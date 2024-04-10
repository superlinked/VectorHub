# Scaling RAG for Production

Retrieval-augmented Generation (RAG) combines Large Language Models (LLMs) with external data to reduce the probability of machine hallucinations - AI-generated information that misrepresents underlying data or reality. When developing RAG systems, scalability is often an afterthought. This creates problems when moving from initial development to production. Having to manually adjust code while your application grows can get very costly and is prone to errors.

Our tutorial provides an example of **how you can develop a RAG pipeline with production workloads in mind from the start**, using the right tools - ones that are designed to scale your application.

## Development vs. production

The goals and requirements of development and production are usually very different. This is particularly true for new technologies like Large Language Models (LLMs) and Retrieval-augmented Generation (RAG), where organizations prioritize rapid experimentation to test the waters before committing more resources. Once important stakeholders are convinced, the focus shifts from demonstrating an application's _potential for_ creating value to _actually_ creating value, via production. Until a system is productionized, its ROI is typically zero.

**Productionizing**, in the context of [RAG systems](retrieval-augmented-generation), involves transitioning from a prototype or test environment to a **stable, operational state**, in which the system is readily accessible and reliable for remote end users, such as via URL - i.e., independent of the end user machine state. Productionizing also involves **scaling** the system to handle varying levels of user demand and traffic, ensuring consistent performance and availability.

Even though there is no ROI without productionizing, organizations often underesimate the hurdles involved in getting to an end product. Productionizing is always a trade-off between performance and costs, and this is no different for Retrieval-augmented Generation (RAG) systems. The goal is to achieve a stable, operational, and scalable end product while keeping costs low. 

Let's look more closely at the basic requirements of an [RAG system](retrieval-augmented-generation), before going in to the specifics of what you'll need to productionize it in a cost-effective but scalable way.

## The basics of RAG

The most basic RAG workflow looks like this:

1. Submit a text query to an embedding model, which converts it into a semantically meaningful vector embedding.
2. Send the resulting query vector embedding to your document embeddings storage location - typically a [vector database](../building-blocks/vector-search/access-patterns#dynamic-access).
3. Retrieve the most relevant document chunks - based on proximity of document chunk embeddings to the query vector embedding.
4. Add the retrieved document chunks as context to the query vector embedding and send it to the LLM.
5. The LLM generates a response utilizing the retrieved context.

While RAG workflows can become significantly more complex, incorporating methods like metadata filtering and retrieval reranking, _all_ RAG systems must contain the components involved in the basic workflow: an embedding model, a store for document and vector embeddings, a retriever, and a LLM.

But smart development, with productionization in mind, requires more than just setting up your components in a functional way. You must also develop with cost-effective scalability in mind. For this you'll need not just these basic components, but more specifically the tools appropriate to configuring a scalable RAG system.

## Developing for scalability: the right tools

### LLM library: LangChain

As of this writing, LangChain, while it has also been the subject of much criticism, is arguably the most prominent LLM library. A lot of developers turn to Langchain to build Proof-of-Concepts (PoCs) and Minimum Viable Products (MVPs), or simply to experiment with new ideas. Whether one chooses LangChain or one of the other major LLM and RAG libraries - for example, LlamaIndex or Haystack, to name our alternate personal favorites - they can _all_ be used to productionize an RAG system. That is, all three have integrations for third-party libraries and providers that will handle production requirements. Which one you choose to interface with your other components depends on the details of your existing tech stack and use case.

For the purpose of this tutorial, we'll use part of the Langchain documentation, along with Ray.

### Scaling with Ray

Because our goal is to build a 1) simple, 2) scalable, _and_ 3) economically feasible option, not reliant on proprietary solutions, we have chosen to use [Ray](https://github.com/ray-project/ray), a Python framework for productionizing and scaling machine learning (ML) workloads. Ray is designed with a range of auto-scaling features that seamlessly scale ML systems. It's also adaptable to both local environments and Kubernetes, efficiently managing all workload requirements.

**Ray permits us to keep our tutorial system simple, non-proprietary, and on our own network, rather than the cloud**. While LangChain, LlamaIndex, and Haystack libraries support cloud deployment for AWS, Azure, and GCP, the details of cloud deployment heavily depend on - and are therefore very particular to - the specific cloud provider you choose. These libraries also all contain Ray integrations to enable scaling. But **using Ray directly will provide us with more universally applicable insights**, given that the Ray integrations within LangChain, LlamaIndex, and Haystack are built upon the same underlying framework.

Now that we have our LLM library sorted, let's turn to data gathering and processing.

## Data gathering and processing

### Gathering the data

Every ML journey starts with data, and that data needs to be gathered and stored somewhere. For this tutorial, we gather data from part of the LangChain documentation. We first download the html files and then create a [Ray dataset](https://docs.ray.io/en/latest/data/data.html) of them.

We start by **installing all the dependencies** that we'll use:

```console
pip install ray langchain sentence-transformers qdrant-client einops openai tiktoken fastapi "ray[serve]"
```

We use the OpenAI API in this tutorial, so we'll **need an API key**. We export our API key as an environmental variable, and then **initialize our Ray environment** like this:

```python
import os
import ray

working_dir = "downloaded_docs"

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

# Setting up our Ray environment
ray.init(runtime_env={
    "env_vars": {
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
    },
    "working_dir": str(working_dir)
})
```

To work with the LangChain documentation, we need to **download the html files and process them**. Scraping html files can get very tricky and the details depend heavily on the structure of the website you’re trying to scrape. The functions below are only meant to be used in the context of this tutorial.

```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def sanitize_filename(filename):
    filename = re.sub(r'[\\/*?:"<>|]', '', filename)  # Remove problematic characters
    filename = re.sub(r'[^\x00-\x7F]+', '_', filename)  # Replace non-ASCII characters
    return filename

def is_valid(url, base_domain):
    parsed = urlparse(url)
    valid = bool(parsed.netloc) and parsed.path.startswith("/docs/expression_language/")
    return valid

def save_html(url, folder):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else os.path.basename(urlparse(url).path)
        sanitized_title = sanitize_filename(title)
        filename = os.path.join(folder, sanitized_title.replace(" ", "_") + ".html")

        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(str(soup))
            print(f"Saved: {filename}")

            links = [urljoin(url, link.get('href')) for link in soup.find_all('a') if link.get('href') and is_valid(urljoin(url, link.get('href')), base_domain)]
            return links
        else:
            return []
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

def download_all(start_url, folder, max_workers=5):
    visited = set()
    to_visit = {start_url}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while to_visit:
            future_to_url = {executor.submit(save_html, url, folder): url for url in to_visit}
            visited.update(to_visit)
            to_visit.clear()

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    new_links = future.result()
                    for link in new_links:
                        if link not in visited:
                            to_visit.add(link)
                except Exception as e:
                    print(f"Error with future for {url}: {e}")
```

Because the LangChain documentation is very large, we'll download only **a subset** of it: **LangChain's Expression Language (LCEL)**, which consists of 28 html pages.

```python
base_domain = "python.langchain.com"
start_url = "https://python.langchain.com/docs/expression_language/"
folder = working_dir

download_all(start_url, folder, max_workers=10)
```

Now that we've downloaded the files, we can use them to **create our Ray dataset**:

```python
from pathlib import Path

# Ray dataset
document_dir = Path(folder)
ds = ray.data.from_items([{"path": path.absolute()} for path in document_dir.rglob("*.html") if not path.is_dir()])
print(f"{ds.count()} documents")
```

Great! But there's one more step left before we can move on to the next phase of our workflow. We need to **extract the relevant text from our html files and clean up all the html syntax**. For this, we import BeautifulSoup to **parse the files and find relevant html tags**.

```python
from bs4 import BeautifulSoup, NavigableString

def extract_text_from_element(element):
    texts = []
    for elem in element.descendants:
        if isinstance(elem, NavigableString):
            text = elem.strip()
            if text:
                texts.append(text)
    return "\n".join(texts)

def extract_main_content(record):
    with open(record["path"], "r", encoding="utf-8") as html_file:
        soup = BeautifulSoup(html_file, "html.parser")

    main_content = soup.find(['main', 'article'])  # Add any other tags or class_="some-class-name" here
    if main_content:
        text = extract_text_from_element(main_content)
    else:
        text = "No main content found."

    path = record["path"]
    return {"path": path, "text": text}

```

We can now use Ray's map() function to run this extraction process. Ray lets us run multiple processes in parallel.

```python
# Extract content
content_ds = ds.map(extract_main_content)
content_ds.count()

```

Awesome! The results of the above extraction are our dataset. Because Ray datasets are optimized for scaled performance in production, they don't require us to make costly and error-prone adjustments to our code when our application grows. 

### Processing the data

To process our dataset, our next three steps are **chunking, embedding, and indexing**.

**Chunking the data**

Chunking - splitting your documents into multiple smaller parts - is necessary to make your data meet the LLM’s context length limits, and helps keep contexts specific enough to remain relevant. Chunks also need to not be too small. When chunks are too small, the information retrieved may become too narrow to provide adequate query responses. The optimal chunk size will depend on your data, the models you use, and your use case. We will use a common chunking value here, one that has been used in a lot of applications.

Let’s define our text splitting logic first, using a standard text splitter from LangChain:

```python
from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Defining our text splitting function
def chunking(document, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)

    chunks = text_splitter.create_documents(
        texts=[document["text"]], 
        metadatas=[{"path": document["path"]}])
    return [{"text": chunk.page_content, "path": chunk.metadata["path"]} for chunk in chunks]
```

Again, we utilize Ray's map() function to ensure scalability:

```python
chunks_ds = content_ds.flat_map(partial(
    chunking, 
    chunk_size=512, 
    chunk_overlap=50))
print(f"{chunks_ds.count()} chunks")
```

Now that we've gathered and chunked our data scalably, we need to embed and index it, so that we can efficiently retrieve relevant answers to our queries.

**Embedding the data**

We use a pretrained model to create vector embeddings for both our data chunks and the query itself. By measuring the distance between the chunk embeddings and the query embedding, we can identify the most relevant, or "top-k," chunks. Of the various pretrained models, we'll use the popular 'bge-base-en-v1.5' model, which, at the time of writing this tutorial, ranks as the highest-performing model of its size on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). For convenience, we continue using LangChain:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy as np
from ray.data import ActorPoolStrategy

def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):
    embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    return embedding_model
```

This time, instead of map(), we want to use map_batches(), which requires defining a class object to perform a **call** on.

```python
class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = get_embedding_model(
            embedding_model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100})
    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "path": batch["path"], "embeddings": embeddings}

# Embedding our chunks
embedding_model_name = "BAAI/bge-base-en-v1.5"
embedded_chunks = chunks_ds.map_batches(
    EmbedChunks,
    fn_constructor_kwargs={"model_name": embedding_model_name},
    batch_size=100, 
    num_gpus=1,
    concurrency=1)
```

**Indexing the data**

Now that our chunks are embedded, we need to **store** them somewhere. For the sake of this tutorial, we'll utilize Qdrant’s new in-memory feature, which lets us experiment with our code rapidly without needing to set up a fully-fledged instance. However, for deployment in a production environment, you should rely on more robust and scalable solutions — hosted either within your own network or by a third-party provider. For example, to fully productionize, we would need to point to our Qdrant (or your preferred hosted vendor) instance instead of using it in-memory. Detailed guidance on self-hosted solutions, such as setting up a Kubernetes cluster, are beyond the scope of this tutorial.

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Initalizing a local client in-memory
client = QdrantClient(":memory:")

client.recreate_collection(
   collection_name="documents",
   vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
)
```

To perform the next processing step, storage, using Ray would require more than 2 CPU scores, making this tutorial incompatible with the free tier of Google Colab. Instead, then, we'll use pandas. Fortunately, Ray allows us to convert our dataset into a pandas DataFrame with a single line of code:

```python
emb_chunks_df = embedded_chunks.to_pandas()
```

Now that our dataset is converted to pandas, we **define and execute our data storage function**:

```python
from qdrant_client.models import PointStruct

def store_results(df, collection_name="documents", client=client):
	  # Defining our data structure
    points = [
        # PointStruct is the data classs used in Qdrant
        PointStruct(
            id=hash(path),  # Unique ID for each point
            vector=embedding,
            payload={
                "text": text,
                "source": path
            }
        )
        for text, path, embedding in zip(df["text"], df["path"], df["embeddings"])
    ]
		
		# Adding our data points to the collection
    client.upsert(
        collection_name=collection_name,
        points=points
    )

store_results(emb_chunks_df)
```

This wraps up the data processing part! Our data is now stored in our vector database and ready to be retrieved.

## Data retrieval

When you retrieve data from vector storage, it's important to use the same embedding model for your query that you used for your source data. Otherwise, vector comparison to surface relevant content may result in mismatched or non-nuanced results (due to semantic drift, loss of context, or inconsistent distance metrics).

```python
import numpy as np 

# Embed query
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
query = "How to run agents?"
query_embedding = np.array(embedding_model.embed_query(query))
len(query_embedding)
```

Recall from above that we measure the distance between the query embedding and chunk embeddings to identify the most relevant, or 'top-k' chunks. In Qdrant’s search, the 'limit' parameter is equivalent to 'k'. By default, the search uses cosine similarity as the metric, and retrieves from our database the 5 chunks closest to our query embedding:

```python
hits = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5  # Return 5 closest points
)

context_list = [hit.payload["text"] for hit in hits]
context = "\n".join(context_list)
```

We rewrite this as a function for later use:

```python
def semantic_search(query, embedding_model, k):
    query_embedding = np.array(embedding_model.embed_query(query))
    hits = client.search(
      collection_name="documents",
      query_vector=query_embedding,
      limit=5  # Return 5 closest points
    )

    context_list = [{"id": hit.id, "source": str(hit.payload["source"]), "text": hit.payload["text"]} for hit in hits]
    return context_list
```

## Generation

We're now very close to being able to field queries and retrieve answers! We've set up everything we need to query our LLM _at scale_. But before querying the model for a response, we want to first inform the query with our data, by **retrieving relevant context from our vector database and then adding it to the query**.

To do this, we use a simplified version of the generate.py script provided in Ray's [LLM repository](https://github.com/ray-project/llm-applications/blob/main/rag/generate.py). This version is adapted to our code and - to simplify and keep our focus on how to scale a basic RAG system - leaves out a bunch of advanced retrieval techniques, such as reranking and hybrid search. For our LLM, we use gpt-3.5-turbo, and query it via the OpenAI API.

```python
from openai import OpenAI

def get_client(llm):
  api_key = os.environ["OPENAI_API_KEY"]
  client = OpenAI(api_key=api_key)
  return client

def generate_response(
    llm,
    max_tokens=None,
    temperature=0.0,
    stream=False,
    system_content="",
    assistant_content="",
    user_content="",
    max_retries=1,
    retry_interval=60,
):
    """Generate response from an LLM."""
    retry_count = 0
    client = get_client(llm=llm)
    messages = [
        {"role": role, "content": content}
        for role, content in [
            ("system", system_content),
            ("assistant", assistant_content),
            ("user", user_content),
        ]
        if content
    ]
    while retry_count <= max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=llm,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                messages=messages,
            )
            return prepare_response(chat_completion, stream=stream)

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""

def response_stream(chat_completion):
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content

def prepare_response(chat_completion, stream):
    if stream:
        return response_stream(chat_completion)
    else:
        return chat_completion.choices[0].message.content
```

Finally, we **generate a response**:

```python
# Generating our response
query = "How to run agents?"
response = generate_response(
    llm="gpt-3.5-turbo",
    temperature=0.0,
    stream=True,
    system_content="Answer the query using the context provided. Be succinct.",
    user_content=f"query: {query}, context: {context_list}")
# Stream response
for content in response:
    print(content, end='', flush=True)
```

To **make using our application even more convenient**, we can simply adapt Ray's official documentation to **implement our workflow within a single QueryAgent class**, which bundles together and takes care of all of the steps we implemented above - retrieving embeddings, embedding the search query, performing vector search, processing the results, and querying the LLM to generate a response. Using this single class approach, we no longer need to sequentially call all of these functions, and can also include utility functions. (Specifically, `Get_num_tokens` encodes our text and gets the number of tokens, to calculate the length of the input. To maintain our standard 50:50 ratio to allocate space to each of input and generation, we use `(text, max_context_length)` to trim input text if it's too long.)

```python
import tiktoken

def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def trim(text, max_context_length):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_context_length])

class QueryAgent:
    def __init__(
        self,
        embedding_model_name="BAAI/bge-base-en-v1.5",
        llm="gpt-3.5-turbo",
        temperature=0.0,
        max_context_length=4096,
        system_content="",
        assistant_content="",
    ):
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100},
        )

        # LLM
        self.llm = llm
        self.temperature = temperature
        self.context_length = int(
            0.5 * max_context_length
        ) - get_num_tokens(  # 50% of total context reserved for input
            system_content + assistant_content
        )
        self.max_tokens = int(
            0.5 * max_context_length
        )  # max sampled output (the other 50% of total context)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(
        self,
        query,
        num_chunks=5,
        stream=True,
    ):
        # Get top_k context
        context_results = semantic_search(
            query=query, embedding_model=self.embedding_model, k=num_chunks
        )

        # Generate response
        document_ids = [item["id"] for item in context_results]
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm=self.llm,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=trim(user_content, self.context_length),
        )

        # Result
        result = {
            "question": query,
            "sources": sources,
            "document_ids": document_ids,
            "answer": answer,
            "llm": self.llm,
        }
        return result
```

To embed our query and retrieve relevant vectors, and then generate a response, we run our QueryAgent as follows:

```python
import json 

query = "How to run an agent?"
system_content = "Answer the query using the context provided. Be succinct."
agent = QueryAgent(
    embedding_model_name="BAAI/bge-base-en-v1.5",
    llm="gpt-3.5-turbo",
    max_context_length=4096,
    system_content=system_content)
result = agent(query=query, stream=False)
print(json.dumps(result, indent=2))
```

## Serving our application

Our application is now running! Our last productionizing step is to serve it. Ray's [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) module makes this step very straightforward. We combine Ray Serve with FastAPI and pydantic. The @serve.deployment decorator lets us define how many replicas and compute resources we want to use, and Ray’s autoscaling will handle the rest. Two Ray Serve decorators are all we need to modify our FastAPI application for production.

```python
import pickle
import requests
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

# Initialize application
app = FastAPI()

class Query(BaseModel):
    query: str

class Response(BaseModel):
    llm: str
    question: str
    sources: List[str]
    response: str

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2, "num_gpus": 1})
@serve.ingress(app)
class RayAssistantDeployment:
    def __init__(self, embedding_model_name, embedding_dim, llm):
        
        # Query agent
        system_content = "Answer the query using the context provided. Be succinct. " \
            "Contexts are organized in a list of dictionaries [{'text': <context>}, {'text': <context>}, ...]. " \
            "Feel free to ignore any contexts in the list that don't seem relevant to the query. "
        self.gpt_agent = QueryAgent(
            embedding_model_name=embedding_model_name,
            llm="gpt-3.5-turbo",
            max_context_length=4096,
            system_content=system_content)

    @app.post("/query")
    def query(self, query: Query) -> Response:
        result = self.gpt_agent(
            query=query.query, 
            stream=False
            )
        return Response.parse_obj(result)
```

Now, we're ready to **deploy** our application:

```python
# Deploying our application with Ray Serve
deployment = RayAssistantDeployment.bind(
    embedding_model_name="BAAI/bge-base-en-v1.5",
    embedding_dim=768,
    llm="gpt-3.5.-turbo")

serve.run(deployment, route_prefix="/")
```

Our FastAPI endpoint is capable of being queried like any other API, while Ray take care of the workload automatically:

```python
# Performing inference
data = {"query": "How to run an agent?"}
response = requests.post(
"https://127.0.0.1:8000/query", json=data
)

try:
  print(response.json())
except:
  print(response.text)
```

Wow! We've been on quite a journey. We gathered our data using Ray and some LangChain documentation, processed it by chunking, embedding, and indexing it, set up our retrieval and generation, and, finally, served our application using Ray Serve. Our tutorial has so far covered an example of how to develop scalably and economically - how to productionize from the very start of development. 

Still, there is one last crucial step.

## Production is only the start: maintenance

To fully productionize any application, you also need to maintain it. And maintaining your application is a continuous task.

Maintenance involves regular assessment and improvement of your application. You may need to routinely update your dataset if your application relies on being current with real-world changes. And, of course, you should monitor application performance to prevent  degradation. For smoother operations, we recommend integrating your workflows with CI/CD pipelines.

### Limitations and future discussion

Other critical aspects of scalably productionizing fall outside of the scope of this article, but will be explored in future articles, including:

- **Advanced Development** Pre-training, finetuning, prompt engineering and other in-depth development techniques
- **Evaluation** Randomness and qualitative metrics, and complex multi-part structure of RAG can make LLM evaluation difficult
- **Compliance** Adhering to data privacy laws and regulations, especially when handling personal or sensitive information

---
## Contributors

- [Pascal Biese, author](https://www.linkedin.com/in/pascalbiese/)
- [Mór Kapronczay, contributor](https://www.linkedin.com/in/m%C3%B3r-kapronczay-49447692/)
- [Robert Turner, editor](https://robertturner.co/copyedit/)
