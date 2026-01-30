# A Comprehensive Tutorial on Advanced Retrieval Augmented Generation

<!-- SEO SUMMARY: Our comprehensive tutorial explores the intricacies of Advanced Retrieval-Augmented Generation (RAG) systems, demonstrating their capacity to enhance natural language processing tasks by integrating retrieval-based techniques with generative AI models. We examine and provide code snippets for essential components such as document cleaning, chunking, embedding, and the strategic (dynamic) assembly of pipelines for efficient data processing and generation. We emphasize these components, along with fine-tuning and post-retrieval optimizations, to guide developers who want to generate high-quality, contextually relevant text output for effective real-world applications.-->

## Advanced RAG - why we need it

Retrieval-Augmented Generation (RAG) aims to improve the quality and effectiveness of language models by augmenting them with retrieved information from external sources. To familiarize yourself with the basics, read our [Introduction to RAG](https://superlinked.com/vectorhub/retrieval-augmented-generation). In its most basic version, RAG often suffers from low retrieval precision, hallucination in generated responses, and ineffective integration of retrieved context into generated output. These problems are especially limiting for applications that require reliable and informative generated content, such as question answering systems, chatbots, and content creation tools.

To address these issues, advances in RAG methods have evolved, as reflected in RAG [terminology](https://arxiv.org/abs/2312.10997). "Advanced RAG" employs pre-retrieval and post-retrieval strategies, refined indexing approaches, and optimized retrieval processes to improve the quality and relevance of the retrieved information. By addressing the challenges faced by "basic" or "naive RAG" in retrieval, generation, and augmentation, advanced RAG enables language models to generate more accurate, comprehensive, and coherent responses.

### Tutorial Overview

This Advanced RAG tutorial examines and provides code examples of pre-retrieval, retrieval, and post-retrieval techniques employed in Advanced RAG systems, including document cleaning, chunking, embedding, and the strategic, dynamic assembly of pipelines for efficient data processing and generation. We emphasize these components, along with fine-tuning and post-retrieval optimization, so that you can set up your RAG pipeline to generate high-quality, contextually relevant text outputs.

Here's what we cover below:

1. Pre-retrieval
- Chunking
- Document embeddings
- Indexing

2. Retrieval
- Hybrid search

3. Post-retrieval
- Reranking

Let's get started.

## Set up

Before we dive into advanced RAG and pre-retrieval, let's **set up** everything we'll need for this tutorial.

We'll build a RAG system with [LlamaIndex](https://docs.llamaindex.ai/en/stable/) using open source (sentence-transformers) embeddings and models from Huggingface. In addition, we'll need the "accelerate" and "bitsandbytes" libraries to load our generative large language model (LLM) in 4-bit. This set up will enable us to run our RAG system efficiently. Indeed, this tutorial is optimized to work within free [Google Colab](https://colab.research.google.com/) GPU environments. **Note** that if you're on Windows or Mac, you may run into trouble setting up [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), as neither are supported yet. There are, however, several free virtual linux environments available, such as Google Colab and Kaggle.

```bash
!pip install llama-index boilerpy3 sentence-transformers fastembed qdrant_client llama-index-vector-stores-qdrant llama-index-embeddings-huggingface llama-index-llms-huggingface accelerate bitsandbytes
```

### Getting and cleaning the data

Data is the lifeblood of any Machine Learning (ML) model, and its quality directly impacts the performance of RAG systems. Cleaning data includes removing noise such as irrelevant information, correcting typos, and standardizing formats - in short, optimizing it for machine processing. Clean data not only improves the efficiency of retrieval and generation but also significantly enhances the quality of the generated text.

For the purposes of this tutorial, we'll fetch **our data** from [VectorHub](https://superlinked.com/vectorhub) articles, and use a simple extraction pipeline that removes all HTML tags. If we were working with messier data, we could write custom scripts to standardize words and correct typos. The cleaning routine you choose depends heavily on your specific use case and goals.

```python
import requests
from boilerpy3 import extractors
from llama_index.core import Document

# You can add more articles if you have >15GB of RAM available
urls = ['https://superlinked.com/vectorhub/retrieval-augmented-generation']

extractor = extractors.KeepEverythingExtractor()

# Make request to URLs
responses = []
for url in urls:
  response = requests.get(url)
  responses.append(response)

# Pass HTML to Extractor
contents = []
for resp in responses:
  content = extractor.get_content(resp.text)
  contents.append(content)

# Convert raw texts to LlamaIndex documents
documents = [Document(text=content) for content in contents]
```

In the pre-processing step above, we use [boilerpy3](https://github.com/jmriebold/BoilerPy3)'s KeepEverythingExtractor to remove all HTML tags but keep relevant content from HTML files. The KeepEverythingExtractor keeps everything that is tagged as content.
  
Now that we have our data source set up and pre-processed, let's turn to some specific Advanced RAG pre-retrieval, retrieval, and post-retrieval techniques that will improve the quality and relevance of the retrieved information, and solve the issues that plague naive RAG: low retrieval precision, hallucination in generated responses, and ineffective integration of retrieved context into generated output.

## Pre-retrieval

### Chunking

Chunking breaks large pieces of text down into more manageable, logically coherent units. Chunking can improve efficiency by focusing your RAG system on the most relevant segments of text when generating responses. Effective chunking strategies dramatically improve the relevance and cohesion of the generated content.

```python
from llama_index.core.node_parser import SentenceWindowNodeParser

# Creating chunks of sentences with a window function
node_parser = SentenceWindowNodeParser.from_defaults(
	window_size=3,
	window_metadata_key="window",
	original_text_metadata_key="original_text",
)
```

To maintain granular control over the chunking process, we choose to split our documents into sentences with overlapping windows. We set the `window_size` to 3 sentences to ensure each chunk is sufficiently detailed yet concise. There is no universal, golden rule for these hyperparameters. We suggest you try out different configurations based on your documents' structure and type. For a thorough evaluation of chunking methods, check out [this article](https://superlinked.com/vectorhub/an-evaluation-of-rag-retrieval-chunking-methods).

### Document embeddings

Embeddings are central to pre-processing, connecting raw text data to the sophisticated algorithms - typically, LLMs - that drive RAG systems. By converting words into numerical vectors, embeddings capture the semantic relationships between different terms, enabling models to understand context and generate relevant responses. Selecting the right embeddings and models is critical for achieving high-quality retrieval and generation.

We use a sentence transformer from [HuggingFace](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) that considers the full context of sentences or even larger text snippets, rather than individual words in isolation, to generate embeddings. This permits a deeper understanding of the text, capturing nuances and meanings that might be lost in word-level embeddings.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# loads mixedbread-ai/mxbai-embed-large-v1
embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Settings.embed_model defined the model we will use for our embeddings
Settings.embed_model = embed_model
```

Specifically, we selected "mixedbread-ai/mxbai-embed-large-v1", a model that strikes a balance between retrieval accuracy and computational efficiency, according to recent performance evaluations in the Hugging Face [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). 

### Indexing

Before we can index, we need to create a database to write our documents and document embeddings to. We use a simple in-memory `QdrantClient`. You can easily replace it with the storage you are using, as long as it is [supported by LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/storing/).

```python
import qdrant_client

# We are using :memory: mode for fast and light-weight experimentation
client = qdrant_client.QdrantClient(
	location=":memory:"
)
```

Now, let's create our vector store, define a collection name and, finally, index our documents. **Note:** because we use a hybrid search technique in this tutorial, it's important to create our vector store with `enable_hybrid=True`. This generates a single index that can handle both sparse and dense vectors.

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(
	client=client, 
	collection_name="tutorial", 
	enable_hybrid=True
)
	
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
	documents,
	storage_context=storage_context
)
```

Indexing organizes the pre-processed data in a structured format, creating an optimized database of tagged and categorized chunks. In our case, indexing is **handled automatically** by the pipeline, based on the structure of our data. Because we split our documents into chunks and then encode those, every data point will be written to the database with a document id, a chunk id, and both the raw text and the corresponding vector. This automatic indexing process will enable our RAG system to retrieve relevant information quickly and accurately.

## Retrieval

### Hybrid Search

Another way to enhance retrieval accuracy is through [hybrid search](https://superlinked.com/vectorhub/optimizing-rag-with-hybrid-search-and-reranking) techniques. In certain contexts, the high precision of a traditional keyword search method like BM25 makes it a valuable addition to vector search. Adopting a hybrid keyword and vector search is very straightforward with LlamaIndex; we simply set `vector_store_query_mode` to "hybrid", and choose a value for the `alpha` parameter, which controls the weighting of the two search methods. Alpha ranges from 0 for pure keyword search to 1 for pure vector search.

```python
	# set Logging to DEBUG for more detailed outputs
	query_engine = index.as_query_engine(
	...,
	vector_store_query_mode="hybrid",
	alpha=0.5,
	...
)
```

This hybrid approach captures both the semantic richness of embeddings and the direct match precision of keyword search, leading to improved relevance in retrieved documents.

So far we've seen how careful pre-retrieval (data preparation, chunking, embedding, indexing) and retrieval (hybrid search) can help improve RAG retrieval results. What about _after_ we've done our retrieval?

## Post-retrieval

### Reranking

Reranking lets us reassess the initial set of retrieved documents and refine the selection based on relevance and context. It often employs more sophisticated or computationally intensive methods that would have been impractical in the initial retrieval (because the initial dataset is larger than our retrieved set). Reranking still takes time - you should always evaluate the performance of your system on your data with _and_ without reranking to make sure the additional latency and cost of reranking are worth it. In our case, we rerank using a TransformersSimilarityRanker model.

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

# Define reranker model
rerank = SentenceTransformerRerank(
	top_n = 10,
	model = "mixedbread-ai/mxbai-embed-large-v1"
)
```

We use the same embedding model that we previously used for embedding our document chunks. Setting `top_n` to 10 keeps the top 10 documents after reranking. Keep in mind that the retrieved chunks have to fit into the context (token) limits of your large language model (LLM). We need to choose a `top_n` number that keeps the combined chunks length within the LLM's context window limits, so that the LLM can process them in a single pass.

## Generation

In the generation phase, we integrate the prepared context into a prompt to produce the final output. To do this, we use an LLM to generate answers based on the context provided by the previous components of the pipeline. To prepare for generation, we set up a prompt template to format the input so that it's optimized for the LLM to understand and respond to.

```python
from llama_index.core import PromptTemplate

query_wrapper_prompt = PromptTemplate("""
  <|system|>
  Using the information contained in the context, 
  give a comprehensive answer to the question.
  If the answer cannot be deduced from the context, do not give an answer.</s>
  <|user|>
  {query_str}</s>
  <|assistant|>"""
)
```

Our generator is [Zephyr-7B-Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), a popular, fine-tuned model based on [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1). Zephyr-7B-Beta is relatively small but highly performant.

```python
from llama_index.llms.huggingface import HuggingFaceLLM

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    model_name="HuggingFaceH4/zephyr-7b-beta",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)

Settings.chunk_size = 512
Settings.llm = llm
```

### Assembling the Query Engine

Now that we have defined all necessary components, we can assemble our LlamaIndex query engine. To review, this includes hybrid search, an LLM for generation, and a reranker. The `similarity_top_k` setting defines how many results will be retrieved by the (hybrid) search. 

```python
# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(
	llm=llm,
	similarity_top_k=10,
	# Rerankers are node_postprocessors in LlamaIndex
	node_postprocessors=[rerank],
	vector_store_query_mode="hybrid",
	alpha=0.5,
)
```

Now, **let's try it out**.
Our example questions will query one of our previous VectorHub tutorials, [An introduction to RAG](https://superlinked.com/vectorhub/retrieval-augmented-generation).

```python
query = "Based on these articles, what are the dangers of hallucination?"

response = query_engine.query(query)
```

Let's see what our advanced RAG system generated:

```python
from IPython.display import Markdown, display

display(Markdown(f"<b>{response}</b>"))
```

"Based on the context provided, the dangers of hallucinations in the context of machine learning and natural language processing are that they can lead to inaccurate or incorrect results, particularly in customer support and content creation. These hallucinations, which are false pieces of information generated by a generative model, can have disastrous consequences in use cases where there's more at stake than simple internet searches. In short, machine hallucinations can be dangerous because they can lead to false information being presented as fact, which can have serious consequences in real-world applications."

Our advanced RAG pipeline result appears to be relatively precise, avoids hallucinations, and effectively integrates retrieved context into generated output. Note: generation is not a fully deterministic process, so if you run this code yourself, you may receive slightly different output.

## Conclusion

In this tutorial, we've covered several critical aspects of setting up an advanced RAG system:

-  **Data Preparation:** Ensuring the data is in the right format and free of noise for optimal processing.

-  **Chunking and Embedding:** Breaking down the text into manageable pieces, and converting them into numerical vectors to capture semantic meaning.

-  **Retrieval:** Finding the most relevant documents based on the query.

-  **Generation:** Crafting a prompt that guides the language model to generate a relevant and accurate answer.

-  **Pipeline Assembly:** Bringing together various components into a cohesive query engine that processes queries end-to-end.

-  **Execution:** Running the query engine to obtain answers to specific queries.

In short, an advanced RAG system should be highly customizable, letting you tweak each component and incorporate different models and strategies at various stages of the pipeline, in order to address the weaknesses of naive RAG. Thus, an advanced RAG system can help tackle a wide range of tasks, from answering questions to generating content based on complex criteria, in a way that addresses the challenges faced by "basic" or "naive RAG" in retrieval, generation, and augmentation.

---

## Contributors

- [Pascal Biese, author](https://www.linkedin.com/in/pascalbiese/)
- [Mór Kapronczay, Editor](https://www.linkedin.com/in/mór-kapronczay-49447692)
- [Robert Turner, Editor](https://www.linkedin.com/in/robertdhayanturner/)
