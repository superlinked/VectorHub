<!--Second of a three-part (monthly) series on RAG evaluation. In this article, we'll demystify an existing evaluation framework (RAGAS), and see how well it does at covering all the layers (model, data ingestion, semantic retrieval, E2E) of evaluation we introduced in our first article.... -->


# Evaluating Retrieval Augmented Generation using RAGAS

*In our [first article](https://superlinked.com/vectorhub/articles/evaluating-retrieval-augmented-generation-framework), we introduced RAG evaluation and its challenges, proposed an evaluation framework, and provided an overview of the various tools and approaches you can use to evaluate your RAG application. In this **second** of three articles (one a month), **we leverage the knowledge and concepts we built in part one to walk you through our first RAG evaluation framework, [RAGAS](https://docs.ragas.io/)**, which is based on Es et al.'s [introductory paper](https://arxiv.org/pdf/2309.15217). RAGAS aims to provide useful and actionable metrics but relies on as little annotated data as possible, making it cheaper and faster than using LLMs to evaluate your RAG pipeline.

To set up our RAGAS evaluation framework, we apply a concept we discussed in the [first article](https://superlinked.com/vectorhub/articles/evaluating-retrieval-augmented-generation-framework) - ground truth. Some ground truth data - a golden set - is needed for evaluating our RAG pipeline's generated responses. It provides a meaningful context for our evaluation metrics. Our first step, therefore, in setting up RAGAS is creating an evaluation dataset, complete with questions, answers, and ground-truth data that takes account of relevant context. Next, we'll look more closely at evaluation metrics, indicating which ones make sense for specific use cases, and how to interpret them. Finally, we'll look at how to use our evaluation dataset to assess our chosen metrics.

This article will take you through:

1) **Creating a synthetic evaluation dataset**
   a. evaluation dataset = the basics
   b. different ways to build one
   c. FAQs and suggestions
2) **Metrics**
   a. different metrics provided by RAGAS
   b. choosing the right metrics for your use cases
   c. learning to decode the metric values
3) **Using the evaluation dataset to assess given metrics**
4) **Code walkthrough and reference notebook**


## Synthetic evaluation dataset

### The basics of evaluation datasets

Golden or evaluation datasets are crucial in the assessment of information retrieval systems. Often, evaluation datasets contain triplets of queries, document ids, and relevance scores. In RAG evaluation, queries are replaced by **questions**, document id is replaced by **chunk_id, chunk text, an answer, or a ground truth**, and instead of relevance scores, we leverage **context texts** to provide an optimal evaluation record. Optionally, we could add **question complexity level** - our RAG system may perform better answering questions of a certain difficulty level.

But generating numerous Question-Context-Answer samples by hand from reference data would not only be tedious but also inefficient. Furthermore, questions crafted by individuals may not achieve the complexity needed for effective evaluation, thereby diminishing the assessment's accuracy. If manually creating RAG evaluation dataset is not a feasible solution, what would be? **What's an efficient and effective way of creating our RAG evaluation dataset**?

Fortunately, we can use the Hugging Face Datasets library to build our evaluation datasets efficiently and effectively. Using this library, we build two Hugging Face datasets: [Source Documentation Dataset](https://huggingface.co/datasets/atitaarora/qdrant_doc) - a dataset with all documentation text with their sources, and an [Evaluation Dataset](https://huggingface.co/datasets/atitaarora/qdrant_doc_qna) - our "golden set" or reference standard, leveraged for evaluation purposes.

**Generating the questions
To make a set of questions that are clear, concise, and related to the information in the source documents, we need to use targeted snippets - selected because they contain relevant information.

>  💡 Note: Some of the techniques below may involve using **OpenAI** to generate the questions on the chunked document.

### Method 1 - Using T5

Our first method uses the [T5](https://huggingface.co/docs/transformers/en/model_doc/t5) model. Make sure to have `pip install transformers` before you try using T5.

```python
from transformers import T5ForConditionalGeneration, T5TokenizerFast
hfmodel = T5ForConditionalGeneration.from_pretrained("ThomasSimonini/t5-end2end-question-generation")
checkpoint = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)
tokenizer.sep_token = '<sep>'
tokenizer.add_tokens(['<sep>'])
model.resize_token_embeddings(len(tokenizer))
# Check the sep_token_id to verify that it was added to the tokenizer
tokenizer.sep_token_id

def hf_run_model(input_string, **generator_args):
  generator_args = {
  "max_length": 256,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
  }
  input_string = "generate questions: " + input_string + " </s>"
  input_ids = tokenizer.encode(input_string, return_tensors="pt")
  res = hfmodel.generate(input_ids, **generator_args)
  output = tokenizer.batch_decode(res, skip_special_tokens=True)
  output = [item.split("<sep>") for item in output]
  return output
```

We use T5 to create a question for the given text:

```python
text_passage = """HNSW is the approximate nearest neighbor search. 
          This means our accuracy improves up to a point of diminishing returns, as we check the index for more similar candidates. 
          In the context of binary quantization, this is referred to as the oversampling rate."""

hf_run_model(text_passage)

##Outputs 
#[['What is the approximate nearest neighbor search?',
#  ' What does HNSW mean in terms of accuracy?',
#  ' In binary quantization what is the oversampling rate?',
#  '']]
```

Though we still need to identify and extract the correct answers manually, _this method handles question generation efficiently and effectively_, creating a pretty sensible set of questions in the process.


### **Method 2 -  Using OpenAI.**

```python
import openai
from getpass import getpass
import os

if not (OPENAI_API_KEY := os.getenv("OPENAI_API_KEY")):
    OPENAI_API_KEY = getpass("🔑 Enter your OpenAI API key: ")
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model ="gpt-3.5-turbo"
client = OpenAI()

 # Function to generate synthetic question-answer pairs
def generate_question_answer(context):

  prompt = f"""Generate a question and answer pair based keeping in mind the following: 
  Please generate a clear and concise question that requires understanding of the content provided in the document chunk. Ensure that the question is specific, relevant, and not too broad.
  Avoid questions such as 'in the given passage or document chunk' kind of questions. Ensure the question is about the concept the document chunk is about.
  Provide a complete , detailed and accurate answer to the question. Make sure that the answer addresses the question directly and comprehensively, drawing from the information provided in the document chunk.
  Use technical terminology appropriately and maintain clarity throughout the response.
  Based on this Context : {context}
  """
        
  response = client.chat.completions.create(
        model = model,
        messages = [ 
        {'role': 'user', 'content': prompt}
          ],
        temperature=0.5,
        max_tokens=200,
        n=1,
        stop=None
    )
  return response.choices[0].message.content
```

Using the subroutine above to generate question and answer/ground-truth pairs as :

```python
# Generate question-answer pairs for the given chunk

context = context = """HNSW is the approximate nearest neighbor search. 
          This means our accuracy improves up to a point of diminishing returns, as we check the index for more similar candidates. 
          In the context of binary quantization, this is referred to as the oversampling rate."""

question_answer_pair = generate_question_answer(context)
print(question_answer_pair)

## Outputs 
#Question: How does the oversampling rate relate to the accuracy of approximate nearest neighbor search in the context of binary quantization?
#Answer: In the context of binary quantization, the oversampling rate refers to the number of similar candidates that are checked in the index during an approximate nearest neighbor search. As we increase the oversampling rate, we are essentially checking more candidates for similarity, which can improve the accuracy of the search up to a certain point. However, there is a point of diminishing returns where further increasing the oversampling rate may not significantly improve the accuracy any further. Therefore, the oversampling rate plays a crucial role in balancing the trade-off between accuracy and computational efficiency in approximate nearest neighbor search using binary quantization.
```

We saw another way to generate questions along with answers/ground-truth using OpenAI prompts this time.

### **Method 3 -  Using RAGAS.**
As we are discussing about RAGAS , it comes with an easier way to generate Question-Context-Ground_Truth set and rather a complete baseline evaluation dataset [utility](https://docs.ragas.io/en/latest/getstarted/testset_generation.html) , using your dataset in just a couple of lines of code as below.

```python
## Test Evaluation Dataset Generation using Ragas
from langchain.docstore.document import Document as LangchainDocument
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# load dataset from which the questions have to be created 
dataset = load_dataset("atitaarora/qdrant_doc", split="train")

#Process dataset into langchain documents
langchain_docs = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in tqdm(dataset)
]

generator = TestsetGenerator.with_openai()
# using the first 10 docs as langchain_docs[:10] to check the sample questions
testset = generator.generate_with_langchain_docs(langchain_docs[:10], test_size=10, 
                                                 raise_exceptions=False, with_debugging_logs=False,
                                                 distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}) 

df = testset.to_pandas()
df.head(10)
```

Which gives us a reasonable set of baseline question-context-ground_truth set as below which could later be used to generate response from our RAG pipeline for evaluation :

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/ragas_baseline_eval_dataset_preview.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/ragas_baseline_eval_dataset_preview.png)

Let's zoom into one of the rows to see what did RAGAS generate for us: 

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/ragas_sample_question.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/ragas_sample_question.png)

The first column is the `question` generated based on the given list of values of `contexts` along with value of `ground_truth` used to evaluate our `answer` which is going to be surfaced as we run the `question` through our RAG pipeline.
It is a good idea to export them as a hugging-face dataset to be utilised later during the evaluation step.
The techniques above let you build a good evaluation dataset including reasonably good questions and ground-truths to evaluate your RAG system.
---

> 💡 An essential aspect to consider is the creation of an evaluation dataset with `answer` for each tuning cycle. This suggests developing a subroutine that facilitates the construction of evaluation dataset in the expected RAGAS format, as illustrated above, utilizing the provided questions and executing them through your RAG system.

A sample subroutine may look like this below :

```python
## Prepare the evaluation dataset to evaluate our RAG system
from datasets import Dataset

# RAGAS Expect ['question', 'answer', 'contexts', 'ground_truths'] format
'''
{
    "question": ['What is quantization?', ...], 
    "answer": [], ## answer
    "contexts": [], ## context
    "ground_truths": [] ## answer expected
}
'''
# loading the eval dataset from HF
qdrant_qa = load_dataset("atitaarora/qdrant_doc_qna", split="train")

questions = []
answers = []
contexts = []
ground_truths = []

for query in qdrant_qa:
    question = query["question"]
    questions.append(question)
    ground_truths.append([query["answer"]])
    
    rag_response = query_with_context(question)
    answers.append(rag_response)
    search_result_context = client.query(collection_name=COLLECTION_NAME, query_text=question,limit=4)
    context = ["document:"+r.document+",source:"+r.metadata['source'] for r in search_result_context]
    contexts.append(context)    

rag_response_data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

## The dataset used to evaluate RAG using RAGAS
## Note this is the dataset needed for evaluation hence has to be recreated everytime changes to RAG config is made
rag_response_dataset = Dataset.from_dict(rag_response_data)
# Might as well export it as a csv with some recognizable experiment name like chunk size 512
rag_response_dataset.to_csv('rag_chunk_512.csv')
```
 

The subroutine here uses an abstraction of your RAG pipeline `query_with_context()` method , we will show a sample naive-rag example in a bit. It uses the older format with `ground_truths` which is the list on ground_truth strings while the latest format asserts to use `ground_truth` as a string value. You can choose your preferred format. 

Now that we have a baseline as well as the subroutine to create evaluation dataset .

Let us understand what all can you measure with RAGAS.

## Metrics

It is a good idea to check out [core concepts](https://docs.ragas.io/en/latest/concepts/index.html) of RAGAS , at this stage.

RAGAS is focused on retrieval and generation stages of RAG along with End-to-End evaluation of RAG and it relies on concept of [metrics driven development](https://docs.ragas.io/en/latest/concepts/metrics_driven.html)

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/ragas_metrics.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/ragas_metrics.png)

### RAGAS's key evaluation metrics

The key evaluation metrics offered by RAGAS are:

- [Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html) - Scaled between 0 and 1, it measures factual consistency of the answer. The preference given to faithfulness score being closer to 1 signifying higher consistency. It is measured as :

```notion
Faithfulness score = (Number of claims in the generated answer that can be inferred from given context) / 
										 (Total number of claims in the generated answer)
```

and uses context and answer fields from .

- [Answer relevancy](https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html) - Answer relevance of an answer is based on direct alignment with the original question, not factuality. The assessment penalizes incomplete or redundant responses using cosine similarity with questions generated from the answer to gauge alignment with the original question. This metric essentially is expected to range between 0 and 1 although due to the nature of cosine similarity it can be expected in the range of -1 to 1.
  This metric is computed using the question, the context and the answer.


- [Context recall](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html) - It is computed based on the ground truth and the retrieved context. The values range between 0 and 1, with higher values indicating better performance. To accurately estimate context recall from the ground truth answer, each sentence in the ground truth answer is analyzed to determine if it aligns with the retrieved context or not. It is calculated as :

```notion
context recall = (Ground truth sentences that can be attributed to context) / 
								 (Number of sentences in Ground truth)
```

- [Context precision](https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html) - It is computed based on question, ground_truth and the contexts from the evaluation dataset. The value ranges between 0 and 1, with higher values indicating better performance.Context Precision is used to evaluate whether all relevant items in a given context are ranked highly. It is calculated as :

```notion
Context Precision@K = (Precision@K * Relevance of K) / 
										  (Total number of relevant items in the top K)

where
Relevance of K =  1 for relevant / 0 for irrelevant items
K = number of chunks
```

- [Context relevancy](https://docs.ragas.io/en/latest/concepts/metrics/context_relevancy.html) - It is computed based on the relevancy of the retrieved context with respect to the question. The value ranges between 0 and 1, with higher values indicating better relevancy. It is calculated as :

```notion
context relevancy = (Number of relevant sentences within the retrieved) / 
										(Total number of sentences in retrieved context)
```

- [Context entity recall](https://docs.ragas.io/en/latest/concepts/metrics/context_entities_recall.html) - This metric evaluates entity retrieval based on comparison with ground truth entities to ensure relevant context coverage.The value ranges between 0 and 1, with higher values indicating better entity recall. It is calculated as :

```notion
context entity recall = (Entities in context ∩ Entities in Ground Truth) / 
												(Entities in Ground Truth)
```

- [Answer semantic similarity](https://docs.ragas.io/en/latest/concepts/metrics/semantic_similarity.html) - This metric evaluates the semantic similarity between the generated answer and the ground truth. This evaluation is based on the ground truth and the answer, with values ranging between 0 and 1. It is calculated as :

```notion
answer similarity score = cosine similarity (Vector of ground truth , Vector of generated answer)
```

- [Answer correctness](https://docs.ragas.io/en/latest/concepts/metrics/answer_correctness.html) - This metric evaluates the accuracy between the generated answer and the ground truth. This evaluation is based on the ground truth and the answer, with values ranging between 0 and 1. It is calculated as :

```notion
answer correctness = factual correctness (ground truth , generated answer ) + answer similarity score

Where factual correctness is the F1 score calculated using ground truth and generated answer.
```


### **Choosing the right metrics for your use cases**

Ragas focuses on evaluating mainly Retrieval and Generation phases of your RAG pipeline and offers metrics to :

- **Evaluate Retrieval**: *context_relevancy* and *context_recall*, which give you the fair measure of the efficiency of your retrieval system.
- **Evaluate Generation**: *faithfulness* to measure the most important aspect when working with LLM i.e. hallucinations and *answer_relevancy* which measures the relevancy of the generated answer wrt to the question.

The mean score of these 4 aspects gives you the `ragas score` which is a single measure of the performance of your RAG system on the aspects of retrieval and generation.

#### **RAGAS in action**

To understand this further lets build a RAG system on top of Qdrant’s documentation and run the evaluations on top of it.
We are going to be using the pre compiled [hugging-face dataset](https://www.google.com/url?q=https://huggingface.co/datasets/atitaarora/qdrant_doc&sa=D&source=editors&ust=1712248322130408&usg=AOvVaw0Bl7gKES2qQyo88tFWPaZZ) consisting of ‘text’ and ‘source’ of the documentation.

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/dataset_preview.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/dataset_preview.png)

We process this dataset using the choice of our text splitter , chunks size and chunk overlap.

```python
# Retrieve the documents / dataset to be used
dataset = load_dataset("atitaarora/qdrant_doc", split="train")

#Process dataset as langchain document for further processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

## Dataset to langchain document
langchain_docs = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in tqdm(dataset)
]

#Document chunk processing
#Processing each document with desired TEXT_SPLITTER_ALGO , CHUNK_SIZE , CHUNK_OVERLAP etc

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = []
for doc in langchain_docs:
    docs_processed += text_splitter.split_documents([doc])
```

We are using [Fastembed](https://qdrant.github.io/fastembed) to make the encoding process of these document chunks lighter and simpler. FastEmbed is a lightweight, fast, Python library built for embedding generation and natively supports embedding creation suitable to use with Qdrant.

```python
## Declaring the intended Embedding Model with Fastembed
from fastembed.embedding import TextEmbedding

pd.DataFrame(TextEmbedding.list_supported_models())
```

You can execute above to get the list of all supported embedding models including the one for generating sparse vector encoding through [SPLADE++](https://huggingface.co/prithivida/Splade_PP_en_v1). We are using default model in this case.

```python
##Initilising embedding model
embedding_model = TextEmbedding()

## For custom model supported by Fastembed
#embedding_model = TextEmbedding(model_name="BAAI/bge-small-en", max_length=512)

## Using Default Model - BAAI/bge-small-en-v1.5
embedding_model.model_name
```

We process the document chunks into text to be processed by Fastembed as :

```python
docs_contents = []
docs_metadatas = []

for doc in docs_processed:
    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
        docs_contents.append(doc.page_content)
        docs_metadatas.append(doc.metadata)
    else:
        # Handle the case where attributes are missing
        print("Warning: Some documents do not have 'page_content' or 'metadata' attributes.")

print("content : ",len(docs_contents))
print("metadata : ",len(docs_metadatas))
```

Now we set up qdrant client to ingest these document chunk encoding into our knowledge store which is Qdrant in this case :

```python

##Uncomment to initialise qdrant client in memory
client = qdrant_client.QdrantClient(
    location=":memory:",
)

##Uncomment below to connect to Qdrant Cloud
#client = qdrant_client.QdrantClient(
#    "QDRANT_URL", 
#    api_key="QDRANT_API_KEY",
#)

## Uncomment below to connect to local Qdrant
#client = qdrant_client.QdrantClient("http://localhost:6333")
```

And define our `collection name`

```python
## Collection name that will be used throughtout in the notebook
COLLECTION_NAME = "qdrant-docs-ragas"
```

There are some collection level operations which are **Good-to-know** in case they are needed :

```python
## General Collection level operations

## Get information about existing collections
client.get_collections()

## Get information about specific collection
#collection_info = client.get_collection(COLLECTION_NAME)
#print(collection_info)

## Deleting collection , if need be
#client.delete_collection(COLLECTION_NAME)
```

Following this , to add the document chunks into Qdrant Collection is as straightforward as :

```python
client.add(collection_name=COLLECTION_NAME, metadata=docs_metadatas, documents=docs_contents)
```

We can verify the total number of document chunks indexed as :

```python
## Ensuring we have expected number of document chunks
client.count(collection_name=COLLECTION_NAME)

##Outputs
#CountResult(count=4431)
```

You can test searching for the document as :

```python
## Searching for document chunks similar to query for context
search_result = client.query(collection_name=COLLECTION_NAME, query_text="what is binary quantization",limit=2)
context = ["document:"+r.document+",source:"+r.metadata['source'] for r in search_result]

for res in search_result:
    print("Id: ", res.id)
    print("Document : " , res.document)
    print("Score : " , res.score)
    print("Source : " , res.metadata['source'])
```

Now that we’re all set with our knowledge store to provide context to our RAG , lets quickly build it. We are going to write a small sub routine to wrap our prompt , context and response collections.

```python
def query_with_context(query):
    
    ## Fetch context from Qdrant
    search_result = client.query(collection_name=COLLECTION_NAME, query_text=query, limit=4)
    
    contexts = [
        "document:"+r.document+",source:"+r.metadata['source'] for r in search_result
    ]
    prompt_start = (
       """ You're assisting a user who has a question based on the documentation.
        Your goal is to provide a clear and concise response that addresses their query while referencing relevant information 
        from the documentation.
        Remember to: 
        Understand the user's question thoroughly.
        If the user's query is general (e.g., "hi," "good morning"), 
        greet them normally and avoid using the context from the documentation.
        If the user's query is specific and related to the documentation, locate and extract the pertinent information.
        Craft a response that directly addresses the user's query and provides accurate information
        referring the relevant source and page from the 'source' field of fetched context from the documentation to support your answer.
        Use a friendly and professional tone in your response.
        If you cannot find the answer in the provided context, do not pretend to know it.
        Instead, respond with "I don't know".
        
        Context:\n"""
    )
    
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    
    prompt = (
        prompt_start + "\n\n---\n\n".join(contexts) + 
        prompt_end
    )

    res = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0,
        max_tokens=636,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    return (res.choices[0].text)
```

Try out running your first query through your RAG pipeline as :

```python
print (query_with_context(query = "what is quantization?"))

##Outputs 
#Quantization is a process that converts traditional float32 vectors into qbit vectors, which can then be used in quantum computing to speed up the search process in ANNs. 
#This is explained in detail in the article "Quantum Quantization and Entanglement" (source: articles/quantum-quantization.md). 
#Additionally, you can also check out the blog post "Binary Quantization and Vector Space Talk" by Andrey Vasnetsov (source: blog/binary-quantization-andrey-vasnetsov-vector-space-talk-001.md) for more information on how this process works.
```

One key thing to understand is that RAGAS work inherently with datasets for evaluations for it is a good idea to create a set of questions and ground truth to help create an evaluation dataset to assess our RAG system. We are leveraging the dataset [https://huggingface.co/datasets/atitaarora/qdrant_doc_qna](https://huggingface.co/datasets/atitaarora/qdrant_doc_qna) to create our evaluation set from our RAG pipeline and execute it as shown [before](https://www.notion.so/Evaluating-RAG-with-RAGAS-ef0de436a00748c7bba9fd7161cd76cc?pvs=21).

**Lets evaluate , how good is our RAG !!**

Its as simple as importing the desired metrics:

```python
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
```

And running the evaluation as :

```python
result = evaluate(
    rag_response_dataset,
    metrics=[
        answer_correctness,
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision
    ],
)

## Lets look at our results

evaluation_result_df_chunk_size_512 = result.to_pandas()

evaluation_result_df_chunk_size_512.head(5)
```

Which outputs as below :

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/rag_eval_ragas.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/rag_eval_ragas.png)

We notice in some of our queries the context precision and answer correctness is quite low (or even zero). Which indicates the prospective issue with the retrieval chain of our RAG system.

This issue could be impacted by :

- Number of chunks retrieved
- Size of chunks
- Chunk overlap size
- Choice of Embedding model

Let’s try to change the size of document chunk retrieval from **4** to **5** and find out if it improves our metrics. We will make a change to our RAG with a context method to retrieve **5** instead of **4** chunks and recreate the evaluation dataset.
Followed by re-running the RAGAS evaluations.

We add some visualisations to enable smoother comparison and this is what it looks after the changes.

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/algo_comparison_chart.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_ragas/algo_comparison_chart.png)

We can see the metrics impacted by this change and can assess if they are moving in the desired direction. Please remember this is an iterative process and you can try our other suggested changes to evaluate what works better for your given usecase.

The complete version of this code and notebook is available at - [https://github.com/qdrant/qdrant-rag-eval/tree/master/workshop-rag-eval-qdrant-ragas](https://github.com/qdrant/qdrant-rag-eval/tree/master/workshop-rag-eval-qdrant-ragas).

Don’t forget to star and contribute your experiments. Goes without saying that any feedback or improvement is very welcome.

The next in series is the similar code walk through of using [Arize Phoenix](https://phoenix.arize.com/) with your RAG system.

Until next time - *Stay tuned* !

## Contributors 
 
- [Atita Arora, Author](https://www.linkedin.com/in/atitaarora/) 
- [Mór Kapronczay, Editor](https://www.linkedin.com/in/mór-kapronczay-49447692) 
- [Robert Turner, Editor](https://www.linkedin.com/in/robertdhayanturner) 
