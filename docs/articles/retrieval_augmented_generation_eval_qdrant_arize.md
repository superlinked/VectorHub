# RAG Evaluation with Arize Phoenix

Following the series :

[Principles of RAG Evaluation](https://superlinked.com/vectorhub/articles/evaluating-retrieval-augmented-generation-framework)

[RAG Evaluation with RAGAS](https://superlinked.com/vectorhub/articles/retrieval-augmented-generation-eval-qdrant-ragas)

For this article we will take you through RAG Evaluation using another popular framework tool called Arize Phoenix. It is an open source tool to evaluate and observe LLM applications, including capabilities to determine the relevance or irrelevance of the documents retrieved by RAG application. It also assists with monitoring models and LLM applications, providing LLMOps fast insights with zero-config observability.

Please check out Phoenix [github]([https://github.com/Arize-ai/phoenix](https://github.com/Arize-ai/phoenix)) or [website]([https://phoenix.arize.com/](https://phoenix.arize.com/)) for more information.

Let us talk about the RAG evaluation aspect of Phoenix in this article.

Just like our [previous article]([https://superlinked.com/vectorhub/articles/retrieval-augmented-generation-eval-qdrant-ragas](https://superlinked.com/vectorhub/articles/retrieval-augmented-generation-eval-qdrant-ragas)) , we’ll be building a RAG pipeline and evaluate it using Phoenix.

The code is available in the [github repo]([https://github.com/qdrant/qdrant-rag-eval/tree/master/workshop-rag-eval-qdrant-arize](https://github.com/qdrant/qdrant-rag-eval/tree/master/workshop-rag-eval-qdrant-arize)) if you prefer to follow along.

There is also a walkthrough workshop [video]([https://www.youtube.com/watch?v=m_J0nFmnrPI](https://www.youtube.com/watch?v=m_J0nFmnrPI)) available for the reference.

This Article will take you through :

- Learning about key concepts and metrics in Phoenix
- Evaluating a Naive RAG using Phoenix
- Key observations
- Building a [Hybrid RAG]([https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking))
- Evaluating a Hybrid RAG using Phoenix
- Concluding thoughts

Lets begin!

**Learning about key concepts and metrics in Phoenix**

Phoenix is an open-source observability library designed for experimentation, evaluation, and troubleshooting. It allows AI Engineers to quickly visualize their data, evaluate performance, track down issues and optimize.

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/arize_products.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/arize_products.png)

It provides a platform to enable evaluation and observability across different layers and stages of an LLM-based system.

Arize offers a comprehensive suite of solutions to meet various needs, from open-source to enterprise-grade offerings. This [LLM tracing](https://docs.arize.com/phoenix/tracing/llm-traces-1) application provides visibility into the topology of your application whether you're running it locally in Jupyter notebook or a Docker container, or pushing your application into staging and production, the solutions provide unparalleled observability into every aspect of your LLM system.

This includes tracing, prompt iteration, search and retrieval, [evaluating specific tasks](https://arize.com/blog-course/llm-evaluation-the-definitive-guide/), and ultimately improving your LLM through fine-tuning.

This article covers **Phoenix**, which is the open-source solution.

The complexity that is involved in building an LLM application is why observability is so important. Each step of the response generation process needs to be monitored, evaluated and tuned to provide the best possible experience. Not only that, certain tradeoffs might need to be made to optimize for speed, cost, or accuracy. In the context of LLM applications, we need to be able to understand the internal state of the system by examining telemetry data such as LLM Traces.

**Traces** are made up of a sequence of `spans`. A span represents a unit of work or operation (think a span of time). It tracks specific operations that a request makes, painting a picture of what happened during the time in which that operation was executed.

LLM Tracing supports the following span kinds:

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/arize_spans_types.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/arize_spans_types.png)

By capturing the building blocks of your application while it's running, Phoenix can provide a more complete picture of the inner workings of your application.

**Evaluating a Naive RAG using Phoenix**

To illustrate this, let's look at an example Naive RAG application similar to the one we built in our [previous article]([https://superlinked.com/vectorhub/articles/retrieval-augmented-generation-eval-qdrant-ragas](https://superlinked.com/vectorhub/articles/retrieval-augmented-generation-eval-qdrant-ragas)) and evaluate it using Phoenix.

We will use the collection name as below :

```python
## Collection Name 
COLLECTION_NAME = "qdrant_docs_arize_dense"
```

to index our Naive RAG documents as below :

```python
## Initializing the space to work with llama-index and related settings
import llama_index
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from phoenix.trace import suppress_tracing
## Uncomment it if you'd like to use FastEmbed instead of OpenAI
## For the complete list of supported models,
##please check https://qdrant.github.io/fastembed/examples/Supported_Models/
from llama_index.embeddings.fastembed import FastEmbedEmbedding

vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

## Uncomment if using FastEmbed
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

## Uncomment it if you'd like to use OpenAI Embeddings instead of FastEmbed
#Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

Settings.llm = OpenAI(model="gpt-4-1106-preview", temperature=0.0)

```

and pushing the docs into our Qdrant collection as :

```python
## Indexing the vectors into Qdrant collection 
from phoenix.trace import suppress_tracing
with suppress_tracing():
  dense_vector_index = VectorStoreIndex.from_documents(
      documents,
      storage_context=storage_context,
      show_progress=True
  )
```

Once the data is pushed in, we will create our `dense-vector retriever` as below :

```python
##Initialise retriever to interact with the Qdrant collection
dense_retriever = VectorIndexRetriever(
    index=dense_vector_index,
    vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
    similarity_top_k=2
)
```

which we’ll use through Phoenix to evaluate our Naive Rag pipeline.

We'll use the pre-compiled **[hugging-face dataset](https://huggingface.co/datasets/atitaarora/qdrant_doc)**, which consists of `text` and `source`, derived from the documentation as before in our previous article, for the evaluation.

Let’s load our eval dataset as :

```python
## Loading the Eval dataset
from datasets import load_dataset
qdrant_qa = load_dataset("atitaarora/qdrant_doc_qna", split="train")
qdrant_qa_question = qdrant_qa.select_columns(['question'])
qdrant_qa_question['question'][:10]

#Outputs
['What is vaccum optimizer ?',
 'Tell me about ‘always_ram’ parameter?',
 'What is difference between scalar and product quantization?',
 'What is ‘best_score’ strategy?',
 'How does oversampling helps?',
 'What is the purpose of ‘CreatePayloadIndexAsync’?',
 'What is the purpose of ef_construct in HNSW ?',
 'How do you use ‘ordering’ parameter?',
 'What is significance of ‘on_disk_payload’ setting?',
 'What is the impact of ‘write_consistency_factor’ ?']

```

Next , we will define our response synthesizer and associate it with  `query engine` which will facilitate collection of traces for the evaluations.

```python
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

#define response synthesizer
response_synthesizer = get_response_synthesizer()

#assemble query engine for dense retriever
dense_query_engine = RetrieverQueryEngine(
                     retriever=dense_retriever,
                     response_synthesizer=response_synthesizer,)
#query_engine = index.as_query_engine()
for query in tqdm(qdrant_qa_question['question'][:10]):
    try:
      dense_query_engine.query(query)
    except Exception as e:
      pass
```

Open the Phoenix UI with the [link]([http://localhost:6006/](http://localhost:6006/)) and click through the queries to better understand how the query engine is performing.

Check the Phoenix UI as your queries run. Your traces should appear in real time.

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run1.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run1.png)

For each trace notice a span breakdown and the list of all the spans can also be seen by clicking on spans tab as below :

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run1_spans.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run1_spans.png)

Phoenix can be used to understand and troubleshoot your application by surfacing:

- **Application latency** - highlighting slow invocations of LLMs, retrievers, etc.
- **Token Usage** - Displays the breakdown of token usage with LLMs to surface up your most expensive LLM calls
- **Runtime Exceptions** - Critical runtime exceptions such as rate-limiting are captured as exception events.
- **Retrieved Documents** - view all the documents retrieved during a retriever call and the score and order in which they were returned
- **Embeddings** - view the embedding text used for retrieval and the underlying embedding model LLM parameters - view the parameters used when calling out to an LLM to debug things like temperature and the system prompts
- **Prompt Templates** - Figure out what prompt template is used during the prompting step and what variables were used
- **Tool Descriptions** - view the description and function signature of the tools your LLM has been given access to
- **LLM Function Calls** - if using OpenAI or other a model with function calls, you can view the function selection and function messages in the input messages to the LLM.

You can export your trace data as a pandas dataframe for further analysis and evaluation.

In this case, we will export our retriever spans into two separate dataframes:

**queries_df**, in which the retrieved documents for each query are concatenated into a single column, retrieved_documents_df, in which each retrieved document is "exploded" into its own row to enable the evaluation of each query-document pair in isolation. This will enable us to compute multiple kinds of evaluations, including:

**relevance:** Are the retrieved documents grounded in the response? Q&A correctness: Are your application's responses grounded in the retrieved context?

**hallucinations:** Is your application making up false information?

We can do this as :

```jsx
queries_df = get_qa_with_reference(px.Client())
retrieved_documents_df = get_retrieved_documents(px.Client())
```

Next, we define our evaluation model and evaluators.

Evaluators are built on top of language models and prompt the LLM to assess the quality of responses, the relevance of retrieved documents, etc., and provide a quality signal even in the absence of human-labeled data.

Pick an evaluator type and instantiate it with the language model you want to use to perform evaluations using our battle-tested evaluation templates. We can do this as :

```python
eval_model = OpenAIModel(
    model="gpt-4-turbo-preview",
)
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QAEvaluator(eval_model)
relevance_evaluator = RelevanceEvaluator(eval_model)

hallucination_eval_df, qa_correctness_eval_df = run_evals(
    dataframe=queries_df,
    evaluators=[hallucination_evaluator, qa_correctness_evaluator],
    provide_explanation=True,
)
relevance_eval_df = run_evals(
    dataframe=retrieved_documents_df,
    evaluators=[relevance_evaluator],
    provide_explanation=True,
)[0]

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df),
    SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval_df),
)
px.Client().log_evaluations(DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df))
```

The evaluations should now appear as annotations on the appropriate spans in Phoenix.

**Key Observations**

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run_evals.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run_evals.png)

This is the aggregated view where we can see `Total Traces` (which is essentially the total number of eval queries run) , `Total Tokens` processed , `Latency P50` , `Latency P99` , `Hallucination`  and `QA Correctness`. Along with Overall `Relevance` metrics like *nDCG , Precision and Recall.*

For the given evaluations for Naive RAG we have : `Hallucination` = 18% and `QA Correctness` = 91%. Lets find out the problematic query/ies and address the issues to achieve our aim - **0% Hallucination and 100% QA Correctness**.

Notice the `evaluations` for each eval questions which captures these for each query too which makes spotting the problematic queries instantly.

We can zoom into each trace  and inspect the spans there-in to provide us with further details on each query and the output from the configured retriever.

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run_trace_details.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run_trace_details.png)

If we inspect into the `Evaluations` tab , we’ll notice that the response was *Hallucinated* even though it is correct per *QA Correctness* metrics.

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run_trace_detail_eval.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/naive_rag_run_trace_detail_eval.png)

This can happen for various reasons, one of which could be **domain / use case specific terminology** where we may **require exact matches**.

So, our next attempt will be to enrich our retriever to be powered by **Dense** as well as **Sparse vectors** to **retrieve content based on semantic similarity as well as the exact matches**.

**Building a Hybrid RAG**

Lets build a new vector index for the same.

```python
## Define a new collection to store your hybrid emebeddings
COLLECTION_NAME_HYBRID = "qdrant_docs_arize_hybrid"
```

For this purpose , we’ll need to use a sparse vector model , we’ll use [Splade++]([https://huggingface.co/prithivida/Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1)) in this case supported through Fastembed as we can verify below :

```python
##List of supported sparse vector models
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
SparseTextEmbedding.list_supported_models()

Outputs
{'model': 'prithivida/Splade_PP_en_v1',
  'vocab_size': 30522,
  'description': 'Independent Implementation of SPLADE++ Model for English',
  'size_in_GB': 0.532,
  'sources': {'hf': 'Qdrant/SPLADE_PP_en_v1'}}
```

We ingest sparse and dense vectors into Qdrant Collection.

We are using **Splade++ model for Sparse Vector** Model and default Fastembed model - **bge-small-en-1.5 for Dense embeddings**.

```python
## Initializing the space to work with llama-index and related settings
import llama_index
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding, SparseEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from typing import List, Tuple

sparse_model_name = "prithivida/Splade_PP_en_v1"

# This triggers the model download
sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)

## Computing sparse vectors
def compute_sparse_vectors(
    texts: List[str],
    ) -> Tuple[List[List[int]], List[List[float]]]:
    indices, values = [], []
    for embedding in sparse_model.embed(texts):
        indices.append(embedding.indices.tolist())
        values.append(embedding.values.tolist())
    return indices, values

## Creating a vector store with Hybrid search enabled
hybrid_vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME_HYBRID,
    enable_hybrid=True,
    sparse_doc_fn=compute_sparse_vectors,
    sparse_query_fn=compute_sparse_vectors)

storage_context = StorageContext.from_defaults(vector_store=hybrid_vector_store)

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

Following above snippet , we push the sparse and dense vectors into our hybrid vector Qdrant collection , as below :

```python
## Note : Ingesting sparse and dense vectors into Qdrant collection
from phoenix.trace import suppress_tracing
with suppress_tracing():
    hybrid_vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
```

You may notice that for some snippets above , we are using :

```python
from phoenix.trace import suppress_tracing
with suppress_tracing():
```

which essentially indicates to phoenix to suppress monitoring for the given code.

Similar to the Naive / Simple RAG before, lets build a retriever for this Hybrid index, a Hybrid Retriever.

As an add on , you can try interacting with Sparse index through sparse retriever too as below:

```python
## Before trying Hybrid search , lets try Sparse Vector Search Retriever 
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.indices.vector_store import VectorIndexRetriever

sparse_retriever = VectorIndexRetriever(
    index=hybrid_vector_index,
    vector_store_query_mode=VectorStoreQueryMode.SPARSE,
    sparse_top_k=2,
)

## Pure sparse vector search
nodes = sparse_retriever.retrieve("What is a Merge Optimizer?")
for i, node in enumerate(nodes):
    print(i + 1, node.text, end="\n")
```

which should retrieve the documents based on `exact match`.

```python
## Let's try Hybrid Search Retriever now using the 'alpha' parameter that controls the weightage between
## the dense and sparse vector search scores.
# NOTE: For hybrid search (0 for sparse search, 1 for dense search)

hybrid_retriever = VectorIndexRetriever(
    index=hybrid_vector_index,
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    sparse_top_k=1,
    similarity_top_k=2,
    alpha=0.1,
)
```

The `alpha` parameter is the magic button here that slides between sparse and dense vector search. For our retriever we have `alpha=0.1` which means 90% of the score accounts from sparse vector search and 10% from the dense vector search, of course we can try different distribution as relevant to the given use case.

Although, it is not advised but should you want to try different value of `alpha` you can do that as:

```python
hybrid_retriever._alpha = 0.1
#or
hybrid_retriever._alpha = 0.9
```

Next , we’ll build a query engine on top of our **Hybrid Retriever as :**

```python
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

#define response synthesizer
response_synthesizer = get_response_synthesizer()

#assemble query engine for hybrid retriever
hybrid_query_engine = RetrieverQueryEngine(
                        retriever=hybrid_retriever,
                        response_synthesizer=response_synthesizer,)
```

which we use to evaluate as before for the same set of eval questions.

<aside>
💡 A noteworthy feature in Phoenix is to switch workspace / projects on-the-fly which is a useful feature to segregate different experiments.

</aside>

**Evaluating a Hybrid RAG using Phoenix**

We will use a separate project / workspace for the Hybrid Search evaluations as :

```python
## Switching phoenix project space
from phoenix.trace import using_project

# Switch project to run evals
with using_project(HYBRID_RAG_PROJECT)
```

and run our evaluation as :

```python
## Switching phoenix project space
from phoenix.trace import using_project

# Switch project to run evals
with using_project(HYBRID_RAG_PROJECT):
# All spans created within this context will be associated with the `HYBRID_RAG_PROJECT` project.

    ##Reuse the previously loaded dataset `qdrant_qa_question`
    
    for query in tqdm(qdrant_qa_question['question'][:10]):
        try:
          hybrid_query_engine.query(query)
        except Exception as e:
          pass
```

We will log and store our eval df’s as before :

```python
## Switching phoenix project space
from phoenix.trace import using_project

queries_df_hybrid = get_qa_with_reference(px.Client(), project_name=HYBRID_RAG_PROJECT)
retrieved_documents_df_hybrid = get_retrieved_documents(px.Client(), project_name=HYBRID_RAG_PROJECT)
```

finally defining our Evaluators for the Hybrid RAG as :

```python

# all spans created within this context will be associated with the `HYBRID_RAG_PROJECT` project.
eval_model = OpenAIModel(
    model="gpt-4-turbo-preview",
)
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QAEvaluator(eval_model)
relevance_evaluator = RelevanceEvaluator(eval_model)

hallucination_eval_df_hybrid, qa_correctness_eval_df_hybrid = run_evals(
    dataframe=queries_df_hybrid,
    evaluators=[hallucination_evaluator, qa_correctness_evaluator],
    provide_explanation=True,
)
relevance_eval_df_hybrid = run_evals(
    dataframe=retrieved_documents_df_hybrid,
    evaluators=[relevance_evaluator],
    provide_explanation=True,
)[0]

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df_hybrid),
    SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval_df_hybrid),
    project_name=HYBRID_RAG_PROJECT,
)
px.Client().log_evaluations(DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df_hybrid),
                            project_name=HYBRID_RAG_PROJECT)
```

You will notice your project will now have 2 workspaces as below :

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/arize_project_workspace_with_2_projects.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/arize_project_workspace_with_2_projects.png)

Lets dig into our hybrid-rag workspace.

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/arize_hybrid_rag_eval.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/arize_hybrid_rag_eval.png)

Let’s take a quick look at the stats and we notice :

`Hallucination` is 0.00 and `QA Correctness` is 1.00 .

Also , notice the question `What is the purpose of ef_construct in HNSW ?` which earlier showed Hallucinated, now shows as :

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/hybrid_rag_trace_detail.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/hybrid_rag_trace_detail.png)

and in the `Evaluation` tab :

![../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/hybrid_rag_trace_eval_tab.png](../assets/use_cases/retrieval_augmented_generation_eval_qdrant_arize/hybrid_rag_trace_eval_tab.png)

seem to be already fixed.

**Concluding thoughts**

In the [previous article]([https://superlinked.com/vectorhub/articles/retrieval-augmented-generation-eval-qdrant-ragas](https://superlinked.com/vectorhub/articles/retrieval-augmented-generation-eval-qdrant-ragas)) we experimented improving RAG Evaluation score by experimenting with different *Retrieval Window sizes* while in this article we covered *Multivector / Hybrid vector* approach to improve and fix our RAG pipeline.

It is suggested to correlate these experiments with your use case.

There are other ways of improving your RAG pipeline ,  such as :

- Experimenting with *different chunk sizes and chunk overlap settings* - based on the nature of queries and response expectations.
- Experimenting with *different embedding model and LLMs* - based on domain understanding and response capabilities as required
- Experimenting with *rerankers* - based of preset or custom relevance criterion
- Experimenting with *query and response enrichment* - based on level of enrichment required before and after retrieval state to help LLM generate the desired response.

The complete code for this article can be found [here]([https://github.com/qdrant/qdrant-rag-eval/tree/master/workshop-rag-eval-qdrant-arize](https://github.com/qdrant/qdrant-rag-eval/tree/master/workshop-rag-eval-qdrant-arize)).

We hope you found this series useful! Don’t forget to star and contribute your experiments. Feedback and suggestions are welcome.