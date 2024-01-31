<!-- SEO: Retrieval augmented generation Evaluation - TODO Summary
 -->

# Evaluating Retrieval Augmented Generation


## Understanding Retrieval-Augmented Generation

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_qdrant.jpg alt="Implementation of RAG using Qdrant as a vector database" data-size="100" />

**Disclaimer**:
The provided implementation serves as an example using  [Qdrant]("https://qdrant.tech"). Alternative Vector Databases are also available. To determine the most suitable Vector Database for your specific use case, please refer to the [Vector DB feature matrix]("https://vdbs.superlinked.com/")

**RAG** stands for **Retrieval Augmented Generation**, and it probably is the most useful application of large language models lately.
It is the technique that combines the strengths of both retrieval and generation models. The retrieval is usually based on dense vector search, in combination with a text generation model like GPT.

RAG has received significant attention due to its ability to enhance content generation by leveraging existing information effectively. 
Its capacity to amalgamate specific, relevant details from multiple sources and generate accurate and relevant content has a lot of potential in various domains like content creation, question & answer application, and information synthesis.

Read more: [here]("https://hub.superlinked.com/retrieval-augmented-generation")

## Why do we need RAG?

You can leverage RAG to enhance [vector search](https://qdrant.tech/documentation/overview/vector-search/), allowing users to generate new content based on the retrieved knowledge.

RAG proves invaluable for tasks requiring detailed, coherent explanations, summaries, or responses that transcend the explicit data stored in vectors. 
While vector search efficiently finds similar items, RAG takes a step further, offering content synthesis and a deeper level of understanding.

If the question is : "When is vector search alone not enough?" 
RAG becomes crucial when users seek to generate new content rather than interact with documents or search results directly. 
It excels in providing contextually rich, informative, and human-like responses, making it an ideal solution for a comprehensive and versatile approach when integrated with vector search.

Your next step is to conduct feasibility studies and correlate the results with your business needs and value expectations.

## Evaluating Retrieval-Augmented Generation

Our experience scaling RAG from Proof-of-concept (POC) to production taught us
about the unknowns. The lesson is to improve and optimize your solution through evaluation.

### Why do you need to evaluate?

Evaluation is a very crucial step to distill a sense of Trust in your application, which establishes reputation, and boosts team morale and confidence.
Needless to say, it also validates that your applications avoid common pitfalls, and it is no different Evaluation is key. It distills a sense of Trust in your application,
which establishes reputation, and boosts team morale and confidence.
It also validates that your applications avoid common pitfalls.
The aspect of relevance assessment through DCG / nDCG or machine learning model through train-validation-test split/overfitting/underfitting analysis addressed similar concerns.

And for some who think LLMs are perfect, they can and do make mistakes too!! 🙂
One problem is that the 'data' frequently changes. Evaluation helps ensure the results are consistent with user expectations.

### If you can't quantify it, you can't improve it!!

<br>For RAG it can be restated as :

### If you can't retrieve it, you can't generate it!!

Our experience has taught us that it is better to evaluate each component in isolation.

As for different components of RAG viz. Information Retrieval - Context Augmentation - Response Generation , we classified challenges of RAG as below :

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_challenges.png alt="Classification of Challenges of RAG Evaluation" data-size="100" />

For evaluation, we need metrics/methodology/frameworks that uses one or more
tools. The tools that you use should cover each component of RAG. Such coverage
ensures granular and thorough measurements.

Evaluation metrics can assess:

- Retrieval effectiveness
- Coherence of generated responses
- Relevance to the retrieved information.

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_granular.jpg alt="Granular Levels of Evaluation of RAG" data-size="100" />

### Let's dive in !!

## Strategies for Evaluation

Now that we have defined different challenges and levels at which we can break down the RAG evaluation, let us zoom into the levels individually. 

### Model Evaluation 

To evaluate our model, we start with this [Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb#leaderboard). We want to ensure that
the model can understand the data that we encode.
The benchmark shown above leverages different public/private datasets to evaluate and report on the different capabilities of individual models.
If you're working with specialized domains, you may want to put together a
specialized dataset to teach the model.

We could either leverage the results here (provided our model belongs to this list) or run relevant 'tasks' for our custom model (instructions provided in the link)

```python
import logging

from mteb import MTEB
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)


model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)
evaluation = MTEB(task_langs=["en"])
evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=["test"])

print("--DONE--")
```

### Data Ingestion Evaluation

We first validate the model, and train it on the language of our domain. 
We can then configure data ingestion into our semantic retrieval store aka vector store.
Various vector databases offer index configurations to influence and enhance the retrieval quality, based on the supported index types (Flat , LSH , HNSW and IVF).
One such example here to improve HNSW [retrieval-quality]("https://qdrant.tech/documentation/tutorials/retrieval-quality/").

To evaluate ingestion, we need to focus on related variables, such as:

* **Chunk size** - represents the size of each segment. This depends on the token limit of our embedding model. It also has a substantial impact on the contextual understanding of the data. That impacts the precision, recall, and relevancy of our results.

* **Chunk overlap** - represents the presence of overlapping information fragments in each segment. This helps with context retention of the information chunks but at the same time must be used with relevant strategies like deduplication, and content normalization to eradicate adverse effects.

* **Chunking/Text splitting strategy** - represents the process of data splitting and further treatment as mentioned above.

A [utility]("https://chunkviz.up.railway.app/") like this seem useful to visualise your apparent chunks.

### Semantic Retrieval Evaluation

The next step is semantic retrieval evaluation. It puts the work you've done so far to a litmus test.

Retrieval is the vital component of the RAG and its evaluation can be addressed as a classic information retrieval evaluation problem.

You need to establish the expectations from the returned results. This can help
us to identify reference metrics and important parameters. Done correctly, it
helps us determine if the documents retrieved at this stage are relevant
results.

We have several existing metrics to guide and define our baseline:

- Precision and Recall or their combination F1 Score
- DCG and nDCG, which relates to their relevance, based on the inclusion or rank of documents in the results

The nature of semantic information retrieval poses a challenge at this stage, as the documents are retrieved beyond the keywords/synonyms/token enrichment-matching.

You can build a reference evaluation set, known as a [Golden Set]("https://www.luigisbox.com/search-glossary/golden-set/"). We could also leverage [T5 Model]("https://huggingface.co/docs/transformers/model_doc/t5") to generate a starter pack for evaluation.

The golden set is a fundamental component in information retrieval evaluation.
It helps in define characteristics of a reference standard to assess the performance, effectiveness, and relevance of a selected retrieval algorithm.
It provides a common ground for objective comparison and improvement of a retrieval process/algorithm.


### End-to-End Evaluation

An end-to-end evaluation covers the response generation of the question. It leverages the provided context through retrieved documents.

Evaluating the quality of responses produced by large language models can be a challenge due to various factors as described above. 

By virtue of nature and design , the answers generated rely on diversity of response which makes it impossible to device a fixed metric or methodology that fits in all domains and use-cases.

To address these difficulties, you may use a blend of existing metrics like the following scores: 

- [BLEU]("https://huggingface.co/spaces/evaluate-metric/bleu") 
- [ROUGE]("https://huggingface.co/spaces/evaluate-metric/rouge") 

You can then combine these score with LLM-based or human evaluation methods.
For more information, see this paper which provides great ideas to establish [Quality Criteria]("https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/c6a53998-09e3-4d17-91fd-c7416d51b250/content") for the same.

To summarize, you want to establish methods to automate evaluating similarity
and content overlap between generated response and reference summaries. You can
also leverage human evaluation to evaluate subjective information, such as context-relevance, novelty, and fluency.

You can also build a classified 'domain - questions' set based on question complexity (easy, medium, hard). It can give you an overall sense of the RAG performance.

Nevertheless, it is a challenge to formulate a comprehensive set of metrics. It
has to appraise the quality of responses from LLMs. It remains an ongoing and intricate issue in the context of natural language processing.

We have laid the foundation, and we know more about the layers of evaluation. 

In the next article, we will describe how you can demystify existing frameworks.

Looking forward to seeing you in the next part!
