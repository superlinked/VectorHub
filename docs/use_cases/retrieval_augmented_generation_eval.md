<!-- SEO: Retrieval augmented generation Evaluation - TODO Summary
 -->

# Evaluating Retrieval Augmented Generation


## Understanding Retrieval-Augmented Generation

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_qdrant.jpg alt="Implementation of RAG using Qdrant as a vector database" data-size="100" />

**Disclaimer**:
The provided implementation serves as an example using  [Qdrant](https://qdrant.tech). Alternative Vector Databases are also available. To determine the most suitable Vector Database for your specific use case, please refer to the [Vector DB feature matrix](https://vdbs.superlinked.com/)

**RAG** stands for **Retrieval Augmented Generation**, and it probably is the most useful application of large language models lately.
It is the technique that combines the strengths of both retrieval and generation models. The retrieval is usually based on dense vector search, in combination with a text generation model like GPT.

RAG has received significant attention due to its ability to enhance content generation by leveraging existing information effectively. 
Its capacity to amalgamate specific, relevant details from multiple sources and generate accurate and relevant content has a lot of potential in various domains like content creation, question & answer application, and information synthesis.

Read more: [here](https://hub.superlinked.com/retrieval-augmented-generation)

## Why do we need RAG?

RAG plays a crucial role in significantly enhancing [vector search](https://qdrant.tech/documentation/overview/vector-search/) with the power of Large Language Model (LLM), 
enabling dynamic content generation based on the retrieved knowledge.
It proves invaluable for tasks requiring detailed, coherent explanations, summaries, or responses that transcend the explicit data stored in vectors.

While vector search efficiently retrieves relevant similar documents/chunks, RAG takes a step further, providing unique and tailored answers for each query.
This distinction ensures that every response is not only relevant but also personalized, offering content synthesis and a deeper level of understanding beyond document retrieval,
contributing to an enriched user experience.

If the question is : "When is vector search alone not enough?" 
RAG becomes indispensable when users seek to generate new content rather than interact with documents or search results directly. 
It excels in providing contextually rich, informative, and human-like responses, making it an ideal solution for a comprehensive and versatile approach when integrated with vector search.

*While we've highlighted the substantial benefits of RAG, it's essential to note that its effectiveness may vary based on your unique business requirements. 
Consider conducting feasibility studies to ensure that RAG aligns to your specific needs and correlates with your value expectations.*

## Evaluating Retrieval-Augmented Generation

Our experience scaling RAG from Proof-of-concept (POC) to production taught us
about the unknowns. The lesson is to improve and optimize your solution through evaluation.

### Why do you need to evaluate?

Evaluation is key. It distills a sense of Trust in your application,
which establishes reputation, and boosts team morale and confidence.
It also validates that your applications avoid common pitfalls.
As we know, LLMs can make mistakes!! 🙂

The 'data' you leverage to build your RAG is likely to be dynamic in nature which may often undergo frequent changes.
Not only are there going to be new queries , this can include data used to build response - like real-time data streams , economic fluctuations , business metrics or research data to name a few. 
Evaluation helps ensure the results are consistent with user expectations.

> If you can't quantify it, you can't improve it!!

For RAG it can be restated as :

> If you can't retrieve it, you can't generate it!!

Our experience has taught us that it is better to evaluate each component in isolation.

As for different components of RAG viz. Information Retrieval , Context Augmentation and Response Generation , we classified challenges of RAG as below :

<figure>
<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_challenges.jpg alt="Classification of Challenges of RAG Evaluation" data-size="100" />
<figcaption align="center">Slide from RAG Evaluation Presentation mentions <a href="https://arxiv.org/abs/2307.03172">'Lost in the Middle' problem</a></figcaption>
</figure>

For effective evaluation, we propose a framework such that the coverage ensures granular and thorough measurements.

Evaluation metrics can assess:

- Retrieval effectiveness - as a measure of accuracy of the retrieved relevant information from the underlying vector database that aligns with user's query intent.  
- Relevance to the retrieved information - as a measure of generated responses being meaningful and aligned with the content and context of retrieved information.
- Coherence of generated responses - as a measure of generated responses being logically connected , fluent and contextually consistent with the user's query.

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_granular.jpg alt="Granular Levels of Evaluation of RAG" data-size="100" />

**Let's dive in !!**

## Strategies for Evaluation

Now that we have defined different challenges and levels at which we can break down the RAG evaluation, let us zoom into the levels individually. 

### Model Evaluation 

To evaluate our model, we start with this [Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard). 
We want to ensure that the model can understand the data that we encode.
The benchmark shown above leverages different public/private datasets to evaluate and report on the different capabilities of individual models.
If you're working with specialized domains, you may want to put together a specialized dataset to train the model.

We could either leverage the results here (provided our model belongs to this list) or run relevant 'tasks' for our custom model (instructions provided in the github [link](https://github.com/embeddings-benchmark/mteb#leaderboard))

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
One such example here to improve HNSW [retrieval-quality](https://qdrant.tech/documentation/tutorials/retrieval-quality/).

To evaluate ingestion, we need to focus on related variables, such as:

* **Chunk size** - represents the size of each segment. This depends on the token limit of our embedding model. It also has a substantial control on the granularity of the information and impact on the contextual understanding of the data. That impacts the precision, recall, and relevancy of our results.

* **Chunk overlap** - represents the presence of overlapping information fragments in each segment. This helps with context retention of the information chunks but at the same time must be used with relevant strategies like deduplication, and content normalization to eradicate adverse effects.

* **Chunking/Text splitting strategy** - represents the process of data splitting and further treatment based on the type of data for e.g. html , markdown , code , or pdf combined with nuances from the use-case like a summarization use-case may split segments based on chapters or paragraphs , legal document assistant may divide documents into sections based on headings and subsections , medical literature assistant may split them based on sentence boundaries or key-concepts. 

A [utility](https://chunkviz.up.railway.app/) like this seems useful to visualise your apparent chunks.

### Semantic Retrieval Evaluation

The next step is semantic retrieval evaluation. It puts the work you've done so far to a litmus test.

Retrieval is a vital component of RAG and its evaluation can be addressed as a classic information retrieval evaluation problem.

You need to establish the expectations from the returned results. This can help
us to identify reference metrics and important parameters. Done correctly, it
helps us determine if the documents retrieved at this stage are relevant
results.

We have several existing metrics to guide and define our baseline:

- Precision and Recall or their combination F1 Score
- DCG and nDCG, which relates to their relevance, based on the inclusion or rank of documents in the results

At this stage, the challenge arises in semantic information retrieval, where retrieval transcends mere keyword matching, encompassing considerations beyond synonyms and token-level enrichment.

You can build a reference evaluation set, known as a [Golden Set](https://www.luigisbox.com/search-glossary/golden-set/). We could also leverage [T5 Model](https://huggingface.co/docs/transformers/model_doc/t5) to generate a starter pack for evaluation.

The golden set is a fundamental component in information retrieval evaluation.
It helps in define characteristics of a reference standard to assess the performance, effectiveness, and relevance of a selected retrieval algorithm.
It provides a common ground for objective comparison and improvement of a retrieval process/algorithm.


### End-to-End Evaluation

An end-to-end evaluation covers the response generation of the question. It leverages the provided context through retrieved documents.

Evaluating the quality of responses produced by large language models can be a challenge due to various factors as described above. 

By virtue of nature and design, the answers generated rely on the diversity of response which makes it impossible to devise a fixed metric or methodology that fits all domains and use-cases.

To address these difficulties, you may use a blend of existing metrics like the following scores: 

- [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu) 
- [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) 

You can then combine these score with LLM-based or human evaluation methods.
For more information, see this paper which provides great ideas to establish [Quality Criteria](https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/c6a53998-09e3-4d17-91fd-c7416d51b250/content) for the same.

To summarize, you want to establish methods to automate evaluating similarity
and content overlap between generated response and reference summaries. You can
also leverage human evaluation to evaluate subjective information, such as context-relevance, novelty, and fluency.

You can also build a classified 'domain - questions' set based on question complexity (easy, medium, hard). It can give you an overall sense of the RAG performance.

Nevertheless, it is a challenge to formulate a comprehensive set of metrics. It
has to appraise the quality of responses from LLMs. It remains an ongoing and intricate issue in the context of natural language processing.

We have laid the foundation, and we know more about the layers of evaluation. 

In the next article, we will describe how you can demystify existing frameworks.

Looking forward to seeing you in the next part!
