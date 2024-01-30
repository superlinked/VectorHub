<!-- SEO: Retrieval augmented generation Evaluation - TODO Summary
 -->

# Evaluating Retrieval Augmented Generation


## Understanding Retrieval-Augmented Generation

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_diagram.jpg alt="Typical implementation of RAG using a vector database" data-size="100" />

RAG stands for Retrieval Augmented Generation and it probably is the most useful application of large language models lately.
It is the technique that combines the strengths of both retrieval and generation models. The retrieval is usually based on dense vector search, in combination with a text generation model like GPT.
RAG has received significant attention due to its ability to enhance content generation by leveraging existing information effectively. Its capacity to amalgamate specific, relevant details from multiple sources and generate accurate and relevant content has a lot of potential in various domains like content creation, question & answer application, and information synthesis.

Read more: [here]("https://hub.superlinked.com/retrieval-augmented-generation")

## Why do we need RAG?

Let’s flip the question to ‘When is vector search alone not enough’?
We established above that ‘vector search’ is one of the key pieces in the retrieval stage of RAG.
But why can’t we just do it with vector search alone?
RAG is the solution meant for when your users do not want to interact with documents or search results but generate new content based on the retrieved knowledge.
This capability is valuable for tasks requiring detailed, coherent explanations, summaries, or responses that go beyond what's explicitly stored in the vectors.
Vector search is excellent for finding similar items efficiently, RAG complements it by providing a deeper level of understanding and content synthesis. It's particularly beneficial for tasks demanding contextually rich, informative, and human-like responses or content generation. Integrating both approaches can offer a more comprehensive and versatile solution for various information retrieval and content creation needs.

That said, it is advisable to conduct feasibility and correlate with your business needs and value expectations.

## Evaluating Retrieval-Augmented Generation

At this point, there are a few hundred if not thousands short demos proving the concept of RAG and how seamlessly it may make any business achieve that extra oomph! using AI.
Helping one of our recent customers implement RAG from POC to production made us realize that there is a lot of unknown that goes in the mix to polish the solution.
The most important here is to improve and optimize your solution through evaluation.

### Why do you need to evaluate?

Evaluation is a very crucial step to distill a sense of Trust in your application, which establishes reputation, and boosts team morale and confidence.
Needless to say, it also validates that your applications avoid common pitfalls and it is no different than how we would evaluate and tune our information retrieval systems or machine learning models.
The aspect of relevance assessment through DCG / nDCG or machine learning model through train-validation-test split/overfitting/underfitting analysis addressed similar concerns.
And for some who think LLMs are perfect, they can and do make mistakes too!! 🙂
The key here is that the ‘data’ either source/query is forever changing hence a good evaluation comes in handy to ensure the results are correlated to what users expect.

### If you can't quantify it, you can't improve it!!
For RAG it can be restated as :
### If you can't retrieve it, you can't generate it!!

Assuming you are aware of the different components of
RAG viz. Information Retrieval - Context Augmentation - Response Generation

We categorised the component-wise challenges to understand it better to evaluate each component in isolation. 

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_challenges.png alt="Classification of Challenges of RAG Evaluation" data-size="100" />

We need an evaluation methodology/framework that is essentially composed of using one/multiple tools, that cover each component of RAG to ensure our measurements are granular and thorough.

An Evaluation metric could be one or a combination of many different metrics to assess the effectiveness of retrieval, coherence of generated responses, and relevance to the retrieved information.

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_granular.jpg alt="Granular Levels of Evaluation of RAG" data-size="100" />

### Let's dive in !!

## Strategies for Evaluation

Now that we have defined different challenges and levels at which we can break down the RAG evaluation, let us zoom into the levels individually, starting from the level 1. 

### Model Evaluation 

The baseline of our model evaluation could be inspired of :
https://github.com/embeddings-benchmark/mteb#leaderboard
The idea is to be able to assess that the data that we are going to encode is comprehensible by the model.
The benchmark above leverages different public/private datasets to evaluate the different capabilities of individual models and report them.
If your business domain is niche there is a good chance that you may have to put together a dataset for this step and it will be worthwhile.
We could either leverage the results here (provided our model belongs to this list) or run relevant ‘tasks’ for our custom model (instructions provided in the link)

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

Once the model is validated to understand our domain language we move to the part where we need to ingest the data into our semantic retrieval aka vector store.
Seeing various blogs and write-ups paying attention to the ‘retrieval and generations’ phases, we realized a lot can happen here. We also did not understand that on our first try ;)
With evaluation on this end - we mean to focus on variables that affect the data ingestion, for example :

* chunk size - represents the size of each segment, our data is split into. This will, in principle, depend on the token limit of our embedding model but has a substantial impact on the contextual understanding thereby affecting the precision, recall, and relevancy of our results.

* chunk overlap - represents the presence of overlapping information fragments in each segment. This helps with context retention of the information chunks but at the same time must be used with relevant strategies like deduplication, and content normalization to eradicate adverse effects.

* chunking/text splitting strategy - represents the process of data splitting and further treatment as mentioned above.

A [utility]("https://chunkviz.up.railway.app/") like this seem very useful to visualise your apparent chunks.

### Semantic Retrieval Evaluation

This stage of evaluation goes into iteration with the previous two stages as it puts them to a litmus test.

Retrieval is driving component of the RAG and may need to be addressed as a classic information retrieval evaluation problem.

One of the keys to evaluate information retrieval is to establish the expectations from the returned results , which will help us identifying our reference metrics and important parameters to establish if the documents retrieved at this stage are relevant to the expected information need.

There are existing metrics to guide and define our baseline like Precision and Recall or their combination F1 Score. We have others metrics like DCG and NDCG to take into account the relevance in regards to the inclusion or rank of relevant documents in the results.

Although the nature of semantic information retrieval poses a bit of challenge at this stage, as the documents are retrieved beyond the keywords/synonyms/token enrichment-matching.

The essence of an idea here is to build a reference evaluation set aka [Golden Set]("https://www.luigisbox.com/search-glossary/golden-set/"). We could also leverage [T5 Model]("https://huggingface.co/docs/transformers/model_doc/t5") to generate a starter pack for evaluation.

The golden set is a fundamental component in information retrieval evaluation, that helps in defining characteristics of reference standard for assessing the performance, effectiveness, and relevance of retrieval algorithm. 
It provides a common ground for objective comparison and improvement of retrieval process/algorithm.

### End-to-End Evaluation

This stage of evaluation covers the evaluation of response generation of the question leveraging the provided context through document retrieved.

Evaluating the quality of responses produced by large language models poses a considerable challenge due to various factors as described in the challenges above. 

By virtue of nature and design , the answers generated rely on diversity of response which makes it impossible to device a fixed metric or methodology that fits in all domains and use-cases.

To address these difficulties, it is often suggested to employ not a single but a blend of existing metrics like [BLEU]("https://huggingface.co/spaces/evaluate-metric/bleu") and [ROUGE]("https://huggingface.co/spaces/evaluate-metric/rouge") scores combined with LLM-based or human evaluation methods. [This]("https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/c6a53998-09e3-4d17-91fd-c7416d51b250/content") paper provides some great ideas around the same.  

In completeness, the idea is to establish a sense to automate evaluating similarity and content overlap between generated response and reference summaries , topped by leveraging human evaluation to aid the assessment of subjective aspects such as context-relevance, novelty, and fluency.

Another simple technique to build a classified 'domain - questions' set on the basis of question complexity as easy , medium , hard to get an overall sense of the RAG performance aids targeted improvements.

Nevertheless, formulating a comprehensive set of metrics for appraising the quality of responses from LLMs remains an ongoing and intricate issue within the context of natural language processing.


Now that we have laid the foundation of the building blocks and the layers of evaluation , in the next part of this series we will continue to look into demystifying some of the existing frameworks that helps us to assess the RAG Evaluation.
Looking forward to seeing you in the next part!
