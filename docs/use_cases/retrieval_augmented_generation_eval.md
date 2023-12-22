<!-- SEO: Retrieval augmented generation Evaluation - TODO Summary
 -->

# Evaluating Retrieval Augmented Generation


## Understanding Retrieval-Augmented Generation

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_qdrant.jpg alt="Implementation of RAG using Qdrant as a vector database" data-size="100" />

RAG stands for Retrieval Augmented Generation and it probably is the most useful application of large language models lately.
It is the technique that combines the strengths of both retrieval and generation models. The retrieval is usually based on dense vector search, in combination with a text generation model like GPT.
RAG has received significant attention due to its ability to enhance content generation by leveraging existing information effectively. Its capacity to amalgamate specific, relevant details from multiple sources and generate accurate and relevant content has a lot of potential in various domains like content creation, question & answer application, and information synthesis.

Read more: https://hub.superlinked.com/retrieval-augmented-generation

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
RAG viz. Information Retrieval - Augmentation - Response Generation
Here we are talking about an evaluation methodology/framework that is essentially composed of using one/multiple tools, that cover each step to ensure our measurements are granular and thorough.

An Evaluation metric could be one or a combination of many different metrics to assess the effectiveness of retrieval, coherence of generated responses, and relevance to the retrieved information.

<img src=/assets/use_cases/retrieval_augmented_generation_eval/rag_granular.jpg alt="Granular Levels of Evaluation of RAG" data-size="100" />

### Let's dive in !!

## Strategies for Evaluation

Now that we have defined different levels at which we can break down the RAG pipeline, let us zoom into the levels individually, starting from the level 1. 

### Model Evaluation 

The baseline of our model evaluation could be inspired of :
https://github.com/embeddings-benchmark/mteb#leaderboard
The idea is to be able to assess that the data that we are going to encode is comprehensible by the model.
The benchmark above leverages different public/private datasets to evaluate the different capabilities of individual models and report them.
If your business domain is niche there is a good chance that you may have to put together a dataset for this step and it will be worthwhile.
We could either leverage the results here (provided our model belongs to this list) or run relevant ‘tasks’ for our custom model (instructions provided in the link)

TODO: Script examples
.

### Data Ingestion Evaluation

Once the model is validated to understand our domain language we move to the part where we need to ingest the data into our semantic retrieval aka vector store.
Seeing various blogs and write-ups paying attention to the ‘retrieval and generations’ phases, we realized a lot can happen here. We also did not understand that on our first try ;)
With evaluation on this end - we mean to focus on variables that affect the data ingestion, for example :

* chunk size - represents the size of each segment, our data is split into. This will, in principle, depend on the token limit of our embedding model but has a substantial impact on the contextual understanding thereby affecting the precision, recall, and relevancy of our results.

* chunk overlap - represents the presence of overlapping information fragments in each segment. This helps with context retention of the information chunks but at the same time must be used with relevant strategies like deduplication, and content normalization to eradicate adverse effects.

* chunking/text splitting strategy - represents the process of data splitting and further treatment as mentioned above.


### Semantic Retrieval Evaluation

This stage of evaluation goes into iteration with the other 2 previous stages as it puts them to a litmus test.
We may have to go back and forth frequently.


### End-to-End Evaluation

This stage of evaluation covers the evaluation of response generation of the question leveraging the provided context through document retrieved.


Now that we have laid the foundation of what are the important pieces and the layers of evaluation, we can further zoom in on the demo application which we are going to use to demystify the impact of each one of them.
