# Your RAG application is a communication system.

Retrieval-Augmented Generation has emerged as a standard approach of implementing LLMs in large organizations. The grounding of LLM generation on existing data ensures an increased robustness to hallucinations, adaptation to in-domain knowledge and updatability. Yet despite proven achievements RAG is still commonly seen as a temporary workaround until the advent of models with an improved context length and more intelligent capabilities.

After designing multiple RAG systems over the past year, I’ve come to believe that RAG signals a more radical transformation: the integration of LLMs in new forms of flexible expert systems. Effective RAG means ingesting even heterogeneous, badly formatted data sources and contextualizing new knowledge on the fly. RAG applications should aspire to become a communication system, taking the best features out of language models (adaptability, universal content translation) but also moderating their shortcomings (alignment mismatch, hallucinations, lack of explainability).

As Yan et al. accurately [called it](https://applied-llms.org/#dont-forget-keyword-search-use-it-as-a-baseline-and-in-hybrid-search): “Models are likely to be the least durable component in the system”. They can be swapped, combined or disabled. What this means for the months and the years to come is that, rather than being replaced by some disruptive LLM technology like long context, RAG will eat LLMs. Embedding search is largely repurposed as an RAG technique. These days, I am mostly fine-tuning with RAG in mind. I recently co-founded PleIAs to get even one step further by pre-training a new generation of language models for RAG end use, and increased support for document processing use cases.

## 1. Language models are few-doc learners.
RAG was not supposed to exist. With pre-training scaling up, LLMs have been largely conceived as “interpolated datasets” holding in memory the collective knowledge of the Internet. The inaugural RAG system was still a form of interpolated dataset: a clever model design associating a seq2seq model with a dense layer trained on Wikipedia ((Lewis et al., 2020)[https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf]).

Interpolated datasets ran into a range of structural issues that proceed from the core nature of LLMs (a model trained at one time on a finite set of data):
* **Specialization**: many professional resources are unlikely to be featured in pre-training data or with a very light exposure that result in high hallucination rates.
* **Updatability**: a database can be continuously maintained in real time while LLMs are limited by their cutoff date.
* **Explainability**: separating the search/memory part does circumvent to some extent the “black box” nature of LLMs.

Instead, reformulation of few-document turned out to be one the most practical use cases for LLMs. The key properties were already built by design in word embeddings nearly a decade ago and have developed ever since. Cross-lingual transfer is surprisingly efficient: many models primarily trained in English can be successfully trained on downstream tasks in other languages. This extends to other forms of language transfer: from formal to informal, from standards to dialects, from generalist to specialized terminologies.

The default paradigm for RAG emerged almost immediately after the release of chatGPT. One of the first systematic descriptions I’ve found is Simon Willison’s “(semantic search answers pattern)[https://simonwillison.net/2023/Jan/13/semantic-search-answers/]” from early January 2023 – search for texts in a semantic database, get the most relevant results, send this input to GPT-3 as an inspiration.

Today, production RAG is increasingly a lot of flows and arrows and practitioners are confronted with the hard reality: there is no generic approach to RAG. Good applications need to anticipate a wide array of edge cases. There are simply just too many breaking points such as bad OCR, poor indexation, suboptimal tokenization, inconsistent document segment, unpredictable queries…

The last mile problem is actually a great opportunity for RAG apps. Despite the eagerness of major AI actors to become encompassing platforms, the last mile won’t be easily solved. The right fit for a use case, a corpus, a specific user target is the hardest part of a RAG system, probably the only actual moat there is in applied generative AI.

In turn, this means that the biggest issue for RAG is evaluation. Preliminary evaluations and benchmarks to select and swap models is not the hardest part — you could even do with synthetic data. The real challenge is evaluation in production of the different components of your system, taking into account their level of specialization. While we strive right now to automate everything, evaluation is a final residual of qualitative human assessment.

## 2. Retrieval: back to future.
As RAG was primarily formulated in an LLM/embedding context, it obfuscated the actual roots of the communication system it came to develop: information retrieval.

### 2.1 The actual roots of RAG: decades of IR and data frictions.
Expert systems and knowledge infrastructures have been existing for decades and dealt with the same data frictions that characterize RAG applications today. Frictions at the entry level, with much additional work needed to fit new content to imperative formats and expectations. Frictions on the receiving ends, as users are not behaving according to the expectations of the search infrastructure.

Back in the 1960s, the first expert systems widely underestimated the massive intermediation work needed to get these systems working. It was hoped that they would soon be seamlessly accessed by professionals: “MEDLINE was intended to be used by medical researchers and clinicians, NASA/RECON was designed for aerospace engineers and scientists.” And yet, “For many reasons, however, most users through the seventies were librarians and trained intermediaries working on behalf of end users.”

It sounds like a familiar pattern. RAG emerged with a similar promise of simplification shock: taking whatever content there is, outputting whatever content the user wants. This is not only the obvious objective of LLM on the output side but also embedding models on the input side. Embeddings are natively multilingual, theoretically resilient to synonyms, variations of language standards and document artifacts. 

The radical promise frequently failed on two fronts: understanding users, understanding data.

### 2.2 Know your users
This is the obvious start of most Information retrieval projects but, until now, I haven’t seen it frequently mentioned in a RAG context: know what your users are doing with your systems.

Overall, RAG applications underestimate the continuity with the previous expert system and the inertia of user practices. I recently analyzed a set of internal user queries with a chatbot in production. One of the most surprising results is that many users were not submitting questions and conversations but unstructured keywords. This is perfectly fitting in the standard information retrieval system, but instruction-based LLM and embedding models are by design trained on question/answer pairs.

Dissociation between use and design should be a primary concern. Without proper care, LLMs and embedding models are not only unable to solve existing data frictions but will introduce new frictions : alignment mismatch with user expectation, poor breakdown of data structure due to tokenization, incongruous length, total lack of control over the system purpose (the infamous “Chevrolet” effect).

### 2.3 Know your data
Like your users, the data will not necessarily agree with the constraints of language models and embedding models.

Context window is an obvious restriction. Despite its gradual lengthening, the indexation of long texts always underperforms. This is one of the biggest mismatches between pre-training data and data in production: the vast majority of Common Crawl texts are less than 512 tokens. Organizations routinely deal with multi-pages PDF.

So you need to split or chunk your document and… this is not trivial. Documents parts are not meant to be standalone units of content and disambiguation requires some level of cross-referencing. The biggest performance gain on my retrieval evaluation was the… systematic inclusion of title hierarchy (document > chapter > section etc.). Simply because in long documents, general information on the topic is given at this scale, but not recalled in specific paragraphs.

Proper contextual support for original data is not only a concern for retrieval but also for generation. Even powerful LLMs are heavily sensitive to document structure. Lack of proper segmentation or unusual structure will result in poor tokenization (typically merging a document separator with a token) and a higher rate of omissions and hallucinations. With tables, a hacky (but verbose) fix I’ve found is to recall the row name/column name couple before each entry.

Finally, retrieval metrics are not necessarily fitting the data. Embedding models rely on a fuzzy concept of “similarity” that is likely to fail the last mile. They are trained on specific sets of documents and questions (typically bases like MS-Marco). This will not necessarily extend well to specialized resources, and, more fundamentally, similarity may not align with user expectations.

### 2.4 Settling for hybrid
As RAG systems naturally evolve with longer documents, enriched with additional context and more and more uninitiated users, simply transposing their usual search habits, the reality sets in: embedding models are barely competitive to classic information retrieval approaches like BM25 and have to be supplemented by it.

What this means in practice: 
* Hybrid indexation is emerging as the new standard: on my latest internal benchmarks, combining a keyword-based search with embedding similarity hit the highest retrieval rate (nearly 90% of resources found among the top ten results). The easiest implementation I’ve found so far is lancedb’s but there is right now a lot of activity in this space.
* Keyword-based search is still a strong alternative. The infrastructure is already there and at a very large scale, embeddification can quickly become prohibitive.
* Evaluation has to be built from the ground up, based on the constraints of your existing infrastructure (especially in regards to data quality/indexation) and the ways users interact with it.

Overall, there is a lot of alpha left in recovering information retrieval methods and indicators in an LLM context, especially as many key features can be now accurately generated/extracted at scale. Jo Kristian Bergum from Vespa has just introduced a very convincing method of data expansion, by grounding an LLM-as-a-Judge on a small relevant dataset, that can be transformed into a large one. Basically, generative AI does not necessarily replace the classic approaches of retrieval evaluation, but completely reshape their logistics. Intensive data work that would have been only available to large scale organizations is suddenly scalable with much less resources.

## 3. Keep the data alive
The evolution of RAG toward a communication system means that the data is no longer just a passive reference stock. It has to be kept alive in two ways: continuously transformed and reshaped to better fit the retrieval objective, constantly circulating across different flows.

### 3.1 You need bad data
The reality of production RAG is that once we get slightly beyond the proof-of-concept phase, the system has to work with whatever data is available. To get to the point, you have to become an expert on bad data.

Scaling is not only a bitter lesson for LLMs: I worked for years in cultural heritage and a library with millions of bad OCR documents is always considerably more useful than a well curated and edited selection of a thousand canon texts. Large organizations are no different. RAG has to work with legacy texts, which means PDFs. And, to stress it again, LLMs are hardly trained on PDFs: common crawl is web/html texts, without artifact and advanced document structures. Bad data is also to be expected on the input side. Users want the best answers with the minimum efforts, and the customer is always right. 

In both cases, LLM trainers do not necessarily have access to the actual bad data processed in production. It can be restricted due to data protection regulations or other organizational concerns, or simply hard to access. The main alternative is looking for placeholders or… generating it. As it turns out Google and probably other major AI providers have been seriously investing in data degradation techniques. Simple data processing heuristics can be already effective (like dropping cases, introducing noise). The most powerful technique I’ve yet found is to fine-tune on existing bad data.

### 3.2 Managing data flows with classifiers
Fortunately, we have the best tools to date for bad (and good) data. Despite their limitations for search, language models are amazing at data processing. I mean here language models in general, not just the popular decoders: the previous generation of encoders and encoder-decoders currently enjoy a much needed revival; they provide a much better parameters / performance / robustness ratio than GPT-4, Llama or Mistral for many data tasks.

Classifiers and reformulators (for want of a better name) are increasingly critical components of my RAG systems. This include:
* Automated re-routing of user queries. A major issue met in more generalist chat system is to decide 1) whether or not the retrieval of external resources should be activated 2) which collection of resources should be prioritized (maybe not all are needed, and this can relieve stress to the system) 3) when does a new round of retrieval become necessary (for instance during a conversation if the topic has noticeably switched). 
* Query expansion. LLM and embedding models struggle with unstructured queries, as they are primarily trained on complete sentences. Query expansion can also rely on additional elements (like past queries) for further contextualization.
* Text segmentation (rather than just “chunking”). Proper segmentation has been proven to be one of the most powerful performance boosts. Yet, as the RAG system expands both in terms of document coverage and user use, it can become prohibitively costly to do so iteratively (typically by parsing different sets of resources, with their own ). Lately, I’ve been experimenting a lot with agnostic encoder models for token-classification tasks, able to reconstruct text parts (including titles, bibliography) from raw text comprehension, regardless of the original format.

For all these things, using an LLM is an overkill that results in performance lag and degradation. Once properly re-trained, encoder models yield comparable results, are very fast and not costly to host and better fitted to the overall purpose of classification. There’s no need to apply structured generation, regex or data constraints as the data design is built-in.

### 3.3 An alternative to generation: synthetic data curation.
I’m expecting that a major direction of research in the months to come will be pre-generation of system output and input. Basically swapping models in production with optimized retrieval of synthetic data. This was the approach retained by Klarna for their integration of ChatGPT. It yields obvious advantages in terms of robustness and security (prompt injection is simply impossible). 

As RAG systems scale, pre-generation provides significant economies of scale: most queries follow a 80-20 pattern and once a question has been solved, there’s little point to regenerate it. Obviously, this approach works best once you have a general grasp of what users are looking for.

A further twist on pre-generation is to expand the original data to also include queries, preferably the type of queries your users are likely to write. And you are accidentally recreating with… an instruction dataset, with a couple of queries and answers. It may be high time to consider fine-tuning 

## 4. Fine-tuning as a communication protocol
In a RAG context, fine-tuning has been frequently misunderstood. It’s neither an alternative nor a good supplement to content retrieval: LLM memory is inconsistent, even more so when it relies on LORA rather than pretraining. 

Fine-tuning is better conceived as a method for enhancing communication, by tweaking the word probabilities of the base model in a specific direction: communication between the model and the user, communication between the resources and the output and, as agent-based approaches are emerging, between different models.

### 4.1 Fine-tuning for/through RAG
Fine-tuning has to come later in the cycle of development. In my experience, good instruction datasets need to be designed not only with RAG in mind but through RAG. My standard protocol of fine-tuning for RAG goes this way:
* Generate queries based on indexed texts. This is a really standard and basic approach of synthetic data universally used for fine-tuning: given a text, invent a question it would answer. In this specific context, it is recommended to prompt the model to create “vague” enough questions that could yield more valid answers than the seeding text. Also to go one step further, you’d want the queries to match the ones sent by your user. I recommend including a random selection 
* Retrieve a selection of documents. Here it may be actually good to sometimes voluntarily degrade the performance of retrieval (for instance using only BM25), as you want the LLM to be more resilient to hard negatives.
* Generate the actual synthesis based on the question and the retrieved documents. This is the most creative part by far, the one that actually controls the final communication output, the end behavior of the model. You don’t have to get it right in one pass. Since LORA fine-tuning is cheap anyway, I start with a simple RAG prompt on an existing model and then continuously reshape and rework this output.

Fine-tuning does not require a lot of examples. Most of my projects are in the 1,000-4,000 instructions range, sufficiently large to warrant text generation, sufficiently small to still be largely an opinionated product. There is no way around it: while LLMs are all about text automation, a good fine-tuning dataset will require a significant amount of careful manual tweaking.

The instruction dataset is in many ways a miniature version of the RAG communication system. You need to approximate as much as possible the future queries your users will send (so preferably based on the past ones), the existing and future documents you will use for grounding and the expected output. In short, you are designing a translation process and, through the instructions, providing as many use cases as possible where this translation went right.

Otherwise, I would not spend so much time optimizing the training design beyond a few hyperparameters (learning rate, batch size, etc.). Data work and base model improvement had always the most impact. Typically, I stopped looking into preference fine-tuning (like DPO) as the time spent in additional data preparation was not worth the very few improvement points.

While this is much less used, the exact same approach (can be used for embedding models)[https://huggingface.co/blog/davanstrien/synthetic-similarity-datasets]. Synthetic data makes it considerably easier to create an instruction dataset mapping the expected format of similarity dataset (including queries and “hard negatives”). You’ll get here the same benefit as LLM fine-tuning: cost-saving (with a much smaller model showing the same level of performance) and appropriateness by bringing the “similarity” score closer to the expectations of your retrieval system.

### 4.2 Purpose of fine-tuning: robustness
This is the primary objective of fine-tuning in a RAG context: reinforcing the overall robustness of document synthesis. Prompt engineering has built-in limitations. Even powerful models have conflicting incentives as they are built for generalists use, obviously beyond your specific RAG system but even beyond RAG in general. LLMs are likely to hallucinate more than you’d want, simply because model providers can only restrict their creativity to a point.

Robustness has several underlying implications in a RAG context.
* **Content fidelity**: the model should generally not add more than what is presented in the retrieved document, at least beyond some general knowledge.
* **Dealing with uncertainty**: queries can fail for a variety of reasons, either because there are no proper answers in the RAG dataset or because retrieval fails. Models should not only anticipate issues but act as a fallback when it happens, typically by stating a negative answer.
* **Tracking the information**: RAG-dedicated models have increasing support for citation and references of each individual statement from the retrieved sources. I think the approach was originally introduced by Perplexity.
* **Setting the stage for verifiability**: as one step further, references can be associated with quote excerpts from the retrieved sources. Beyond the potential gain of accuracy, this process makes it easier for lay users to navigate to the original sources. Just making available the complete original document turns out to be very cumbersome for fact-checking.

Our RAG projects at PleIAs were conceived in the context of high constraints for verifiability (public sector, large legacy privacy companies). We took inspiration from Wikipedia. The collaborative encyclopedia had to deal with similar constraints as an expert RAG system: texts cannot be attributed to individual authors that take responsibility for publication and trustworthiness. Accuracy stems mostly from verifiability, by associating individual statements to secondary sources and easing verification.

While it is definitely possible to get to the same output using a powerful model (and this was actually the source for the instruction dataset), this has some limitations:
* **Prompts will not systematically work**. While generating the instruction dataset, I’ve had to discard 5% of the generated output. This is not much, but you have to combine this uncertainty with all the others (retrieval failure, imprecise query) and this adds up to the stress of your RAG system.
* **Prompts can saturate**, especially if on top of asking for an advanced verifiability check, you are also making other recommendations for style (see next section).
* **Prompts are open to injection and security issues**. This is the infamous “Chevrolet” effect where a help support system turns out to be a thin wrapper of ChatGPT.

Assessing model robustness is the tricky part. Standard benchmarks do not apply: what you are going to measure is actually a loss of performance as the specialization through retraining only results in a loss of general performance.

As far as accuracy goes, the most straightforward way is to compare the model output and the original documents. It works obviously best if you are already clarifying the attribution of individual statements to specific documents through generated references. For this process you can use an LLM-as-a-judge or, if you provide actual quotes, a straight comparison with the original text (as an additional confusing factor, this is called a “text alignment” algorithm, but nothing to do with fine-tuning or RLHF).

### 4.3 Purpose of fine-tuning: relieving data frictions
A RAG system is made primarily for its users. Hence one of the biggest issues of using a generalist model like GPT-3.5/4 on a long term basis : you have an audience alignment gap. This is even more striking if you’re working in another language than English (like myself): ChatGPT feels immediately “alien” with weird twists of languages that are not to be found in standard French.

Reshaping style is the most shocking impact of fine-tuning. A few months ago I released an extreme version of user-LLM communication: MonadGPT, an instruction model trained on 17th century English. What took me the longest was to come up with the most fitting communication scheme: train the models on a couple of answers from historical texts and generated questions in contemporary English, simply because I should absolutely not expect the users to mimic 17th century language. Users should not have to mimic the style of any document resources either. 

Similarly to MonadGPT you should look for the following alignment design:
* Generated queries should be as close as possible as the real queries sent by users, which means likely purposefully generating “bad” data. In my current instruction dataset, most queries have no punctuation, some are in all caps or missing words, with typos.
* Generated answers have to convey the general “brand” style of your system. Do not hesitate to re-generate the answer, reformulate the existing one, or even rewrite them manually.

Relatedly, you have to anticipate the “persona” of your model. Whether you like it or not, users of interactive systems will ask meta-referential questions: who are? What can you do? Many advanced LLM systems overlooked this aspect to their detriment (with new models trained from scratch erroneously stating they are ChatGPT, since this is the default AI persona in much of the training data these days). I recommend here to combine both prepared answers in the instruction dataset with a series of “biographical” documents about the model in the RAG dataset.

To assess style and frictions, the best way may ultimately be to collect feedback from your users. You don’t need anything fancy: I would just recommend an open-ended comment section somewhere (and green/red buttons).

# 5. Will RAG redefine LLMs?
By mid-2024, we are probably at a crossroad. There's always the possibility of a major breakthrough in LLM research (Q*, Monte Carlo Tree Search, diffusion models, or state space models…) that could shift the focus back to model architecture. If not LLMs will be increasingly pushed to the backstage of larger, more complex systems.

RAG has emerged as the main production use case for LLMs and will be at the forefront of model orchestration. Orchestration already happened in many ways. Embedding models are language models, and so are classifiers. Integrating multiple decoder models in production will just be a scaling challenge: the pattern is already there.

We are now at the early stages of a new generation of LLMs and SLMs made for RAG and RAG-associated tasks. The first models are already there. CommandR from Cohere, has been designed from the ground up with better support for source quotation and document interaction. The smallest variant of Phi-3 from Microsoft aims to operate best in contexts where knowledge and memory are separated from reasoning capacity.

In the current state of technology adoption, specialization is solving multiple complex equations. It reduces the overall memory footprint, as small models well integrated in the pipeline can suddenly become competitive with gigantic frontier models. It ensures the model can run locally if needs be and process private organizational data without any liability. It limits the amount of work needed to deploy the model in production as many requirements of a good RAG application (source verifiability, better grounding, better understanding of original sources) can be addressed at the pretraining level.

At PleIAs we intend to move even further in this direction. While several intermediaries focus on fitting internal data better for LLM processing, we intend to bring models closer to the type of content actually used in production. Our current generation of models is trained on a large corpus of non-HTML documents (especially PDF) in a wide variety of formats and intentionally featuring many examples of digitization artifacts (OCR noise, bad segmentation) Our starting assumption is that a model can only get good at correcting things it has seen. 

Like with any integrated search and document processing system, a critical aspect of a RAG application is redundancy and fallbacks. Everything will fail at some point: the unexpected query, the wrong document, the bad token. While pre-processing of RAG dataset does a lot, we believe that some of this burden should fall on the model themselves, ensuring they are sufficiently resilient to the content people actually use.

I believe that another overlooked factor that will propel a new wave of specialized or retrained models: RAG datasets are growing. Even for applications with a well-targeted audience, there are strong incentives to expand the range of indexed context, cover more use cases. The improved quality and efficiency of search retrieval reverses means that less related content (for instance generalist newspaper articles or even some parts of common crawl) are increasingly usable.

A very large RAG dataset, properly documented and augmented with synthetic data looks a lot like… pretraining data. Both large scale RAG and the recent focus of pre-training on “quality data” invites us to think about “data orchestration”. Synthetic data is a hot topic in pre-training right now, not just as a source of new data, but as a powerful tool to process, clean and expand on existing resources. I’m expecting the same process to reverberate on the production side. Especially as data transactions hold potentially the promise to cut costs on model transactions: it’s even likely that, in a near future, most RAG calls will return pre-generated texts.

Despite the current fascination for gigantic models, LM development may rely more on mid-size organizations with advanced data expertise. Diminishing returns for scale, increasing returns for customization convergently bring a future where RAG has absorbed LLMs.

