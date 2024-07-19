# Your RAG application is a communication system

It's become routine for large organizations to employ Retrieval-Augmented Generation when using LLMs on their data. By grounding LLM generation in existing data, RAG ensures increasingly robust resistance to hallucinations, adaptation to in-domain knowledge, and updatability. Still, RAG is typically understood as a temporary workaround, to be made redundant after the eventual advent of LLMs with improved context length and better intelligence.

After designing multiple RAG systems over the past year, I’ve come to believe that RAG systems represent a more radical transformation that will be here for years to come, more enduring than any LLM model they happen to integrate. RAG systems have the capacity to address the fundamental challenges of information retrieval, meeting the user and the data where they're at. Effective RAG can take the best features of language models (adapatability, universal content translation) but also moderate their shortcomings (alignment mismatch, hallucinations, and lack of explainability), ingesting even heterogeneous, badly formatted data sources, and contextualizing new knowledge on the fly.

Recently, I find myself fine-tuning LLMs with RAG in mind, and recently cofounded PleIAs to pre-train a new generation of LLMs for RAG end use and document-processing use cases. LLMs, as Yan et al. [predicted](https://applied-llms.org/#dont-forget-keyword-search-use-it-as-a-baseline-and-in-hybrid-search), are "likely to be the least durable component in the system" - swapped, combined, disabled. RAG applications, rather than being replaced by some disruptive revolution in LLMs, like long context, can be nothing less than flexible, expert communication systems, and should serve as the focal end point of fine-tuning whatever LLM your RAG uses.
Embedding search is largely repurposed as a RAG technique. 

outline/intro... 



## 1. Language models are few-doc learners
RAG was not supposed to exist. With pre-training scaling up, LLMs have been largely conceived as “interpolated datasets” holding in memory the collective knowledge of the Internet. The inaugural RAG system was still a form of interpolated dataset: a clever model design associating a seq2seq model with a dense layer trained on Wikipedia ([Lewis et al., 2020](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)).

Interpolated datasets ran into a range of structural issues that result from the fact that LLMs are models trained at a point in time on a finite set of data:
* **Specialization**: professional resources are often not or only lightly included in pre-training data, resulting in high hallucination rates
* **Updatability**: a database can be continuously maintained in real time while LLMs are limited by their cutoff date
* **Explainability**: separating the search/memory part does circumvent to some extent the “black box” nature of LLMs

Instead, reformulation of few-document turned out to be one the most practical use cases for LLMs. The key properties were already built by design in word embeddings nearly a decade ago and have developed ever since. Cross-lingual transfer is surprisingly efficient: many models primarily trained in English can be successfully trained on downstream tasks in other languages. This extends to other forms of language transfer: from formal to informal, from standards to dialects, from generalist to specialized terminologies.

The default paradigm for RAG emerged almost immediately after the release of chatGPT. One of the first systematic descriptions I’ve found is Simon Willison’s “[semantic search answers pattern](https://simonwillison.net/2023/Jan/13/semantic-search-answers/)” from early January 2023 – search for texts in a semantic database, get the most relevant results, send this input to GPT-3 as an inspiration.

Today, production RAG is increasingly a lot of flows and arrows and practitioners are confronted with the hard reality: there is no generic approach to RAG. Good applications need to anticipate a wide array of edge cases. There are simply just too many breaking points, e.g., bad OCR, poor indexation, suboptimal tokenization, inconsistent document segment, unpredictable queries, etc.

The last mile problem is actually a great opportunity for RAG apps. Despite the eagerness of major AI actors to become encompassing platforms, the last mile won’t be easily solved. The right fit for a use case, a corpus, a specific user target is the hardest part of a RAG system, probably the only actual moat there is in applied generative AI.

In turn, this means that the biggest issue for RAG is evaluation. Preliminary evaluations and benchmarks to select and swap models is not the hardest part — you could even do with synthetic data. The real challenge is evaluation in production of the different components of your system, taking into account their level of specialization. While we strive right now to automate everything, evaluation is a final residual of qualitative human assessment.

## 2. Retrieval: back to future
Because RAG was primarily formulated in an LLM/embedding context, it obfuscated the actual roots of the communication system it came to develop: information retrieval.

### 2.1 The actual roots of RAG: decades of IR and data frictions
Expert systems and knowledge infrastructures have existed for decades and dealt with the same data frictions that characterize RAG applications today. At the entry level, much additional work is needed to fit new content to imperative formats and expectations. On the receiving end, users don't behave according to the expectations of the search infrastructure.

Back in the 1960s, the first expert systems widely underestimated the massive intermediation work needed to get them working. It was hoped that they would soon be seamlessly accessed by professionals: “MEDLINE was intended to be used by medical researchers and clinicians, NASA/RECON was designed for aerospace engineers and scientists.” And yet, “For many reasons, however, most users through the seventies were librarians and trained intermediaries working on behalf of end users.”

(LLM and) RAG emerged with a similar promise of simplification shock: taking whatever content there is, outputting whatever content the user wants. This is not only the obvious objective of LLM on the output side but also embedding models on the input side. Embeddings are natively multilingual, theoretically resilient to synonyms, variations of language standards and document artifacts.

The radical promise frequently failed on two fronts: understanding users, understanding data.

### 2.2 Know your users
This is the obvious start of most information retrieval projects but, until now, I haven’t seen it frequently mentioned in a RAG context: know what your users are doing with your systems.

Overall, RAG applications underestimate the continuity with the previous expert system and the inertia of user practices. I recently analyzed a set of internal user queries with a chatbot in production. One of the most surprising results is that many users were not submitting questions and conversations but unstructured keywords. This practice works perfectly in a standard information retrieval system, but instruction-based LLM and embedding models are by design trained on question/answer pairs.

Dissociation between use and design should be a primary concern. Without proper care, LLMs and embedding models are not only unable to solve existing data frictions but will introduce new frictions: alignment mismatch with user expectation, poor breakdown of data structure due to tokenization, incongruous length, total lack of control over the system purpose (the infamous “Chevrolet” effect).

### 2.3 Know your data
Like your users, the data will not necessarily agree with the constraints of language models and embedding models.

Context window is an obvious restriction. Despite its gradual lengthening, the indexation of long texts always underperforms. This is one of the biggest mismatches between pre-training data and data in production: the vast majority of Common Crawl texts are less than 512 tokens. Organizations routinely deal with multi-page PDFs.

So you need to split or chunk your document, which is not trivial. Documents parts are not meant to be standalone units of content, and disambiguation requires some level of cross-referencing. The biggest performance gain on my retrieval evaluation was the systematic inclusion of title hierarchy (document > chapter > section, etc.) - in long documents general information on the topic is given at this scale, but not recalled in specific paragraphs.

Proper contextual support for original data is not only a concern for retrieval but also for generation. Even powerful LLMs are heavily sensitive to document structure. Lack of proper segmentation or unusual structure will result in poor tokenization (typically, merging a document separator with a token) and a higher rate of omissions and hallucinations. With tables, a hacky (but verbose) fix I’ve found is to recall the row name/column name couple before each entry.

Finally, retrieval metrics don't necessarily fit the data. Embedding models rely on a fuzzy concept of “similarity” that is likely to fail the last mile. They are trained on specific sets of documents and questions (typically bases like MS-Marco). This will not necessarily extend well to specialized resources, and, more fundamentally, similarity may not align with user expectations.

### 2.4 Settling for hybrid
As RAG systems naturally evolve with longer documents, enriched with additional context and more and more uninitiated users, simply transposing their usual search habits, the reality sets in: embedding models can barely compete with classic information retrieval approaches like BM25 and have to be supplemented by it.

What this means in practice:
* Hybrid indexation is emerging as the new standard: on my latest internal benchmarks, combining a keyword-based search with embedding similarity hit the highest retrieval rate (nearly 90% of resources found among the top ten results). The easiest implementation I’ve found so far is lancedb’s, but the jury's still out - there's a lot of activity in this space right now.
* Keyword-based search is still a strong alternative. The infrastructure is already there and at a very large scale, embeddification can quickly become prohibitive.
* Evaluation has to be built from the ground up, based on the constraints of your existing infrastructure (especially with regard to data quality/indexation) and the ways users interact with it.

Overall, there is a lot of alpha left in recovering information retrieval methods and indicators in an LLM context, especially as many key features can now be accurately generated/extracted at scale. Jo Kristian Bergum from Vespa has just introduced a [very convincing method of data expansion](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/), by grounding an LLM-as-a-Judge on a small relevant dataset, that can be transformed into a large one. Basically, generative AI doesn't replace the classic approaches of retrieval evaluation; it can completely reshape their logistics. Intensive data work that would have been available only to large scale organizations is suddenly scalable with far fewer resources.

## 3. Keep the data alive
The evolution of RAG toward a communication system means that the data is no longer just a passive reference stock. It has to be kept alive in two ways: continuously transformed and reshaped to better fit the retrieval objective, constantly circulating across different flows.

### 3.1 You need bad data
The reality of production RAG is that once we get slightly beyond the proof-of-concept phase, the system has to work with whatever data is available. To productionize, you have to become an expert on bad data.

Scaling is not only a bitter lesson for LLMs: I worked for years in cultural heritage, and a library with millions of bad OCR documents is far more useful than a well curated and edited selection of a thousand canon texts. Large organizations are no different. RAG has to work with legacy texts, which means PDFs. And, to stress it again, LLMs are hardly trained on PDFs: common crawl is web/html texts, without artifact and advanced document structures. We should also expect bad data on the input side. Users want the best answers with minimum effort, and the customer is always right.

In both cases, LLM trainers generally don't have access to the bad data actually processed in production - either because it's restricted by data protection regulations or other organizational barriers, or because it's simply hard to access. The usual workaround is looking for placeholders, or generating it. As it turns out, Google (and probably other major AI providers) have been seriously investing in data degradation techniques. Simple data processing heuristics can be already effective (e.g., dropping cases, introducing noise). The most powerful technique I’ve found to date is fine-tuning on existing bad data.

### 3.2 Managing data flows with classifiers
Fortunately, we have the best tools to date for processing bad (and good) data: language models, despite their limitations for search, are amazing at data processing. I mean here language models in general, not just the popular decoders: the previous generation of encoders and encoder-decoders currently enjoy a much needed revival; they provide a much better parameters / performance / robustness ratio than GPT-4, Llama or Mistral for many data tasks.

Classifiers and reformulators (for want of a better name) are increasingly critical components of my RAG systems. This includes:
- * Automated re-routing of user queries. A major issue of more generalist chat systems is deciding 1) whether or not to activate the retrieval of external resources 2) which collection of resources to prioritize (maybe not all are needed; reducing volume can relieve stress to the system) 3) when a new round of retrieval becomes necessary (e.g., when the topic of a conversation has changed).
- * Query expansion. LLM and embedding models, because they're trained primarily on complete sentences, struggle with unstructured queries. You may require further contextualization by adding elements (e.g., past queries).
- * Text segmentation (rather than just “chunking”). Proper segmentation has proven to be one of the most powerful performance boosts. But as user traffic increases, and more document coverage is needed, iteratively expanding your RAG system (typically by parsing different sets of resources with existing ones) can become prohibitively costly. Lately, I’ve been experimenting a lot with agnostic encoder models for token-classification tasks, able to reconstruct text parts (including titles, bibliography) from raw text comprehension, regardless of the original format.

For automated query rerouting, query expansion, and text segmentation, using an LLM is overkill and results in performance lag and degradation. Once properly re-trained, encoder models yield comparable results, are very fast, not expensive to host, and better fitted to the overall purpose of classification. In addition, encoders don't require structured generation, regex, or data constraints - the data design is already built-in.

### 3.3 An alternative to generation: synthetic data curation
I’m expecting that a major direction of research in the months to come will be pre-generation of system output and input - i.e., basically, swapping models in production with optimized retrieval of synthetic data. This was the approach retained by Klarna for their integration of ChatGPT. It yields obvious advantages in terms of robustness and security (prompt injection is simply impossible).

As RAG systems scale, pre-generation provides significant economies of scale: most queries follow a 80-20 pattern and once a question has been solved, there’s little point in regenerating it. Obviously, this approach works best once you have a general grasp of what users are looking for.

A further twist on pre-generation is to expand the original data to also include queries, preferably the type of queries your users are likely to write. And you are accidentally recreating with… an instruction dataset, with a couple of queries and answers. It may be high time to consider fine-tuning.

## 4. Fine-tuning as a communication protocol
In a RAG context, fine-tuning has been frequently misunderstood. It’s neither an alternative nor a good supplement to content retrieval: LLM memory is inconsistent, even moreso when it relies on LORA rather than pretraining. 

Fine-tuning is better conceived as a method for enhancing communication, by tweaking the word probabilities of the base model in a specific direction: communication between the model and the user, communication between the resources and the output and, as agent-based approaches are emerging, communication between different models.

### 4.1 Fine-tuning for/through RAG
Fine-tuning has to come later in the cycle of development. In my experience, good instruction datasets need to be designed not only with RAG in mind but through RAG. My standard protocol of fine-tuning for RAG goes this way:
- * Generate queries based on indexed texts. This is a basic and universal approach when using synthetic data for fine-tuning: for a given text, invent a question it would answer. In this specific context, I recommend to prompting the model to create questions “vague” enough to yield other valid answers than just the seeding text. Also, to go one step further, you want the queries to match the ones sent by your user. I recommend including a random selection.
- * Retrieve a broad selection of documents. Here it may be good sometimes to voluntarily degrade retrieval performance (e.g., using only BM25) - you want the LLM to be more resilient to hard negatives.
- * Generate the synthesis based on the question and the retrieved documents. This is the most creative part by far, one that controls the final communication output - the end behavior of the model. You don’t have to get it right in one pass; LORA fine-tuning is cheap. I start with a simple RAG prompt on an existing model, and then continuously reshape and rework this output.

Fine-tuning doesn't require a lot of examples. Most of my projects are in the 1,000-4,000 instructions range, sufficiently large to warrant text generation, sufficiently small to still be mostly an opinionated product. There is no way around it: while LLMs are all about text automation, a good fine-tuning dataset will require a significant amount of careful manual tweaking.

The instruction dataset is in many ways a miniature version of the RAG communication system. You need to approximate as much as possible the future queries your users will send (so preferably based on the past ones), the existing and future documents you will use for grounding and the expected output. In short, you are designing a translation process and, through the instructions, providing as many use cases as possible where this translation went right.

Otherwise, I would not spend much time optimizing the training design beyond a few hyperparameters (learning rate, batch size, etc.). Data work and base model improvement always have the most impact. I've generally stopped looking into preference fine-tuning (like DPO); the time spent in additional data preparation was not worth the very few improvement points.

While it's far less common, the exact same approach [can be used for embedding models](https://huggingface.co/blog/davanstrien/synthetic-similarity-datasets). Synthetic data makes it considerably easier to create an instruction dataset mapping the expected format of the similarity dataset (including queries and “hard negatives”). You’ll get here the same benefit as LLM fine-tuning: cost-saving (with a much smaller model showing the same level of performance) and appropriateness by bringing the “similarity” score closer to the expectations of your retrieval system.

### 4.2 Purpose of fine-tuning: a) robustness
Reinforcing the overall robustness of document synthesis is the primary objective of fine-tuning in a RAG context. Prompt engineering has built-in limitations. Even powerful models have conflicting incentives as they are built for general use, as opposed to being specifically designed for your specific RAG system, or even for RAG objectives generally. LLMs are likely to hallucinate more than you’d want, simply because model providers can only restrict their creativity to a point.

Robustness has several underlying implications in a RAG context.
* **Content fidelity**: the model should generally not add more than what is presented in the retrieved document, at least beyond some general knowledge.
* **Dealing with uncertainty**: queries can fail for a variety of reasons, either because there are no proper answers in the RAG dataset or because retrieval fails. Models should not only anticipate issues but act as a fallback when it happens, typically by stating a negative answer.
* **Tracking the information**: RAG-dedicated models have increasing support for citation and references of each individual statement from the retrieved sources. I think this approach was originally introduced by Perplexity.
* **Setting the stage for verifiability**: as one step further, references can be associated with quote excerpts from the retrieved sources. Beyond the potential gain of accuracy, this process makes it easier for lay users to navigate to the original sources. Just making available the complete original document turns out to be very cumbersome for fact-checking.

Our RAG projects at PleIAs were conceived in the context of high constraints for verifiability (public sector, large legacy privacy companies). We took inspiration from the collaborative encyclopedia, Wikipedia, who faced constraints similar to those of an expert RAG system: texts cannot be attributed to individual authors who take responsibility for their publication and trustworthiness. Accuracy stems mostly from verifiability, by associating individual statements with secondary sources and easing verification.

While it's possible to get to the same output using a powerful model (and this was actually the source for the instruction dataset), this has some limitations:
* **Prompts won't work systematically**. While generating the instruction dataset, I’ve had to discard 5% of the generated output. This is not much, but when you combine this uncertainty with all the others (retrieval failure, imprecise query) and this adds up to the stress of your RAG system.
* **Prompts can saturate**, especially if on top of asking for an advanced verifiability check, you are also making other recommendations for style (see section 4.3 below).
* **Prompts are vulnerable to injection and security issues**. This is the infamous “Chevrolet” effect where a help support system turns out to be a thin wrapper of ChatGPT.

**Assessing model robustness is tricky**. Standard benchmarks don't apply; because specialization through retraining results only in a general performance loss, this is what you are measuring.

The most straightforward way of measuring accuracy is to compare the model output and the original documents. This comparison obviously works best if you're already clarifying the attribution of individual statements to specific documents through generated references. For this process you can use LLM-as-a-judge, or, if you provide actual quotes, a straight (?) comparison with the original text (as an additional confusing factor, this is called a “text alignment” algorithm, but has nothing to do with fine-tuning or RLHF).

### 4.3 Purpose of fine-tuning: b) relieving data friction
Generalist models like GPT-3.5/4 for your RAG system, on a long term basis, creates an audience alignment gap. This is even more striking if (like myself) you’re working in another language than English: ChatGPT feels immediately “alien” with weird twists of language that are not to be found in standard French.

Reshaping style is the most shocking impact of fine-tuning. A few months ago I released an extreme version of user-LLM communication: MonadGPT, an instruction model trained on 17th century English. What took up the most time was coming up with a fitting communication scheme - i.e., training the models on a couple of answers from historical texts and generated questions in contemporary English, so that users aren't expected to mimic 17th century English, or the style of any of the document resources.

When fine-tuning, you should try to achieve this kind of style alignment in your design:
* Generated queries should be as close as possible to real queries sent by users, which probably means purposefully generating “bad” data. In my current instruction dataset, most queries have no punctuation, some are in all caps or missing words, with typos.
* Generated answers have to convey the general “brand” style of your system. Do not hesitate to re-generate an answer, reformulate an existing one, or even rewrite it manually.

Relatedly, you have to anticipate the “persona” of your model. Like it or not, users of interactive systems will ask meta-referential questions: Who are you? What can you do? Many advanced LLM systems overlooked this aspect to their detriment; because the default training data's persona these days is ChatGPT, new models trained from scratch erroneously say they are ChatGPT. To establish a unique persona for your model, you should combine both prepared answers in the instruction dataset and a series of “biographical” documents about the model in the RAG dataset.

The best way to assess style and related frictions is probably to simply collect feedback from your users. You don’t need anything fancy: an open-ended comment section somewhere (and green/red buttons) should work just fine.

## 5. Will RAG redefine LLMs?
Now, at mid-2024, we're probably at a crossroads. A major breakthrough in LLM research (e.g., Q*, Monte Carlo Tree Search, diffusion models, state space models, etc.) that shifts the focus from RAG systems back to model architecture is not impossible. But if this doesn't happen, LLMs will be increasingly pushed to the backstage of larger, more complex systems.

RAG has emerged as the main production use case for LLMs and will be at the forefront of model orchestration. In fact, in many ways orchestration has already happened. Embedding models are language models, and so are classifiers. Integrating multiple decoder models in production will just be a scaling challenge: the pattern is already there.

We are now in the early stages of a new generation of LLMs and SLMs made for RAG and RAG-associated tasks. We already have the first models - CommandR from Cohere, designed from the ground up with better support for source quotation and document interaction. The smallest variant of Phi-3 from Microsoft aims to operate best in contexts where knowledge and memory are separated from reasoning capacity.

In the current state of technology adoption, specialization is solving multiple complex problems. Specialization:
- reduces the overall memory footprint, as small models well integrated in the pipeline can suddenly become competitive with gigantic frontier models
- ensures that a model can run locally if necessary, and processes private organizational data without any liability
- limits the amount of work needed to deploy the model in production, because many requirements of a good RAG application (source verifiability, better grounding, better understanding of original sources) can be addressed at the pretraining level

At PleIAs, we aim to move even further in this direction. While several intermediaries focus on fitting internal data better for LLM processing, we work to bring models closer to the type of content actually used in production. Our current generation of models is trained on a large corpus of non-HTML documents (especially PDF) in a wide variety of formats and intentionally featuring many examples of digitization artifacts (OCR noise, bad segmentation) - a model can only get good at correcting things it has seen.

As with any integrated search and document processing system, a critical aspect of any RAG application is redundancy and fallbacks. Everything fails at some point - the unexpected query, the wrong document, the bad token. While pre-processing the RAG dataset achieves a lot, we believe that some of the redundancy and fallback burden should fall on the model itself, to ensure that it's sufficiently resilient to the content people actually use.

Another overlooked factor for propelling a new wave of specialized or retrained models is that RAG datasets are growing. Even for applications with a well-targeted audience, there are strong incentives to expand the range of indexed context - to cover more use cases. The improved quality and efficiency of search retrieval reverses means that content that is less related (for instance generalist newspaper articles or even some parts of common crawl) is increasingly usable.

A very large RAG dataset, properly documented and augmented with synthetic data, looks a lot like pretraining data. Both large scale RAG and the recent focus of pre-training on “quality data” invites us to think about “data orchestration”. Synthetic data is a hot topic in pre-training right now, not just as a source of new data, but as a powerful tool to process, clean and expand on existing resources. I’m expecting the same process to reverberate on the production side. It’s even likely that, in the near future, because data transactions have the potential to cut costs on model transactions, most RAG calls will return pre-generated texts.

Despite the current fascination for gigantic models, LM development may rely more on mid-size organizations with advanced data expertise. Diminishing returns for scale, increasing returns for customization convergently bring a future where RAG has absorbed LLMs.

## Contributors
[Pierre-Carl Langlais, author](https://www.linkedin.com/in/pierre-carl-langlais-b0105b10/)
[Mór Kapronczay, editor](https://www.linkedin.com/in/m%C3%B3r-kapronczay-49447692/)
[Robert Turner, editor](https://www.linkedin.com/in/robertdhayanturner)