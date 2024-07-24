# Your RAG application is a communication system

It's become routine for large organizations to employ Retrieval-Augmented Generation when using LLMs on their data. By grounding LLM generation in existing data, RAG ensures increasingly robust resistance to hallucinations, adaptation to in-domain knowledge, and updatability. Still, RAG is typically understood as a temporary workaround, to be made redundant after the eventual advent of LLMs with improved context length and better intelligence.

After designing multiple RAG systems over the past year, I’ve come to believe that RAG systems represent a more radical transformation that will be here for years to come, more enduring than any LLM model they happen to integrate. RAG systems have the capacity to address the fundamental challenges of information retrieval, meeting the user and the data where they're at. Effective RAG can take the best features of language models (adapatability, universal content translation) but also moderate their shortcomings (misalignment, hallucinations, and lack of explainability), ingesting even heterogeneous, badly formatted data sources, and contextualizing new knowledge on the fly.

The majority of vector search deployments today employ RAG. RAG has become so integral to vector search, I recently not only find myself fine-tuning LLMs with RAG in mind, but also cofounded PleIAs to pre-train a new generation of LLMs for RAG end use and document-processing use cases. RAG applications are not likely to be replaced by some disruptive revolution in LLMs, like long context. It's rather LLMs, as Yan et al. [predicted](https://applied-llms.org/#dont-forget-keyword-search-use-it-as-a-baseline-and-in-hybrid-search), that are "likely to be the least durable component in the system" - swapped, combined, disabled.

RAG can be nothing less than a flexible, expert communication system - **a feedback loop between LLMs and the data & user landscape of a company**. RAG applications, therefore, should serve as the focal end point of fine-tuning whatever LLM your RAG uses. Let's see how, in more detail, below.

## outline:

1. how RAG addresses information retrieval challenges faced by but predating LLMs; 
production RAG is dynamic - always solving the last mile problem - closing the loop between data and users via LLMs...; crucial in this = evaluation...
2. info retrieval challenges are data friction challenges (between data, system, and users); must take advantage of the best tools even if they're not new tools... hybrid approach
3. reality = need bad data..
4. ...

## 1. RAG - a comm system solution for few-doc learning (LLM) limitations

LLMs are powerful. One of their most valuable properties is language transfer - rephrasing and reformulating an existing text in a new style or for a different audience. LLMs' ability to handle, for example, cross-lingual transfer, is surprisingly efficient - many models trained primarily in English can be trained successfully on downstream tasks in other languages. This extends to other forms of language transfer: formal to informal, standards to dialects, generalist to specialized terminologies.

But LLMs are also limited, and these limitations result from the fact that LLMs are few doc-learners. LLMs have largely been conceived as “interpolated datasets” (holding in memory the collective knowledge of the Internet), and they share the challenges of interpolated datasets. (Even the inaugural RAG system was still a form of interpolated dataset: a clever design associating a seq2seq model with a dense layer trained on Wikipedia ([Lewis et al., 2020](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)).)

Like other interpolated datasets, LLMs are trained at a point in time on a finite set of data, and this creates problems. Problems that can to some degree be mitigated by RAG systems:

* **Specialization**: professional resources are often not or only lightly included in pre-training data, resulting in high hallucination rates. *RAG systems can provide coverage of these resources*.
* **Updatability**: while LLMs are limited by their cutoff date, *RAG systems can be continuously maintained in real time*.
* **Explainability**: LLMs are a “black box”; *RAG systems can provide transparency, tracing statements to original sources*.

The default paradigm and basic conceptualization of RAG emerged almost immediately after the release of chatGPT. Simon Willison’s “[semantic search answers pattern](https://simonwillison.net/2023/Jan/13/semantic-search-answers/)” is one of the first (January 2023) systematic descriptions: search for texts in a semantic database, get the most relevant results, send this input to GPT-3 as inspiration.Following this basic strategy, RAG helps mitigate the limitations of few document learning by grounding an LLM on existing data at inference time.

**But RAG doesn't solve LLM limitations once and for all**. *In production*:

* A. there's *no one-size-fits-all RAG solution*, as people working on production RAG today soon realize. Good applications are increasingly a lot of flows and arrows, anticipating a wide array of edge cases. There are simply too many breaking points, e.g., bad OCR, poor indexation, suboptimal tokenization, inconsistent document segment, unpredictable queries, etc., to settle on one generic approach.
* B. your solution to the last mile problem (indexing, retrieval, and generation) depends heavily on your use case: This won't be easily handled by LLMs, despite major AI actors' eagerness to create innovative panacea platforms. Probably the only actual moat in finding the right fit for your use case, corpus, and specific user target is in applied generative AI - i.e., in a specifically tailored RAG system.
* C. RAG evaluation is tricky: While you can use synthetic data to do preliminary evaluations and benchmarks for selecting and swapping models, the real challenge is evaluation in production of your system's different components, taking into account their level of specialization. Though we'd like to automate everything, including evaluation, here at least it continues to require qualitative human involvement.

These ongoing hurdles are at once challenges *and* opportunities for RAG applications qua communication systems, employing LLMs in a particular landscape of specific data and users. In a way, this has always been RAG's *raison d'etre*. But, as communication systems, RAG applications can perform better by taking advantage of a broad range of techniques, not all of them new, as we'll see below.

## 2. Retrieval: many solutions - not everything new is better

RAG is a new solution, and not the only one, to an issue that LLMs aimed to solve: information retrieval. Our retrieval solutions need to be adaptive, incorporating the best performing parts of techniques, regardless of when they originated.

### 2.1 The actual roots of RAG: decades of information retrieval and data frictions

Expert systems and knowledge infrastructures have existed for decades, and dealt with the same data frictions that RAG applications have to deal with today. On the data ingestion side, you still need to do additional work to fit new content to mandatory formats. On the query side, users don't act the way the search infrastructure expects.

The first expert systems emerged in the 1960s, with the expectation that professionals themselves would access them seamlessly. They wildly underestimated the intermediation needed to get them working. MEDLINE, for example, was designed for medical researchers and clinicians, and NASA/RECON, for aerospace engineers and scientists. But most MEDLINE and NASA/RECON users through the 1970s were librarians and trained intermediaries, working on behalf of end users.

The advent of LLMs and RAG also promised a radical simplification of information retrieval - taking whatever content there is, and outputting whatever content the user wants. This is the objective of LLM on the output side, and embedding models on the input side. Embeddings are natively multilingual, theoretically resilient to synonyms, variations of language standards, and document artifacts.

But the radical promise of LLMs and RAG, as with the previous expert systems, was not fulfilled, on the same two fronts: understanding users, and understanding data.

### 2.2 Know your users

It may seem obvious that knowing what your users are doing with your system is a necessary prerequisite to designing good information retrieval. But until recently, user behavior in the context of RAG has barely been discussed.

In general, RAG applications underestimate the inertia of user practices; users continue behaving the same way they did with older expert systems. I recently analyzed a set of internal user queries in a production chatbot, and was surpised to see that many users were not submitting questions but unstructured keywords. This practice works perfectly in a standard information retrieval system, but instruction-based LLM and embedding models are designed to be trained on question/answer pairs.

Inattentiion to user - design friction in LLMs and embedding models can result in poor performance, disappointed user expectations, tokenization that fails to adequately represent data structure, incongruity of input vs output lengths, and an overall lack of control over the system's purpose.

### 2.3 Know your data

Like your users, your data won't necessarily comply with the constraints of LLMs and embedding models.

Context window, though it's gradually gotten longer, is still a limiting factor - the indexation of long texts continues to underperform. This results directly from a mismatch between pre-training data and data in production: the vast majority of Common Crawl texts are less than 512 tokens, whereas organizations routinely deal with multi-page PDFs.

Limited context windows require you to split or chunk your document, which is not trivial. Documents parts are not meant to be standalone units of content, and disambiguation requires some level of cross-referencing. The biggest performance gain on my retrieval evaluation,especially on projects in the French public service sector, came from the systematic inclusion of title hierarchy (document > chapter > section, etc.); in long documents, general information on the topic is given at this scale, but not recalled in specific paragraphs.

Proper contextual support for original data is a concern in not just retrieval but also *generation*. Even powerful LLMs (frontier LLMs) are highly sensitive to document structure. Lack of proper segmentation or unusual structure will result in poor tokenization (typically, merging a document separator with a token), and a higher rate of omission and hallucination. With tables at least, a hacky (but verbose) fix is to recall the row name/column name couple before each entry.

Besides data context issues, there are also often issues with retrieval metrics.

Existing retrieval metrics, which rely on a binary relevance model, may not work well with embedding models, which rely on “similarity” to represent ingested data. Moreover, embedding models are trained on specific sets of documents and questions (typically bases like MS-Marco), and may not perform as well on specialized resources. Also, users' understanding of what data counts as similar may differ from what an embedding model represents as similar.

To meet users' expectations and do justice to our data sources, we can't just simply rely on LLM embedding models or set RAG systems. We need a different, more flexible approach.

### 2.4 Settling for hybrid

Even with recent advances (longer document limits, additional context enrichment), embedding models in RAG systems can barely compete with and have to be supplemented by classic information retrieval approaches, like BM25. This suggests a direction: recover info retrieval methods and indicators that can form part of a more flexible, better performing RAG comms system.

In practical terms:

* Hybrid indexation is emerging as the new standard. On my latest internal benchmarks, combining a keyword-based search with embedding similarity achieved the highest retrieval rate (nearly 90% of relevant resources in the top ten results). The easiest implementation I’ve found so far is LanceDB’s, but the jury's still out - there's a lot of activity in this space right now.
* Keyword-based search is still a strong alternative to vector search, especially when dealing with longer texts or more expansive context information. Embedding at a very large scale can become prohibitive, while the infrastructure for keyword search already exists.
* Evaluation has to be built from the ground up, using criteria based on the constraints of your existing infrastructure (especially with regard to data quality/indexation), and the ways users interact with it.

In this time of LLMs, older information retrieval methods and indicators continue to hold a lot of unrealized value, especially now that it's possible to generate/extract many key data features at scale. Jo Kristian Bergum from Vespa, for example, has [convincingly demonstrated](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/) how classic info retrieval evaluation design and metrics (precision at k, recall) can be effectively repurposed using emerging practices in AI, such as LLM-as-a-Judge - grounded on a small relevant dataset, but scalable. Intensive data work that would have been available only to large scale organizations is now scalable with far fewer resources.

Generative AI within a RAG communication system shouldn't be looking to replace the classic approaches of retrieval evaluation; it should instead reshape their logistics to take full advantage of them.

## 3. Keep the data alive

A proper RAG communication system should treat data no longer as a passive reference stock but rather a living entity - in two ways. 
Data should be:

* a) continuously transformed and reshaped to better fit the retrieval objective, and
* b) constantly circulated across different flows.

### 3.1 You need bad data

As soon as you get even slightly beyond the proof-of-concept phase, your RAG system has to work with whatever data is available. To productionize, you have to become an expert on bad data.

Scaling is a bitter pill, and not just for LLMs: I worked in cultural heritage for years, and a library with millions of "bad", uncleaned OCR documents gets far more user interaction than one with a thousand perfectly (and expensively) cleaned and edited OCR texts. This also holds for large organizations. Their RAG has to work with legacy texts - i.e., PDFs, whereas LLMs are trained mostly on Common Crawl (web/html archives), not the digitization artifacts and advanced document structures of PDF documents. LLM trainers also don't have access to the bad data actually processed in production - from both users (minimal effort, "lazy" queries) and documents, either because of a poor retrieval system or they're restricted by data protection regulations, other internal organizational barriers.

But there are workarounds. The usual one is looking for placeholders, or generating them. Google (and probably other major AI providers), for example, have been seriously investing in data degradation techniques. Simple data processing heuristics (e.g., dropping cases, introducing noise) will do the job. My own experience has shown fine-tuning on existing bad data to be the most powerful way of preparing a RAG system for production use cases.

### 3.2 Managing data flows with classifiers

For processing bad (and good) data, we take advantage of LLM data processing power. But here too the best approach is using the best LLMs for production use cases, and not simply the newest or most popular decoders. The previous generation of encoders and encoder-decoders provide much better parameters / performance / robustness ratios for many data tasks than GPT-4, Llama or Mistral. I increasingly build RAG systems with classifiers or "reformulators" that can reliably handle:

* Automated re-routing of user queries. A major issue of more generalist chat systems is deciding 1) whether or not to activate the retrieval of external resources, 2) which collection of resources to prioritize (maybe not all are needed; reducing volume can relieve stress to the system), and 3) when a new round of retrieval becomes necessary (e.g., when the topic of a conversation has changed).
* Query expansion. LLM and embedding models, because they're trained primarily on complete sentences, struggle with unstructured queries. You may require further contextualization by adding elements (e.g., past queries).
* Text segmentation (rather than just “chunking”). Proper segmentation has proven to be one of the most powerful performance boosts. But as user traffic increases, and more document coverage is needed, iteratively expanding your RAG system (typically by parsing different sets of resources with existing ones) can become prohibitively costly. Lately, I’ve been experimenting a lot with agnostic encoder models for token-classification tasks to reconstruct text parts (including titles, bibliography) from raw text comprehension, regardless of the original format.

For automated query rerouting, query expansion, and text segmentation, using an LLM is overkill and results in performance lag and degradation. Once properly re-trained, encoder models yield comparable results, are very fast, not expensive to host, and better fitted to the overall purpose of classification. In addition, encoders don't require structured generation, regex, or data constraints - the data design is already built-in.

### 3.3 An alternative to generation: synthetic data curation

In the months to come, I anticipate a lot of research on pre-generation of system input and output from a model, rather than generation at inference time. As RAG systems scale, pre-generation (solving and storing query answers beforehand) provides significant economies of scale, preferably when you have a general grasp of what queries or topics users ask frequently. Most queries follow a 80-20 rule (80% of questions are about 20% of topics); once a question is solved and stored, there's no need to regenerate it, saving computational resources.

It's also possible to expand the original dataset to include the types of queries your users are likely to write. At this point, a RAG dataset starts to look like an instruction dataset, structured around a couple of answers and (generated) queries, and therefore also becomes a resource for fine-tuning and reconfiguring the model's behavior.

## 4. Fine-tuning as a communication protocol

In a RAG context, fine-tuning is neither an alternative nor simply a good supplement to content retrieval: LLM memory is inconsistent, even moreso when it relies on LoRA (Low-Rank Adaptation) rather than pretraining. Fine-tuning is better conceived as a **method for enhancing communication**, by tweaking the word probabilities of the base model in a specific direction: communication between the model and the user, communication between the resources and the output, and, as agent-based approaches become more prominent, communication between different models.

### 4.1 Fine-tuning for/through RAG

Fine-tuning has to come later in the cycle of development. In my experience, good instruction datasets need to be designed not only with RAG in mind but through (i.e., using) RAG. My standard protocol for RAG-based fine-tuning goes like this:

* Generate queries based on indexed texts. This is a basic and universal approach when using synthetic data for fine-tuning: for a given text, invent a question it would answer. I recommend prompting the model to create questions “vague” enough to yield other (not the seeding text) valid answers. Also, to make the queries to match those sent by your users, you should include in the prompt a random selection of past queries, constraining the model so it produces realistic synthetic data - i.e., with potential imperfections (all lower case, including syntax errors, etc.).
* Retrieve a broad selection of documents. You might even voluntarily degrade retrieval performance (e.g., use only BM25) - to make the LLM more resilient to hard negatives.
* Generate a cohesive, comprehensive response - as a synthesis based on the question and the retrieved documents. This creative and crucial part controls the final communication output of the model. You don’t have to get it right in one pass; LoRA fine-tuning is cheap. I start with a simple RAG prompt on an existing model, and then continuously reshape and rework this output.

A good fine-tuning dataset, though it requires a significant amoung of careful manual tweaking, does *not* require a lot of examples. Most of my projects are in the 1,000-4,000 instructions range, sufficiently large to warrant text generation, sufficiently small to still be mostly an opinionated product. The instruction dataset is in many ways a miniature version of the entire RAG communication system. It needs to approximate in format and style the future queries your users will send, the existing and future documents you will use for grounding, and the expected output. In short, you are designing a translation process and - through the instructions - providing it all the use cases where this translation went right.

The preparation of the instruction dataset should be your main focus. I don't spend much time optimizing the training design beyond a few hyperparameters (learning rate, batch size, etc.). Work on the instruction dataset and base model improvement always have the most impact on performance. I've also generally stopped looking into preference fine-tuning (like DPO); the time spent was not worth the very few improvement points.

While it's far less common, this approach - fine-tuning your instruction dataset using RAG-generated synthetic data - [can be applied to embedding models](https://huggingface.co/blog/davanstrien/synthetic-similarity-datasets). Synthetic data makes it considerably easier to create an instruction dataset that maps the expected format of the similarity dataset (including queries and “hard negatives”). Fine-tuning your embedding models with synthetic data will confer the same benefits as LLM fine-tuning: cost savings (a much smaller model that demonstrates the same level of performance) and appropriateness by bringing the “similarity” score closer to the expectations of your retrieval system.

### 4.2 Fine-tuning for robustness

LLMs are built for general use, rather than specifically designed for your use cases. Prompt engineering on them can only take you so far. The primary objective of fine-tuning a RAG system is to improve its ability to generate documents that are accurate, relevant, and reliable in terms of your specific use case/s. 

Robust RAG document synthesis requires:

* **Content fidelity**: the model should generally not add more than what is presented in the retrieved document (i.e., hallucinate), at least beyond some general knowledge.
* **Dealing with uncertainty**: queries can fail either because there are no proper answers in the RAG dataset, or because retrieval fails. A good model will anticipate this, and as a fallback acknowledge that it can't find a suitable answer.
* **Tracking the information**: RAG-dedicated models have increasing support for providing citation and references for each statement they retrieve from a source. (This approach was probably introduced by Perplexity.)
* **Setting the stage for verifiability**: further, references can be linked to specific quotations excerpted from the sources they were retrieved from, potentially improving accuracy and making it easier for users to navigate to original sources - for less cumbersome fact-checking.

Our RAG projects at PleIAs were conceived in the context of severe constraints on verifiability (public sector, large legacy privacy companies). We took inspiration from Wikipedia, who faced similar constraints: because texts can't be attributed to individual authors who take responsibility for their publication and trustworthiness, accuracy stems mostly from verifiability - associating individual statements with exact passages in secondary sources. While Frontier LLMs can achieve high quality performance without fine-tuning (and this was actually [our first approach](https://huggingface.co/AgentPublic/guillaumetell-7b)), we have opted for the RAG fine-tuning approach to avoid the **limitations of Frontier LLMs**:

* **Prompts won't work systematically**. While generating instruction datasets for RAG, I’ve had to discard 5-10% of the generated output, which added to other stresses on your RAG system (retrieval failure, imprecise query) is non-trivial.
* **Prompts can saturate**, especially if, on top of asking for an advanced verifiability check, you make style recommendations (see section 4.3 below).
* **Prompts are vulnerable to injection and security issues** - e.g., a help support system that turns out to be a thin wrapper of ChatGPT.
* **Assessing model robustness is tricky**. Standard benchmarks don't accurately reflect the performance of specialized models retrained for a special task, and instead register a general performance loss.

The most straightforward way to measure accuracy is comparing the model output with the original documents. This works best when you're simply checking a system you've set up for attributing specific statements to their respective documents using generated references. You can use LLM-as-a-judge, or, if you provide actual quotes, a direct comparison with the original text (i.e., via a “text alignment” algorithm).

### 4.3 Fine-tuning to relieve data friction

On a long term, using generalist models like GPT-3.5/4 in your RAG system creates an audience alignment gap. This gap is even more pronounced if (like me) you’re working in a language other than English - ChatGPT introduces strange ways of saying things that no one speaking standard French would use.

In my experience, fine-tuning is the most effective method of reshaping the style and overall method of communication of an LLM. A few months ago, I released MonadGPT, an instruction model trained on 17th century English. The fitting communication scheme took the most time - i.e., fine-tuning an LLM on a couple of answers from historical texts, and generated questions in contemporary English, so that users aren't expected to mimic 17th century English, or the style of any of the document resources.

When fine-tuning, you should aim for the following kind of style alignment:

* Generated queries (as discussed above) should be as close as possible to real user queries, which probably means purposefully generating “bad” data. In my current instruction dataset, most queries have no punctuation, some are in all caps, are missing words, and/or have typos.
* Generated answers should convey the general “brand” of your system. Do not hesitate to re-generate an answer, reformulate an existing one, or even rewrite it manually.
* Anticipate your model's "persona". Interactive system users will ask meta-referential questions: Who are you? What can you do? New models' training data persona is typically ChatGPT, so some erroneously state that they are ChatGPT. To establish a unique persona for your model, build redudancy into it - with both prepared answers in the instruction dataset (i.e., fine-tuning) and a series of “biographical” documents in the RAG dataset (vector store, for reference at inference time).

The best way to assess style (and related) frictions is probably to simply collect feedback from your users. You don’t need anything fancy: an open-ended comment section somewhere (and green/red buttons) works just fine.

## 5. Will RAG redefine LLMs?

Now, in mid-2024, we're probably at a crossroads. Though a major breakthrough in LLM research (e.g., Q*, Monte Carlo Tree Search, diffusion models, state space models, etc.) that shifts the focus from RAG systems back to model architecture is not impossible, LLMs seem to be increasingly pushed backstage, while larger, more complex RAG systems are garnering increasing attention.

As the main production use case for LLMs, and at the forefront of model orchestration, RAG systems employ LLMs as embedding models and classifiers. Integrating multiple decoder models in production will follow once they can be scaled. We are in the early stages of a new generation of LLMs and SLMs *made for* RAG and RAG-associated tasks - e.g., CommandR from Cohere are designed from the ground up with better support for source quotation and document interaction. Microsoft's smallest Phi-3 variant (a local LLM used in RAG setups) is optimized for contexts where knowledge and memory are separated from reasoning capacity, as they are in RAG.

RAG's ability to take advantage of specialization through model orchestration gives RAG an advantage in solving multiple complex problems:

* reducing the overall memory footprint; small models well integrated in the pipeline can compete with gigantic frontier models
* ensuring that a model can run locally and process private organizational data without liability
* limiting the amount of work needed to deploy the model in production; many requirements (source verifiability, better grounding, better understanding of original sources) can be addressed at the pretraining level

At PleIAs, we aim to move even further in this direction. While several intermediaries focus on fitting internal data better for LLM processing, we work to bring models closer to the type of content actually used in production. Our current generation of models is trained on a large corpus of non-HTML documents (especially PDF) in a wide variety of formats and intentionally featuring many examples of digitization artifacts (OCR noise, bad segmentation) - a model can only get good at correcting things it has seen.

As with any integrated search and document processing system, a critical aspect of any RAG application is redundancy and fallbacks. Everything fails at some point - the unexpected query, the wrong document, the bad token. While pre-processing the RAG dataset achieves a lot, we believe that some of the redundancy and fallback burden should fall on the model itself, to ensure that it's sufficiently resilient to the content people actually use.

Another overlooked factor for propelling a new wave of specialized or retrained models is that RAG datasets are growing. Even for applications with a well-targeted audience, there are strong incentives to expand the range of indexed context - to cover more use cases. The improved quality and efficiency of search retrieval reverses means that content that is less related (for instance generalist newspaper articles or even some parts of common crawl) is increasingly usable.

A very large RAG dataset, properly documented and augmented with synthetic data, looks a lot like pretraining data. Both large scale RAG and the recent focus of pre-training on “quality data” invites us to think about “data orchestration”. Synthetic data is a hot topic in pre-training right now, not just as a source of new data, but as a powerful tool to process, clean and expand on existing resources. I’m expecting the same process to reverberate on the production side. It’s even likely that, in the near future, because data transactions have the potential to cut costs on model transactions, most RAG calls will return pre-generated texts.

Despite the current fascination for gigantic models, LM development may rely more on mid-size organizations with advanced data expertise. Diminishing returns for scale, increasing returns for customization convergently bring a future where RAG has absorbed LLMs.

## Contributors
[Pierre-Carl Langlais, author](https://www.linkedin.com/in/pierre-carl-langlais-b0105b10/)
[Mór Kapronczay, editor](https://www.linkedin.com/in/m%C3%B3r-kapronczay-49447692/)
[Robert Turner, editor](https://www.linkedin.com/in/robertdhayanturner)