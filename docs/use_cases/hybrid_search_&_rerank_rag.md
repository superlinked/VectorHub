<!-- SEO: Retrieval-Augmented Generation (RAG) is revolutionizing traditional search engines and AI methodologies for information retrieval. However, standard RAG systems often lack efficiency and precision when dealing with extensive data repositories. Substituting the search approach with a hybrid method and reordering the outcomes using a semantic ranker considerably enhances performance, indicating potential for large-scale implementations.
-->

# Hybrid Search + Rerank: Optimizing RAG

RAG (Retrieval Augmented Generation) has been the preferred strategy for AI developers all around the world to power their LLMs with up-to-date knowledge. It introduces a new age of information retrieval and search which has enabled many AI technologies we use often, like the Bing Chatbot, Google Bard or even the ChatPDF. A RAG architecture assists an LLM in identifying the most pertinent context and providing precise responses, similar to how a librarian aids an individual in choosing a preferred book from a vast collection.

![Conventional RAG Workflow](docs\assets\use_cases\hybrid_search_&_rerank_rag\RAGDiagram.png "Fig 1")

As depicted in the figure, the retriever fetches relevant data from the knowledge base whenever a query is passed and the LLM can answer the query by examining the context provided. The accuracy of the response increases with the precision of the retrieved context. RAG solves a major issue in LLMs called hallucination where LLMs fail to respond with relevant context to the query and spit out irrelevant or false data which it thinks is true and thus gives the phenomenon its name.

## Challenges & Shortcomings of RAG

Even though RAG has been spearheading recent developments in this space, it has a lot to improve on what it does. Conventional RAG systems drag behind with latency when dealing with a sufficiently large knowledge base. When there is a vast amount of data in the knowledge base, conducting a similarity search across the entire corpus can be quite time-intensive. They also struggle to find relevant passages from the corpora using simple similarity search algorithms. 
Normal RAG systems find it difficult to match exact names like “Joe Biden” or “Salvador Dali” in the query and knowledge base. It also misses out on abbreviations like GAN or LLaMA when their full forms are present in the query or knowledge base.  

## Why Hybrid Search + RAG?

Vector similarity works well even when there are spelling mistakes in the query because these typically don’t alter the overall meaning of the sentence. However, for precise word or abbreviation matching, vector similarity may not suffice because the abbreviations and names just dissolve in the vector embeddings along with the words around them. So, we need something to keep tabs on keywords also. 
How can we solve such issues? One way is to tweak the search method to accommodate keywords. However, this would make the retrieval system weaker as identifying similarities is a fundamental objective when we decide to transform data into embeddings. The perfect way to do this is to take the best from semantic search and keyword search approaches while mitigating their limitations as much as possible. This keyword-sensitive semantic search is what we call “Hybrid Search”.

![Hybrid Search in RAG](docs\assets\use_cases\hybrid_search_&_rerank_rag\HybridSearch.png "Fig 2")

Microsoft discusses this concept in their article: “Azure AI Search: Outperforming vector search with hybrid retrieval and ranking capabilities”.  Document chunking plays a crucial role in deciding the speed of search and the performance of the retrieval system. The whole document is divided into chunks with a fixed token length and indexed. Usually, the BM25 similarity search algorithm is employed for keyword matching in embeddings. BM25 is a way of finding the most relevant documents for a given query by looking at two things:
- How often the query words appear in each document (the more, the better).
- How rare the query words are across all the documents (the rarer, the better).
When it comes to vector similarity, an algorithm such as cosine similarity search would be adequate.  The results from vector and keyword searches are combined using the Reciprocal Rank Fusion technique. This method ranks each passage according to its place in the keyword and vector outcome lists and subsequently merges these rankings to generate a unified result list. 
Hybrid retrieval thus utilizes the combined advantages of keyword and vector retrieval to identify the passages most pertinent to the query.

## The final touch of Rerank

A response by hybrid search does not solve our issue completely. A hybrid search scans the entire corpus to find all possible sections that could contain the answer. Typically, algorithms yield the top-k matches. However, the challenge lies in the fact that these top-k matches may not always include the relevant sections, or conversely, not all relevant sections may be within these top-k matches. At this point, we recognize the necessity to rank all the retrieved content based on a score that indicates semantic relevance with the query.

![Hybrid Search + Rerank](docs\assets\use_cases\hybrid_search_&_rerank_rag\Rerank.png "Fig 3")

The responses from the retriever are passed to a semantic scoring model. Semantic scoring models are transformer models that take in queries and documents to produce a score in a calibrated range. There are many models like ember that are available to use. In the Azure AI approach, 0 stands for irrelevant and the maximum score is at 4. After reranking, a list of documents is returned, sorted according to relevance score, from highest to lowest.  Results are arranged in a sequence based on their scores and incorporated into the response payload of the query. The relevance score will be based on the semantic similarity calculated between the query and the document.
Now, the LLM gets the most relevant part of the document that can satisfy what the user has asked. It then generates appropriate responses based on this content.

## Results

Hybrid search and reranking have shown significant progress in RAG performance. The quality of the retrieved content has increased a lot by employing hybrid search which in turn will aid the generation phase of the RAG architecture. The recall score of the model can be assessed by the fact that almost all relevant content is present in the top-k retrievals. 
Here are some comparisons with various approaches on benchmark and real-world datasets from the Azure AI article.

![Comparison of Retrieval Modes](docs\assets\use_cases\hybrid_search_&_rerank_rag\RetrievalComparison.png "Fig 4")


The below chart from Azure AI shows the percentage of queries showing high-quality results in the top-5 retrievals.

![Percentage of queries where high-quality chunks are found in the top 1 to 5 results](docs\assets\use_cases\hybrid_search_&_rerank_rag\Performancegraph.png "Fig 5")

With hybrid search and semantic ranker, over 75% of queries show high-quality results in top-5 retrievals and almost 60% of queries give relevant results right in the top-1 retrieval. 

## Moving Forward

We have discussed how hybrid search + rerank offers enhanced performance in RAG systems for keyword matching and refinement in response. The higher recall rates of the retriever have aided in better responses from the LLM. Retrieval Augmented Generation is transforming the way we work with search engines and documents and so we expect to see much more developments in this space. Another thing we learned from this approach here is that substituting an entire method in RAG for either search or response generation yields superior results compared to minor tweaks made sporadically. 
Hope you enjoyed the article. Thank you for engaging.

## References

[Azure AI](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167)
[Dify.ai](https://dify.ai/blog/hybrid-search-rerank-rag-improvement)
