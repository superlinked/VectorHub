<!-- SEO: We evaluate some Retrieval methods in Retrieval-Augmented Generation (RAG), comparing different popular chunking techniques, leaderboard single- and multi-vector embedding models, and reranking to see how they affect outcome accuracy and relevance on benchmark datasets - HotpotQA, SQUAD, and QuAC. Impressive results achieved by models ColBERT v2, WhereIsAI/UAE-Large-V1, BAAI/bge-large-en-v1.5, SentenceSplitter chunking, and TinyBERT-L-2-v2 reranker.-->

# An evaluation of RAG Retrieval Chunking Methods

Choosing a RAG Retrieval method that suits your use case can be daunting. Are some methods better suited to specific tasks and types of datasets than others? Are there trade-offs between performance and resource requirements you need to be aware of? How do different chunking techniques, embedding models, and reranking interact to impact performance results? Evaluation can help answer these questions.

To **evaluate the relative performance of several different, prominent chunking methods within the Retrieval component of a RAG system**, we looked at how they performed 1) on different leaderboard datasets, 2) using different parameters and embedding models, and 3) along several ranking metrics - MRR, NDCG@k, Recall@k, Precision@k, MAP@k and Hit-Rate, with k’s of 1, 3, 7, and 10.

Below, we present our datasets, chunking methods, embedding models, rerankers, and, finally, the outcomes of our research, within each dataset, and across all datasets.

## Datasets

We performed our evaluation of chunking methods on the following three datasets:

  1. [Dataset HotpotQA](https://huggingface.co/datasets/hotpot_qa?row=16)
  2. [Dataset SQUAD](https://huggingface.co/datasets/squad?row=0)
  3. [Dataset QuAC](https://huggingface.co/datasets/quac)

These datasets are widely used benchmarks in the field of Question Answering. They also contain a wide variety of questions, so we can use them to evaluate the performance of different methods across different query types.

## Chunking, with different parameters and embedding models

We experimented with [LlamaIndex- and LangChain-implemented **chunking methods**](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html), using different parameter combinations and embedding models:

  1. SentenceSplitter
  2. SentenceWindowNodeParser
  3. SemanticSplitterNodeParser
  4. TokenTextSplitter
  5. RecursiveCharacterTextSplitter ([LangChain](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter))

We chose **embedding models** from the top ranks of the MTEB Leaderboard:

  1. [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
  2. [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
  3. [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
  4. [sentence-transformers/all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1)
  5. [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)

In addition to the above single-vector representation models, we also tested a **multi-vector embedding and Retrieval** model, [ColBERT v2](https://huggingface.co/colbert-ir/colbertv2.0) using [RAGatouille](https://github.com/bclavie/RAGatouille/tree/main). 

ColBERT v2 embeds each text as a matrix of token-level embeddings, permitting more fine-grained interactions between parts of the text than with single-vector representation. RAGatouille provides optional chunking using LlamaIndex SentenceSplitter.

Finally, we also tested the effect on performance of different **rerankers** after Retrieval:

  1. [cross-encoder/ms-marco-TinyBERT-L-2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2)
  2. [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
  3. [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
  4. [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
  5. [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)

## Summary of outcomes

### Model

**ColBERT-based embedding** - using SentenceSplitter - **and Retrieval** proved to be the **most efficient method of improving result accuracy and relevance**, with an average performance advantage of about 10% over the second best method. This method's performance superiority held for all the datasets we tested.

### Chunking

The **SentenceSplitter** performed better than the other chunking methods, contrary to our expectations. We had intuitively assumed that the naive **SentenceSplitter** (which takes in chunk_size and overlap parameters) would be the _least_ efficient chunking method, and that instead the **SemanticSplitterNodeParser**’s performance would be _superior_ - because, for any given sentence, the latter method creates chunks based on breakpoints computed from semantic dissimilarities of the preceding and succeeding sentences. This assumption turned out to be false. **Why?**

The **superior performance of SentenceSplitter over SemanticSplitterNodeParser** appears to illustrate that, despite the advantages of semantic evaluation: 1) a sentence is a very natural level of granularity for containing meaningful information, and 2) high semantic similarity measures can result from noise rather than meaningful similarities. Vector representations of semantically similar sentences (provided there is no word overlap) are often more distant from each other than we might expect based on the sentences’ meaning. Words can mean different things in different contexts, and word embeddings basically encode the "average" meaning of a word, so they can miss context-specific meaning in particular instances. In addition, sentence embeddings are aggregates of word embeddings, which further "averages away" some meaning.

As a result, retrieval performance can suffer when we 1) do our chunking by splitting text on the basis of something other than the borders of sentences, 2) merge text segments into groups on the basis of semantic similarity rather than using a fixed length (i.e., basically 1-2 sentences).

### Reranking

Using rerankers after retrieval (as a post-processing step) was very good at improving result accuracy and relevance. In particular, reranker model **cross-encoder/ms-marco-TinyBERT-L-2-v2**, with only 4.3M parameters, was highly efficient in terms of inference speed, and also outperformed the larger models consistently across all three datasets.

Now, let’s take a look at our dataset-specific outcomes.

## Dataset HotpotQA results

![Chunking methods performance results on HotpotQA dataset](assets/use_cases/evaluation_of_RAG_retrieval_chunking_methods/mlflow_hotpotqa.png)

_Chunking methods performance results on HotpotQA dataset (above)_

On the HotpotQA dataset, the best performance came from **ColBERT v2**, which used the default **SentenceSplitter** chunker from LlamaIndex, with a max_document_length of 512. This method achieved an MRR of 0.3123 and Recall@10 of 0.5051.

The **second** best performance came from using SentenceSplitter with a chunk size of 128, embedding model **WhereIsAI/UAE-Large-V1**, with 335M parameters, and reranker **cross-encoder/ms-marco-TinyBERT-L-2-v2**. In fact, all the other single-vector embedding models, combined with SentenceSplitter chunking and the TinyBERT reranker, performed about as well as WhereIsAI/UAE-Large-V1, with minor differences. This includes model BAAI/bge-small-en-v1.5; it performed on par with the larger models despite being only 1/10th their size. 

The single-vector embedding models performed about as well as each other whether reranking was applied or not. Reranking improved their performance by about the same percentage for all these models. This was true not just for this dataset, but also across our other datasets (SQUAD and QuAC).

## Dataset SQUAD results

![Chunking methods performance on SQUAD dataset](assets/use_cases/evaluation_of_RAG_retrieval_chunking_methods/mlflow_squad.png)

_Chunking methods performance results on SQUAD dataset (above)_

On the SQUAD dataset, the **best ColBERT experiment** produced an MRR of 0.8711 and Recall@10 of 0.9581. These values are very high. We think this may suggest that the model was trained on SQUAD, though the ColBERT v2 paper mentions only evaluation of the Dev partition of SQUAD, which we didn't use.

On this dataset, the **BAAI/bge-m3** model, using the same **cross-encoder/ms-marco-TinyBERT-L-2-v2** reranker, produced the **second** best results - an MRR of 0.8286 and Recall@10 of 0.93. Without a reranker, BAAI/bge-m3’s MRR was 0.8063 and Recall@10 was 0.93.

BAAI/bge-m3’s scores are also (like ColBERT’s) **high**. It’s possible that this model was also trained on SQUAD, but [Huggingface doesn’t provide an exhaustive list of BAAI/bge-m3’s training datasets](https://huggingface.co/BAAI/bge-m3).

We tested multiple rerankers on this dataset of 278M-560M parameters, but they performed significantly worse than the small (TinyBERT) model, in addition to having much slower inference speeds. 

## Dataset QuAC results

![Chunking methods performance on QuAC dataset](assets/use_cases/evaluation_of_RAG_retrieval_chunking_methods/mlflow_quac.png)

_Chunking methods performance results on QuAC dataset (above)_

On the QuAC dataset, the **ColBERT experiment** achieved an MRR of 0.2207 and Recall@10 of 0.3144.

The **second** best performing model was **BAAI/bge-large-en-v1.5** with **SentenceSplitter**, chunk size of 128 and chunk overlap of 16, combined with the same **TinyBERT reranker**. The other models, when using the same reranker, performed roughly on par with this model.

Without the reranker, the different chunking methods, with the exception of the SemanticSplitter, would produce comparable results.

## In sum...

Here’s a tabular summary of our best performing methods for handling RAG Retrieval.

| Dataset      | Model                 | Chunker          | Reranker        | MRR   | Recall@10 |
| ------------ | --------------------- | ---------------- | --------------- | ----- | --------- |
| All datasets | ColBERT v2            | SentenceSplitter | None            | + 8%  | + 12%     |
| HotpotQA     | ColBERT v2            | SentenceSplitter | None            | 0.3123| 0.5051    |
| HotpotQA     | WhereIsAI/UAE-Large-V1| SentenceSplitter | TinyBERT-L-2-v2 | 0.2953| 0.4257    |
| SQUAD        | ColBERT v2            | SentenceSplitter | None            | 0.8711| 0.9581    |
| SQUAD        | BAAI/bge-m3           | SentenceSplitter | TinyBERT-L-2-v2 | 0.8286| 0.93      |
| SQUAD        | BAAI/bge-m3           | SentenceSplitter | None            | 0.8063| 0.93      |
| QuAC         | ColBERT v2            | SentenceSplitter | None            | 0.2207| 0.3144    |
| QuAC         | BAAI/bge-large-en-v1.5| SentenceSplitter | TinyBERT-L-2-v2 | 0.1975| 0.2766    |

Our **best performing method** for handling RAG Retrieval on all datasets was **ColBERT v2 with SentenceSplitter chunking.

Our **other (single-vector) embedding models**, though trailing in performance behind ColBERT v2 (with SentenceSplitter), tended to perform **about the same** as each other, both when they were combined with reranking and when they weren’t, across all three datasets.

**SentenceSplitter chunking** surprised us by outperforming SemanticSplitterNodeParser, but upon further reflection, these outcomes suggest that sentences are natural delimiters of meaning, and semantic “averaging” of meaning may miss context-specific relevance. 

Finally, **reranker model TinyBERT** proved to be the most efficient at improving model performance, outperforming even the larger rerankers.

## Contributors

- [Kristóf Horváth, author](https://www.linkedin.com/in/kristof-horvath-0301/)
- [Mór Kapronczay, contributor](https://www.linkedin.com/in/mór-kapronczay-49447692)
- [Robert Turner, contributor-editor](https://www.linkedin.com/in/robertdhayanturner/)
