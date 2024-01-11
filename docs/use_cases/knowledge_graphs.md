# Using Knowledge Graphs + Embeddings to Further Augment RAG Systems

We look at the limitations of not just LLMs but also standard RAG solutions to LLM's knowledge and reasoning gaps, and examine the ways Knowledge Graphs combined with vector embeddings can fill these gaps - through graph embedding constraints, judicious choice of reasoning techniques, careful retrieval design, collaborative filtering, and flywheel learning. 

## Introduction

Large Language Models (LLMs) mark a watershed moment in natural language processing, creating new abilities in conversational AI, creative writing, and a broad range of other applications. But they have limitations. While LLMs can generate remarkably fluent and coherent text from nothing more than a short prompt, LLM knowledge is not real-world data, but rather restricted to patterns learned from training data. In addition, LLMs can't do logical inference or synthesize facts from multiple sources; as queries become more complex and open-ended, LLM responses become contradictory or nonsense.

Retrieval Augmented Generation (RAG) systems have filled some of the LLM gaps by surfacing external source data using semantic similarity search on vector embeddings. Still, because RAG systems don't have access to network structure data (the interconnections between contextual facts), they struggle to achieve true relevance, aggregate facts, and perform chains of reasoning.

Knowledge Graphs (KGs), by encoding real-world entities and their connections, overcome the above deficiencies of pure vector search. KGs enable complex, multi-hop reasoning, across diverse data sources, thereby representing a more comprehensive understanding of the knowledge space. 

Let's take a closer look at how we can combine vector embeddings and KGs, fusing surface-level semantics, structured knowledge, and logic to unlock new levels of reasoning, accuracy, and explanatory ability in LLMs. 

We start by exploring the inherent weaknesses of relying on vector search in isolation, and then show how to combine KGs and embeddings complementarily, to overcome the limitations of each.


## RAG Vector Search: process and limits

Most RAG systems employ vector search on a document collection to surface relevant context for the LLM. This process has **several key steps**:

- 1. **Text Encoding**: Using embedding models, like BERT, the RAG system encodes and condenses passages of text from the corpus as dense vector representations, capturing semantic meaning.
- 2. **Indexing**: To enable rapid similarity search, these passage vectors are indexed within a high-dimensional vector space. Popular methods include ANNOY, Faiss, and Pinecone.
- 3. **Query Encoding**: An incoming user query is encoded as a vector representation, using the same embedding model.
- 4. **Similarity Retrieval**: Using distance metrics like cosine similarity, the system runs a search over the indexed passages to find closest neighbors to the query vector.
- 5. **Passage Return**: The system returns the most similar passage vectors, and extracts the corresponding original text to provide context for the LLM.
  
This RAG Vector Search pipeline has **several key limitations**:

- Passage vectors can't represent inferential connections (i.e., context), and therefore usually fail to encode the query's full semantic intent.
- Key relevant details embedded in passages (across sentences) are lost in the process of condensing entire passages into single vectors.
- Each passage is matched independently, so facts can't be connected or aggregated.
- The ranking and matching process for determining relevancy remains opaque; we can't see why the system prefers certain passages to others.
- There's no encoding of relationships, structure, rules, or any other connections between content.

RAG, because it focuses only on semantic similarity, is unable to reason across content, so it fails to really understand not only queries but also the data RAG retrieves. The more complex the query, the poorer RAG's results become.

## Incorporating Knowledge Graphs

Knowledge Graphs, on the other hand, represent information in an interconnected network of entities and relationships, enabling more complex reasoning across content.

How do KGs augment retrieval?

- 1. **Explicit Facts** — KGs preserve key details by capturing facts directly as nodes and edges, instead of condensed into opaque vectors.
- 2. **Contextual Details** — KG entities possess rich attributes like descriptions, aliases, and metadata that provide crucial context.
- 3. **Network Structure** — KGs capture real-world relationships - rules, hierarchies, timelines, and other connections - between entities.
- 4. **Multi-Hop Reasoning** — Queries can traverse relationships, and infer across multiple steps, to connect and derive facts from diverse sources.
- 5. **Joint Reasoning** — Entity Resolution can identify and link references that pertain to the same real-world object, enabling collective analysis.
- 6. **Explainable Relevance** — Graph topology lets us transparently analyze the connections and relationships that determine why certain facts are retrieved as relevant.
- 7. **Personalization** — KGs capture and tailor query results according to user attributes, context, and historical interactions.

In sum, whereas RAG performs matching on disconnected nodes, KGs enable graph traversal search and retrieval of interconnected contextual, search for query-relevant facts, make the ranking process transparent, and encode structured facts, relationships, and context to enable complex, precise, multi-step reasoning. As a result, compared to pure vector search, KGs can improve relevance and explanatory power.

But KG retrieval can be optimized further by applying certain constraints.

## Using Constraints to Optimize Embeddings from Knowledge Graphs

Knowledge Graphs represent entities and relationships that can be vector embedded to enable mathematical operations. These representations and retrieval results can be improved further by adding some **simple but universal constraints**:

- **Non-Negativity Constraints** — Restricting entity embeddings to values between 0 and 1 ensures focus on entities' positive properties only, and thereby improves interpretability.
- **Entailment Constraints** — Encoding expected logic rules like symmetry, inversion, and composition directly as constraints on relation embeddings ensures incorporation of those patterns into the representations.
- **Confidence Modeling** — Soft constraints using slack variables can encode different confidence levels of logic rules depending on evidence.
- **Regularization** — Introduces constraints that impose useful inductive biases to help pattern learning, without making optimization significantly more complex; only a projection step is added.

In addition to **improving interpretability**, **ensuring expected logic rules**, **permitting evidence-based rule confidence levels**, and **improving pattern learning**, constraints can _also_:
- **improve explainability** of the reasoning process; structured constraints make visible the patterns learned by the model; and
- **improve accuracy** of unseen queries; constraints improve generalization by restricting the hypothesis space to compliant representations.

In short, applying some simple constraints can augment Knowledge Graph embeddings to produce more optimized, explainable, and logically compliant representations, with inductive biases that mimic real-world structures and rules, resulting in more accurate and interpretable reasoning, without much additional complexity.

## Choosing a reasoning framework that matches your use case

Knowledge Graphs require reasoning to derive new facts, answer queries, and make predictions. But there are a diverse range of reasoning techniques, whose respective strengths can be combined to fit the requirements of specific use cases.

| Reasoning framework | Method | Pros | Cons |
| ---- | ---- | ---- | ---- |
| **Logical Rules** | Express knowledge as logical axioms and ontologies | Sound and complete reasoning through theorem proving | Limited uncertainty handling |
| **Graph Embeddings** | Embed KG structure for vector space operations | Handle uncertainty | Lack expressivity |
| **Neural Provers** | Differentiable theorem proving modules combined with vector lookups | Adaptive | Opaque reasoning |
| **Rule Learners** | Induce rules by statistical analysis of graph structure and data | Automate rule creation | Uncertain quality |
| **Hybrid Pipeline** | Logical rules encode unambiguous constraints | Embeddings provide vector space operations. Neural provers fuse benefits through joint training. | |
| **Explainable Modeling** | Use case-based, fuzzy, or probabilistic logic to add transparency | Can express degrees uncertainty and confidence in rules | |
| **Iterative Enrichment** | Expand knowledge by materializing inferred facts and learned rules back into the graph | Provides a feedback loop | |

The key to creating a suitable pipeline is identifying the types of reasoning required and mapping them to the right combination of appropriate techniques.

## Preserving Quality Information Flow to the LLM

Retrieving knowledge Graph facts for the LLM introduces information bottlenecks. Careful design can mitigate these bottlenecks by ensuring relevance. Here are some methods for doing that:

- **Chunking** — Splitting content into small chunks improves isolation. But it loses surrounding context, hindering reasoning across chunks.
- **Summarization** — Generating summaries of chunks condenses key details, highlighting their significance. This makes context more concise.
- **Metadata** — Attaching summaries, titles, tags, etc. preserves the source content's context.
- **Query Rewriting** — Rewriting a more detailed version of the original query better tailors retrieval to the LLM’s needs.
- **Relationship Modeling** — KG traversals preserve connections between facts, maintaining context.
- **Information Ordering** — Ordering facts chronologically or by relevance optimizes information structure.
- **Explicit Statements** — Converting implicit knowledge into explicit facts facilitates reasoning.

To preserve quality information flow to the LLM to maximize its reasoning ability, you need to strike a balance between granularity and cohesiveness. KG relationships help contextualize isolated facts. Techniques that optimize the relevance, structure, explicitness, and context of retrieved knowledge help maximize the LLM's reasoning ability.

## Unlocking Reasoning Capabilities by Combining KGs and Embeddings

**Knowledge Graphs** provide structured representations of entities and relationships. KGs empower complex reasoning through graph traversals, and handle multi-hop inferences. **Embeddings** encode information in the vector space for similarity-based operations. Embeddings enable efficient approximate search at scale, and surface latent patterns.

**Combining KGs and embeddings** permits their respective strengths to overcome each other’s weaknesses, and improve reasoning capabilities, in the following ways:

- **Joint Encoding** — Embeddings are generated for both KG entities and KG relationships. This distills statistical patterns in the embeddings.
- **Neural Networks** — Graph neural networks (GNNs) operate on the graph structure and embedded elements through differentiable message passing. This fuses the benefits of both KGs and embeddings.
- **Reasoning Flow** — KG traversals gather structured knowledge. Then, embeddings focus the search and retrieve related content at scale.
- **Explainability** — Explicit KG relationships help make the reasoning process transparent. Embeddings lend interpretability.
- **Iterative Improvement** — Inferred knowledge can expand the KG. GNNs provide continuous representation learning.

While KGs enable structured knowledge representation and reasoning, embeddings provide the pattern recognition capability and scalability of neural networks, augmenting reasoning capabilities in the kinds of language AI that require both statistical learning and symbolic logic.

## Improving Search with Collaborative Filtering

You can use collaborative filtering's ability to leverage connections between entities to enhance search, by taking the following steps:

- 1. **Knowledge Graph** — Construct a KG with nodes representing entities and edges representing relationships.
- 2. **Node Embedding** — Generate an embedding vector for certain key node properties like title, description, and so on.
- 3. **Vector Index** — Build a vector similarity index on the node embeddings.
- 4. **Similarity Search** — For a given search query, find the nodes with the most similar embeddings.
- 5. **Collaborative Adjustment** — Propagate and adjust similarity scores based on node connections, using algorithms like PageRank.
- 6. **Edge Weighting** — Weight adjustments on the basis of edge types, strengths, confidence levels, etc.
- 7. **Score Normalization** — Normalize adjusted scores to preserve relative rankings.
- 8. **Result Reranking** — Reorder initial search results on the basis of adjusted collaborative scores.
- 9. **User Context** — Further adapt search results based on user profile, history, and preferences.

## Fueling Knowledge Graphs with Flywheel Learning

Knowledge Graphs unlock new reasoning capabilities for language models by providing structured real-world knowledge. But KGs aren't perfect. They contain knowledge gaps, and have to update to remain current. Flywheel Learning can help remediate these problems, improving KG quality by continuously analyzing system interactions and ingesting new data.

### Building the Knowledge Graph Flywheel

Building an effective KG flywheel requires:

1. **Instrumentation** — logging all system queries, responses, scores, user actions, and so on, to provide visibility into how the KG is being used.
2. **Analysis** — aggregating, clustering, and analyzing usage data to surface poor responses and issues, and identify patterns indicating knowledge gaps.
3. **Curation** — manually reviewing problematic responses and tracing issues back to missing or incorrect facts in the graph.
4. **Remediation** — directly modifying the graph to add missing facts, improve structure, increase clarity, etc., and fixing the underlying data issues.
5. **Iteration** — continuously looping through the above steps.

Each iteration through the loop further enhances the Knowledge Graph.

Flywheels can also handle high-volume ingestion of streamed live data.

### Streaming Data Ingestion

- You can keep your KG current by continuously ingesting live data sources like news and social media.
- Specialized infrastructure can handle high-volume ingestion into the graph.

### Active Learning

Streaming data pipelines, while continuously updating the KG, will not necessarily fill all knowledge gaps. To handle these, flywheel learning also:
- generates queries to identify and fill critical knowledge gaps; and
- discovers holes in the graph, formulates questions, retrieves missing facts, and adds them.

### The Flywheel Effect

Each loop of the flywheel analyzes current usage patterns and remediates more data issues, incrementally improving the quality of the Knowledge Graph. The flywheel process thus enables the KG and language model to co-evolve and improve in accordance with feedback from real-world system operation. Flywheel learning provides a scaffolding for continuous, automated improvement of the Knowledge Graph, tailoring it to fit the language model's needs. This powers the accuracy, relevance, and adaptability of the language model.

## Conclusion:

In sum, to achieve human-level performance, language AI must be augmented by retrieving external knowledge and reasoning. Where LLMs and RAG struggle with representing the context and relationships between real-world entities, Knowledge Graphs excel. The Knowledge Graph's structured representations permit complex, multi-hop, logical reasoning over interconnected facts. 

Still, while KGs provide previously missing information to language models, KGs can't surface latent patterns the way that language models working on vector embeddings can. Together, KGs and embeddings provide a highly productive blend of knowledge representation, logical reasoning, and statistical learning. And embedding of KGs can be optimized by applying some simple constraints.

Finally, KG's aren't perfect; they have knowledge gaps and need updating. Flywheel Learning can make up for KG knowledge gaps through live system analysis, and handle continuous, large volume data updates to keep the KG current. Flywheel learning thus enables the co-evolution of KGs and LLMs to achieve better reasoning, accuracy, and relevance in language AI applications.

The partnership of KGs and embeddings provides the building blocks moving language AI to true comprehension — conversation agents that understand context and history, recommendation engines that discern subtle preferences, and search systems that synthesize accurate answers by connecting facts. As we continue to improve our solutions to the challenges of constructing high-quality Knowledge Graphs, benchmarking, noise handling, and more, a key role will no doubt be played by hybrid techniques combining symbolic and neural approaches. 
