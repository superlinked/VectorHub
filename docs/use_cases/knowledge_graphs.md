# Using Knowledge Graphs + Embeddings to Further Augment RAG Systems

We look at the limitations of not just LLMs but also standard RAG solutions to LLM's knowledge and reasoning gaps, and examine the ways KGs with vector embeddings can fill these gaps and lay the foundation of future progress - through careful incorporation, judicious choice of reasoning techniques, collaborative filtering, and flywheel learning. 


Large Language Models (LLMs) mark a watershed moment in natural language processing, creating new abilities in conversational AI, creative writing, and a broad range of other applications. But they have limitations. While LLMs can generate remarkably fluent and coherent text from nothing more than a short prompt, LLM knowledge is not real-world but rather restricted to patterns learned from training data. In addition, LLMs can't do logical inference or synthesize facts from multiple sources; as queries become more complex and open-ended, LLM responses become contradictory or nonsense.

Retrieval Augmented Generation (RAG) systems have filled some of the LLM gaps by surfacing external source data using semantic similarity search on vector embeddings. Still, because RAG systems don't have access to network structure data - the interconnections between contextual facts, they struggle to achieve true relevance, aggregate facts, and perform chains of reasoning.

Knowledge Graphs (KGs), by encoding real-world entities and their connections, overcome the above deficiencies of pure vector search. KGs enable complex, multi-hop reasoning, across diverse data sources, thereby representing a more comprehensive understanding of the knowledge space. 

Let's take a closer look at how we can combine vector embeddings and knowledge graphs, fusing surface-level semantics, structured knowledge, and logic to unlock new levels of reasoning, accuracy and explanatory ability in LLMs. 

We start by exploring the inherent weaknesses of relying on vector search in isolation, and then show how to combine knowledge graphs and embeddings complementarily, to overcome the limitations of each.


## RAG Vector Search: process and limits

Most RAG systems employ vector search on a document collection to surface relevant context for the LLM. This process has **several key steps**:

- 1. **Text Encoding**: Using embedding models, like BERT, the RAG system encodes and condenses passages of text from the corpus as dense vector representations, capturing semantic meaning.
- 2. **Indexing**: To enable rapid similarity seach, these passage vectors are indexed within a high-dimensional vector space. Popular methods include ANNOY, Faiss, and Pinecone.
- 3. **Query Encoding**: An incoming user query is encoded as a vector representation, using the same embedding model.
- 4. **Similarity Retrieval**: Using distance metrics like cosine similarity, the system runs a search over the indexed passages to find closest neighbors to the query vector.
- 5. **Passage Return**: The system returns the most similar passage vectors, and extracts the corresponding original text to provide context for the LLM.
  
This RAG Vector Search pipeline has **several key limitations**:

- Passage vectors can't represent inferential connections (i.e., context), and therefore often fail to encode the query's full semantic intent.
- Key relevant details embedded in passages (across sentences) are lost in the process of condensing entire passages into single vectors.
- Each passage is matched independently, so facts can't be connected or aggregated.
- The ranking and matching process for determining relevancy remains opaque; we can't see why the system prefers certain passages to others.
- No encoding of relationships, structure, rules, or any other connections between content.

RAG, because it focuses only on semantic similarity, is unable to reason across content, so it fails to really understand both queries and the data RAG retrieves. The more complex the query, the poorer RAG's results become.

## Incorporating Knowledge Graphs

Knowledge graphs, on the other hand, represent information in an interconnected network of entities and relationships, enabling more complex reasoning across content.

How do KGs augment retrieval?

- 1. **Explicit Facts** — KGs preserves key details by capturing facts directly as nodes and edges instead of condensed into opaque vectors.
- 2. **Contextual Details** — KG entities possess rich attributes like descriptions, aliases, and metadata that provide crucial context.
- 3. **Network Structure** — KGs capture real-world relationships - rules, hierarchies, timelines, and other connections - between entities.
- 4. **Multi-Hop Reasoning** — Queries can traverse relationships, and infer across multiple steps, to connect and derive facts from diverse sources.
- 5. **Joint Reasoning** — Entity Resolution can ID and link references that pertain to the same real-world object, enabling collective analysis.
- 6. **Explainable Relevance** — Graph topology lets us transparently analyze the connections and relationships that determine why certain facts are retrieved as relevant.
- 7. **Personalization** — KGs capture and tailor query results according to user attributes, context, and historical interactions.

In sum, whereas RAG performs matching on disconnected nodes, knowledge graphs enable: 
a graph traversal search and retrieval of interconnected contextual, query-relevant facts, make the ranking process transparent, encode structured facts, relationships, and context to enable complex, precise, multi-step reasoning. As a result, compared to pure vector search, KGs can improve relevance and explanatory power.

But KG retrieval can be optimized further by applying certain constraints.

## Optimizing Knowledge Graph Embeddings Using Constraints

Knowledge graphs represent entities and relationships as vector embeddings to enable mathematical operations. These representations and retrieval results can be improved further by adding some simply but universal constraints:

- 1. **Non-Negativity Constraints** — Restricting entity embeddings to values between 0 and 1 ensures focus on entities' positive properties only, and thereby improves interpretability.
- 2. **Entailment Constraints** — Encoding expected logic rules like symmetry, inversion, and composition directly as constraints on relation embeddings ensures incorporation of those patterns into the representations.
- 3. **Confidence Modeling** — Soft constraints using slack variables can encode different confidence levels of logic rules depending on evidence.
- 4. **Regularization** — Introduces constraints that impose useful inductive biases to help pattern learning, without making optimization significantly more complex; only a projection step is added.

In addition to **improving interpretability**, **ensuring expected logic rules**, **permitting evidence-based rule confidence levels**, and **improving pattern learning**, constraints can _also_:
- **improve explainability** of the reasoning process; structured constraints make visible the patterns learned by the model; and
- **improve accuracy** of unseen queries; constraints improve generalization by restricting the hypothesis space to compliant representations.

In short, applying some simple constraints can augment knowledge graph embeddings to produce more optimized, explainable, and logically compliant representations, with inductive biases that mimic real-world structures and rules, resulting in more accurate and interpretable reasoning, without much additional complexity.

## Choosing a reasoning framework that matches your use case

Knowledge Graphs require reasoning to derive new facts, answer queries, and make predictions. But there are a diverse range of reasoning techniques, whose respective strengths can be combined to match specific use cases.

| Reasoning framework | Method | Pros | Cons |
| ---- | ---- | ---- | ---- |
| **Logical Rules** | Express knowledge as logical axioms and ontologies | Sound and complete reasoning through theorem proving | Limited uncertainty handling |
| **Graph Embeddings** | Embed knowledge graph structure for vector space operations | Handle uncertainty | Lack expressivity |
| **Neural Provers** | Differentiable theorem proving modules combined with vector lookups | Adaptive | Opaque reasoning |
| **Rule Learners** | Induce rules by statistical analysis of graph structure and data | Automate rule creation | Uncertain quality |
| **Hybrid Pipeline** | Logical rules encode unambiguous constraints | Embeddings provide vector space operations. Neural provers fuse benefits through joint training. | |
| **Explainable Modeling** | Use case-based, fuzzy, or probabilistic logic to add transparency | Can express degrees uncertainty and confidence in rules | |
| **Iterative Enrichment** | Expand knowledge by materializing inferred facts and learned rules back into the graph | Provides a feedback loop | |
  
The key is identifying the types of reasoning required and mapping them to appropriate techniques. A composable pipeline combining logical formalisms, vector representations, and neural components provides both robustness and explainability.

## Preserving Information Flow to the LLM

Retrieving knowledge graph facts for the LLM introduces information bottlenecks. Careful design preserves relevance:
- **Chunking** — Splitting content into small chunks improves isolation but loses surrounding context. This hinders reasoning across chunks.
- **Summarization** — Generating summaries of chunks provides more concise context. Key details are condensed to highlight significance.
- **Metadata** — Attaching summaries, titles, tags etc as metadata maintains context about the source content.
- **Query Rewriting** — Rewriting the original query into a more detailed version provides retrieval that is better targeted to the LLM’s needs.
- **Relationship Modeling** — Knowledge graph traversals preserve connections between facts, maintaining context.
- **Information Ordering** — Ordering facts chronologically or by relevance optimizes information structure for the LLM.
- **Explicit Statements** — Converting implicit knowledge into explicit facts stated for the LLM makes reasoning easier.

The goal is to optimise the relevance, context, structure and explicitness of the retrieved knowledge to maximize reasoning ability. A balance needs to be struck between granularity and cohesiveness. The knowledge graph relationships aid in contextualizing isolated facts.

## Unlocking Reasoning Capabilities

Knowledge graphs and embeddings each have strengths that overcome the other’s weaknesses when combined:
- **Knowledge Graphs** — Provide structured representation of entities and relationships. Empower complex reasoning through graph traversals. Handle multi-hop inferences.
- **Embeddings** — Encode information in vector space for similarity-based operations. Enable efficient approximate search at scale. Surface latent patterns.
- **Joint Encoding** — Embeddings are generated for entities and relationships in the knowledge graph. This distills statistical patterns.
- **Neural Networks** — Graph neural networks operate on the graph structure and embedded elements through differentiable message passing. This fuses benefits.
- **Reasoning Flow** — Knowledge graph traversals first gather structured knowledge. Then embeddings focus the search and retrieve related content at scale.
- **Explainability** — Explicit knowledge graph relationships provide explainability for the reasoning process. Embeddings lend interpretability.
- **Iterative Improvement** — Inferred knowledge can expand the graph. GNNs provide continuous representation learning.

The partnership enables structured knowledge representation and reasoning augmented by the pattern recognition capability and scalability of neural networks. This is key to advancing language AI requiring both statistical learning and symbolic logic.

## Improving Search with Collaborative Filtering

Collaborative filtering leverages connections between entities to enhance search:

- 1. **Knowledge Graph** — Construct a knowledge graph with nodes representing entities and edges representing relationships.
- 2. **Node Embeddings** — Generate an embedding vector for certain key node properties like title, description etc.
- 3. **Vector Index** — Build a vector similarity index on the node embeddings.
- 4. **Similarity Search** — For a search query, find nodes with most similar embeddings.
- 5. **Collaborative Adjustment** — Based on a node’s connections, propagate and adjust similarity scores using algorithms like PageRank.
- 6. **Edge Weighting** — Weight adjustments based on edge types, strengths, confidence levels etc.
- 7. **Score Normalization** — Normalize adjusted scores to maintain relative rankings.
- 8. **Result Reranking** — Rerank initial results based on adjusted collaborative scores.
- 9. **User Context** — Further adapt based on user profile, history and preferences.

## Fueling Knowledge Graphs with Flywheel Learning

Knowledge graphs unlocked new reasoning capabilities for language models by providing structured world knowledge. But constructing high-quality graphs remains challenging This is where flywheel learning comes in — continuously improving the knowledge graph by analyzing system interactions.

### The Knowledge Graph Flywheel
1. **Instrumentation** — Log all system queries, responses, scores, user actions etc. Provide visibility into how the knowledge graph is being used.
2. **Analysis** — Aggregate usage data to surface poor responses. Cluster and analyze these responses to identify patterns indicating knowledge gaps.
3. **Curation** — Manually review problematic responses and trace issues back to missing or incorrect facts in the graph.
4. **Remediation** — Directly modify the graph to add missing facts, improve structure, increase clarity etc. Fix the underlying data issues.
5. **Iteration** — Continuously loop through the steps above. Each iteration further enhances the knowledge graph.

### Streaming Data Ingestion

- Streaming live data sources like news and social media provides a constant flow of new information to keep the knowledge graph current.
- Specialized infrastructure handles high-volume ingestion into the graph.

### Active Learning

- Use query generation to identify and fill critical knowledge gaps, beyond what streaming provides.
- Discover holes in the graph, formulate questions, retrieve missing facts, and add them.

### The Flywheel Effect

With each loop, the knowledge graph gets incrementally stronger through analysis of usage patterns and remediation of data issues. The improved graph empowers better system performance.

This flywheel process enables the knowledge graph and language model to co-evolve based on feedback from real-world usage. The graph is actively tailored to fit the model’s needs.

In summary, flywheel learning provides a scaffolding for continuous, automated improvement of the knowledge graph through analysis of system interactions. This powers the accuracy, relevance and adaptability of language models relying on the graph.

## Conclusion :

To reach human-level intelligence, language AI needs to incorporate external knowledge and reasoning. This is where knowledge graphs come into the picture. Knowledge graphs provide structured representations of real-world entities and relationships, encoding facts about the world and connections between them. This allows complex logical reasoning across multiple steps by traversing interconnected facts.

However, knowledge graphs have their own limitations like sparsity and lack of uncertainty handling. This is where graph embeddings help — by encoding knowledge graph elements in a vector space, embeddings allow statistical learning from large corpora to surface latent patterns. They also enable efficient similarity-based operations.

Neither knowledge graphs nor embeddings on their own are sufficient for human-like language intelligence. But together, they provide the perfect blend of structured knowledge representation, logical reasoning, and statistical learning. Knowledge graphs overlay symbolic logic and relationships on top of the pattern recognition capability of neural networks.

Techniques like graph neural networks further unify these approaches via differentiable message passing over graph structure and embeddings. The symbiosis enables systems that leverage both statistical learning as well as symbolic logic — combining the strengths of neural networks and structured knowledge representation.

This partnership provides building blocks for the next generation of AI that moves beyond just eloquence to true comprehension — conversational agents that understand context and history, recommendation engines that discern nuanced preferences, search systems that synthesize answers by connecting facts.

Challenges still remain in constructing high-quality knowledge graphs, benchmarking, noise handling, and more. But the future is bright for hybrid techniques spanning symbolic and neural approaches. As knowledge graphs and language models continue advancing, their integration will unlock new frontiers in explainable, intelligent language AI.
