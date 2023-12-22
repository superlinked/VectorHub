Embeddings + Knowledge Graphs: The Ultimate Tools for RAG Systems

The advent of large language models (LLMs) , trained on vast amounts of text data, has been one of the most significant breakthroughs in natural language processing. The ability of these models to generate remarkably fluent and coherent text with just a short prompt has opened up new possibilities for conversational AI, creative writing, and a wide array of other applications.
However, despite their eloquence, LLMs have some key limitations. Their knowledge is restricted to patterns discerned from the training data, which means they lack true understanding of the world.
Their reasoning ability is also limited — they cannot perform logical inferences or synthesize facts from multiple sources. As we ask more complex, open-ended questions, the responses start becoming nonsensical or contradictory.
To address these gaps, there has been growing interest in retrieval-augmented generation (RAG) systems. The key idea is to retrieve relevant knowledge from external sources to provide context for the LLM to make more informed responses.
Most existing systems retrieve passages using semantic similarity of vector embeddings. However, this approach has its own drawbacks like lack of true relevance, inability to aggregate facts, and no chain of reasoning.
This is where knowledge graphs come into the picture. Knowledge graphs are structured representations of real-world entities and relationships. They overcome the deficiencies of pure vector search by encoding interconnections between contextual facts. Traversing knowledge graphs enables complex multi-hop reasoning across diverse information sources.
In this article, we dive deep on how combining vector embeddings and knowledge graphs can unlock new levels of reasoning, accuracy and explanatory ability in LLMs. This partnership provides the perfect blend of surface-level semantics along with structured knowledge and logic.
Like our own minds, LLMs need both statistical learning as well as symbolic representations.
We first explore the inherent weaknesses of relying solely on vector search in isolation.
We then elucidate how knowledge graphs and embeddings can complement each other, with neither technique alone being sufficient.
The Limits of Raw Vector Search

Vector Search Is Not All You Need
Introduction

Most RAG systems rely on a vector search process over passages from a document collection to find relevant context for the LLM. This process has several key steps:
Text Encoding: The system encodes passages of text from the corpus into vector representations using embedding models like BERT. Each passage gets condensed into a dense vector capturing semantic meaning.
Indexing: These passage vectors get indexed in a high-dimensional vector space to enable fast similarity search. Popular methods include ANNOY, Faiss, and Pinecone.
Query Encoding: When a user query comes in, it also gets encoded into a vector representation using the same embedding model.
Similarity Retrieval: A similarity search is run over the indexed passages to find those closest to the query vector based on distance metrics like cosine similarity.
Passage Return: The most similar passage vectors are returned, and the original text is extracted to provide context for the LLM.
This pipeline has several key limitations:
The passage vectors may not fully capture the semantic intent of the query. Important context ends up overlooked because embeddings fail to represent certain inferential connections.
Nuances get lost in condensing entire passages to single vectors. Key relevant details embedded across sentences get obscured.
Matching is done independently for each passage. There is no joint analysis across different passages to connect facts and derive answers requiring aggregation.
The ranking and matching process is opaque, providing no transparency into why certain passages are deemed more relevant.
Only semantic similarity is encoded, with no representations of relationships, structure, rules and other diverse connections between content.
The singular focus on semantic vector similarity results in retrieval that lacks true comprehension. As queries get more complex, these limitations become increasingly apparent in the inability to reason across retrieved content.
Incorporating Knowledge Graphs
Knowledge graphs represent information in an interconnected network of entities and relationships, enabling more complex reasoning across content.
Here’s how they augment retrieval:
Explicit Facts — Facts are directly captured as nodes and edges instead of condensed into opaque vectors. This preserves key details.
Contextual Details — Entities contain rich attributes like descriptions, aliases, and metadata that provide crucial context.
Network Structure — Relationships model real-world connections between entities, capturing rules, hierarchies, timelines, etc.
Multi-Hop Reasoning — Queries can traverse relationships to connect facts from diverse sources. Answers requiring inference across multiple steps can be derived.
Joint Reasoning — Entity Resolution links references to the same real-world object, allowing collective analysis.
Explainable Relevance — Graph topology provides transparency into why certain facts are relevant based on their connections.
Personalization — User attributes, context, and historical interactions are captured to tailor results.
Rather than isolated matching, knowledge graphs enable a graph traversal process to gather interconnected contextual facts relevant to the query. Explainable rankings are possible based on topology. The rich knowledge representation empowers more complex reasoning.
Knowledge graphs augment retrieval by encoding structured facts, relationships and context to enable precise, multi-step reasoning. This provides greater relevance and explanatory power compared to pure vector search.
Incorporating Knowledge Graphs with Embeddings & Constraints

Improving Knowledge Graph Embedding Using Simple Constraints
Embedding knowledge graphs (KGs) into continuous vector spaces is a focus of current research. Early works performed…
arxiv.org

Knowledge graphs represent entities and relationships as vector embeddings to enable mathematical operations. Additional constraints can make the representations more optimal:
Non-Negativity Constraints — Restricting entity embeddings to positive values between 0 and 1 induces sparsity. This models only their positive properties explicitly and improves interpretability.
Entailment Constraints — Encoding expected logic rules like symmetry, inversion, composition directly as constraints on relation embeddings enforces those patterns.
Confidence Modeling — Soft constraints with slack variables can encode varying confidence levels of logic rules based on evidence.
Regularization — Constraints impose useful inductive biases without making optimization significantly more complex. Only a projection step is added.
Explainability — The structured constraints provide transparency into the patterns learned by the model. This explains the reasoning process.
Accuracy — Constraints improve generalization by reducing the hypothesis space to compliant representations. This improves accuracy on unseen queries.
Adding simple but universal constraints augments knowledge graph embeddings to produce more optimized, explainable, and logically compliant representations. The embeddings gain inductive biases that mimic real-world structures and rules. This results in more accurate and interpretable reasoning without much additional complexity.
Integrating Diverse Reasoning Frameworks

Papers with Code - Graph Agent: Explicit Reasoning Agent for Graphs

paperswithcode.com

Knowledge graphs require reasoning to derive new facts, answer queries, and make predictions. Different techniques have complementary strengths:
Logical Rules — Express knowledge as logical axioms and ontologies. Sound and complete reasoning through theorem proving. Limited uncertainty handling.
Graph Embeddings — Embed knowledge graph structure for vector space operations. Handle uncertainty but lack expressivity.
Neural Provers — Differentiable theorem proving modules combined with vector lookups. Adaptive but opaque reasoning.
Rule Learners — Induce rules by statistical analysis of graph structure and data. Automates rule creation but uncertain quality.
Hybrid Pipeline — Logical rules encode unambiguous constraints. Embeddings provide vector space operations. Neural provers fuse benefits through joint training.
Explainable Modeling — Use case-based, fuzzy, or probabilistic logic to add transparency. Express uncertainty and confidence in rules.
Iterative Enrichment — Expand knowledge by materializing inferred facts and learned rules back into the graph. Provides a feedback loop.
The key is identifying the types of reasoning required and mapping them to appropriate techniques. A composable pipeline combining logical formalisms, vector representations, and neural components provides both robustness and explainability.
Preserving Information Flow to the LLM
Retrieving knowledge graph facts for the LLM introduces information bottlenecks. Careful design preserves relevance:
Chunking — Splitting content into small chunks improves isolation but loses surrounding context. This hinders reasoning across chunks.
Summarization — Generating summaries of chunks provides more concise context. Key details are condensed to highlight significance.
Metadata — Attaching summaries, titles, tags etc as metadata maintains context about the source content.
Query Rewriting — Rewriting the original query into a more detailed version provides retrieval that is better targeted to the LLM’s needs.
Relationship Modeling — Knowledge graph traversals preserve connections between facts, maintaining context.
Information Ordering — Ordering facts chronologically or by relevance optimizes information structure for the LLM.
Explicit Statements — Converting implicit knowledge into explicit facts stated for the LLM makes reasoning easier.
The goal is optimizing relevance, context, structure and explicitness of the retrieved knowledge to maximize reasoning ability. A balance needs to be struck between granularity and cohesiveness. The knowledge graph relationships aid in contextualizing isolated facts.
Unlocking Reasoning Capabilities
Knowledge graphs and embeddings each have strengths that overcome the other’s weaknesses when combined:
Knowledge Graphs — Provide structured representation of entities and relationships. Empower complex reasoning through graph traversals. Handle multi-hop inferences.
Embeddings — Encode information in vector space for similarity-based operations. Enable efficient approximate search at scale. Surface latent patterns.
Joint Encoding — Embeddings are generated for entities and relationships in the knowledge graph. This distills statistical patterns.
Neural Networks — Graph neural networks operate on the graph structure and embedded elements through differentiable message passing. This fuses benefits.
Reasoning Flow — Knowledge graph traversals first gather structured knowledge. Then embeddings focus the search and retrieve related content at scale.
Explainability — Explicit knowledge graph relationships provide explainability for the reasoning process. Embeddings lend interpretability.
Iterative Improvement — Inferred knowledge can expand the graph. GNNs provide continuous representation learning.
The partnership enables structured knowledge representation and reasoning augmented by the pattern recognition capability and scalability of neural networks. This is key to advancing language AI requiring both statistical learning and symbolic logic.
Improving Search with Collaborative Filtering
Collaborative filtering leverages connections between entities to enhance search:
Knowledge Graph — Construct a knowledge graph with nodes representing entities and edges representing relationships.
Node Embeddings — Generate an embedding vector for certain key node properties like title, description etc.
Vector Index — Build a vector similarity index on the node embeddings.
Similarity Search — For a search query, find nodes with most similar embeddings.
Collaborative Adjustment — Based on a node’s connections, propagate and adjust similarity scores using algorithms like PageRank.
Edge Weighting — Weight adjustments based on edge types, strengths, confidence levels etc.
Score Normalization — Normalize adjusted scores to maintain relative rankings.
Result Reranking — Rerank initial results based on adjusted collaborative scores.
User Context — Further adapt based on user profile, history and preferences.
Fueling Knowledge Graphs with Flywheel Learning

Fueling the RAG Engine : The Data Flywheel
Building a high-performing retrieval-augmented generation (RAG) system that continuously improves requires implementing…
ai.plainenglish.io

Knowledge graphs unlocked new reasoning capabilities for language models by providing structured world knowledge. But constructing high-quality graphs remains challenging. This is where flywheel learning comes in — continuously improving the knowledge graph by analyzing system interactions.
The Knowledge Graph Flywheel
Instrumentation — Log all system queries, responses, scores, user actions etc. Provide visibility into how the knowledge graph is being used.
Analysis — Aggregate usage data to surface poor responses. Cluster and analyze these responses to identify patterns indicating knowledge gaps.
Curation — Manually review problematic responses and trace issues back to missing or incorrect facts in the graph.
Remediation — Directly modify the graph to add missing facts, improve structure, increase clarity etc. Fix the underlying data issues.
Iteration — Continuously loop through the steps above. Each iteration further enhances the knowledge graph.
Streaming Data Ingestion
Streaming live data sources like news and social media provides a constant flow of new information to keep the knowledge graph current.
Specialized infrastructure handles high volume ingestion into the graph.
Active Learning
Use query generation to identify and fill critical knowledge gaps, beyond what streaming provides.
Discover holes in the graph, formulate questions, retrieve missing facts, and add them.
The Flywheel Effect
With each loop, the knowledge graph gets incrementally stronger through analysis of usage patterns and remediation of data issues. The improved graph empowers better system performance.
This flywheel process enables the knowledge graph and language model to co-evolve based on feedback from real-world usage. The graph is actively tailored to fit the model’s needs.
In summary, flywheel learning provides a scaffolding for continuous, automated improvement of the knowledge graph through analysis of system interactions. This powers the accuracy, relevance and adaptability of language models relying on the graph.
Conclusion :
To reach human-level intelligence, language AI needs to incorporate external knowledge and reasoning. This is where knowledge graphs come into the picture. Knowledge graphs provide structured representations of real-world entities and relationships, encoding facts about the world and connections between them. This allows complex logical reasoning across multiple steps by traversing interconnected facts.
However, knowledge graphs have their own limitations like sparsity and lack of uncertainty handling. This is where graph embeddings help — by encoding knowledge graph elements in a vector space, embeddings allow statistical learning from large corpora to surface latent patterns. They also enable efficient similarity-based operations.
Neither knowledge graphs nor embeddings on their own are sufficient for human-like language intelligence. But together, they provide the perfect blend of structured knowledge representation, logical reasoning, and statistical learning. Knowledge graphs overlay symbolic logic and relationships on top of the pattern recognition capability of neural networks.
Techniques like graph neural networks further unify these approaches via differentiable message passing over graph structure and embeddings. The symbiosis enables systems that leverage both statistical learning as well as symbolic logic — combining the strengths of neural networks and structured knowledge representation.
This partnership provides building blocks for the next generation of AI that moves beyond just eloquence to true comprehension — conversational agents that understand context and history, recommendation engines that discern nuanced preferences, search systems that synthesize answers by connecting facts.
Challenges still remain in constructing high-quality knowledge graphs, benchmarking, noise handling, and more. But the future is bright for hybrid techniques spanning symbolic and neural approaches. As knowledge graphs and language models continue advancing, their integration will unlock new frontiers in explainable, intelligent language AI.
