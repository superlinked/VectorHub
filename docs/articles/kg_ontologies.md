# Knowledge Graphs and Organizational Ontologies


## Introduction
Large Language Models (LLMs) harness the power of continuous representations of data in a vector space. They can orchestrate workflows, analyze data, write letters, generate code, and a wide variety of other complex tasks. Heavy investment in LLM development by big tech - Google, Amazon, Apple, Meta, Microsoft - is testament to this power. The survival and well-being of organizations working in AI, and in data more generally, depends on harnessing this power. To do this, AI organizations have to solve two related data issues. In addition to (1) reducing LLM hallucinations and returning relevant, accurate results in their data products, AI organizations need to (2) harness and protect (maintain org boundaries around) the goldmine of data every corp is sitting on.

For all of these tasks, Knowledge Graphs (KGs) and their ontologies (schema layers) can help. 

Indeed, if LLMs reflect one system of human thinking and data representation - continuous and fuzzy, “intuitive,” then KGs represent the other - discrete and reliable, ”abstract”, and knowable. KGs’ reliability and knowability allow them to reduce LLM hallucinations and efficiently return relevant, accurate results, and harness and safeguard the inherent power of organizational data. All through connecting and consolidating (i.e., integrating) the data an organization already has, according to an ontology (i.e., common set of semantics).


## LLMs and KGs - the basics

To understand how KGs can work with LLMs to improve retrieval and harness organizational data, we need to first understand LLMs’ continuous data representation, and KGs’ discrete data representation. 

### LLMs and continuous data representation

Imagine a two-dimensional space (see the diagram below), with fruitiness on the y-axis, and techiness on the x-axis. ‘Banana’ and ‘grape’ score high on fruitiness and low on techiness. Vice versa for ‘Microsoft’ and ‘Google’. ‘Apple,’ on the other hand, is more complex; it’s both a tech giant and a fruit. We need to add dimensions that capture more meaning and context to properly represent it. In fact, each term in an LLM typically occupies a unique position across thousands of dimensions, with each dimension representing a unique aspect of meaning. (Beyond two or three dimensions, a vector space becomes too complex for humans to comprehend.) Each word or word-part’s position in this “latent space” encodes its meaning.

Simplified representation of vector embeddings (on 2 dimensions) and (inset) knowledge graph

An LLM is basically a compression of the web. The LLM reads docs on web, and trying to do next word prediction. It looks back using the transformer model to see how one word relates to another, and creates embedding vectors that are like coordinates.

A vector is like a coordinate - eg. the coordinate for “king” will “zoom” me to the coordinate position for “king”. But I can also take the embedding vector for king, take away the embedding vector for “man”, and add the embedding vector for “woman”, and this new coordinate will now zoom me to the position of queen. This is a very powerful way of representing the terms. But this is a continuous space; the coordinate (vector) will take me roughly (not precisely) to the relevant area of the vector space.

So how does the representation of data in a Knowledge Graph (KG) differ from the representation of data in an LLM?

### KGs and discrete knowledge representation

LLMs represent data (mostly text) continuously. Vector embeddings in LLMs are continuous, fuzzy.. “One fact or concept flows into the next in a continuous multi-dimensional space.” LLMs are powerful, flexible, but not always reliable, and ultimately incomprehensible. 
KGs, on the other hand, are precise and discrete. KGs make each data item into a distinct node and connections between them into discrete edges. They are less flexible than LLMs but they are reliable and explicable. 


KGs are built for the integration of factual data. In addition to text (the domain of LLMs), KGs can easily capture data contained in tables, trees (e.g., json, xml), and images.


### Ontology / schema layer

The key to a KG’s power is its ontology, or schema layer. A KG’s ontology structures the KG and makes it intelligible. The ontology is a semantic data model defining the types of things in a domain. Ontologies include classes, relationships, and attributes. KGs are created by applying an ontology (or schema layer) to a set of data points. This schema layer permits the KG to represent linked entities in a domain. Schemas allow searches to use the classes, relationships, and attributes in an ontology (schema layer) to inform searches with precise definitions. 

Over 44% of the web takes advantage of the power of KG schemas by tagging data items with JSON-LDs (Javascript Object Notation for Linked Data). Every data item tagged in this way can be connected by their formal relationship as set out in the schema’s classes, relationships, and attributes.

For example, if I want to indicate that Jane Doe and John Smith are colleagues, I can do it using their JSON-LDs, and obtain a distributed graph. Simplified, such a graph would look like this: 
Jane Doe <--- Colleagues ---> John Smith


Each island of JSON-LD points back to schema.org (maintained by the JSON-LD Working Group), with contributions from a broad community. Schema.org has a common set of schemas for things people mostly search for on web - e.g., products, flights, bookings. For example, if I search for a specific recipe on google, I get very specific results based on the google knowledge graph constructed from islands of JSON-LDs.


## Using KGs for data org survival
So… how should AI organizations take advantage of the power of graphs to survive? 
1) By using KGs in your organization’s data products to insert ontological context into LLM prompts - thereby reducing hallucination and returning relevant, accurate results. And,
2) By unifying the organization’s data into a comprehensive organizational KG that harnesses and protects the organization’s goldmine of data. Let’s take a closer look at each of these in turn.
1) Using a KG to insert ontological context into the prompt. 
A given question comes in, that question can be referred back to classes in the ontology or facts in the KG (e,g., on my repo), which can be injected into the context of the prompt of the LLM, influencing the way the LLM responds. This is Retrieval Augmented Generation (RAG). In its simplest form, RAG employs the following basic, step-by-step approach to connecting a KG to an LLM:

In almost every KG, we have a short RDFS (Resource Description Framework Schema) label description and a longer DC (Dublic Core) description of each node. The RDFS is a human readable description of a node’s URI. The DC metadata elements include title, creator, subject, description, etc., and are often represented in the URI itself. We can do a direct query and pull out every single description, then do a call out to openAI to get an embedding vector for each of those descriptions. 
	
Extract relevant nodes
```python
# Begin by pulling all the nodes that you wish to index from your Knowledge Graph, including their descriptions:

rows = rdflib_graph.query('SELECT * WHERE {?uri dc:description ?desc}')
```
Generate embedding vectors
```python
# Employ your large language model to create an embedding vector for the description of each node:

node_embedding = openai.Embedding.create(input = row.desc, model=model) ['data'][0]['embedding']
```

Build a vector store
```
# Store the generated embedding vectors in a dedicated vector store:

index = faiss.IndexFlatL2(len(embedding))
index.add(embedding)
```
Query with natural language
```
# When a user poses a question in natural language, convert the query into an embedding vector using the same language model. Then, leverage the vector store to find the nodes with the lowest cosine similarity to the query vector:


question_embedding = openai.Embedding.create(input = question, model=model) ['data'][0]['embedding']
d, i = index.search(question_embedding, 100)
```

Semantic post-processing
```
# To further enhance the user experience, apply post-processing techniques to the retrieved related nodes. This step refines the results and presents information in a way that best provides users with actionable insights.
```

For example, I pass the description text for “Jennifer Aniston” into my LLM, and now can store the fact that this discrete node (representing “Jennifer Aniston”) in my KG relates to the Jennifer Aniston textual description in embedding vector space (in the LLM). After this, when a user comes and does a query for “Jennifer Aniston”, I can turn the query into an embedding vector, locate the closest embedding vectors in the continuous vector space, and then find the related node within the discrete KG, and return a relevant result.



Because I can control my KG, I can limit my query results to exclude / reduce hallucination, and improve result precision and accuracy.

In addition to 1) using my KG to insert context into my prompts (above), I can also harness and protect (maintain org boundaries around) the goldmine of data my corp is sitting on by 2) creating a unified organizational KG using schema.org, and connecting it to my LLM.


2) Creating a unified org KG, and connecting it to LLM
KGs can harness and protect (maintain organizational boundaries around) the goldmine of data every corporation is sitting on. Using a cellular analogy, an organization should erect a cellular membrane - i.e., a data boundary, a firewall, that contains and selectively (safely) exposes its proprietary data products. The cellular membrane contains the organization’s semantic data mesh.


The organizational semantic data mesh should be powered by organizational KGs, allowing the organization to use LLMs to realize and capture the free energy bound up in the chaotic jumble of its databases. An organization’s KGs (structured by the organization’s ontology) can act as a kind of Markov Blanket - the minimal subset of variables containing all the information needed to infer/predict a target random variable - anything that’s useful for organizational purposes.

With the blanket / membrane in place, an AI organization can safely expose - i.e., distribute - parts of its internal network-shaped data (a large connected graph) as fragments, at the membrane surface. With a well-defined ontology in place, an organization can use its KG to securely cooperate with other organizations, and maintain private networks with informational boundaries.

### Creating a unified organizational KG - schema.org

Any organization can use schema.org to create a unified KG with their own data resources, thereby integrating them in a comprehensible, defined way - i.e., in an ontology (which can be amended on demand) that represents the connections between data items. Schema.org is open source and can be downloaded from github, and you can stand up that server behind a firewall within your organization.

You can take schema.org’s base types as your starting model, but can develop your own particular semantics related to your own particular business. Eg. for a bank, their will be “trade” and “risk”, if you’re a railway, “tracks” and “trains”, if you’re a hospital, “patients”, “beds”, “medicines”. If we give a URI (functionally, a URL on the web) to each and every item in the graph, it becomes globally identifiable. Each data node is a data item, a unique network address, and an artificial neuron in a neural network, linked together by their edges (defined in the JSON-LD).

Embed the definitions salient to your own particular business, sticking as close as possible to the actual, working semantics of the real people in your business, into your own version of schema.org. In a large organization there’ll be between 5000-100000 separate apps and databases, each with 1000s of different tables in it, each table with 100s of different columns, in sum, a vast complex. By publishing the data in each application or database in JSON-LD, referencing the well-defined semantics in your schema.org, you have your organizational KG.

Your organizational KG should not be treated as a new database. Rather, for each use case, you need only download the chunk of the graph (pre-connected, pre-integrated) that you need. 

In other words, “within your organization, let a thousand knowledge graphs bloom.”

### Data catalog

At the level of the whole org, you want a representation of the schema and linked data - the data catalog - an inventory of your data assets, to facilitate data management and use (data discovery, collaboration and knowledge sharing, quality and metadata management, etc.). Each semantic data product/department within the organization should publish a catalog of the data it has available. What is it that your department’s product has to offer the business? Which datasets should your particular organizational product make available to everyone else? Once every department/product publishes their own data catalog, you should collect them together via a schema, connecting all of them. With this organizational data catalog - which makes all your data accessible to you - in place, your semantic layer is complete.

Go from this:


…to this:


Through the semantic layer, I can use business concepts to access data which is in a graph format (working memory graph).

Now that you have a complete organizational KG, you can take full advantage of value of your organization’s internal data by connecting it to an instance of an LLM. Connecting your KG to an LLM can: 1 enrich your own data, providing more comprehensive information, 2 achieve semantic interoperability - your data can be understood not just by your org but across different systems that recognize the URIs, 3 provide more complete views of entities represented by the well-known URIs 4 improve the ability to query your data, using the URIs as hooks to pull in relevant data from various sources 5 maintain consistency and standardization in your data.

### Connecting your KG to LLM graph data

In addition to using schema.org to create a unified organization KG, you can connect your schema.org KG to LLM data provided as a JSON-LD graph. Because the URIs that JSON-LDs rely on are well-known - recognizable by multiple systems - they can serve as a common language that different data sources can understand and use to communicate. Once you have the LLM’s JSON-LD graph, you can map from the graph’s well-known URIs to matching items in your own internal data. 

Diagrammatically, we can depict this as a working memory graph. (For example, wikidata is a large, online and open source KG that an LLM can read and give to us as a JSON-LD.)


CONCLUSION 
In sum, KGs present an organizational opportunity to companies who work with LLMs. They provide a way of ameliorating LLM hallucinations, but also a method for uniying the trove of data your organization houses but may not yet reap the full potential benefits of. By building a strong ontology (schema layer) in your KG, you indicate the classes, relationships, and attributes that are priorities for you, You can then use your KG (its discrete representation of your prioirites) to more judiciously harness the power of an LLM’s continuous vector space to improve, protect, and develop your data and data products.