# Key Access Patterns

The access patterns deployed in Vector Search significantly impact storage, query efficiency, and infrastructure alignment, which are consequential in optimizing your retrieval system for your intended application.

### Static In-Memory Access

The static in-memory access pattern is ideal when working with a relatively small set of vectors, typically fewer than one million, that don't change frequently.

In this pattern, the entire vector set is loaded into your application's memory, enabling quick and efficient retrieval, without external databases or storage. This setup ensures blazing-fast access times, making it perfect for low data scenarios requiring real-time retrieval.

#### Static Access Implementation

You can use libraries like NumPy and Facebook AI Similarity Search (FAISS) to implement static access in Python. These tools allow direct cosine similarity queries on your in-memory vectors.

For smaller vector sets (less than 100,000 vectors), NumPy may be efficient, especially for simple cosine similarity queries. However, if the vector corpus grows substantially or the query requirements become more complex, an "in-process vector database" like LanceDB or Chroma is better.

#### Service Restart Considerations

During service restarts or shutdowns, static in-memory access requires you to reload the entire vector set into memory. This costs time and resources and affects the system's overall performance during the initialization phase.

In sum, static in-memory access is **suitable for compact vector sets that remain static**. It offers fast and efficient real-time access directly from memory. 

But when the vector corpus grows substantially and is updated frequently, you’ll need a different access pattern: dynamic access.

### Dynamic Access 

Dynamic access is particularly useful for managing larger datasets, typically exceeding one million vectors, and scenarios where vectors are subject to frequent updates – for example, sensor readings, user preferences, real-time analytics, and so on. Dynamic access relies on specialized databases and search tools designed for managing and querying high-dimensional vector data; these databases and tools efficiently handle access to evolving vectors and can retrieve them in real-time, or near real-time.

Several types of technologies allow dynamic vector access, each with its own tradeoffs:

1. **Vector-Native Vector Databases** (e.g., [Weaviate](https://weaviate.io/), [Pinecone](https://www.pinecone.io/), [Milvus](https://zilliz.com/what-is-milvus), [Vespa](https://vespa.ai/), [Qdrant](https://qdrant.tech/)): are designed specifically for vector data, and optimized for fast, efficient similarity searches on high-dimensional data. However, they may not be as versatile when it comes to traditional data operations.  

2. **Hybrid Databases** (e.g., [MongoDB](https://www.mongodb.com/), [PostgreSQL with pgvector](https://github.com/pgvector/pgvector/), [Redis with VSS module](https://redis.com/blog/rediscover-redis-for-vector-similarity-search/)): offer a combination of traditional and vector-based operations, providing greater flexibility in managing data. However, hybrid databases may not perform vector searches at the same level or with the same vector-specific features as dedicated vector databases.  

3. **Search Tools** (e.g., [Elasticsearch](https://www.elastic.co/)): are primarily created to handle text search but also provide some Vector Search capabilities. Search tools like Elasticsearch let you perform both text and Vector Search operations, without needing a fully featured database.  

Here's a simplified side-by-side comparison of each database type’s pros and cons:

| Type | Pros | Cons |
| ---------------------------------- | ---------------------------------- | --------------------------------------- |
| *Vector-Native Vector Databases* | High performance for vector search tasks | May not be as versatile for traditional data operations |
| *Hybrid Databases* | Support for both traditional and vector operations | Slightly lower efficiency in handling vector tasks |
| *Search Tools* | Accommodate both text and vector search tasks | May not provide the highest efficiency in vector tasks |

### Batch Access

Batch processing is an optimal approach when working with **large vector sets, typically exceeding one million, that require collective, non-real-time processing**. This pattern is particularly useful for Vector Management tasks such as **model training** or **precomputing nearest neighbors**, which are crucial steps in building Vector Search services.

To be sure your batch processing setup fits your application, you need to keep the following **key considerations** in mind:

**1. Storage Technologies**:
You need scalable, reliable storage to house your vectors during the batch pipeline. Various technologies offer different levels of efficiency:
- **Object Storage (e.g., Amazon S3, Google Cloud Storage, Azure Blob Storage)**: These solutions are cost-effective for storing static data, including substantial vector sets. They scale well and integrate with various cloud-based data processing engines. However, these solutions may not facilitate rapid, real-time communication because they're designed for data at rest.
- **Distributed File Systems (e.g., HDFS, GlusterFS)**: Designed for storing vast amounts of data across multiple servers, distributed file systems offer redundancy and high-throughput access. They seamlessly integrate with big data processing tools like Hadoop or Spark. However, setting up and maintaining them can be complex, especially in cloud environments.

**2. Data Serialization Formats**:
For storing vectors efficiently, you need compact data formats that conserve space and support fast read operations:
 - **Avro and Parquet**: These data serialization formats work well with big data tools like Hadoop or Spark. They effectively compress data, support schema evolution, and integrate smoothly with cloud-based storage. Avro is suitable for write-heavy loads, while Parquet optimizes read-heavy operations.
 - **Compressed NumPy Arrays**: For simpler use cases, you can serialize and compress NumPy arrays that hold vectors. This straightforward approach integrates well with Python-based processing pipelines but is probably not ideal for very large-scale or non-Python environments.

**3. Execution Environment**:
You have two primary options for executing batch-processing tasks:
 - **On-Premise Execution**: If you have to process large volumes of data on your infrastructure, tools like Hadoop and Apache Spark offer complete control over the processing environment but may require complex setup and maintenance.
 - **Cloud Services**: Cloud platforms like Amazon EMR (Elastic MapReduce) provide convenient and scalable batch processing solutions. They manage processing clusters, making scaling resources up or down according to your requirements easier .


In short, which storage technology, data serialization format, and execution environment you choose for your batch processing use case depends on various considerations, including:

- The size of your vector dataset,
- Whether your data is static or dynamic,
- Your workload scalability requirements,
- Whether your dataset is stored across multiple servers,
- Whether you require real-time querying,
- Your integration needs (i.e., big data processing),
- Your desired level of processing control,
- and your available resources and time for set up and maintenance. 

<img src=assets/building_blocks/vector_search/bb3-2.png alt="Choosing a batch processing setup" data-size="100" />

---
## Contributors

- [Daniel Svonava](https://www.linkedin.com/in/svonava/)
- [Paolo Perrone](https://www.linkedin.com/in/paoloperrone/)
- [Robert Turner, editor](https://robertturner.co/copyedit)
