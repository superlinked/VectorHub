<!-- TODO: Replace this text with a summary of article for SEO -->

# Vector Search

<!-- TODO: Cover image: 
1. You can create your own cover image and put it in the correct asset directory,
2. or you can give an explanation on how it should be and we will help you create one. Please tag arunesh@superlinked.com or @AruneshSingh (GitHub) in this case. -->

## Introduction
You've unlocked a wealth of potential after transforming raw data into vector embeddings. These vectors encapsulate the essence of your data, but their full value isn't realized until you apply them in your Vector Search & Management.

Vector Search & Management is the bridge between the latent, abstract mathematical representations of your data and their real-world applications. Your Vector Management stores, structures, and prepares your vector data for various machine learning tasks. Using Vector Search, you can perform efficient and relevant data retrieval from extensive data repositories. 

Vector Search & Management empower critical objectives:

**Quality Evaluation**: Application success depends on your vector quality. You can use Vector Search to thoroughly assess and fine-tune the performance of embeddings of all types, including word, image, document, product, music & audio, face, tag, and features (in ML).

**Model Training**: Vector representations are essential for training models in multiple domains, including transfer learning (e.g., language translation, image style transfer), reinforcement learning (e.g., autonomous driving, robotic control), content-based recommendation systems (e.g., e-commerce, music), anomaly detection (e.g., manufacturing quality control, network anomaly detection), active learning (e.g., image classification, text sentiment analysis), and embedding space analysis (e.g., document clustering, user behavior profiling).

**Real-Time Retrieval**: In live systems, Vector Search is the foundation for real-time retrieval, powering functions like visual similarity search in online image search engines, streaming platforms as well as video, music, and news suggestions on news sites, content tagging and classification on, e.g., social media sites, online reverse image search, and voice search in voice assistants and search engines.

How you use your Vector Search & Management to achieve these objectives depends on your use case requirements and constraints. Your specific implementation depends on how your use case fits a few Core Parameters, how you tailor Nearest Neighbor Search, and which Key Access Patterns you utilize. 

Let’s look at each of these below.

## Core Parameters in Vector Search & Management

To understand Vector Search & Management and their role in retrieval systems, we need to explore the key requirements that shape their implementation. These requirements vary with the specific use cases and tasks the retrieval system aims to accomplish.

Let's break down the retrieval system’s key defining parameters:

**Update Frequency**:
Vectors can remain static over time or undergo frequent updates. In many retrieval systems, vectors are used to represent evolving data that requires regular refreshing. For example, on e-commerce platforms, product recommendations must adapt as new products are added, and customer preferences change. 

In contrast, in a static library of scientific articles, vector updates happen less frequently. The frequency at which the vectors change crucially affects how they are managed and accessed.

**Access Patterns**:
Access patterns – how vectors are queried and retrieved within the retrieval system – can vary widely, ranging from real-time, on-the-fly, queries for nearest neighbors of single vectors, to batch processing and integration with other data. Emergency response systems (e.g., 911 dispatch), for instance, rely on quick access real-time similarity searches to find the closest available emergency responders to a reported incident location. Here, low-latency access can mean the difference between life and death. 

Whereas, a video processing pipeline handling large sets of video frames for analysis will mainly focus on batch processing. Access patterns play a critical role in determining the efficiency of vector retrieval, and therefore, whether real-time or batch processing is preferred.
 
**Priorities**:
The priorities of your particular application determine what matters most: minimizing latency, maximizing throughput, or ensuring a high level of accuracy. For example, low-latency access to real-time data is paramount for making split-second decisions in a financial trading system. In contrast, a data analytics platform might prioritize high throughput to process large volumes of data quickly.

What you choose to prioritize affects your retrieval system's architecture and the trade-offs made during implementation.

An efficient and effective Vector Search & Management approach must carefully consider and quantify the retrieval system’s update frequency, access patterns, and priorities to meet the requirements of the intended use case, whether it’s providing instant e-commerce product recommendations, facilitating real-time decision-making, conducting in-depth data analysis, or something else entirely.

![Three core considerations of Vector Search](assets/building_blocks/vector_search/bb3-1.png)

The key takeaway from the visual above is these three corners interact to determine the design of our system. Depending on our project, we may prioritize one corner over the others. For example, if we're building a recommendation system for an online store, we would emphasize "real-time" interactions and "speed" to offer customers instant, personalized recommendations.


## Nearest Neighbor Search Algorithms:

Scanning to calculate the similarity between vectors quickly is at Vector Search's heart. Vector similarity scores encoded by your embedding model/s store valuable feature or characteristic information about your data that can be used in various applications (e.g., content recommendation, clustering, data analysis). 

### Full Scan

Let’s say we have a dataset of 1 million 1000-dimensional vectors and want to find similar vectors for a given query quickly. Naively, this would require 1 billion operations per query – a full scan nearest neighbor search. Full scan is an exhaustive, brute force approach to nearest neighbor search. It sequentially checks every record or data block of the dataset. 

A full scan is simple, easy to implement, suitable when your dataset has less than 1M vectors, and not constantly changing.

But as your dataset grows past 1M vectors, full scan takes more and more time and becomes more resource-intensive. We can significantly expedite our scan using an approximate nearest neighbors (ANN) search.

### ANN Algorithms

Approximate Nearest Neighbors (ANN) is a class of algorithms that expedite vector similarity search through approximations, avoiding the high computational cost of exhaustive brute force (full scan) search.

ANN algorithms like locality-sensitive hashing (LSH) and hierarchical navigable small world (HNSW) can provide a tunable balance between precision and speed. HNSW and other similar methods permit very high recall, approaching brute force levels but at faster speeds.

ANN provides a crucial advantage over brute force search for large, high-dimensional datasets. [ANN benchmarks’ “Benchmarking Results”](https://ann-benchmarks.com/) demonstrate that brute force algorithms provide the highest precision but at a tradeoff cost of fewer QPS (queries per second). Hybrid approaches combining ANN and exact search (i.e., full scan) can provide both speed and accuracy.

To select an ANN method appropriate to your application, you need to evaluate metrics like recall and latency in relation to the dataset scale, dimensionality, and data distribution requirements of your use case. 


#### Code example: full scan vs. ANN

Let’s look at some example code, demonstrating linear (full) scan and ANN search.

```python
import numpy as np
import faiss
```

Create dataset of 1 million 1000-dim vectors
```python
num_vectors = 1000000
vector_dim = 1000
dataset = np.random.rand(num_vectors, vector_dim).astype('float32')
```

Define query vector
```python
query_vector = np.random.rand(vector_dim).astype('float32')
```

Create FAISS index
```python
index = faiss.IndexFlatL2(vector_dim)
```
Add vectors to index
```python
index.add(dataset)
```

Linear scan search
```python
start = time.time()
distances, indices = index.search(query_vector.reshape(1, vector_dim), 1)
print("Linear scan time: ", time.time() - start)
```

Switch to IVFFlat index for approximate search

```python
index = faiss.index_factory(vector_dim, "IVF1024,Flat")
index.train(dataset)
index.add(dataset)

start = time.time()
distances, indices = index.search(query_vector.reshape(1, vector_dim), 1) 
print("ANN time: ", time.time() - start)
```

This code demonstrates how an ANN algorithm, like IVFFlat from the FAISS library, indexes the vectors, allowing quick narrowing of the search space. This lets you significantly speed up your scan, compared to a linear scan, especially as your data set becomes larger.

Different ANN implementations provide different optimization tradeoffs. Choosing the right ANN method requires benchmarking options like HNSW, IVF, and LSH on your dataset. 

### Quantization

Search speed is not the only challenge when handling datasets with more than 1 million vectors. Once a dataset is too large to fit into the RAM of a single machine, you may be forced to shard the index to multiple machines. This is costly and increases system complexity. Fortunately, you can use quantization to reduce the index size, without substantially reducing retrieval quality.

Quantization reduces the memory required to store vectors by compressing the data but preserving relevant information. There are three types of vector quantization: scalar, binary, and product.

**Scalar** quantization reduces the dimensionality of your data by representing each vector component with fewer bits, reducing the amount of memory required for vector storage and speeding up the search process. Scalar quantization achieves a good balance of compression, accuracy, and speed, and is used widely.

**Binary** quantization represents each vector component as binary codes. Binary is the fastest quantization method. But it’s only efficient for high dimensional vectors and a centered distribution of vector components. As a result, binary quantization should only be used with tested models.

**Product** quantization splits the vector dimensions into smaller subvectors, each independently quantized into an index using the subvector’s set of codewords, representing data point clusters, and then combined into a single index representing the original high-dimensional vector. Product quantization provides better compression (minimal memory footprint) but is slower and less accurate than scalar quantization.

ANN search can be combined with quantization to enable running high-performance [Vector Search] pipelines on huge datasets.

## Key Access Patterns

The access patterns deployed in Vector Search significantly impact storage, query efficiency, and infrastructure alignment, which are consequential in optimizing your retrieval system for your intended application.

### Static In-Memory Access

The static in-memory access pattern is ideal when working with a relatively small set of vectors, typically fewer than one million, that don't change frequently.

In this pattern, the entire vector set is loaded into your application's memory, enabling quick and efficient retrieval, without external databases or storage. This setup ensures blazing-fast access times, making it perfect for low data scenarios requiring real-time retrieval.

#### Static Access Implementation

You can use libraries like NumPy and Facebook AI Similarity Search (FAISS) to implement static access in Python. These tools allow direct cosine similarity queries on your in-memory vectors.

For smaller vector sets (less than 100,000 vectors), NumPy may be efficient, especially for simple cosine similarity queries. However, if the vector corpus grows substantially or the query requirements become more complex, an "in-process vector database" like Lancedb or Chroma is better.

#### Service Restart Considerations

During service restarts or shutdowns, static in-memory access requires you to reload the entire vector set into memory. This costs time and resources and affects the system's overall performance during the initialization phase.

In sum, static in-memory access is suitable for compact vector sets that remain static. It offers fast and efficient real-time access directly from memory. 

But when the vector corpus grows substantially and is updated frequently, you’ll need a different access pattern: dynamic access.

### Dynamic Access 

Dynamic access is particularly useful for managing larger datasets, typically exceeding one million vectors, and scenarios where vectors are subject to frequent updates – for example, sensor readings, user preferences, real-time analytics, and so on. Dynamic access relies on 
specialized databases and search tools designed for managing and querying high-dimensional vector data; these databases and tools efficiently handle access to evolving vectors and can retrieve them in real-time, or near real-time.

Several types of technologies allow dynamic vector access, each with its own tradeoffs:

1. **Vector-Native Vector Databases** (e.g., **[Weaviate](https://weaviate.io/), [Pinecone](https://www.pinecone.io/), [Milvus](https://zilliz.com/what-is-milvus), [Vespa](https://vespa.ai/)): are designed specifically for vector data, and optimized for fast, efficient similarity searches on high-dimensional data. However, they may not be as versatile when it comes to traditional data operations.

2. **Hybrid Databases** (e.g., [MongoDB](https://www.mongodb.com/), [PostgreSQL with pgvector](https://github.com/pgvector/pgvector/), [Redis with VSS module](https://redis.com/blog/rediscover-redis-for-vector-similarity-search/)): offer a combination of traditional and vector-based operations, providing greater flexibility in managing data. However hybrid databases may not perform vector searches at the same level or with the same vector-specific features as dedicated vector databases.

3. **Search Tools** (e.g., [Elasticsearch](https://www.elastic.co/))**: are primarily created to handle text search but also provide some Vector Search capabilities. Search tools like Elasticsearch are useful when you need to perform both text and Vector Search operations, without needing a fully featured database.

Here's a simplified side-by-side comparison of each database type’s pros and cons:

| Type | Pros | Cons |
| ---------------------------------- | ---------------------------------- | --------------------------------------- |
| *Vector-Native Vector Databases* | High performance for vector search tasks | May not be as versatile for traditional data operations |
| *Hybrid Databases* | Support for both traditional and vector operations | Slightly lower efficiency in handling vector tasks |
| *Search Tools* | Accommodate both text and vector search tasks | May not provide the highest efficiency in vector tasks |

### Batch Access

Batch processing is an optimal approach when working with large vector sets, typically exceeding one million, that require collective, non-real-time processing. This pattern is particularly useful for Vector Management tasks such as model training or precomputing nearest neighbors, which are crucial steps in building Vector Search services.

To be sure your batch processing setup fits your application, you need to keep the following key considerations in mind:

**1. Storage Technologies**:
You need scalable, reliable storage to house your vectors during the batch pipeline. Various technologies offer different levels of efficiency:
- **Object Storage (e.g., Amazon S3, Google Cloud Storage, Azure Blob Storage)**: These solutions are cost-effective for storing static data, including substantial vector sets. They scale well and integrate with various cloud-based data processing engines. However, these solutions may not facilitate rapid, real-time communication because they're designed for data at rest.
- **Distributed File Systems (e.g., HDFS, GlusterFS)**: Designed for storing vast amounts of data across multiple servers, distributed file systems offer redundancy and high-throughput access. They seamlessly integrate with big data processing tools like Hadoop or Spark. However, setting up and maintaining them can be complex, especially in cloud environments.

**2. Data Serialization Formats**:
For storing vectors efficiently, you need compact data formats that conserve space and support fast read operations:
- **Avro and Parquet**: These data serialization formats work well with big data tools like Hadoop or Spark. They effectively compress data, support schema evolution, and integrate smoothly with cloud-based storage. Avro is suitable for write-heavy loads, while Parquet optimizes read-heavy operations.
- **Compressed NumPy Arrays**: For simpler use cases, you can serialize and compress NumPy arrays that hold vectors. This straightforward approach integrates well with Python-based processing pipelines but is probably not ideal for very large-scale or non-Python environments.

**3. Execution Environment**:
You have two primary choices for executing batch-processing tasks:
   - **On-Premise Execution**: If you must process large volumes of data on your infrastructure, tools like Hadoop and Apache Spark offer complete control over the processing environment but may require complex setup and maintenance.
   - **Cloud Services**: Cloud platforms like Amazon EMR (Elastic MapReduce) provide convenient and scalable batch processing solutions. They manage processing clusters, making scaling resources up or down easier based on your requirements.


In short, which storage technology, data serialization format, and execution environment you choose for your batch processing use case depends on various considerations. 
These considerations include 
- The size of your vector dataset,
- Whether your data is static or dynamic,
- Your workload scalability requirements,
- Whether your dataset is stored across multiple servers,
- Whether you require real-time querying,
- Your integration needs (big data processing),
- Your desired level of processing control,
- and your available resources and time for set up and maintenance. 

![Choosing a batch processing setup](assets/building_blocks/vector_search/bb3-2.png)

## Conclusion
At its core, Vector Search & Management provide the critical link between vectorized data and machine learning models that can extract insights and guide decisions. But designing and implementing your vector storage, indexing, and retrieval strategy to match the requirements of your use case involves several considerations:

First, the update frequency of your vector embeddings. Your use case may warrant an architecture optimized for real-time streaming data or higher-latency batch processing. 

Second, the access pattern you use to perform nearest neighbor search. Whether it’s responsive user queries or scheduled large batch analysis - your data pipelines and query patterns must be structured to match your access needs. 

Third, your use case will require its own particular prioritization of latency, throughput, and accuracy. To optimise your application's objectives, you must balance latency, throughput, and accuracy concerns.

Tying this together with high-quality vector embeddings produces a modern, scalable retrieval stack. Rich vectors encode important relationships combined with performant search, enabling responsive or large-scale analytics over these representations.

---
## Contributors

- [Daniel Svonava](https://www.linkedin.com/in/svonava/)
- [Paolo Perrone](https://www.linkedin.com/in/paoloperrone/)
- [Robert Turner, editor](https://robertturner.co/copyedit)
