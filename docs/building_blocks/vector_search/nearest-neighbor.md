# Nearest Neighbor Search Algorithms

**Scanning to calculate the similarity between vectors quickly** is at the heart of Vector Search. Vector similarity scores encoded by your embedding model/s store valuable feature or characteristic information about your data that can be used in various applications (e.g., content recommendation, clustering, data analysis). There are several ways to perform nearest neighbor search.

### Full Scan

Let’s say we have a dataset of 1 million 1000-dimensional vectors and want to quickly find similar vectors for a given query. Naively, this would require 1 billion operations per query – a full scan nearest neighbor search. Full scan is an **exhaustive, brute force approach** to nearest neighbor search. It sequentially checks every record or data block of the dataset. 

A full scan is simple, easy to implement, suitable when your dataset has less than 1M vectors, and not constantly changing.

But as your dataset grows past 1M vectors, or is updated frequently, full scan takes more and more time and becomes more resource-intensive. We can significantly expedite our scan using an approximate nearest neighbors (ANN) search.

### ANN Algorithms

Approximate Nearest Neighbors (ANN) is a class of algorithms that **expedite** vector similarity search through approximations, avoiding the high computational cost of exhaustive brute force (full scan) search.

ANN algorithms like locality-sensitive hashing (LSH) and hierarchical navigable small world (HNSW) can provide a tunable balance between precision and speed. HNSW and other similar methods permit very high recall, approaching brute force levels but at faster speeds.

ANN provides a crucial advantage over brute force search for large, high-dimensional datasets. [ANN-benchmarks’ “Benchmarking Results”](https://ann-benchmarks.com/) demonstrate that brute force algorithms provide the highest precision but at a tradeoff cost of fewer QPS (queries per second). **Hybrid** approaches combining ANN and exact search (i.e., full scan) can provide both speed and accuracy.

To select an ANN method appropriate to your application, you need to evaluate metrics like recall and latency in relation to your dataset scale, dimensionality, and data distribution requirements. 


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

The code snippet above shows how an ANN algorithm, like IVFFlat from the FAISS library, indexes the vectors, allowing **quick narrowing of the search space**. This lets you significantly speed up your scan, compared to a linear scan, especially as your data set becomes larger.

Different ANN implementations provide different optimization tradeoffs. Choosing the right ANN method requires benchmarking options like HNSW, IVF, and LSH on your dataset.

### Quantization

Search speed is not the only challenge when handling datasets with more than 1 million vectors. Once a dataset is too large to fit into the RAM of a single machine, you may be forced to shard the index to multiple machines. This is costly and increases system complexity. Fortunately, you can use quantization to **reduce the index size, without substantially reducing retrieval quality**.

Quantization reduces the memory required to store vectors by compressing the data but preserving relevant information. There are **three types** of vector quantization: scalar, binary, and product.

**Scalar** quantization reduces the dimensionality of your data by representing each vector component with fewer bits, reducing the amount of memory required for vector storage and speeding up the search process. Scalar quantization achieves a good balance of compression, accuracy, and speed, and is used widely.

**Binary** quantization represents each vector component as binary codes. Binary is the fastest quantization method. But it’s only efficient for high dimensional vectors and a centered distribution of vector components. As a result, binary quantization should only be used with tested models.

**Product** quantization splits the vector dimensions into smaller subvectors, each independently quantized into an index using the subvector’s set of codewords, representing data point clusters, and then combined into a single index representing the original high-dimensional vector. Product quantization provides better compression (minimal memory footprint) but is slower and less accurate than scalar quantization.

ANN search can be combined with quantization to enable running high-performance [Vector Search] pipelines on huge datasets.

---
## Contributors

- [Daniel Svonava](https://www.linkedin.com/in/svonava/)
- [Paolo Perrone](https://www.linkedin.com/in/paoloperrone/)
- [Robert Turner, editor](https://robertturner.co/copyedit)
