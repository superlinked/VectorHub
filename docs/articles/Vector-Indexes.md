# Vector Indexes

This article takes you through the basics of implementating vector indexing in Python using LlamaIndex. But first, let's briefly introduce vector indexes, why they're so important, different common types, and popular use cases.

## Why you need vector indexes

Running AI applications depends on vectors, often called [embeddings](https://superlinked.com/vectorhub/building-blocks/vector-compute/embedding-models) - dense data representations, generated via complex mathematical operations to capture key attributes of source information. When a user submits a query, it's also embedded in the vector database, and vectors that are close to it are deemed relevant (i.e., similar), and returned. It's possible to query these vectors *without* a vector index, but that would be a very inefficient, brute-force process, checking every single vector in the database to see if they match the query vector, one by one. Which, if the database is large, takes a long time. Vector indexing, using specialized algorithms that understand the data and create groups of matching elements, speeds up our similarity searches.

Similarity searches use algorithms to calculate vector closeness using metrics like Euclidean or Jacobian distance. For small datasets, where accuracy is more important than efficiency, you can use K-Nearest Neighbors to pinpoint exactly the closest near neighbors to your query. But as datasets become larger, and your AI application requires efficiency, it's better to use [Approximate Nearest Neighbour](https://superlinked.com/vectorhub/building-blocks/vector-search/nearest-neighbor-algorithms) (ANN), which returns results that are accurate enough very quickly.

Vector indexing is crucial in Retrieval-Augmented Generation (RAG), which improves LLMs' ability to sift through extensive data and efficiently find relevant vectors. Say a user queries an LLM about the "benefits of apples". Because the attritbutes of apples are stored in embeddings and indexed inside the vector database, the LLM can recognize an apple as a "fruit", and retrieve more general information about fruits - including, for example, bananas and oranges - to construct a more information-rich response.

## Types of indexing

Not all vector indexes are the same. Let's take a look at some of the most commonly used ones, and when it makes sense to use them.

### Flat indexing

Flat indexing is the most basic type of indexing - it stores vectors in the database as they are. No special operation is performed to modify or optimize them. Flat indexing is a brute-force approach: you generate similarity scores by comparing query vectors with every other vector in the database, but indexing enables you to do these computations in parallel, speeding up the process.

Flat indexing employs a very precise algorithm, returning perfectly accurate results as top k closest matches, but at a cost. Despite enabling parallel processing, flat indexing is computationally expensive and not ideal if your database consists of millions of records or more.

### Locality-Sensitive Hashing (LSH)

LSH uses a hashing function to compute a hash for each vector element, then groups vectors with similar hashes into buckets. The query vector is also hashed into a bucket with similar vectors thereby reducing the search space, and dramatically improving efficiency.

### Inverted File Index (IVF)

IVF, like LSH, groups data elements to improve vector search efficiency. But instead of hashing, IVF uses clustering techniques to prefilter data. Simpler IVF techniques may use K-means clustering to create cluster centroids. At query time, the query vector is compared to partition centroids to find the closest clusters, then vector search happens within those partitions. There are variations of IVF, each performing differently in terms of storage and retrieval efficiency. Let's take a look at a few: IVF_FLAT, IVF_PQ, and IVF_SQ.

**IVF_FLAT**

The flat variation clusters the vectors and creates centroids but stores each vector as it is with no additional processing. As a result, IVF_FLAT searches within clusters linearly (i.e., brute-force), providing good accuracy. But IVF_FLAT stores whole vectors and requires more memory, so it becomes slow with large datasets.

**IVF_PQ**

Inverted File Product Quantization (IVF_PQ) algorithm reduces storage requirements by:

* 1. dividing the vector space into centroid clusters
* 2. breaking each vector within the cluster into smaller chunks - e.g., a 12-dimensional vector can be broken into 3 chunks (sub-vectors), each with 4 dimensions
* 3. quantizing the chunks (4-dimensional sub-vectors) into bits, significantly reducing storage requirements

At query time, the same algorithm is applied to the query vector.

IVF_PQ saves on space, and, because we’re comparing smaller vectors, improves vector search times. Some accuracy is lost.

**IVF_SQ**

Inverted File Scalar Quantization uses a simpler algorithm than IVF_PQ.

* 1. divide the dataset, clustering the data points into smaller manageable chunks ("inverted lists")
* 2. create bins: for each vector dimension, determine minimum (start values) and maximum values, calculate step sizes ( = (max - min) / # bins)
* 3. convert floating-point vectors into scalar integer vectors by dividing each vector dimension into bins
* 4. assign each quantized vector to nearest chunk (inverted list)

Suppose we have a vector, x=[1.2,3.5,−0.7,2.1]. It has 4 dimensions, so we'll define 4 quantization bins:

* Bin 0: [−1.0,0.0)[-1.0,0.0)[−1.0,0.0)
* Bin 1: [0.0,1.0)[0.0,1.0)[0.0,1.0)
* Bin 2: [1.0,2.0)[1.0,2.0)[1.0,2.0)
* Bin 3: [2.0,4.0)[2.0,4.0)[2.0,4.0)

Each vector element will be distributed into a bin as follows:

* x1=1.2, falls into Bin 2 [1.0,2.0), and quantizes to 2
* x2=3.5, falls into Bin 3 [2.0,4.0), and quantizes to 3
* x3=−0.7, falls into Bin 0 [−1.0,0.0), and quantizes to 0
* x4=2.1, falls into Bin 3 [2.0,4.0), and quantizes to 3

The final quantized vector becomes [2,3,0,3].

IVF_SQ makes sense when dealing with medium to large datasets where memory efficiency is important.

### DiskANN

Most ANN algorithms - including those above - are designed for in-memory computation. But when you're dealing with *big data*, in-memory computation can be a bottleneck. Disk-based ANN [DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf) is built to leverage Solid-State Drives' (SSDs') large memory and high-speed capabilities. DiskANN indexes vectors using the Vamana algorithm, a graph-based indexing structure that minimizes the number of sequential disk reads required during, by creating a graph with a smaller search "diameter" - the max distance between any two nodes (representing vectors), measured as the least number of hops (edges) to get from one to the other. This makes the search process more efficient, especially for the kind of large-scale datasets that are stored on SSDs.

By using a SSD to store and search its graph index, DiskANN can be cost-effective, scalable, and efficient.

### Scalable Progressive Approximate Nearest Neighbor Search (SPANN)

[SPANN](https://openreview.net/forum?id=-1rrzmJCp4) can handle even larger datasets than DiskANN with high accuracy and low latency by taking a hybrid indexing approach that leverages both in-memory and disk-based storage. SPANN stores centroid points of posting lists in memory, and stores the posting lists (corresponding to clusters of similar vectors) themselves in the disk. SPANN can then very quickly - without accessing the disk - identify which clusters are relevant to a query.

To mitigate potential bottlenecks in disk read-write operations, SPANN builds its indexes using a hierarchical balanced clustering algorithm that ensures that posting lists are roughly the same length, and augments the posting list by including inside a cluster data points that are on the edges of that cluster. 

improving recall by ensuring that the posting lists are more comprehensive and include points that are relevant but might be on the edge of the clusters.


...
to ensure that posting lists are balanced in size. They also implement dynamic pruning to select clusters that match data points smartly. It first selects the top k closest clusters based on the distance between the centroids and the query. From the selected vectors, a relevance score between the posting lists and the query is calculated to filter the search further and only retrieve the most relevant data points.
...
------
 "balance the length of posting lists and augment the posting list by adding the points in the closure of the corresponding clusters. In the search stage, we use a query-aware scheme to dynamically prune the access of unnecessary posting lists."


Taking this hybrid approach, in addition to scaling horizontally (by adding more machines), SPANN can achieve high performance levels while handling larger scale datasets than DiskANN.

------

### Hierarchical Navigable Small Worlds (HNSW)

HNSW is a complicated but one of the most popular and efficient techniques for vector indexing. It combines proximity graphs and skip lists to divide the vector search space into layers. The lowest layer of the graph represents every vector as a vertex. All the vertices are connected based on their distance. Vertices are also connected across different layers depending on the position of other vectors in that layer. As we move up the multi-layer graph, the data points are grouped together, leaving fewer vertices at each level (similar to a skip list).

During the search, the query vector is placed on the highest layer, which is matched with the closest vector using an ANN algorithm. We then move to the next layer based on the earlier selected closest neighbor and repeat the same process. This is continued until the final layer is reached.

## Vector indexes in practical use cases

The similarity-matching capabilities of vector databases are used in various interesting applications. Some of these include:

* **Retrieval Augmented Generation (RAG)**: [RAG](https://superlinked.com/vectorhub/articles/advanced-retrieval-augmented-generation) uses vector indexing techniques to query relevant documents from an external database. These allow the LLM to construct a well-thought-out, accurate, and informative response for the user.

* **Image search using text queries**: Vector indexing powers [semantic search in image databases](https://superlinked.com/vectorhub/articles/retrieval-from-image-text-modalities) such as those in modern smartphones. This allows the user to input text queries and return images described by natural language.

* **Searching through large text documents**: Conventional algorithms can be inefficient in searching large corpora. Vector indexes quickly sift through the text and retrieve exact and similar matching documents. This is helpful as text queries can be tricky sometimes. For example, if a user searches for ‘movies related to time travel’, the terms `movie` and `time travel` will help reach all relevant information in the text, even if the exact terms are not present.

* **Advanced search on e-commerce websites**: E-commerce stores can employ vector indexing to improve user experience. Users can describe the product they want and the search algorithm can present all matching results. moreover, if a specific product is not available, the ANN algorithm can help retrieve similar items to keep the user intrigued.

## Basic Implementation of Vector Indexing using ANN

In this section, we will implement vector indexing in Python to understand how vectors are stored and retrieved using the Approximate Nearest Neighbour approach. We will first implement flat indexing (Brute force) and then use clustering to implement Inverted File indexing and reduce retrieval time.

### Flat Indexing

This implementation will use the basic Pandas and Numpy libraries, so no additional installation is required. Let's import the required libraries.

```python
import numpy as np
import pandas as pd
```

#### Creating Data Vectors

For this tutorial, we will generate dummy random vectors and use them as data points in our algorithm.

```python
# Generate 10 random 3D vectors
random_vectors = np.random.rand(10, 3)

# Print the random 3D vectors
for i, vector in enumerate(random_vectors):
    print(f"Vector {i+1}: {vector}")
```

```bash
Vector 1: [0.80981775 0.1963886  0.02965684]
Vector 2: [0.56029493 0.62403894 0.56696611]
Vector 3: [0.47052236 0.1559688  0.87332513]
Vector 4: [0.80995196 0.00508334 0.10529516]
Vector 5: [0.55410133 0.96722797 0.96957061]
Vector 6: [0.20098567 0.0527692  0.65003235]
Vector 7: [0.34715575 0.41244063 0.72056698]
Vector 8: [0.30936325 0.47881762 0.75795186]
Vector 9: [0.67403853 0.23895253 0.87895722]
Vector 10: [0.8207672  0.21424442 0.20621931]
```

#### Distance Calculation

Now we need a function to calculate the distance between the different vectors. We will implement the Euclidean Distance to approximate the relation between the vectors.

```python
def calculate_euclidean_distance(point1, point2):

    # calculate the euclidean distance between the provided points
    return np.sqrt(sum((point1 - point2)**2))
```

#### Vector Querying

We now have everything in place to query the data store. We will now use a test data point and retrieve the top-k closest matches. The algorithm will:

1. Generate a random test point.
2. Calculate the distances between the test point and all vectors in the data store (This is the brute force approach).
3. Retreive the top k vectors closest to the test point.

```python
# generate a random test vector
test_point = np.random.rand(1, 3)[0]
print("Test Vector:", test_point)
```

```plaintext
Test Vector: [0.98897054 0.03674464 0.59036213]
```

The following loop will calculate the euclidean distance between the provided points.

```python
distances = []

# calculate the distance from each point in our database
for i, vector in enumerate(random_vectors):
    d = calculate_euclidean_distance(test_point, vector)
    distances.append((i,d))
```

```bash
print("Distances: ")
for val in distances:
    print(f"Vector {val[0]+1}: {val[1]}")
```

```bash
Distances: 
Vector 1: 0.9624227010223827
Vector 2: 0.47065580576204386
Vector 3: 0.18677305533181324
Vector 4: 0.9402397257996893
Vector 5: 0.6393489353421491
Vector 6: 0.49525570368173166
Vector 7: 0.27160367994034773
Vector 8: 0.2958891670880358
Vector 9: 0.20437726721718957
Vector 10: 0.801081734496685
```

A quick glance shows that vector 3, 9, and 7 are the top 3 closest matches. Let's write the function to automatically retreive these.

```python
# get top k vectors
def get_top_k(random_vectors, distances, k):

    # sort the distances in ascending order
    sorted_distance = sorted(distances, key=lambda distances: distances[1])
    # retreive the top k smallest distance indexes
    top_k = sorted_distance[:k]

    # top macthes
    top_matches = []
    for idx in top_k:
        # Get the first element of the tuple as the index
        idx_to_get = idx[0]  
        top_matches.append(random_vectors[idx_to_get])

    return top_matches
```

```python
# Retreive the top 3 matches
top_matching_vectors = get_top_k(random_vectors, distances, 3)
top_matching_vectors
```

```bash
[array([0.47052236, 0.1559688 , 0.87332513]),
 array([0.67403853, 0.23895253, 0.87895722]),
 array([0.34715575, 0.41244063, 0.72056698])]
```

The above data points can represent text, images, or any other encoded data form. This way, flat indexing can be used to implement similarity search.

### IVF

Now we will see how we can improve the efficiency of search by clustering the vector data points. We will use the K-means clustering algorithm to create segements of the vector database and reduce the search space at runtime.

We will first need to import the Kmeans implementation from SkLearn.

```python
from sklearn.cluster import KMeans
```

We will also increase the data store size just to make things interesting. Let's generate 100 random vectors this time.

```python
# generate large number of data points
data_points  = np.random.rand(100, 3)
```

#### K-means Implementation

Now, we initialize the Kmeans algorithm.

```python
kmeans = KMeans(init="k-means++",n_clusters=3,n_init=10,max_iter=300,random_state=42)
```

By default, here, we select the data to be seperated into 3 clusters. However, using algorithms like the elbow method to determine the optimal clusters is better. Let's create clusters using our initialized method.

```python
# Fit k-means on our data points
kmeans.fit(data_points)
```

It is important to note that clustering can increase time complexity for large databases, but it is a one-time operation and will improve query times later.

Let's store the cluster centroids and label data points according to the clusters.

```python
# store centroids and data point associations
centroids = {}
for i, c in enumerate(kmeans.cluster_centers_):
    centroids[i] = c

data_point_clusters = {i:[] for i in np.unique(kmeans.labels_)}
for i, l in enumerate(kmeans.labels_):
    data_point_clusters[l].append(data_points[i])
```

```python
centroids
```

```bash
{0: array([0.41329647, 0.21550016, 0.76765634]),
 1: array([0.77927426, 0.49457407, 0.36967834]),
 2: array([0.23936467, 0.73028761, 0.3946528 ])}
```

#### Vector Querying

Now, we need to query a test vector. The algorithm will
1. Generate a random test point.
2. Calculate the closest cluster to the test point using the centroid vectors.
3. Calculate the Euclidean distances with vectors in the closest cluster only.
4. Retreive the closest vectors from the calculated distances.

```python
# Generate a test point
test_point = np.random.rand(1, 3)[0]
test_point
```

```bash
array([0.19125138, 0.01180168, 0.65204789])
```

We need to implement functions that will calculate the closest matching cluster and retrieve the Top-k matching vectors.

```python
def get_closest_centroid(test_point, centroids):
    '''
    Return the centroid label closest to the test point
    '''

    distances = {}
    for cent_label, cent in centroids.items():
        # using the same function defined in earlier implementation
        distances[cent_label] = calculate_euclidean_distance(test_point, cent)
    
    # get minimum label
    min_key = min(distances, key=distances.get)

    return min_key
```

```python
def get_top_k_from_macthing_cluster(data_point_clusters, closest_cluster, test_point, k = 3):
    '''
    Function to calculate nearest neighbors for test point but only in the selected cluster
    '''

    dist_for_cluster = []

    # calculate the distance from each point in our database
    for i, vector in enumerate(data_point_clusters[closest_cluster]):
        d = calculate_euclidean_distance(test_point, vector)
        dist_for_cluster.append((i,d))

    # sort the distances in ascending order
    sorted_distance = sorted(dist_for_cluster, key=lambda dist_for_cluster: dist_for_cluster[1])
    # retreive the top k smallest distance indexes
    top_k = sorted_distance[:k]

    # top matches
    top_matches = []
    for idx in top_k:
        # Get the first element of the tuple as the index
        idx_to_get = idx[0]  
        top_matches.append(data_point_clusters[closest_cluster][idx_to_get])

    return top_matches
```

Now, we just need to pass our test point to the above functions.

```python
# get the closest cluster
closest_cluster = get_closest_centroid(test_point, centroids)

# get the top k vector from the closest cluster
macthes = get_top_k_from_macthing_cluster(data_point_clusters, closest_cluster, test_point, k = 3)

print("Test Point: ", test_point)
print("Closest Matches: ", macthes)
```

```bash
Test Point:  [0.19125138 0.01180168 0.65204789]
Closest Matches:  [array([0.1338926 , 0.04131207, 0.66420534]), array([0.09719324, 0.01207222, 0.80169845]), array([0.30091393, 0.14300888, 0.82777153])]
```

Since the algorithm only queries the matching cluster, it significantly reduces the search complexity. Instead of performing 100 computations (The size of the database), it only has to look through the cluster space. 

## Conclusion

Vector indexing is a powerful technique for optimizing the retrieval of vectorized information. Various indexing techniques use clustering algorithms to group similar information and reduce the search space for efficient data search.

Moreover, the technique also allows the approximate nearest neighbor search (ANN) to retrieve similar matching indexes. This allows users to retrieve matching information if an exact match is not found. It also enables an information-rich response as it can gather additional data from matching vectors.

Vector indexes empower some critical modern-day applications. Perhaps the most important one has been RAG, which allows LLMs to access additional information from an external vector store. The ANN search allows an LLM to retrieve all relevant information about a query and formulate an accurate response. Other benefits of vector indexes are found in search applications such as image search and eCommerce platforms.

## Contributors

[Haziqa Sajid, author](https://www.linkedin.com/in/haziqa-sajid-22b53245/)
[Mór Kapronczay, editor](https://www.linkedin.com/in/m%C3%B3r-kapronczay-49447692/)
[Robert Turner, editor](https://www.linkedin.com/in/robertdhayanturner/)
