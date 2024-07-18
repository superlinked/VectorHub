# Vector Indexes
As large language models (LLMs), like ChatGPT, gain traction, vector data has become the new norm. A vector database stores data in the form of vectors, as opposed to scalars, in a conventional relational database management system (RDBMS). While scalar data is represented by a single element, vector data has multiple attributes that help build a semantic relationship between the different data elements.

A major benefit of vector data is that it allows for similarity search using vector indexes. The similarity search opens avenues for various new applications like Retrieval Augmented Generation (RAG). This blog will discuss vector indexes in detail. We will also look at popular types of indexing and implement vector indexing in Python using LlamaIndex.

## Overview of Vector Indexes
Vector data is often called [‘Embeddings’](https://superlinked.com/vectorhub/building-blocks/vector-compute/embedding-models), and these are generated via complex mathematical operations. The embeddings are a dense data representation, capturing key attributes describing the information. These attributes allow vectors to be arranged so that similar data points are closer. This operation, called vector indexing, has various benefits, including faster search operation, similarity matching, and pattern identification. The indexing operation requires specialized algorithms that understand the data and create groups of matching elements.

Vector indexing also benefits Retrieval-Augmented Generation (RAG), allowing LLMs to sift extensive data and find relevant vectors efficiently. This is important since efficiency is key to modern AI application development. Moreover, it also allows LLMs to filter information similar to the user’s query so that the model can present all relevant information. Imagine an LLM is asked about the benefits of apples. The model will retrieve information from a vector database, and since it recognizes an apple as a fruit, it can also query information regarding bananas and oranges and construct a more information-rich response.

The similarity search uses an [Approximate Nearest Neighbour](https://superlinked.com/vectorhub/building-blocks/vector-search/nearest-neighbor-algorithms) (ANN) algorithm to find matching data vectors. It calculates how close or distant vectors are using distance metrics like Euclidean distance or Jacobian distance. Closer points are grouped together and referenced in relevant queries.

### Types of Indexing
Let’s discuss some popular indexes involved in vector indexing.

#### Flat Indexing
Flat indexing is the most basic type of indexing, as it stores vectors in the database as they are. No special operation is performed for modification or optimization. When a query vector arrives, it is compared with every other vector within the database to generate a similarity score. Due to this, flat indexing is also considered a brute-force approach. Once the similarity scores are calculated, the top k closest matches are retrieved.

Flat Indexing is a very precise algorithm, as it retrieves vectors with perfect accuracy. However, it is computationally expensive and not ideal in cases where the database consists of millions of records or more.

#### Locality Sensitive Hashing (LSH)
LSH optimizes the vector search operation by dividing the database elements into buckets. Here, we first use a hashing function to compute hashes for each vector element. Then, based on the hash’s similarity, the vectors are grouped together into buckets. Intuitively, each bucket contains a matching vector.
Now, when a query vector appears, it is first hashed using the same function. Then, based on a hash, it is assigned a bucket containing all its similar vectors. The query vector now only needs to be compared with the bucket vectors, reducing the search space and dramatically improving the efficiency.

#### Inverted File (IVF)
IVF works similarly to LSH and creates groups of data. But rather than hashing it, it uses clustering techniques. The techniques can vary depending on the implementation, but simpler techniques may use K-means clustering and then use the cluster centroids as a reference for query vectors. The query vector is then compared with only its associated data cluster to improve efficiency. IVF has a few variations, each improving the storage and retrieval efficiency.

##### IVFFLAT
The flat variation clusters the vectors and creates centroids but stores each vector as it is. It performs no additional processing beyond the clustering, and a brute-force approach is required to search through any given cluster.

##### IVFPQ
Inverted File Product Quantization (IVFPQ) improves the vector storage cost by quantizing the data. It begins by clustering the data points, but before storing the vectors, it quantizes the data. Each vector in a cluster is divided into sub-vectors, and each sub-vector is encoded into bits using product quantization.

At search time, the query vector is first associated with a cluster. Then, it is quantified, and the encoded sub-vectors are compared with the encoded data within the cluster. This storage method has various benefits, such as it saves on space, and since we’re comparing smaller vectors, it improves vector search times.
##### IVFSQ
Inverted File Scalar Quantization (IVFSQ) is similar to IVFPQ but uses a simpler quantization algorithm. It begins by clustering the data points and then quantizing each vector separately. The quantization algorithm quantizes each vector dimension. The first step is to create bins to map the vector elements to. Each element is mapped to a bin, and depending on the bin range, the floating point number is converted to a scalar integer.

Suppose we have a vector x=[1.2,3.5,−0.7,2.1]. As it has 4 dimensions, we will define 4 quantization bins such as the following:
- Bin 0: [−1.0,0.0)[-1.0, 0.0)[−1.0,0.0)
- Bin 1: [0.0,1.0)[0.0, 1.0)[0.0,1.0)
- Bin 2: [1.0,2.0)[1.0, 2.0)[1.0,2.0)
- Bin 3: [2.0,4.0)[2.0, 4.0)[2.0,4.0)

Each vector element will be distributed into a bin as follows: 
- For x1=1.2: It falls into Bin 2 [1.0,2.0), so it is quantized to 2.
- For x2=3.5: It falls into Bin 3 [2.0,4.0), so it is quantized to 3.
- For x3=−0.7: It falls into Bin 0 [−1.0,0.0), so it is quantized to 0.
- For x4=2.1: It falls into Bin 3 [2.0,4.0), so it is quantized to 3.

The final quantized vector becomes [2,3,0,3].

#### DiskANN
Most ANN algorithms are designed for in-memory computation, which can be a bottleneck when working with big data. [DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf) is built to leverage the large memory and high-speed capabilities of Solid-State Drives (SSDs). It uses a graph-based structure to index the vector data points. The graph nodes represent individual vectors and are connected to other nodes based on their similarity. For this implementation, the authors introduce Vamana, a special graph structure with a smaller "diameter" (the maximum number of hops needed to reach any two points), minimizing the number of sequential disk reads required during the search.

The graph index is entirely stored on disk and accessed during search. The overall approach is cost-effective and scalable and implements an efficient search strategy.

#### Scalable Progressive Approximate Nearest Neighbor Search (SPANN)

[SPANN](https://openreview.net/forum?id=-1rrzmJCp4) improves upon DiskANN by introducing a memory-disk hybrid approach designed for billion-scale datasets. The approach clusters the data and stores the centroids in memory, but data clusters (or pointing lists) on disk.

Since it involves read-write operations with the disk, these can also become a bottleneck. The authors use a few techniques to reduce disk read operations and storage costs. They use a balanced hierarchical clustering algorithm to ensure that posting lists are balanced in size. They also implement dynamic pruning to select clusters that match data points smartly. It first selects the top k closest clusters based on the distance between the centroids and the query. From the selected vectors, a relevance score between the pointing lists and the query is calculated to filter the search further and only retrieve the most relevant data points.

#### Hierarchical Navigable Small Worlds (HNSW)
HNSW is a complicated but one of the most popular and efficient techniques for vector indexing. It combines proximity graphs and skip lists to divide the vector search space into layers. The lowest layer of the graph represents every vector as a vertex. All the vertices are connected based on their distance. Vertices are also connected across different layers depending on the position of other vectors in that layer. As we move up the multi-layer graph, the data points are grouped together, leaving fewer vertices at each level (similar to a skip list).

During the search, the query vector is placed on the highest layer, which is matched with the closest vector using an ANN algorithm. We then move to the next layer based on the earlier selected closest neighbor and repeat the same process. This is continued until the final layer is reached.

## Vector Indexes in Practical Use Cases
The similarity-matching capabilities of vector databases are used in various interesting applications. Some of these include:

- **Retrieval Augmented Generation (RAG)**: [RAG](https://superlinked.com/vectorhub/articles/advanced-retrieval-augmented-generation) uses vector indexing techniques to query relevant documents from an external database. These allow the LLM to construct a well-thought-out, accurate, and informative response for the user.

- **Image search using text queries**: Vector indexing powers semantic search in [image databases](https://superlinked.com/vectorhub/articles/retrieval-from-image-text-modalities) such as those in modern smartphones. This allows the user to input text queries and return images described by natural language.

- **Searching through large text documents**: Conventional algorithms can be inefficient in searching large corpora. Vector indexes quickly sift through the text and retrieve exact and similar matching documents. This is helpful as text queries can be tricky sometimes. For example, if a user searches for ‘movies related to time travel’, the terms `movie` and `time travel` will help reach all relevant information in the text, even if the exact terms are not present.

- **Advanced search on e-commerce websites**: E-commerce stores can employ vector indexing to improve user experience. Users can describe the product they want and the search algorithm can present all matching results. moreover, if a specific product is not available, the ANN algorithm can help retrieve similar items to keep the user intrigued.

## Basic Implementation of Vector Indexing using ANN
In this section, we will implement vector indexing in Python to understand how vectors are stored and retrieved using the Approximate Nearest Neighbour approach. We will first implement flat indexing (Brute force) and then use clustering to implement Inverted File indexing and reduce retrieval time.
### Flat Indexing
This implementation will use the basic Pandas and Numpy libraries, so no additional installation is required. Let's import the required libraries.

```python
import numpy as np
import pandas as pd
```
#### Creating Data Vectors
For this tutorial we will generate dummy random vectors and use them as data point in our algorithm.
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
Now we need to function to calculate the distance between the different vectors. We will implement the Euclidean Distance to approximate the relation between the vectors.
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

```Python
# generate a random test vector
test_point = np.random.rand(1, 3)[0]
print("Test Vector:", test_point)
```
```
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
```Python
# Retreive the top 3 matches
top_matching_vectors = get_top_k(random_vectors, distances, 3)
top_matching_vectors
```
```Bash
[array([0.47052236, 0.1559688 , 0.87332513]),
 array([0.67403853, 0.23895253, 0.87895722]),
 array([0.34715575, 0.41244063, 0.72056698])]
```
The above data points can represent text, images, or any other encoded data form. This way, flat indexing can be used to implement similarity search.
### IVF
Now we will see how we can improve the efficiency of search by clustering the vector data points. We will use the K-means clustering algorithm to create segements of the vector database and reduce the search space at runtime.

We will first need to import the Kmeans implementation from SkLearn.
```Python
from sklearn.cluster import KMeans
```
We will also increase the data store size just to make things interesting. Let's generate 100 random vectors this time.
```python
# generate large number of data points
data_points  = np.random.rand(100, 3)
```
#### K-means Implementation
Now, we initialize the Kmeans algorithm.
```Python
kmeans = KMeans(init="k-means++",n_clusters=3,n_init=10,max_iter=300,random_state=42)
```
By default, here, we select the data to be seperated into 3 clusters. However, using algorithms like the elbow method to determine the optimal clusters is better. Let's create clusters using our initialized method.
```Python
# Fit k-means on our data points
kmeans.fit(data_points)
```
It is important to note that clustering can increase time complexity for large databases, but it is a one-time operation and will improve query times later.

Let's store the cluster centroids and label data points according to the clusters.
```Python
# store centroids and data point associations
centroids = {}
for i, c in enumerate(kmeans.cluster_centers_):
    centroids[i] = c

data_point_clusters = {i:[] for i in np.unique(kmeans.labels_)}
for i, l in enumerate(kmeans.labels_):
    data_point_clusters[l].append(data_points[i])
```
```Python
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

```Python
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
```Python
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


