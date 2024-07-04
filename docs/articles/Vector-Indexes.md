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

Flat Indexing is a very precise algorithm, as it retrieves vectors with great accuracy. However, it is computationally expensive and not ideal in cases where the database consists of millions of records or more.

#### Locality Sensitive Hashing (LSH)
LSH optimizes the vector search operation by dividing the database elements into buckets. Here, we first use a hashing function to compute hashes for each vector element. Then, based on the hash’s similarity, the vectors are grouped together into buckets. Intuitively, each bucket contains a matching vector.
Now, when a query vector appears, it is first hashed using the same function. Then, based on a hash, it is assigned a bucket containing all its similar vectors. The query vector now only needs to be compared with the bucket vectors, reducing the search space and dramatically improving the efficiency.

#### Inverted File (IVF)
IVF works similarly to LSH and creates groups of data. But rather than hashing it using clustering techniques. The techniques can vary depending on the implementation but simpler techniques may use K-means clustering and then use the cluster centroids as reference for query vectors. The query vector is then compared with only its associated data cluster to improve efficiency.

#### Hierarchical Navigable Small Worlds (HNSW)
HNSW is a complicated but one of the most popular and efficient techniques for vector indexing. It combines proximity graphs and skip lists to divide the vector search space into layers. The lowest layer of the graph represents every vector as a vertex. All the vertices are connected based on their distance. Vertices are also connected across different layers depending on the position of other vectors in that layer. As we move up the multi-layer graph, the data points are grouped together, leaving fewer vertices at each level (similar to a skip list).

During the search, the query vector is placed on the highest layer, which is matched with the closest vector using an ANN algorithm. We then move to the next layer based on the earlier selected closest neighbor and repeat the same process. This is continued until the final layer is reached.

## Vector Indexes in Practical Use Cases
The similarity-matching capabilities of vector databases are used in various interesting applications. Some of these include:

- **Retrieval Augmented Generation (RAG)**: [RAG](https://superlinked.com/vectorhub/articles/advanced-retrieval-augmented-generation) uses vector indexing techniques to query relevant documents from an external database. These allow the LLM to construct a well-thought-out, accurate, and informative response for the user.

- **Image search using text queries**: Vector indexing powers semantic search in [image databases](https://superlinked.com/vectorhub/articles/retrieval-from-image-text-modalities) such as those in modern smartphones. This allows the user to input text queries and return images described by natural language.

- **Searching through large text documents**: Conventional algorithms can be inefficient in searching large corpora. Vector indexes quickly sift through the text and retrieve exact and similar matching documents. This is helpful as text queries can be tricky sometimes. For example, if a user searches for ‘movies related to time travel’, the terms `movie` and `time travel` will help reach all relevant information in the text, even if the exact terms are not present.

- **Advanced search on e-commerce websites**: E-commerce stores can employ vector indexing to improve user experience. Users can describe the product they want and the search algorithm can present all matching results. moreover, if a specific product is not available, the ANN algorithm can help retrieve similar items to keep the user intrigued.

## Basic Implementation of Vector Index
To implement vector indexes, we first need a vector store to store the data in the form of vector embeddings. There are various frameworks available that implement vector stores. However, in this tutorial, we will use the LlamaIndex vector store and demonstrate its built-in capabilities to store and query vector indexes. Additionally, by default, LlamaIndex uses GPT LLM for creating embedding and formulating a natural language response so you will also need an OpenAI key to run the code.

We will start by installing and importing the required libraries.
```bash
pip install openai llama-index
```

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import openai
```
### Loading Data as Vectors
After initialization, we can use the built-in document loader from llama index. We are using a couple of essays from [Paul Graham](https://www.paulgraham.com/articles.html), which will be converted to embeddings and stored in the vector store.
```python
# Initialize OpenAI key
openai.api_key = 'YOUR_OPENAI_KEY'

# Load documents and build index
documents = SimpleDirectoryReader(
    input_files=["/content/data.txt"]
).load_data()
index = VectorStoreIndex.from_documents(documents)
```
Our .txt files are stored in a folder named ‘data’. The SimpleDirectoryReader loops over all available files and tries to read them as text. The VectorStoreIndex module then converts the loaded data into vector indexes.

### Querying the Vector Store
Once the indexes are created, we can query the database for results. However, since this implementation uses the GPT LLM, we can formulate our query in natural language, and the model should understand it.

Our loaded documents consist of an interview with the founder of Airbnb and a technical article about programming principles, so our query should be relevant to these.
```python
query_engine = index.as_query_engine()

# input a query
response=query_engine.query("Did the interviewer have any cool names for the AirBnb Founders and was the it?")
print(response)
```

```bash
>> The interviewer had a cool nickname for Brian Chesky, which was "The Tasmanian Devil."

```
Let’s try another.
```python
response=query_engine.query("Benefits of bottom-up programming")
print(response)
```
```bash
>> Bottom-up programming yields programs that are smaller, more agile, and easier to read. It promotes code re-use, reduces the size and complexity of programs, and helps clarify ideas about program design by identifying patterns and encouraging simpler redesigns.
```
### Saving and Re-loading Indexes
Once the data is processed, the embeddings can be saved for later use.
```python
# Save indexes on disk
index.storage_context.persist()

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
loaded_index = load_index_from_storage(storage_context)
```
The first command above saves the indexes and related metadata in a JSON format. The data is saved in a directory named ‘storage’ and can be loaded easily whenever needed.

Conclusion
Vector indexing is a powerful technique for optimizing the retrieval of vectorized information. Various indexing techniques use clustering algorithms to group similar information and reduce the search space for efficient data search.

Moreover, the technique also allows the approximate nearest neighbor search (ANN) to retrieve similar matching indexes. This allows users to retrieve matching information if an exact match is not found. It also enables an information-rich response as it can gather additional data from matching vectors.

Vector indexes empower some critical modern-day applications. Perhaps the most important one has been RAG, which allows LLMs to access additional information from an external vector store. The ANN search allows an LLM to retrieve all relevant information about a query and formulate an accurate response. Other benefits of vector indexes are found in search applications such as image search and eCommerce platforms.


