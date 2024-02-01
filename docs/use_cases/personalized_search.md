<!-- TODO: Replace this text with a summary of article for SEO -->

# Personalized Search: Harnessing the Power of Vector Embeddings

<!-- TODO: Cover image: 
1. You can create your own cover image and put it in the correct asset directory,
2. or you can give an explanation on how it should be and we will help you create one. Please tag arunesh@superlinked.com or @AruneshSingh (GitHub) in this case. -->

Imagine a world where your online searches return results that truly understand your needs, a world where you don't have
to know the exact words to find what you're looking for. This isn't a vision of some distant possible future; it's
happening now. Companies like Pinterest, Spotify, eBay, Airbnb, and Doordash have already taken advantage of the
treasure trove of insights inherent in data – data that is
[growing exponentially, and projected to surpass 175 zettabytes by 2025](https://www.forbes.com/sites/tomcoughlin/2018/11/27/175-zettabytes-by-2025)
– to significantly improve user experience and engagement, conversion rates, and customer satisfaction. Spotify, for
example, has been able to enhance its music recommendation system, leading to a more than 10% performance improvement in
session and track recommendation tasks, and a
[sizeable boost in user engagement and satisfaction](https://doi.org/10.1145/3383313.3412248).

And _how_ have they done this? What's enabled these companies to harvest the inherent power of data to their benefit?

The answer is vector embeddings.

Vector embeddings let you return more _relevant_ results to your search queries by: 1) querying the _meaning_ of the
search terms, as opposed to just looking for search keyword _matches_; and 2) informing your search query with the
meaning of personal preference data, through the addition of a personal preference vector.

Let's look first at how vector embeddings improve the relevance of search query results generally, and then at how
vector embeddings permit us to use the meaning of personal preferences to create truly personalized searches.

<img src=assets/use_cases/personalized_search/vector_embeddings.png alt="Illustration of vector embeddings" data-size="100" />

## Vector search vs. keyword-based search

Vector embeddings are revolutionizing the way we search and retrieve information. They work by converting data into
numerical representations, known as vectors. Conversion into vectors allows the search system to consider the
_semantics_ – the underlying meaning – of the data when performing a search.

Imagine you're searching for a book in an online store. With traditional keyword-based search, you would need to know
the exact title or author's name. But using vector embeddings, you can simply describe the book's theme or plot, and the
search system retrieves relevant results. This is because vector embeddings understand the _meaning_ of the query,
rather than just matching on the keywords in your query.

### How do vector embeddings return relevant results?

The power of vector embeddings lies in their ability to quantify similarity between vectors. This is done using a
distance metric. One of the most commonly used distance metrics is cosine similarity, which measures how close one
vector is to another; the distance between vectors is a measure of how similar the two pieces of data they represent
are. In this way, vector search is able to return relevant results even when the exact terms aren't present in the
query.

### Handling embedding model input limits

The embedding models used for vector search _do_ have maximum input length limits that users need to consider. The
twelve best-performing models, based on the
[Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), are limited to an
input size of 512 tokens. (The 13th best has an exceptional input size limit of 8192 tokens.)

But we can handle this input size limitation by segmenting the data into smaller parts that fit the model's token
constraints, or by adopting a sliding window technique. Segmenting involves cutting the text into smaller pieces that
can be individually vectorized. The sliding window method processes the text in sections, with each new window
overlapping the previous one, to maintain context between portions. These techniques adjust the data to the model's
requirements, allowing for detailed vector representation of larger texts.

Say, for example, you're searching for a specific article about a local festival in a newspaper's digital archive. The
system identifies the lengthy Sunday edition where the piece appeared, and then, to ensure a thorough search, breaks the
edition down, analyzing it article by article, much like going page by page, until it pinpoints the local festival
article you're looking for.

### But it's not only text that you can search!

The general-purpose nature of vector embeddings makes it possible to represent almost any form of data, from text to
images to audio. Back in our bookstore, vector embeddings can handle more than just title searches. We can also
represent a transaction as a vector. Each dimension of the vector represents a different attribute, such as the
transaction amount, date, or product category. By comparing these transaction vectors, the search system can identify
patterns or anomalies that would be difficult to spot with traditional search methods.

### Great! But what can I use to get started?

Using a vector database – a system designed to store and perform semantic search at scale – you can compare the query
vector with vectors stored in the database and return the top-k most similar ones. The key components of a vector
database include a vector index, a query engine, partitioning/sharding capabilities, replication features, and an
accessible API. Vector databases are categorized into vector-native databases, hybrid databases, and search engines.
Notable vector database providers include [Pinecone](https://pinecone.io), [Milvus](https://milvus.io), and
[Weaviate](https://weaviate.io).

| Key Component | Description | | --------------------- | ------------------------------------------------------- | |
Vector Index | Allows fast and efficient retrieval of similar vectors | | Query Engine | Performs optimized similarity
computations on the index | | Partitioning/Sharding | Enables horizontal scaling | | Replication | Ensures reliability
and data integrity | | API | Allows for efficient vector CRUD operations |

## How can I personalize my search (using a vector database)?

Let's illustrate, in a simple code snippet, how we might personalize a query – by adding a user preference vector:

```python
from transformers import BertTokenizer, BertModel
import torch

# Initialize an open-source embedding model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Embed search queries
def embed_query(query):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=32)
    outputs = model(**inputs)
    return ... # return output from last layer

# Sample embedding
query = "Looking for a thrilling mystery novel with a female lead."
query_embedding = embed_query(query)

# Assume we have a user preference vector from user’s past interactions
user_preference_vector = load_user_preference_vector(...)  # Placeholder Vector

# Mix the query embedding with the user preference vector
query_weight = 0.7
user_preference_weight = 0.3
biased_query_embedding = query_weight * query_embedding + user_preference_weight * user_preference_vector
```

In this code example, we convert a search query into a vector using an
[open-source, pretrained BERT model from Hugging Face](https://huggingface.co/bert-base-uncased) (you can try this out
online yourself by following the link). We also have a user preference vector, which is usually based on a user's past
clicks or choices. We then arithmetically "add" the query vector and the user preference vector to create a new query
vector that reflects both the user input and user preferences.

<img src=assets/use_cases/personalized_search/vector_space.png alt="Use cases of personalized search with vector embeddings" data-size="100" />

## Conclusions and next steps 😊

Vector embeddings are revolutionizing the way we interact with and use data. By enabling more accurate and contextually
relevant search results, they are paving the way for a new era of data-driven insights and decision-making. It's not
only early adopters like Pinterest, Spotify, eBay, Airbnb, and Doordash who have
[reaped the benefits of vector search integration](https://rockset.com/blog/introduction-to-semantic-search-from-keyword-to-vector-search/).
Any company can take advantage of vector search to enhance user experience and engagement. Home Depot, for example,
responded to increased online activity during the COVID pandemic period by integrating vector search, leading to
[improved customer service and a boost in online sales](https://www.datanami.com/2022/03/15/home-depot-finds-diy-success-with-vector-search/).
The future of search is here, and it's powered by vector embeddings.

So, what's next? How can you start implementing personalized search in your organization? There are plenty of resources
and tools available to help you get started. For instance, you can check out this
[guide on implementing vector search](https://hub.superlinked.com/vector-search) or this
[tutorial on using vector embeddings](https://hub.superlinked.com/vector-compute).

## Share your thoughts and stay updated

What are your thoughts on personalized search using vector embeddings? Have you used this technology in your
organization? If you'd like to contribute an article to the conversation, don't hesitate to
[get in touch](https://github.com/superlinked/VectorHub)!

Stay Updated: Drop your email in the footer to stay up to date with new resources coming out of VectorHub and
Superlinked.

Your feedback shapes VectorHub! Found an Issue or Have a Suggestion? If you spot something off in an article or have a
topic you want us to dive into, create a GitHub issue and we'll get on it!

______________________________________________________________________

## Contributors

- [Michael Jancen-Widmer, author](https://www.contrarian.ai)
- [Robert Turner, editor](https://robertturner.co/copyedit)
