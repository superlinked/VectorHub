# Personalized Search: Harnessing the Power of Vector Embeddings

## Introduction

Imagine a world where your online searches return results that truly understand your needs. A world where you don't have to know the exact words to find what you're looking for. This isn't a distant future; it's happening now, with personalized search using vector embeddings. Companies like Pinterest, Spotify, eBay, Airbnb, and Doordash have integrated vector search into their platforms, seeing significant improvements in user experience and engagement, leading to higher conversion rates and customer satisfaction.

Spotify, for example, uses vector embeddings to enhance its music recommendation system. When evaluated on a large-scale dataset, the use of vector embeddings led to a performance improvement of upwards of 10% in session and track recommendation tasks, showcasing a significant boost in user engagement and satisfaction on the platform [(Hansen et al., 2020)](https://doi.org/10.1145/3383313.3412248). So, what's the secret behind this powerful tool? Let's dive in.

## Why are there limits to traditional search methods?

In this digital age, organizations across the globe are grappling with an exponential surge in data. By 2025, the world is projected to have over 175 zettabytes of data, [source](https://www.forbes.com/sites/tomcoughlin/2018/11/27/175-zettabytes-by-2025). This data, both structured and unstructured, is a treasure trove of insights waiting to be unlocked. The key to this treasure chest? Personalized search using vector embeddings.

## What is a vector embedding?

Vector embeddings are revolutionizing the way we search and retrieve information. They work by converting data into a numerical representation, known as a vector. This process allows the search system to consider the semantics, or the underlying meaning, of the data when performing a search.

![Illustration of vector embeddings](../assets/use_cases/personalized_search/embeddings.png)

For instance, consider a user searching for a book on an online store. With traditional keyword-based search, the user would need to know the exact title or author's name. But with vector embeddings, the user could simply describe the book's theme or plot, and the search system would retrieve relevant results. This is because vector embeddings understand the 'meaning' of the query, rather than just matching keywords.

### How does it search?

The power of vector embeddings lies in their ability to quantify the similarity between two vectors. This is done using a distance metric, with cosine similarity being one of the most commonly used metrics. Simply put, cosine similarity measures how close two vectors are to each other, helping to determine how similar two pieces of data are. It returns relevant results even when the exact terms aren't present in the query. This is made possible by the general-purpose nature of vector embeddings, which can represent almost any form of data, from text to images to audio.

When vectorizing data, it's crucial to remember that embedding models have a limit on the maximum input length. The twelve best-performing models, based on the [Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/spaces/mteb/leaderboard), are limited to an input size of 512 tokens, while the 13th best has an exceptional input size limit of 8192 tokens.

For example, consider a user searching for a news article on a particular topic. If the article is longer than the model's maximum context length, the search system would need to split the article into smaller pieces or use a sliding window approach to capture the entire document in a single vector. This ensures that the search results are accurate and relevant, regardless of the length of the source document.

## But it's not only text that you can search!

Contrary to popular belief, vector embeddings are not only applicable to unstructured data. In fact, they have always been integral to deep learning models and can be used to derive insights from structured data as well.

For instance, consider a database of customer transactions. Each transaction can be represented as a vector, with each dimension representing a different attribute such as the transaction amount, date, or product category. By comparing these vectors, the search system can identify patterns or anomalies that would be difficult to spot with traditional search methods.

## Great! But what can I use to get started?

Using a vector database, which is a system designed to store and perform semantic search at scale, you can compare the query vector with the vectors stored in the database and return the top-k most similar ones. The key components of a vector database include a vector index, a query engine, partitioning/sharding capabilities, replication features, and an accessible API. Furthermore vector databases are categorized into vector-native, hybrid, and search engines. Notable vector database providers include [Pinecone](https://pinecone.io), [Milvus](https://milvus.io), and [Weaviate](https://weaviate.io).

| Key Components        | Descriptions                                            |
| --------------------- | ------------------------------------------------------- |
| Vector Index          | Allows fast and efficient retrieval of similar vectors  |
| Query Engine          | Performs optimized similarity computations on the index |
| Partitioning/Sharding | Enables horizontal scaling                              |
| Replication           | Ensures reliability and data integrity                  |
| API                   | Allows for efficient vector CRUD operations             |

## How can I use this for personalized search?

Before diving into various applications, let's illustrate how we might personalize a query using a simple code snippet:

```python
from transformers import BertTokenizer, BertModel
import torch

# Initialize an open-source embedding models
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

In this code example, we convert a search query into a vector using an [open-source, pretrained BERT model from Hugging Face](https://huggingface.co/bert-base-uncased) (You can also try this out online using the link). We also have a user preference vector, which would be usually based on a user's past clicks or choices. We then arithmetically "add" the query vector and the user preference vector to create a new query vector that reflects both the user input as well as user preferences.

![Use cases of personalized search with vector embeddings](../assets/use_cases/personalized_search/vector_space.png)

## Conclusions and next steps 😊

Vector embeddings are revolutionizing the way we interact with and use data. By enabling more accurate and contextually relevant search results, they are paving the way for a new era of data-driven insights and decision-making. Companies like Pinterest, Spotify, eBay, Airbnb, and Doordash are among the early adopters of vector search, integrating it into their platforms to enhance user experience and engagement [source](https://rockset.com/blog/introduction-to-semantic-search-from-keyword-to-vector-search/). Home Depot, for instance, saw the surge in online business due to the COVID pandemic as the perfect opportunity to implement vector search, suggesting significant value in this technology during periods of increased online activity [source](https://www.datanami.com/2022/03/15/home-depot-finds-diy-success-with-vector-search/). The future of search is here, and it's powered by vector embeddings.

So, what's next? How can you start implementing personalized search in your organization? There are plenty of resources and tools available to help you get started. For instance, you can check out this [guide on implementing vector search](link) or this [tutorial on using vector embeddings](link).

What are your thoughts on personalized search using vector embeddings? Have you used this technology in your organization? Share your experiences and join the conversation!

Stay Updated: Drop your email in the footer box to keep all the latest guides coming your way.

Found an Issue or Have a Suggestion? Your feedback shapes VectorHub! If you spot something off in the article or have a topic you want us to dive into, create a GitHub issue and we're happy to provide you with that information 😊.

---

## Author

- [Michael Jancen-Widmer](https://www.contrarian.ai) from Contrarian AI
