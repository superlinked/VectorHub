# Why You Don't Need Re-Ranking: Understanding the Superlinked Vector Layer

## Takeaways

- Superlinked removes the need for re-ranking by combining semantic and
  numeric signals in a single vector index.

- Dynamic query-time weighting in Superlinked allows for real-time
  adjustments without the need to re-embed data.

- Superlinked simplifies search by unifying text, number, and category
  data into one index, unlike traditional systems that use multiple
  indices.

- Hard filtering before search boosts precision and speed, unlike
  traditional post-search filtering.

- Superlinked reduces latency and computational costs, providing faster
  and more efficient search results.

***Meta Title:** Superlinked Boosts Search Speed Without Re-Ranking
Steps*

***Meta Description:** Discover how Superlinked eliminates the need for
re-ranking in vector search systems. Learn about its unified multimodal
vectors, dynamic intent capture that improve search relevance, speed,
and scalability.*

When it comes to vector search, it's not just about matching words.
Understanding the meaning behind them is equally important. But there
are challenges. Sometimes, factors like text meaning, popularity, and
recency can lead to results that aren't quite right. This is because
vector search isn't always perfect at making precise matches.

To fix this, many systems use a technique called **re-ranking**. This
involves using a separate model that processes both the query and the
initial results together to reorder them based on their relevance to the
query. However, this process comes with its own set of problems. It
takes up a lot of computing power and slows things down, especially when
working with large datasets.

[Now, imagine if your search infrastructure was smarter even before you
hit the database. The key idea here is that with Superlinked, your
search system can understand what you want and adjust accordingly. This
reduces the need for re-ranking altogether. Superlinked improves search
results by embedding multiple signals directly into the search index.
This makes the results more relevant, faster, and more efficient, all
without the need for extra steps.]{.mark}

This article discusses how Superlinked eliminates the need for
re-ranking by embedding multiple signals directly into unified vector
spaces.

## Understanding vector re-ranking in modern search systems

Vector re-ranking improves initial search results by adding a secondary
scoring process. It uses neural networks or cross-encoders to prioritize
documents that are most relevant to the context. Unlike the initial
vector search, which converts documents into vectors before querying,
re-ranking processes the query and documents together. It then reorders
the results more precisely based on relevance. This method is commonly
used in [Retrieval-Augmented Generation
(RAG)](https://superlinked.com/vectorhub/articles/retrieval-augmented-generation)
pipelines and semantic search systems to fill the gaps of traditional
retrieval methods like vector similarity search.

However, as with everything, reranking comes with challenges. This
process often involves cross-encoders which compute similarity scores
for each query-document pair individually and may require augmenting
results with domain-specific metadata. While this increases precision,
it also adds latency and computing costs. These factors can be
challenging in production environments, particularly in RAG and
[semantic search
systems](https://github.com/superlinked/superlinked/blob/main/notebook/semantic_search_news.ipynb).

## Superlinked's approach to eliminating re-ranking

Superlinked is a Python framework designed for AI engineers to build
high-performance search and recommendation systems that unify structured
and unstructured data into multi-modal vectors. It bridges the gap
between data and vector databases by using a mixture of encoders to
combine text semantics, numerical ranges, and categorical attributes
into unified embeddings. This eliminates the need for post-retrieval
re-ranking.

Superlinked's core innovation lies in these abilities:

1.  **Mixture of encoders at index time**

2.  **Dynamic intent capture at query time**

3.  **Hard filtering before vector search**

![](media/unified.png)

<p align="center">Unified vector search without re-ranking</p>

### Mixture of encoders at index time

Superlinked addresses the need for re-ranking by embedding data across
different modalities, such as text and numerical attributes, into
separate vectors. These vectors are then combined into a single
[multimodal
vector](https://docs.superlinked.com/concepts/multiple-embeddings),
ensuring that all aspects of the data are captured.

For example, paragraphs and their like_count values are embedded
separately using TextSimilaritySpace and NumberSpace, then combined into
a single index:

<pre><code class="language-python">
# Define Spaces for text and numerical data

body_space = TextSimilaritySpace(text=paragraph.body, model="sentence-transformers/all-mpnet-base-v2")  
like_space = NumberSpace(number=paragraph.like_count, min_value=0, max_value=100)  
paragraph_index = Index([body_space, like_space])  # Unified multimodal vector
</code></pre>

This approach streamlines search processes by preserving structured
attributes like popularity metrics alongside unstructured text
semantics, improving retrieval relevance without additional layers or
complex filtering.

### Dynamic intent capture at query time

Superlinked also enables us to fine-tune search relevance through
dynamic [query-time
weighting](https://docs.superlinked.com/concepts/dynamic-parameters).
For example, a query can prioritize text similarity over "like counts"
by adjusting weights at runtime without re-embedding data:
<pre><code class="language-python">

# Weight text similarity 2x more than likes

body_query = Query(index, weights={body_space: 1.0, like_space: 0.5})

</code></pre>

Superlinked optimizes our search by allowing dynamic control over
assigning weights to attributes. All weights are kept on the query-side
vector. This allows us to prioritize what matters most based on our use
case without the overhead of re-embedding or re-indexing.

Superlinked supports two flexible methods for applying weights:

1.  **At query definition:** Since each attribute, like text or numbers,
    is embedded separately, we can assign specific weights to them when
    defining the query. This means we can experiment with what matters
    most for our search without re-embedding or changing our dataset.

2.  **At query execution:** We can define placeholder parameters within
    the query and fill them in dynamically at runtime. This gives us the
    ability to adjust weights dynamically, offering more control over
    what's considered important even after the query has been defined.

![](media/weights.png)

<p align="center">Two ways to weight the query</p>

### Hard filtering before vector search

[Hard
filtering](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/feature/hard_filtering.ipynb)
is essential when we need to ensure that certain items are excluded from
our results. In Superlinked, this can be achieved using the ".filter"
within a query:

<pre><code class="language-python">
# Exclude paragraphs by author "Adam" and enforce minimum length

query = (  
    Query(index)  
    .find(paragraph)  
    .filter((paragraph.author != "Adam") & (paragraph.length > 300))  
)
</code></pre>

This allows filtering out paragraphs based on specific authors,
excluding items below length thresholds, and combining conditions using
logical operators.

Superlinked also supports set operations. This helps filter results by a
list of possible values or exclude results from a list.

<pre><code class="language-shell">
# Filter results containing BOTH "fresh" and "useful" tags
query.filter(paragraph.tags.contains_all(["fresh", "useful"]))
</code></pre>

Hard filtering eliminates the need for time-consuming reranking later by
retrieving only the most relevant items based on defined criteria.

## A product search use case

To show how traditional reranking compares to Superlinked's unified
multi-space querying, let's compare how each handles the query:

"Find affordable wireless headphones with noise cancellation under \$200
and high ratings."

Both methods capture semantic similarity using transformer-based
embeddings. But they handle metadata like price, rating, and category
very differently. The traditional approach uses a reranker model that
scores only the text, and we have to write extra logic to blend in
metadata like price and rating afterward. Superlinked makes it easier by
natively supporting multimodal scoring, letting us assign dynamic
weights at query time, and apply hard filters directly.

### Installations

We begin by installing the necessary packages.

<pre><code class="language-shell">
# Install required packages

!pip install rerankers
!pip install superlinked
</code></pre>

### Imports

We import the core libraries needed for both the traditional reranking
approach and the Superlinked-based approach.

<pre><code class="language-python">
# Imports

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rerankers import Reranker
from superlinked import framework as sl
</code></pre>

### Dataset

We define a product dataset with text descriptions, numerical price and
rating fields, and a categorical field for category.

<pre><code class="language-python">
# Sample product data

products = [
    {
        "id": "p1",
        "title": "Premium Wireless Headphones",
        "description": "High-end wireless headphones with active noise cancellation (ANC), 30hr battery. Original price $350, now discounted to $199.",
        "price": 199,
        "rating": 4.8,
        "category": "electronics"
    },
    {
        "id": "p2",
        "title": "Budget Noise-Canceling Earbuds",
        "description": "Affordable wireless earbuds with basic noise cancellation. 20hr battery. Ideal for casual use.",
        "price": 89,
        "rating": 4.2,
        "category": "electronics"
    },
    {
        "id": "p3",
        "title": "Studio-Grade ANC Headphones",
        "description": "Professional noise-canceling headphones with Hi-Res audio. Priced at $210.",
        "price": 210,
        "rating": 4.7,
        "category": "electronics"
    }
]
</code></pre>

### Define query and embed descriptions

We initialize the SentenceTransformer model to compute semantic
embeddings for product descriptions. This is used in both the
traditional and Superlinked pipelines.

<pre><code class="language-python">
# User query and embedding with SentenceTransformer

query_text = "Find affordable wireless headphones with noise cancellation under $200 and high ratings"

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

embeddings = model.encode([p["description"] for p in products])
</code></pre>

### Traditional approach

We use a reranker model (mxbai-rerank-large-v1) from the rerankers
library. This model only takes query and documents, not metadata, and
returns a ranked list based on text relevance.

<pre><code class="language-python">
# Initialize reranker

ranker = Reranker("mixedbread-ai/mxbai-rerank-large-v1")

reranked = ranker.rank(
    query=query_text,
    docs=[p["description"] for p in products],
    doc_ids=[p["id"] for p in products]
)
</code></pre>


#### Process results with metadata

We combine reranker output with metadata by weighting the reranker score
and product rating. We also apply a filter to exclude products priced
over \$200.

<pre><code class="language-python">
# Reranker-based hybrid ranking with price filtering

def process_traditional_results(reranked_output, max_price=200):
    results = []
    
    for doc in reranked_output.top_k(len(products)):
        product = next(p for p in products if p["id"] == doc.doc_id)
        
        # Hard filter on price
        if product["price"] > max_price:
            continue
        
        # Combine reranker score with rating
        combined_score = (doc.score * 0.6) + (product["rating"] / 5 * 0.4)
        
        results.append({
            "title": product["title"],
            "price": product["price"],
            "rating": product["rating"],
            "reranker_score": doc.score,
            "final_score": combined_score
        })
    
    return pd.DataFrame(results).sort_values("final_score", ascending=False)

# Display results
traditional_df = process_traditional_results(reranked)

print("\nTraditional Approach Results:")
display(traditional_df)
</code></pre>

#### Results

![](media/traditionalresults.png)

### Superlinked approach

We define a structured schema for products and create multiple
similarity spaces:

- Text similarity

- Numeric similarity

- Categorical similarity

<pre><code class="language-python">
# Define product schema for Superlinked
@sl.schema
class Product:
    id: sl.IdField
    title: sl.String
    description: sl.String
    price: sl.Integer
    rating: sl.Float
    category: sl.String

product = Product()

# Define similarity spaces
text_space = sl.TextSimilaritySpace(
    text=product.description,
    model="sentence-transformers/all-mpnet-base-v2"
)

price_space = sl.NumberSpace(
    number=product.price,
    mode=sl.Mode.MINIMUM,
    min_value=0,
    max_value=500
)

rating_space = sl.NumberSpace(
    number=product.rating,
    mode=sl.Mode.MAXIMUM,
    min_value=0,
    max_value=5
)

category_space = sl.CategoricalSimilaritySpace(
    category_input=product.category,
    categories=["electronics", "fashion", "home", "sports", "books"],
    negative_filter=-1.0,
    uncategorized_as_category=False
)
</code></pre>


#### Superlinked index and executor setup

We configure the Superlinked index using the defined spaces. The
InMemorySource and Executor manage querying and updates.

<pre><code class="language-python">
# Create index with filterable fields
product_index = sl.Index(
    [text_space, price_space, rating_space, category_space],
    fields=[product.category, product.price]
)

# Load source and run executor
source = sl.InMemorySource(product)

executor = sl.InMemoryExecutor(
    sources=[source],
    indices=[product_index]
)

app = executor.run()

source.put(products)
</code></pre>


#### Superlinked query with weights and filters

We construct a unified query with dynamic weights assigned to text,
price, and rating spaces. We also apply hard filters to enforce price is
below 200\$ and category falls is limited to electronics.

<pre><code class="language-python">
# Unified multimodal query with dynamic weights and hard filters
query = (
    sl.Query(product_index, weights={
        text_space: 0.5,
        price_space: 0.3,
        rating_space: 0.2
    })
    .find(product)
    .similar(text_space.text, query_text)
    .filter(product.category == "electronics")
    .filter(product.price <= 200)
    .select_all()
)

# Execute query
result = app.query(query)

# Display results as DataFrame
sl.PandasConverter.to_pandas(result)
</code></pre>


#### Results

![](media/superlinkedresults.png)

This use case shows how Superlinked makes search simpler and more
powerful. It gives you a unified way to handle search, whereas in the
traditional setup, you have to manually piece together reranking logic,
filters, and weights. You also save on compute since there's no need to
re-embed or re-rank results multiple times.

## Benefits of Superlinked's vector layer over vector re-ranking

Here\'s a quick comparison between Superlinked and traditional
re-ranking systems across key features to show how they differ in
performance, usability, and scalability.

  | **Feature**                         | **Superlinked Approach**                                                                                                                                                   | **Traditional Re-ranking**                                                                                                                        |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Unified Multi-space Indexing**    | Uses a mixture of encoders to combine multiple data types (text, numbers, categories) into a single index, enabling simultaneous consideration during search.             | Requires separate indices for each data type during re-ranking.                                                                                     |
| **Dynamic Query-time Weighting**    | Allows adjustment of the importance of different embedding components during query execution without re-embedding the dataset.                                            | Adjustments require reprocessing or re-embedding data, leading to increased complexity and latency.                                               |
| **Hard Filtering Before Search**    | Applies business rules and filters prior to vector search, reducing the search space and improving performance.                                                            | Filtering is typically applied post-search, which can be less efficient and may return less relevant results.                                     |
| **Event-driven Personalization**    | Supports real-time personalization by updating user vectors based on behavioral events, enhancing recommendation relevance.                                               | Personalization often requires batch processing and lacks real-time adaptability.                                                                 |
| **VectorSampler Utility**          | Provides direct access to export vectors from the index into a VectorCollection (NumPy array and ID list), facilitating analysis and debugging.                          | Lacks built-in tools for easy extraction and analysis of vector data.                                                                             |
| **Integration with Vector Databases** | Compatible with various vector databases like Redis, Qdrant, and MongoDB, allowing flexible deployment options.                                                           | Integration often requires custom connectors and additional configuration.                                                                         |
| **Performance and Scalability**     | Optimized for high-throughput and low-latency scenarios, capable of handling complex, multi-dimensional data efficiently.                                                 | Performance can degrade with increased data complexity and volume due to processing overhead.                                                     |
| **Ease of Use and Flexibility**     | Provides a Python framework with intuitive APIs, reducing boilerplate code and simplifying the development of advanced search and recommendation systems.                | Development often involves extensive boilerplate code and complex configurations.                                                                 |


## Conclusion

Modern users expect fast, intelligent responses to complex queries that
blend semantics with filters and real-world context. The real challenge
in traditional vector search isn't just poor re-ranking; it's weak
initial retrieval. If the first layer of results misses the right
signals, no amount of re-sorting will fix it. That's where Superlinked
changes the game.

Superlinked combines structured and unstructured data into unified
multimodal vectors, enabling the most relevant results to surface early
without re-ranking. This enhances both the accuracy and the speed of
search systems, making your search infrastructure smarter before you
even hit the database.

Ready to upgrade your search and recommendations? Explore
[Superlinked](https://superlinked.com/) and build smarter,
faster results with real-world data.
