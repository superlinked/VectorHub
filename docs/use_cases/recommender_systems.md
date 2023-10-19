<!--SEO SUMMARY: Personalized product recommendations, according to a study by IBM, can lead to a 10-30% increase in revenue. But creating truly personalized experiences requires extensive user and product data, which is not always available. For example, providers with restrictions on collecting user data may only have basic product information, like genre and popularity. While collaborative filtering approaches work reasonably well when extensive user and product data is scarce, they leave significant gains on the table. Incorporating side data with a recommender based on collaborative filtering can significantly increase recommendation quality (> 20% gains in precision), leading to greater customer satisfaction, more traffic, and millions in additional revenue.-->

# A Recommender System: Collaborative Filtering with Sparse Metadata

![](assets/use_cases/recommender_systems/recommender.jpg)

Recommender Systems are increasingly important, given the plethora of products offered to users/customers. Beginning approximately twenty years ago, fashion retailers developed basic versions of content-based recommenders that did increase user engagement (compared with the no-recommendations approach). But when the capabilities of event-tracking systems improved, it became possible to integrate new signals that could help provide even better recommendations. At the moment of this writing, fashion retailers have adopted more sophisticated recommendation systems that ingest not only users' purchasing/viewing history but also user metadata (age, location, spending habits, mood, etc.) and item metadata (category, popularity, etc.).
​
But not everyone has access to the kind or amount of metadata fashion retailers do. Sometimes only scarce side is available. Public service providers with their own on-demand audio and video platform, for example, are legally restricted in collecting user metadata. Typically, they use collaborative filtering (CF) approaches, which employ historical interactions (data consisting of user-item pairs) to extrapolate user preferences via similarities of all users' browsing/purchasing history. Such companies still have item data - genre, popularity, and so on - that can be used to improve the quality of recommendations. Developers often disregard this side information because it is scarce. While CF works reasonably well in this use case - i.e., extrapolating user preferences via similarities of all users' browsing/purchasing history, we can improve the recommendation quality (thereby increasing user engagement) of CF by adding available side information, even if it's scarce. More precisely, there are libraries that allow us to ‘inject’ side information ([LightFM](https://making.lyst.com/lightfm/docs/home.html), for example). Even the most efficient and effective collaborative filtering models, such as [ALS Matrix Factorization (MF)](http://yifanhu.net/PUB/cf.pdf) (the Python [‘implicit’](https://github.com/benfred/implicit) library) or [EASE] (https://arxiv.org/abs/1905.03375) (Embarrassingly Shallow Autoencoder), can be extended and improved using side information.

## Recommender systems as graphs

Matrix factorization is a common collaborative filtering approach. After a low-rank matrix approximation, we have two sets of vectors, one representing the users and the other representing the items. The inner product of a user and item vector estimates the rating this user gave to this particular item. We can represent this process using a graph. In this graph, users and items are graph nodes, and predicted ratings are edge weights between them. The graph is bipartite, meaning that links appear only between nodes belonging to different groups. The list of recommendations made to a user correspond to the most likely new item-connections for this user-node. We can easily represent side info injection as graph nodes - for example, a "genre" node that links related items.
​
Using this graphical representation, it is easy to see how matrix factorization can be extended to include additional metadata. What we want is to somehow let the algorithm know about the new links, which help group similar items or users together. In other words, we want to "inject" new nodes – nodes that link the nodes belonging to the same group. How do we do this? Have a look at the illustration below:

![](assets/use_cases/recommender_systems/dummy_nodes.jpg)

There are three users: u1, u2 and u3; and four items: i1, i2, i3, i4. The user u1 has interacted with items i1 and i3. There is a dummy user, who links items that have the same color (i1, i3, i4). By coupling similar items, the dummy user helps the model identify related content. This increases the chances of item i4 being recommended to user u1.

**Adaptation**

We need only add dummy data to enrich a CF approach: when only item side information is available, add dummy users; when only user side information is available, add dummy items; when both user and item information is available, add both dummy users and dummy items. Obviously, the dummy nodes should not be included in the recommendations; these nodes are only there to help ‘inject’ some ‘commonality’ of the nodes belonging to a certain group.
​
The same approach (as used above with MF) can be used with, for example, EASE, or with an explicitly graph-based approach for recommendations such as [PageRank](https://scikit-network.readthedocs.io/en/latest/use_cases/recommendation.html). In fact, with PageRank, the walks would include the dummy nodes.
​
A question remains: how should dummy user interactions be weighted (rated)? We suggest you first start with low weights, see how recommendation quality changes, and iteratively adjust your weights to fine-tune.

**Minimal Code for Adding Dummy Users**

Here is some minimal Python code demonstrating the addition of dummy users, one for each category:

``` python
import implicit
import numpy as np
import scipy.sparse as sp

# Create a synthetic user-item interaction matrix
user_item = sp.coo_matrix((np.random.randint(0, 2, 100), (np.random.randint(0, 10, 100), np.random.randint(0, 10, 100))))

# Create synthetic item categories
item_categories = ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A']

# Create dummy users for each unique category
unique_categories = list(set(item_categories))
dummy_users = {category: i + user_item.shape[0] for i, category in enumerate(unique_categories)}

# Extend the user-item matrix with dummy interactions
extended_user_item = sp.vstack([
    user_item,
    sp.coo_matrix((len(unique_categories), user_item.shape[1]))
]).tocsr()  # Ensure the matrix is in CSR format

# Add dummy interactions
# For each item, add an interaction with its corresponding dummy user
rows, cols, data = [], [], []
for item_idx, category in enumerate(item_categories):
    rows.append(dummy_users[category])
    cols.append(item_idx)
    data.append(1)  # or another value indicating the strength of the dummy interaction

# Add dummy interactions to the extended_user_item matrix
dummy_interactions = sp.coo_matrix((data, (rows, cols)), shape=extended_user_item.shape)

print(rows, cols, data)

extended_user_item += dummy_interactions

print(extended_user_item)

# Fit the ALS model
model = implicit.als.AlternatingLeastSquares(factors=50)
model.fit(extended_user_item.T)  # Transpose to item-user format as implicit library expects item-user matrix

# Generate recommendations
# Note: Ensure not to recommend dummy users
user_id = 0  # Example user ID
recommended_items = model.recommend(user_id, extended_user_item[user_id], N=5, filter_already_liked_items=True)

recommended_items
```

## A real world use case

We evaluated an ALS matrix factorization model with genre dummy users on an audio-on-demand platform with over 250k items (genres such as "comedy", "documentary", etc.). Compared to their production system, adding dummy nodes increased recommendation accuracy by over 10%. Obviously, the addition of dummy nodes increases computational and memory complexity, but in most cases this is a negligible compromise, given the scarcity of side information. Though the platform we evaluated had over 250K items in the catalog, there were only a few hundred item categories.

## Numerical data as side information

A natural question: what to do when the side information is numerical, not categorical? With numerical side information, we advise pursuing one of the following two approaches:

1. Inject a dummy user, with scaled numerical values for weights
2. Analyze the distribution of the numerical data, and build categories based on value ranges

## When to use side-data-injected CF vs. neural CF

Injecting dummy nodes provides a lightweight way to improve recommendations when only limited side data is available. The simplicity of this approach, compared to neural models, lies in keeping training fast and preserving interpretability. Dummy nodes are ideal for sparse categorical features like genre, but may underperform for dense numerical data. With rich side information, neural collaborative filtering is preferred, despite increased complexity.

Overall, dummy nodes offer a transparent way to gain value from side data, with little cost when features are simple. They balance accuracy and speed for common cases. We recommend using this technique when you have sparse categorical metadata and want straightforward gains without major modeling changes.

---
## Contributors

- [Mirza Klimenta, PhD](https://www.linkedin.com/in/mirza-klimenta/)
