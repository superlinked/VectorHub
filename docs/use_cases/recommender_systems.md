<!--SEO SUMMARY: According to an IBM study, personalized product recommendations can lead to a 10-30% increase in revenue. However, creating truly personalized experiences requires extensive user and product data, which is not always available. For example, providers with restrictions on collecting user data may only have basic product information like genre and popularity. While collaborative filtering approaches can still work reasonably well, they leave significant gains on the table. Incorporating side data with a recommender based based on collaborative filtering can significantly increase recommendation quality (> 20% gains in precision), leading to greater customer satisfaction, more traffic, millions in additional revenue. -->

# A Recommender System: Collaborative Filtering with Sparse Metadata

![](assets/use_cases/recommender_systems/recommender.jpg)

Recommender Systems are becoming increasingly important, given the plethora of products offered to users/customers. Some twenty years ago fashion retailers developed basic versions of content-based recommenders that did increase users engagement (compared with the variant which offered no recommendations), but as soon as the capabilities of the tracking systems improved, 
it became possible to integrate new signals that could help provide even better recommendations. At the moment of this writing, fashion retailers use far more sophisticated approaches that use not only users' purchasing/viewing history, but also of user metadata (age, location, spending habits, mood, etc.) and item metadata (category, popularity, etc.)

However, it is sometimes the case that only scarce metadata about the users or the items is available to us. Public service providers, with their own on-demand audio and video platform, for example, are restricted in collecting user metadata. Such companies still have item data - genre, popularity, and so on - that can be used to improve the quality of the recommendations. Still, developers often disregard this side information, because it is scarce, and proceed with collaborative-filtering (CF) approaches, which make use of the historical interactions only (data consisting of user-item pairs). While CF work reasonably well in this use case - extrapolate user preferences via similarities of all users' browsing/purchasing history - can we improve the quality of the recommendations (increase users' engegement) using available side information even when it's scarce? More precisely, can we use the collaborative filtering paradigm in this case, with this scarce side info? Yes, there are libraries that allow us to ‘inject’ side information ([LightFM](https://making.lyst.com/lightfm/docs/home.html), for example). But there is also a way to extend the most efficient and effective collaborative filtering models, such as [ALS Matrix Factorization (MF)](http://yifanhu.net/PUB/cf.pdf) (the Python [‘implicit’](https://github.com/benfred/implicit) library) or [EASE] (https://arxiv.org/abs/1905.03375) (Embarassingly Shallow Autoencoder), to make use of the side information.


### Recommender Systems as Graphs

Matrix factorization is a common collaborative filtering approach. After a low-rank matrix approximation, we have two sets of vectors, one represeting the users and the other representing the items. The inner product of a user and item vector estimates the rating or interaction strength. We can view this process through a graph perspective. Users and items become graph nodes. Predicted ratings are edge weights between them. Recommendations identify the most likely new connections for a user. With both users and items being nodes of a (bipartite) graph, we can easily inject additional nodes as needed - for example, a "genre" node that links related items - this couples similar items, which improves recommendations.

Once the problem is understood as inherently a graph problem, it is easy to see how the model can be extended to include additional metadata. What we want to achieve is to somehow let the algorithm know about the new links, which would help group similar items or users together. In other words, we want to ‘inject’ new nodes, which would link the nodes belonging to the same group. How could this be achieved? Have a look at illustration below:

![](assets/use_cases/recommender_systems/dummy_nodes.jpg)

There are three users, u1, u2 and u3, and four items, i1, i2, i3, i4. The user U1 has interacted with items i1 and i3. Now, there is a Dummy User, who links the items that have the same color (i1, i3, i4). 
The dummy user couples similar items, helping the model identify related content. This increases the chances of item i4 to be recommended to user u1.

**Adaptation**

As discussed above, we only need to add dummy data: in case of only item side information available, add dummy users; in case of only user side information available, add dummy items; in case both the user and the item information is available, add both the dummy users and the dummy items. Obviously, the dummy nodes should not be part of recommendations; these nodes are only there to help ‘inject’ some ‘commonality’ of the nodes belonging to a certain group.

Beside MF, the same approach can be used with EASE, for example, or with an explicitly graph-based approach for recommendations such as [PageRank](https://scikit-network.readthedocs.io/en/latest/use_cases/recommendation.html). In fact, with PageRank, the walks would include the dummy nodes.

One question remains: which weights (ratings) to put for the dummy user interactions? We suggest you first start with low weights, and see how the quality of the recommendations changes.

**Minimal Code**

Here is some minimal Python code, demonstrating the addition of dummy users, one for each category:

``` python
import implicit
import numpy as np
import scipy.sparse as sp

# Creating a synthetic user-item interaction matrix
user_item = sp.coo_matrix((np.random.randint(0, 2, 100), (np.random.randint(0, 10, 100), np.random.randint(0, 10, 100))))

# Creating synthetic item categories
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

### Potential problems

Obviously, the addition of dummy nodes increases the computational and memory complexity, but in most cases this is a negligible compromise, given the scarcity of the side information. In an audio-on-demand platform, with 250K items in the catalog, there were a few hundred item categories only.

#### Real-world use case

We evaluated an ALS matrix factorization model with genre dummy users on an audio platform with over 250k items (genre such as "comedy", "documentary", etc.). Compared to their production system, 
it increased recommendation accuracy by over 10%.

#### Numerical data as side information

A natural question: what to do when the side information is numerical, not categorical? With numerical side information, we advise following one of the following two approaches::

1. Inject a dummy user, with scaled numerical values for weights
2. Analyze the distribution of the numerical data, and build categories based on value ranges


### Comparison with LightFM and Neural Collaborative Filtering

Injecting dummy nodes provides a lightweight way to improve recommendations when only limited side data is available. The simplicity of this approach is in keeping training fast and in preserving interpretability, compared to neural models. Dummy nodes are ideal for sparse categorical features like genre, but may underperform for dense numerical data. With rich side information, neural collaborative filtering is preferred, despite increased complexity. 

Overall, dummy nodes offer a transparent way to gain value from side data, with little cost when features are simple. They balance accuracy and speed for common cases. We recommend using this technique when you have sparse categorical metadata and want straightforward gains without major modeling changes.

---
## Contributors

- [Mirza Klimenta, PhD](https://www.linkedin.com/in/mirza-klimenta/)