<!--SEO SUMMARY: Recommender Systems can be improved by incorporating side information about users and items, even when this information is scarce. According to an IBM study, personalized product recommendations can lead to a 10-30% increase in revenue. However, creating truly personalized experiences requires extensive user and product data, which is not always available. For example, providers with restrictions on collecting user data may only have basic product information like genre and popularity. While collaborative filtering approaches can still work reasonably well, they leave significant gains on the table. By incorporating side data, no matter how limited, recommendation quality can potentially increase significantly (over 20% in precision), leading to greater customers's satisfaction, more traffic, millions in additional revenue. -->

# Recommender Systems

![](assets/use_cases/recommender_systems/recommender.jpg)

Recommender Systems are becoming increasingly important, given the plethora of products offered to the users/customers. For example, in early days, fashion retailers developed basic versions of content-based recommenders that served quite well, but soon the need emerged for more relevant recommendations that could increase user’s engagement. At the moment of this writing, fashion retailers use far more sophisticated approaches that make use not only of the user’s purchasing/viewing history, but also of user and item metadata: user age, location, spending habits, mood, etc.; item category, popularity, etc.

However, it is sometimes the case that only scarce side information about the users or the items is available to us. Consider, for example, a public service provider that has its own audio- and video-on-demand platform; such companies have restrictions with regard to collecting/storing user metadata. But even in this case, we still have useful information about the items that are served, such as genre, popularity, etc. Still, developers often disregard this side information, because it is scarce, and proceed with collaborative-filtering (CF) approaches, which make use of the historical interactions only (data consisting of user-item pairs). While CF work reasonably well in this use case - extrapolate user preferences via similarities of all-user’s browsing/purchasing history - one question remains: can we improve on the quality by (somewhow) making use of available information, regardless how scarce? More precisely, can we use the collaborative filtering paradigm in this case, with this scarce side info? Yes, there are libraries that allow us to ‘inject’ side information (LightFM, for example). But there is also a way to extend the most efficient and effective collaborative filtering models, such as ALS Matrix Factorization (MF) (the Python ‘implicit’ library) or EASE, to make use of the side information.


### Recommender Systems as Graphs

Matrix factorization is a common collaborative filtering approach. It represents users and items as vectors in a shared latent space. The inner product of a user and item vector estimates the rating or interaction strength. We can view this process through a graph perspective. Users and items become graph nodes. Predicted ratings are edge weights between them. Recommendations identify the most likely new connections for a user. With this graph view, we can easily inject additional nodes as needed. For example, a "genre" node that links related items. This couples similar items which improves recommendations.

(We deliberately took a detour and observed the problem via graphs. Note that this approach is known as ‘joint matrix factorization’ in the literature).

Once the problem is understood as inherently a graph problem, it is easy to see how additional information can be added. What we want to achieve is to somehow let the algorithm know about the new links, which would help group similar items or users together. In other words, we want to ‘inject’ new nodes, which would link the nodes belonging to the same group. How could this be achieved? Have a look at illustration below:

![](assets/use_cases/recommender_systems/dummy_nodes.jpg)

There are three users, U1, U2 and U3, and four items, I1, I2, I3, I4. The user U1 has interacted with items I1 and I3. Now, there is a Dummy User, who links the items that have the same color (I1, I3, I4). 
The dummy user couples similar items, helping the model identify related content. This increases the chances of item I4 to be recommended to user U1.

#### Adaptation

As discussed above, we only need to add dummy data: in case of only item side information available, add dummy users; in case of only user side information available, add dummy items; in case both the user and the item information is available, add both the dummy users and the dummy items. Obviously, the dummy nodes should not be part of recommendations; these nodes are only there to help ‘inject’ some ‘commonality’ of the nodes belonging to a certain group.

Beside MF, the same approach can be used with EASE, for example, or with an explicitly graph-based approach for recommendations such as PageRank. In fact, with PageRank, the walks would include the dummy nodes.

One question remains: which weights (ratings) to put for the dummy user interactions? We suggest you first start with low weights, and see how the quality of the recommendations changes.

### Potential problems

Obviously, the addition of dummy nodes increases the computational and memory complexity, but in most cases this is a negligible compromise, given the scarcity of the side information. In an audio-on-demand platform, with 250K items in the catalog, there were a few hundred item categories only.

#### Real-world use case

We evaluated an ALS matrix factorization model with genre dummy users on an audio platform with over 250k items (genre such as "comedy", "documentary", etc.). Compared to their production system, 
it increased recommendation accuracy by over 10%.

#### Numerical data as side information

A natural question: what to do in case the side information is numerical, not categorical? In this case we advise either of the following two:

1. Inject a dummy user, with weights being scaled numerical values
2. Analyze the distribution of the numerical data, and build categories based on value ranges


### Comparison with LightFM and Neural Collaborative Filtering

Injecting dummy nodes provides a lightweight way to improve recommendations when only limited side data is available. The simplicity keeps training fast and preserves interpretability compared to neural models. Dummy nodes are ideal for sparse categorical features like genre, but may underperform for dense numerical data. With rich side information, neural collaborative filtering would be preferred despite increased complexity. Overall, dummy nodes offer a transparent way to gain value from side data, with little cost when features are simple. They balance accuracy and speed for common cases. You would choose this technique when you have sparse categorical metadata and want straightforward gains without major modeling changes.

---
## Contributors

- [Mirza Klimenta, PhD](https://www.linkedin.com/in/mirza-klimenta/)