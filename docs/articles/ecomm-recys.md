# Creating personalized, real-time recommendation systems - a [notebook](https://github.com/superlinked/superlinked/blob/main/notebook/recommendations_e_commerce.ipynb) article

Pioneered by the likes of Google and AirBnB, vector embedding-powered recommendation systems are increasingly popular. In this article, we will walk you through how to use the Superlinked library to create a RecSys that provides more relevant matches, and can be updated in real-time, with feedback loops defined by user interactions.

Vector embeddings have revolutionized recommendation systems by representing users and items as high-dimensional vectors in a latent space, capturing similarities and relationships. This approach surpasses traditional methods reliant on sparse categorical data, enabling more accurate and personalized recommendations. Their compact and dense nature facilitates efficient computation and scalability, which is vital for real-time and large-scale scenarios. 

## Challenges of using vector embeddings for recommendations

Using vector embeddings in building recommendation systems presents several challenges that need to be addressed for effective implementation. These challenges include:
- Quality and relevance: Embedding generation process, architecture, and data must be carefully considered.
- Sparse and noisy data: Embeddings may struggle with incomplete or noisy input. Sparse data is the crux of the cold-start problem.
- Scalability: Efficient methods for large datasets are needed; otherwise latency will be an issue.

Superlinked allows you to combine all the different pieces of information you have about an entity into a rich multimodal vector, making it easier for you capture what is important to users when making recommendations.

## How Superlinked helps you create hyper-personalized experiences users love

RecSys personalisation can be challenging because, typically, the available user-specific data is sparse. Using the Superlinked library, you can capture events data and use this information to create a feedback loop that lets you update vectors in real time to reflect user preferences.

In the example below, you’ll use the following elements of the Superlinked library to build your recommendation system for e-commerce:
min_max number spaces: for understanding customer reviews and pricing information
text-similarity space: for semantic understanding of product information
Events schema and Effects to modify vectors 
Query time weights - to define how you want the data to be treated when you run the query, avoiding the need to re-embed the whole dataset to optimize your experiment

This approach can tackle both the cold start problem and personalize recommendations. 

## Building an e-commerce recommendation engine with Superlinked

We’re building a recommender system for an e-commerce site that sells mainly clothing.

To start, we have the following product deta
- number of reviewers
- product ratings
- textual description
- name of product (usually containing brand name)
- category
  
We have two users (user_1 and user_2) whom we can differentiate by either:
- which of three products offered during registration they choose, or
- more general characteristics explained in the below paragraph (price, reviews)
 
Users have preferences in textual characteristics of products (description, name, category), and (according to classical economics) ceteris paribus prefer products that:
- cost less
- have a lot of reviews
- have higher ratings

We’ll set up our spaces up to reflect that.

This initial setup can be used in cold-start scenarios - we’ll recommend items for users we know very little about. [only their on-registration product choice?]

Once our RecSys is up and running, we’ll have behavioral data. Users will click on certain products, buy certain products, etc. We can capture this events data, and use it to create feedback loops, updating our vectors to reflect user preferences and improving the quality of recommendations. 

### Setting up Superlinked

First, we need to install the library and import the classes.

```python
%pip install superlinked==6.0.0

import altair as alt
import os
import pandas as pd
import sys


from superlinked.framework.common.embedding.number_embedding import Mode
from superlinked.framework.common.schema.schema import schema
from superlinked.framework.common.schema.event_schema import event_schema
from superlinked.framework.common.schema.schema_object import String, Integer
from superlinked.framework.common.schema.schema_reference import SchemaReference
from superlinked.framework.common.schema.id_schema_object import IdField
from superlinked.framework.common.parser.dataframe_parser import DataFrameParser
from superlinked.framework.dsl.executor.in_memory.in_memory_executor import (
   InMemoryExecutor,
   InMemoryApp,
)
from superlinked.framework.dsl.index.index import Index
from superlinked.framework.dsl.index.effect import Effect
from superlinked.framework.dsl.query.param import Param
from superlinked.framework.dsl.query.query import Query
from superlinked.framework.dsl.source.in_memory_source import InMemorySource
from superlinked.framework.dsl.space.text_similarity_space import TextSimilaritySpace
from superlinked.framework.dsl.space.number_space import NumberSpace


alt.renderers.enable("mimetype")
pd.set_option("display.max_colwidth", 190)
```

Now that the library's installed and classes imported, we can explore the dataset. Per above, initial user preference data comes from the users' 

```python
# the user preferences come from the user being prompted to select a product out of 3 - those will be the initial preferences
# this is done in order to give somewhat personalised recommendations
user_df: pd.DataFrame = pd.read_json(USER_DATASET_URL)
user_df

NROWS = int(os.getenv("NOTEBOOK_TEST_ROW_LIMIT", sys.maxsize))
products_df: pd.DataFrame = (
   pd.read_json(PRODUCT_DATASET_URL)
   .reset_index()
   .rename(columns={"index": "id"})
   .head(NROWS)
)
# convert price data to int
products_df["price"] = products_df["price"].astype(int)
print(products_df.shape)
products_df.head()
```


We have data on which of three products user_1 and user_2 chose on registration:

```python
# the user preferences come from the user being prompted to select a product out of 3 - those will be the initial preferences
# this is done in order to give somewhat personalised recommendations
user_df: pd.DataFrame = pd.read_json(USER_DATASET_URL)
user_df
```

![User product pref at registration](..assets/use_cases/ecomm_recsys/user_product_pref-at_registration.png)

We all have distribution data showing how many products are at different price points, have different review counts, and have different ratings.



### Building out the index for vector search

Superlinked’s library contains a set of core building blocks that we use to construct the index and manage the retrieval. You can read about these building blocks in more detail [here](https://github.com/superlinked/superlinked/blob/main/notebook/feature/basic_building_blocks.ipynb).

Let’s put this library’s building blocks to use in our EComm RecSys.

First you need to define your Schema to tell the system about your data. 

```python
# schema is the way to describe the input data flowing into our system - in a typed manner
@schema
class ProductSchema:
   description: String
   name: String
   category: String
   price: Integer
   review_count: Integer
   review_rating: Integer
   id: IdField




@schema
class UserSchema:
   preference_description: String
   preference_name: String
   preference_category: String
   id: IdField




@event_schema
class EventSchema:
   product: SchemaReference[ProductSchema]
   user: SchemaReference[UserSchema]
   event_type: String
   id: IdField


# we instantiate schemas
product = ProductSchema()
user = UserSchema()
event = EventSchema()
```

Next, you use Spaces to say how you want to treat each part of the data when embedding. In space definitions, we describe how to embed inputs so that they reflect the semantic relationships in our data. Each Space is optimized to embed the data so as to return the highest possible quality of retrieval results. Which Spaces are used depends on your datatype.

```python
# textual inputs are embedded in a text similarity space powered by a sentence_transformers model
description_space = TextSimilaritySpace(
   text=[user.preference_description, product.description],
   model="sentence-transformers/all-distilroberta-v1",
)
name_space = TextSimilaritySpace(
   text=[user.preference_name, product.name],
   model="sentence-transformers/all-distilroberta-v1",
)
category_space = TextSimilaritySpace(
   text=[user.preference_category, product.category],
   model="sentence-transformers/all-distilroberta-v1",
)


# NumberSpaces encode numeric input in special ways to reflect a relationship
# here we express relationships to price (lower the better), or ratings and review counts (more/higher the better)
price_space = NumberSpace(
   number=product.price, mode=Mode.MINIMUM, min_value=25, max_value=1000
)
review_count_space = NumberSpace(
   number=product.review_count, mode=Mode.MAXIMUM, min_value=0, max_value=100
)
review_rating_space = NumberSpace(
   number=product.review_rating, mode=Mode.MAXIMUM, min_value=0, max_value=4
)
# create the index using the defined spaces
product_index = Index(
   spaces=[
       description_space,
       name_space,
       category_space,
       price_space,
       review_count_space,
       review_rating_space,
   ]
)
# parse our data into the schemas - not matching column names can be conformed to schemas using the mapping parameter
product_df_parser = DataFrameParser(schema=product)
user_df_parser = DataFrameParser(
   schema=user, mapping={user.preference_description: "preference_desc"}
)


# setup our application
source_product: InMemorySource = InMemorySource(product, parser=product_df_parser)
source_user: InMemorySource = InMemorySource(user, parser=user_df_parser)
executor: InMemoryExecutor = InMemoryExecutor(
   sources=[source_product, source_user], indices=[product_index]
)
app: InMemoryApp = executor.run()


# load the actual data into our system
source_product.put([products_df])
source_user.put([user_df])
```

Now that you’ve got your data defined in Spaces, you’re ready to play with your data and optimize the results.

### Tackling the RecSys cold-start problem

Let's first showcase what we can do without events - our cold-start solution.

Here we define a user query that just searches with the user's preference vector configuration options are the importance (weights) of each input type (space)

```python
user_query = (
   Query(
       product_index,
       weights={
           description_space: Param("description_weight"),
           name_space: Param("name_weight"),
           category_space: Param("category_weight"),
           price_space: Param("price_weight"),
           review_count_space: Param("review_count_weight"),
           review_rating_space: Param("review_rating_weight"),
       },
   )
   .find(product)
   .with_vector(user, Param("user_id"))
   .limit(Param("limit"))
)
# simple recommendations for our user_1
# these are only based on the initial product the user chose when first entering our site
simple_result = app.query(
   user_query,
   user_id="user_1",
   description_weight=1,
   name_weight=1,
   category_weight=1,
   price_weight=1,
   review_count_weight=1,
   review_rating_weight=1,
   limit=TOP_N,
)


simple_result.to_pandas()
```



We can also just give the user products that generally seem appealing. For example, items with a low price, and a lot of good reviews (we can play around with the weights to tune those relationships, too)

```python
general_result = app.query(
   user_query,
   user_id="user_1",
   description_weight=0,
   name_weight=0,
   category_weight=0,
   price_weight=1,
   review_count_weight=1,
   review_rating_weight=1,
   limit=TOP_N,
)


general_result.to_pandas() 
```



Then you can optimise further by giving additional weight to the category space, to reccomend more “women clothing jackets” products.

```python
women_cat_result = app.query(
   search_query,
   user_id="user_1",
   query_text="women clothing jackets",
   description_weight=1,
   name_weight=1,
   category_weight=10,
   price_weight=1,
   review_count_weight=1,
   review_rating_weight=1,
   limit=TOP_N,
)


women_cat_result.to_pandas()
```



### Using events data to create personalized experiences

Now fast-forward a month. Our users made some interactions on our platform. User_1 did some more, while user_2 only did some.
Let's now utilize their behavioral data, represented as events and their effects, for our two example users:
- a user interested in casual and leisure products
- a user interested in elegant products for going out and formal work occasions

```python
events_df = (
   pd.read_json(EVENT_DATASET_URL)
   .reset_index()
   .rename(columns={"index": "id"})
   .head(NROWS)
)
events_df = events_df.merge(
   products_df[["id"]], left_on="product", right_on="id", suffixes=("", "r")
).drop("idr", axis=1)
events_df
```



You can set up different actions to show certain levels of interest

```python
event_weights = {
   "clicked_on": 0.2,
   "buy": 1,
   "put_to_cart": 0.5,
   "removed_from_cart": -0.5,
}
```

Then personalize your recommendations to the user by setting up events to impact the retrieval.

```python
# for a new index, all data has to be put into the source again
source_product.put([products_df])
source_user.put([user_df])
source_event.put([events_df])
# a query only searching with the user's vector the preferences are now much more personalised thanks to the events
personalised_query = (
   Query(
       product_index_with_events,
       weights={
           description_space: Param("description_weight"),
           category_space: Param("category_weight"),
           name_space: Param("name_weight"),
           price_space: Param("price_weight"),
           review_count_space: Param("review_count_weight"),
           review_rating_space: Param("review_rating_weight"),
       },
   )
   .find(product)
   .with_vector(user, Param("user_id"))
   .limit(Param("limit"))
)
```

The following two snippets show the impact of a slight (vs. a large) personalisation weighting on your results.

```python
# with small weight on the spaces the events affected, we mainly just alter the results below position 4
general_event_result = app_with_events.query(
   personalised_query,
   user_id="user_1",
   description_weight=1,
   category_weight=1,
   name_weight=1,
   price_weight=1,
   review_count_weight=1,
   review_rating_weight=1,
   limit=TOP_N,
)


general_event_result.to_pandas().join(
   simple_result.to_pandas(), lsuffix="", rsuffix="_base"
)[["description", "id", "description_base", "id_base"]]
```

You can see mainly the lower results have changed from the previous iteration



```python
# with larger weight on the the event affected spaces, more totally new items appear in the TOP10
event_weighted_result = app_with_events.query(
   personalised_query,
   user_id="user_1",
   query_text="",
   description_weight=5,
   category_weight=1,
   name_weight=1,
   price_weight=1,
   review_count_weight=1,
   review_rating_weight=1,
   limit=TOP_N,
)


event_weighted_result.to_pandas().join(
   simple_result.to_pandas(), lsuffix="", rsuffix="_base"
)[["description", "id", "description_base", "id_base"]]
```

Whereas here the results are very different



And that’s how you can create personalised recommendation systems that account for both the semantic meaning of your user query and their preferences using Superlinked.

[Try it out](https://colab.research.google.com/github/superlinked/superlinked/blob/main/notebook/recommendations_e_commerce.ipynb)
