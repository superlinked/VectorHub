# Semantic search in business news

[Article built around this notebook: https://github.com/superlinked/superlinked/blob/main/notebook/semantic_search_news.ipynb] 

Semantic search is revolutionizing how we discover and consume news articles, offering a more intuitive and efficient method for finding relevant content and curating personalized news feeds. By understanding the nuances of language and the underlying concepts within news articles, semantic search can surface articles that align closely with the user's interests, preferences, and browsing history. 

Still, implementing effective semantic search for news articles presents several challenges, including:
Response optimization: Balancing factors in semantic search algorithms requires complex optimization processes.
Scalability and performance: Efficient indexing and retrieval mechanisms are needed to handle the vast volume of news articles.

Superlinked is designed to handle these challenges, so that you can prioritize similarity or freshness when recommending articles to users.

We’ll use the following parts of the Superlinked library to build a semantic search-powered application to identify news articles:

- Recency space - for understanding the freshness of different news articles
- TextSimilarity space - to interpret the meaning of different articles and identify similarities between them
- Query time weights - top define how you want the data to be treated when you run the query, avoiding the need to re-embed the whole dataset to optimize your experiment

...
This notebook implements semantic search in [news articles](https://www.kaggle.com/datasets/rmisra/news-category-dataset). The dataset is filtered for news in the 'BUSINESS' category.

We are embedding

headlines
news body (short description)
and date
to be able to search for

notable events, or
related articles to a specific story.
There is a possibility to skew the results towards older or fresher news, and also to influence the results using a specific search term.

In [2], change `mimetype` to ‘colab’ in alt.renderers.enable

## Boilerplate

First, we'll install Superlinked.

```python
%pip install superlinked==9.43.0
```

And then do our imports and constants...

(Note: Below, change `alt.renderers.enable(“mimetype”)` to `alt.renderers.enable('colab')` if you’re running this in google colab. Keep “mimetype” if you’re executing in github.)

```python
from datetime import datetime, timedelta, timezone

import os
import sys
import altair as alt
import pandas as pd

from superlinked.evaluation.charts.recency_plotter import RecencyPlotter
from superlinked.framework.common.dag.context import CONTEXT_COMMON, CONTEXT_COMMON_NOW
from superlinked.framework.common.dag.period_time import PeriodTime
from superlinked.framework.common.schema.schema import Schema
from superlinked.framework.common.schema.schema_object import String, Timestamp
from superlinked.framework.common.schema.id_schema_object import IdField
from superlinked.framework.common.parser.dataframe_parser import DataFrameParser
from superlinked.framework.dsl.executor.in_memory.in_memory_executor import (
    InMemoryExecutor,
    InMemoryApp,
)
from superlinked.framework.dsl.index.index import Index
from superlinked.framework.dsl.query.param import Param
from superlinked.framework.dsl.query.query import Query
from superlinked.framework.dsl.query.result import Result
from superlinked.framework.dsl.source.in_memory_source import InMemorySource
from superlinked.framework.dsl.space.text_similarity_space import TextSimilaritySpace
from superlinked.framework.dsl.space.recency_space import RecencySpace

alt.renderers.enable("mimetype") # NOTE: to render altair plots in colab, change 'mimetype' to 'colab'
alt.data_transformers.disable_max_rows()
alt.data_transformers.disable_max_rows()
pd.set_option("display.max_colwidth", 190)
```

```python
YEAR_IN_DAYS = 365
TOP_N = 10
DATASET_URL = "https://storage.googleapis.com/superlinked-notebook-news-dataset/business_news.json"
# as the dataset contains articles from 2022 and before, we can set our application's "NOW" to this date
END_OF_2022_TS = int(datetime(2022, 12, 31, 23, 59).timestamp())
EXECUTOR_DATA = {CONTEXT_COMMON: {CONTEXT_COMMON_NOW: END_OF_2022_TS}}
```

### Prepare & explore dataset

```python
NROWS = int(os.getenv("NOTEBOOK_TEST_ROW_LIMIT", str(sys.maxsize)))
business_news = pd.read_json(DATASET_URL, convert_dates=True).head(NROWS)
```

```python
# we are going to need an id column
business_news = business_news.reset_index().rename(columns={"index": "id"})
# we need to handle the timestamp being set in milliseconds
business_news["date"] = [
    int(date.replace(tzinfo=timezone.utc).timestamp()) for date in business_news.date
]
```

```python
# a sneak peak into the data
business_news.head()
```

![sneak peak into data](../assets/use_cases/semantic_search_news/sneak_peek.png)

### Understand release date distribution

```python
# some quick transformations and an altair histogram
years_to_plot: pd.DataFrame = pd.DataFrame(
    {
        "year_of_publication": [
            int(datetime.fromtimestamp(ts).year) for ts in business_news["date"]
        ]
    }
)
alt.Chart(years_to_plot).mark_bar().encode(
    alt.X("year_of_publication:N", bin=True, title="Year of publication"),
    y=alt.Y("count()", title="Count of articles"),
).properties(width=400, height=400)
```

![count of articles by year of publication](../assets/use_cases/semantic_search_news/count_article-by-year_publication.png)

The largest period time should be around 11 years as the oldest article is from 2012.

As most articles are between 2012-2017, therefore, it also makes sense to differentiate across the relatively scarce recent period of 4 years.

It can also make sense to give additional weight to more populous time periods - small differences can be amplified by adding extra weight compared to regions where the data is scarce and differences are larger on average.

## Set up Superlinked

```python
# set up schema to accommodate our inputs
class NewsSchema(Schema):
    description: String
    headline: String
    release_timestamp: Timestamp
    id: IdField
```

```python
news = NewsSchema()
```

```python
# textual characteristics are embedded using a sentence-transformers model
description_space = TextSimilaritySpace(
    text=news.description, model="sentence-transformers/all-mpnet-base-v2"
)
headline_space = TextSimilaritySpace(
    text=news.headline, model="sentence-transformers/all-mpnet-base-v2"
)
# release date is encoded using our recency embedding algorithm
recency_space = RecencySpace(
    timestamp=news.release_timestamp,
    period_time_list=[
        PeriodTime(timedelta(days=4 * YEAR_IN_DAYS), weight=1),
        PeriodTime(timedelta(days=11 * YEAR_IN_DAYS), weight=2),
    ],
    negative_filter=0.0,
)
```

```python
# we create an index of our spaces
news_index = Index(spaces=[description_space, headline_space, recency_space])
```

```python
# simple query will serve us right when we simply want to search the dataset with a search term
# the term will search in both textual fields
# and we will have to option to weight certain inputs' importance
simple_query = (
    Query(
        news_index,
        weights={
            description_space: Param("description_weight"),
            headline_space: Param("headline_weight"),
            recency_space: Param("recency_weight"),
        },
    )
    .find(news)
    .similar(description_space.text, Param("query_text"))
    .similar(headline_space.text, Param("query_text"))
    .limit(Param("limit"))
)

# news query on the other hand will search in the database with the vector of a news article
# weighting possibility is still there
news_query = (
    Query(
        news_index,
        weights={
            description_space: Param("description_weight"),
            headline_space: Param("headline_weight"),
            recency_space: Param("recency_weight"),
        },
    )
    .find(news)
    .with_vector(news, Param("news_id"))
    .limit(Param("limit"))
)
```

```python
dataframe_parser = DataFrameParser(
    schema=news,
    mapping={news.release_timestamp: "date", news.description: "short_description"},
)
```

```python
source: InMemorySource = InMemorySource(news, parser=dataframe_parser)
executor: InMemoryExecutor = InMemoryExecutor(
    sources=[source], indices=[news_index], context_data=EXECUTOR_DATA
)
app: InMemoryApp = executor.run()
```

```python
source.put([business_news])
```

## Understanding recency

```python
recency_plotter = RecencyPlotter(recency_space, context_data=EXECUTOR_DATA)
recency_plotter.plot_recency_curve()
```

## Queries

```python
# quick helper to present the results in a notebook
def present_result(
    result_to_present: Result,
    cols_to_keep: list[str] | None = None,
) -> pd.DataFrame:
    if cols_to_keep is None:
        cols_to_keep = [
            "description",
            "headline",
            "release_date",
            "id",
            "similarity_score",
        ]
    # parse result to dataframe
    df: pd.DataFrame = result_to_present.to_pandas()
    # transform timestamp back to release year. Ts is in milliseconds originally hence the division
    df["release_date"] = [
        datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        for timestamp in df["release_timestamp"]
    ]
    return df[cols_to_keep]
```

Let's search for one of the biggest acquisitions of the last decade! We are going to set recency's weight to 0 as it does not matter at this point.

```python
result = app.query(
    simple_query,
    query_text="Microsoft acquires LinkedIn",
    description_weight=1,
    headline_weight=1,
    recency_weight=0,
    limit=TOP_N,
)

present_result(result)
```

![microsoft acquires linkedin](../assets/use_cases/semantic_search_news/microsoft_acquires_linkedin.png)

The first result is about the deal, others are related to some aspect of the query. Let's try upweighting recency to see a recent big acquisition jump to the second place.

```python
result = app.query(
    simple_query,
    query_text="Microsoft acquires LinkedIn",
    description_weight=1,
    headline_weight=1,
    recency_weight=1,
    limit=TOP_N,
)

present_result(result)
```

![microsoft linkedin recency upweighted](../assets/use_cases/semantic_search_news/microsoft_linkedin_recency_upweighted.png)

Subsequently we can also search with the news article about Elon Musk offering to buy Twitter. As the dataset is quite biased towards old articles, what we get back is news about either Elon Musk or Twitter.

```python
result = app.query(
    news_query,
    description_weight=1,
    headline_weight=1,
    recency_weight=0,
    news_id="849",
    limit=TOP_N,
)

present_result(result)
```

![musk acquires twitter](../assets/use_cases/semantic_search_news/musk_twitter.png)

That we can start biasing towards recency, navigating the tradeoff of letting less connected but recent news into the mix. 

```python
result = app.query(
    news_query,
    description_weight=1,
    headline_weight=1,
    recency_weight=1,
    news_id="849",
    limit=TOP_N,
)

present_result(result)
```

![musk twitter recency](../assets/use_cases/semantic_search_news/musk_twitter_recency.png)