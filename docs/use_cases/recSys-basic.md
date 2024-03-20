# RecSys for Beginners

## Why do we build Recommender Systems?

Recommender Systems are central to nearly every web platform that offers things - movies, clothes, any kind of commodity - to users. Recommenders analyze patterns of user behavior to suggest items they might like but would not necessarily discover on their own, items similar to what they or users similar to them have liked in the past. Personalized recommendation systems are reported to increase sales, boost user satisfaction, and improve engagment on a broad range of platforms, including, for example, Amazon, Netflix, and Spotify. Building one yourself may seem daunting. Where do you start? What are the necessary components?

Below, we'll show you how to build a very simple recommender system. The rationale for our RecSys comes from our general recipe for providing recommendations, which is based on user-type (activity level):

| **interaction level** | -> | **recommendation approach** |
| ----------------- | -- | ---------------- |
| no interactions (cold start) | -> | most popular items |
| some interactions | -> | content-based items |
| more interactions | -> | collaborative filtering (interaction-based) items |

Our RecSys also lets you adopt use-case-specific strategies depending on whether a content- or interaction-based approach makes more sense. Our example system, which suggests news articles to users, therefore consists of two parts:
    
1. a **content-based recommender** - the model identifies and recommends items similar to the context item. To motivate readers to read more content, we show them a list of recommendations, entited "Similar Articles."
2. a **collaborative filtering (interaction-based) recommender** - this type of model first identifies users with an interaction history similar to the current user's, collects articles these similar users have interacted with, excluding articles the user's already seen, and recommends these articles as an "Others also read" or "Personalized Recommendations" list. These titles tell the user that the list is personalized - generated specifically for them.

Let's get started.

## Our RecSys build

We build our recommenders using a news dataset, [downloadable here](https://www.kaggle.com/datasets/yazansalameh/news-category-dataset-v2). Down below, once we move on to the collaborative filtering model, we'll link you to our user-article interaction data.

But first, **let's set up and refine our dataset**.

```python
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity  
```


```python
pd.options.display.max_columns = None
```


```python
from sentence_transformers import SentenceTransformer
```


```python
BERT_SENT = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
```


```python
news_articles = pd.read_json("News_Category_Dataset_v2.json", lines = True)
```


```python
news_articles.head()
```

|    | category      | headline                                            | authors        | link                                                | short_description                                  | date       |
|---:|:--------------|:----------------------------------------------------|:---------------|:----------------------------------------------------|:---------------------------------------------------|:-----------|
|  0 | CRIME         | There Were 2 Mass Shootings In Texas Last Week...  | Melissa Jeltsen| `https://www.huffingtonpost.com/entry/t...`  | She left her husband. He killed their children... | 2018-05-26 |
|  1 | ENTERTAINMENT | Will Smith Joins Diplo And Nicky Jam For The 2...  | Andy McDonald  | `https://www.huffingtonpost.com/entry/w...`  | Of course it has a song.                          | 2018-05-26 |
|  2 | ENTERTAINMENT | Hugh Grant Marries For The First Time At Age 57   | Ron Dicker     | `https://www.huffingtonpost.com/entry/h...`  | The actor and his longtime girlfriend Anna Ebe... | 2018-05-26 |
|  3 | ENTERTAINMENT | Jim Carrey Blasts 'Castrato' Adam Schiff And D... | Ron Dicker     | `https://www.huffingtonpost.com/entry/j...`  | The actor gives Dems an ass-kicking for not fi... | 2018-05-26 |
|  4 | ENTERTAINMENT | Julianna Margulies Uses Donald Trump Poop Bags... | Ron Dicker     | `https://www.huffingtonpost.com/entry/j...`  | The "Dietland" actress said using the bags is ... | 2018-05-26 |


Using our dataset's columns, e.g., `category`, `headline`, `short_description`, `date`, we can extract only instances published after 2018-01-01.


```python
news_articles.shape
```




    (200853, 6)




```python
news_articles = news_articles[news_articles['date'] >= pd.Timestamp(2018, 1, 1)]
```


```python
news_articles.shape
```




    (8583, 6)



By filtering out articles published on or before 2018-01-01, we've refined our article set down from around 200K to roughly 8.5K.

Next, we remove news articles with short headlines (shorter than 7 words), and then drop duplicates based on headline.


```python
news_articles = news_articles[news_articles['headline'].apply(lambda x: len(x.split()) > 6)]
news_articles.shape[0]
```




    8429




```python
# drop duplicates
news_articles = news_articles.sort_values('headline', ascending=False).drop_duplicates('headline', keep=False)
print(f"Total number of articles after removing duplicates: {news_articles.shape[0]}")
```

    Total number of articles after removing duplicates: 8384



```python
print("Total number of articles : ", news_articles.shape[0])
print("Total number of authors : ", news_articles["authors"].nunique())
print("Total number of categories : ", news_articles["category"].nunique())
```

    Total number of articles :  8384
    Total number of authors :  876
    Total number of categories :  26


## 1. Content-based recommender

Next, we implement the first of our models: the content-based recommender. This recommender creates the same recommended list for all readers of a given article, displayed under the title "Similar Articles."

To identify which news articles are similar to a given (or "context") article, we obtain embeddings of the text associated with all articles in our refined dataset. Once we have the embeddings, we employ cosine similarity to retrieve the most similar articles. We use a model from the Sentence Transformers family that is often used for text-embedding tasks.


```python
def get_text_and_mappings(df):

    corpus = df['headline_description'].tolist()
    
    # generate mappings
    ids_count_map = {row_id: index for index, row_id in enumerate(df['article_id'])}
    count_ids_map = {index: row_id for index, row_id in enumerate(df['article_id'])}
    
    return corpus, ids_count_map, count_ids_map
```


```python
def compute_vectors(corpus, model):

    print('Calculating Embeddings of articles...')
    vectors = model.encode(corpus)
    print('Embeddings calculated!')
    
    return vectors
```

We add one more column to the dataset, populated with values concatenating the headline and the news article description. We do this so we can embed the text corpus of this column - i.e., calculate a vector representation of the articles.


```python
news_articles["headline_description"] = news_articles['headline'] + ' ' + news_articles['short_description']
news_articles.head()
```

    
|      | category      | headline                                            | authors                                             | link                                                | short_description                                  | date       | headline_description                               |
|-----:|:--------------|:----------------------------------------------------|:----------------------------------------------------|:----------------------------------------------------|:---------------------------------------------------|:-----------|:----------------------------------------------------|
| 2932 | QUEER VOICES  | ‘Will & Grace’ Creator To Donate Gay Bunny Boo...   | Elyse Wanshel                                       | `https://www.huffingtonpost.com/entry/w...`        | It's about to be a lot easier for kids in Mike... | 2018-04-02 | ‘Will & Grace’ Creator To Donate Gay Bunny Boo...  |
| 4487 | QUEER VOICES  | ‘The Voice’ Blind Auditions Make History With ...  | Lyndsey Parker, Yahoo Entertainment                 | `https://www.huffingtonpost.com/entry/t...`        | Austin Giorgio, 21: “How Sweet It Is (To Be Lo... | 2018-03-06 | ‘The Voice’ Blind Auditions Make History With ...  |
| 8255 | QUEER VOICES  | ‘The Penumbra’ Is The Queer Audio Drama You Di...  | Sarah Emily Baum, ContributorFreelance Writer      | `https://www.huffingtonpost.com/entry/t...`        | Young, fun, fantastical and, most notably, inc... | 2018-01-05 | ‘The Penumbra’ Is The Queer Audio Drama You Di...  |
|  744 | COMEDY        | ‘The Opposition’ Gives Trump A Hot Lawyer Of H...  | Ed Mazza                                            | `https://www.huffingtonpost.com/entry/t...`        | He's here to make a "strong case" for the pres... | 2018-05-11 | ‘The Opposition’ Gives Trump A Hot Lawyer Of H...  |
| 2893 | ENTERTAINMENT | ‘Stranger Things’ Fans Will Be Able To Visit T...  | Elyse Wanshel                                       | `https://www.huffingtonpost.com/entry/s...`        | Hawkins is headed to Hollywood, Orlando and Si... | 2018-04-03 | ‘Stranger Things’ Fans Will Be Able To Visit T...  |


We make index serve as article id, as follows:


```python
news_articles['article_id'] = news_articles.index
news_articles.head()
```


|      | category      | headline                                            | authors                                             | link                                                | short_description                                  | date       | headline_description                               | article_id |
|-----:|:--------------|:----------------------------------------------------|:----------------------------------------------------|:----------------------------------------------------|:---------------------------------------------------|:-----------|:----------------------------------------------------|------------|
| 2932 | QUEER VOICES  | ‘Will & Grace’ Creator To Donate Gay Bunny Boo...   | Elyse Wanshel                                       | `https://www.huffingtonpost.com/entry/will-grac...` | It's about to be a lot easier for kids in Mike... | 2018-04-02 | ‘Will & Grace’ Creator To Donate Gay Bunny Boo...  | 2932       |
| 4487 | QUEER VOICES  | ‘The Voice’ Blind Auditions Make History With ...  | Lyndsey Parker, Yahoo Entertainment                 | `https://www.huffingtonpost.com/entry/the-voice...` | Austin Giorgio, 21: “How Sweet It Is (To Be Lo... | 2018-03-06 | ‘The Voice’ Blind Auditions Make History With ...  | 4487       |
| 8255 | QUEER VOICES  | ‘The Penumbra’ Is The Queer Audio Drama You Di...  | Sarah Emily Baum, ContributorFreelance Writer      | `https://www.huffingtonpost.com/entry/the-penum...` | Young, fun, fantastical and, most notably, inc... | 2018-01-05 | ‘The Penumbra’ Is The Queer Audio Drama You Di...  | 8255       |
|  744 | COMEDY        | ‘The Opposition’ Gives Trump A Hot Lawyer Of H...  | Ed Mazza                                            | `https://www.huffingtonpost.com/entry/trump-hot...` | He's here to make a "strong case" for the pres... | 2018-05-11 | ‘The Opposition’ Gives Trump A Hot Lawyer Of H...  | 744        |
| 2893 | ENTERTAINMENT | ‘Stranger Things’ Fans Will Be Able To Visit T...  | Elyse Wanshel                                       | `https://www.huffingtonpost.com/entry/stranger-...` | Hawkins is headed to Hollywood, Orlando and Si... | 2018-04-03 | ‘Stranger Things’ Fans Will Be Able To Visit T...  | 2893       |



```python
articles_simple = news_articles[['article_id', 'headline_description']]
articles_simple.head()
```


|      | article_id | headline_description                               |
|-----:|------------|:----------------------------------------------------|
| 2932 | 2932       | ‘Will & Grace’ Creator To Donate Gay Bunny Boo...  |
| 4487 | 4487       | ‘The Voice’ Blind Auditions Make History With ... |
| 8255 | 8255       | ‘The Penumbra’ Is The Queer Audio Drama You Di... |
|  744 | 744        | ‘The Opposition’ Gives Trump A Hot Lawyer Of H... |
| 2893 | 2893       | ‘Stranger Things’ Fans Will Be Able To Visit T...  |


For computational efficiency, we collect the first 500 articles. Feel free to experiment with a different subset or the full dataset, whatever fits your use case.


```python
articles_sample = articles_simple.head(500)
```


```python
corpus, ids_count_map, count_ids_map = get_text_and_mappings(articles_sample)
```


```python
print("Loading the model...")
model_bert = SentenceTransformer(BERT_SENT)
```

    Loading the model...



```python
vectors = compute_vectors(corpus, model_bert)
```

    Calculating Embeddings of articles...
    Embeddings calculated!



```python
import operator
from numpy import dot
from numpy.linalg import norm

def get_cosine_sim(count_id, a, vectors):
    
    ids_scores = []
    for count in range(len(vectors)):
        if count == count_id:
            continue
        b = vectors[count]
        cos_sim = dot(a, b) / (norm(a) * norm(b))
        ids_scores.append((count, cos_sim))

    ids_scores.sort(key=operator.itemgetter(1), reverse=True)

    return ids_scores

def get_similar_vectors(count_id, vectors, N):

    a = vectors[count_id]
    
    ids_scores = get_cosine_sim(count_id, a, vectors)
    
    return ids_scores[:N]

def get_similar_articles(count_id, vectors, N, count_ids_map):

    ids_scores = get_similar_vectors(count_id, vectors, N)
    original_ids = [count_ids_map[a[0]] for a in ids_scores]

    return original_ids
```


```python
def print_article_text(corpus, ids_count_map, similar_article_ids):
    
    #print(f'{N} movies/shows similar to\n {corpus[count_id]}:\n')
    for article_id in similar_article_ids:
        print(corpus[article_id])
        print('-' * 30)
```

Below, we choose a context article, and then search for articles similar to it. Our aim is to see whether the short article descriptions are enough to evaluate whether the recommended articles are indeed similar to the context article.

```python
original_id = 744
count_id = ids_count_map[original_id]
print("Context Article: ", count_id, corpus[count_id])
```

    Context Article:  3 ‘The Opposition’ Gives Trump A Hot Lawyer Of His Own He's here to make a "strong case" for the president.



```python
k = 3 # how many similar articles to retrieve
```


```python
similar_articles = get_similar_articles(count_id, vectors, k, count_ids_map)
similar_articles
```




    [3705, 8208, 7878]



Next, we map the article ids to count ids. This lets us index articles in the corpus by their count ids.

```python
similar_article_ids = [ids_count_map[id_] for id_ in similar_articles]
similar_article_ids
```




    [277, 276, 367]




```python
corpus[count_id] # context article
```




    '‘The Opposition’ Gives Trump A Hot Lawyer Of His Own He\'s here to make a "strong case" for the president.'




```python
print_article_text(corpus, ids_count_map, similar_article_ids) # similar articles
```

    White House Lawyer Insists Trump Isn't Considering Firing Mueller Republican lawmakers have insisted that Trump let the special counsel to do his job.
    ------------------------------
    White House Lawyer Misled Trump To Prevent James Comey's Dismissal: Report The deputy counsel was reportedly trying to prevent an obstruction investigation.
    ------------------------------
    Wednesday's Morning Email: Judge Halts Trump Administration's Plan To Kill DACA While a lawsuit proceeds.
    ------------------------------


Our context article (above) was about Donald Trump. Our recommended articles also mention Mr. Trump. On a first glance evaluation of our content-based recommender looks good - it makes intuitive sense that people who read the context article would also be interested in reading our recommended articles.

Our system should of course be able to handle scenarios where a user has read more than one article. We therefore test our content-based recommender model to see if it can identify two different context articles, and find articles similar/relevant to both. We do this using simple vector averaging before doing our cosine similarity search. We use articles from the Entertainment and Business sections.


```python
bv_ind = 2893 # entertainment
bv_count_id = ids_count_map[bv_ind]
bv_vect = vectors[bv_count_id]

print("1st Context Article: ", bv_count_id, corpus[bv_count_id])
```

    1st Context Article:  4 ‘Stranger Things’ Fans Will Be Able To Visit The Upside Down IRL Hawkins is headed to Hollywood, Orlando and Singapore this fall.



```python
en_ind = 3510 # business
en_count_id = ids_count_map[en_ind]
en_vect = vectors[en_count_id]

print("2nd Context Article: ", en_count_id, corpus[en_count_id])
```

    2nd Context Article:  69 YouTube Quietly Escalates Crackdown On Firearm Videos The video site is expanding restrictions following the Florida massacre.



```python
avg_vect = average_array = (bv_vect + en_vect) / 2

similar_articles_ = get_cosine_sim(count_id, avg_vect, vectors)
```


```python
similar_article_ids_ = [ids_count_map[id_[0]] for id_ in similar_articles_ if id_[0] in ids_count_map][:k]
```


```python
print_article_text(corpus, ids_count_map, similar_article_ids_)
```

    Whistleblower Leaked Michael Cohen's Financials Over Potential Cover-Up: Report The whistleblower said two files about Cohen's business dealings are missing from a government database.
    ------------------------------
    What You Missed About The Saddest Death In 'Avengers: Infinity War' Directors Joe and Anthony Russo answer our most pressing questions.
    ------------------------------
    Will Ferrell And Molly Shannon Cover The Royal Wedding As 'Cord And Tish' They should cover everything.
    ------------------------------

Our content-based model successfully provides articles relevant to both of the context articles.

### Evaluating recommender models

The gold standard for evaluating recommender models is to A/B test - launch the models, assign a fair amount of traffic to each, then see which one has a higher click-through-rate. But a **relatively easy way to get a first-glimpse evaluation** of a recommender model (whether content-based or user-interaction-based) is to **'manually' inspect the results**, the way we've already done above. In our use case - a news platform, for example, we could get someone from the editorial team to check if our recommended articles are similar enough to our context article.

Manual evaluation provides a sense of the relevance and interpretability of the recommendations. But manual evaluation remains relatively subjective and not scalable. To get a more objective (and scalable) evaluation, we can compliment our manual evaluation by obtaining metrics - precision, recall, and rank. We use manual evaluation for both our content-based and collaborative filtering (interaction-based) models, and run metrics on the latter. Let's take a closer look at these collaborative filtering models.

## 2. Collaborative filtering recommenders

To be able to provide personalized article recommendations to our users, we need to use interaction-based models in our RecSys. Below, we provide implementations of two collaborative filtering approaches that can provide user-specific recommendations, in lists we title "Recommendations for you," "Others also read," or "Personalized Recommendations." Our implementations, called "Similar Vectors" and "Matrix Factorization," address the cold-start problem, and deploy some basic evaluation metrics - precision, recall, rank - that will tell us which model performs better.

### Generating a user-item interaction dataset

To keep things simple, we'll first create a simulated user-article interaction dataset with the following assumptions:

Users are randomly assigned specific interests (e.g., "politics", "entertainment", "comedy", "travel", etc.). Articles are already categorized according to "interest", so we will simply 'match' users to their preferred interest category. We also assign a rating to the interaction: ratings ranging from 3 - 5 indicate a match between user interest and article category, ratings of 1 - 2 indicate no match. For our purposes, we'll filter out the low rating interactions.


```python
import random

def create_users(num_users, categories):
    interests = categories # + ['other']  # include 'other' for users with general interests
    return [{'user_id': i + 1, 'interest': random.choice(interests)} for i in range(num_users)]

# generate user-article interactions
def generate_interactions(users, articles_df):
    interactions = []
    for user in users:
        user_interest = user['interest']
        for _, article in articles_df.iterrows():
            # bias: Higher probability of higher rating for interest match
            # if article['category'] == user_interest or user_interest == 'other':
            if article['category'] == user_interest:
                rating = random.randint(3, 5)
            else:
                rating = random.randint(1, 2) # (1, 3)
            interactions.append({
                'user_id': user['user_id'],
                'article_id': article['article_id'],
                'rating': rating
            })
    return pd.DataFrame(interactions)
```

For computational efficiency, we sample the first 3K articles, and limit the number of users to 300.


```python
articles = news_articles.head(3000)
```


```python
# create 300 users
num_users = 300
categories = articles['category'].unique().tolist()
users_dynamic = create_users(num_users, categories)
```


```python
# cenerate the user-article interactions dataset
interactions = generate_interactions(users_dynamic, articles)
print(interactions.head())
```

       user_id  article_id  rating
    0        1        2932       1
    1        1        4487       1
    2        1        8255       1
    3        1         744       1
    4        1        2893       2



```python
interactions.shape
```




    (900000, 3)




```python
# collecting interactions with ratings greater than or equal to 3
interactions = interactions[interactions['rating'] >= 3]
print(interactions.shape)
```

    (25768, 3)



```python
user_id = 74
```

user_id 74, for example, has been assigned an interest in "travel". Let's see if we've successfully matched this user with "travel" articles.


```python
specific_articles = interactions[interactions['user_id'] == user_id]['article_id'].unique().tolist()
```


```python
def print_articles_from_list(articles):
    
    for id_ in articles:
        print(news_articles.loc[id_]['headline_description'] + "\n")
```


```python
print_articles_from_list(specific_articles)
```

    United Airlines Mistakenly Flies Family's Dog To Japan Instead Of Kansas City The mix-up came just a day after a puppy died aboard a United flight.
    
    The 10 Best Hotels In The US In 2018, Revealed BRB, booking a trip to San Antonio immediately ✈️
    
    Rogue Cat Rescued After Hiding Out In New York Airport For Over A Week Pepper is safe and sound!
    
    Yelp Users Are Dragging Trump Hotels By Leaving ‘S**thole’ Reviews “Perhaps the Trump brand could take some lessons from Norway, where they have the BEST hotels."
    
    You Can Fly Around The World For Less Than $1,200 See four cities across four continents in 16 days ✈️
    
    Take A Virtual Disney Vacation With Stunning New Google Street View Maps Visit Disneyland and Disney World on the same day without leaving home.
    
    These Gorgeous Secret Lagoons Exist For Only Three Months A Year Lençóis might give off Saharan vibes, but the park is not technically a desert.
    
    The 5 Best (And Most Affordable) Places To Travel In April If you’re smart about where you place that pin on the map, you can even trade some rainy days for sunshine.
    
    The World's Best Food Cities, According To TripAdvisor Will travel for food 🍜
    
    The Most Popular U.S. Destinations Of 2018, According To TripAdvisor Get inspired for your next getaway.
    
    These Rainbow Mountains In Peru Look Like They’re Straight Out Of A Dr. Seuss Book Sometimes we can’t help but wonder if Mother Nature specifically designed some places just for Instagram.
    
    The Most Popular Destinations In The World, According To TripAdvisor And where to stay in these spots when you visit.
    
    Why People Like To Stay In Places Where Celebrities Have Died Inside the world of dark tourism.
    
    The 5 Best (And Most Affordable) Places To Travel in March 1. Miami, Florida
    
    United Airlines Temporarily Suspends Cargo Travel For Pets The decision follows multiple pet-related mishaps, including the death of a puppy.
    
    The One Thing You’re Forgetting In Your Carry-On While you can’t bring that liter of SmartWater through TSA, you absolutely can bring the empty bottle to refill at a water fountain once you’re past security.
    
    What The Southwest Flight Can Teach Us About Oxygen Masks A former flight attendant called out passengers for wearing their masks incorrectly.
    
    Your Emotional Support Spider And Goat Are Now Banned On American Airlines The airline released a list of prohibited animals after seeing a 40 percent rise in onboard companions.
    
    Space Mountain With The Lights On Is A Freaky Experience See Disney's most famous coaster in a whole new light.
    
    Wild Brawls Turn Carnival Ship Into 'Cruise From Hell' Twenty-three passengers were removed for "disruptive and violent acts.”
    
    What Flight Attendants Really Wish You'd Do On Your Next Flight Take. off. your. headphones.
    
    United Bans Many Popular Dog And Cat Breeds From Cargo Holds After Pet Deaths The airline unveiled new restrictions in its pet transportation policy.
    
    Why There Are Tiny Holes At The Bottom Of Windows On Planes If you’ve ever been on a plane, chances are, you’ve probably looked out your window, only to notice a tiny hole at the bottom
    
    These Are The Most Expensive Travel Days Of The Year We all love a good travel deal, so avoid these two days. There are two days of the year that are the absolute most expensive
    
    Roller Coaster Riders Suspended 100 Feet In The Air, Facing Down, After Malfunction An "abnormality" halted the coaster at the worst possible time.
    



```python
set([news_articles.loc[id_]['category'] for id_ in specific_articles])
```




    {'TRAVEL'}


Success! We see that user_id 74 has been matched with articles appropriate to their interest in "travel". Our user-interaction dataset appears to be effective in matching users with articles they would be interested in.

### Training our interaction-based models

Now that we've successfully created our user-interaction dataset, let's get into the details of our two collaborative filtering models. **To train them, we need to create training and test data**.

In our **first collaborative filtering model**, which we'll call "**Similar Vectors**," we use training data to create a vector for each user, populated by ratings they've given to news articles. Once we've created our user vectors, we can retrieve the most similar users via, for example, cosine similarity. And once similar users are identified, we can easily collect articles they've viewed that the context user hasn't yet seen.

Our **second collaborative filtering model** is a **Matrix Factorization** model, presented in [this paper](http://yifanhu.net/PUB/cf.pdf), with an efficient implementation in `implicit` package available [here](https://github.com/benfred/implicit).

### Cold-start problem

Recall the tiers of our general recipe for providing recommendations to particular user-types (based on their activity level). Our two collaborative filtering models can recommend items only to users that were part of training. And our content-based recommender can only recommend articles to users who have at least one interaction with content. Because new users start "cold" - without any platform activity, we have to recommend articles to them using a different strategy. One common solution to the cold start problem is to present these users with a list of the most popular items among all users.


```python
most_popular_articles = train['article_id'].value_counts().index.tolist()[:5]
most_popular_articles
```




    [6622, 1005, 2561, 3622, 1861]




```python
print_articles_from_list(most_popular_articles)
```

    These 'No Promo Homo' Laws Are Hurting LGBTQ Students Across America Seven states still have laws that specifically target gay students.
    
    Trump Defends Gina Haspel, His Nominee For CIA Director, And Her Record Of Torture “We have the most qualified person, a woman, who Democrats want Out because she is too tough on terror,” Trump tweeted Monday.
    
    Trump Condemns 'Sick' Syria Disaster Yet Slams The Door On Countless Refugees The U.S. has resettled only 44 Syrian refugees since October.
    
    Trump Congratulates Putin On Totally Expected Victory In Russian Election It seems the U.S. president at least has no hard feelings toward Russia.
    
    Trump Considering 'Full Pardon' Of Late Boxing Champion Jack Johnson Johnson, the first black heavyweight champion, was arrested for driving his girlfriend across state lines.
    

By suggesting Similar Articles, we hope to start moving cold start users to higher levels of platform activity, and expose them to our content-based and interaction-based recommendation approaches.

### Train-test split

Before proceeding to training and testing of our collaborative filtering models, we need to split our interactions dataset into a training set and a test set.

```python
train = interactions.head(20000)
test = interactions.tail(interactions.shape[0] - train.shape[0])
```

### Similar Vectors model

With our interactions dataset split, we can set up our Similar Vectors model.

```python
# function to recommend articles for a given user
def recommend_articles_sv(user_id, user_article_matrix, user_similarity_df, top_n=5):
    
    # identify similar users
    n_similar_users = 3
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:n_similar_users+1].index
    
    # get articles interacted by similar users with ratings > 0
    articles_seen_by_similar = user_article_matrix.loc[similar_users]
    articles_seen_by_similar = articles_seen_by_similar[articles_seen_by_similar > 0].stack().index.tolist()
    
    # get articles interacted by the target user with ratings > 0
    articles_seen_by_user = user_article_matrix.loc[user_id]
    articles_seen_by_user = articles_seen_by_user[articles_seen_by_user > 0].index.tolist()
    
    #print(len(articles_seen_by_similar), len(articles_seen_by_user))
    # filter out articles the user has already seen
    recommended_articles = [article for user, article in articles_seen_by_similar if article not in articles_seen_by_user]
    
    # select unique articles and limit the number of recommendations
    recommended_articles = list(set(recommended_articles))[:top_n]
    
    return recommended_articles
```


```python
user_article_matrix = train.pivot_table(index='user_id', columns='article_id', values='rating', fill_value=0)

# compute cosine similarity between users
user_similarity = cosine_similarity(user_article_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_article_matrix.index, columns=user_article_matrix.index)
```


```python
user_id = 74
```


```python
recommended_articles_sv = recommend_articles_sv(user_id, user_article_matrix, user_similarity_df, top_n=5)
recommended_articles_sv
```





    [5129, 5199, 4146, 5395, 1299]


Next, let's set up our Matrix Factorization model.

### Matrix Factorization (MF)

A general MF model takes a list of triplets (user, item, rating), and then tries to create vector representations for both users and items, such that the inner product of user-vector and item-vector is as close to the rating as possible. 

The specific MF model we use actually incorporates a weight component in the calculation of the inner product, and restricts the ratings to a constant value of 1, as shown in the following snippet.


```python
from typing import Iterable

import itertools
import os
import threadpoolctl
import implicit

from tqdm import tqdm
from scipy.sparse import csr_matrix

# as recommended by the implicit package warning
threadpoolctl.threadpool_limits(1, "blas")
```



```python
def build_user_and_item_mappings(users, items):
    users_map = {user: idx for idx, user in enumerate(users)}
    items_map = {game: idx for idx, game in enumerate(items)}
    return users_map, items_map
```


```python
def build_matrix(data, rating_col, users_map, items_map,
    user_col = "user_id",
    item_col= "article_id"):
    
    records = (
        data
        .loc[:, [user_col, item_col, rating_col]]
        .groupby(by=[user_col, item_col])
        .agg({rating_col: "sum"})
        .reset_index()
        .assign(**{
            user_col: lambda x: x[user_col].map(users_map),
            item_col: lambda x: x[item_col].map(items_map),
        })
    )
    return csr_matrix(
        (records[rating_col], (records[user_col], records[item_col])),
        shape=(len(users_map), len(items_map))
    )
```

**MF model parameters**

The MF model has several parameters:

- alpha: a float whose magnitude basically separates items users have interacted with from those they haven't
- factors: determines the length (dimensionality) of the vectors the model generates for both users and items
- iterations: the number of iterations involved in numerical optimization process when training the model

We pass these hyperparameters as follows.

```python
def train_model(
    train_matrix,
    test_matrix,
    alpha: float = 40,
    factors: int = 50,
    iterations: int = 30,
    show_progress: bool = False
):
    model = implicit.als.AlternatingLeastSquares(
        alpha=alpha,
        factors=factors,
        iterations=iterations,
    )
    model.fit(train_matrix, show_progress=show_progress)
    
    return model
```


```python
train.head()
```


|      | user_id | article_id | rating |
|-----:|---------|------------|--------|
| 379822 | 127    | 1047       | 3      |
| 621877 | 208    | 57         | 5      |
| 857185 | 286    | 1958       | 5      |
| 894788 | 299    | 37         | 5      |
| 255164 | 86     | 8436       | 5      |


```python
train_data = train.groupby(by=["user_id", "article_id"]).agg("sum").reset_index()
```


```python
# building player and bet mappings
users_map, items_map = build_user_and_item_mappings(train_data["user_id"].unique(), train_data["article_id"].unique())
print(f"Number of users: {len(users_map)}")
print(f"Number of articles: {len(items_map)}")
```

    Number of users: 299
    Number of articles: 3000



```python
model = implicit.als.AlternatingLeastSquares()
```


```python
train_matrix = build_matrix(train_data, rating_col='rating', users_map=users_map, items_map=items_map)
```


```python
model.fit(train_matrix, show_progress=True)
```

    100%|███████████████████████████████████████████| 15/15 [00:00<00:00, 89.22it/s]



```python
items_map_inv = {val:key for key, val in items_map.items()}
```


```python
def recommend_articles_mf(model, original_user_id, users_map, items_map, k=5):
    
    user_id = users_map[original_user_id]
    recos = set(model.recommend(user_id, train_matrix[user_id], N=k)[0])
    
    return [items_map[r] for r in recos]
```


```python
user_id = 74
```


```python
recommended_articles_mf = recommend_articles_mf(model, user_id, users_map, items_map_inv)
recommended_articles_mf
```




    [1299, 2524, 5129, 5199, 5395]




```python
train.head()
```

    
|      | user_id | article_id | rating |
|-----:|--------:|-----------:|-------:|
| 379822 | 127 | 1047 | 3 |
| 621877 | 208 | 57 | 5 |
| 857185 | 286 | 1958 | 5 |
| 894788 | 299 | 37 | 5 |
| 255164 | 86 | 8436 | 5 |


Now that we have both our Similar Vectors and Matrix Factorization models set up, let's start evaluating them, first 'manually', and then using metrics - precision, recall, and rank.

### Evaluating our collaborative filtering models - manual, and metrics

**'Manual' evaluation**

Below, we list the articles that the context user has actually read, and 'manually' compare this list to the lists generated by our two models.


```python
def get_seen_articles_by_user(df, user_id):
    
    return df[df['user_id'] == user_id]['article_id'].tolist()
```


```python
seen_articles = get_seen_articles_by_user(test, user_id)
len(seen_articles)
```




    7




```python
set([news_articles.loc[id_]['category'] for id_ in seen_articles])
```




    {'TRAVEL'}




```python
set([news_articles.loc[id_]['category'] for id_ in recommended_articles_sv])
```




    {'TRAVEL'}




```python
set([news_articles.loc[id_]['category'] for id_ in recommended_articles_mf])
```




    {'TRAVEL'}



Evaluating manually, we can see (above) that both of our models recommend items that belong to the 'travel' category, which means that both models produce lists that are relevant. This is a good intuitive start to evaluating our two interaction-based models. But to provide a more objective (and scalable) evaluation of our models, we need some quantitative metrics.

**Evaluation metrics**

Measuring values such as precision and recall are good ways to complement our manual evaluation of our recSys's quality. Precision and recall evaluate recommendation relevancy in two different ways. **Precision** measures the proportion of recommended items that are relevant, while **recall** assesses the proportion of relevant items that are recommended. High precision means that most of the recommended items are relevant, and high recall means that most of the relevant items are recommended. Let's perform such an evaluation of our models, below.

```python
def precision_and_recall_at_k(train, test, recommended, k):
    
    train_user_ids = train['user_id'].unique().tolist()
    test_user_ids = test['user_id'].unique().tolist()
    
    common_user_ids = list(set(train_user_ids).intersection(set(test_user_ids)))
    print('Number of common users: ', len(common_user_ids))

    precision = 0
    recall = 0
    
    for user_id in common_user_ids:
        #recommended_articles = recommend_articles(user_id, user_article_matrix, user_similarity_df, top_n=k)
        recommended_articles = recommended[user_id]
        test_articles = get_seen_articles_by_user(test, user_id)
        
        intersection_count = len(list(set(recommended_articles).intersection(set(test_articles))))
        
        # precision
        if k > 0:
            precision += intersection_count / k
        
        # recall
        if len(test_articles) > 0:
            recall += intersection_count / len(test_articles)
    
    # division by zero is handled
    if len(common_user_ids) > 0:
        average_precision = precision / len(common_user_ids)
        average_recall = recall / len(common_user_ids)
    else:
        average_precision = 0
        average_recall = 0
    
    return average_precision, average_recall
```

We first extract a list of users who appear in both the training set and the test set, because - as we discussed in the "Cold Start" section - our models can generate recommendations for only those users whose interactions the models have been trained on.


```python
train_user_ids = train['user_id'].unique().tolist()
test_user_ids = test['user_id'].unique().tolist()
    
common_user_ids = list(set(train_user_ids).intersection(set(test_user_ids)))
len(common_user_ids)
```




    284






```python
recos_mf = {user_id: recommend_articles_mf(model, user_id, users_map, items_map_inv) 
                         for user_id in common_user_ids}
```



```python
precision_and_recall_at_k(train, test, recos_sv, k=5)
```

    Number of common users:  284





    (0.784507042253521, 0.6387401993297399)




```python
precision_and_recall_at_k(train, test, recos_mf, k=5)
```

    Number of common users:  284





    (0.6542253521126759, 0.635305934716874)



**Position-based metrics**

Recall and precision consider only the **number** of items common to both the recommendations and the test set. We can obtain a more complete picture by also generating metrics - for example, Mean Reciprocal Rank (MRR) - that measure the **position (rank)** of relevant items. MRR can provide an indication of whether our model does a good job of recommending the most relevant items to users first.

MRR is calculated as the average of the reciprocal ranks of the first correct answer for a set of queries or users. The reciprocal rank is the inverse of the rank at which the first relevant item appears; for example, if the first relevant item appears in the third position, the reciprocal rank is 1/3.

MRR is particularly useful when the position of the first relevant recommendation is more consequential than the presence of other relevant items in the list. It quantifies how effective a recSys is at providing the most relevant result as early as possible in a recommendation list. High MRR values indicate a system that often ranks the most relevant items higher, thereby increasing the probability of user satisfaction in scenarios where users are likely to consider only the top few recommendations or answers.


```python
def calculate_mrr(common_user_ids, recommended, k):
    
    reciprocal_ranks = []
    
    for user_id in common_user_ids:

        #recommended_articles = recommend_articles(user_id, user_article_matrix, user_similarity_df, top_n=k) #recommend_to_user(user_id)
        recommended_articles = recommended[user_id]
        actual_articles = test[test['user_id'] == user_id]['article_id'].tolist()
        
        # find the rank of the first relevant (actual) article in the recommendations
        rank = None
        for i, article_id in enumerate(recommended_articles):
            if article_id in actual_articles:
                rank = i + 1  # adding 1 because index starts at 0, but ranks start at 1
                break
        
        # if a relevant article was found in the recommendations, calculate its reciprocal rank
        if rank:
            reciprocal_ranks.append(1 / rank)
    
    #print(reciprocal_ranks)
    # calculate MRR
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    return mrr
```


```python
calculate_mrr(common_user_ids, recos_sv, k)
```




    1.0




```python
calculate_mrr(common_user_ids, recos_mf, k)
```




    0.7596692111959288


## Collaborative filter results

Both precision-recall and MRR results indicate that, for our simulated dataset at least, the Similar Vectors approach gives better recommendations than the Matrix Factorization model. However, it's important to note that our models may perform differently with real-world data. 

## In sum...

In sum, we've implemented a RecSys that can handle the broad range of use cases encountered by any web platform that recommends things to users. Our RecSys incorporates three different approaches to handle recommendations for users of all three (zero, low, and higher) activity level types, as well as content-based ("Similar articles") and personalized (interaction-based) strategies ("e.g., "Recommendations for you," etc.) amenable to different sections of your web platform.

## Contributors

- [Dr. Mirza Klimenta](https://www.linkedin.com/in/mirza-klimenta/)
- [Robert Turner, editor](https://robertturner.co/copyedit)
