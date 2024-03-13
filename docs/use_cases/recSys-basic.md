# Rec Sys for Beginners

## Why do we build Recommender Systems?

Recommender Systems are central to nearly every web platform offering things - movies, clothes, any kind of commodity - to users. Recommenders analyze patterns of user behavior to suggest items they might like but not necessarily discover on their own, items similar to what they or users similar to them have liked in the past. Personalized recommendation systems are reported to increase sales, boost user satisfaction, and improve engagment on a broad range of platforms, including Amazon, Netflix, and Spotify. Building one yourself may seem daunting. Where do you start? What are the necessary components?

Below, we'll show how to build a very simple recommender system. Our example system suggests news articles to users, and consists of just two parts:
    
    1. a content-based recommender - the model identifies and recommends items similar to the context item. To motivate readers to read more content, we show them a list of recommendations, entited "Similar Articles."
    2. a collaborative-filtering recommender - based on the user's past interactions, the model first identifies users with an interaction history similar to the current user's, collects articles these similar users have interacted with, excluding articles the user's already seen, and recommends these articles as an "Others also read" or "Personalized Recommendations" list, indicating to the user that this list is specifically generated for them.

We build our recommenders using a news dataset, [downloadable here](https://www.kaggle.com/datasets/yazansalameh/news-category-dataset-v2). Down below, once we move on to the collaborative-filtering model, we'll link you to user-article interaction data. But first, **let's set up and refine our dataest**.

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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>authors</th>
      <th>link</th>
      <th>short_description</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CRIME</td>
      <td>There Were 2 Mass Shootings In Texas Last Week...</td>
      <td>Melissa Jeltsen</td>
      <td>https://www.huffingtonpost.com/entry/texas-ama...</td>
      <td>She left her husband. He killed their children...</td>
      <td>2018-05-26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTERTAINMENT</td>
      <td>Will Smith Joins Diplo And Nicky Jam For The 2...</td>
      <td>Andy McDonald</td>
      <td>https://www.huffingtonpost.com/entry/will-smit...</td>
      <td>Of course it has a song.</td>
      <td>2018-05-26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ENTERTAINMENT</td>
      <td>Hugh Grant Marries For The First Time At Age 57</td>
      <td>Ron Dicker</td>
      <td>https://www.huffingtonpost.com/entry/hugh-gran...</td>
      <td>The actor and his longtime girlfriend Anna Ebe...</td>
      <td>2018-05-26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENTERTAINMENT</td>
      <td>Jim Carrey Blasts 'Castrato' Adam Schiff And D...</td>
      <td>Ron Dicker</td>
      <td>https://www.huffingtonpost.com/entry/jim-carre...</td>
      <td>The actor gives Dems an ass-kicking for not fi...</td>
      <td>2018-05-26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENTERTAINMENT</td>
      <td>Julianna Margulies Uses Donald Trump Poop Bags...</td>
      <td>Ron Dicker</td>
      <td>https://www.huffingtonpost.com/entry/julianna-...</td>
      <td>The "Dietland" actress said using the bags is ...</td>
      <td>2018-05-26</td>
    </tr>
  </tbody>
</table>
</div>



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

Next, we remove news with short headlines (shorter than 7 words), and then drop duplicates based on headline.


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

Next, we'll implement our content-based recommender. This recommender creates the same recommended list for all users, displayed under the title "Similar Articles."

To identify which news articles are similar to a given article, we obtain embeddings of the text associated with all articles in our refined dataset. Once we have the embeddings, we use cosine similarity to retrieve the most similar articles. We use a model from the Sentence Transformers family that is often used for text-embedding tasks.


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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>authors</th>
      <th>link</th>
      <th>short_description</th>
      <th>date</th>
      <th>headline_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2932</th>
      <td>QUEER VOICES</td>
      <td>‘Will &amp; Grace’ Creator To Donate Gay Bunny Boo...</td>
      <td>Elyse Wanshel</td>
      <td>https://www.huffingtonpost.com/entry/will-grac...</td>
      <td>It's about to be a lot easier for kids in Mike...</td>
      <td>2018-04-02</td>
      <td>‘Will &amp; Grace’ Creator To Donate Gay Bunny Boo...</td>
    </tr>
    <tr>
      <th>4487</th>
      <td>QUEER VOICES</td>
      <td>‘The Voice’ Blind Auditions Make History With ...</td>
      <td>Lyndsey Parker, Yahoo Entertainment</td>
      <td>https://www.huffingtonpost.com/entry/the-voice...</td>
      <td>Austin Giorgio, 21: “How Sweet It Is (To Be Lo...</td>
      <td>2018-03-06</td>
      <td>‘The Voice’ Blind Auditions Make History With ...</td>
    </tr>
    <tr>
      <th>8255</th>
      <td>QUEER VOICES</td>
      <td>‘The Penumbra’ Is The Queer Audio Drama You Di...</td>
      <td>Sarah Emily Baum, ContributorFreelance Writer</td>
      <td>https://www.huffingtonpost.com/entry/the-penum...</td>
      <td>Young, fun, fantastical and, most notably, inc...</td>
      <td>2018-01-05</td>
      <td>‘The Penumbra’ Is The Queer Audio Drama You Di...</td>
    </tr>
    <tr>
      <th>744</th>
      <td>COMEDY</td>
      <td>‘The Opposition’ Gives Trump A Hot Lawyer Of H...</td>
      <td>Ed Mazza</td>
      <td>https://www.huffingtonpost.com/entry/trump-hot...</td>
      <td>He's here to make a "strong case" for the pres...</td>
      <td>2018-05-11</td>
      <td>‘The Opposition’ Gives Trump A Hot Lawyer Of H...</td>
    </tr>
    <tr>
      <th>2893</th>
      <td>ENTERTAINMENT</td>
      <td>‘Stranger Things’ Fans Will Be Able To Visit T...</td>
      <td>Elyse Wanshel</td>
      <td>https://www.huffingtonpost.com/entry/stranger-...</td>
      <td>Hawkins is headed to Hollywood, Orlando and Si...</td>
      <td>2018-04-03</td>
      <td>‘Stranger Things’ Fans Will Be Able To Visit T...</td>
    </tr>
  </tbody>
</table>
</div>



We make index serve as article id, as follows:


```python
news_articles['article_id'] = news_articles.index
news_articles.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>authors</th>
      <th>link</th>
      <th>short_description</th>
      <th>date</th>
      <th>headline_description</th>
      <th>article_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2932</th>
      <td>QUEER VOICES</td>
      <td>‘Will &amp; Grace’ Creator To Donate Gay Bunny Boo...</td>
      <td>Elyse Wanshel</td>
      <td>https://www.huffingtonpost.com/entry/will-grac...</td>
      <td>It's about to be a lot easier for kids in Mike...</td>
      <td>2018-04-02</td>
      <td>‘Will &amp; Grace’ Creator To Donate Gay Bunny Boo...</td>
      <td>2932</td>
    </tr>
    <tr>
      <th>4487</th>
      <td>QUEER VOICES</td>
      <td>‘The Voice’ Blind Auditions Make History With ...</td>
      <td>Lyndsey Parker, Yahoo Entertainment</td>
      <td>https://www.huffingtonpost.com/entry/the-voice...</td>
      <td>Austin Giorgio, 21: “How Sweet It Is (To Be Lo...</td>
      <td>2018-03-06</td>
      <td>‘The Voice’ Blind Auditions Make History With ...</td>
      <td>4487</td>
    </tr>
    <tr>
      <th>8255</th>
      <td>QUEER VOICES</td>
      <td>‘The Penumbra’ Is The Queer Audio Drama You Di...</td>
      <td>Sarah Emily Baum, ContributorFreelance Writer</td>
      <td>https://www.huffingtonpost.com/entry/the-penum...</td>
      <td>Young, fun, fantastical and, most notably, inc...</td>
      <td>2018-01-05</td>
      <td>‘The Penumbra’ Is The Queer Audio Drama You Di...</td>
      <td>8255</td>
    </tr>
    <tr>
      <th>744</th>
      <td>COMEDY</td>
      <td>‘The Opposition’ Gives Trump A Hot Lawyer Of H...</td>
      <td>Ed Mazza</td>
      <td>https://www.huffingtonpost.com/entry/trump-hot...</td>
      <td>He's here to make a "strong case" for the pres...</td>
      <td>2018-05-11</td>
      <td>‘The Opposition’ Gives Trump A Hot Lawyer Of H...</td>
      <td>744</td>
    </tr>
    <tr>
      <th>2893</th>
      <td>ENTERTAINMENT</td>
      <td>‘Stranger Things’ Fans Will Be Able To Visit T...</td>
      <td>Elyse Wanshel</td>
      <td>https://www.huffingtonpost.com/entry/stranger-...</td>
      <td>Hawkins is headed to Hollywood, Orlando and Si...</td>
      <td>2018-04-03</td>
      <td>‘Stranger Things’ Fans Will Be Able To Visit T...</td>
      <td>2893</td>
    </tr>
  </tbody>
</table>
</div>




```python
articles_simple = news_articles[['article_id', 'headline_description']]
articles_simple.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>article_id</th>
      <th>headline_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2932</th>
      <td>2932</td>
      <td>‘Will &amp; Grace’ Creator To Donate Gay Bunny Boo...</td>
    </tr>
    <tr>
      <th>4487</th>
      <td>4487</td>
      <td>‘The Voice’ Blind Auditions Make History With ...</td>
    </tr>
    <tr>
      <th>8255</th>
      <td>8255</td>
      <td>‘The Penumbra’ Is The Queer Audio Drama You Di...</td>
    </tr>
    <tr>
      <th>744</th>
      <td>744</td>
      <td>‘The Opposition’ Gives Trump A Hot Lawyer Of H...</td>
    </tr>
    <tr>
      <th>2893</th>
      <td>2893</td>
      <td>‘Stranger Things’ Fans Will Be Able To Visit T...</td>
    </tr>
  </tbody>
</table>
</div>



For computational efficiency, we collect the first 500 articles. Feel free to experiment with a different subset or the full dataset.


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



Next, we map the article ids to count ids. This lets us index articles in corpus by their count ids.

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


Our context article (above) was about Donald Trump. Our recommended articles also mention Mr. Trump. And it makes intuitive sense that people who read the context article would also be interested in reading our recommendations.

We now try to identify two different articles, and find articles similar/relevant to both. We do this using simple vector averaging before doing our cosine similarity search. We use articles from the Entertainment and Business sections.


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


### Evaluating recommender systems

**One way to evaluate a content-based recommender system is to 'manually' inspect the results**, the way we've done already, above. In our use case, a news platform, for example, we could get someone from the editorial team to check if our recommended articles are similar enough to our context article.

But to evaluate/compare **two or more recommenders** (whether they are content-based or user-interaction-based), the **gold standard is A/B-testing**. This simply means launching the models, assigning a fair amount of traffic to each, then basically seeing which one has a higher click-through-rate.

## 2. Collaborative-filtering recommenders

Below, we provide implementations of two collaborative filtering approaches that give personalized recommendations to users, in lists titled "Recommendations for you," "Others also read," or "Personalized Recommendations." Our implementations address the cold-start problem, and deploy some basic evaluation metrics that will tell us which model performs better.

### Generating user-item interactions

To keep things simple, we'll create a simulated user-article interaction dataset with the following assumptions:

Users have specific interests (e.g., "politics", "entertainment", "comedy"). Articles are already categorized, so we will simply 'match' users to their preferred category. We also assign a rating to the interaction: ratings ranging from 3 - 5 indicate a taste match, otherwise 1 - 2. For our purposes, we'll filter out the low rating interactions, but we leave the option for further exploration.


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

We see above that user_id 74 has an interest in "travel". Let's see if we have matched this user with relevant articles.


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



We have successfully matched articles with this user's interest ("travel").


Let's turn to our two collaborative filtering models. **To train our models, we need to create train and test data**.

We will provide two models. For the **first model**, which we'll call "**Similar Vectors**," the training data will be used to create a vector for each user, populated by ratings given to news articles. Once we've created our user vectors, we can retrieve the most similar users via, for example, cosine similarity. And once similar users are identified, we can easily collect articles they've viewed that the context user hasn't yet seen.

The **second model** is a **Matrix Factorization** model presented in [this paper](http://yifanhu.net/PUB/cf.pdf), with an efficient implementation in `implicit` package available [here](https://github.com/benfred/implicit).

### Cold-start problem

Our two models can recommend items only to users that were part of training. Because new users start without any interactions, we have to recommend articles to them using a different strategy. One common solution is to present these users with a list of the most popular items.


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
    

This cold start strategy is one tier of our more general recipe for providing recommendations to particular user-types (where user-type is determined by user activity on the platform):

- no interactions -> most popular items
- some interactions -> content-based items
- more interactions -> collaborative filtering


Before proceeding to training and testing, we need to split our interactions dataset into a training set and a test set.

### Train-test split


```python
train = interactions.head(20000)
test = interactions.tail(interactions.shape[0] - train.shape[0])
```

### Similar Vectors model
Here's the code for our Similar Vectors model.

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

### MF model parameters

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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>article_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>379822</th>
      <td>127</td>
      <td>1047</td>
      <td>3</td>
    </tr>
    <tr>
      <th>621877</th>
      <td>208</td>
      <td>57</td>
      <td>5</td>
    </tr>
    <tr>
      <th>857185</th>
      <td>286</td>
      <td>1958</td>
      <td>5</td>
    </tr>
    <tr>
      <th>894788</th>
      <td>299</td>
      <td>37</td>
      <td>5</td>
    </tr>
    <tr>
      <th>255164</th>
      <td>86</td>
      <td>8436</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>article_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>379822</th>
      <td>127</td>
      <td>1047</td>
      <td>3</td>
    </tr>
    <tr>
      <th>621877</th>
      <td>208</td>
      <td>57</td>
      <td>5</td>
    </tr>
    <tr>
      <th>857185</th>
      <td>286</td>
      <td>1958</td>
      <td>5</td>
    </tr>
    <tr>
      <th>894788</th>
      <td>299</td>
      <td>37</td>
      <td>5</td>
    </tr>
    <tr>
      <th>255164</th>
      <td>86</td>
      <td>8436</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### 'Manual' evaluation

Below, we list the articles that the context user has actually read, and compare this list to the lists generated by our two models.


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



Evaluating manually, we can see (above) that both of our models recommend items that belong to the 'travel' category, which means that both models produce lists that are relevant. But to provide a more insightful evaluation of our models, we need to measure them numerically.

### Evaluation Metrics

Another, probably more effective way to ascertain the quality of a recommender is to measure values such as precision and recall. These metrics evaluate recommendation relevancy in two different ways. Precision measures the proportion of recommended items that are relevant, while recall assesses the proportion of relevant items that are recommended. High precision means that most of the recommended items are relevant, and high recall means that most of the relevant items are recommended. Let's perform such an evaluation, below.


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

We first extract a list of users which appear in both the training set and the test set, because - as discussed in the Cold-start section - our models can generate recommendations for only those users whose interactions the models have been trained on.


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



### Position-based metrics

Recall and precision consider only the **number** of items common to both the recommendations and the test set. We can obtain a more complete picture by also generating metrics - for example, Mean Reciprocal Rank (MRR) - that measure the **position (rank)** of relevant items. MRR can provide an indication of whether our model does a good job of recommending the most relevant items to users first.

MRR is calculated as the average of the reciprocal ranks of the first correct answer for a set of queries or users. The reciprocal rank is the inverse of the rank at which the first relevant item appears; for example, if the first relevant item appears in the third position, the reciprocal rank is 1/3.

MRR is particularly useful when the position of the first relevant recommendation is more consequential than the presence of other relevant items in the list. MRR quantifies how effective a system is at providing the most relevant result as early as possible in a recommendation list. High MRR values indicate a system that often ranks the most relevant items higher, thereby increasing the probability of user satisfaction in scenarios where users are likely to consider only the top few recommendations or answers.


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


## Our results, for simulated data

Both precision-recall and MRR results indicate that for our simulated dataset the Similar Vectors approach gives better recommendations than the Matrix Factorization model. However, it's important to note that our models may perform differently with real-world data.


```python

```
