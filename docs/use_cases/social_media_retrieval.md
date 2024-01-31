<!-- SEO SUMMARY:  -->

# TODO: Add SEO Summary

# Social media retrieval system


# TODO: What will you learn?
# TODO: What problem do we solve?

In this article, you will learn how to build a real-time retrieval system for social media data. In our particular scenario, we will use only my LinkedIn posts, but they can easily be extended to other platforms that support written content, such as X, Instagram, or Medium.

As social media data platforms produce data at high frequencies, your vector DB can easily remain behind. Thus, we used a streaming engine to move data from the raw data source to the vector DB in real-time.

Even though we will explain only the retrieval part of an RAG system in this article, you can quickly hook the retrieved LinkedIn posts to an LLM for post analysis or personalized content generation.

**That being said, in this article, you will learn:**

- to build a streaming pipeline to ingest LinkedIn posts into a vector DB in real-time
- to clean, chunk, and embed LinkedIn posts
- build a retrieval client to query your LinkedIn post collection
- use reranking to improve the retrieval step
- visualize the retrieval step using UMAP

We won't dive into the basics of building a retrieval or RAG system but focus only on the specifics of our use case. In case you want to refresh your mind on RAG systems, check out this excellent article from VectorHub: [Retrieval Augmented Generation](https://hub.superlinked.com/retrieval-augmented-generation)


## 1. System design

# TODO: Add image

The retrieval system is split into 2 detached components:
1. The streaming ingestion pipeline
2. The retrieval client

### 1.1. The streaming ingestion pipeline

The streaming ingestion pipeline is used for [Change Data Capture (CDC)](https://hub.superlinked.com/12-data-modality#fpwX4) between a data source that contains the raw LinkedIn posts and the vector DB used for retrieval.

In a real-world scenario, the streaming pipeline will listen to a queue populated by all the changes made to the source database. But as our primary focus is the retrieval system, we simulated the queue with a couple of JSON files.

The streaming pipeline is built in Python using Bytewax and cleans, chunks, and embeds the LinkedIn posts before loading them into a [Qdrant](https://qdrant.tech/) vector DB. 


#### Why do we need a stream engine?

As LinkedIn posts (or any other social media data) are posted frequently, your vector DB can quickly become out of sync. You could build a batch pipeline that runs every minute, but the best way is to use a streaming pipeline that immediately takes every new item, preprocesses it and loads it into the vector DB.

By doing so, you are ensured that you have access to all the latest LinkedIn posts with minimal delay.


#### What is Bytewax?

[Bytewax](https://github.com/bytewax/bytewax) is a streaming engine built in Rust with a Python interface. Which means you get the best of both worlds:
- The fantastic speed and reliability of Rust
- The ease of use and ecosystem of Python


### 2.2. The retrieval client

The retrieval client will be a standard Python module that preprocesses the user queries and searches the vector DB for most similar results. It is decoupled from the streaming ingestion pipeline through the Qdrant vector DB.

Similar to the training-serving skew, it is essential to preprocess the ingested posts and queries using the same functions. 

The beauty of using a semantic-based retrieval system is that you are very flexible in how you can query your LinkedIn post collection. For example, you can find similar posts using another post or any other question or sentence.

Also, to improve the retrieval system, we used a reranking step.

Lastly, to better understand and explain the retrieval step, we will visualize it using UMAP.


## 2. Data

We will ingest 215 LinkedIn posts from [my profile - Paul Iusztin](https://www.linkedin.com/in/pauliusztin/). Even though how the posts are ingested is simulated through JSON files, the posts themself are authentic.

Before diving into the code, let's look over a LinkedIn post to address the challenges it will introduce.

```json
{
    "text": "What is the 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝗰𝗲 between your 𝗠𝗟 𝗱𝗲𝘃𝗲𝗹𝗼𝗽𝗺𝗲𝗻𝘁 and 𝗰𝗼𝗻𝘁𝗶𝗻𝘂𝗼𝘂𝘀 𝘁𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗲𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁𝘀?\nThey might do the same thing, but their design is entirely different ↓\n𝗠𝗟 𝗗𝗲𝘃𝗲𝗹𝗼𝗽𝗺𝗲𝗻𝘁 𝗘𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁\nAt this point, your main goal is to ingest the raw and preprocessed data through versioned artifacts (or a feature store), analyze it & generate as many experiments as possible to find the best:\n- model\n- hyperparameters\n- augmentations\nBased on your business requirements, you must maximize some specific metrics, find the best latency-accuracy trade-offs, etc.\nYou will use an experiment tracker to compare all these experiments.\nAfter you settle on the best one, the output of your ML development environment will be:\n- a new version of the code\n- a new version of the configuration artifact\nHere is where the research happens. Thus, you need flexibility.\nThat is why we decouple it from the rest of the ML systems through artifacts (data, config, & code artifacts).\n𝗖𝗼𝗻𝘁𝗶𝗻𝘂𝗼𝘂𝘀 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗘𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁\nHere is where you want to take the data, code, and config artifacts and:\n- train the model on all the required data\n- output a staging versioned model artifact\n- test the staging model artifact\n- if the test passes, label it as the new production model artifact\n- deploy it to the inference services\nA common strategy is to build a CI/CD pipeline that (e.g., using GitHub Actions):\n- builds a docker image from the code artifact (e.g., triggered manually or when a new artifact version is created)\n- start the training pipeline inside the docker container that pulls the feature and config artifacts and outputs the staging model artifact\n- manually look over the training report -> If everything went fine, manually trigger the testing pipeline\n- manually look over the testing report -> if everything worked fine (e.g., the model is better than the previous one), manually trigger the CD pipeline that deploys the new model to your inference services\nNote how the model registry quickly helps you to decouple all the components.\nAlso, because training and testing metrics are not always black & white, it is tough to 100% automate the CI/CD pipeline.\nThus, you need a human in the loop when deploying ML models.\nTo conclude...\nThe ML development environment is where you do your research to find better models:\n- 𝘪𝘯𝘱𝘶𝘵: data artifact\n- 𝘰𝘶𝘵𝘱𝘶𝘵: code & config artifacts\nThe continuous training environment is used to train & test the production model at scale:\n- 𝘪𝘯𝘱𝘶𝘵: data, code, config artifacts\n- 𝘰𝘶𝘵𝘱𝘶𝘵: model artifact\n.\n↳ See this strategy in action in my 𝗧𝗵𝗲 𝗙𝘂𝗹𝗹 𝗦𝘁𝗮𝗰𝗸 𝟳-𝗦𝘁𝗲𝗽𝘀 𝗠𝗟𝗢𝗽𝘀 𝗙𝗿𝗮𝗺𝗲𝘄𝗼𝗿𝗸 FREE course: 🔗\nhttps://lnkd.in/d_GVpZ9X\nhashtag\n#\nmachinelearning\nhashtag\n#\nmlops\nhashtag\n#\ndatascience"
}
```

As you can see, during our preprocessing step, we have to take care of the following aspects that are not compatible with the embedding model:
- emojis
- bold, italic text
- URLs
- exceed the context window of the embedding model
- other non-ASCII characters 

## 3. Streaming ingestion pipeline

### 3.1. Bytewax Flow

### 3.2. Extract LinkedIn posts

### 3.3. Clean LinkedIn posts

### 3.4. Chunk & Embed LinkedIn posts

### 3.6. Load processed posts to Qdrant


## 4. Retrieval client

### 4.1. Preprocess query

### 4.2. Plain retrieval

### 4.3. Visualize retrieval

### 4.4. Rerank

## 5. More examples


## Conclusion


### Future steps:
- add an intermediat NoSQL DB between the RAW NoSQL DB & the cleaned posts
- change embedding or rerank model



---

## Contributors

- [Paul Iusztin](https://www.linkedin.com/in/pauliusztin/)
- [Decoding ML](https://decodingml.substack.com/)