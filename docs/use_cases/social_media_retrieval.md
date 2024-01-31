<!-- SEO SUMMARY:  -->

# TODO: Add SEO Summary

# Social media retrieval system

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

Before diving into the code, let's look over a LinkedIn post to address the challenges it will introduce ↓

```json
[
    ...
    {
        "text": "What is the 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝗰𝗲 between your 𝗠𝗟 𝗱𝗲𝘃𝗲𝗹𝗼𝗽𝗺𝗲𝗻𝘁 and 𝗰𝗼𝗻𝘁𝗶𝗻𝘂𝗼𝘂𝘀 𝘁𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗲𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁𝘀?\nThey might do the same thing, but their design is entirely different ↓\n𝗠𝗟 𝗗𝗲𝘃𝗲𝗹𝗼𝗽𝗺𝗲𝗻𝘁 𝗘𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁\nAt this point, your main goal is to ingest the raw and preprocessed data through versioned artifacts (or a feature store), analyze it & generate as many experiments as possible to find the best:\n- model\n- hyperparameters\n- augmentations\nBased on your business requirements, you must maximize some specific metrics, find the best latency-accuracy trade-offs, etc.\nYou will use an experiment tracker to compare all these experiments.\nAfter you settle on the best one, the output of your ML development environment will be:\n- a new version of the code\n- a new version of the configuration artifact\nHere is where the research happens. Thus, you need flexibility.\nThat is why we decouple it from the rest of the ML systems through artifacts (data, config, & code artifacts).\n𝗖𝗼𝗻𝘁𝗶𝗻𝘂𝗼𝘂𝘀 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗘𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁\nHere is where you want to take the data, code, and config artifacts and:\n- train the model on all the required data\n- output a staging versioned model artifact\n- test the staging model artifact\n- if the test passes, label it as the new production model artifact\n- deploy it to the inference services\nA common strategy is to build a CI/CD pipeline that (e.g., using GitHub Actions):\n- builds a docker image from the code artifact (e.g., triggered manually or when a new artifact version is created)\n- start the training pipeline inside the docker container that pulls the feature and config artifacts and outputs the staging model artifact\n- manually look over the training report -> If everything went fine, manually trigger the testing pipeline\n- manually look over the testing report -> if everything worked fine (e.g., the model is better than the previous one), manually trigger the CD pipeline that deploys the new model to your inference services\nNote how the model registry quickly helps you to decouple all the components.\nAlso, because training and testing metrics are not always black & white, it is tough to 100% automate the CI/CD pipeline.\nThus, you need a human in the loop when deploying ML models.\nTo conclude...\nThe ML development environment is where you do your research to find better models:\n- 𝘪𝘯𝘱𝘶𝘵: data artifact\n- 𝘰𝘶𝘵𝘱𝘶𝘵: code & config artifacts\nThe continuous training environment is used to train & test the production model at scale:\n- 𝘪𝘯𝘱𝘶𝘵: data, code, config artifacts\n- 𝘰𝘶𝘵𝘱𝘶𝘵: model artifact\n.\n↳ See this strategy in action in my 𝗧𝗵𝗲 𝗙𝘂𝗹𝗹 𝗦𝘁𝗮𝗰𝗸 𝟳-𝗦𝘁𝗲𝗽𝘀 𝗠𝗟𝗢𝗽𝘀 𝗙𝗿𝗮𝗺𝗲𝘄𝗼𝗿𝗸 FREE course: 🔗\nhttps://lnkd.in/d_GVpZ9X\nhashtag\n#\nmachinelearning\nhashtag\n#\nmlops\nhashtag\n#\ndatascience",
        "image": "https://media.licdn.com/dms/image/D4D10AQEdpFdpJSlDKQ/image-shrink_800/0/1701156624205?e=1705082400&v=beta&t=jxPE3kyWPThjTX_XDPpcMOeSnaBplBodZgaU5ukMN3c"
    }
    ...
]
```

As you can see, during our preprocessing step, we have to take care of the following aspects that are not compatible with the embedding model:
- emojis
- bold, italic text
- URLs
- exceed the context window of the embedding model
- other non-ASCII characters 

## 3. Settings

It is good practice to have a single place to configure your application. We used `pydantic` to quickly implement a `AppSettings` class that contains all the default settings and can be overwritten by other files such as `.env` or `yaml`.

```python
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.prod"))

    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 256
    EMBEDDING_SIZE: int = 384
    EMBEDDING_MODEL_DEVICE: str = "cpu"
    VECTOR_DB_OUTPUT_COLLECTION_NAME: str = "linkedin_posts"

    # Variables loaded from .env file
    QDRANT_URL: str = "localhost:6333"
    QDRANT_API_KEY: Optional[str] = None


settings = AppSettings()
```

These constants will be spread across all the components.

## 4. Streaming ingestion pipeline

Let's dive into the streaming pipeline from top to bottom ↓

### 4.1. Bytewax Flow

Looking at the Bytewax flow, we can quickly understand all the steps of the streaming pipeline.

The first stage is to ingest every LinkedIn post from our JSON files. 

Next, every map operation has a single responsibility, such as:
- validating the ingested data using a`RawPost` `pydantic` model
- cleaning the post
- chunking the posts; we have used a `flat_map` operation as the chunking operation will output a list of `ChunkedPost` objects, which we want to flatten out
- embed the posts
- load the posts to a Qdrant vector DB

```python
def build_flow():
    embedding_model = EmbeddingModelSingleton()

    flow = Dataflow("flow")

    stream = op.input("input", flow, JSONSource(["data/paul.json"]))
    stream = op.map("raw_post", stream, RawPost.from_source)
    stream = op.map("cleaned_post", stream, CleanedPost.from_raw_post)
    stream = op.flat_map(
        "chunked_post",
        stream,
        lambda cleaned_post: ChunkedPost.from_cleaned_post(
            cleaned_post, embedding_model=embedding_model
        ),
    )
    stream = op.map(
        "embedded_chunked_post",
        stream,
        lambda chunked_post: EmbeddedChunkedPost.from_chunked_post(
            chunked_post, embedding_model=embedding_model
        ),
    )
    op.inspect("inspect", stream, print)
    op.output(
        "output", stream, QdrantVectorOutput(vector_size=model.embedding_size)
    )
    
    return flow
```

By wrapping every state of the post into a different `pydantic` model, we can quickly validate the data at each step and reuse the code in the retrieval module.

### 4.2. Clean LinkedIn posts

The raw LinkedIn posts are initially wrapped in a `RawPost` `pydantic` class to enforce static typing and the domain model, which is crucial for validating and modeling our data:

```python
class RawPost(BaseModel):
    post_id: str
    text: str
    image: Optional[str]

    @classmethod
    def from_source(cls, k_v: Tuple[str, dict]) -> "RawPost":
        k, v = k_v

        return cls(post_id=k, text=v["text"], image=v.get("image", None))
```

Following this strategy, we will write the next `pydantic` class representing the state of a cleaned post:

```python
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)
from src.cleaning import (
    remove_emojis_and_symbols,
    replace_urls_with_placeholder,
    unbold_text,
    unitalic_text,
)

class CleanedPost(BaseModel):
    post_id: str
    raw_text: str
    text: str
    image: Optional[str]

    @classmethod
    def from_raw_post(cls, raw_post: RawPost) -> "CleanedPost":
        cleaned_text = CleanedPost.clean(raw_post.text)

        return cls(
            post_id=raw_post.post_id,
            raw_text=raw_post.text,
            text=cleaned_text,
            image=raw_post.image,
        )
        
    @staticmethod
    def clean(text: str) -> str:
        cleaned_text = unbold_text(text)
        cleaned_text = unitalic_text(cleaned_text)
        cleaned_text = remove_emojis_and_symbols(cleaned_text)
        cleaned_text = clean(cleaned_text)
        cleaned_text = replace_unicode_quotes(cleaned_text)
        cleaned_text = clean_non_ascii_chars(cleaned_text)
        cleaned_text = replace_urls_with_placeholder(cleaned_text)
        
        return cleaned_text
```

The `from_raw_post` factory method takes an instance of the `RawPost` as input and uses the `clean()` method to clean the text to make it compatible with the embedding model.

As you can see, we address all the concerns highlighted in the `2. Data` section, such as bolded text, removing emojis, cleaning nonascii chars, etc.

Here is what the cleaned post looks like:

```json
{
    "text": "What is the difference between your ML development and continuous training environments?\nThey might do the same thing, but their design is entirely different  \nML Development Environment\nAt this point, your main goal is to ingest the raw and preprocessed data through versioned artifacts (or a feature store), analyze it & generate as many experiments as possible to find the best:\n- model\n- hyperparameters\n- augmentations\nBased on your business requirements, you must maximize some specific metrics, find the best latency-accuracy trade-offs, etc.\nYou will use an experiment tracker to compare all these experiments.\nAfter you settle on the best one, the output of your ML development environment will be:\n- a new version of the code\n- a new version of the configuration artifact\nHere is where the research happens. Thus, you need flexibility.\nThat is why we decouple it from the rest of the ML systems through artifacts (data, config, & code artifacts).\nContinuous Training Environment\nHere is where you want to take the data, code, and config artifacts and:\n- train the model on all the required data\n- output a staging versioned model artifact\n- test the staging model artifact\n- if the test passes, label it as the new production model artifact\n- deploy it to the inference services\nA common strategy is to build a CI/CD pipeline that (e.g., using GitHub Actions):\n- builds a docker image from the code artifact (e.g., triggered manually or when a new artifact version is created)\n- start the training pipeline inside the docker container that pulls the feature and config artifacts and outputs the staging model artifact\n- manually look over the training report -> If everything went fine, manually trigger the testing pipeline\n- manually look over the testing report -> if everything worked fine (e.g., the model is better than the previous one), manually trigger the CD pipeline that deploys the new model to your inference services\nNote how the model registry quickly helps you to decouple all the components.\nAlso, because training and testing metrics are not always black & white, it is tough to 100% automate the CI/CD pipeline.\nThus, you need a human in the loop when deploying ML models.\nTo conclude...\nThe ML development environment is where you do your research to find better models:\n- input: data artifact\n- output: code & config artifacts\nThe continuous training environment is used to train & test the production model at scale:\n- input: data, code, config artifacts\n- output: model artifact\n.\n  See this strategy in action in my The Full Stack 7-Steps MLOps Framework FREE course:  \n[URL]\nhashtag\n#\nmachinelearning\nhashtag\n#\nmlops\nhashtag\n#\ndatascience"
}
```


### 4.3. Chunk

Now, we are ready to chunk the cleaned posts ↓

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

class ChunkedPost(BaseModel):
    post_id: str
    chunk_id: str
    full_raw_text: str
    text: str
    image: Optional[str]

    @classmethod
    def from_cleaned_post(
        cls, cleaned_post: CleanedPost, embedding_model: EmbeddingModelSingleton
    ) -> list["ChunkedPost"]:
        chunks = ChunkedPost.chunk(cleaned_post.text, embedding_model)

        return [
            cls(
                post_id=cleaned_post.post_id,
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                full_raw_text=cleaned_post.raw_text,
                text=chunk,
                image=cleaned_post.image,
            )
            for chunk in chunks
        ]
        
    @staticmethod
    def chunk(text: str, embedding_model: EmbeddingModelSingleton) -> list[str]:
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"], chunk_size=500, chunk_overlap=0
        )
        text_sections = character_splitter.split_text(text)

        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=50,
            tokens_per_chunk=embedding_model.max_input_length,
            model_name=embedding_model.model_id,
        )

        chunks = []
        for text_section in text_sections:
            chunks.extend(token_splitter.split_text(text_section))
            
        return chunks
```

We have used the `RecursiveCharacterTextSplitter` class from LangChain to separate paragraphs delimited by `\n\n,` as these have a high chance of starting a different topic within the posts.

Afterward, as we are restricted by the `max_input_length` of the embedding model, we have used the `SentenceTransformersTokenTextSplitter` class to split the paragraphs further.

This is a standard strategy in chunking text for retrieval systems.

One last thing to point out is that we dynamically computed the `chunk_id` using the `MD5` deterministic digital signatures, ensuring that we didn't ingest duplicates into the vector DB. 

### 4.4. Embed

The last step before loading the posts to Qdrant is to compute the embedding of each chunk. To do so, we have a different `pydantic` model that exclusively calls the given embedding model ↓

```python
class EmbeddedChunkedPost(BaseModel):
    post_id: str
    chunk_id: str
    full_raw_text: str
    text: str
    text_embedding: list
    image: Optional[str] = None
    score: Optional[float] = None
    rerank_score: Optional[float] = None

    @classmethod
    def from_chunked_post(
        cls, chunked_post: ChunkedPost, embedding_model: EmbeddingModelSingleton
    ) -> "EmbeddedChunkedPost":
        return cls(
            post_id=chunked_post.post_id,
            chunk_id=chunked_post.chunk_id,
            full_raw_text=chunked_post.full_raw_text,
            text=chunked_post.text,
            text_embedding=embedding_model(chunked_post.text, to_list=True),
            image=chunked_post.image,
        )

    def to_payload(self) -> tuple[str, np.ndarray, dict]:
        return (
            self.chunk_id,
            self.text_embedding,
            {
                "post_id": self.post_id,
                "text": self.text,
                "image": self.image,
                "full_raw_text": self.full_raw_text,
            },
        )
```

### 4.5. Load to Qdrant

To keep things concise, to load the LinkedIn posts to Qdrant, you have to override a Bytewax `sync` class that is used as **output** in a Bytewax flow:

```python
class QdrantVectorSink(StatelessSinkPartition):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = settings.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        self._client = client
        self._collection_name = collection_name

    def write_batch(self, chunks: list[EmbeddedChunkedPost]):
        ids = []
        embeddings = []
        metadata = []
        for chunk in chunks:
            chunk_id, text_embedding, chunk_metadata = chunk.to_payload()

            ids.append(chunk_id)
            embeddings.append(text_embedding)
            metadata.append(chunk_metadata)

        self._client.upsert(
            collection_name=self._collection_name,
            points=Batch(
                ids=ids,
                vectors=embeddings,
                payloads=metadata,
            ),
        )
```

Within this class, you must overwrite the `write_batch` method, where we serialize every `EmbeddedChunkedPost` to a payload to map the data as Qdrant expects. Finally, we load the serialized data to the vector DB.


## 5. Retrieval client

### 5.1. Preprocess query

### 5.2. Plain retrieval

### 5.3. Visualize retrieval

### 5.4. Rerank

## 6. More examples


## Conclusion


### Future steps:
- add an intermediat NoSQL DB between the RAW NoSQL DB & the cleaned posts
- change embedding or rerank model



---

## Contributors

- [Paul Iusztin](https://www.linkedin.com/in/pauliusztin/)
- [Decoding ML](https://decodingml.substack.com/)