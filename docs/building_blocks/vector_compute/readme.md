<!-- TODO: Replace this text with a summary of article for SEO -->

# Vector Compute

<!-- TODO: Cover image: 
1. You can create your own cover image and put it in the correct asset directory,
2. or you can give an explanation on how it should be and we will help you create one. Please tag arunesh@superlinked.com or @AruneshSingh (GitHub) in this case. -->

## Introduction 

“Software is eating the world,” penned Marc Andreessen back in 2011. Now, more than a decade later, there’s a new guest at the table, and “AI is eating software.” More products and processes, including the creation of software itself, are being powered by advanced Machine Learning.

But it’s not an all-you-can-eat buffet. ML can’t just ingest anything it wants, whenever it wants. Building a good ML-powered system involves overcoming two common problems: _organizing_ your data in a way lets you quickly retrieve relevant information, and, relatedly, _representing_ your data in a way that makes it easy to feed into your ML models.

<!-- IMAGE 1 GOES HERE -->

These two problems are related. Indeed, they converge as parts of what is in essence _the_ defining challenge of many ML systems: turning your data into vector embeddings – that is, connecting your [Data Sources](https://hub.superlinked.com/data-sources) to your [Vector Search & Management](https://hub.superlinked.com/vector-search) system.

We call this the Vector Compute problem and this article explores how to build & use systems that solve it.

## What is Vector Compute?

In basic terms, Vector Compute is the infrastructure responsible for the training and management of vector embedding models, and the application of these models to your data in order to produce vector embeddings. These vectors then power the information retrieval and other machine learning systems across your organization.

## Vector Compute and ETL, not the same thing

<!-- IMAGE 2 GOES HERE -->

The role Vector Compute fills for your information retrieval system is similar to the role ETL tools like fivetran fill for your data warehouse. As in ETL, in Vector Compute you have to Extract the right information, Transform it into Vector Embeddings, and Load it into your Vector Search solution or cloud storage. 

There are, however, 2 important distinctions between ETL and Vector Compute:

1) Your _ETL_ system has to interact with dozens of data sources - loading data from all kinds of messy places, focus on cleaning it up, set and validate a schema, and keep everything as simple as possible while the data landscape of your company changes, over time.
In contrast, the data sources that feed into your _Vector Compute_ stack are likely those used as destinations by your ETL systems, like the data warehouse, your core database, or message queue solution. This means that in Vector Compute there can be fewer sources and they are likely of higher quality.

2) Often,the Transform step in _ETL_ contains just a simple aggregation, a join, or it fills in a missing value from a default. These operations are easy to express in SQL, making it easy to test whether they work.
In _Vector Compute_, on the other hand, we convert a diverse set of data into vector embeddings, which is a machine learning problem rather than a pure data engineering problem. This means that Vector Compute takes much longer to build than a simple ETL pipeline, more resources to run because you’re using ML models, and is more difficult to test – ML models don’t follow a simple “if A then B” logic, but instead operate within a spectrum of evaluation that is variable and task-specific.

In short, Vector Compute uses the data delivered by the ETL stack as input, and transforms it into vector embeddings, which are then used to organize it and extract insight from it. 

Finally, the key challenge you will face when building a Vector Compute system is the development and configuration of embedding models.

## Embedding Models, the heart of Vector Compute

At the core of Vector Compute are embedding models – machine learning models applied to raw data to generate vector embeddings.

Embedding models turn features extracted from high-dimensional data, with large numbers of attributes or dimensions, like text, images, or audio, into lower-dimensional but dense mathematical representations – i.e., vectors. You can also apply embedding models to structured data like tabular datasets or graphs. An e-commerce company, for example, might ingest thousands of rows of user activity logs with features such as date, product viewed, purchase amount, and so on. Embedding models encode these heterogeneous features as embedding vectors, with 100s of dimensions that highlight otherwise latent relationships and patterns, thereby enabling nearest neighbor search, clustering, or further modeling.

The mathematical vector representations that result from the vector embedding conversion process are readily amenable to processing by machine learning algorithms.

_Word_ embeddings, for example, capture nuanced linguistic similarities between terms that would be opaque to computers operating on just raw text strings. 

_Image_ embeddings (vector representations of images) can be fed, along with metadata, like labels, into traditional deep learning models to train them for tasks like image classification. Without image embedding, feeding images into deep learning models requires a specialized feature extraction step and specific model architecture – like a [convolutional neural network or a restricted Boltzmann machine](https://www.hindawi.com/journals/am/2022/3351256/). Using image embedding, you can work on computer vision tasks yourself; you don’t need specialized computer vision engineers to get started.

Furthermore, compared to complex raw data, vectors have far fewer dimensions, making them more efficient for tasks like storage, transfer, and retrieval. A low dimensional vector (i.e., array of numerical values) such as [102, 000, 241, 317, 004], for instance, encodes a wealth of semantic features and relationships. The continuity of the embedding space enables gradient-based optimization, which is central to Machine Learning modeling.

In summary, embedding models can efficiently turn obscure raw data into structured vector representations that reveal otherwise hidden patterns – patterns that computers can effectively model using ML.

But what does the embedding process look like?

### The embedding process visualized

Embedding maps data onto a high-dimensional vector space, often between 500-2000 dimensions, depending on the complexity of the underlying data. However, for visualization purposes, popular dimensionality reduction techniques like [UMAP](https://umap-learn.readthedocs.io/en/latest/) (Uniform Manifold Approximation and Projection) or [t-SNE](https://lvdmaaten.github.io/tsne/) (t-Distributed Stochastic Neighbor Embedding) can be used to project these dense vectors into 2D scatterplots that approximate the relative distances and relationships between data points, as shown in [this embedding projector](https://projector.tensorflow.org).

<!-- IMAGE 3 GOES HERE -->

If, for example, your vector space represented restaurant reviews, these could be embedded based on ratings, food quality, and customer responses. Similar reviews would cluster together in the vector space, while dissimilar ones would be further apart. Reviews mentioning “good food” and “tasty dishes” would be close together, as their meaning is similar.

### A new kind of model: Pre-trained Embedding

Until the early to mid-2010s, embedding was handled exclusively by ML teams building custom models from scratch. Custom models rely heavily on large volumes of task-specific data and require expert data scientists doing development and refinement for months.

Recent advances in [transfer learning](https://ai.plainenglish.io/transfer-learning-in-deep-learning-leveraging-pretrained-models-for-improved-performance-b4c49f2cd644) research have enabled _another_ kind of model, pre-trained on large more broad datasets, using more computational power than 99.9% of companies have available. Custom models continue to play a crucial role in Vector Compute, excelling at some tasks. But pre-trained models have performed better than custom models on others – in particular, text, image, and audio embedding. And, importantly, compared to building your own models, pre-trained models make it easier and faster to get to a working solution for your use case.

Let’s look at some examples:

[Llama-2](https://ai.meta.com/llama/), developed by Meta, is a set of LLMs that have been pre-trained on a publically available corpus of data, with variants trained on 7B, 13B, 34B and 70B parameters. Llama-2 achieves highly impressive results on language tasks, but [as a decoder model](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder) it is not well suited for vector embeddings.

Another example is OpenAI, which leverages ELMo and GPT for unsupervised pre-training to create robust general linguistic representations. Read how [OpenAI has improved language understanding with unsupervised learning](https://openai.com/blog/language-unsupervised/).

Besides Llama-2 and OpenAI, there are other prominent embedding models, ones that specialize in specific data types:

For _images_, there are breakthrough model architectures like [ResNet](https://arxiv.org/abs/1512.03385) (Residual Neural Network) that utilize skip connections to enable training extremely deep convolutional neural networks, and Google’s [Inception](https://arxiv.org/abs/1409.4842), which achieves human-level accuracy on image classification by pre-training on large labeled datasets like [ImageNet](https://www.image-net.org). By employing advanced techniques (including skip connections and concatenated convolutions), these approaches effectively learn visual features that can be transferred to downstream tasks.

In the _audio_ domain, [DeepSpeech](https://arxiv.org/abs/1412.5567) and [Wav2Vec](https://arxiv.org/abs/1904.05862) have demonstrated strong speech recognition and audio understanding through pre-training on thousands of hours of speech data. Models like Wav2Vec can generate embeddings directly from raw audio input.

For the purpose of generating vector embeddings, the top scoring models on Huggingface’s [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) are pre-trained encoder transformer models.

The primary advantage of pre-trained models is their ability to learn powerful general-purpose representations from vast datasets – for example, [CommonCrawl](https://commoncrawl.org/), which contains billions of web pages and is an ideal data source for training.

Leveraging pre-trained models as a starting point offers remarkable computational efficiency and enhanced performance, especially in scenarios where task-specific data is limited.

Still, pre-trained models have limitations; whether you should use a custom or pre-trained model depends on your task requirements.

## Pre-trained and Custom models: when to use which?

Your task's unique requirements dictate whether you should use a custom or a pre-trained model.

<!-- IMAGE 4 GOES HERE -->

Whereas _pre-trained models_ shine in domains such as text, image, and audio processing by leveraging large, generic datasets to enhance performance and efficiency, _custom models_ are essential in areas like graph embeddings, time series, and categorical data processing, where specific patterns and characteristics require tailored solutions, for example, [Time2Vec](https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e) or [GraphSAGE](https://github.com/williamleif/GraphSAGE). 

Graph Embeddings, such as [Node2Vec](https://snap.stanford.edu/node2vec/), have a variety of use cases, including recommender systems. [Graph embedding models](https://towardsdatascience.com/knowledge-graph-embeddings-101-2cc1ca5db44f) learn the relationships between entities in a knowledge graph using low-dimensional embeddings, making it more efficient to compute various similarity and inference tasks. 

There has been much debate on the use of transformer models for time series data embeddings. For example, after much initial hype over Zhou et al’s transformer-based Informer model, unveiled in [a 2021 paper](https://arxiv.org/abs/2012.07436) that won an AAAI 2021 Outstanding Paper Award, subsequent research appeared to show that such transformer-based models were outperformed on time series forecasting tasks by simpler [linear algorithms](https://machine-learning-made-simple.medium.com/why-do-transformers-suck-at-time-series-forecasting-46ae3a4d6b11), including the DLinear model. This debate is still very much alive, with transformer-based proponents maintaining that [Autoformer custom models perform better](https://huggingface.co/blog/autoformer). Whichever side of this debate you stand on, the fact remains: if you work with time series data, you need custom models.

Custom models are more useful and perform better than pre-trained models where data is atypical, structured and or proprietary. Pre-trained models, on the other hand, are typically designed for general applications and may not perfectly align with specific downstream tasks. 


So, is there a way of preserving the efficiency and performance advantages of a pre-trained model, permitting a faster go-to-market, but _also_ align it with specific downstream tasks?

One approach is _fine-tuning_.

|Aspect|Custom Models|Pre-Trained Models|
|---|---|---|
|Purpose|Tailored to specific tasks or domains|Trained on large generic datasets, then fine-tuned for specific tasks|
|Strengths|Excelling in unique domains with specialized requirements|Offering computational efficiency and enhanced performance when task-specific data is limited|
|Examples|Graph embeddings, time series analysis, category embeddings|BERT, ResNet, Wav2Vec and other models in text, image, and audio processing|
|Customization|Customization and fine-tuning to address specific nuances of data and task|Adjusted to align with task specifics through fine-tuning|
|Use Cases|Essential in areas with distinct patterns and specialized needs|Effective in domains like text, image, and audio processing where large, generic datasets enhance performance|
|Balancing Act|Achieving superior results by finding a balance between custom and pre-trained models|Recognizing that customization and fine-tuning are often necessary, even when using pre-trained models|

### Fine-tuning pre-trained models for specific tasks

The process of adapting a pre-trained model to a given task is called fine-tuning. Fine-tuning leverages the wealth of knowledge already learned by the pre-trained model on large datasets, while specializing it for the problem at hand using small-task-specific data for a specific downstream requirement.

Some examples:

Research has shown that fine-tuning Llama-2 to specific tasks can improve its performance significantly, in particular on the Llama-7b & Llama-13b parameter versions. [A case study fine-tuning the Llama-13b variant](https://www.anyscale.com/blog/fine-tuning-llama-2-a-comprehensive-case-study-for-tailoring-models-to-unique-applications) observed an increase in accuracy from 58% to 98% on functional representations, 42% to 89% on SQL generation, and 28% to 47% on GSM. Using fine-tuning on smaller models allows you to increase performance while managing the costs of running the models, ultimately making them more practical to implement into production.

Fine-tuning OpenAI on downstream NLP tasks has achieved [state-of-the-art results in commonsense reasoning, semantic similarity, and reading comprehension](https://openai.com/blog/language-unsupervised/). Further research has demonstrated GPT-4 performance on certain tasks at a level comparable to a human. But before we conclude that this amounts to artificially general intelligence (AGI), it is worth pointing out that there remain [several areas in which GPT-4 falls short of human-level performance](https://arxiv.org/pdf/2303.12712.pdf). For example, the model struggles to calibrate for confidence (i.e., it doesn’t know when it should be confident vs. when it is simply guessing). Also, the model’s context is extremely limited, it functions “statelessly” – once it’s trained, the model is fixed; it isn’t able to update itself with prior interactions or information, or adapt to a changing environment. Another frequently observed challenge is that the model regularly hallucinates, making up facts and figures, a problem compounded by its poor confidence calibration.

#### Fine-tuning adds layers and avoids overfitting

Instead of training all layers of the model from scratch, fine-tuning typically involves modifying and re-training just the last few layers of the model, specializing them for the given task. During fine-tuning, the main part of the pre-trained model remains unchanged. Fine-tuning requires only a fraction of the data that would be required to train the whole model from scratch, and it’s also computationally much cheaper.

<!-- IMAGE 5 GOES HERE -->

Fine-tuning helps avoid overfitting. Overfitting occurs when a model exposed to only small amounts of task-specific data memorizes inherent patterns, but can't generalize well. Overfitting leads to poor performance on real-world unseen data. By retraining only the last few layers of a pre-trained model, fine-tuning makes overfitting less likely, because the unchanged layers continue to provide generally useful features. In contrast, training an entire complex model on a small amount of data from scratch is very susceptible to overfitting.


To fine-tune a pre-trained model, you first need to obtain a quality dataset relevant to your specific problem. For common tasks like sentiment analysis, there are public benchmark datasets that can be used.

For example, let's take a look at some illustrative input/output pairs from the [IMDB movie reviews dataset](https://paperswithcode.com/dataset/imdb-movie-reviews), which can be employed to fine-tune a pre-trained model designed for sentiment analysis:

_Input_: "Absolutely loved this movie! The acting was superb, and the storyline kept me engaged from start to finish. Highly recommended!"

_Output_: Positive sentiment

_Input_: "This film turned out to be a significant disappointment. The plot left me puzzled, the acting was merely average, and I struggled to maintain interest. Not worth watching."

_Output_: Negative sentiment

For _proprietary_ applications, companies invest in human annotation to create labeled datasets reflecting their custom use cases and data. For successful fine-tuning, it’s key to possess a sufficient volume and variety of examples so the pre-trained model can adapt its knowledge to the new domain.

#### The fine-tuning cycle: training, hyperparameter tuning, performance measurement

The fine-tuning procedure involves repeatedly training the model on the downstream labeled data, adjusting key hyperparameters like learning rate and batch size, and evaluating performance on a held-out validation set after each iteration.

This cycle of training, hyperparameter tuning, and performance measurement is repeated until the model plateaus and stops showing significant gains on your particular dataset. The goal is to maximize metrics like accuracy and F1-score, to reach acceptable performance levels for your specific use case and data.

Libraries like [Hugging Face's Transformers](https://huggingface.co/docs/transformers/index) and [spaCy](https://spacy.io) _simplify_ this experimentation process. They provide optimized implementations of impressive pre-trained models, along with tools to rapidly run training iterations and fine-tune hyperparameters.

Fine-tuning, therefore, can adapt pre-trained models to perform more successfully on specific downstream tasks. However, even fine-tuned pre-trained models have limitations.

#### The limits of fine-tuning

Fine-tuning works well when the fine-tuning data is of the same type and modality as the pre-trained model's _initial_ training data – for example, fine-tuning on a language model using text. A fine-tuned LLM is the right solution for a straightforward language task like sentiment classification.

But on its own a fine-tuned LLM can’t properly solve broader use-cases involving more context, like product recommendations or fraud detection. 

To make a good product recommendation, for example, you need to input a variety of data of different types and modalities – including product imagery, recency (when it was launched), user preferences, and product description. To combine these factors in a way that maps to your task requirements and does it efficiently, your solution has to include _both_ a model trained to define recency – i.e., a _custom_ model – and a model that can understand relevant details from the product description and initial search query – i.e., a _pre-trained_ model (such as GPT-4). 

In other words, to get Vector Compute right, you need to develop both intricate custom models to integrate and harmonize diverse data types, _and_ pre-trained models that, with high-quality in-domain data, perform better on some in-domain tasks. Because they use general-purpose or fine-tuned representations from vast datasets, re-trained models also reduce startup costs and improve computational efficiency.

An effective vector retrieval stack is a single, unified system that assigns, coordinates, and configures custom models and pre-trained models respectively, or in combination, to handle the tasks they are designed for. Such robust homegrown solutions will be increasingly important given the broad and ever expanding application of Vector Compute to solve real world problems in a spectrum of domains, partially enumerated below.

## Applications of Vector Compute

**Personalized Search** 
In e-commerce, Vector Compute fuels tailored product recommendations, taking into account user behavior and preferences, ensuring a more personalized shopping experience.
For content-driven platforms, such as news websites, Vector Compute transforms content recommendations into a personalized journey, analyzing users' reading habits to suggest articles and topics that align with their interests.

**Recommender Systems**
Streaming platforms like Netflix employ Vector Compute to suggest movies and TV shows based on user preferences, ensuring a personalized viewing experience.
E-commerce sites use recommender systems to propose additional products by considering a user's browsing and purchase history, enhancing the overall user experience.

**RAG (Retrieval Augmented Generation)**
Chatbots for customer service use RAG to search knowledge bases and respond intelligently to user inquiries, improving issue resolution.
RAG can retrieve and synthesize answers for search engines like Wolfram Alpha by finding relevant structured data to augment natural language queries.

**Fraud & Safety**
Vector Compute is a critical tool for detecting credit card fraud. It assesses transaction data for anomalies, flagging potentially fraudulent activities.
In cybersecurity, Vector Compute identifies suspicious network activity, creating embeddings for network traffic data to detect patterns that deviate from normal behavior, potentially indicating security breaches or attacks.

**Clustering & Anomaly Detection**
Businesses employ clustering with Vector Compute to group customers based on behavior and preferences, allowing for tailored marketing strategies.
In manufacturing, Vector Compute identifies anomalies in sensor data, preventing equipment failures and maintaining product quality.

**Cybersecurity**
Vector Compute is at the forefront of intrusion detection, identifying abnormal patterns in network traffic and flagging potential security threats.
In malware detection, antivirus software employs Vector Compute to quickly recognize new instances of malicious code by creating embeddings of known malware patterns.

These example applications indicate the breadth and depth of Vector Compute’s impact – enhancing user experiences, ensuring digital safety, and enabling the relevance and quality of digital content.


## Conclusion

As Machine Learning takes on an increasingly prominent and broad role in handling and realizing value from data, more organizations in a range of domains need an effective vector retrieval stack – one that organizes your data in a way lets you quickly retrieve relevant information, and represents your data in a way that makes it easy to feed into your ML models. To build this kind of solution, you need the right combination and configuration of embedding models – connecting your [Data Sources](https://hub.superlinked.com/data-sources) to your [Vector Search & Management](https://hub.superlinked.com/vector-search).

While generic pre-trained models fail to capture the nuances of proprietary data, developing custom models from scratch is expensive and risky. Fine-tuned pre-trained models with some high-quality in-domain data can outperform large custom models, while avoiding overfitting. But even better optimization results from intricate, home-grown solutions that develop custom models and integrate them with (fine-tuned) pre-trained models into a single system – one that assigns each type of model alone, or in combination with the other, to tasks each is best suited to. The future of Vector Compute lies in the development of this kind of solution.

---
## Contributors

- [Daniel Svonava](https://www.linkedin.com/in/svonava/)
- [Paolo Perrone](https://www.linkedin.com/in/paoloperrone/)
- [Robert Turner, editor](https://robertturner.co/copyedit)
