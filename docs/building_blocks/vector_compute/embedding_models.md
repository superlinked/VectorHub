At the core of Vector Compute are embedding models – machine learning models applied to raw data to generate vector embeddings.

Embedding models turn features extracted from high-dimensional data, with large numbers of attributes or dimensions, like text, images, or audio, into lower-dimensional but dense mathematical representations – i.e., vectors. You can apply embedding models to structured data like tabular datasets or graphs. 

An e-commerce company, for example, might ingest thousands of rows of user activity logs with features such as date, product viewed, purchase amount, and so on. Embedding models encode these heterogeneous features as embedding vectors, with 100s of dimensions highlighting otherwise latent relationships and patterns, thereby enabling nearest neighbor search, clustering, or further modeling.

The mathematical vector representations that result from the vector embedding conversion process are readily **amenable to processing by machine learning algorithms**.

**Word** embeddings, for example, capture nuanced linguistic similarities between terms that would be opaque to computers operating on just raw text strings. 

**Image** embeddings (vector representations of images) can be fed, along with metadata, like labels, into traditional deep learning models to train them for tasks like image classification. Without image embedding, feeding images into deep learning models requires a specialized feature extraction step and specific model architecture – like a [convolutional neural network or a restricted Boltzmann machine](https://www.hindawi.com/journals/am/2022/3351256/). 

However, you can *use image embedding to work on computer vision tasks yourself* – you don’t need specialized computer vision engineers to get started.

Furthermore, compared to complex raw data, vectors have far fewer dimensions, making them more efficient for tasks like storage, transfer, and retrieval. A low dimensional vector (i.e., an array of numerical values) such as [102, 000, 241, 317, 004] encodes a wealth of semantic features and relationships. **The continuity of the embedding space enables gradient-based optimization, which is central to Machine Learning modelling**.

![Image Embeddings](assets/building_blocks/vector_compute/bb2-4.png)

In summary, embedding models can efficiently turn obscure raw data into structured vector representations that reveal otherwise hidden patterns – patterns that computers can effectively model using ML.

But what does the embedding process look like?

### The embedding process visualized

Embedding maps data onto a high-dimensional vector space, often between 500-2000 dimensions, depending on the complexity of the underlying data. However, for visualization purposes, popular dimensionality reduction techniques like [UMAP](https://umap-learn.readthedocs.io/en/latest/) (Uniform Manifold Approximation and Projection) or [t-SNE](https://lvdmaaten.github.io/tsne/) (t-Distributed Stochastic Neighbor Embedding) can be used to project these dense vectors into 2D scatterplots that approximate the relative distances and relationships between data points, as shown in [this embedding projector](https://projector.tensorflow.org).

![Embedding process](assets/building_blocks/vector_compute/bb2-5.png)

If, for example, your vector space represented restaurant reviews, these could be embedded based on ratings, food quality, and customer responses. **Similar reviews would cluster together in the vector space, while dissimilar ones would be further apart**. Reviews mentioning “good food” and “tasty dishes” would be close together, as their meaning is similar.

### A new kind of model: Pre-trained Embedding

Until the early to mid-2010s, embedding was handled exclusively by ML teams building custom models from scratch. Custom models rely heavily on large volumes of task-specific data and require expert data scientists to develop and refine for months.

Recent advances in [transfer learning](https://ai.plainenglish.io/transfer-learning-in-deep-learning-leveraging-pretrained-models-for-improved-performance-b4c49f2cd644) research have enabled _another_ kind of model, pre-trained on large, broad datasets, using more computational power than 99.9% of companies have available. Custom models continue to play a crucial role in Vector Compute, excelling at some tasks. 

However, pre-trained models have performed better than custom models on others – in particular, text, image, and audio embedding. And, importantly, compared to building your own models, pre-trained models make it **easier and faster** to get to a working solution for your use case.

Let’s look at some examples:

[Llama-2](https://ai.meta.com/llama/), developed by Meta, is a set of LLMs pre-trained on a publically available corpus of data, with variants trained on 7B, 13B, 34B and 70B parameters. Llama-2 achieves highly impressive results on language tasks, but [as a decoder model](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder) it is not well suited to vector embeddings.

Another example is OpenAI, which leverages ELMo and GPT for unsupervised pre-training to create robust general linguistic representations. Read how [OpenAI has improved language understanding with unsupervised learning](https://openai.com/blog/language-unsupervised/).

Besides Llama-2 and OpenAI, there are **other prominent embedding models**, ones that **specialize in specific data types**:

For **images**, there are breakthrough model architectures like [ResNet](https://arxiv.org/abs/1512.03385) (Residual Neural Network) that utilize skip connections to enable training extremely deep convolutional neural networks, and Google’s [Inception](https://arxiv.org/abs/1409.4842), which achieves human-level accuracy on image classification by pre-training on large labeled datasets like [ImageNet](https://www.image-net.org). By employing advanced techniques (including skip connections and concatenated convolutions), these approaches effectively learn visual features that can be transferred to downstream tasks.

In the **audio** domain, [DeepSpeech](https://arxiv.org/abs/1412.5567) and [Wav2Vec](https://arxiv.org/abs/1904.05862) have demonstrated strong speech recognition and audio understanding through pre-training on thousands of hours of speech data. Models like Wav2Vec can generate embeddings directly from raw audio input.

To generate vector embeddings, the top-scoring models on Huggingface’s [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) are pre-trained encoder transformer models.

The **primary advantage of pre-trained models is their ability to learn powerful general-purpose representations from vast datasets** – for example, [CommonCrawl](https://commoncrawl.org/), which contains billions of web pages and is an ideal data source for training.

Leveraging pre-trained models as a starting point offers remarkable computational efficiency and enhanced performance, **especially in scenarios where task-specific data is limited**.

Still, pre-trained models have limitations; whether you should use a custom or pre-trained model depends on your task requirements.
