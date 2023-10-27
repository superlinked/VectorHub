<!-- TODO: Replace this text with a summary of article for SEO -->

## Introduction

A robust vector retrieval system relies on a thoughtful selection of data sources that align with the system’s objectives. Different use cases demand different kinds and combinations of data. Conversely, different types and combinations of data make possible different kinds of operations.

If, for example, your organization runs a personalized movie recommendation system, you need data on customer preferences and viewing history. Whereas if you operate an automatic fraud detection system, your primary data source is transactional data. 

In other words, data sources form the bedrock upon which your retrieval system stands. 
It’s important, therefore, to understand what different types and combinations of data are available, and what they make possible – both generally and in your specific instance. You need a data map.


## The data map - A Mosaic of Data Types and Combinations

An organization’s data is rarely uniform. Instead, the data landscape of any organization is a mosaic that can be characterized in terms of its velocity (ranging from stream to batch data), its modality / type (spanning structured to unstructured data), and its sources (originating from in-house systems and/or third-party providers).

Where your data falls along these three dimensions (velocity, modality, source) determines what you can do with it – that is, what use cases you can accommodate, and how you should configure your data retrieval stack to meet your objectives. In order to shape your data retrieval stack optimally, you need to build a mental model of your available data and understand its potentialities.

### Example: Pinterest technology stack
Let’s look at an example that showcases how diverse data types can be combined to meet various organizational objectives: Pinterest’s technology stack.

<!---- ![](IMAGE NUMBER 1 GOES HERE) --->

1. **Event Streaming Platform**:
    - Pinterest relies on an **event streaming platform** to collect real-time event data. This platform allows them to easily collect, enrich, and transform event attributes.
    - In practice, the event streaming system likely handles real-time data ingestion, ensuring that events from user interactions, content uploads, and other actions are captured efficiently. 
    - These events can be considered **semi-structured**, as they follow a specific format (e.g., JSON) but may vary in content.

2. **Data Storage and Querying**:
    - For **persistent storage**, Pinterest utilizes technologies like **DynamoDB** (a NoSQL database service by AWS). DynamoDB provides scalability, reliability, and security for Pinterest’s massive dataset.
    - The architecture supports various querying needs:
        - Real-time exploration of raw incoming data for monitoring and analytics (semi-structured data).
        - Cached queries that can instantly be loaded into applications and reports for customer-facing metrics (structured data).
        - The data stored in DynamoDB likely includes:
            - User profiles (structured data).
            - Event logs (semi-structured data).
            - Images and other media (unstructured data).

3. **Data Management Culture**:
    - Pinterest has a strong culture of **data-driven decision-making**. They likely store a wide variety of data types related to user behavior, content preferences, engagement metrics, and more.
    - Their data stack includes mechanisms to handle:
        - Structured data (e.g., user profiles): This includes well-defined attributes with specific formats.
        - Semi-structured data (e.g., event logs): These follow a general structure but may have varying fields or additional context.
        - Unstructured data (e.g., images): These are typically files without a fixed schema.

4. **Querying and Analysis**:
    - Pinterest’s engineers use **Querybook**, a collaborative big data hub that allows internal users to query and analyze data efficiently. Each **DataDoc** in Querybook consists of cells (text, query, or chart) that facilitate exploration and visualization.
    - The system likely supports complex queries for trend analysis, anomaly detection, recommendation algorithms, and personalized content delivery.

5. **Additional Components**:
    Beyond event streaming and storage, Pinterest’s stack likely includes:
	  - Frontend technologies (for their visual discovery engine).
	  - Backend frameworks (for serving APIs).
	  - Infrastructure tools (for scaling).
	  - Machine learning libraries (for recommendation models).

Keep in mind that this is a high-level overview based on available information on Stackshare. For precise details about Pinterest’s specific data types and components, you may want to explore their engineering resources directly, [here](https://medium.com/pinterest-engineering/looking-inside-the-technology-that-powers-pinterest-2e8bd1cfc329) and [here](https://medium.com/pinterest-engineering/open-sourcing-querybook-pinterests-collaborative-big-data-hub-ba2605558883).


## Understanding Velocity

The choice of data processing velocity is pivotal in determining the kind of data retrieval and vector compute tasks you can perform. Different velocities offer distinct advantages and make different use cases possible. Here's a breakdown of the three primary velocity categories:

1. **Batch Processing**
- **Technologies**: Apache Spark, Hadoop MapReduce
- **Example Data**: Historical sales data, customer records, monthly financial reports. 
- **Properties**: Batch processing involves processing data in fixed-size chunks or batches, typically scheduled at specific intervals (e.g., daily, weekly, or monthly). It provides the ability to handle large volumes of data efficiently but lacks real-time responsiveness. For instance, a product recommendation system for an online store might opt for batch updates. Updating recommendations daily or every few hours may suffice, given the trade-off between customer value and computational resources.
- **Formats**: Common formats include CSV, Parquet, Avro, or any format that suits the specific data source.
- **Databases:** Databases such as PostgreSQL, MySQL, and MongoDB commonly store structured data that can be batch-processed for your retrieval system.
- **ETL-Able Systems:** Systems like Magento for e-commerce or MailChimp for marketing can be employed to extract and batch-process this data for your retrieval stack.

2. **Micro-Batch Processing**
- **Technologies**: Apache Flink, Apache Beam
- **Example Data**: Social media posts, IoT sensor data, small-scale e-commerce transactions.
- **Properties**: Micro-batch processing is a compromise between batch and stream. It processes data in smaller, more frequent batches, allowing for near-real-time updates. It's suitable for use cases that require a balance between real-time processing and resource efficiency.
- **Formats**: Often similar to batch processing with data structured in formats like JSON or Avro.

3. **Stream Processing**
- **Technologies**: Apache Kafka, Apache Flink, Apache Storm
- **Example Data**: Social media feeds, stock market transactions, sensor readings, clickstream data. A credit card company aiming to detect fraudulent transactions in real-time benefits from stream data. Real-time detection can prevent financial losses and protect customers from fraudulent activities.
- **Properties**: Stream processing handles data in real-time, making it highly dynamic. It's designed to support immediate updates and changes, making it ideal for use cases that require up-to-the-second insights.
- **Formats**: Data in stream processing is often in JSON, Avro, or other formats optimized for fast data transmission.
- **Databases:** Data Warehouses like Snowflake, Redshift, or BigQuery often function in near-real time. Your retrieval stack should be designed to accommodate the slight delays associated with these data warehouses.


It is also possible to combine batch and stream processing. This enables you to leverage both the immediacy of real-time updates and the depth of historical data. But reconciling stream and batch processes in a single system introduces some trade-off decisions. How you combine batch and stream processes to meet your use case requirements requires careful consideration.


## Reconciling Streaming and Batch Sources 

When choosing and configuring the data sources for a vector retrieval system, it’s important to consider tradeoffs between data velocity, complexity, and model size. 


<!---- ![](IMAGE NUMBER 2 GOES HERE) --->

### Velocity vs Complexity Tradeoff

- **Streaming data**: This option allows for real-time vector computations but comes with constraints on model complexity due to latency concerns. Streaming data may require simpler embeddings or lookups.
- **Batch data**: As opposed to streaming data, batch data enables complex models, including large transformers. However, with batch data, updates are less frequent, making it suitable for asynchronous needs.
- **Hybrid Approaches**: These merge streaming filters or lookups with batch retraining, offering responsiveness along with in-depth analysis.

| **Aspect** | **Streaming Data** | **Batch Data** | **Hybrid Approaches**|
|------------------------|------------------------|------------------------|------------------------|
|**Model Complexity**| Constraints on model complexity due to latency | Enables complex models, including large transformers | Balances streaming with batch retraining for analysis |
|**Update Frequency**| Real-time updates | Less frequent updates, suitable for asynchronous use | Combines streaming with batch for responsiveness and depth |
|**Cost-effectiveness**| Higher resource requirements | More cost-effective due to lower resource requirements | A balance of cost-effectiveness with real-time features |
|**Data Consistency**| Real-time updates, potential data consistency | Ensures data consistency | A balance of real-time and consistency |
|**Timeliness & Relevance**| Timely but may have lower content value | Timely and relevant | Timely and relevant |

How you configure your data sources to balance velocity and complexity ultimately manifests in the types of architectures you use in your vector search system.

**Example Architectures**

Within a vector search system, there are three basic architecture types that balance velocity and complexity in different ways:

- **Streaming**: Think of architectures like Recurrent Neural Networks (RNNs), shallow neural networks, and indexing/retrieval models.
- **Batch**: Large pretrained transformers, custom deep learning models, and even architecture search models.
- **Hybrid**: Combine indexing with periodic retraining or deploying a two-stage filtering and analysis setup.

Choosing the right architecture depends on your goals:

**Key Considerations**
- **Synchronicity Needs:** Determine whether real-time updates are essential or if daily or weekly updates suffice.
- **Infrastructure Availability:** Check if you have specialized streaming hardware at your disposal.
- **Model Accuracy Requirements:** Consider whether approximate real-time performance meets your needs.


The trade-off between velocity and complexity is not the only important consideration in building your vector retrieval system. You also need to balance model size with response time.

### Model Size vs Response Time Tradeoff

- **Streaming:** Typically involves models under 100MB in size, such as shallow networks, distillation models, and compact neural networks like MobileNet.
- **Batch:** Here, you're looking at models exceeding 1GB in size, including giant transformers and custom deep networks.
- **Hybrid:** This is the sweet spot, combining smaller streaming models with larger batch models to find an optimal balance between size and speed.

As with velocity and complexity, how you balance size with response time is consequential for the types of architectures you use in your vector search system.

**Example Architectures**
- **Streaming**: You might come across RNNs, CNNs with fewer than 10 layers, and efficient models like MiniLM or TinyML. Even simple dense networks or XGBoost can fit the bill.
- **Batch Architectures**: Think of big players like BERT, GPT-2/3, T5, EfficientNet, and custom transformers. Also, don't forget about neural architecture search models.
- **Ensemble and Stacked Models**: In some cases, using ensemble and stacked models can provide the best of both streaming and batch architectures, combining model sizes and optimizing for various needs.

### Additional Key Considerations for Vector Retrieval System Design
Besides the velocity-complexity and model size-response time tradeoffs above, there are some additional considerations pivotal to ensuring that you meet your vector retrieval system’s objectives:
- **Response Latency Constraints**: Always keep an eye on response latency constraints to make sure your models can meet real-time demands.
- **Model Accuracy Requirements**: Consider your precise model accuracy requirements and whether your infrastructure can handle the chosen model sizes effectively.
- **Data Value**: Recognize the value of processed data, as high-value data holds critical information that can lead to valuable insights and informed decision-making.

In short, keeping these key considerations in mind while you balance velocity with complexity, and model size with response times will help ensure that you configure your specific vector retrieval system optimally.


### Kappa vs. Lambda:

In the context of real-time data processing, the choice between Kappa and Lambda architectures involves evaluating trade-offs between velocity, complexity, and data value:

- **Kappa Architecture**: Emphasizes real-time processing. Data is ingested as a continuous stream and processed in real-time to support immediate decisions. This approach is well-suited for use cases with high velocity, such as fraud detection systems in financial institutions.

- **Lambda Architecture**: Combines batch and real-time processing. Data is ingested both as a stream and in batch mode. This approach offers the advantage of capturing high-velocity data while maintaining the value of historical data. It's suitable when there's a need for comprehensive insights that take into account both real-time and historical data.

| Aspect | Kappa Architecture | Lambda Architecture |
|------------------------|--------------------------------------------------------|-------------------------------------------------------------|
| *Real-Time Processing* | Emphasizes real-time data processing. Data is ingested as a continuous stream and processed in real-time to support immediate decisions. | Combines batch and real-time processing. Data is ingested as both a stream and batch, allowing for comprehensive analysis. |
| *Simplicity* | Offers a simpler architecture, with a single processing pipeline for real-time data. | Has a more complex architecture, with separate processing pipelines for real-time and batch data. |
| *Scalability* | Highly scalable for handling high-velocity data streams. Scales well with growing data loads. | Scalable but may involve more management due to the dual processing pipelines. |
| *Latency* | Provides low-latency processing, making it suitable for use cases that require immediate decisions, such as fraud detection. | Offers lower latency for real-time processing, but higher latency for batch processing. |
| *Complex Analysis* | May be less suitable for complex batch analysis. Real-time data is the primary focus. | Supports both real-time and batch processing, allowing for more comprehensive and complex analysis. |
| *Data Consistency* | May sacrifice some degree of data consistency to prioritize real-time processing. | Ensures strong data consistency between real-time and batch views, making it suitable for data analytics and historical comparisons. |
| *Use Cases* | Ideal for use cases with high velocity, such as real-time monitoring, fraud detection, and sensor data processing. | Suitable for applications where historical data analysis, complex event processing, and reconciling real-time and batch views are required. |

## Data Modality / Type
Whether your data is structured, unstructured, or a hybrid is a crucial consideration when evaluating data sources. The nature of the data source your vector retrieval system uses shapes how that data should be managed and processed.


### Unstructured Data
Unstructured data encompasses a wide variety of information that doesn't adhere to a fixed structure or definition. This type of data is often characterized by its raw, unordered, and noisy nature. Examples of unstructured data include text data, image data, audio data, and video data. Let's take a closer look at each type:

**Text Data:**
- **Example Data:** Social media posts, news articles, chat transcripts, product reviews.
- **Typical Formats:** Plain text, JSON, XML, HTML, PDF, CSV (for tabular text data).
- **Datasets:**
    - Kaggle: [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
    - Hugging Face: [Text Classification](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=trending)

**Image Data:**
- **Example Data:** Photographs, medical images, satellite imagery.
- **Typical Formats:** JPEG, PNG, TIFF.
- **Datasets:**
    - Kaggle: [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
    - Hugging Face: [Image Segmentation]([https://huggingface.co/transformers/main_classes/datasets.html](https://huggingface.co/datasets?task_categories=task_categories:image-segmentation&sort=trending))

**Audio Data:**
- **Example Data:** Speech recordings, music, environmental sounds.
- **Typical Formats:** WAV, MP3, FLAC.
- **Datasets:**
    - Kaggle: [Urban Sound Classification](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
    - Hugging Face: [Audio Classification](https://huggingface.co/datasets?task_categories=task_categories:audio-classification&sort=trending)

**Video Data:**
- **Example Data:** Movie clips, surveillance footage, video streams.
- **Typical Formats:** MP4, AVI, MOV.
- **Datasets:**
    - Kaggle: [YouTube-8M Segments - Video Classification](https://www.kaggle.com/c/youtube8m)
    - Hugging Face: [Video Classification](https://huggingface.co/datasets?task_categories=task_categories:video-classification&sort=trending)


### Multi-Task Evaluation (Benchmark) datasets
GLUE (General Language Understanding Evaluation) and SuperGLUE (Super General Language Understanding Evaluation) are multi-task benchmark datasets designed to assess model performance across various NLP tasks.

1. **GLUE**
    - Description: GLUE contains diverse NLP tasks like sentiment analysis, text classification, and textual entailment. These tasks rely on semi-structured text input-output pairs rather than completely free-form text.
    - Format: JSON/CSV with text snippets and corresponding labels.
    - [Hugging Face GLUE Datasets](https://huggingface.co/datasets/glue)
2. **SuperGLUE**
    - Description: SuperGLUE introduces more complex language tasks like question answering and coreference resolution, also based on semi-structured text.
    - Format: JSON/CSV with text inputs and labels.
    - [Hugging Face SuperGLUE Datasets](https://huggingface.co/datasets/super_glue)

While GLUE and SuperGLUE are useful for benchmarking language models, it would be inaccurate to describe them solely as unstructured text datasets, since many of the tasks involve semi-structured input-output pairs.


### Structured Data
Structured data adheres to predefined formats, organized categories, and a fixed set of data fields, making it highly organized and readily analyzable. Unlike unstructured data, which is raw and lacks a specific format, structured data is well-ordered and typically fits neatly into tables or databases.

Here are some examples of structured data types, links to relevant datasets, typical formats, and considerations relevant to each type of structured data:

**Tabular Data:**
- **Example Data:** Sales records, customer information, financial statements.
- **Typical Formats:** CSV, Excel spreadsheets, SQL databases.
- **Datasets:**
   - [Kaggle Datasets](https://www.kaggle.com/datasets): Kaggle offers a wide range of structured datasets covering various domains.
   - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php): UCI's repository provides many structured datasets for machine learning.
- **Considerations:** When working with tabular data, consider data quality, missing values, and the choice of columns or features relevant to your analysis. You may need to preprocess data and address issues such as normalization and encoding of categorical variables.
- **Systems:** Structured data often lives in relational database management systems (RDBMS) like MySQL, PostgreSQL, or cloud-based solutions like AWS RDS.

**Graph Data:**
- **Example Data:** Social networks, organizational hierarchies, knowledge graphs.
- **Typical Formats:** Graph databases (e.g., Neo4j), edge-list or adjacency matrix representation.
- **Datasets:**
   - [Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/data/): Offers a collection of real-world network datasets.
   - [KONECT](http://konect.cc/networks/): Provides a variety of network datasets for research.
- **Considerations:** In graph data, consider the types of nodes, edges, and their attributes. Pay attention to graph algorithms for traversing, analyzing, and extracting insights from the graph structure.
- **Systems:** Graph data is often stored in graph databases, but it can also be represented using traditional RDBMS with specific schemas for relations.

**Time Series Data:**
- **Example Data:** Stock prices, weather measurements, sensor data.
- **Typical Formats:** CSV, JSON, time-series databases (e.g., InfluxDB).
- **Datasets:**
   - [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/): Covers economic and research data from various countries, including the USA, Germany, and Japan.
   - [UCI Time Series Data Repository](https://timeseriesdata.ucr.edu/): Provides various time series datasets for research and benchmarking.
- **Considerations:** Time series data requires dealing with temporal aspects, seasonality, trends, and handling irregularities. It may involve time-based feature engineering and modeling techniques like ARIMA or LSTM.
- **Systems:** Time series data can be stored in specialized time-series databases (e.g., InfluxDB, TimescaleDB) or traditional databases with timestamp columns.

**Spatial Data:**
- **Example Data:** Geographic information, maps, GPS coordinates.
- **Typical Formats:** Shapefiles (SHP), GeoJSON, GPS coordinates in CSV.
- **Datasets:**
   - [Natural Earth Data](https://www.naturalearthdata.com/): Offers free vector and raster map data.
   - [OpenStreetMap (OSM) Data](https://www.openstreetmap.org/): Provides geospatial data for mapping and navigation.
- **Considerations:** Spatial data often involves geographic analysis, mapping, and visualization. Understanding coordinate systems, geospatial libraries, and map projections is important.
- **Systems:** Spatial data can be stored in specialized Geographic Information Systems (GIS) or in databases with spatial extensions (e.g., PostGIS for PostgreSQL).

Each of these structured data types comes with its own unique challenges and characteristics. In particular, it’s important to pay attention to data quality and pre-processing to make choices that are aligned with your vector retrieval system.


## Maintaining data currency using Change Data Capture (CDC)

In any data retrieval system, a key requirement is ensuring vectors accurately reflect the latest state of source data. As underlying data changes – e.g., product updates, user activities, sensor readings – corresponding vector representations must also be kept current.

One approach to keeping data current is batch recomputation – periodically rebuilding all vectors from scratch as new data arrives. But batch recomputation ignores incremental changes between batches.

Change Data Capture (CDC) provides a more efficient alternative – capturing granular data changes as they occur and incrementally updating associated vectors. Using CDC, an e-commerce site, for example, can stream new customer interactions to rapidly update user vectors. Or a real-time anomaly detection system can employ CDC to incorporate sensor measurement changes to adapt equipment representations.

By consuming change events from databases and other sources, and then propagating those to downstream systems, CDC enables low latency vector updates in response to data modifications rather than bloated rebuilds. It’s possible, as a result, to maintain up-to-date vectors vital for real-time applications and mirror source system states as they shift. In these ways, CDC plays an integral role in keeping vectorized data aligned with changing realities.

The following visualization shows how CDC can be implemented within a streaming data retrieval system. A primary database emits CDC into a queue, which is then consumed like any other streaming data source:

1. 1. **Primary Database**:
    - The primary database, which can be MySQL, PostgreSQL, SQL Server, or other database management system, serves as the source of data changes.
    - It continuously captures changes to its tables, including inserts, updates, and deletes, in its transaction logs.
2. **Change Data Capture (CDC)**:
    - CDC technology acts as the bridge between the primary database and your retrieval system, detecting and managing incremental changes at the data source.
    - A dedicated **capture process** is responsible for reading these changes from the transaction log of the primary database.
    - These changes are then organized and placed into corresponding **change tables** within the CDC system.
    - These change tables essentially store metadata about what data has changed and when those changes occurred.
3. **Queue (Message Broker)**:
    - The CDC system efficiently transmits these organized changes into a message queue, which can be implemented using technologies like Apache Kafka or RabbitMQ.
    - The message queue acts as a buffer and plays a crucial role in ensuring the reliable delivery of change events to downstream consumers.
4. **Streaming Data Consumers**:
    - Applications, analytics pipelines, and other data consumers subscribe to the message queue.
    - They actively consume change events in real time, just as they would with any other streaming data source.
    - These consumers treat CDC data as a continuous flow of real-time data, making it readily available for processing and analysis.

<!---- ![](IMAGE NUMBER 3 GOES HERE) --->

## Conclusion 

Building an effective vector retrieval stack requires a deep understanding of its objectives,  available data sources, and how these interact in the context of your retrieval system. Building a data map should help you a) balance velocity and complexity, model size and response time, and b) consider response latency, model accuracy, and the value of processed data.

Keeping these considerations in mind will help you effectively navigate the landscape of vector retrieval.

## Contributors

- [Daniel Svonava](https://www.linkedin.com/in/svonava/)
- [Paolo Perrone](https://www.linkedin.com/in/paoloperrone/)
