# Data Modality

## Data Modality / Type

Whether your data is structured, unstructured, or hybrid is crucial when evaluating data sources. The nature of the data source your vector retrieval system uses shapes how that data should be managed and processed.

### Unstructured Data

Unstructured data encompasses a wide variety of information that doesn't adhere to a fixed structure or definition. This data type is often characterized by its raw, unordered, and noisy nature. Examples of unstructured data include natural language text, image, audio, and video data. Let's take a closer look at each type:

**Text Data**
- **Example Data:** Social media posts, news articles, chat transcripts, product reviews.
- **Typical Formats:** Plain text, JSON, XML, HTML, PDF, CSV (for tabular text data).
- **Datasets:**
    - Kaggle: [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
    - Hugging Face: [Text Classification](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=trending)

**Image Data**
- **Example Data:** Photographs, medical images, satellite imagery, generative AI-created images.
- **Typical Formats:** JPEG, PNG, TIFF.
- **Datasets:**
    - Kaggle: [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
    - Github: [Unsplash 4.8M Photos, Keywords, Searches](https://github.com/unsplash/datasets)

**Audio Data**
- **Example Data:** Speech recordings, music, environmental sounds.
- **Typical Formats:** WAV, MP3, FLAC.
- **Datasets:**
    - Kaggle: [Urban Sound Classification](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
    - Hugging Face: [Audio Classification](https://huggingface.co/datasets?task_categories=task_categories:audio-classification&sort=trending)

**Video Data**
- **Example Data:** Movie clips, surveillance footage, video streams.
- **Typical Formats:** MP4, AVI, MOV.
- **Datasets:**
    - Kaggle: [YouTube-8M Segments - Video Classification](https://www.kaggle.com/c/youtube8m)
    - Hugging Face: [Video Classification](https://huggingface.co/datasets?task_categories=task_categories:video-classification&sort=trending)

#### Multi-Task Evaluation datasets

When building information retrieval systems, choosing the right dataset for evaluation is key. As a rule, you should always do this using your company’s or product’s own data.

However, there are also instances when using a 3rd party dataset for evaluation is possible, preferable, or even the only option. Most commonly, this occurs when:
Your own data quality or volume is insufficient for robust evaluation.
You want to standardize your evaluation results by running your system against a standardized evaluation dataset available in the market.

Let’s look at a couple of evaluation dataset examples for language models:

GLUE (General Language Understanding Evaluation) and SuperGLUE (Super General Language Understanding Evaluation) are multi-task benchmark datasets designed to assess model performance across various NLP tasks.

**GLUE** (General Language Understanding Evaluation)
- **Description**: GLUE contains diverse NLP tasks like sentiment analysis, text classification, and textual entailment. These tasks rely on semi-structured text input-output pairs rather than completely free-form text.
- **Format**: JSON/CSV with text snippets and corresponding labels.
- [Hugging Face GLUE Datasets](https://huggingface.co/datasets/glue)  
  
**SuperGLUE** (Super General Language Understanding Evaluation)
- **Description**: SuperGLUE introduces more complex language tasks like question answering and coreference resolution, which are also based on semi-structured text.
- **Format**: JSON/CSV with text inputs and labels.
- [Hugging Face SuperGLUE Datasets](https://huggingface.co/datasets/super_glue)

While GLUE and SuperGLUE are useful for benchmarking language models, it would be inaccurate to describe them solely as unstructured text datasets, since many tasks involve semi-structured input-output pairs.

### Structured Data

All major enterprises today run on structured data. Structured data adheres to predefined formats, organized categories, and a fixed set of data fields, making it easier to start working with. But, beware, even structured datasets may be of poor quality, with many missing values or poor schema compliance.

Here are some examples of structured data types, links to example datasets, typical formats, and considerations relevant to each type of structured data:

**Tabular Data**
- **Example Data:** Sales records, customer information, financial statements.
- **Typical Formats:** CSV, Excel spreadsheets, SQL databases.
- **Datasets:**
   - [Kaggle Datasets](https://www.kaggle.com/datasets): Kaggle offers a wide range of structured datasets covering various domains.
   - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php): UCI's repository provides many structured datasets for machine learning.
- **Considerations:** When working with tabular data, consider data quality, missing values, and the choice of columns or features relevant to your analysis. You may need to preprocess data and address issues such as normalization and encoding of categorical variables.
- **Systems:** Structured data often lives in relational database management systems (RDBMS) like MySQL, PostgreSQL, or cloud-based solutions like AWS RDS.

**Graph Data**
- **Example Data:** Social networks, organizational hierarchies, knowledge graphs.
- **Typical Formats:** Graph databases (e.g., Neo4j), edge-list or adjacency matrix representation.
- **Datasets:**
   - [Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/data/): Offers a collection of real-world network datasets.
   - [KONECT](http://konect.cc/networks/): Provides a variety of network datasets for research.
- **Considerations:** In graph data, consider the types of nodes, edges, and their attributes. Pay attention to graph algorithms for traversing, analyzing, and extracting insights from the graph structure.
- **Systems:** Graph data is often stored in graph databases like [Neo4j](https://neo4j.com/), [ArangoDB](https://github.com/arangodb/arangodb), or [Apollo](https://www.apollographql.com/), but it can also be represented using traditional RDBMS with specific schemas for relations.

**Time Series Data**
- **Example Data:** Stock prices, weather measurements, sensor data.
- **Typical Formats:** CSV, JSON, time-series databases (e.g., InfluxDB).
- **Datasets:**
   - [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/): Covers economic and research data from various countries, including the USA, Germany, and Japan.
   - [The Google Trends Dataset](https://trends.google.com/trends/?ref=hackernoon.com)
- **Considerations:** Time series data requires dealing with temporal aspects, seasonality, trends, and handling irregularities. It may involve time-based feature engineering and modeling techniques, like [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) or other sequential models, like [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory).
- **Systems:** Time series data can be stored in specialized time-series databases (e.g., InfluxDB, TimescaleDB, [KX](https://kx.com/)) or traditional databases with timestamp columns.

**Spatial Data**
- **Example Data:** Geographic information, maps, GPS coordinates.
- **Typical Formats:** Shapefiles (SHP), GeoJSON, GPS coordinates in CSV.
- **Datasets:**
   - [Natural Earth Data](https://www.naturalearthdata.com/): Offers free vector and raster map data.
   - [OpenStreetMap (OSM) Data](https://www.openstreetmap.org/): Provides geospatial data for mapping and navigation.
- **Considerations:** Spatial data often involves geographic analysis, mapping, and visualization. Understanding coordinate systems, geospatial libraries, and map projections is important.
- **Systems:** Spatial data can be stored in specialized Geographic Information Systems (GIS) or in databases with spatial extensions (e.g., PostGIS for PostgreSQL).

**Logs Data**
- **Example Data:** Some examples of different logs include: system event logs that monitor traffic to an application, detect issues, and record errors causing a system to crash, or user behaviour logs, which track actions a user takes on a website or when signed into a device.
- **Typical Formats:** [CLF](https://en.wikipedia.org/wiki/Common_Log_Format) or a custom text or binary file that contains ordered (timestamp action) pairs.
- **Datasets:**
    - [loghub](https://github.com/logpai/loghub): A large collection of different system log datasets for AI-driven log analytics.
- **Considerations:** How long you want to save the log interactions and what you want to use them for – i.e. understanding where errors occur, defining “typical” behaviour – are key considerations for processing this data. For further details on what to track and how, see this [Tracking Plan](https://segment.com/academy/collecting-data/how-to-create-a-tracking-plan/) course from Segment.
- **Systems:** There are plenty of log management tools, for example [Better Stack](https://betterstack.com/logs), which has a pipeline set up for ClickHouse, allowing real-time processing, or [Papertrail](https://www.papertrail.com/), which can ingest different syslogs txt log file formats from Apache, MySQL, and Ruby.  


Each of these structured data types comes with its own unique challenges and characteristics. In particular, paying attention to data quality and pre-processing is important to make choices aligned with your vector retrieval system.


## Keeping your retrieval stack up to date with Change Data Capture

In any data retrieval system, a key requirement is ensuring the underlying representations (i.e. vector embeddings) accurately reflect the latest state of source data. As underlying data changes – e.g., product updates, user activities, sensor readings – corresponding vector representations must also be kept current.

One approach to updating your data is batch recomputation – periodically rebuilding all vectors from scratch as the new data piles up. But batch recomputation ignores incremental changes between batches.

**Change Data Capture** (CDC) provides a more efficient alternative – capturing granular data changes as they occur and incrementally updating associated vectors. Using CDC, an e-commerce site, for example, can stream product catalog changes to update product vectors rapidly. Or, a real-time anomaly detection system can employ CDC to incorporate user account changes to adapt user vectors. As a result, CDC plays an integral role in keeping vectorized data aligned with changing realities.

The visualization below shows how CDC can be implemented within a streaming data retrieval system. A primary database emits CDC into a queue, which is then consumed like any other streaming data source:

1. **Primary Database**:
    - The primary database, which can be MySQL, PostgreSQL, SQL Server, or other database management system, serves as the source of data changes.
    - It continuously captures changes to its tables, including inserts, updates, and deletes, in its transaction logs.
2. **Change Data Capture (CDC)**:
    - CDC technology, for example [Kebola](https://www.keboola.com/) or [Qlik Replicate](https://www.qlik.com/us/products/qlik-replicate), acts as the bridge between the primary database and your retrieval system, detecting and managing incremental changes at the data source.
    - A dedicated **capture process** is responsible for reading these changes from the transaction log of the primary database.
    - These changes are then organized and placed into corresponding **change tables** within the CDC system.
    - These change tables store metadata about what data has changed and when those changes occurred.
3. **Queue (Message Broker)**:
    - The CDC system efficiently transmits these organized changes into a message queue, which can be implemented using technologies like Apache Kafka or RabbitMQ.
    - The message queue acts as a buffer and is crucial in ensuring the reliable delivery of change events to downstream consumers.
4. **Streaming Data Consumers**:
    - Applications, analytics pipelines, and other data consumers subscribe to the message queue.
    - Streaming data consumers actively consume change events in real-time, just as they would with any other streaming data source.
    - These consumers treat CDC data as a continuous flow of real-time data, making it readily available for processing and analysis.

![CDC with streaming data](assets/building_blocks/data_sources/bb1-3.png)
