# 1.1 Data Velocity

The choice of data processing velocity is pivotal in determining the kind of data retrieval and vector compute tasks you can perform. Different velocities offer distinct advantages and make different use cases possible. Here's a breakdown of the three primary velocity categories:

### 1. **Batch Processing**

| Overview | Example |
| ---- | ------------- |
|**Technologies**| Open source tools like Apache Spark, or proprietary batch pipeline systems in Google BigQuery, AWS Batch, or Snowflake.| 
|**Example Data**| Historical sales data, customer records, monthly financial reports.|
|**Properties**| Batch processing involves processing data in fixed-size chunks or batches, typically scheduled at specific intervals (e.g., daily, weekly, or monthly). It can handle large volumes of data efficiently but lacks real-time responsiveness. For instance, a product recommendation system for an online store email newsletter might opt for batch updates. Updating recommendations once per week may suffice, given that the email is also sent once a week.|
|**Data Formats**|Common formats include CSV, Parquet, Avro, or any format that suits the specific data source.|
|**Databases**| Data storage systems like AWS S3 / Google Cloud Storage and document databases like MongoDB commonly store data that can be batch-processed for your retrieval system.|
|**ETL-Able Systems**| Systems like Magento for e-commerce or MailChimp for marketing are typical sources for batch accessed and processed data in your retrieval stack. |

### 2. **Micro-Batch Processing**

| Overview | Example |
| ---- | ------------- |
|**Technologies**| Apache Spark Structured Streaming, Apache Flink, Apache Beam.|
|**Example Data**| Social media posts, IoT sensor data, small-scale e-commerce transactions.|
|**Properties**| Micro-batch processing compromises batch and stream. It processes data in smaller, more frequent batches, allowing for near-real-time updates. It's suitable for use cases that balance real-time processing and resource efficiency.|
|**Formats**| Often similar to batch processing, with data structured in formats like JSON or Avro.|

### 3. **Stream Processing**

| Overview | Example |
| ---- | ------------- |
|**Technologies**| Apache Kafka, Apache Storm, Amazon Kinesis, [hazelcast](https://hazelcast.com/), [bytewax](https://github.com/bytewax/bytewax), [quix](https://quix.io/), [streamkap](https://streamkap.com/), [decodable](https://www.decodable.co/).|
|**Example Data**| Social media feeds, stock market transactions, sensor readings, clickstream data, ad requests and responses. A credit card company aiming to detect fraudulent transactions in real-time benefits from streaming data. Real-time detection can prevent financial losses and protect customers from fraudulent activities.|
|**Properties**| Stream processing handles data in real-time, making it highly dynamic. It's designed to support immediate updates and changes, making it ideal for use cases that require up-to-the-second insights.|
|**Formats**| Data in stream processing is often in Protobuf, Avro, or other formats optimized for small footprint and fast serialization.|
|**Databases**| Real-time databases include [Clickhouse](https://clickhouse.com/), [Redis](https://redis.com/), and [RethinkDB](https://rethinkdb.com/). There are also in-memory databases, such as [DuckDB](https://duckdb.org/) and [KuzuDB](https://kuzudb.com/), which can be used to create real-time dashboards. However, depending on the deployment strategy chosen, these databases may lose the data once the application is terminated.|

Most systems deployed in production at scale combine stream and batch processing. This enables you to leverage the immediacy of real-time updates and the depth of historical data. But reconciling stream and batch processes in a single system introduces trade-off decisions – trade-off decisions you must make to keep your data consistent across systems and across time.

## Reconciling Streaming and Batch Sources 

When choosing and configuring the data sources for a vector retrieval system, it’s important to consider tradeoffs between data velocity, complexity, and model size. 

![Velocity-complexity tradeoff](assets/building_blocks/data_sources/bb1-2.png)

### Velocity vs Complexity Tradeoff

Different data sources impact how the data is then ingested and processed by your data & ML infrastructure. For a deep dive on how your choice of data sources affects the downstream systems in your information retrieval stack, see our article on [Vector Compute](https://hub.superlinked.com/vector-compute). For the purposes of this discussion, you can think about data source choice in terms of a tradeoff between data velocity, retrieval quality, and engineering complexity.

- **Streaming data**: This option allows for real-time vector computations but, due to latency concerns, comes with constraints on model complexity. Streaming data may require simpler embeddings or lookups.
- **Batch data**: Unlike streaming data, batch data enables complex models, including large transformers. But with batch data, updates are less frequent, making it suitable for asynchronous needs.
- **Hybrid Approaches**: These merge streaming filters or lookups with batch retraining, offering responsiveness alongside in-depth analysis.

| **Aspect** | **Streaming Data** | **Batch Data** | **Hybrid Approaches**|
|------------------------|------------------------|------------------------|------------------------|
|**Model Complexity**| Constraints on model complexity due to latency | Enables complex models, including large transformers | Balances streaming with batch retraining for analysis |
|**Update Frequency**| Real-time updates | Less frequent updates, suitable for asynchronous use | Combines streaming with batch for responsiveness and depth |
|**Cost-effectiveness**| Higher resource requirements | More cost-effective due to lower resource requirements | A balance of cost-effectiveness with real-time features |
|**Data Consistency**| Real-time updates, potential data consistency | Ensures data consistency | A balance of real-time and consistency |
|**Timeliness & Relevance**| Timely but may have lower content value | Timely and relevant | Timely and relevant |

How you configure your data sources to balance velocity and complexity ultimately manifests in the types of machine learning model architectures you use in your vector search system.

**Example Architectures**

Within a vector search system, there are three basic architecture types. These architectures balance velocity and complexity in different ways:

- **Streaming**: Think of architectures like Recurrent Neural Networks (RNNs), shallow neural networks, and indexing/retrieval models.
- **Batch**: Large pre-trained transformers, custom deep learning models, and architecture search models.
- **Hybrid**: Combine indexing with periodic retraining or deploying a two-stage filtering and analysis setup.

Choosing the right architecture depends on your goals:

**Key Considerations**

- **Synchronicity Needs:** Determine whether real-time updates to the vector embeddings are essential or if daily or weekly updates suffice.
- **Model Accuracy Requirements:** Consider whether approximate real-time performance meets your needs. For example, speed is key in preventing further loss in fraud detection. Therefore, real-time data is important. When detecting fraud, the cost of a false positive is often less than the risk posed by a false negative, so it makes sense to prioritize velocity. In a recommendation system, on the other hand, having true real-time user interaction data may be less important; the customer takes time to evaluate the current options. Therefore, you can get away with near real-time performance.

The trade-off between velocity and complexity is not the only important consideration in building your vector retrieval system. You also need to balance model size with response time.

### Model Size vs Response Time Tradeoff

- **Streaming:** Typically involves models under 100MB in size, such as shallow networks, distillation models, and compact neural networks like MobileNet.
- **Batch:** Here, you're looking at models exceeding 1GB in size, including giant transformers and custom deep networks.
- **Hybrid:** This is the sweet spot, combining smaller streaming models with larger batch models to find an optimal balance between size and speed.

As with velocity and complexity, balancing size with response time is consequential for the types of architectures you use in your vector search system.

**Example Architectures**
- **Streaming**: You might come across RNNs, CNNs with fewer than 10 layers, and efficient models like MiniLM or TinyML. Even simple dense networks or XGBoost can fit the bill.
- **Batch Architectures**: Think of big players like BERT, GPT-3/4, T5, EfficientNet, and custom transformers. Also, don't forget about neural architecture search models.
- **Ensemble and Stacked Models**: In some cases, ensemble and stacked models can provide the best of streaming and batch architectures, combining model sizes and optimizing for various needs.

### Additional Key Considerations for Vector Retrieval System Design

Besides the velocity-complexity and model size-response time tradeoffs above, there are some additional considerations pivotal to ensuring that you meet your vector retrieval system’s objectives:
- **Response Latency Constraints**: Always monitor response latency constraints to ensure your models can meet real-time demands.
- **Model Accuracy Requirements**: Consider your precise model accuracy requirements and whether your infrastructure can effectively handle the chosen model sizes.
- **Data Value**: Recognize the value of processed data, as high-value data holds critical information that can lead to valuable insights and informed decision-making.

In short, keeping these key considerations in mind while you balance velocity with complexity, and model size with response times, will help ensure that you configure your specific vector retrieval system optimally.


### Kappa vs. Lambda

In the context of real-time data processing, the choice between Kappa and Lambda architectures involves evaluating trade-offs between velocity, complexity, and data value:

- **Kappa Architecture**: Emphasizes real-time processing. Data is ingested as a continuous stream and processed in real-time to support immediate decisions. This approach is well-suited to use high-velocity cases, such as fraud detection systems in financial institutions.

- **Lambda Architecture**: Combines batch and real-time processing. Data is ingested both as a stream and in batch mode. This approach offers the advantage of capturing high-velocity data while maintaining the value of historical data. It's suitable when there's a need for comprehensive insights that consider both real-time and historical data.

| Aspect | Kappa Architecture | Lambda Architecture |
|------------------------|--------------------------------------------------------|-------------------------------------------------------------|
| *Real-Time Processing* | Emphasizes real-time data processing. Data is ingested as a continuous stream and processed in real-time to support immediate decisions. | Combines batch and real-time processing. Data is ingested as a stream and batch, allowing for comprehensive analysis. |
| *Simplicity* | Offers a simpler architecture, with a single processing pipeline for real-time data. | Has a more complex architecture, with separate processing pipelines for real-time and batch data. |
| *Scalability* | Highly scalable for handling high-velocity data streams. Scales well with growing data loads. | Scalable but may involve more management due to the dual processing pipelines. |
| *Latency* | Provides low-latency processing, making it suitable for use cases that require immediate decisions, such as fraud detection. | Offers lower latency for real-time processing, but higher latency for batch processing. |
| *Complex Analysis* | May be less suitable for complex batch analysis. Real-time data is the primary focus. | Supports both real-time and batch processing, allowing for more comprehensive and complex analysis. |
| *Data Consistency* | May sacrifice some data consistency to prioritize real-time processing. | Ensures strong data consistency between real-time and batch views, making it suitable for data analytics and historical comparisons. |
| *Use Cases* | Ideal for use cases with high velocity, such as real-time monitoring, fraud detection, and sensor data processing. | Suitable for applications where historical data analysis, complex event processing, and reconciling real-time and batch views are required. |

