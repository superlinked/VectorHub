# Building Blocks

Building blocks are the atomic units of creating a vector retrieval stack. If you want to create a vector retrieval
stack that's ready for production, you'll need to have a few key components in place. These include:

- Data sources: You can get your data from a variety of sources, including relational databases like PSQL and MySQL,
  data pipeline tools like Kafka and GCP pub-sub, data warehouses like Snowflake and Databricks, and customer data
  platforms like Segment. The goal here is to extract and connect your data so that it can be used in your vector stack.
- Vector computation: This involves turning your data into vectors using models from Huggingface or your own custom
  models. You'll also need to know where to run these models and how to bring all of your computing infrastructure
  together using tools like custom spark pipelines or products like Superlinked. The ultimate goal is to have
  production-ready pipelines and models that are ready to go.
- Vector search & management: This is all about querying and retrieving vectors from Vector DBs like Weaviate and
  Pinecone, or hybrid DBs like Redis and Postgres (with pgvector). You'll also need to use search tools like Elastic and
  Vespa to rank your vectors. The goal is to make the vectors indexable and search for relevant vectors when needed.

## Contents

- [Data Sources](https://hub.superlinked.com/data-sources)
- [Vector Compute](https://hub.superlinked.com/vector-compute)
- [Vector Search & Management](https://hub.superlinked.com/vector-search)
