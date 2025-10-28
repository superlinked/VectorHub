# Semantic Search for Cyber Threat Intelligence
_Extracting hacker forum insights with Superlinked_


## Takeaways

- **Keyword search isn't enough**: attackers use slang, misspellings, and evolving terms that break exact-match search.
- **Superlinked spaces** make it easy to combine text similarity and recency into one search pipeline.
- **Query-time weighting** lets analysts tune focus (fresh threats vs deeper background chatter).
- Adding an **LLM summarization step** transforms raw forum noise into analyst-ready daily briefings.

## Why this matters

Cyber Threat Intelligence (CTI) teams deal with overwhelming amounts of raw text: hacker forum chatter, underground marketplace discussions, and security news articles.  
Traditional keyword searches catch only exact matches - if an analyst searches _"finance frauds"_ but the forum thread says _"cc dumps"_, the match is lost.

Semantic search solves this by understanding **meaning** rather than just keywords.  
In this demo, I built a semantic search pipeline for **threat intelligence articles** using **Superlinked**, and tested it with ~1,700 documents from hacker forums and security news sources

## Dataset

I exported ~1,700 articles from a CTI collection system. Each record contains:

- **Title** (short description of the post)
- **Body** (longer content text)
- **Date** (ISO timestamp)
- **ID** (unique identifier)

Examples:

<img width="1784" height="300" alt="image" src="https://github.com/user-attachments/assets/425a61d4-fc9f-40c3-ad6f-47d301428692" />

This type of noisy, varied dataset is typical in CTI - exactly where semantic search adds value.

## Schema and Spaces

### Schema
```python
class ThreatIntelSchema(sl.Schema):
    title: sl.String
    body: sl.String
    date: sl.Timestamp
    id: sl.IdField
```

### Semantic Spaces

I defined three spaces:

- **Body space**: semantic similarity on article text.
- **Title space**: semantic similarity on titles.
- **Recency space**: weights newer posts higher, but still allows historical context.

```python
body_space = sl.TextSimilaritySpace(
    text=threat_intel_schema.body,
    model="sentence-transformers/all-MiniLM-L6-v2"
)
title_space = sl.TextSimilaritySpace(
    text=threat_intel_schema.title,
    model="sentence-transformers/all-MiniLM-L6-v2"
)
recency_space = sl.RecencySpace(
    timestamp=threat_intel_schema.date,
    period_time_list=[
        sl.PeriodTime(timedelta(days=4*365), weight=1),
        sl.PeriodTime(timedelta(days=11*365), weight=2),
    ],
    negative_filter=0.0,
)
```

## Index and Queries

### Index
```python
index = sl.Index(spaces=[body_space, title_space, recency_space])
executor = sl.InMemoryExecutor(
    sources=[sl.InMemorySource(threat_intel_schema, parser=dataframe_parser)],
    indices=[index],
    context_data=EXECUTOR_DATA
)
app = executor.run()
```

### Simple Query
```python
simple_query = (
    sl.Query(
        index,
        weights={
            body_space: sl.Param("body_weight"),
            title_space: sl.Param("title_weight"),
            recency_space: sl.Param("recency_weight"),
        },
    )
    .find(threat_intel_schema)
    .similar(body_space, sl.Param("query_text"))
    .similar(title_space, sl.Param("query_text"))
    .select([threat_intel_schema.body, threat_intel_schema.title, threat_intel_schema.date])
    .limit(sl.Param("limit"))
)
```

## Example Searches

Query: _"finance fraud"_
- Surfaces discussions on **phishing scams**, **banking malware**, and **underground dumps markets**.
- Recency weighting prioritizes newer fraud campaigns over older discussions.

<img width="2335" height="534" alt="image" src="https://github.com/user-attachments/assets/ec2af043-5dcc-4846-a977-94b9ed80952e" />

## Why this matters for CTI

- **Better recall**: semantic search retrieves threats missed by keywords.
- **Freshness**: recency weighting ensures analysts see what matters now.
- **Summaries**: LLM turns noisy forum chatter into clear, daily briefings.

For CTI teams, this means saving time and catching emerging threats earlier.

## Conclusion

This project shows how **Superlinked enables semantic search in practical cybersecurity scenarios**. 
With just a small dataset of ~1,700 articles, I was able to build a pipeline that retrieves meaningful signals and generates analyst-ready summaries.
For CTI teams, this can mean the difference between **reactive defense** and **early detection of emerging threats**.
Superlinked makes it possible to bring semantic search and retrieval-augmented intelligence directly into CTI workflows.

