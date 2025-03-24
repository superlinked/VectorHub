# Simplifying Complex Research with AI

Keeping up-to-date with the vast number of research papers published regularly can be challenging and time-consuming. An AI assistant capable of efficiently locating relevant research, summarizing key insights, and answering specific questions from these papers could significantly streamline this process.

## Where to start building a research assistant system ?

Traditionally, building such a system involves complexity and considerable resource investment. Search systems typically retrieve an initial broad set of documents based on relevance and subsequently apply a secondary reranking process to refine and reorder results. While reranking enhances accuracy, it significantly increases computational complexity, latency, and overhead due to the extensive data retrieval initially required. Superlinked addresses this complexity by combining structured numeric and categorical embeddings with semantic text embeddings, providing comprehensive multimodal vectors.
This method significantly enhances search accuracy by preserving attribute-specific information within each embedding.

## Build an agentic system with Superlinked

This article shows how to build an agent system using a Kernel agent to handle queries. If you want to follow along, here is the [colab](links.superlinked.com/research_ai_agent_nb)

This AI agent can do three main things:

1. Find Papers: Search for research papers by topic (e.g. “quantum computing”) and then rank them by relevance and recency.
2. Summarize papers: Condense the retrieved papers into bite-sized insights.
3. Answer questions: Extract answers directly from the specific research papers based on targeted user queries.

Superlinked eliminates the need for re-ranking methods as it improves the vector search relevance. Superlinked's RecencySpace will be used which specifically encodes temporal metadata, prioritizing recent documents during retrieval, and eliminating the need for computationally expensive reranking. For example, if two papers have the same relevance - the one that is most recent will rank higher.

### Step 1 : Setting up the toolbox

```python
%%capture
!pip3 install openai pandas sentence-transformers transformers superlinked==19.21.1
```

To make things easier and more modular, I created an Abstract Tool class. This will simplify the process of building and adding tools

```python
import pandas as pd
import superlinked.framework as sl
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from tqdm import tqdm
from google.colab import userdata

# Abstract Tool Class
class Tool(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def use(self, *args, **kwargs) -> Any:
        pass


# Get API key from Google Colab secrets
try:
    api_key = userdata.get('OPENAI_API_KEY')
except KeyError:
    raise ValueError("OPENAI_API_KEY not found in user secrets. Please add it using Tools > User secrets.")

# Initialize OpenAI Client
api_key = os.environ.get("OPENAI_API_KEY", "your-openai-key")  # Replace with your OpenAI API key
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)
model = "gpt-4"
```

### Step 2 : Understanding the Dataset

This example uses a dataset containing approximately 10,000 AI research papers available on [Kaggle](https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset). To make it easy, simply run the cell below, and it will automatically download the dataset to your working directory. You may also use your own data sources, such as research papers or other academic content. If you decide to do so, all you need to do is adjust the schema design slightly and update the column names.

```python
import pandas as pd

!wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1FCR3TW5yLjGhEmm-Uclw0_5PWVEaLk1j' -O arxiv_ai_data.csv
```

For now, to make things run a bit quicker, we will use a smaller subset of the papers just to speed things up but feel free to try the example using the full dataset. An important technical detail here is that the timestamps from the dataset will be converted from string timestamps (like '1993-08-01 00:00:00+00:00') into pandas datetime objects.  This conversion is necessary because it allows us to perform date/time operations.

```python
df = pd.read_csv('arxiv_ai_data.csv').head(100)

# Convert to datetime but keep it as datetime (more readable and usable)
df['published'] = pd.to_datetime(df['published'])

# Ensure summary is a string
df['summary'] = df['summary'].astype(str)

# Add 'text' column for similarity search
df['text'] = df['title'] + " " + df['summary']
```

```python
Debug: Columns in original DataFrame: ['authors', 'categories', 'comment', 'doi', 'entry_id', 'journal_ref' 'pdf_url', 'primary_category', 'published', 'summary', 'title', 'updated']
```

### Understanding the Dataset Columns

Below is a brief overview of the key columns in our dataset, which will be important in the upcoming steps:

1. `published`: The publication date of the research paper.
2. `summary`: The abstract of the paper, providing a concise overview.
3. `entry_id`: The unique identifier for each paper from arXiv.

For this demonstration, we specifically focus on four columns: `entry_id`, `published`, `title`, and `summary`. To optimize retrieval quality, the title and summary are combined into a single, comprehensive text column, which forms the core of our embedding and search process.

A Note on Superlinked’s In-Memory Indexer : Superlinked’s in-memory indexing stores our dataset directly in RAM, making retrieval exceptionally fast which is ideal for real-time searches and rapid prototyping. For this proof-of-concept with 1,000 research papers, leveraging an in-memory approach significantly enhances query performance, eliminating delays associated with disk access.

### Step 3 : Defining the Superlinked Schema

To move ahead, there is a need for schema to map our data. We have set up `PaperSchema` with key fields:

```python
class PaperSchema(sl.Schema):
    text: sl.String
    published: sl.Timestamp  # This will handle datetime objects properly
    entry_id: sl.IdField
    title: sl.String
    summary: sl.String

paper = PaperSchema()
```

### Defining Superlinked Spaces for Effective Retrieval

An essential step in organizing and effectively querying our dataset involves defining two specialized vector spaces: TextSimilaritySpace and RecencySpace.

1. TextSimilaritySpace

The `TextSimilaritySpace` is designed to encode textual information—such as the titles and abstracts of research papers into vectors. By converting text into embeddings, this space significantly enhances the ease and accuracy of semantic searches. It is optimized specifically to handle longer text sequences efficiently, enabling precise similarity comparisons across documents.

```python
text_space = sl.TextSimilaritySpace(
    text=sl.chunk(paper.text, chunk_size=200, chunk_overlap=50),
    model="sentence-transformers/all-mpnet-base-v2"
)
```

2. RecencySpace

The `RecencySpace` captures temporal metadata, emphasizing the recency of research publications. By encoding timestamps, this space assigns greater significance to newer documents. As a result, retrieval results naturally balance content relevance with publication dates, favoring recent insights.

```python
recency_space = sl.RecencySpace(
    timestamp=paper.published,
    period_time_list=[
        sl.PeriodTime(timedelta(days=365)),      # papers within 1 year
        sl.PeriodTime(timedelta(days=2*365)),    # papers within 2 years
        sl.PeriodTime(timedelta(days=3*365)),    # papers within 3 years
    ],
    negative_filter=-0.25
)
```

Think of RecencySpace as a time-based filter, similar to sorting your emails by date or viewing Instagram posts with the newest ones first. It helps answer the question, 'How fresh is this paper?'

- Smaller timedeltas (like 365 days) allow for more granular, yearly time-based rankings.
- Larger timedeltas (like 1095 days) create broader time periods.

The `negative_filter` penalizes very old papers. To explain it more clearly, consider the following example where two papers have identical content relevance, but their rankings will depend on their publication dates.

```markdown
Paper A: Published in 1996 
Paper B: Published in 1993

Scoring example:
- Text similarity score: Both papers get 0.8
- Recency score:
  - Paper A: Receives the full recency boost (1.0)
  - Paper B: Gets penalized (-0.25 due to negative_filter)

Final combined scores:
- Paper A: Higher final rank
- Paper B: Lower final rank
```

These spaces are key to making the dataset more accessible and effective. They allow for both content-based and time-based searches, and really helpful in understanding the relevance and recency of research papers. This provides a powerful way to organize and search through the dataset based on both the content and the publication time.

### Step 4 : Building the index

Next, the spaces are fused into an index which is the search engine's core:

```python
paper_index = sl.Index([text_space, recency_space])
```

Then the DataFrame is mapped to the schema and loaded in batches (10 papers at a time) into an in-memory store:

```python
# Parser to map DataFrame columns to schema fields
parser = sl.DataFrameParser(
    paper,
    mapping={
        paper.entry_id: "entry_id",
        paper.published: "published",
        paper.text: "text",
        paper.title: "title",
        paper.summary: "summary",
    }
)

# Set up in-memory source and executor
source = sl.InMemorySource(paper, parser=parser)
executor = sl.InMemoryExecutor(sources=[source], indices=[paper_index])
app = executor.run()

# Load the DataFrame with a progress bar using batches
batch_size = 10
data_batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
for batch in tqdm(data_batches, total=len(data_batches), desc="Loading Data into Source"):
    source.put([batch])
```

The in-memory executor is why Superlinked shines here—1,000 papers fit snugly in RAM, and queries fly without disk I/O bottlenecks.

### Step 5 : Crafting the query

Next is the query creation. This is where the template for crafting queries is created. To manage this, we need a query template that can balance both relevance and recency. Here’s what that would look like:

```python
# Define the query
knowledgebase_query = (
    sl.Query(
        paper_index,
        weights={
            text_space: sl.Param("relevance_weight"),
            recency_space: sl.Param("recency_weight"),
        }
    )
    .find(paper)
    .similar(text_space, sl.Param("search_query"))
    .select(paper.entry_id, paper.published, paper.text, paper.title, paper.summary)
    .limit(sl.Param("limit"))
)
```

This allows us to pick whether to prioritize the content (relevance_weight) or the recency (recency_weight) - a very useful combo for our agent's needs.

### Step 6 : Building tools

Now comes the tooling part.

We will be creating three tools ...

1. `Retrieval Tool` : This tool is crafted by plugging into Superlinked’s index, letting it pull the top 5 papers based on a query. It balances relevance (1.0 weight) and recency (0.5 weight) to accomplish the “find papers” goal. What we want is to find the papers which are relevant to the query. So, if the query is: “What quantum computing papers were published between 1993 and 1994?”, then the retrieval tool will retrieve those papers, summarize them one by one, and return the results.

```python
class RetrievalTool(Tool):
    def __init__(self, df, app, knowledgebase_query, client, model):
        self.df = df
        self.app = app
        self.knowledgebase_query = knowledgebase_query
        self.client = client
        self.model = model

    def name(self) -> str:
        return "RetrievalTool"

    def description(self) -> str:
        return "Retrieves a list of relevant papers based on a query using Superlinked."

    def use(self, query: str) -> pd.DataFrame:
        result = self.app.query(
            self.knowledgebase_query,
            relevance_weight=1.0,
            recency_weight=0.5,
            search_query=query,
            limit=5
        )
        df_result = sl.PandasConverter.to_pandas(result)
        # Ensure summary is a string
        if 'summary' in df_result.columns:
            df_result['summary'] = df_result['summary'].astype(str)
        else:
            print("Warning: 'summary' column not found in retrieved DataFrame.")
        return df_result
```

Next up is the `Summarization Tool`. This tool is designed for cases where a concise summary of a paper is needed. In order to use it, it will be provided with `paper_id`, which is the ID of the paper that needs to be summarized. If a `paper_id` is not provided, the tool will not work as these IDs are a requirement in order to find the corresponding papers in the dataset.

```python
class SummarizationTool(Tool):
    def __init__(self, df, client, model):
        self.df = df
        self.client = client
        self.model = model

    def name(self) -> str:
        return "SummarizationTool"

    def description(self) -> str:
        return "Generates a concise summary of specified papers using an LLM."

    def use(self, query: str, paper_ids: list) -> str:
        papers = self.df[self.df['entry_id'].isin(paper_ids)]
        if papers.empty:
            return "No papers found with the given IDs."
        summaries = papers['summary'].tolist()
        summary_str = "\n\n".join(summaries)
        prompt = f"""
        Summarize the following paper summaries:\n\n{summary_str}\n\nProvide a concise summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
```

Finally, we have the `QuestionAnsweringTool`. This tool chains the `RetrievalTool` to fetch the relevant papers and then uses them to answer the questions. If no relevant papers are found to answer the questions, it will provide an answer based on general knowledge

```python
class QuestionAnsweringTool(Tool):
    def __init__(self, retrieval_tool, client, model):
        self.retrieval_tool = retrieval_tool
        self.client = client
        self.model = model

    def name(self) -> str:
        return "QuestionAnsweringTool"

    def description(self) -> str:
        return "Answers questions about research topics using retrieved paper summaries or general knowledge if no specific context is available."

    def use(self, query: str) -> str:
        df_result = self.retrieval_tool.use(query)
        if 'summary' not in df_result.columns:
            # Tag as a general question if summary is missing
            prompt = f"""
            You are a knowledgeable research assistant. This is a general question tagged as [GENERAL]. Answer based on your broad knowledge, not limited to specific paper summaries. If you don't know the answer, provide a brief explanation of why.

            User's question: {query}
            """
        else:
            # Use paper summaries for specific context
            contexts = df_result['summary'].tolist()
            context_str = "\n\n".join(contexts)
            prompt = f"""
            You are a research assistant. Use the following paper summaries to answer the user's question. If you don't know the answer based on the summaries, say 'I don't know.'

            Paper summaries:
            {context_str}

            User's question: {query}
            """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
```

### Step 7 :  Building the Kernel Agent

Next is the Kernel Agent. It functions as the central controller, ensuring smooth and efficient operation. Acting as the core component of the system, the Kernel Agent coordinates communication by routing queries according to their intent when multiple agents operate concurrently. In single-agent systems, such as this one, the Kernel Agent directly uses the relevant tools to manage tasks effectively.

```python
class KernelAgent:
    def __init__(self, retrieval_tool: RetrievalTool, summarization_tool: SummarizationTool, question_answering_tool: QuestionAnsweringTool, client, model):
        self.retrieval_tool = retrieval_tool
        self.summarization_tool = summarization_tool
        self.question_answering_tool = question_answering_tool
        self.client = client
        self.model = model

    def classify_query(self, query: str) -> str:
        prompt = f"""
        Classify the following user prompt into one of the three categories:
        - retrieval: The user wants to find a list of papers based on some criteria (e.g., 'Find papers on AI ethics from 2020').
        - summarization: The user wants to summarize a list of papers (e.g., 'Summarize papers with entry_id 123, 456, 789').
        - question_answering: The user wants to ask a question about research topics and get an answer (e.g., 'What is the latest development in AI ethics?').

        User prompt: {query}

        Respond with only the category name (retrieval, summarization, question_answering).
        If unsure, respond with 'unknown'.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=10
        )
        classification = response.choices[0].message.content.strip().lower()
        print(f"Query type: {classification}")
        return classification

    def process_query(self, query: str, params: Optional[Dict] = None) -> str:
        query_type = self.classify_query(query)
        if query_type == 'retrieval':
            df_result = self.retrieval_tool.use(query)
            response = "Here are the top papers:\n"
            for i, row in df_result.iterrows():
                # Ensure summary is a string and handle empty cases
                summary = str(row['summary']) if pd.notna(row['summary']) else ""
                response += f"{i+1}. {row['title']} \nSummary: {summary[:200]}...\n\n"
            return response
        elif query_type == 'summarization':
            if not params or 'paper_ids' not in params:
                return "Error: Summarization query requires a 'paper_ids' parameter with a list of entry_ids."
            return self.summarization_tool.use(query, params['paper_ids'])
        elif query_type == 'question_answering':
            return self.question_answering_tool.use(query)
        else:
            return "Error: Unable to classify query as 'retrieval', 'summarization', or 'question_answering'."
```

At this stage, all components of the Research Agent System have been configured. The system can now be initialized by providing the Kernel Agent with the appropriate tools, after which the Research Agent System will be fully operational.

```python
retrieval_tool = RetrievalTool(df, app, knowledgebase_query, client, model)
summarization_tool = SummarizationTool(df, client, model)
question_answering_tool = QuestionAnsweringTool(retrieval_tool, client, model)

# Initialize KernelAgent
kernel_agent = KernelAgent(retrieval_tool, summarization_tool, question_answering_tool, client, model)
```

Now let's test the system..

```python
# Test query
print(kernel_agent.process_query("Find papers on quantum computing in last 10 years"))
```

Running this activates the `RetrievalTool`. It will fetch the relevant papers based on both relevance and recency, and return the relevant columns. If the returned result includes the summary column (indicating the papers were retrieved from the dataset), it will use those summaries and return them to us.

```python
Query type: retrieval
Here are the top papers:
1. Quantum Computing and Phase Transitions in Combinatorial Search 
Summary: We introduce an algorithm for combinatorial search on quantum computers that
is capable of significantly concentrating amplitude into solutions for some NP
search problems, on average. This is done by...

1. The Road to Quantum Artificial Intelligence 
Summary: This paper overviews the basic principles and recent advances in the emerging
field of Quantum Computation (QC), highlighting its potential application to
Artificial Intelligence (AI). The paper provi...

1. Solving Highly Constrained Search Problems with Quantum Computers 
Summary: A previously developed quantum search algorithm for solving 1-SAT problems in
a single step is generalized to apply to a range of highly constrained k-SAT
problems. We identify a bound on the number o...

1. The model of quantum evolution 
Summary: This paper has been withdrawn by the author due to extremely unscientific
errors....

1. Artificial and Biological Intelligence 
Summary: This article considers evidence from physical and biological sciences to show
machines are deficient compared to biological systems at incorporating
intelligence. Machines fall short on two counts: fi...
```

Let's try one more query, this time, let's do a summarization one..

```python
print(kernel_agent.process_query("Summarize this paper", params={"paper_ids": ["http://arxiv.org/abs/cs/9311101v1"]}))
```

```python
Query type: summarization
This paper discusses the challenges of learning logic programs that contain the cut predicate (!). Traditional learning methods cannot handle clauses with cut because it has a procedural meaning. The proposed approach is to first generate a candidate base program that covers positive examples, and then make it consistent by inserting cut where needed. Learning programs with cut is difficult due to the need for intensional evaluation, and current induction techniques may need to be limited to purely declarative logic languages.
```

I hope this example has been helpful for developing AI agents and agent-based systems. Much of the retrieval functionality demonstrated here was made possible by Superlinked, so please consider starring the [repository](links.superlinked.com/research_ai_agent_repo) for future reference when accurate retrieval capabilities are needed for your AI agents!


## Contributors

- [Vipul Maheshwari, author](https://www.linkedin.com/in/vipulmaheshwarii/)
- [Filip Makraduli, reviewer](https://www.linkedin.com/in/filipmakraduli/)