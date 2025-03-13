I’ve always wanted an AI assistant that could make it easier to go through a collection of research papers, answer my questions, summarize the content, and give me the important points. In short, I wanted a research helper that would make the whole process simpler. That’s when I decided to create it

### What’s the Plan?

I wanted to build an agentic system that could do three main things:

1. Find Papers: Search for research papers by topic (e.g., “quantum computing”) and rank them by relevance and recency.
2. Summarize: Crunch those papers into bite-sized insights fast when I asked it to do so.
3. Answer Questions: Pull answers from the specific papers which I tell it to do and based on that it answer those questions

In all of it, Superlinked is the backbone here, As it improves the vector search relevance by encoding metadata together with your unstructured data into vectors. This powers a vector database that understands both the text's meaning and the time factor. I mean when we will stitch the schema design later on, we will be using `RecencySpace` which will literally help us to add the relevancy of the papers during the retrieval (ps: without using any rerankers) that means during the retrieval, if the two papers have the same kind of relevancy, it will prioritize paper which is more recent. More on this later.

For the LLM reasoning, I plan to use OpenAI’s models. However, you can easily replace it with any LLM of your choice. And last but not least,  I made a Kernel Agent to handle the queries. If you want to follow along, here is the [colab](https://colab.research.google.com/drive/1DZ13m8lTPsGFVW0KuHnjP5Z7z5xmFOaa?usp=sharing)

Too much talking, let's go now..

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

# Initialize OpenAI Client
api_key = os.environ.get("OPENAI_API_KEY", "your-api-key")  # Replace with your OpenAI API key
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)
model = "gpt-3.5-turbo"
```

### Step 2 : Data needs

What does the data look like? For this experiment, I used a dataset consisting of 10,000 AI research papers from [Kaggle](https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset). To make it easy, simply run the cell below, and it will automatically download the dataset to your working directory. You can also replace this with your own data sources, whether they are research papers or other academic content. If you decide to do so, all you need to do is adjust the schema design slightly and update the column names.

```python
import pandas as pd

!wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1FCR3TW5yLjGhEmm-Uclw0_5PWVEaLk1j' -O arxiv_ai_data.csv
```

For now, I’m going to work with just 1,000 papers—mainly to speed things up for the demo. SHHHHHH :)

There’s one more thing: I’ll convert the published data from string timestamps (like '1993-08-01 00:00:00+00:00') into pandas datetime objects. This conversion is necessary because it allows us to perform date/time operations.

```python
df = pd.read_csv('arxiv_ai_data.csv').head(1000)

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

Let me briefly explain the columns and what they contain, as this will be helpful later on:

1. `published`: The date the paper was published.
2. `summary`: The abstract of the paper.
3. `entry_id`: The arXiv entry ID for the paper.

For our experiment, the only columns we need are `entry_id`, `published`, `title`, and `summary`. To make things more effective, we can combine the title and the summary of each paper to create a comprehensive text, which will be the core of our work. So `text` column will contains the `title` and the `summary`

Just a bit on the Superlinked's in-memory indexer. So basically I think it's a game-changer for our small dataset of 1,000 papers. I mean it stores data in the computer’s memory, which means queries are lightning-fast—no waiting for disk access. This is perfect for real-time applications and prototyping, like our proof-of-concept.

### Step 3 : Defining the Superlinked Schema

To move ahead, we will need a schema to map our data. I set up `PaperSchema` with key fields:

```python
class PaperSchema(sl.Schema):
    text: sl.String
    published: sl.Timestamp  # This will handle datetime objects properly
    entry_id: sl.IdField
    title: sl.String
    summary: sl.String

paper = PaperSchema()
```

Now, the most important part of the process is creating the spaces. There are two spaces we need to create for our schema. The first one is `TextSimilaritySpace`. This space plays a crucial role in organizing and searching through the research papers. Essentially, it converts the text data we provide into vectors, making it much easier to search through them. The TextSimilaritySpace is specifically designed to efficiently encode longer text sequences using specialized models, enabling effective similarity comparisons between different pieces of text

```python
text_space = sl.TextSimilaritySpace(
    text=sl.chunk(paper.text, chunk_size=200, chunk_overlap=50),
    model="sentence-transformers/all-mpnet-base-v2"
)
```

On the other hand, `RecencySpace` focuses on the importance of time in the context of research papers. It encodes timestamps, prioritizing the recency of the papers. This ensures that more recent papers are given greater weight in the similarity calculations, allowing for a balance between the relevance of the content and the publication time of the papers.

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

Next, I fused the spaces into an index, A.K.A the search engine’s core:

```python
paper_index = sl.Index([text_space, recency_space])
```

Then, I mapped the DataFrame to the schema and loaded it in batches (10 papers at a time) into in-memory store:

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

Next, we move on to query creation. This is where we set up a template for crafting queries. To manage this, we need a query template that can balance both relevance and recency. Here’s what it looks like:

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

This lets me puts on how much I care about content (relevance_weight) versus freshness (recency_weight) — a killer combo for our agentic needs.

### Step 6 : Building tools

Now comes the tooling part, here we go lads.

Ok , So We’ll be creating three tools for our agent to choose from. These tools will help find papers, summarize them if needed, and answer relevant questions.

1. `Retrieval Tool` : This tool is crafted by hooking into Superlinked’s index, letting it pull the top 5 papers based on a query. It balances relevance (1.0 weight) and recency (0.5 weight) to nail the “find papers” goal.  I mean what we wanted is to find the papers which are relevant to the query. So if the query is something like, "What are the Quantum computers papers are published in b/w 1993 and 1994" , then the retrieval tool will retrieve those papers, will bring out the summary of those papers one by one, and will return the result.

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

Next up is the `Summarization Tool`. This tool is designed for cases where we need a concise summary of a paper. To use it, we’ll provide the `paper_id`, which is the ID of the paper we want summarized. Keep in mind that if you don’t provide the `paper_id`, the tool won’t work, as it requires these IDs to look up the corresponding papers in the dataset.

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

Next, we have the Kernel Agent. Whenever I build an agent-based system, I always include the Kernel Agent—it acts like the 'boss.' It really makes everything run smoothly. The Kernel Agent is the centerpiece of the system. If you have multiple agents working in parallel, it acts as a router, directing queries based on the intent. In a single-agent system like this one, you simply use the Kernel Agent with the relevant tools.

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

By this point, everything is set up for our Research Agent System. All you need to do is initialize the Kernel Agent with the tools, and voilà—our Research Agent system is ready to roll.

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

When you run this, the `RetrievalTool` will be activated. It will fetch the relevant papers based on both relevance and recency, and return the relevant columns. If the returned result includes the summary column (indicating the papers were retrieved from the dataset), it will use those summaries and return them to us.

```python
Query type: retrieval
Here are the top papers:
1. Quantum Computing and Phase Transitions in Combinatorial Search 
Summary: We introduce an algorithm for combinatorial search on quantum computers that
is capable of significantly concentrating amplitude into solutions for some NP
search problems, on average. This is done by...

2. The Road to Quantum Artificial Intelligence 
Summary: This paper overviews the basic principles and recent advances in the emerging
field of Quantum Computation (QC), highlighting its potential application to
Artificial Intelligence (AI). The paper provi...

3. Solving Highly Constrained Search Problems with Quantum Computers 
Summary: A previously developed quantum search algorithm for solving 1-SAT problems in
a single step is generalized to apply to a range of highly constrained k-SAT
problems. We identify a bound on the number o...

4. The model of quantum evolution 
Summary: This paper has been withdrawn by the author due to extremely unscientific
errors....

5. Artificial and Biological Intelligence 
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

I mean it's so cool.

Adios Amigos, until next time..