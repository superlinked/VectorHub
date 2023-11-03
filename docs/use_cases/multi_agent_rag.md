<!-- SEO: Retrieval-augmented generation (RAG) has shown great promise for powering conversational AI. However, existing implementations of RAG struggle with relevance, latency, and coherence. A multi-agent architecture that divides responsibilities across specialized retrieval, ranking, reading, and orchestration agents can overcome these limitations and unlock further gains.
 -->

# Retrieval Augmented Generation

<!-- TODO: Cover image -->

## Enhancing RAG with a Multi-Agent System

Retrieval-augmented generation (RAG) has shown great promise for powering conversational AI. However, in most RAG systems today, a single model handles the full workflow of query analysis, passage retrieval, contextual ranking, summarization, and prompt augmentation. This results in suboptimal relevance, latency, and coherence. A multi-agent architecture that factors responsibilities across specialized retrieval, ranking, reading, and orchestration agents, operating asynchronously, allows each agent to focus on its specialized capability using custom models and data. Multi-agent RAG is thus able to improve relevance, latency, and coherence overall. 

While multi-agent RAG is not a panacea – for simpler conversational tasks a single RAG agent may suffice – multi-agent RAG outperforms single agent RAG when your use case requires reasoning over diverse information sources. This article explores a multi-agent RAG architecture and quantifies its benefits.

## RAG Challenges and Opportunities

Retrieval augmented generation faces several key challenges that limit its performance in real-world applications.

First, existing retrieval mechanisms struggle to identify the most relevant passages from corpora containing millions of documents. Simple similarity functions often return superfluous or tangential results. When retrieval fails to return the most relevant information, it leads to suboptimal prompting.

Second, retrieving supplementary information introduces latency; if the database is large, this latency can be prohibitive. Searching terabytes of text with complex ranking creates wait times that are too long for consumer applications.

In addition, current RAG systems fail to appropriately weight the original prompt and retrieved passages. Without dynamic contextual weighting, the model can become over-reliant on the retrievals (resulting in reduced control or adaptablity in generating meaningful responses).

## Multi-agent RAGs address real-world challenges

Specialized agents with divided responsibilities can help address the challenges that plague single-agent architectures, and unlock RAG's full potential. 
By factoring RAG into separable subtasks executed concurrently by collaborative and specialized retriever, ranker, reader, and orchestrator agents, multi-agent RAG can mitigate single-agent RAG's relevance, scalability, and latency limitations. This allows RAG to scale efficiently to enterprise workloads.

Let's break multi-agent RAG into its parts:

First, a query understanding / parsing agent comprehends the query, breaking it down and describing in different sub queries.

Then x number of retriever agents focus solely on efficient passage retrieval from the document corpus, based on the sub queries. These retriever agents employ vector similarity search or knowledge graph retrieval based searches to quickly find potentially relevant passages, minimizing latency.

The ranker agent evaluates the relevance of the retrieved passages using additional ranking signals like source credibility, passage specificity, and lexical overlap. This provides a relevance-based filtering step. This agent might be using ontology for example as a way to rerank retrieved information. 

The reader agent summarizes lengthy retrieved passages to succinct snippets containing only the most salient information. This distills the context down to key facts.

Finally, the orchestrator agent dynamically adjusts the weighting and integration of the prompt and filtered ranked context passages to optimize the final augmented prompt.

By dividing the workload across specialized agents, factored RAG is achieved allowing gains in relevance, reduced latency, better summarization, and optimized prompting.

The modular architecture also provides flexibility to add more agents, like a visualizer agent to inspect system behavior. And to substitute alternate implementations of any agent’s capability.

[combine two following passages, and check against content above:]
For example, retriever agents can focus solely on efficient passage retrieval using optimized vector similarity searches. Dedicated reader agents can analyze retrieved context and summarize the most salient information. Orchestrator agents can dynamically adjust relevance weightings between the prompt and context.
For example, dedicated retriever agents can efficiently search large corpora using optimized vector indexes, reader agents can distill retrieved passages down to salient facts, and an orchestrator agent can dynamically adjust prompt hybridization to maximize coherence.


## Benefits of multi-agent RAG architecture:
- Focused specialization _improves relevance and quality_. Retriever agents leverage tailored similarity metrics, rankers weigh signals like source credibility, and readers summarize context.
- Asynchronous operation _reduces latency_ by parallelizing retrieval. Slow operations don't block faster ones.
- Adding more retriever agents _allows easy horizontal scaling_, and optional _incorporation of new data sources_.
- Modular components allow iterative enhancement over time.

Experiments demonstrate that multi-agent RAG significantly improves appropriateness, coherence, and correctness compared to single-agent RAG. The future potential of multi-agent architectures for conversational systems is promising.




## Example with Autogen library : https://github.com/microsoft/autogen

1. AssistantAgent : They are given a name, a system message, and a configuration object (llm_config). The system message is a string that describes the role of the agent. The llm_config object is a dictionary that contains functions for the agent to perform its role.

2. user_proxy is an instance of UserProxyAgent. It is given a name and several configuration options. The is_termination_msg option is a function that determines when the user wants to terminate the conversation. The human_input_mode option is set to "NEVER", which means the agent will never ask for input from a human. The max_consecutive_auto_reply option is set to 10, which means the agent will automatically reply to up to 10 consecutive messages without input from a human. The code_execution_config option is a dictionary that contains configuration options for executing code.

```python
def mock_understand_query(query):
    # Mock function to understand the query and break it down into subqueries
    pass

def mock_rank_passages(passages):
    # Mock function to rank the retrieved passages based on relevance
    pass

def mock_summarize_passages(passages):
    # Mock function to summarize the retrieved passages
    pass

def mock_adjust_weighting(prompt, context_passages):
    # Mock function to adjust the weighting and integration of the prompt and filtered ranked context passages
    pass

import asyncio

async def mock_retrieve_passages_vector_search(subqueries):
    # Mock function to retrieve relevant passages based on the subqueries using vector search
    pass

async def mock_retrieve_passages_kg(subqueries):
    # Mock function to retrieve relevant passages based on the subqueries using knowledge graph
    pass

async def mock_retrieve_passages_sql(subqueries):
    # Mock function to retrieve relevant passages based on the subqueries using SQL
    pass

async def mock_retrieve_passages(subqueries):
    # Create tasks for each retrieval method
    tasks = [
        mock_retrieve_passages_vector_search(subqueries),
        mock_retrieve_passages_kg(subqueries),
        mock_retrieve_passages_sql(subqueries),
    ]

    # Run the tasks concurrently and wait for all of them to complete
    await asyncio.gather(*tasks)

llm_config = {
    "understand_query": mock_understand_query,
    "retrieve_passages": mock_retrieve_passages,
    "rank_passages": mock_rank_passages,
    "summarize_passages": mock_summarize_passages,
    "adjust_weighting": mock_adjust_weighting,
}

# QueryUnderstandingAgent
query_understanding_agent = autogen.AssistantAgent(
    name="query_understanding_agent",
    system_message="You must use X function. You are only here to understand queries. You intervene First.",
    llm_config=llm_config
)

retriever_agent_vector = autogen.AssistantAgent(
    name="retriever_agent_vector",
    system_message="You must use Y function. You are only here to retrieve passages using vector search. You intervene at the same time as other Retriever agents.",
    llm_config=llm_config_vector
)

retriever_agent_kg = autogen.AssistantAgent(
    name="retriever_agent_kg",
    system_message="You must use Z function. You are only here to retrieve passages using knowledge graph. You intervene at the same time as other Retriever agents.",
    llm_config=llm_config_kg
)

retriever_agent_sql = autogen.AssistantAgent(
    name="retriever_agent_sql",
    system_message="You must use A function. You are only here to retrieve passages using SQL. You intervene at the same time as other Retriever agents.",
    llm_config=llm_config_sql
)

# RankerAgent
ranker_agent = autogen.AssistantAgent(
    name="ranker_agent",
    system_message="You must use B function. You are only here to rank passages. You intervene in third position. ",
    llm_config=llm_config
)

# ReaderAgent
reader_agent = autogen.AssistantAgent(
    name="reader_agent",
    system_message="You must use C function. You are only here to summarize passages. You intervene in fourth position. ",
    llm_config=llm_config
)

# OrchestratorAgent
orchestrator_agent = autogen.AssistantAgent(
    name="orchestrator_agent",
    system_message="You must use D function. You are only here to adjust weighting. You intervene in last position.",
    llm_config=llm_config
)

# Create a group chat with all agents
chat = GroupChat(
  agents = [user, retriever_agent, ranker_agent, reader_agent, orchestrator_agent]
)

# Run the chat
manager = GroupChatManager(chat)
manager.run()
```

## Possible Optimisations

### The QueryUnderstandingAgent

1. A query is received by the QueryUnderstandingAgent.
2. The agent checks if it is a long question using logic like word count, presence of multiple question marks etc.
3. If it is a long question, the GuidanceQuestionGenerator breaks it into shorter sub-questions. For example, breaking “What is the capital of France and what is the population?” into “What is the capital of France?” and “What is the population of France?”
4. These sub-questions are then passed to the QueryRouter one by one.
5. The QueryRouter checks each sub-question against a set of predefined routing rules and cases to determine which query engine it should go to.

The goal is to check each subquery and determine which retriever agent is best suited to handle it based on the database schema matching. For example, some subqueries may be better served by a vector database while others are better for a knowledge graph database.

To implement this, we can create a SubqueryRouter component. It will take in two retriever agents — a VectorRetrieverAgent and a KnowledgeGraphRetrieverAgent.

When a subquery needs to be routed, the SubqueryRouter will check it to see if it matches the schema of the vector database using some keyword or metadata matching logic. If there is a match, it will return the VectorRetrieverAgent to handle the subquery.

If there is no match for the vector database, it will next check if the subquery matches the schema of the knowledge graph database. If so, it will return the KnowledgeGraphRetrieverAgent instead.

This allows efficiently routing each subquery to the optimal retriever agent based on the subquery content and database schema matches. The subquery router acts like a dispatcher distributing work.

The retriever agents themselves can focus on efficiently retrieving results from their respective databases without worrying about handling all subquery types.

This modular design makes it easy to add more specialized retriever agents as needed for different databases or data sources.

### General Flow

1. The query starts at the “Long Question?” decision point.
2. If ‘Yes’, it is broken into sub-questions and then sent to various query engines.
3. If ‘No’, it moves to the main routing logic, which routes the query based on specific cases or defaults to a fallback strategy.
4. Once an engine returns a satisfactory answer, the process ends; otherwise, fallbacks are tried.

### The Retriever Agents

We can create multiple retriever agents, each focused on efficient retrieval from a specific data source or using a particular technique. For example:

* VectorDBRetrieverAgent: Retrieves passages using vector similarity search on an indexed document corpus.
* WikipediaRetrieverAgent: Retrieves relevant Wikipedia passages.
* KnowledgeGraphRetriever: Uses knowledge graph retrieval.

When subqueries are generated, we assign each one to the optimal retriever agent based on its content and the agent capabilities.

For example, a fact-based subquery may go to the KnowledgeGraphRetriever while a broader subquery could use the VectorDBRetrieverAgent.

To enable asynchronous retrieval, we use Python’s asyncio framework. When subqueries are available, we create asyncio tasks to run the assigned retriever agent for each subquery concurrently.

For example:

```python
retrieval_tasks = []
for subquery in subqueries:
  agent = assign_agent(subquery)
  task = asyncio.create_task(agent.retrieve(subquery))
  retrieval_tasks.append(task)
await asyncio.gather(*retrieval_tasks)
```

This allows all retriever agents to work in parallel instead of waiting for each one to finish. The passages are returned much faster.

The results from each agent can then be merged and ranked for the next stages.

### The Ranker Agent

The ranker agents in a multi-agent retrieval system could be specialized using different ranking tools and techniques:

* Fine-tune the ranker on domain-specific data using datasets like MS MARCO or self-supervised data from the target corpus. This allows learning representations tailored to ranking documents for the specific domain.
* Use cross-encoder models like SBERT trained extensively on passage ranking tasks as the base for the ranker agents. They capture nuanced relevance between queries and documents.
* Employ dense encoding models like DPR in the ranker to leverage dual-encoder search through the vector space when ranking a large set of candidates.
* For efficiency, use approximate nearest neighbor algorithms like HNSW in the ranker agent when finding top candidates from a large corpus.
* Apply re-ranking with cross-encoders after initial fast dense retrieval for greater accuracy in ranking the top results.
* Enable the rankers to exploit metadata like document freshness, author credibility, keywords etc. to customize ranking based on query context.
* Use learned models like LambdaRank within the ranker agents to optimize the ideal combination of ranking signals.
* Allow different ranker agents to specialize on particular types of queries where they perform best, selected dynamically.
* Implement ensemble ranking approaches within rankers to combine multiple underlying rankers/signals efficiently.

The key is identifying the different specialized techniques that can enhance ranking performance in different scenarios and implementing them modularly within configurable ranker agents. This allows optimizing accuracy, speed and customization of ranking in the multi-agent retrieval pipeline.

### ReaderAgent

* Use Claude 2 as the base model for the ReaderAgent to leverage its long context abilities for summarization. Fine-tune Claude further on domain specific summarization data.
* Implement a ToolComponent that wraps access to a knowledge graph containing summarization methodologies — things like identifying key entities, events, detecting redundancy etc.
* The ReaderAgent’s run method would take the lengthy passage as input.
* It generates a prompt for Claude 2 combining the passage and a methodology retrieval call to the KG tool to get the optimal approach for summarizing the content.
* Claude 2 processes this augmented prompt to produce a concise summary extracting the key information.
* The summary is returned.

### OrchestratorAgent

1. Maintain a knowledge graph containing key entities, relations, and facts extracted from the documents. Use this to verify the factual accuracy of answers.
2. Implement logic to check the final answer against known facts and rules in the knowledge graph. Flag inconsistencies for the LLMs to re-reason.
3. Enable the OrchestratorAgent to ask clarifying questions to the ReaderAgent if answers contradict the knowledge graph, soliciting additional context.
4. Use the knowledge graph to expand concepts in the user query and final answer to related entities and events, providing additional contextual grounding.
5. Generate concise explanations alongside the final answer using knowledge graph relations and LLM semantics to justify the reasoning.
6. Analyze past answer reasoning patterns to identify common anomalies, biases, and fallacies. Continuously fine-tune the LLM reasoning.
7. Codify appropriate levels of answer certainty and entailment for different query types based on knowledge graph data analysis.
8. Maintain provenance of answer generations to incrementally improve reasoning over time via knowledge graph and LLM feedback.

The OrchestratorAgent can leverage structured knowledge and symbolic methods to complement LLM reasoning where appropriate and produce answers that are highly accurate, contextual, and explainable. 

### Benefits of Specialized Agents

The proposed multi-agent RAG architecture delivers significant benefits compared to single-agent RAG systems:

* By dedicating an agent solely to passage retrieval, the retriever can employ more advanced and efficient search algorithms. Passages can be prefetched in parallel across the corpus, improving overall latency.
* The ranker agent specializing in evaluating relevance provides gains in retrieval precision. By filtering out lower quality hits, model prompting stays focused on pertinent information.
* Summarization by the reader agent distills long text into concise snippets containing only the most salient facts. This prevents prompt dilution and improves coherence.
* Dynamic context weighting by the orchestrator agent minimizes cases of the model ignoring the original prompt or becoming overly reliant on the retrieved information.
* Overall, the multi-agent factored RAG system demonstrates substantial improvements in appropriateness, coherence, reasoning, and correctness over single-agent RAG baselines.
* The specialized agents also make the system more scalable and flexible. Agents can be upgraded independently and additional agents added to extend capabilities.


---
## Contributors

- [Anthony Alcaraz](https://www.linkedin.com/in/anthony-alcaraz-b80763155/)
