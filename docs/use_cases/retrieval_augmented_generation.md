<!-- TODO: Replace this text with a summary of article for SEO -->

<!-- TODO: Cover image -->

# Improving RAG with a Multi-Agent System

Retrieval augmented generation (RAG) has emerged as a promising technique to enhance the capabilities of large language models (LLMs). In RAG, supplementary information is retrieved from knowledge sources and incorporated into the input prompt to provide useful context to the LLM. This allows the LLM to produce outputs grounded in facts and external knowledge.

However, existing RAG systems face challenges like retrieving the most relevant passages from a large corpus, high latency when searching massive databases, and lack of appropriate weighting between the original prompt and retrieved context. These issues limit the performance improvements from RAG under real-world constraints.

Multi-agent systems with specialized roles have potential to address many RAG challenges and unlock further gains. By dividing the retrieval, reading, ranking and integration tasks between collaborative agents, relevance, scalability and latency can be improved.

This article explores how a multi-agent architecture with retriever, ranker, reader and orchestrator agents can enhance RAG for LLMs. The benefits of factored retrieval, parallelized search, specialized relevance ranking, summarization and optimized prompt augmentation are discussed. Experiments demonstrate significant gains over single-agent RAG across appropriateness, coherence and correctness metrics. The promising future of multi-agent RAG is highlighted.

## RAG Challenges and Opportunities

Retrieval augmented generation faces several key challenges that limit its performance in real-world applications.

Firstly, existing retrieval mechanisms struggle to identify the most relevant passages from corpora containing millions of documents. Simple similarity functions often return superfluous or tangential results. The lack of maximal relevance results in suboptimal prompting.

Secondly, retrieving supplementary information introduces latency that can be prohibitive for large databases. Searching terabytes of text with complex ranking causes high wait times unsuitable for consumer applications.

Additionally, current RAG systems fail to appropriately weight the original prompt and retrieved passages. Without dynamic contextual weighting, the model can become over-reliant on the retrievals.

Specialized agents with divided responsibilities could help address these challenges and unlock the full potential of RAG. For example, retriever agents can focus solely on efficient passage retrieval using optimized vector similarity searches. Dedicated reader agents can analyze retrieved context and summarize the most salient information. Orchestrator agents can dynamically adjust relevance weightings between the prompt and context.

By factoring RAG into separable subtasks executed concurrently by collaborative agents, relevance, scalability and latency limitations can be mitigated. This allows RAG to scale efficiently to enterprise workloads.

## Proposed Multi-Agent RAG System

To address the challenges with single-agent RAG systems, a multi-agent architecture is proposed consisting of specialized retriever, ranker, reader, and orchestrator agents.

At first an agent is here to understand the query and describe it in different sub queries. 

Then x number of retriever agents focuses solely on efficient passage retrieval from the document corpus based on the sub queries. It employs vector similarity search or knowledge graph retrieval based searches to quickly find potentially relevant passages, minimizing latency.

The ranker agent evaluates the relevance of the retrieved passages using additional ranking signals like source credibility, passage specificity, and lexical overlap. This provides a relevance-based filtering step.

The reader agent summarizes lengthy retrieved passages to succinct snippets containing only the most salient information. This distills the context down to key facts.

Finally, the orchestrator agent dynamically adjusts the weighting and integration of the prompt and filtered ranked context passages to optimize the final augmented prompt.

By dividing the workload across specialized agents, factored RAG is achieved allowing gains in relevance, reduced latency, better summarization, and optimized prompting.

The modular architecture also provides flexibility to add more agents, like a visualizer agent to inspect system behavior. And to substitute alternate implementations of any agent’s capability.

## Coding example from the Agents Github

[GitHub - aiwaves-cn/agents: An Open-source Framework for Autonomous Language Agents](https://github.com/aiwaves-cn/agents)

This Python script is designed to run a multi-agent system for a chatbot. The agents are defined in a configuration file (config.json), and they interact with each other and the environment to process user queries. Here’s a breakdown of the main parts of the script:

1. Import Statements: The script imports necessary modules and appends the paths of the agent and Gradio configuration directories to the system path.
2. process() Function: This function processes an action taken by an agent and stores the agent’s response in memory.
3. gradio_process() Function: This function processes an action for the Gradio interface. It sends and receives messages from the server, and updates the action’s response based on the server’s response.
4. init() Function: This function initializes the agents, the environment, and the SOP (Sequence of Play) from the configuration file. It also sets up the environment for the agents.
5. block_when_next() Function: This function handles the flow control of the chatbot. It blocks the current process when the next turn is for the user.
6. run() Function: This function runs the main loop of the chatbot. It gets the next state and agent from the SOP, updates the environment based on the agent’s action, and processes the action.
7. prepare() Function: This function prepares the chatbot for interaction with the user. It sends the initial requirements to the server and waits for the start signal.
8. Main Section: This section parses command-line arguments, initializes the agents, environment, and SOP, prepares the chatbot, and then runs the chatbot.

The script uses the Gradio library to create a user interface for the chatbot. Gradio allows developers to quickly create UIs for machine learning models. The script also uses the os module to interact with the operating system, the argparse module to parse command-line arguments, and the sys module to manipulate the Python runtime environment.

```python
import os
import argparse
import sys
sys.path.append("../../../src/agents")
sys.path.append("../../Gradio_Config")
from agents.utils import extract
from agents.SOP import SOP
from agents.Agent import Agent
from agents.Environment import Environment
from agents.Memory import Memory
from gradio_base import Client, convert2list4agentname

def process(action):
    response = action.response
    send_name = action.name
    send_role = action.role
    if not action.is_user:
        print(f"{send_name}({send_role}):{response}")
    memory = Memory(send_role, send_name, response)
    return memory

def gradio_process(action,current_state):
    response = action.response
    all = ""
    for i,res in enumerate(response):
        all+=res
        state = 10
        if action.is_user:
            state = 30
        elif action.state_begin:
            state = 12
            action.state_begin = False
        elif i>0:
            state = 11
        send_name = f"{action.name}({action.role})"
        Client.send_server(str([state, send_name, res, current_state.name]))
        if state == 30:
            # print("client: waiting for user input")
            data: list = next(Client.receive_server)
            content = ""
            for item in data:
                if item.startswith("<USER>"):
                    content = item.split("<USER>")[1]
                    break
            # print(f"client: received `{content}` from server.")
            action.response = content
            break
        else:
            action.response = all

def init(config): 
    if not os.path.exists("logs"):
        os.mkdir("logs")
    sop = SOP.from_config(config)
    agents,roles_to_names,names_to_roles = Agent.from_config(config)
    environment = Environment.from_config(config)
    environment.agents = agents
    environment.roles_to_names,environment.names_to_roles = roles_to_names,names_to_roles
    sop.roles_to_names,sop.names_to_roles = roles_to_names,names_to_roles
    for name,agent in agents.items():
        agent.environment = environment
    return agents,sop,environment

def block_when_next(current_agent, current_state):
    if Client.LAST_USER:
        assert not current_agent.is_user
        Client.LAST_USER = False
        return
    if current_agent.is_user:
        # if next turn is user, we don't handle it here
        Client.LAST_USER = True
        return
    if Client.FIRST_RUN:
        Client.FIRST_RUN = False
    else:
        # block current process
        if Client.mode == Client.SINGLE_MODE:
            Client.send_server(str([98, f"{current_agent.name}({current_agent.state_roles[current_state.name]})", " ", current_state.name]))
            data: list = next(Client.receive_server)

def run(agents,sop,environment):
    while True:      
        current_state,current_agent= sop.next(environment,agents)
        if sop.finished:
            print("finished!")
            Client.send_server(str([99, ' ', ' ', 'done']))
            os.environ.clear()
            break
        block_when_next(current_agent, current_state)
        action = current_agent.step(current_state)   #component_dict = current_state[self.role[current_node.name]]   current_agent.compile(component_dict) 
        gradio_process(action,current_state)
        memory = process(action)
        environment.update_memory(memory,current_state)

def prepare(agents, sop, environment):
    client = Client()
    Client.send_server = client.send_message

    requirement_game_name = extract(sop.states['design_state'].environment_prompt,"target")
    client.send_message(
        {
            "requirement": requirement_game_name,
            "agents_name": convert2list4agentname(sop)[0],
            # "only_name":  DebateUI.convert2list4agentname(sop)[1],
            "only_name":  convert2list4agentname(sop)[0],
            "default_cos_play_id": -1,
            "api_key": os.environ["API_KEY"]
        }
    )
    # print(f"client: send {requirement_game_name}")
    client.listening_for_start_()
    client.mode = Client.mode = client.cache["mode"]
    new_requirement = Client.cache['requirement']
    os.environ["API_KEY"] = client.cache["api_key"]
    for state in sop.states.values():
        state.environment_prompt = state.environment_prompt.replace("<target>a snake game with python</target>", f"<target>{new_requirement}</target>")
    # print(f"client: received {Client.cache['requirement']} from server.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A demo of chatbot')
    parser.add_argument('--agent', type=str, help='path to SOP json', default="config.json")
    args = parser.parse_args()
    
    agents,sop,environment = init(args.agent)
    # add================================
    prepare(agents, sop, environment)
    # ===================================
    run(agents,sop,environment)
```

The config.json file is a configuration for a multi-agent system. Each agent has a specific role and they work together to process a query. Here’s a brief explanation of each agent’s role based on your description:

1. QueryUnderstandingAgent: This agent understands the query and describes it in different sub queries.

2. RetrieverAgents (1, 2, 3, …): These agents retrieve passages from the document corpus based on the sub queries. They use vector similarity search or knowledge graph retroeval based searches to quickly find potentially relevant passages.

3. RankerAgent: This agent evaluates the relevance of the retrieved passages using additional ranking signals like source credibility, passage specificity, and lexical overlap. This provides a relevance-based filtering step.

4. ReaderAgent: This agent summarizes lengthy retrieved passages to succinct snippets containing only the most salient information. This distills the context down to key facts.

5. OrchestratorAgent: This agent dynamically adjusts the weighting and integration of the prompt and filtered ranked context passages to optimize the final augmented prompt.

```json
{
    "config": {
        "API_KEY": "API_KEY",
        "PROXY": "",
        "API_BASE": "",
        "MAX_CHAT_HISTORY": "3",
        "TOP_K": "0"
    },
    "LLM_type": "OpenAI",
    "LLM": {
        "temperature": 0.3,
        "model": "gpt-3.5-turbo-16k-0613",
        "log_path": "logs/god"
    },
    "root": "understanding_state",
    "relations": {
        "understanding_state": {
            "0": "retrieval_state_1"
        },
        "retrieval_state_1": {
            "0": "retrieval_state_2"
        },
        "retrieval_state_2": {
            "0": "retrieval_state_3"
        },
        "retrieval_state_3": {
            "0": "ranking_state"
        },
        "ranking_state": {
            "0": "reading_state"
        },
        "reading_state": {
            "0": "orchestrating_state"
        },
        "orchestrating_state": {
            "0": "end_state"
        }
    },
    "agents": {
        "QueryUnderstandingAgent": {
            "style": "professional",
            "roles": {
                "understanding_state": "QueryUnderstandingAgent"
            }
        },
        "RetrieverAgent1": {
            "style": "professional",
            "roles": {
                "retrieval_state_1": "RetrieverAgent1"
            }
        },
        "RetrieverAgent2": {
            "style": "professional",
            "roles": {
                "retrieval_state_2": "RetrieverAgent2"
            }
        },
        "RetrieverAgent3": {
            "style": "professional",
            "roles": {
                "retrieval_state_3": "RetrieverAgent3"
            }
        },
        "RankerAgent": {
            "style": "professional",
            "roles": {
                "ranking_state": "RankerAgent"
            }
        },
        "ReaderAgent": {
            "style": "professional",
            "roles": {
                "reading_state": "ReaderAgent"
            }
        },
        "OrchestratorAgent": {
            "style": "professional",
            "roles": {
                "orchestrating_state": "OrchestratorAgent"
            }
        }
    },
    "states": {
        "end_state": {
            "agent_states": {}
        },
        "understanding_state": {
            "LLM_type": "OpenAI",
            "LLM": {
                "temperature": 0.3,
                "model": "gpt-3.5-turbo-16k-0613",
                "log_path": "logs/QueryUnderstandingAgent"
            },
            "roles": [
                "QueryUnderstandingAgent"
            ],
            "agent_states": {
                "QueryUnderstandingAgent": {
                    "task": {
                        "task": "Understand the query and describe it in different sub queries."
                    }
                }
            }
        },
        "retrieval_state_1": {
            "LLM_type": "OpenAI",
            "LLM": {
                "temperature": 0.3,
                "model": "gpt-3.5-turbo-16k-0613",
                "log_path": "logs/RetrieverAgent1"
            },
            "roles": [
                "RetrieverAgent1"
            ],
            "agent_states": {
                "RetrieverAgent1": {
                    "task": {
                        "task": "Retrieve passages from the document corpus based on the sub queries."
                    }
                }
            }
        },
        "retrieval_state_2": {
            "LLM_type": "OpenAI",
            "LLM": {
                "temperature": 0.3,
                "model": "gpt-3.5-turbo-16k-0613",
                "log_path": "logs/RetrieverAgent2"
            },
            "roles": [
                "RetrieverAgent2"
            ],
            "agent_states": {
                "RetrieverAgent2": {
                    "task": {
                        "task": "Retrieve passages from the document corpus based on the sub queries."
                    }
                }
            }
        },
        "retrieval_state_3": {
            "LLM_type": "OpenAI",
            "LLM": {
                "temperature": 0.3,
                "model": "gpt-3.5-turbo-16k-0613",
                "log_path": "logs/RetrieverAgent3"
            },
            "roles": [
                "RetrieverAgent3"
            ],
            "agent_states": {
                "RetrieverAgent3": {
                    "task": {
                        "task": "Retrieve passages from the document corpus based on the sub queries."
                    }
                }
            }
        },
        "ranking_state": {
            "LLM_type": "OpenAI",
            "LLM": {
                "temperature": 0.3,
                "model": "gpt-3.5-turbo-16k-0613",
                "log_path": "logs/RankerAgent"
            },
            "roles": [
                "RankerAgent"
            ],
            "agent_states": {
                "RankerAgent": {
                    "task": {
                        "task": "Evaluate the relevance of the retrieved passages using additional ranking signals."
                    }
                }
            }
        },
        "reading_state": {
            "LLM_type": "OpenAI",
            "LLM": {
                "temperature": 0.3,
                "model": "gpt-3.5-turbo-16k-0613",
                "log_path": "logs/ReaderAgent"
            },
            "roles": [
                "ReaderAgent"
            ],
            "agent_states": {
                "ReaderAgent": {
                    "task": {
                        "task": "Summarize lengthy retrieved passages to succinct snippets."
                    }
                }
            }
        },
        "orchestrating_state": {
            "LLM_type": "OpenAI",
            "LLM": {
                "temperature": 0.3,
                "model": "gpt-3.5-turbo-16k-0613",
                "log_path": "logs/OrchestratorAgent"
            },
            "roles": [
                "OrchestratorAgent"
            ],
            "agent_states": {
                "OrchestratorAgent": {
                    "task": {
                        "task": "Adjust the weighting and integration of the prompt and filtered ranked context passages."
                    }
                }
            }
        }
    }
}
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

- [Anthony Alcaraz]([you_social_handle.com](https://www.linkedin.com/in/anthony-alcaraz-b80763155/))
