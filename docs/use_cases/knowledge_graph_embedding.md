<!-- SEO: Introduction to Knowledge Graphs. Introduction to Knowledge Graph Embedding. Introduction to the DistMult algorithm. Example code for training KGE model. Example code for evaluating KGE model. Example code for answering questions with a KGE model. KGE and LLM comparison on QA task.  -->

# Learning Scemantic Representations for Knowledge Graphs

<!-- TODO: Cover image: 
1. You can create your own cover image and put it in the correct asset directory,
2. or you can give an explanation on how it should be and we will help you create one. Please tag arunesh@superlinked.com or @AruneshSingh (GitHub) in this case. -->

In today's world, LLMs (Large Language Models) are everywhere, doing all sorts of language-related tasks really well. They're the go-to solution for understanding and using text in many different ways.

However, in some specific areas, there are other more specialized approaches that might give better results for specific tasks. 

This post digs into when these popular language models might not be the best choice. We'll see how specialized methods, like KGE (Knowledge Graph Embedding) algorithms, could actually be better for certain tasks.

## What are Knowledge Graphs?

Now, let's zoom in on KGs (Knowledge Graphs). We use KGs to describe how different entities, like people, places, or more generally "things", relate to each other. For example a KG can show us how a famous writer is linked to their books or how a book is connected to its received awards:

![Knowledge Graph example](../assets/use_cases/knowledge_graph_embedding/small_kg.png)

In certain areas where understanding these specific connections is crucial - like recommendation systems, search engines, information retrieval, etc. - KGs step in as specialized tools. They help computers grasp the detailed relationships between things.

## What is Knowledge Graph Embedding (KGE)?

KGE algorithms take this tangled complex web of connections and turn it into something AI systems can understand better: vectors. This might raise a question - if we already have knowledge of the connections between nodes and their relations, then why do we need to take the effort to learn embeddings?

The challenge with KGs is that they are usually incomplete. This means that there might be some edges that should ideally be present but are missing. These missing links could be the result of inaccuracies in the data collection process, or it could simply be a reflection of the imperfect nature of our data source. According to [this article](https://towardsdatascience.com/neural-graph-databases-cc35c9e1d04f), in large open-source knowledge bases we can observe a significant amount of incompleteness: 

“… in Freebase, 93.8% of people have no place of birth and [78.5% have no nationality](https://aclanthology.org/P09-1113.pdf), about 68% of people [do not have any profession](https://dl.acm.org/doi/abs/10.1145/2566486.2568032), while in Wikidata, about [50% of artists have no date of birth](https://arxiv.org/abs/2207.00143), and only [0.4% of known buildings have information about height](https://dl.acm.org/doi/abs/10.1145/3485447.3511932).”

These imperfections, whether minor or major, can pose significant difficulties if we solely rely on the graph for information. In such a scenario, KGE algorithms prove to be extremely beneficial. They allow us to retrieve the missing data by utilizing the learned semantic meaning of nodes and links.

### How KGE algorithms work?

In general, KGE algorithms work by defining a similarity function in the embedding space, and learn the model by minimizing a loss function that penalizes the discrepancy between this similarity function and some notion of similarity in the graph. KGE algorithms can differ in the choice of the similarity function and the definition of node similarity in the graph.

The simplest approach is to consider nodes that are connected by an edge as similar. Then, the task of learning node embeddings can be defined as a classification task. Given the embeddings of two nodes and a relation, the task is to determine how likely it is that they are similar (connected).

For our demo, we've opted for the DistMult KGE algorithm. It works by representing the likelihood of relationships between entities (the similarity function) as a bilinear function. Essentially, it assumes that the score of a given triple (comprising a head entity $h$, a relationship $r$, and a tail entity $t$) can be computed as $h^T \text{diag}(r) t$. 

![DistMult similarity function](../assets/use_cases/knowledge_graph_embedding/distmult.png)

[(Image source)](https://data.dgl.ai/asset/image/ke/distmult.png)

The model parameters are learned by minimizing the cross entropy between real and corrupted triplets. This process allows the model to learn the intricate relationships within the KG.

In the following two sections we will walk through how you can:

1. Build and train a DistMult model.
2. Use the model to answer questions.

## Build and Train a KGE model

In our demo, we'll use a subgraph of the Freebase Knowledge Graph, a database of general facts (now transferred to Wikidata after its 2014 shutdown).
This graph contains 14541 different entities, 237 different relation types and 310116 edges in total. 

You can load the graph as follows:

```python
from torch_geometric.datasets import FB15k_237
train_data = FB15k_237("./data", split='train')[0]
```
We will use PyTorch Geometric, a library built on top of PyTorch, to construct and train the model. This library is specifically designed for building machine learning models on graph-structured data.

The implementation of the DistMult algorithm lies under the `torch_geometric.nn` package. To create the model, we need to specify the following three parameters:

- `num_nodes`: The number of distinct entities in the graph (in our case, 14541)
- `num_relations`: The number of distinct relations in the graph (in our case, 237)
- `hidden_channels`: The dimensionality of the embedding space (for this, we'll use 64)

For additional configuration of the model, please refer to the [documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.DistMult.html).

```python
from torch_geometric.nn import DistMult
model = DistMult(
    num_nodes=train_data.num_nodes, 
    num_relations=train_data.num_edge_types, 
    hidden_channels=64
)
```

The process of model training in PyTorch follows a standard set of steps:

The first step involves the initialization of an optimizer. The optimizer is a fundamental part of machine learning model training as it adjusts the parameters of the model in order to reduce the loss. In this example we will use the [`Adam`](LINK) optimizer.

```python
import torch.optim as optim
# 1. initialize optimizer
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
```

Next, a data loader is created. The purpose of this loader is to return a batch iterator over the entire dataset. The batch size can be adjusted depending on the specific requirements of the model and the capacity of the hardware. The loader not only provides an efficient way to load the data but also shuffles it to ensure that the model is not influenced by the order of the training samples.

```python
# 2. create data loader on the training set
loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=2000,
    shuffle=True,
)
```

The final step is the execution of the training loop. This is where the actual learning takes place. The model processes each batch of data, then we compare the output to the expected output (labels). The model parameters are then adjusted to bring the outputs closer to the labels. This process continues until the model's performance on a validation set reaches an acceptable level or a predefined number of iterations has been completed (in this example, we'll stick with the latter option).
```python
# 3. usual torch training loop
EPOCHS = 20
model.train()
for e in range(EPOCHS):
    l = []
    for batch in loader:
        opt.zero_grad()
        loss = model.loss(*batch)
        l.append(loss.item())
        loss.backward()
        opt.step()
    print(f"Epoch {e} loss {sum(l) / len(l):.4f}")
```

Now that we have a trained model, we can do some experiments to see how well the learned embeddings capture semantic meaning. To do so, we will construct 3 fact triplets and then we'll use the model to score these triplets. The triplets are:
1. France contains Burgundy (which is true)
2. France contains Rio de Janeiro (which is not true)
3. France contains Bonnie and Clyde (which does not make sense at all)
```python
# Get node and relation IDs
france = nodes["France"]
rel = edges["/location/location/contains"]
burgundy = nodes["Burgundy"]
riodj = nodes["Rio de Janeiro"]
bnc = nodes["Bonnie and Clyde"]

# Define triples
head_entities = torch.tensor([france, france, france], dtype=torch.long)
relationships = torch.tensor([rel, rel, rel], dtype=torch.long)
tail_entities = torch.tensor([burgundy, riodj, bnc], dtype=torch.long)

# Score triples using the model
scores = model(head_entities, relationships, tail_entities)
print(scores.tolist())
>>> [3.890, 2.069, -2.666]

# Burgundy gets the highest score
# Bonnie and Clyde gets the lowest (negative) score
```

## Answering questions with the model

Next, we'll show how we can apply the trained model to answer questions. Let's consider this question: "What is Guy Ritchie's profession?"
To answer this question, we start by finding the embedding vectors of "Guy Ritchie" and the relation "profession."
```python
# Accessing node and relation embeddings
node_embeddings = model.node_emb.weight
relation_embeddings = model.rel_emb.weight

# Accessing embeddings for specific entities and relations
guy_ritchie = node_embeddings[nodes["Guy Ritchie"]]
profession = relation_embeddings[edges["/people/person/profession"]]
```
Remember, the DistMult algorithm models connections as a bilinear function of the (head, relation, tail) triplet, so we can express our question as: <Guy Ritchie, profession, ?>. Whichever node maximizes this expression will be the answer of the model.

```python
# Creating embedding for the query based on the chosen relation and entity
query = guy_ritchie * profession

# Calculating scores using vector operations
scores = node_embeddings @ query

# Find the index for the top 5 scores
sorted_indices = scores.argsort().tolist()[-5:][::-1]
# Get the score for the top 5 index
top_5_scores = scores[sorted_indices]

>>> [('film producer', 3.171), # Correct answer
 ('author', 2.923),
 ('film director', 2.676),
 ('writer', 2.557),
 ('artist', 2.522)]
```

Remarkably, the model managed to correctly answer the question, despite the actual fact not being present in the training graph. This impressive feat indicates the model's ability to interpret and infer information that isn't explicitly included in the graph (incompleteness).

Furthermore, an interesting observation can be made in relation to the top five relevant entities identified by the model. All of these were professions, which suggests that the model has successfully learned and understood the concept of a "profession". This understanding goes beyond merely recognizing the term "profession"; the model has evidently grasped the broader context and implications associated with the concept.

Moreover, a deeper look into these five professions reveals an intriguing pattern - they're all closely related to the film industry. This pattern suggests that the model has not only understood the concept of a profession but also managed to capture the semantic meaning of the question, which was related to a film director. Thus, the model was able to link the general concept of a profession to the specific context of the film industry, a testament to its ability to capture and interpret semantic meaning.

In conclusion, the model's performance in this scenario demonstrates its potential in understanding concepts, interpreting context, and extracting semantic meaning.

You can find all the code for this demonstration in [this](https://drive.google.com/file/d/1G3tJ6Nn_6hKZ8HZGpx8OHpWwGqp_sQtF/view?usp=sharing) notebook.

## Comparing KGE with LLM performance on a large Knowledge Graph

In this section, we're comparing the performance of Knowledge Graph Embeddings (KGE) and Language Models (LLMs) on a dataset called ogbl-wikikg2. This dataset is drawn from Wikidata and includes 2.5 million unique entities, 535 types of relations, and 17.1 million fact triplets. We'll evaluate their performance using hit rates (the ratio of correct answers), following the guidelines provided [here](https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2).

Our approach involved creating textual representations for each node within the graph. We did this by crafting sentences that describe their connections, like this: "[node] [relation1] [neighbor1], [neighbor2]. [node] [relation2] [neighbor3], [neighbor4]. ..." These textual representations were then fed into a LLM – specifically, the `BAAI/bge-base-en-v1.5` model available on [HuggingFace](https://huggingface.co/BAAI/bge-base-en-v1.5). The resulting embeddings from this process served as our node embeddings.

For queries, we took a similar textual representation approach, creating descriptions of the query but omitting the specific entity in question. With these representations in hand, we utilized a dot product similarity to find and rank relevant answers.

For the KGE algorithm, we employed DistMult with a 250-dimensional embedding space.


### Results
You can see the results on the Open Graph Benchmark query set in the table below:

| metric/model  | Random | LLM | DistMult|
| --- | --- | --- | --- |
| HitRate@1 |  0.001 | 0.0055 | **0.065** |
| HitRate@3 |  0.003 | 0.0154 | **0.150** |
| HitRate@10 |  0.010 | 0.0436 | **0.307** |

The table above clearly shows that LLM performs three times better than the method of randomly ordering nodes. It is also clear that KGE significantly outperforms LLM, with hit rates almost ten times higher. Notably, DistMult can find the correct answer on its first attempt more frequently than LLM can in ten attempts. It's important to point out that we used only 250-dimensional embeddings with DistMult, while LLM outputs 768-dimensional vectors, which makes the performance difference even more noticeable.

These results strongly support that KGE is more suitable than LLMs for tasks where relational information is super important.

## Limitations

One reason why the LLM approach might struggle with performance is due to the formulation used, where each node is mapped to a sequence of sentences describing its connections. This method tends to overload the input text with an extensive amount of information for a single node. LLMs are typically not trained to handle such broad and diverse information within a single context; their strength lies in processing more focused and specific textual information.

While DistMult stands as a simple but powerful tool for embedding KGs, it does come with limitations:
1. Cold start problem: When the graph evolves or changes over time, DistMult can't represent new nodes introduced later on, or can't model the effect of new connections introduced to the graph.
2. It struggles with complex questions: While it excels in straightforward question-answering scenarios, the DistMult model falls short when faced with complex questions that demand a deeper comprehension extending beyond immediate connections. Other KGE algorithms better suit such tasks.



---
## Contributors

- [Richárd Kiss, author](https://www.linkedin.com/in/richard-kiss-3209a1186/)