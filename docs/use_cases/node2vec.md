# POST2 Leveraging Relational Information

## Introduction

Information comes in various forms such as text, images, relations and more, each capturing a different perspective of the world. Relations are particularly interesting because they capture the interactions between different entities and can provide insights into the structure and dynamics of the network they form. In this post, we will focus on how we can leverage relational information to learn meaningful representations of entities in a network. 

We will introduce two node representation learning techniques, Node2Vec trough an example dataset. This dataset is …

We can load the dataset as follows:

```python
from torch_geometric.datasets import AttributedGraphDataset
ds = AttributedGraphDataset("./data", "wiki")[0]
```

We will evaluate representations by measuring classification performance on this dataset. A KNN classifier will be used with 15 neighbors and cosine similarity as the similarity metric. Accuracy and Macro F1 scores will be computed to measure the classification performance.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def evaluate(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    model = KNeighborsClassifier(n_neighbors=15, metric="cosine")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print("Accuracy", f1_score(y_test, y_pred, average="micro"))
    print("F1 macro", f1_score(y_test, y_pred, average="macro"))
```

First, we will explore how text-based representations can be used to solve the classification problem:

```python
evaluate(ds.x, ds.y)
>>> Accuracy 0.849
>>> F1 macro 0.849
```

This is not bad, let’s see if we can do better by utilizing relational information.

## Learning node embeddings with Node2Vec

Before delving into the details, let's briefly understand node embeddings. These are vector representations of nodes in a network, similar to how we convert words into vectors, such as Word2Vec. Essentially, these representations capture the structural role and properties of the nodes in the network.

Node2Vec is an algorithm that employs the Skip-Gram method to learn node representations. It operates by modeling the conditional probability of encountering a context node given a source node in node sequences (random walks):

`P(context∣source) ~ exp(⟨w[context],w[source]⟩)`

The embeddings (`w`) are learned by maximizing the co-occurance probability for (source,context) pairs drawn from the true data distribution (positive pairs), and at the same time minimizing pairs that are drawn from a synthetic noise distribution. This process ensures that the embedding vectors of similar nodes are close in the embedding space, while dissimilar nodes are further apart.

The random walks are sampled according to a policy, which is guided by 2 parameters: return (p), and in-out (q).

- The return parameter (p) impacts the likelihood of returning to the previous node. A higher p leads to more locally focused walks.
- The in-out parameter (q) affects the likelihood of visiting nodes in the same or different neighborhood. A higher q encourages Depth First Search, while a lower q promotes Breadth First Search.

These parameters provide a balance between neighborhood exploration and local context. Adjusting p and q can be used to capture different characteristics of the graph.

### Node2Vec embeddings

In our example, we use the `torch_geometric` implementation. We initialize the model by specifying the following attributes:

- `edge_index`: This is a tensor containing the graph's edges in an edge list format.
- `embedding_dim`: This denotes the number of dimensional embeddings we desire.

By default, the `p` and `q` parameters are set to 1, resulting in ordinary random walks.

```python
from torch_geometric.nn import Node2Vec
device="cuda"
n2v = Node2Vec(
    edge_index=ds.edge_index, 
    embedding_dim=128, 
    walk_length=20,
    context_size=10,
    sparse=True
).to(device)
```

The next steps include initializing the data loader and the optimizer. The role of the data loader is to generate training batches. In our case, it will sample the random walks, create skip-gram pairs, and generate corrupted pairs.

The optimizer is used to update the model weights to minimize the loss (cross entropy). In our case, we are using the sparse variant of the Adam optimizer.

```python
loader = n2v.loader(batch_size=128, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(n2v.parameters(), lr=0.01)
```

In the block below, we conduct the actual model training: We iterate over the training batches, calculate the loss, and apply gradient steps.

```python
n2v.train()
for epoch in range(100):
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch: {epoch:03d}, Loss: {total_loss / len(loader):.4f}')
```

Finally, now that we have a fully trained model, we can evaluate the learned embeddings using the `evaluate` function we defined earlier.

```python
embeddings = n2v().detach().cpu() # Access node embeddings
evaluate(embeddings, ds.y)
>>> Accuracy: 0.628
>>> F1 macro: 0.617
```

This is worse than the text based embeddings, but remember, this graph describes our entities from an other perspective - this is not the same information as the text based embeddings. Let’s see if we can improve by combining the two embeddings.

### Node2Vec + Text based embeddings

A straightforward method to combine embeddings from different sources is by concatenating them dimension-wise. For instance, suppose we have text-based embeddings `v_text` and Node2Vec embeddings `v_n2v`. The fused representation would then be `v_fused = torch.cat((v_n2v, v_text), dim=1)`. If these new embeddings are input into a KNN algorithm that uses dot product similarity, the resulting scores would equate to the sum of the individual text-based and Node2Vec scores. However, before combining the two representations, let’s look at the L2 norm distribution of both embeddings:

![L2 norm distribution of text based and Node2Vec embeddings](../assets/use_cases/node2vec/l2_norm.png)

From the plot, it's evident that the scales of the embedding vector lengths differ. When we want to use them together, the one with the larger magnitude will overshadow the smaller one. To mitigate this, we'll equalize their lengths by dividing each one by its average length. However, this still not necessarily yields the best performance. To optimally combine the two embeddings, we'll introduce a weighting factor: `x = torch.cat((alpha * v_n2v, v_text), dim=1)`. To determine the appropriate `alpha` value, we'll employ a 1D grid search approach. The results of this approach are displayed in the subsequent graph.

![Grid search for alpha](../assets/use_cases/node2vec/grid_search_alpha.png)

Now, we can evaluate the combined representation using the score of alpha that we've obtained (0.388).

```python
v_n2v = normalize(n2v().detach().cpu())
v_text = normalize(ds.x)

x = np.concatenate((best_alpha*v_n2v,v_text), axis=1)
evaluate(x, ds.y)
>>> Accuracy 0.901
>>> F1 macro 0.900
```

## Conclusion


| Embedding | Text Based | Node2Vec | Combined |
| --- | --- | --- | --- |
| F1 (macro) |  0.001 | 0.0055 | **0.065** |
| Accuracy |  0.003 | 0.0154 | **0.150** |

---
## Contributors

- [Richárd Kiss, author](https://www.linkedin.com/in/richard-kiss-3209a1186/)

---

[ SAGE DRAFT FROM HERE ]

## Learning node embeddings with GraphSAGE

GraphSAGE, unlike transductive methods such as Node2Vec, creates node embeddings by incorporating features from neighboring nodes. This approach is beneficial as it enables the embedding of nodes that were not present during training.

This inductive behaviour is achieved by using a particular type of Neural Network designed specifically for graph-structured data, known as Graph Neural Networks (GNN). The central concept behind GNN layers is to build the hidden representation by combining the node's features with the aggregated features of its neighboring nodes, as illustrated in the picture below.

[GNN image]

Here [https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/) you can learn more about GNNs.

The model learns weights by minimizing the cross-entropy in the link prediction task, which involves predicting if an edge exists between two nodes in the graph. The link probability is modeled in a similar way to how Node2Vec models co-occurrence probability, with a slight difference: instead of learning static embeddings for each node, we use a GNN to construct the embeddings: $sigma(<GNN(x1), GNN(x2)>)$

### GraphSAGE embeddings

Here we are using the `torch_geometric` implementation of the GraphSAGE algorithm, similarly as before. First we create the model by initializing a `GraphSAGE` object:

```python
from torch_geometric.nn import GraphSAGE
sage = GraphSAGE(
    ds.num_node_features, hidden_channels=256, out_channels=128, num_layers=2
).to(device)
```

…

```python
from torch_geometric.loader import LinkNeighborLoader
loader = LinkNeighborLoader(
    ds,
    batch_size=512,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[15,10],
    transform=T.NormalizeFeatures(),
    num_workers=4
)
```

…

```python
optimizer = torch.optim.Adam(sage.parameters(), lr=0.01)
```

…

```python
def train():
    sage.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # create node representations
        h = sage(batch.x, batch.edge_index)
        # take head and tail representations
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        # compute pairwise edge scores
        pred = (h_src * h_dst).sum(dim=-1)
        # apply cross entropy
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / ds.num_nodes

for epoch in range(1, 50):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```