<!-- SEO: Introduction to Node Representation Learning. Introduction to Node2Vec. Introduction to GraphSAGE. Example code for training Node2Vec. Example code for training GraphSAGE. Node2Vec GraphSAGE comparison. Node2Vec and GraphSAGE pro con. -->

# Representation Learning on Graph Structured Data

## Introduction

Different types of information, like words, pictures, and connections between things, show us different sides of the world. Relationships, especially, are interesting because they show how things interact and create networks. In this post, we'll talk about how we can use these relationships to understand and describe things in a network better.

We're diving into a real-life example to explain how entities can be turned into vectors using their connections, a common practice in machine learning. The dataset we're going to work with is the a subset of the Cora citation network. It comprises 2708 scientific papers (nodes) and the connections indicate citations between them. Each paper has a BoW (Bag-of-Words) descriptor containing 1433 words. 

The papers in the dataset are also divided into 7 different topics, each paper belongs to exactly one of them. In this article we are going to explore how well we can predict the topic of a paper using different information sources.

The dataset can be loaded as follows:

```python
from torch_geometric.datasets import Planetoid
ds = Planetoid("./data", "Cora")[0]
```

We will evaluate representations by measuring the classification performance (Accuracy and macro F1). We'll use a KNN (K-Nearest Neighbors) classifier with 15 neighbors and cosine similarity as the similarity metric:

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

First, we'll see how the well the BoW representations can be used to solve the classification problem:

```python
evaluate(ds.x, ds.y)
>>> Accuracy 0.738
>>> F1 macro 0.701
```

Additionally, we wanted to see if citations show up in the BoW features. So, we made a plot that compares connected and not connected pairs of papers based on how similar their BoW features are.

![BoW cosine similarity edge counts](../assets/use_cases/node_representation_learning/bins_bow.png)

In this plot, we divided the groups (shown on the y-axis) to have about the same number of pairs in each. The only exception was the 0-0.04 group, where lots of pairs had no similar words - they couldn't be split into smaller groups.

From the plot, it's clear that connected nodes usually have higher cosine similarities. This means papers that cite each other often use similar words. But when we ignore zero similarities, papers that have note cited each other seem to have a wide range of common words.

Even though some information about the connectivity is present in the BoW features, it is not sufficient to reconstruct the citation graph accurately. This might be problematic if the network structure contains additional information necessary for solving the paper classification problem. If we could extract that supplementary information, theoretically, we might be able to build a more accurate classifier. In the following sections, we will look at two methods for learning node representations that capture node connectivity more accurately.

## Learning node embeddings with Node2Vec

Before delving into the details, let's briefly understand node embeddings. These are vector representations of nodes in a network. Essentially, these representations capture the structural role and properties of the nodes in the network.

Node2Vec is an algorithm that employs the Skip-Gram method to learn node representations. It operates by modeling the conditional probability of encountering a context node given a source node in node sequences (random walks):

$P(\text{context}|\text{source}) = \frac{1}{Z}\exp(w_{c}^Tw_s) $

Here $w_c$ and $w_s$ are the embeddings of the context node $c$ and source node $s$ respectively. The variable $Z$ serves as a normalization constant, which, for computational efficiency, is never explicitly computed.

The embeddings are learned by maximizing the co-occurance probability for (source,context) pairs drawn from the true data distribution (positive pairs), and at the same time minimizing for pairs that are drawn from a synthetic noise distribution. This process ensures that the embedding vectors of similar nodes are close in the embedding space, while dissimilar nodes are further apart (w.r.t. dot product).

The random walks are sampled according to a policy, which is guided by 2 parameters: return $p$, and in-out $q$.

- The return parameter $p$ impacts the likelihood of returning to the previous node. A higher p leads to more locally focused walks.
- The in-out parameter $q$ affects the likelihood of visiting nodes in the same or different neighborhood. A higher q encourages Depth First Search, while a lower q promotes Breadth First Search.

These parameters provide a balance between neighborhood exploration and local context. Adjusting $p$ and $q$ can be used to capture different characteristics of the graph.

### Node2Vec embeddings

In our example, we use the `torch_geometric` implementation of the Node2Vec algorithm. We initialize the model by specifying the following attributes:

- `edge_index`: a tensor containing the graph's edges in an edge list format.
- `embedding_dim`: specifies the dimensionality of the embedding vectors.

By default, the `p` and `q` parameters are set to 1, resulting in ordinary random walks. For additional configuration of the model, please refer to the [documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html).

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

The next steps include initializing the data loader and the optimizer. 

The role of the data loader is to generate training batches. In our case, it will sample the random walks, create skip-gram pairs, and generate corrupted pairs by replacing either the head or tail of the edge from the noise distribution.

The optimizer is used to update the model weights to minimize the loss. In our case, we are using the sparse variant of the Adam optimizer.

```python
loader = n2v.loader(batch_size=128, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(n2v.parameters(), lr=0.01)
```

In the code block below, we conduct the actual model training: We iterate over the training batches, calculate the loss, and apply gradient steps.

```python
n2v.train()
for epoch in range(200):
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
>>> Accuracy: 0.822
>>> F1 macro: 0.803
```

These results are better than using BoW representations! 

As previously with BoW features, let's look at if conencted nodes separate by cosine similarity from not connected node pairs.

![N2V cosine similarity edge counts](../assets/use_cases/node_representation_learning/bins_n2v.png)

This time we can see a well defined separation, meaning that these embeddings capture the connectivity of the graph much better.

Let’s see if we can further improve classification performance by combining the two information sources, relations and textual features.

### Node2Vec + Text based embeddings

A straightforward approach for combining embeddings from different sources is by concatenating them dimension-wise. We have BoW features `v_bow` and Node2Vec embeddings `v_n2v`. The fused representation would then be `v_fused = torch.cat((v_n2v, v_bow), dim=1)`. However, before combining the two representations, let’s look at the L2 norm distribution of both embeddings:

![L2 norm distribution of text based and Node2Vec embeddings](../assets/use_cases/node_representation_learning/l2_norm.png)

From the plot, it's clear that the scales of the embedding vector lengths differ. When we want to use them together, the one with the larger magnitude will overshadow the smaller one. To mitigate this, we'll  divide each embedding vector by their average length. However, this still not necessarily yields the best performance. To optimally combine the two embeddings, we'll introduce a weighting factor ($\alpha$). The combined representations are constructed as `x = torch.cat((alpha * v_n2v, v_bow), dim=1)`. To determine the appropriate value for $\alpha$, we'll employ a 1D grid search approach. The results are displayed in the following plot.

![Grid search for alpha](../assets/use_cases/node_representation_learning/grid_search_alpha_bow.png)

Now, we can evaluate the combined representation using the value of alpha that we've obtained (0.517).

```python
v_n2v = normalize(n2v().detach().cpu())
v_bow = normalize(ds.x)

x = np.concatenate((best_alpha*v_n2v,v_bow), axis=1)
evaluate(x, ds.y)
>>> Accuracy 0.852
>>> F1 macro 0.831
```
The results show that by combining the representations obtained from solely the network structure and text of the paper can improve performance. Specifically, in our case, this fusion contributed to a 3.6% improvement from the Node2Vec only and 15.4% from the BoW only classifiers.

This sounds really good however, what if we are given new papers to classify? 
Unlike BoW, which can be generated easily, Node2Vec features require retraining the entire model. Even if we initiate from prior embeddings, adapting these for new entities proves inconvenient. Node2Vec's limitation lies in its inability to generate embeddings for entities not present during its training phase. However, this doesn't mean that Node2Vec is useless. In scenarios where the graph is static, it is still a very robust and powerful tool.

For dynamic networks, where entities evolve or new ones emerge, inductive approaches like GraphSAGE come into play.

## Learning inductive node embedding with GraphSAGE

GraphSAGE is an inductive representation learning algorithm that leverages GNNs (Graph Neural Networks) to create node embeddings. Instead of learning static node embeddings for each node, it learns an aggregation function on node features that outputs node representations. This also means that in this model combines node features with network structure, so we don't have to manually combine the two information sources later on. 

The GraphSAGE layer is defined as follows:

$h_i^{(k)} = \sigma(W (h_i^{(k-1)} + \underset{j \in \mathcal{N}(i)}{\Sigma}h_j^{(k-1)}))$

Here $\sigma$ is a nonlinear activation function, $W^{(k)}$ is a learnable parameter of layer $k$, and $\mathcal{N}(i)$ is the set of neighboring nodes of node $i$. As in traditional Neural Networks, we can stack multiple GNN layers. The resulting multi layer GNN will have wider receptive field, i.e. it will be able to consider information from bigger distances thanks to the recursive neighborhood aggregation.

To learn the model parameters, the authors suggest two approaches:
1. If we are dealing with a supervised setting, we can train the network similarly, how we train a conventional NN for the supervised task (for example using Cross Entropy for classification or Mean Squared Error for regression)
2. If we only have access to the graph itself, we can approach model training as an unsupervised task, where the goal is to predict the presence of the edges in the graph based on the node embeddings. In this case the link probabilities are defined as $P(j \in \mathcal{N}(i)) \approx \sigma(h_i^Th_j)$. The loss function is the Negative Log Likelihood of the presence of the edge and $P$.

It is also possible to combine the two approaches by using a linear combination of the two loss functions.
However, in this example we are going stick with the unsupervised variant.

### GraphSAGE embeddings

Here we are using the `torch_geometric` implementation of the GraphSAGE algorithm, similarly as before. First we create the model by initializing a `GraphSAGE` object. We are using a 1 layer GNN, meaning that our model will receive node features from a distance of at most 1. We will have 256 hidden and 128 output dimensions.

```python
from torch_geometric.nn import GraphSAGE
sage = GraphSAGE(
    ds.num_node_features, hidden_channels=256, out_channels=128, num_layers=1
).to(device)
```

The optimizer is constructed in the usual PyTorch fashion. Once again, we'll use `Adam`:

```python
optimizer = torch.optim.Adam(sage.parameters(), lr=0.01)
```

Next the data loader is constructed, this will generate training batches for us. As we are using the unsupervised objective, this loader will:
1. Select a batch of node pairs which are connected by an edge (positive samples).
2. Sample negative examples by either modifying the head or tail of the positive samples. The amount of negative samples per edge is defined by the `neg_sampling_ratio` parameter, which we set to 1. This means for each positive sample we'll have exactly one negative sample.
3. We sample neighbors for a depth of 1 for each selected node. The `num_neighbors` parameter allows us to specify the number of sampled neighbors in each depth. This is particuarly useful when we are dealing with dense graphs and/or multi layer GNNs. Limiting the considered neighbors will decouple computational complexity from the actual node degree. However, in our particular case we set the number to `-1` indicating that we want to sample all of the neighbors.

```python
from torch_geometric.loader import LinkNeighborLoader
loader = LinkNeighborLoader(
    ds,
    batch_size=1024,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[-1],
    transform=T.NormalizeFeatures(),
    num_workers=4
)
```

Here we can see how a batch, returned by the loader actually looks like:

```python
print(next(iter(loader)))
>>> Data(x=[2646, 1433], edge_index=[2, 8642], edge_label_index=[2, 2048], edge_label=[2048], ...)
```
In the `Data` object `x` contains the BoW node features, the `edge_label_index` tensor contains the head and tail node indices for the positive and negative samples, `edge_label` is the binary target for these pairs (1 for positive 0 for negative samples). The `edge_index` tensor holds the adjacency list for the current batch of nodes. 

Now we can train our model as follows:
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

for epoch in range(100):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

Next, we can embed nodes and evaluate the embeddings on the classification task:

```python
embeddings = sage(normalize(ds.x), ds.edge_index).detach().cpu()
evaluate(embeddings, ds.y)
>>> Accuracy 0.844
>>> F1 macro 0.820
```

The results are slightly worse than the results we got by combining Node2Vec with BoW features. However, remember that with this model we can embed completely new nodes too. If our scenario requires inductiveness, GraphSAGE might be a better choice.

## Using better node representations
The Bag-of-Word representation is a simple way for embedding textual documents. However, it has limitations, such as losing the order of specific words and it is not necessarily good in capturing semantic meaning.

We explored LLM-based embeddings, which excel in capturing semantic meaning more effectively. We used the `all-mpnet-base-v2` model available on [Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for embedding the title and abstract of each paper. The results obtained with LLM only, Node2Vec combined with LLM and GraphSAGE trained on LLM features can be fund in the following table along with the relative improvement compared to using the BoW features:

| Metric  | LLM | Node2Vec |  GraphSAGE |
| --- | --- | --- | --- |
| Accuracy  | 0.816 (+10%) | **0.867** (+1.7%) |  0.852 (+0.9%) |
| F1 (macro)  | 0.779 (+11%) | **0.840** (+1%) | 0.831 (+1.3%) |



## Conclusion
From the results we can draw the following conclusions (on this dataset):
1. LLM feature beat BoW features in all scenarios
2. Combining the text based representations with network structure results in an increased classification performance
3. We achieved the best results using Node2Vec with LLM features

Finally, we included some pros and cons for both node representation learning algorithms:

| Aspect | Node2Vec | GraphSAGE|
| --- | --- | --- |
| Generalizing to new nodes | No | Yes |
| Inference time | Constant | We have control over the inference time |
| Accomodating different graph types and objectives | By setting the $p$ and $q$ parameters we can adapt the representations to our fit | Limited control | 
| Combining with other representations | Concatenation | By design the model learns to map node representations to embeddings |
| Dependency on additional representations | Relies solely on graph data |Relies on quality and availability of node representations; impacts model performance if lacking |
| Embedding flexibility | Very flexible node representations | Neighboring nodes can't have much variation in their representations

---
## Contributors

- [Richárd Kiss, author](https://www.linkedin.com/in/richard-kiss-3209a1186/)
