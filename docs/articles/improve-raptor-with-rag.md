![image.png](../assets/use_cases/improve-raptor-with-rag/raptor-1.png)

Traditional [RAG](https://vipul-maheshwari.github.io/2024/02/14/rag-application-with-langchain) setups often split documents into fixed-size chunks, but this can lead to problems in maintaining the semantic coherence of the text. If a key concept spans multiple chunks, and we only retrieve one chunk, the LLM might lack the full understanding of the idea, leading to incomplete or misleading responses. As a result, crucial ideas and relationships between concepts may be overlooked, leading to incomplete or inaccurate responses.

Additionally, In a flat retrieval structure where all the retrieved chunks are treated equally, this can dilute the importance of critical sections. For example, if one section of the document has key insights but gets buried among less relevant chunks, the model won't know which parts to prioritize unless we introduce more intelligent weighting or hierarchical structures. I mean it becomes really difficult during the retrieval to weigh which chunk is more important and might be better suitable as a context. 

### What is RAPTOR?

RAPTOR, which stands for Recursive Abstractive Processing for Tree Organized Retrieval, is a new technique which solves the problems mentioned before. Think of RAPTOR as a librarian who organizes information in a tree-like structure. Instead of simply stacking books in a pile, it clusters similar titles together, creating a hierarchy that narrows as you ascend. Each cluster of books represents a group of related documents, and at the top of each cluster, there’s a summary that encapsulates the key points from all the books below it. This process continues all the way to the top of the tree, providing a comprehensive view of the information—it's like having both a magnifying glass and a telescope!

To visualize this further, think of the leaves of the tree as document chunks. These chunks are grouped into clusters to generate meaningful summaries, which then become the new leaves of the tree. This recursive process repeats until reaching the top.  

### Key terms to look out for

Before we dive in, let’s quickly review some key terms that will be useful as we explore **RAPTOR** tech. I just want to put it up here to make sure you are comfortable with the nitty tech details as we go along. 

1. **GMM Clustering**: Gaussian Mixture Models (GMM) group data into clusters based on statistical probabilities. So instead of rigidly classifying each instance into one category like K-means, GMM generates K-Gaussian distributions that consider the entire training space. This means that each point can belong to one or more distributions.
2. **Dimensionality Reduction**: This process simplifies the data by reducing the number of variables while retaining essential features. It’s particularly important for understanding high-dimensional datasets like embeddings.
3. **UMAP**: Uniform Manifold Approximation and Projection (UMAP) is a powerful dimensionality reduction algorithm we’ll use to shrink the size of our data point embeddings. This reduction makes it easier for clustering algorithms like GMM to cluster high-dimensional embeddings.
4. **BIC and Elbow Method**: Both techniques help identify the optimal number of clusters in a dataset. The Bayesian Information Criterion (BIC) evaluates models based on their fit to the data while penalizing complexity. The Elbow Method plots explained variance against the number of clusters, helping to pinpoint where adding more clusters offers diminishing returns. For our purposes, we’ll leverage both methods to determine the best number of clusters.

### How it actually works?

Now that you’re familiar with the key terms (and if not, no worries—you’ll catch on as we go!), let’s dive into how everything actually works under the hood of RAPTOR. 

- **Starting Documents as Leaves**: The leaves of the tree represent a set of initial documents, which are our text chunks.
- **Embedding and Clustering**: The leaves are embedded and clustered. The authors utilize the UMAP dimensionality reduction algorithm to minimize the embedding size of these chunks. For clustering, Gaussian Mixture Models (GMM) are employed to ensure effective grouping, addressing the challenges posed by high-dimensional vector embeddings.
- **Summarizing Clusters**: Once clustered, these groups of similar chunks are summarised into higher-level abstractions nodes. Each cluster acts like a basket for similar documents, and the individual summaries encapsulate the essence of all nodes within that cluster. This process builds from the bottom up, where nodes are clustered together to create summaries that are then passed up the hierarchy.
- **Recursive Process**: This entire procedure is recursive, resulting in a tree structure that transitions from raw documents (the leaves) to more abstract summaries, with each summary derived from the clusters of various nodes.

![image.png](../assets/use_cases/improve-raptor-with-rag/raptor-2.png)

### Building the RAPTOR

Now that we’ve unpacked how it all works (and you’re still with me hopefully, right?), let’s shift gears and talk about how we actually build the RAPTOR tree. 

**Setup and Imports**

```python
pip install lancedb scikit-learn openai torch sentence_transformers tiktoken umap-learn PyPDF2
```

```python
import os
import uuid
import tiktoken
import re
import numpy as np
import pandas as pd
import transformers
import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt
from openai import OpenAI
from typing import List, Tuple, Optional, Dict
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer

openai_api_key = "sk-XXXXXXXXXXXXXXX"
client = OpenAI(api_key=openai_api_key)
```

### Creating the Chunks

Setting up RAPTOR is pretty straightforward and builds on what we’ve already covered. The first step is to break down our textual documents into smaller chunks. Once we have those, we can convert them into dense vector embeddings.

```python
import os
import PyPDF2

# Function to extract text from a PDF file
def extract_pdf_text(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to split text into chunks with overlap
def split_text(text, chunk_size=1000, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# Function to process all PDFs in a directory
def process_directory(directory_path, chunk_size=1000, chunk_overlap=50):
    all_chunks = []
    # Iterate over all PDF files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")
            
            # Step 1: Extract text from the PDF
            pdf_text = extract_pdf_text(file_path)
            
            # Step 2: Split the extracted text into chunks
            chunks = split_text(pdf_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Append chunks from this file to the overall list
            all_chunks.extend(chunks)
    
    return all_chunks

directory_path = os.path.join(os.getcwd(), "data")  # Specify your directory path
chunk_size = 1000
chunk_overlap = 50

# Process all PDF files in the directory and get the chunks
chunks = process_directory(directory_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Optional: Print the number of chunks and preview some of the chunks
print(f"Total number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks as a preview
    print(f"Chunk {i+1}:\n{chunk}\n")
```

Now that we have our chunks, it’s time to dive into the recursive processing to create summarized nodes. For the embedding part, I’ll be using the `all-MiniLM-L6-v2` model from Sentence Transformers, but feel free to choose any embedding model that suits your needs—it’s entirely up to you!

```python
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

### Clustering and Dimensionality Reduction

Now we have our embedded chunks, it’s time to step on for the next set of tasks. When diving into RAPTOR, one of the biggest hurdles we encounter is the high dimensionality of vector embeddings. Traditional clustering methods like Gaussian Mixture Models (GMM) often struggle with this complexity, making it tough to effectively cluster high-dimensional data chunks. To tackle this challenge, we turn to **Uniform Manifold Approximation and Projection (UMAP)**. UMAP excels at simplifying data while preserving the essential structures that matter most.

A key factor in UMAP's effectiveness is the **`n_neighbors`** parameter. This setting dictates how much of the data's neighbourhood UMAP considers during dimensionality reduction. In simpler terms, it helps you choose between zooming in on details or taking a broader view:

- **Higher `n_neighbors`:** A higher value encourages UMAP to "look at many neighbors," which helps maintain the **global structure** of the data. This results in larger, more general clusters.
- **Lower `n_neighbors`:** Conversely, lowering `n_neighbors` prompts UMAP to "focus on close relationships," enabling it to preserve the **local structure** and form smaller, more detailed clusters.

**Think of it this way:** Imagine you’re at a party. If you take a step back and look around (high `n_neighbors`), you can see the whole room—where the groups are forming, who’s mingling, and the general vibe. But if you lean in closer to a specific group (low `n_neighbors`), you can hear their conversation and pick up on the nuances, like inside jokes or shared interests. Both perspectives are valuable; it just depends on what you want to understand.

In RAPTOR, we leverage this flexibility in `n_neighbors` to create a **hierarchical clustering structure**. We first run UMAP with a higher `n_neighbors` to identify the **global clusters**—the broad categories. Then, we narrow the focus by lowering the value to uncover **local clusters** within those broader groups. This two-step approach ensures we capture both large-scale patterns and intricate details.

### Well, TL;DR:

1. **Dimensionality Reduction** helps manage high-dimensional data, and UMAP is our primary tool for that.
2. The **`n_neighbors`** parameter controls the balance between seeing the "big picture" and honing in on local details.
3. The clustering process begins with **global clusters** (using high `n_neighbors`), followed by a focus on **local clusters** with a lower setting of `n_neighbors`.

```python
def dimensionality_reduction(
    embeddings: np.ndarray,
    target_dim: int,
    clustering_type: str,
    metric: str = "cosine",
) -> np.ndarray:
    if clustering_type == "local":
        n_neighbors = max(2, min(10, len(embeddings) - 1))
        min_dist = 0.01
    elif clustering_type == "global":
        n_neighbors = max(2, min(int((len(embeddings) - 1) ** 0.5), len(embeddings) // 10, len(embeddings) - 1))
        min_dist = 0.1
    else:
        raise ValueError("clustering_type must be either 'local' or 'global'")

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=target_dim,
        metric=metric,
    )
    return umap_model.fit_transform(embeddings)
```

I plan to leverage both the Elbow Method and the Bayesian Information Criterion (BIC) to pinpoint the optimal number of clusters for our analysis.

```python
def compute_inertia(embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    return np.sum(np.min(np.sum((embeddings[:, np.newaxis] - centroids) ** 2, axis=2), axis=1))

def optimal_cluster_number(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    number_of_clusters = np.arange(1, max_clusters + 1)
    inertias = []
    bic_scores = []
    
    for n in number_of_clusters:
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        labels = gmm.fit_predict(embeddings)
        centroids = gmm.means_
        inertia = compute_inertia(embeddings, labels, centroids)
        inertias.append(inertia)
        bic_scores.append(gmm.bic(embeddings))
    
    inertia_changes = np.diff(inertias)
    elbow_optimal = number_of_clusters[np.argmin(inertia_changes) + 1]
    bic_optimal = number_of_clusters[np.argmin(bic_scores)]
    
    return max(elbow_optimal, bic_optimal)

def gmm_clustering(
    embeddings: np.ndarray, 
    threshold: float, 
    random_state: int = SEED
) -> Tuple[List[np.ndarray], int]:
    n_clusters = optimal_cluster_number(embeddings, random_state=random_state)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=2)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs] 
    return labels, n_clusters  
```

### Tree Construction

Now that we’ve wrapped up the clustering part, let’s talk about how we build our hierarchical tree. After several rounds of clustering and summarization (while keeping track of how deep we go), here’s what we have:

- **Leaf Nodes:** These are our original text chunks, forming the base of the tree.
- **Summary Nodes:** As we go up the tree, each node acts like a quick summary of its child nodes, capturing the main idea of the cluster.
- **Hierarchical Embeddings:** The summary nodes can also become the new nodes at their level. Each of these nodes gets its own vector embedding, representing the summarized meaning. So, we’re essentially adding more nodes while enriching them with summaries.

The process flows nicely: we embed the chunks, reduce their dimensions using UMAP, cluster them with Gaussian Mixture Models, start with a broad overview, and then zoom in for more detailed clusters before summarizing.

```python
def clustering_algorithm(
    embeddings: np.ndarray,
    target_dim: int,
    threshold: float,
    random_state: int = SEED
) -> Tuple[List[np.ndarray], int]:
    if len(embeddings) <= target_dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))], 1
    
    # Global clustering
    reduced_global_embeddings = dimensionality_reduction(embeddings, target_dim, "global")
    global_clusters, n_global_clusters = gmm_clustering(reduced_global_embeddings, threshold, random_state=random_state)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Local clustering within each global cluster
    for i in range(n_global_clusters):
        global_cluster_mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings = embeddings[global_cluster_mask]

        if len(global_cluster_embeddings) <= target_dim + 1:
            # Assign all points in this global cluster to a single local cluster
            for idx in np.where(global_cluster_mask)[0]:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], total_clusters)
            total_clusters += 1
            continue

        try:
            reduced_local_embeddings = dimensionality_reduction(global_cluster_embeddings, target_dim, "local")
            local_clusters, n_local_clusters = gmm_clustering(reduced_local_embeddings, threshold, random_state=random_state)

            # Assign local cluster IDs
            for j in range(n_local_clusters):
                local_cluster_mask = np.array([j in lc for lc in local_clusters])
                global_indices = np.where(global_cluster_mask)[0]
                local_indices = global_indices[local_cluster_mask]
                for idx in local_indices:
                    all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

            total_clusters += n_local_clusters
        except Exception as e:
            print(f"Error in local clustering for global cluster {i}: {str(e)}")
            # Assign all points in this global cluster to a single local cluster
            for idx in np.where(global_cluster_mask)[0]:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], total_clusters)
            total_clusters += 1

    return all_local_clusters, total_clusters
    
 def generate_summary(context: str) -> str:
    prompt = f"""
    Provide the Summary for the given context. Here are some additional instructions for you:

    Instructions:
    1. Don't make things up, Just use the contexts and generate the relevant summary.
    2. Don't mix the numbers, Just use the numbers in the context.
    3. Don't try to use fancy words, stick to the basics of the language that is being used in the context.

    Context: {context}
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7
    )
    summary = response.choices[0].message.content.strip()
    return summary

def embed_clusters(
    texts: List[str],
    target_dim: int = 10,
    threshold: float = 0.1
) -> pd.DataFrame:
    textual_embeddings = np.array(embedding_model.encode(texts))
    clusters, number_of_clusters = clustering_algorithm(textual_embeddings, target_dim, threshold)
    print(f"Number of clusters: {number_of_clusters}")
    return pd.DataFrame({
        "texts": texts,
        "embedding": list(textual_embeddings),
        "clusters": clusters
    })

def embed_cluster_summaries(
    texts: List[str],
    level: int,
    target_dim: int = 10,
    threshold: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_clusters = embed_clusters(texts, target_dim, threshold)
    main_list = []
    
    for _, row in df_clusters.iterrows():
        for cluster in row["clusters"]:
            main_list.append({
                "text": row["texts"],
                "embedding": row["embedding"],
                "clusters": cluster
            })
    
    main_df = pd.DataFrame(main_list)
    unique_clusters = main_df["clusters"].unique()
    if len(unique_clusters) == 0:
        return df_clusters, pd.DataFrame(columns=["summaries", "level", "clusters"])

    print(f"--Generated {len(unique_clusters)} clusters--")

    summaries = []
    for cluster in unique_clusters:
        text_in_df = main_df[main_df["clusters"] == cluster]
        unique_texts = text_in_df["text"].tolist()
        text = "------\n------".join(unique_texts)
        summary = generate_summary(text)
        summaries.append(summary)

    df_summaries = pd.DataFrame({
        "summaries": summaries,
        "level": [level] * len(summaries),
        "clusters": unique_clusters
    })

    return df_clusters, df_summaries

def recursive_embedding_with_cluster_summarization(
    texts: List[str],
    number_of_levels: int = 3,
    level: int = 1,
    target_dim: int = 10,
    threshold: float = 0.1
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    if level > number_of_levels:
        return {}
    
    results = {}
    df_clusters, df_summaries = embed_cluster_summaries(texts, level, target_dim, threshold)
    results[level] = (df_clusters, df_summaries)
    
    if df_summaries.empty or len(df_summaries['clusters'].unique()) == 1:
        print(f"No more unique clusters found at level {level}. Stopping recursion.")
        return results
    
    if level < number_of_levels:
        next_level_texts = df_summaries['summaries'].tolist()
        next_level_results = recursive_embedding_with_cluster_summarization(
            next_level_texts, 
            number_of_levels, 
            level + 1,
            target_dim,
            threshold
        )
        results.update(next_level_results)
    
    return results
 
```

Okay, the code might seem a bit daunting at first glance, but don’t worry! Just give it a couple of looks, and it will start to make sense. Essentially, we’re just following the flow I mentioned earlier. 

```python
def process_text_hierarchy(
    texts: List[str], 
    number_of_levels: int = 3,
    target_dim: int = 10,
    threshold: float = 0.1
) -> Dict[str, pd.DataFrame]:
    hierarchy_results = recursive_embedding_with_cluster_summarization(
        texts, number_of_levels, target_dim=target_dim, threshold=threshold
    )
    
    processed_results = {}
    for level, (df_clusters, df_summaries) in hierarchy_results.items():
        if df_clusters.empty or df_summaries.empty:
            print(f"No data for level {level}. Skipping.")
            continue
        processed_results[f"level_{level}_clusters"] = df_clusters
        processed_results[f"level_{level}_summaries"] = df_summaries
    
    return processed_results

results = process_text_hierarchy(chunks, number_of_levels=3)
```

![image.png](../assets/use_cases/improve-raptor-with-rag/raptor-3.png)

### Inference

Now that we have our tree structure with leaf nodes at the bottom and summarized nodes in between, it’s time to query the RAG. There are two main methods for navigating the RAPTOR tree: Tree Traversal and Collapsed Tree Retrieval.

1. **Tree Traversal Retrieval:** This method systematically explores the tree, starting from the root node. It first selects the top-k most relevant root nodes based on their cosine similarity to the query embedding. Then, for each selected root node, its children are considered in the next layer, where the top-k nodes are again selected based on their cosine similarity to the query vector. This process repeats until we reach the leaf nodes. Finally, the text from all the selected nodes is concatenated to form the retrieved context.
2. **Collapsed Tree Retrieval:** This approach simplifies things by viewing the tree as a single layer. Here, it directly compares the query embedding to the vector embeddings of all the leaf nodes (the original text chunks) and summary nodes. This method works best for factual, keyword-based queries where you need specific details.

![Reference : https://arxiv.org/pdf/2401.18059](../assets/use_cases/improve-raptor-with-rag/raptor-4.png)

In the collapsed tree retrieval, we flatten the tree into one layer, retrieving nodes based on cosine similarity until we reach a specified number of ***top k documents***. In our code, we’ll gather the textual chunks from earlier, along with the summarized nodes at each level for all the clusters, to create one big list of texts that includes both the root documents and the summarized nodes.  

To be honest, if you look closely, we’ve been essentially adding more data points (chunks) to our RAG setup all along. Using RAPTOR, we now have both the original chunks and the summarized chunks for each cluster. Now, we’ll simply embed all these new data points and store them in a vector database along with their embeddings and use them for RAG.

```python
raptor_texts = []
for level, row in results.items():
    if level.endswith("clusters"):
        raptor_texts.extend(row["texts"])
    else:
        raptor_texts.extend(row["summaries"])
        
raptor_embeddings = embedding_model.encode(raptor_texts)
len(raptor_embeddings)
```

### Setting up Vector Database and RAG

Now it’s smooth sailing! We’ll just set up a LanceDB vector database to store our embeddings and query our RAG setup. 

```python
raptor_embeddings = embedding_model.encode(raptor_texts)

raptor_dict = {"texts": [], "embeddings": []}
for texts, embeddings in zip(raptor_texts, raptor_embeddings):
    raptor_dict["texts"].append(texts)
    raptor_dict["embeddings"].append(embeddings.tolist())
 
```

```python
import lancedb
import pyarrow as pa
from lancedb.pydantic import Vector, LanceModel

uri = "lancedb_database"
db = lancedb.connect(uri)

class RAG(LanceModel):
    texts : str
    embeddings : Vector(384)

table_name = "rag_with_raptor"
raptor_table = db.create_table(table_name, schema = RAG, mode="overwrite")
raptor_table.add(rag_raptor_df)
raptor_table.create_fts_index("texts", replace=True)
```

Time to generate the results..

```python
def generate_results(
    query : str,
    context_text : str
) -> str:

    prompt = f"""
    Based on the context provided, use it to answer the query. 

    query : {query}

    Instructions:
    1. Don't make things up, Just use the contexts and generate the relevant answer.
    2. Don't mix the numbers, Just use the numbers in the context.
    3. Don't try to use fancy words, stick to the basics of the language that is being used in the context.
    
    {context_text}
    """
    response = client.chat.completions.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers query and give the answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    return answer
```

```python
query = "NTT DATA's net income attributable to shareholders increased from ¥69,227 million in Q3 FY2020 to ¥110,191 million in Q3 FY2021. How does this growth align with their acquisition strategy, particularly considering their stated reasons for acquiring Nexient, LLC and the provisional goodwill recorded in this transaction?"
```

Now In our query, there are several key points that must be addressed when crafting the answers. First, we need to note the increase in net income from ¥69,227 million in Q3 FY2020 to ¥110,191 million in Q3 FY2021. Second, we should examine how this growth aligns with NTT DATA's acquisition strategy, particularly their reasons for acquiring Nexient, LLC, and the provisional goodwill recorded in the transaction. With this context in mind, I created a VANILLA RAG to compare its results with those of RAPTOR RAG.

```python
normal_embeddings = embedding_model.encode(chunks) # default chunks from our data

normal_dict = {"texts": [], "embeddings": []}
for texts, embeddings in zip(chunks, normal_embeddings):
    normal_dict["texts"].append(texts)
    normal_dict["embeddings"].append(embeddings.tolist())
    
rag_normal_df = pd.DataFrame(normal_dict)

table_name = "rag_without_raptor"
normal_table = db.create_table(table_name, schema = RAG, mode="overwrite")
normal_table.add(rag_normal_df)
normal_table.create_fts_index("texts", replace=True)
```

With RAPTOR, we now have an increased number of chunks due to the addition of cluster-level summary nodes alongside the default chunks we had earlier. 

![image.png](../assets/use_cases/improve-raptor-with-rag/raptor-5.png)

### D-Day

```python
raptor_contexts = raptor_table.search(query).limit(5).select(["texts"]).to_list()
raptor_context_text = "------\n\n".join([context["texts"] for context in raptor_contexts])
raptor_context_text = "------\n\n" + raptor_context_text

normal_contexts = normal_table.search(query).limit(5).select(["texts"]).to_list()
normal_context_text = "------\n\n".join([context["texts"] for context in normal_contexts])
normal_context_text = "------\n\n" + normal_context_text

raptor_answer = generate_results(query, raptor_context_text)
normal_answer = generate_results(query, normal_context_text)
```

![image.png](../assets/use_cases/improve-raptor-with-rag/raptor-6.png)

When we are comparing RAPTOR RAG with Vanilla RAG, it’s clear that RAPTOR performs better. Not only does RAPTOR retrieve details about the financial growth, but it also effectively connects this growth to the broader acquisition strategy, pulling relevant context from multiple sources. It excels in situations like this, where the query requires insights from various pages, making it more adept at handling complex, layered information retrieval.  

And that’s a wrap for this article! If you want to dig into the intricacies of how everything works, I’d suggest checking out the official RAPTOR [GitHub repository](https://github.com/parthsarthi03/raptor/tree/master) for more info and resources. For an even deeper dive, the official [paper](https://arxiv.org/pdf/2401.18059) is a great read and highly recommended!  Here is the Google [colab](https://colab.research.google.com/drive/1I3WI0U4sgb2nc1QTQm51kThZb2q4MXyr?usp=sharing) for your reference.