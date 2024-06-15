# Movie Recommendation using vector database

This article presents a comprehensive guide on creating a movie recommendation system using vector similarity search and multi-label genre classification. By leveraging a vector database, we can streamline the whole process. I'll demonstrate how integrating these two techniques can significantly improve the recommendation experience. 

Here's what we cover below:

1. Data ingestion and preprocessing techniques for movie metadata
2. Training a Doc2Vec model for generating the embeddings
3. Training a Neural network for genre classification task
4. Using Doc2Vec, LanceDB vector database and the trained classifier to get the relevant recommendations

Let's get started!

## Why use embeddings for recommendation systems?

Scrolling through streaming platforms can be frustrating when the movie suggestions don't match our interests. Building recommendation systems is a complex task, as there isn't one metric that can measure the quality of recommendations. To improve this, we can combine embeddings and vector database for better recommendations. These embeddings serve dual purposes: they can either be directly used as input to a classification model for genre classification or stored in a vector database for retrieval purposes. By storing embeddings in a vector database, efficient retrieval and query search for recommendations become possible at a later stage.

This architecture offers a holistic understanding of the underlying processes involved.

![image]()

## Data Ingestion and preprocessing techniques for movie metadata

Our initial task involves gathering and organizing information about movies. This includes gathering extensive details such as the movie's type, plot summary, genres, audience ratings, and more. Thankfully, we have access to a robust dataset on [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) containing information from various sources for approximately 45,000 movies. Please dowload the data from the Kaggle and place it inside your working directory. Watch for a file named `movies_metadata.csv`. 

If you require additional data, you can supplement the dataset by extracting information from platforms like Rotten Tomatoes, IMDb, or even box-office records. Our next step involves extracting core details from the dataset and generating a universal summary for each movie. We will begin by combining the movie's title and genre into a single textual string. This text will then be tagged to create `TaggedDocument` instances, which will later be used to train the Doc2Vec model.

Before moving forward, let's install the relevant libraries to make our life easier..

```python
pip install torch scikit-learn pylance lancedb nltk gensim lancedb scipy==1.12
```

Next, we'll proceed with the ingestion and preprocessing of the data. To simplify the process, we'll work with chunks of 1000 movies at a time. For clarity, I'll only include movie indices with non-null values for genres and titles. This approach ensures that we're working with high-quality, relevant data for our analysis.

```python
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import nltk
nltk.download('punkt') 

# Read data from CSV file
movie_data = pd.read_csv('movies_metadata.csv', low_memory=False)
movie_data.dropna(subset=['genres', 'title'], inplace=True)  # Ensure no missing genres or titles
modified_movie_data = movie_data[movie_data['genres'].apply(lambda x: len(eval(x)) > 0 if isinstance(x, str) else False)]
modified_movie_data = modified_movie_data.drop_duplicates(subset='title')
print(f"Modified movie data shape : {modified_movie_data.shape}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_data(movie_data_chunk):
    movie_info = []
    data = []

    for i in range(len(movie_data_chunk)):
        title = movie_data_chunk.iloc[i]['title']
        genres = movie_data_chunk.iloc[i]['genres']
        
        try:
            movies_text = ''
            genres_list = eval(genres)
            genre_names = ', '.join([genre['name'] for genre in genres_list])
            movies_text += "Genres: " + genre_names + '\n'
            movies_text += "Title: " + title + '\n'
            data.append(movies_text)
            movie_info.append((title, genre_names))
        
        except Exception as e:
            continue

    return data, movie_info

# Preprocess data and extract genres for the first 1000 moviesbat
batch_size = 1000
movie_info = []
complete_data = []
for chunk_start in tqdm(range(0, len(modified_movie_data), batch_size), desc="Processing chunks..."):
    movie_data_chunk = modified_movie_data.iloc[chunk_start:chunk_start+batch_size]
    chunk_movie_data, chunk_movie_info = preprocess_data(movie_data_chunk)
    movie_info.extend(chunk_movie_info)
    complete_data.extend(chunk_movie_data)

# preproces the documents, and create TaggedDocuments
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                              tags=[str(i)]) for i,
               doc in enumerate(complete_data)]
```

## Generating embeddings using Doc2Vec

Next, we'll utilize the Doc2Vec model to generate embeddings for each movie based on the preprocessed text. We'll allow the Doc2Vec model to train for several epochs to capture the essence of the various movies and their metadata in the multidimensional latent space. This process will help us represent each movie in a way that captures its unique characteristics and context.

```python
def train_doc2vec_model(tagged_data, num_epochs=10):
    doc2vec_model = Doc2Vec(vector_size=100, min_count=2, epochs=num_epochs)
    doc2vec_model.build_vocab(tqdm(tagged_data, desc="Building Vocabulary"))
    for epoch in range(num_epochs):
        doc2vec_model.train(tqdm(tagged_data, desc=f"Epoch {epoch+1}"), total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    
    return doc2vec_model

doc2vec_model = train_doc2vec_model(tagged_data)
doc2vec_model.save("doc2vec_model")
document_vectors = [doc2vec_model.infer_vector(
    word_tokenize(doc.lower())) for doc in complete_data]
```

The `train_doc2vec_model` function trains a Doc2Vec model on the tagged movie data, producing 100-dimensional embeddings for each movie. These embeddings act as input features for the neural network. With our current training setup, we are sure that movies with similar genres will be positioned closer to each other in the latent space, reflecting their thematic and content similarities.

## Extracting the unique genre labels

Now, our focus shifts to compiling the names of relevant movies along with their genres. For doing that, we'll employ a tool called `MultiLabelBinarizer` to transform these genres into a more comprehensible format.

To illustrate this, let's consider a movie with three genres: 'drama', 'comedy', and 'horror'. Using the `MultiLabelBinarizer`, we'll represent these genres with lists of 0s and 1s. If a movie belongs to a particular genre, it will be assigned a 1, and if it doesn't, it will receive a 0. Consequently, each row in our dataset will indicate which genres are associated with a specific movie. This approach simplifies the genre representation for easier analysis.

Let's take the movie "Top Gun Maverick" as a reference. We'll associate its genres using binary encoding. Suppose this movie is categorized only under 'drama', not 'comedy' or 'horror'. When we apply the `MultiLabelBinarizer` the representation would be: Drama: 1, Comedy: 0, Horror: 0. This signifies that "Top Gun Maverick" is classified as a drama but not as a comedy or horror movie. We'll replicate this process for all the movies in our dataset to identify the unique genre labels present in our data.

## Training a NN for genre_classification task

We'll define a neural network consisting of four linear layers with ReLU activations. The final layer utilizes softmax activation to generate probability scores for various genres. If your objective is primarily classification within the genre spectrum, where you input a movie description to determine its relevant genres, you can establish a threshold value for the multi-label softmax output. This allows you to select the top 'n' genres with the highest probabilities.

Here's the neural network class, hyperparameter settings, and the corresponding training loop for training our model.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

genres_list = []
for i, row in modified_movie_data.iterrows():
    genres = [genre['name'] for genre in eval(row['genres'])]
    genres_list.append(genres)

mlb = MultiLabelBinarizer()
genre_labels = mlb.fit_transform(genres_list)

embeddings = []
for i, doc in enumerate(complete_data):
    embeddings.append(document_vectors[i])

X_train, X_test, y_train, y_test = train_test_split(embeddings, genre_labels, test_size=0.2, random_state=42)

X_train_np = np.array(X_train, dtype=np.float32)
y_train_np = np.array(y_train, dtype=np.float32)
X_test_np = np.array(X_test, dtype=np.float32)
y_test_np = np.array(y_test, dtype=np.float32)

X_train_tensor = torch.tensor(X_train_np)
y_train_tensor = torch.tensor(y_train_np)
X_test_tensor = torch.tensor(X_test_np)
y_test_tensor = torch.tensor(y_test_np)

class GenreClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(GenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # Adjust the dropout rate as needed

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Move model to the selected device
model = GenreClassifier(input_size=100, output_size=len(mlb.classes_)).to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
batch_size = 64

train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
```

That's it! We've successfully trained a Neural network for our genre classification task. You can check if the model is performing well or not in terms of the genre classification.

```python
from sklearn.metrics import f1_score

model.eval()
with torch.no_grad():
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)  # Move test data to device
    outputs = model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')


thresholds = [0.1] * len(mlb.classes_)
thresholds_tensor = torch.tensor(thresholds, device=device).unsqueeze(0)

# Convert the outputs to binary predictions using varying thresholds
predicted_labels = (outputs > thresholds_tensor).cpu().numpy()

# Convert binary predictions and actual labels to multi-label format
predicted_multilabels = mlb.inverse_transform(predicted_labels)
actual_multilabels = mlb.inverse_transform(y_test_np)

# Print the Predicted and Actual Labels for each movie
for i, (predicted, actual) in enumerate(zip(predicted_multilabels, actual_multilabels)):
    print(f'Movie {i+1}:')
    print(f'    Predicted Labels: {predicted}')
    print(f'    Actual Labels: {actual}')


# Compute F1-score
f1 = f1_score(y_test_np, predicted_labels, average='micro')
print(f'F1-score: {f1:.4f}')

# Saving the trained model
torch.save(model.state_dict(), 'trained_model.pth')
```

## A movie recommendation system

To create our movie recommendation system, we'll implement a process where users input a movie title, and we generate relevant recommendations based on that input. Key to this approach is utilizing Doc2Vec embeddings, which we'll store in a vector database for efficient retrieval.

Here's how it works: When a user enters a movie title as a query, we'll first retrieve its embeddings from our vector database. With these embeddings in hand, our next step involves identifying 'n' number of movies whose embeddings closely resemble those of our query movie. We can achieve this by employing similarity metrics such as cosine similarity or by determining the smallest Euclidean distances. By leveraging these techniques, our system will provide users with personalized movie recommendations that are closely aligned with their initial movie interests. This approach ensures an intuitive and effective recommendation process tailored to individual preferences based on the genres.

For simplification, I've opted to use [LanceDB](https://lancedb.com/), an open-source vector database known for its blazing speed, high-level security (since our data remains local), versioning, and built-in search capabilities.

```python
import lancedb
import numpy as np
import pandas as pd

data = []

for i in range(len(movie_info)):  # Iterate over movie_info, not modified_movie_data
    embedding = document_vectors[i]
    title, genres = movie_info[i]  # Correctly access title and genres from movie_info
    data.append({"title": title, "genres": genres, "vector": embedding})

db = lancedb.connect(".db")
tbl = db.create_table("doc2vec_embeddings", data, mode="Overwrite")
db["doc2vec_embeddings"].head()
```

While this code may initially appear complex, it's essentially structured to handle each row in the table as a distinct movie entry. Within each row, you'll find columns dedicated to essential movie details such as title, genres, and embeddings.

## Using Doc2Vec Embeddings to get the relevant recommendations.

Our recommendation engine combines a neural network-based genre prediction model with a vector similarity search to provide relevant movie recommendations.

For a given query movie, first, we use our trained neural network to predict its genres. Based on these predicted genres, we filter our movie database to include only those movies that share at least one genre with the query movie, achieved by constructing an appropriate SQL filter. We then perform a vector similarity search on this filtered subset to retrieve the most similar movies based on their vector representations. This approach ensures that the recommended movies are not only similar in terms of their vector characteristics but also share genre preferences with the query movie, resulting in more relevant and personalized recommendations.

```python
# Function to get genres for a single movie query
def get_genres_for_query(model, query_embedding, mlb, thresholds, device):
    model.eval()
    with torch.no_grad():
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(query_tensor)
        thresholds = [0.001] * len(mlb.classes_)
        thresold_tensor = torch.tensor(thresholds, device=device).unsqueeze(0)
        predicted_labels = (outputs >= thresold_tensor).cpu().numpy()
        predicted_multilabels = mlb.inverse_transform(predicted_labels)
        return predicted_multilabels


def movie_genre_prediction(movie_title):
    movie_index = modified_movie_data.index[modified_movie_data['title'] == movie_title].tolist()[0]
    query_embedding = document_vectors[movie_index]
    predicted_genres = get_genres_for_query(model, query_embedding, mlb, [0.1] * len(mlb.classes_), device=device)
    return predicted_genres
```

And now, after all the groundwork, we've arrived at the final piece of the puzzle. Let's generate some relevant recommendations using embeddings and LanceDB.
 
```python
def get_recommendations(title):
    pd_data = pd.DataFrame(data)
    title_vector = pd_data[pd_data["title"] == title]["vector"].values[0]
    predicted_genres = movie_genre_prediction(title)
    genres_movie = predicted_genres[0]  # Assuming predicted_genres is available

    genre_conditions = [f"genres LIKE '%{genre}%'" for genre in genres_movie]
    where_clause = " OR ".join(genre_conditions)

    result = (
        tbl.search(title_vector)
        .metric("cosine")
        .limit(10)
        .where(where_clause)
        .to_pandas()
    )
    return result[["title"]]

get_recommendations("Toy Story")
```

![results]()