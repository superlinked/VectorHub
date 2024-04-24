# Build a Vector-Search based Recommender System

## Overview

In this guide, you will:
- Learn how to use a reliable and performant database optimized for vectors (Rockset) along with OpenAI embeddings to build a recommendation engine
- Build a dynamic web application using vanilla CSS, HTML, JavaScript, and Flask, that seamlessly integrates with Rockset and OpenAI APIs to create a recommendation system.
- Find an end-to-end Colab notebook that you can run without any dependencies on your local operating system: [Recsys_workshop](https://colab.research.google.com/drive/1rD08fTiCGJiKFDDNZyJh92QN4_dz6Hcu)

## Introduction
A real-time recommender system with an optimized and efficient [architecture](https://rockset.com/blog/a-blueprint-for-a-real-world-recommendation-system/) can add value to an organization by enhancing personalization, user engagement, and ultimately increasing user satisfaction.

Building a recommendation system that efficiently deals with high-dimensional data to find accurate, relevant, and similar items in a large dataset requires effective and efficient vectorization, vector indexing, search, and retrieval. This, in turn, demands robust databases with vector capabilities. For this post, we will use Rockset as the database and OpenAI embedding models to vectorize the dataset.  Other vector databases are available, you can find examples [here](https://vdbs.superlinked.com/).

There are several components of vector search required to build a recommendation engine, we won't dive into them here however you can read more about them [here](https://superlinked.com/vectorhub/building-blocks/vector-search/introduction). The key pieces for this example are a database optimized for metadata filtering, vector search, and keyword search, supporting sub-second search, aggregations, and joins at scale, is needed.

**Overview of the Recommendation WebApp**
The image below shows the workflow of the application we'll be building. We have unstructured data i.e., game reviews in our case. We'll generate vector embeddings for all of these reviews through OpenAI model and store them in the database. Then we'll use the same OpenAI model to generate vector embeddings for our search query and match it with the review vector embeddings using a similarity function such as the nearest neighbor search, dot product or approximate neighbor search. Finally, we will have our top 10 recommendations ready to be displayed.

![overview](../assets/use_cases/vector_search_based_recommender_system/image3.png)

## Steps to build the Recommender System using Rockset and OpenAI Embedding

Let's begin with signing up for Rockset and OpenAI and then dive into all the steps involved within the Google Colab notebook to build our recommendation webapp:

### Step 1: Initiate your vector database

In this example we are using [Rockset](https://rockset.com/create/) as a vector database. Once signed up, create an [API key](https://console.rockset.com/apikeys) to use in the backend code:

```python
import os
os.environ["ROCKSET_API_KEY"] = "XveaN8L9mUFgaOkffpv6tX6VSPHz####"
```

### Step 2: Create a new Collection and Upload Data

For this tutorial, we'll be using Amazon product review [data](https://drive.google.com/file/d/1EEvCUqKIH6LuLLGRjo6ayCEXqT4Z-ZVE/view?usp=drive_link). Download this on your local machine so it can be uploaded to your collection.

We'll be uploading the [sample_data.json](https://drive.google.com/file/d/1EEvCUqKIH6LuLLGRjo6ayCEXqT4Z-ZVE/view?usp=drive_link) file to Rockset.

Note: In practice, the data is usually ingested from another streaming service but for keeping things simple and building a demo application we are using a sample from a public dataset.

### Step 3: Create OpenAI API Key

To convert data into [embeddings](https://platform.openai.com/docs/api-reference/embeddings/create), we'll use OpenAI's model.

After signing up, go to [API Keys](https://platform.openai.com/api-keys) and create a secret key. Don't forget to copy and save your key that will look similar to "sk-***********************". Like Rockset's API key, save your OpenAI key in the environment so it can easily used throughout the code:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-####"
```
### Step 4: Create a Query Lambda on Rockset

[Query Lambdas](https://docs.rockset.com/documentation/docs/query-lambdas) are named, parameterized SQL queries stored in Rockset that can be executed from a dedicated REST endpoint. Using Query Lambdas, you can save your SQL queries as separate resources and manage them successfully through development and production.

Let's create one for our tutorial. We'll be using the following Query Lambda with parameters: embedding, brand, min_price, max_price and limit.

```sql
SELECT
  asin,
  title,
  brand,
  description,
  estimated_price,
  brand_tokens,
  image_ur1,
  APPROX_DOT_PRODUCT(embedding, VECTOR_ENFORCE(:embedding, 1536, 'float')) as similarity
FROM
    commons.sample s
WHERE estimated_price between :min_price AND :max_price
AND ARRAY_CONTAINS(brand_tokens, LOWER(:brand))
ORDER BY similarity DESC
LIMIT :limit;
```
This parameterized query does the following:
- It retrieves data from the "sample" table in the "commons" schema. And selects specific columns like ASIN, title, brand, description, estimated_price, brand_tokens, and image_ur1.
- It also computes the similarity between the provided embedding and the embedding stored in the database using the **APPROX_DOT_PRODUCT** function. 
- The query filters results based on the estimated_price falling within the provided range, the brand containing the specified value, and then sorts the results based on similarity in descending order.
- Finally, it limits the number of returned rows based on the provided limit parameter.


To build this Query Lambda, query the collection made in step 2 by clicking on **Query this collection** and pasting the parameterized query above into the Rockset query editor.

### Frontend Overview

The final step to create a web application includes implementing a frontend design using vanilla HTML,CSS and a bit of JavaScript along with backend implementation using Flask, a lightweight Pythonic web framework.

The frontend [page](https://github.com/ankit1khare/rockset-vector-search/blob/main/templates/index.html) looks as shown below:

![frontend](../assets/use_cases/vector_search_based_recommender_system/image1.png)

Let's break down the HTML file to understand its structure and components. The code provided consists of the following components:

1.  **HTML Structure:**
    -  The basic structure of the webpage includes a sidebar, header, and product grid container.
2.  **Sidebar:**
    -  The sidebar contains search filters such as brands, min and max price, etc., and buttons for user interaction.

3.  **Product Grid Container:**
    -  The container populates product cards dynamically using JavaScript to display product information i.e. image, title, description, and price.

4.  **JavaScript Functionality:**
    -  It is needed to handle interactions such as toggling full descriptions, populating the recommendations, and clearing search form inputs.

5.  **CSS Styling:**
    -  Implemented for responsive design to ensure optimal viewing on various devices and improve aesthetics.

Check out the full code behind this front-end [here](https://github.com/ankit1khare/rockset-vector-search/blob/main/templates/index.html).

### Backend Overview

Flask makes creating web applications in Python easier by rendering the HTML and CSS files via single-line commands. You can check the backend code that Flask utilizes [here](https://github.com/ankit1khare/rockset-vector-search/blob/main/app.py).

Initially, the Get method will be called and the HTML file will be rendered. As there will be no recommendation at this time, the basic structure of the page will be displayed on the browser. After this is executed, we can fill the form and submit it thereby utilizing the POST method to get some recommendations.

Let's dive into the main components of the code as we did for the front-end:

1.  **Flask App Setup:**
    1.  A Flask application named app is defined along with a route for both GET and POST requests at the root URL ("/").

2.  **Index function:**
    1.  Function built to primarily handle both GET and POST requests.
    2.  If it's a POST request:
        1.  Extracts form data from the frontend.
        2.  Calls a set of functions to process the input data and fetch recommended results from Rockset database.
        3.  Fetches recommended product images by searching through all the product images saved in this [directory](https://github.com/ankit1khare/rockset-vector-search/tree/main/static).
        4.  Renders the index.html template with the results.
    3.  If it's a GET request:
        1.  Renders the index.html template with the search form.
```python

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract data from form fields
        inputs = get_inputs()

        search_query_embedding = get_openai_embedding(inputs, client)
        rockset_key = os.environ.get('ROCKSET_API_KEY')
        region = Regions.usw2a1
        records_list = get_rs_results(inputs, region, rockset_key, search_query_embedding)

        folder_path = 'static'
        for record in records_list:
            # Extract the identifier from the URL
            identifier = record["image_url"].split('/')[-1].split('_')[0]
            file_found = None
            for file in os.listdir(folder_path):
                if file.startswith(identifier):
                    file_found = file
                    break
            if file_found:
                # Overwrite the record["image_url"] with the path to the local file
                record["image_url"] = file_found
                record["description"] = json.dumps(record["description"])
                # print(f"Matched file: {file_found}")
            else:
                print("No matching file found.")

        # Render index.html with results
        return render_template('index.html', records_list=records_list, request=request)

    # If method is GET, just render the form
    return render_template('index.html', request=request)
```

3. **Data Processing Functions:**
    1. get_inputs(): Extracts form data from the request.

    ```python
    def get_inputs():
        search_query = request.form.get('search_query')
        min_price = request.form.get('min_price')
        max_price = request.form.get('max_price')
        brand = request.form.get('brand')
        limit = request.form.get('limit')

        return {
            "search_query": search_query,
            "min_price": min_price,
            "max_price": max_price,
            "brand": brand,
            "limit": limit
        }
    ```
    2. get_openai_embedding(): Uses OpenAI to get embeddings for search queries.
    ```python

    def get_openai_embedding(inputs, client):
        # openai.organization = org
        # openai.api_key = api_key

        openai_start = (datetime.now())
        response = client.embeddings.create(
            input=inputs["search_query"],
            model="text-embedding-ada-002"
            )
        search_query_embedding = response.data[0].embedding
        openai_end = (datetime.now())
        elapsed_time = openai_end - openai_start

        return search_query_embedding

    ```
    3. get_rs_results(): Utilizes Query Lambda created earlier in Rockset and returns recommendations based on user inputs and embeddings.
    ```python

    def get_rs_results(inputs, region, rockset_key, search_query_embedding):
        print("\nRunning Rockset Queries...")
        # Create an instance of the Rockset client
        rs = RocksetClient(api_key=rockset_key, host=region)

        rockset_start = (datetime.now())
        # Execute Query Lambda By Version
        rockset_start = (datetime.now())
        api_response = rs.QueryLambdas.execute_query_lambda_by_tag(
            workspace="commons",
            query_lambda="recommend_games",
            tag="latest",
            parameters=[
                {
                    "name": "embedding",
                    "type": "array",
                    "value": str(search_query_embedding)
                },
                {
                    "name": "min_price",
                    "type": "int",
                    "value": inputs["min_price"]
                },
                {
                    "name": "max_price",
                    "type": "int",
                    "value": inputs["max_price"]
                },
                {
                    "name": "brand",
                    "type": "string",
                    "value": inputs["brand"]
                }
                {
                     "name": "limit",
                     "type": "int",
                     "value": inputs["limit"]
                }
            ]
        )
        rockset_end = (datetime.now())
        elapsed_time = rockset_end - rockset_start

        records_list = []
        for record in api_response["results"]:
            record_data = {
                "title": record['title'],
                "image_url": record['image_ur1'],
                "brand": record['brand'],
                "estimated_price": record['estimated_price'],
                "description": record['description']
            }
            records_list.append(record_data)

        return records_list
    ```

Overall, the Flask backend processes user input and interacts with external services (OpenAI and Rockset) to provide dynamic content to the frontend. It extracts form data from the frontend, generates OpenAI embeddings for text queries, and utilizes Query Lambda at Rockset to find recommendations.


Now, you are ready to run the flask server and access it via your internet browser. Our application is up and running. Let's add some parameters in the bar and get some recommendations. The results will be displayed on an HTML template as shown below.

![frontend](../assets/use_cases/vector_search_based_recommender_system/image2.png)

**Note: The tutorial's entire code is available on** [**GitHub**](https://github.com/ankit1khare/rockset-vector-search/tree/main)**. For a quick-start online implementation, a end-to-end runnable** [**Colab notebook**](https://colab.research.google.com/drive/1WcJggQWYayVIQpKFQVZ80H74x0tky8Pa?usp=sharing#scrollTo=XzggkmXJ_Bly) **is also configured.** 

The methodology outlined in this tutorial can serve as a foundation for various other applications beyond recommendation systems. By leveraging the same set of concepts and using embedding models and a vector database, you are now equipped to build applications such as semantic search engines, customer support chatbots, and real-time data analytics dashboards.

