import os
import json
from json.decoder import JSONDecodeError
import requests
from helpers import Item, ItemType, Items

# load files.json into items
items = Items()
try:
    items.load_from_file("files.json")
except JSONDecodeError as e:
    print('JSON Structure is invalid.')
    exit(1)
except Exception as e:
    print('Unknown error occured.')
    print(e)
    exit(1)

BASE_URL = os.getenv('STRAPI_URL', "")
API_KEY = os.getenv('STRAPI_API_KEY', "")

session = requests.Session()

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Example blog post data
blog_post_data = {
    "data": {
        "github_url": "http://facebook.com",
        "content": "This is a sample blog post created via API call."
    }
}

url = f"{BASE_URL}/api/blogs/4"

# Send POST request to create a blog post
create_response = session.put(
    url, headers=headers, data=json.dumps(blog_post_data))

# Check if the request was successful
if create_response.status_code == 200:
    print("Blog post created successfully")
else:
    print("Failed to create blog post:", create_response.text)
