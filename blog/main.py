import os
import json
import argparse
from json.decoder import JSONDecodeError
import requests
from urllib.parse import urljoin
from helpers import Item, ItemType, StrapiBlog, StrapiBlogType
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path

args = None

BASE_URL = os.getenv('STRAPI_URL', "")
API_KEY = os.getenv('STRAPI_API_KEY', "")

paths_to_search = []

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

def arg_parse():
    global args
    parser = argparse.ArgumentParser(description="VectorHub Strapi Upload")
    parser.add_argument('--directories', help='Path to json which describes the directories to parse')
    args = parser.parse_args()

def load_items_from_json(directories: str) -> list:
    if os.path.exists(directories):
        items = []
        try:
            with open(directories, 'r') as file:
                data = json.load(file)
                for item_data in data:
                    items.append(Item.from_dict(item_data))
        except JSONDecodeError as e:
            print('JSON Structure is invalid.')
            exit(1)
        except Exception as e:
            print('Unknown error occured.')
            print(e)
            exit(1)
        return items
    else:
        print(f"{directories} does not exist.")
        exit(1)

def fetch_paths(node: Item, current_path=""):
    global paths_to_search
    # Update the current path with the node's path
    current_path = f"{current_path}/{node.path}" if current_path else node.path

    # If the node has children, recurse on each child
    if node.has_blogs:
        paths_to_search.append(current_path)
    if node.children and len(node.children) > 0:
        for child in node.children:
            fetch_paths(child, current_path)
    # else:
        # If there are no children, print the current path
        # print(current_path)

def find_files_to_upload(items: list):
    global paths_to_search

    for item in items:
        fetch_paths(item)

    files = []

    extension = 'md'
    
    for path in paths_to_search:
        folder_path = Path(path)
        folder_files = folder_path.glob(f"*.{extension}")
        for file in folder_files:
            if 'readme.md' not in str(file).lower():
                files.append(str(file))

    return files


def build_blog_object(filepath: str) -> StrapiBlog:
    with open(filepath, 'r') as file:
        content = file.read()
        blog = StrapiBlog(content, filepath, datetime.now().strftime("%Y-%m-%d"), StrapiBlogType.USECASE)
        print(blog.get_slug())
        return blog

def upload_blog(blog: StrapiBlog):
    base_url = urljoin(BASE_URL, 'api/blogs')
    search_url = base_url + f"?filters[slug_url][$eq]={blog.get_slug()}"
    session = requests.Session()

    response = session.get(search_url, headers=headers)

    if response.status_code == 200:
        responses = json.loads(response.text)['data']

        if len(responses) > 0:
            # Blog already exists at this slug
            id = json.loads(response.text)['data'][0]['id']

            url = f"{base_url}/{id}"
            create_response = session.put(url, headers=headers, data=json.dumps(blog.get_post_json()))
        else:
            # Its a new blog
            url = base_url
            create_response = session.post(url, headers=headers, data=json.dumps(blog.get_post_json()))


if __name__ == "__main__":
    arg_parse()
    items = load_items_from_json(args.directories)

    files = find_files_to_upload(items)

    for file in tqdm(files):
        blog = build_blog_object(file)
        upload_blog(blog)
