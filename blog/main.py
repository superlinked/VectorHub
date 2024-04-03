import os
import json
import argparse
from json.decoder import JSONDecodeError
import requests
from urllib.parse import urljoin
from helpers import Item, ItemType, StrapiBlog
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path

args = None

BASE_URL = os.getenv('STRAPI_URL', "")
API_KEY = os.getenv('STRAPI_API_KEY', "")

paths_to_search = []
existing_filepaths_discovered = {}

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


def load_existing_blogs(page_num=1):
    global existing_filepaths_discovered
    base_url = urljoin(BASE_URL, 'api/blogs')
    search_url = base_url + f"?pagination[page]={page_num}"

    session = requests.Session()

    response = session.get(search_url, headers=headers)
    if response.status_code == 200:
        data = json.loads(response.text)['data']
        if len(data) > 0:
            for item in data:
                existing_filepaths_discovered[item['attributes']['filepath']] = {'discovered': False, 'id': -1}
            load_existing_blogs(page_num+1)


def fetch_paths(node: Item, current_path=""):
    global paths_to_search
    # Update the current path with dthe node's path
    current_path = f"{current_path}/{node.path}" if current_path else node.path

    # If the node has children, recurse on each child
    if node.has_blogs:
        paths_to_search.append(current_path)
    if node.children and len(node.children) > 0:
        for child in node.children:
            fetch_paths(child, current_path)


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
                files.append({
                    'path': str(file),
                    'time': datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d")
                })

    return files


def build_blog_object(file_obj: dict) -> StrapiBlog:
    filepath = file_obj['path']
    with open(filepath, 'r') as file:
        content = file.read()
        blog = StrapiBlog(content, filepath, file_obj['time'])
        return blog

def upload_blog(blog: StrapiBlog):
    base_url = urljoin(BASE_URL, 'api/blogs')
    filepath = blog.get_filepath()
    search_url = base_url + f"?filters[filepath][$eqi]={filepath}&publicationState=preview"
    session = requests.Session()

    if filepath in existing_filepaths_discovered:
        existing_filepaths_discovered[filepath]['discovered'] = True

    response = session.get(search_url, headers=headers)

    if response.status_code == 200:
        responses = json.loads(response.text)['data']
        print(f'Uploading filepath: {blog.get_filepath()}')
        if len(responses) > 0:
            # Blog already exists at this filepath
            id = json.loads(response.text)['data'][0]['id']

            blog.set_slug_url(json.loads(response.text)['data'][0]['attributes']['slug_url'])
            blog.set_published_at(json.loads(response.text)['data'][0]['attributes']['publishedAt'])

            url = f"{base_url}/{id}"
            create_response = session.put(url, headers=headers, data=json.dumps(blog.get_post_json()))
        else:
            # Its a new blog
            url = base_url
            create_response = session.post(url, headers=headers, data=json.dumps(blog.get_post_json()))

        if create_response.status_code == 200:
            if filepath in existing_filepaths_discovered:
                create_response_text = json.loads(create_response.text)
                existing_filepaths_discovered[filepath]['id'] = create_response_text['data']['id']
        else:
            print(f'Error in parsing blog: {filepath}')
            print(create_response.text)
            exit(1)

def delete_old_blogs():
    global existing_filepaths_discovered

    base_url = urljoin(BASE_URL, 'api/blogs')
    session = requests.Session()

    for filepath in existing_filepaths_discovered:
        if not existing_filepaths_discovered[filepath]['discovered']:
            print(f"Deleting filepath: {filepath}")
            if existing_filepaths_discovered[filepath]['id'] > 0:
                url = f"{base_url}/{id}"
                response = session.delete(url, headers=headers)


if __name__ == "__main__":
    arg_parse()
    items = load_items_from_json(args.directories)

    load_existing_blogs()

    files = find_files_to_upload(items)

    print('Uploading blogs')
    for file in tqdm(files):
        blog = build_blog_object(file)
        upload_blog(blog)

    print('Deleting blogs')
    delete_old_blogs()
