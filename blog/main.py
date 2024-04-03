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
existing_slugs_discovered = {}

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
    global existing_slugs_discovered
    base_url = urljoin(BASE_URL, 'api/blogs')
    search_url = base_url + f"?pagination[page]={page_num}"

    session = requests.Session()

    response = session.get(search_url, headers=headers)
    if response.status_code == 200:
        data = json.loads(response.text)['data']
        if len(data) > 0:
            for item in data:
                existing_slugs_discovered[item['attributes']['slug_url']] = {'discovered': False, 'id': -1}
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
    slug = blog.get_slug()
    search_url = base_url + f"?filters[slug_url][$eq]={slug}"
    session = requests.Session()

    if slug in existing_slugs_discovered:
        existing_slugs_discovered[slug]['discovered'] = True

    response = session.get(search_url, headers=headers)

    if response.status_code == 200:
        responses = json.loads(response.text)['data']
        print(f'Uploading slug: {blog.get_slug()}')
        if len(responses) > 0:
            # Blog already exists at this slug
            id = json.loads(response.text)['data'][0]['id']

            url = f"{base_url}/{id}"
            create_response = session.put(url, headers=headers, data=json.dumps(blog.get_post_json()))
        else:
            # Its a new blog
            url = base_url
            create_response = session.post(url, headers=headers, data=json.dumps(blog.get_post_json()))

        if create_response.status_code == 200:
            if slug in existing_slugs_discovered:
                create_response_text = json.loads(create_response.text)
                existing_slugs_discovered[slug]['id'] = create_response_text['data']['id']
        else:
            print(f'Error in parsing blog: {slug}')
            print(create_response.text)
            exit(1)

def delete_old_blogs():
    global existing_slugs_discovered

    base_url = urljoin(BASE_URL, 'api/blogs')
    session = requests.Session()

    for slug in existing_slugs_discovered:
        if not existing_slugs_discovered[slug]['discovered']:
            print(f"Deleting slug: {slug}")
            if existing_slugs_discovered[slug]['id'] > 0:
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
