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

BASE_URL = os.getenv("STRAPI_URL", "")
API_KEY = os.getenv("STRAPI_API_KEY", "")

paths_to_search = []
existing_filepaths_discovered = {}

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def arg_parse():
    global args
    parser = argparse.ArgumentParser(description="VectorHub Strapi Upload")
    parser.add_argument(
        "--directories", help="Path to json which describes the directories to parse"
    )
    args = parser.parse_args()


def load_items_from_json(directories: str) -> list:
    if os.path.exists(directories):
        try:
            with open(directories, "r") as file:
                data = json.load(file)
                return [Item.from_dict(item_data) for item_data in data]
        except JSONDecodeError:
            print("❌ Invalid JSON structure.")
            exit(1)
        except Exception as e:
            print("❌ Unknown error while reading directory JSON:")
            print(e)
            exit(1)
    else:
        print(f"{directories} does not exist.")
        exit(1)


def load_existing_blogs(page_num=1):
    """Loads all blogs currently in Strapi."""
    global existing_filepaths_discovered

    base_url = urljoin(BASE_URL, "api/blogs")
    search_url = f"{base_url}?pagination[page]={page_num}&pagination[pageSize]=100"

    session = requests.Session()
    response = session.get(search_url, headers=headers)

    if response.status_code == 200:
        data = response.json().get("data", [])
        if not data:
            return
        for item in data:
            filepath = item.get("filepath")
            if filepath:
                existing_filepaths_discovered[filepath] = {
                    "discovered": False,
                    "id": item["id"],
                }
        load_existing_blogs(page_num + 1)
    else:
        print(f"⚠️ Failed to load blogs: {response.status_code} {response.text}")


def fetch_paths(node: Item, current_path=""):
    """Recursively collect directories containing blogs."""
    global paths_to_search

    current_path = f"{current_path}/{node.path}" if current_path else node.path

    if node.has_blogs:
        paths_to_search.append(current_path)
    if node.children:
        for child in node.children:
            fetch_paths(child, current_path)


def find_files_to_upload(items: list):
    global paths_to_search

    for item in items:
        fetch_paths(item)

    files = []
    extension = "md"

    for path in paths_to_search:
        folder_path = Path(path)
        for file in folder_path.glob(f"*.{extension}"):
            if "readme.md" not in str(file).lower():
                files.append(
                    {
                        "path": str(file),
                        "time": datetime.fromtimestamp(os.path.getmtime(file)).strftime(
                            "%Y-%m-%d"
                        ),
                    }
                )
    return files


def build_blog_object(file_obj: dict) -> StrapiBlog:
    filepath = file_obj["path"]
    with open(filepath, "r") as file:
        content = file.read()
    return StrapiBlog(content, filepath, file_obj["time"])


def upload_blog(blog: StrapiBlog):
    """Uploads or updates a blog to Strapi v5."""
    base_url = urljoin(BASE_URL, "api/blogs")
    filepath = blog.get_filepath()
    search_url = f"{base_url}?filters[filepath][$eqi]={filepath}"

    session = requests.Session()

    if filepath in existing_filepaths_discovered:
        existing_filepaths_discovered[filepath]["discovered"] = True

    response = session.get(search_url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Error fetching blog {filepath}: {response.text}")
        return

    existing = response.json().get("data", [])
    print(f"📤 Uploading filepath: {filepath}")

    if existing:
        # Blog already exists
        blog_id = existing[0]["documentId"]
        blog.set_slug_url(existing[0].get("slug_url"))
        blog.set_published_at(existing[0].get("publishedAt"))
        meta_desc = existing[0].get("meta_desc")
        if meta_desc:
            blog.meta_desc = meta_desc
        else:
            blog.meta_desc = blog.title

        url = f"{base_url}/{blog_id}"
        create_response = session.put(
            url, headers=headers, data=json.dumps(blog.get_post_json())
        )
    else:
        # New blog
        blog.meta_desc = blog.title
        create_response = session.post(
            base_url, headers=headers, data=json.dumps(blog.get_post_json())
        )

    if create_response.status_code not in (200, 201):
        print(f"❌ Failed to upload blog: {filepath}", create_response.text)
        exit(1)


def delete_old_blogs():
    """Deletes blogs that were not re-uploaded."""
    global existing_filepaths_discovered

    base_url = urljoin(BASE_URL, "api/blogs")
    session = requests.Session()

    for filepath, info in existing_filepaths_discovered.items():
        if not info["discovered"]:
            print(f"🗑️ Deleting filepath: {filepath}")
            blog_id = info["id"]
            if blog_id:
                url = f"{base_url}/{blog_id}"
                response = session.delete(url, headers=headers)
                if response.status_code not in (200, 204):
                    print(f"⚠️ Error deleting {filepath}: {response.text}")


if __name__ == "__main__":
    arg_parse()
    items = load_items_from_json(args.directories)

    load_existing_blogs()

    files = find_files_to_upload(items)

    print("📦 Uploading blogs...")
    for file in tqdm(files):
        blog = build_blog_object(file)
        upload_blog(blog)

    print("🧹 Cleaning up deleted blogs...")
    delete_old_blogs()
