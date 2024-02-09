import json
import os

import requests

# Constants for the script
DIRECTORY = "docs/tools/vdb_table/data"
GITHUB_API_URL = "https://api.github.com/repos/"


def get_github_stars(github_url, headers):
    # Extract the owner and repo name from the GitHub URL
    global GITHUB_API_URL
    parts = github_url.split("/")
    owner_repo = "/".join(parts[-2:])
    response = requests.get(f"{GITHUB_API_URL}{owner_repo}", headers=headers)
    if response.status_code == 200:
        return response.json()["stargazers_count"]
    else:
        print(f"Failed to fetch stars for {github_url}: {response.status_code}")
        return None


def update_json_files(directory, headers=None):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r+", encoding="utf-8") as json_file:
                data = json.load(json_file)
                github_url = data.get("github_stars", {}).get("source_url", "")
                if github_url:
                    stars = get_github_stars(github_url, headers)
                    if stars is not None:
                        data["github_stars"]["value"] = stars
                        # Write the updated data back to the file
                        json_file.seek(0)  # Rewind to the start of the file
                        json.dump(data, json_file, indent=2)
                        json_file.truncate()  # Remove any leftover content


if __name__ == "__main__":
    update_json_files(DIRECTORY)
