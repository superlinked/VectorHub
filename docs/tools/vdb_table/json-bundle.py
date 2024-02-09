import json
import os

import requests

# Constants for the script
DIRECTORY = "docs/tools/vdb_table"
GITHUB_API_URL = "https://api.github.com/repos/"


def get_github_stars(github_url, headers):
    # Extract the owner and repo name from the GitHub URL
    parts = github_url.split("/")
    owner_repo = "/".join(parts[-2:])
    response = requests.get(f"{GITHUB_API_URL}{owner_repo}", headers=headers)
    if response.status_code == 200:
        return response.json()["stargazers_count"]
    else:
        print(f"Failed to fetch stars for {github_url}: {response.status_code}")
        return None


def update_json_files(directory, headers):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r+", encoding="utf-8") as json_file:
                data = json.load(json_file)
                github_url = data.get("links", {}).get("github", "")
                if github_url:
                    stars = get_github_stars(github_url, headers)
                    if stars is not None:
                        data["github_stars"] = stars
                        # Write the updated data back to the file
                        json_file.seek(0)  # Rewind to the start of the file
                        json.dump(data, json_file, indent=2)
                        json_file.truncate()  # Remove any leftover content


# if __name__ == "__main__":
#     # The GitHub token should be set as a secret in your GitHub repo settings
#     # and passed to the script via environment variable.
#     token = os.environ.get("GITHUB_TOKEN", "YOUR_GITHUB_TOKEN")
#     headers = {"Authorization": f"token {token}"}

update_json_files(DIRECTORY)  # past it was like update_json_files(DIRECTORY , headers)
