import json
import os
from datetime import datetime, timedelta
from urllib.parse import urlparse

import pypistats
import requests
from tqdm.auto import tqdm

# Constants for the script
DIRECTORY = "docs/tools/vdb_table/data"

GITHUB_API_URL = "https://api.github.com/repos/"
DOCKER_HUB_API_URL = "https://hub.docker.com/v2/repositories/"
NPM_API_URL = "https://api.npmjs.org/downloads/point/"
PYPI_API_URL = "https://api.pepy.tech/api/v2/projects"
RUST_CRATES_API_URL = "https://crates.io/api/v1/crates"


def get_github_stars(github_url, headers=None):
    global GITHUB_API_URL
    parts = github_url.split("/")
    owner_repo = "/".join(parts[-2:])
    response = requests.get(f"{GITHUB_API_URL}{owner_repo}", headers=headers)
    if response.status_code == 200:
        return response.json()["stargazers_count"]
    else:
        print(f"Failed to fetch stars for {github_url}: {response.status_code}")
        return None


def get_docker_pulls(namespace, repo_name, headers=None):
    global DOCKER_HUB_API_URL
    response = requests.get(
        f"{DOCKER_HUB_API_URL}{namespace}/{repo_name}/", headers=None
    )
    if response.status_code == 200:
        return response.json()["pull_count"]
    else:
        print(
            f"Failed to fetch pulls for {namespace}/{repo_name}: {response.status_code}"
        )
        return None


def get_npm_package_info(npm_package):
    response = requests.get(f"https://registry.npmjs.org/{npm_package}")
    if response.status_code == 200:
        data = response.json()
        first_release_date = data.get("time")["created"]
        return first_release_date
    else:
        print(
            f"Failed to fetch npm package info for {npm_package}: {response.status_code}"
        )
        return None


def get_npm_downloads(npm_package, headers=None, start_date=None):
    global NPM_API_URL
    if start_date is None:
        release_date = get_npm_package_info(npm_package)
        if release_date:
            # Subtract one day from the release date for the start range
            start_date = datetime.strptime(
                release_date, "%Y-%m-%dT%H:%M:%S.%fZ"
            ) - timedelta(days=1)

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = (datetime.now() + +timedelta(days=1)).strftime("%Y-%m-%d")
    response = requests.get(
        f"{NPM_API_URL}{start_date}:{end_date}/{npm_package}", headers
    )
    if response.status_code == 200:
        return response.json()["downloads"]
    else:
        print(
            f"Failed to fetch npm downloads for {npm_package}: {response.status_code}"
        )
        return None


def get_pypi_downloads(pypi_package, headers=None):
    global PYPI_API_URL
    headers["X-Api-Key"] = os.getenv("PYPI_API_KEY", "")
    response = requests.get(f"{PYPI_API_URL}/{pypi_package}", headers=headers)
    if response.status_code == 200:
        return response.json()["total_downloads"]
    else:
        print(
            f"Failed to fetch pypi downloads for {pypi_package}: {response.status_code}"
        )
        return None


def get_pypi_downloads_last_90(
    pypi_package, headers=None, start_date=None, end_date=None
):
    stats = json.loads(
        pypistats.overall(
            pypi_package,
            mirrors=True,
            format="json",
            start_date=start_date,
            end_date=end_date,
        )
    )
    if stats and "data" in stats:
        return stats["data"][0]["downloads"]
    else:
        print(f"Failed to fetch pypi downloads for {pypi_package}")
        return None


def get_rust_downloads(crate_name):
    global RUST_CRATES_API_URL
    response = requests.get(f"{RUST_CRATES_API_URL}/{crate_name}")
    if response.status_code == 200:
        try:
            return response.json()["crate"]["downloads"]
        except Exception as e:
            print(f'Failed to fetch total downloads for rust crate {crate_name} with exception: {e}')
            return None
    else:
        print(
            f"Failed to fetch total downloads for rust crate {crate_name}: {response.status_code}"
        )
        return None


def get_rust_downloads_last_90(crate_name):
    global RUST_CRATES_API_URL
    response = requests.get(f"{RUST_CRATES_API_URL}/{crate_name}/downloads")
    if response.status_code == 200:
        try:
            downloads_data = response.json()["version_downloads"]
            last_90_days_date = datetime.now() - timedelta(days=90)
            downloads_last_90_days = sum(
                item["downloads"]
                for item in downloads_data
                if datetime.strptime(item["date"], "%Y-%m-%d") >= last_90_days_date
            )
            return downloads_last_90_days
        except Exception as e:
            print(f'Failed to fetch downloads for the last 90 days for rust crate {crate_name} with exception: {e}')
            return None
    else:
        print(
            f"Failed to fetch downloads for the last 90 days for rust crate {crate_name}: {response.status_code}"
        )
        return None


def update_json_files(directory, headers=None):
    if headers is None:
        headers = {}

    sources = ["github_stars", "docker_pulls", "npm_downloads", "pypi_downloads", "crates_io_downloads"]

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r+", encoding="utf-8") as json_file:
                data = json.load(json_file)
                github_url = data.get("github_stars", {}).get("source_url", "")
                dockerhub_url = data.get("docker_pulls", {}).get("source_url", "")
                npm_url = data.get("npm_downloads", {}).get("source_url", "")
                pypi_url = data.get("pypi_downloads", {}).get("source_url", "")
                rust_url = data.get("crates_io_downloads", {}).get("source_url", "")

                for source in sources:
                    if "value_90_days" not in data[source]:
                        data[source]["value_90_days"] = 0

                if dockerhub_url:
                    parsed_dockerhub_path = str(urlparse(dockerhub_url).path)
                    docker_namespace = (
                        list(parsed_dockerhub_path.strip().split("/"))[-2]
                        if "/_/" not in parsed_dockerhub_path
                        else "library"
                    )
                    docker_repo_name = list(parsed_dockerhub_path.strip().split("/"))[
                        -1
                    ]
                    pulls = get_docker_pulls(docker_namespace, docker_repo_name)
                    if pulls is not None:
                        data["docker_pulls"]["value"] = pulls

                if github_url:
                    stars = get_github_stars(github_url, headers)
                    if stars is not None:
                        data["github_stars"]["value"] = stars

                if npm_url:
                    npm_package_name = list(
                        npm_url.split("https://www.npmjs.com/package/")
                    )[1].strip()
                    downloads = get_npm_downloads(npm_package_name, headers)
                    if downloads is not None:
                        data["npm_downloads"]["value"] = downloads

                    start_date = datetime.now() - timedelta(days=90)
                    downloads = get_npm_downloads(npm_package_name, headers, start_date)
                    if downloads is not None:
                        data["npm_downloads"]["value_90_days"] = downloads

                if pypi_url:
                    pypi_package_name = (
                        list(pypi_url.split("https://pypi.org/project/"))[1]
                        .strip()
                        .strip("/")
                    )
                    downloads = get_pypi_downloads(pypi_package_name, headers)
                    if downloads is not None:
                        data["pypi_downloads"]["value"] = downloads

                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=90)
                    downloads = get_pypi_downloads_last_90(
                        pypi_package_name,
                        headers,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )
                    if downloads is not None:
                        data["pypi_downloads"]["value_90_days"] = downloads

                if rust_url:
                    rust_crate_name = (
                        list(rust_url.split("https://crates.io/crates/"))[1]
                        .strip()
                        .strip("/")
                    )
                    downloads = get_rust_downloads(rust_crate_name)
                    if downloads is not None:
                        data["crates_io_downloads"]["value"] = downloads

                    downloads_last_90 = get_rust_downloads_last_90(rust_crate_name)
                    if downloads_last_90 is not None:
                        data["crates_io_downloads"]["value_90_days"] = downloads_last_90

                # Write the updated data back to the file
                json_file.seek(0)  # Rewind to the start of the file
                json.dump(data, json_file, indent=2)
                json_file.truncate()  # Remove any leftover content


if __name__ == "__main__":
    update_json_files(DIRECTORY)
