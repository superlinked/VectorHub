import os
import requests

from tqdm.auto import tqdm

from datetime import date

from urllib.parse import urljoin

DUB_CO_API_TOKEN = os.getenv("DUB_CO_API_TOKEN")

class DubLink:
    def __init__(self, id, url, short_url, key) -> None:
        self.id = id
        self.url = url
        self.short_url = short_url
        self.count = -1
        self.key = key

    def set_click_cnt(self, count) -> None:
        self.count = count

    def __str__(self) -> str:
        return str(self.id)

BASE_URL = "https://api.dub.co"
headers = {
    'Authorization': f'Bearer {DUB_CO_API_TOKEN}',
    'Content-Type': 'application/json'
}

def upload_zapier_hook_data(data, zap_id) -> None:
    url = f"https://hooks.zapier.com/hooks/catch/{zap_id}/"
    r = requests.post(url, json=data)
    print(r.status_code)


def get_shortened_url_objs() -> list:
    global headers, DUB_CO_API_TOKEN, BASE_URL

    print('Fetching list of shortened URLs', flush=True)
    
    url_objs = []

    req = requests.get(urljoin(BASE_URL, "links"), headers=headers)
    if req.status_code < 300:
        data = req.json()
        for row in tqdm(data):
            url_objs.append(
                DubLink(
                    id=row['id'],
                    url=row['url'],
                    short_url=row['shortLink'],
                    key=row['key']
                )
            )
    
    print(f'Found {len(url_objs)} links', flush=True)

    return url_objs


def get_link_click_cnt(linkId) -> int:
    global headers, DUB_CO_API_TOKEN, BASE_URL

    params = {
        'linkId': f'{linkId}',
        'interval': '7d'
    }

    req = requests.get(urljoin(BASE_URL, "analytics"), headers=headers, params=params)
    if req.status_code < 300:
        data = req.json()
        return data['clicks']
    
    return 0

def get_links_clicks(url_objs) -> list:
    print('Fetching click count for every URL', flush=True)

    today = date.today()
    zap_data = {
        "time": str(today)
    }

    for obj in tqdm(url_objs):
        obj.set_click_cnt(get_link_click_cnt(obj.id))
        zap_data[f"{obj.key}"] = int(obj.count)

    upload_zapier_hook_data(zap_data, "5388531/2urr55c")

shortened_url_objs = get_shortened_url_objs()
get_links_clicks(shortened_url_objs)