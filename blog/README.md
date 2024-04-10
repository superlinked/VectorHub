# Blog Uploads

## Setup
```bash
cd blog
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run
```bash
export STRAPI_URL="https://mystrapi.strapiapp.com"
export STRAPI_API_KEY="mystrapi-api-key"
source blog/env/bin/activate
python blog/main.py --directories blog/directories.json
```