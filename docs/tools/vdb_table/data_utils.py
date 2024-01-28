import os
import json
import argparse
import openpyxl
import re
import humanfriendly
import glob
from jsonschema import validate
from url_normalize import url_normalize
from collections import OrderedDict

class JsonProp():
    def __init__(self, prop) -> None:
        self.prop = prop
    
    def id(self):
        return self.prop[0]
    
    def type(self):
        # Parse the type - either just {type: value} or the $ref thing from jsonschema
        if 'type' in self.prop[1]:
            return self.prop[1]['type']
        return self.prop[1].get('allOf')[0]['$ref'].removeprefix('#/$defs/')
    
    def group(self):
        # $comment is a reserved prop in jsonschema, we use it for display name and comment with | as separator
        return self.prop[1].get('$comment','').split('|')[0].strip()
    
    def name(self):
        # $comment is a reserved prop in jsonschema, we use it for display name and comment with | as separator
        return self.prop[1].get('$comment','').split('|')[1].strip()
    
    def comment(self):
        # $comment is a reserved prop in jsonschema, we use it for display name and comment with | as separator
        name_and_comment = self.prop[1].get('$comment','').split('|')
        if len(name_and_comment) > 1:
            return name_and_comment[2].strip()
        return ''

    def __str__(self):
        return "%s, %s, %s, %s, %s"%(self.id(), self.type(), self.group(), self.name(), self.comment())

class JsonSchemaWrapper(): # Switch to XLSX
    def __init__(self, schema) -> None:
        self.schema = schema

    def props(self):
        return [JsonProp(prop) for prop in self.schema['properties'].items()]
    
    def prop_by_id(self, id):
        for prop in self.props():
            if prop.id() == id:
                return prop
        return None

class JsonValueFactory():    
    def backfillLinkFromValue(value, hyperlink):
        URL_REGEX = r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"
        if hyperlink:
            return hyperlink
        parsed_url = re.search(URL_REGEX, value)
        return parsed_url.group() if parsed_url else ""
                
    def convertValueToJson(schema, value, hyperlink):
        return getattr(JsonValueFactory, schema.type())(value, hyperlink)
    
    def bytesWithSource(value, hyperlink):
        try:
            num_bytes = humanfriendly.parse_size(value)
        except:
            num_bytes = 0
        return  {
            "bytes": num_bytes,
            "unlimited": True if "unlimited" in value.lower() else False,
            "source_url": hyperlink,
            "comment": ""
        }

    def featureWithSource(value, hyperlink):
        support = ""
        if "✅" in value: support = "full"
        if "🟨" in value: support = "partial"
        if "❌" in value: support = "none"
        return {
            "support": support,
            "source_url": JsonValueFactory.backfillLinkFromValue(value, hyperlink),
            "comment": "" if not value else value.translate({ord(x): '' for x in ["✅","❌","🟨"]}).strip()
        }

    def integer(value, _hyperlink):
        try:
            return int(value)
        except:
            return 0

    def integerWithSource(value, hyperlink):
        try:
            integer = int(value)
        except:
            integer = 0
        return {    
            "value": integer,
            "unlimited": True if "unlimited" in value.lower() else False,
            "source_url": hyperlink,
            "comment": ""
        }

    def links(value, _hyperlink):
        link_list = value.split("|")
        return {
            "docs": url_normalize(link_list[0]),
            "github": url_normalize(link_list[1]),
            "website": url_normalize(link_list[2]),
            "vendor_discussion": url_normalize(link_list[3]),
            "poc_github": url_normalize("https://github.com/%s"%(link_list[4])),
            "slug": link_list[5]
        }

    def string(value, _hyperlink):
        return value

    def stringListWithSource(value, hyperlink):
        return {
            "value": [val.strip().lower() for val in value.split(",")],
            "source_url": hyperlink,
            "comment": ""
        }

    def stringWithSource(value, hyperlink):
        return {    
            "value": value,
            "source_url": JsonValueFactory.backfillLinkFromValue(value, hyperlink),
            "comment": ""
        }
    
class XLSXWrapper():
    # Preparation steps for the raw CSV that have to be done manually:
    # 1. Fill in missing headers based on the vendor.schema.json file. Currently there are 2:
    #    Links
    #    In-built Structured Data Embedding creation
    # 2. Replace new-line characters in the file with spaces to make sure you have a valid CSV.
    HEADER_TO_SCHEMA_ID = {
        "DB | Attributes": "name",
        "Links": "links",
        "License": "license",
        "Development Language": "dev_languages",
        "Github Stars": "github_stars",
        "First Release of Vector Search": "vector_launch_year",
        "Metadata Filtering": "metadata_filter",
        "Hybrid Search": "hybrid_search",
        "Facets (Aggregations with Count)": "facets",
        "GeoSearch Support": "geo_search",
        "Multiple vectors per point": "multi_vec",
        "Sparse Vectors Support": "sparse_vectors",
        "BM25 support": "bm25",
        "Full-text Search Engine": "full_text",
        "In-built Text Embeddings creation (Bring-your-own-model)": "embeddings_text",
        "In-built Image Embedding creation": "embeddings_image",
        "In-built Structured Data Embedding creation": "embeddings_structured",
        "Calls LLM internally for RAG": "rag",
        "Recommendations API": "recsys",
        "Langchain integration": "langchain",
        "Llama index integration": "llamaindex",
        "Open-source & free to self-host": "oss",
        "Managed Cloud Offering": "managed_cloud",
        "Pricing": "pricing",
        "Embeddable": "in_process",
        "Multi-tenancy Support": "multi_tenancy",
        "Disk-based Index": "disk_index",
        "Ephemeral Index support (without server)": "ephemeral",
        "Sharding": "sharding",
        "Metadata/Doc size limit": "doc_size",
        "Max Dimensions": "vector_dims"
    }

    def map_header_to_schema_id(self, header):
        return self.HEADER_TO_SCHEMA_ID.get(header.replace("\n",""), None)

    def __init__(self, xlsx_path, xlsx_sheet, table_schema) -> None:
        self.xlsx_path = xlsx_path
        self.table_schema = table_schema
        self.xlsx_sheet = xlsx_sheet
    
    def value_to_json(self, value, schema):
        val = value.value
        if not val:
            val = None
        elif isinstance(val, float):
            val = str(int(val))
        else:
            val = str(val)

        return JsonValueFactory.convertValueToJson(
            schema, "" if not val else val,
            "" if not value.hyperlink else value.hyperlink.target)

    def row_to_json(self, header, row):
        # Collect the properties from the row.
        data = {}
        for i, value in enumerate(row):
            key = self.HEADER_TO_SCHEMA_ID.get(header[i].value, None)
            if key: 
                data[key] = self.value_to_json(value, self.table_schema.prop_by_id(key))
        # Order the properties according to the schema.
        result = OrderedDict()
        for prop in self.table_schema.props():
            result[prop.id()] = data[prop.id()]

        return result
    
    def to_json(self, output_dir):
        workbook = openpyxl.load_workbook(self.xlsx_path, data_only=True)
        sheet = workbook[self.xlsx_sheet]
        header_row = []
        for i, row in enumerate(sheet.iter_rows()):
            if not row[0].value: break
            if i == 0:
                header_row = row
            else:
                print("Processing", row[0].value)
                output_obj = self.row_to_json(header_row, row)
                with open(os.path.join(output_dir, output_obj["links"]["slug"]+".json"), "w") as output_file:
                    json.dump(output_obj, output_file, indent=2)

class CLI():
    # Use XLSX instead of CSV to preserve the hyperlinks in the original data.
    def xlsx_to_json(self, path_to_schema, output_dir, path_to_xlsx, xlsx_sheet):
        table_schema = JsonSchemaWrapper(json.load(open(path_to_schema, "r")))
        xlsx = XLSXWrapper(path_to_xlsx, xlsx_sheet, table_schema)
        xlsx.to_json(output_dir)
    
    def json_to_bundle(self, data_glob):
        obj_list = []
        for name in glob.glob(data_glob):  
            obj_list.append(json.load(open(name, "r"), object_pairs_hook=OrderedDict))
        with open("bundle.json", "w") as output_file:
            json.dump(obj_list, output_file, indent=2)
    
    def json_validate(self, path_to_schema, data_glob):
        schema_obj = json.load(open(path_to_schema, "r"))
        for name in glob.glob(data_glob):  
            vendor_obj = json.load(open(name, "r"))
            validate(instance=vendor_obj, schema=schema_obj)


# Executes when your script is called from the command-line:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI for managing the VDB json files in /data')
    parser.add_argument('command', help='The command you want to perform: xlsx_to_json')
    parser.add_argument('-sp','--schema_path', help='Path to the schema.json file.')
    parser.add_argument('-xp','--xlsx_path', help='Path to the legacy XSLX file.')
    parser.add_argument('-xs','--xlsx_sheet', help='Sheet to use in the XLSX file.')
    parser.add_argument('-od','--output_dir', help='Output directory for the vendor JSONs.')
    parser.add_argument('-dd','--data_glob', help='Glob pattern for the vendor JSON data.')

    args = parser.parse_args()
    
    cli = CLI()
    if args.command == 'xlsx_to_json':
        cli.xlsx_to_json(args.schema_path, args.output_dir, args.xlsx_path, args.xlsx_sheet)

    if args.command == 'json_to_bundle':
        cli.json_to_bundle(args.data_glob)
    
    if args.command == 'json_validate':
        cli.json_validate(args.schema_path, args.data_glob)

