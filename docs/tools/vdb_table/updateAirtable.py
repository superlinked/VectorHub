import os
import requests
from dotenv import load_dotenv
import json
from pyairtable import Api
from pyairtable.orm import Model, fields as F

load_dotenv()
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("BASE_ID")
TABLE_ID = os.getenv("TABLE_ID")


api = Api(airtable_api_key)
base = api.base(BASE_ID)
table = api.table(BASE_ID, TABLE_ID)
# schema_obj = json.load(open("vendor.schema.json", "r"))


#TODO
def get_json_data():

    return list_of_fields, records



def update_table_fields():

    #TODO
    #code to check if there is a new field (column) added in the JSON, 
    #if yes then create that field in the table

    table_schema = table.schema()
    return table_schema


def update_table_records():

    list_of_fields = ["Database Name",
                      "Open-source & free to self-host", 
                      "Managed Cloud Offering",
                      "Disk-based Index",
                      "Multi-tenancy Support",
                      "In-built Text Embeddings creation(Bring-your-own-model)",
                      "In-built Image Embedding creation",
                      "Metadata Filtering",
                      "Embeddable",
                      "Multiple vectors per point",
                      "Langchain integration",
                      "Llama index integration",
                      "Hybrid Search",
                      "BM25 support",
                      "Sparse Vectors Support",
                      "Full-text Search Engine",
                      "Facets (Aggregations with Count)",
                      "GeoSearch Support",
                      "Metadata/Doc size limit",
                      "Max Dimensions",
                      "Ephemeral Index support(without server)",
                      "Sharding",
                      "License",
                      "Development Language",
                      "Github Stars",
                      "First Release of Vector Search",
                      "Pricing",
                      "Calls LLM internally for RAG",
                      "Recommendations API",
                      "Personalization",
                      "User events (clickstream)"
                    ]

    class VDB(Model):
        field_0 = F.TextField(list_of_fields[0])
        field_1 = F.RichTextField(list_of_fields[1])
        # field_2 = F.RichTextField(list_of_fields[2])
        # field_3 = F.RichTextField(list_of_fields[3])
        # field_4 = F.RichTextField(list_of_fields[4])
        # field_5 = F.RichTextField(list_of_fields[5])
        # field_6 = F.RichTextField(list_of_fields[6])
        # field_7 = F.RichTextField(list_of_fields[7])
        # field_8 = F.RichTextField(list_of_fields[8])
        # field_9 = F.RichTextField(list_of_fields[9])
        # field_10 = F.RichTextField(list_of_fields[10])
        # field_11 = F.RichTextField(list_of_fields[11])
        # field_12 = F.RichTextField(list_of_fields[12])
        # field_13 = F.RichTextField(list_of_fields[13])
        # field_14 = F.RichTextField(list_of_fields[14])
        # field_15 = F.RichTextField(list_of_fields[15])
        # field_16 = F.RichTextField(list_of_fields[16])
        # field_17 = F.RichTextField(list_of_fields[17])
        # field_18 = F.MultipleSelectField(list_of_fields[18])
        # field_19 = F.SelectField(list_of_fields[19])
        # field_20 = F.RichTextField(list_of_fields[20])
        # field_21 = F.RichTextField(list_of_fields[21])
        # field_22 = F.MultipleSelectField(list_of_fields[22])
        # field_23 = F.MultipleSelectField(list_of_fields[23])
        # field_24 = F.NumberField(list_of_fields[24])
        # field_25 = F.SelectField(list_of_fields[25])
        # field_26 = F.RichTextField(list_of_fields[26])
        # field_27 = F.RichTextField(list_of_fields[27])
        # field_28 = F.RichTextField(list_of_fields[28])
        # field_29 = F.RichTextField(list_of_fields[29])
        # field_30 = F.RichTextField(list_of_fields[30])

        class Meta:
            base_id = BASE_ID
            table_name = TABLE_ID
            api_key = airtable_api_key

    database_instance = VDB(
        field_0="Pinecone",
        field_1="✅",
    )

    assert database_instance.save()
    print(database_instance.exists())
    print(database_instance.id)



update_table_records()