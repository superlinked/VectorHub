from jsonschema import validate
import json

schema_obj = json.load(open("vendor.json.schema", "r"))
vendor_obj = json.load(open("vendorX.json", "r"))

validate(instance=vendor_obj, schema=schema_obj)

