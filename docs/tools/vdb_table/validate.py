from jsonschema import validate
import json
import glob

schema_obj = json.load(open("vendor.schema.json", "r"))
for name in glob.glob('data/*'):  
    vendor_obj = json.load(open(name, "r"))
    validate(instance=vendor_obj, schema=schema_obj)

