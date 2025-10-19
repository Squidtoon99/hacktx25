import json, yaml, sys
from jsonschema import Draft202012Validator
schema = json.load(open("strategy.schema.json"))
doc = yaml.safe_load(open("strategy.yaml"))
Draft202012Validator(schema).validate(doc)
print("valid")
