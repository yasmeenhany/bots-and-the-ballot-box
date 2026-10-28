import argparse
import json

parser = argparse.ArgumentParser(description="Print the first N objects of a JSON list (e.g. node.json).")
parser.add_argument("path", nargs="?", default="node.json")
parser.add_argument("-n", type=int, default=10)
args = parser.parse_args()

with open(args.path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for obj in data[:args.n]:
    print(obj)
