import argparse
import csv
import json


def csv_to_json(csv_file_path, json_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        rows = list(csv.DictReader(csvfile))
    with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
        json.dump(rows, jsonfile, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CSV file to a JSON list of row objects.")
    parser.add_argument("csv_file")
    parser.add_argument("json_file")
    args = parser.parse_args()
    csv_to_json(args.csv_file, args.json_file)
