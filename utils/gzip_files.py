import argparse
import gzip
import shutil

parser = argparse.ArgumentParser(description="Gzip a file, writing <file>.gz next to it.")
parser.add_argument("path", help="e.g. node.json")
args = parser.parse_args()

with open(args.path, "rb") as f_in:
    with gzip.open(args.path + ".gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
