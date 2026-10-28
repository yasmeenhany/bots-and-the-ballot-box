"""Convert the raw 2024 election tweet CSVs into TwiBot-20-style node.json / edge.csv.

Walks --input-dir for *.csv files: the raw dataset from Balasubramanian et al. (2024),
arXiv:2411.00376, after unzipping it with utils/unzip_all_files.py.

node.json  one entry per user, with profile fields and a list of their posts
edge.csv   source_id, relation, target_id with relation in {reply, retweet, post}
"""

import argparse
import ast
import datetime
import json
import os
import re

import pandas as pd

# Read IDs as strings so pandas never turns them into floats. Note that the raw dataset
# already stores reply-target IDs as floats (e.g. '1.486069441259397e+18'); those digits are
# lost upstream, which is why preprocess.py matches IDs as floats.
ID_COLUMNS = {'id_str': str, 'in_reply_to_user_id_str': str, 'inReplyToUser': str}
MISSING_REPLY_VALUES = {'nan', 'none', 'pw'}
EDGE_COLUMNS = ['source_id', 'relation', 'target_id']

# Match datetime.datetime(YYYY, MM, DD, HH, MM, SS, ...)
datetime_pattern = re.compile(
    r"datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+))?(?:,[^)]+)?\)"
)


def replace_datetime(match):
    y, m, d, h, mi, s = match.groups()
    s = s or 0
    # Format as ISO string (UTC)
    dt = datetime.datetime(int(y), int(m), int(d), int(h), int(mi), int(s), tzinfo=datetime.timezone.utc)
    return f"'{dt.isoformat()}'"


def clean_user_str(s):
    if not isinstance(s, str):
        return s
    return datetime_pattern.sub(replace_datetime, s)


def parse_user_dict(s):
    try:
        return ast.literal_eval(clean_user_str(s))
    except Exception as e:
        print(f"Failed to parse: {e}\nID: {s}\n")
        return None


def add_nodes(df, all_nodes):
    """Add each tweet's author (and the tweet) to all_nodes. Returns df with a user_id_str column."""
    df['user'] = df['user'].apply(parse_user_dict)
    df['user_id_str'] = df['user'].apply(lambda x: x.get('id_str') if isinstance(x, dict) else None)

    for col in ['replyCount', 'retweetCount', 'likeCount', 'quoteCount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Drop any rows where user info is missing
    df = df[df['user_id_str'].notnull()].copy()

    for _, row in df.iterrows():
        user_dict = row['user']
        uid = user_dict.get('id_str')
        if not uid:
            continue

        tweet = {
            "id": row.get('id_str', ""),
            "content": row.get('rawContent', ""),
            "replyCount": row.get('replyCount', 0),
            "retweetCount": row.get('retweetCount', 0),
            "likeCount": row.get('likeCount', 0),
            "quoteCount": row.get('quoteCount', 0)
        }
        if uid in all_nodes:
            all_nodes[uid]["posts"].append(tweet)
            continue

        all_nodes[uid] = {
            "id": uid,
            "name": user_dict.get("username", ""),
            # Known issue: the raw CSVs have no top-level "username" column, so this is
            # always "". Kept as-is so reruns reproduce the published tensors; see README.
            "username": row.get("username", ""),
            "public_metrics": {
                "followers_count": user_dict.get("followersCount", 0),
                "friends_count": user_dict.get("friendsCount", 0),
                "statuses_count": user_dict.get("statusesCount", 0),
            },
            "url": user_dict.get("url", ""),
            "profile_image_url": user_dict.get("profileImageUrl", ""),
            "location": user_dict.get("location", ""),
            "verified": user_dict.get("verified", False),
            "protected": user_dict.get("protected", False),
            "description": user_dict.get("rawDescription", ""),
            "created_at": user_dict.get("created", None),
            "posts": [tweet]
        }
    return df


def create_edges(df):
    """Build reply (user -> user), retweet (user -> tweet) and post (user -> tweet) edges."""
    df = df[df['user_id_str'].notnull()].copy()
    df['user_id_str'] = df['user_id_str'].astype(str)

    reply_col = next((c for c in ['in_reply_to_user_id_str', 'inReplyToUser'] if c in df.columns), None)
    if reply_col:
        is_reply = ~df[reply_col].astype(str).str.lower().isin(MISSING_REPLY_VALUES)
    else:
        is_reply = pd.Series(False, index=df.index)
    replies = df[is_reply]
    retweets = df[df['retweetedTweet'] == True]
    posts = df[(df['retweetedTweet'] == False) & ~is_reply]

    return pd.concat([
        pd.DataFrame({"source_id": replies['user_id_str'], "relation": "reply", "target_id": replies[reply_col]})
        if reply_col else pd.DataFrame(columns=EDGE_COLUMNS),
        pd.DataFrame({"source_id": retweets['user_id_str'], "relation": "retweet", "target_id": retweets['id_str']}),
        pd.DataFrame({"source_id": posts['user_id_str'], "relation": "post", "target_id": posts['id_str']}),
    ], ignore_index=True)


def write_edges(edge_batches, path, header):
    edges = pd.concat(edge_batches, ignore_index=True).drop_duplicates(subset=EDGE_COLUMNS)
    edges.to_csv(path, mode='w' if header else 'a', index=False, header=header)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', required=True, help='folder containing the raw tweet CSVs (searched recursively)')
    parser.add_argument('--output-dir', default='.', help='where to write node.json and edge.csv')
    parser.add_argument('--flush-every', type=int, default=10, help='write edges to disk every N files')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    edge_path = os.path.join(args.output_dir, "edge.csv")
    node_path = os.path.join(args.output_dir, "node.json")

    csv_paths = sorted(os.path.join(subdir, f)
                       for subdir, _, files in os.walk(args.input_dir)
                       for f in files if f.endswith(".csv"))

    all_nodes = {}
    edge_batches = []
    wrote_header = False
    for count, path in enumerate(csv_paths, start=1):
        print(path)
        df = pd.read_csv(path, low_memory=False, dtype=ID_COLUMNS)
        df = add_nodes(df, all_nodes)
        edge_batches.append(create_edges(df))

        if count % args.flush_every == 0 or count == len(csv_paths):
            write_edges(edge_batches, edge_path, header=not wrote_header)
            wrote_header = True
            edge_batches = []
            print(f"Completed {count}/{len(csv_paths)}")

    print("Saving nodes to json file")
    with open(node_path, "w") as f:
        json.dump(list(all_nodes.values()), f, indent=2)
    print("Completed")


if __name__ == "__main__":
    main()
