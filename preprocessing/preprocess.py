"""Turn node.json / edge.csv into the feature tensors BotMoE expects.

Adapted from the TwiBot-20 BotRGCN preprocessing scripts:
https://github.com/LuoUndergradXJTU/TwiBot-22/blob/master/src/BotRGCN/twibot_20/preprocess_1.py
https://github.com/LuoUndergradXJTU/TwiBot-22/blob/master/src/BotRGCN/twibot_20/preprocess_2.py
Fields that don't exist in the 2024 election data were removed.

Outputs (written to --output-dir):
    uid_to_user_index.pkl     exact user ID (str) -> row index in every tensor
    edge_index.pt, edge_type.pt
    num_properties_tensor.pt  followers, active days, screen-name length, friends, statuses (z-scored)
    cat_properties_tensor.pt  default profile image, protected, verified
    each_user_tweets.npy      user index -> list of tweet indices
    des_tensor.pt             RoBERTa embedding of each user's description
    tweets_tensor.pt          mean RoBERTa embedding of each user's first 20 tweets

The two embedding files are skipped if they already exist in --output-dir, and the tweet
embedding resumes from its last checkpoint if a previous run was interrupted.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

DEFAULT_PROFILE_IMAGE = 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png'
EMBEDDING_DIM = 768
MAX_TWEETS_PER_USER = 20


def id_key(value):
    """Key used to match IDs between node.json and edge.csv.

    The raw dataset stores reply-target IDs as floats, so edge.csv contains values
    like '1.4835964632212644e+18' whose last digits are lost. Comparing IDs as floats
    lets those rows still match their users. Non-numeric IDs (e.g. TwiBot's 'u123')
    are compared as strings.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def create_user_and_tweet(node):
    print('processing raw data')

    user_rows = []
    tweet_rows = []

    for _, user in node.iterrows():
        user_id = str(user.get("id"))
        user_rows.append({
            "id": user_id,
            "name": user.get("name", ""),
            "username": user.get("username", ""),
            "public_metrics": user.get("public_metrics", {}),
            "url": user.get("url", ""),
            "profile_image_url": user.get("profile_image_url", ""),
            "location": user.get("location", ""),
            "verified": user.get("verified", False),
            "description": user.get("description", ""),
            "created_at": user.get("created_at", "")
        })

        for post in user.get("posts", []):
            tweet_rows.append({
                "id": str(post.get("id")),
                "user_id": user_id,
                "content": post.get("content", ""),
                "replyCount": post.get("replyCount", 0),
                "retweetCount": post.get("retweetCount", 0),
                "likeCount": post.get("likeCount", 0),
                "quoteCount": post.get("quoteCount", 0)
            })

    return pd.DataFrame(user_rows), pd.DataFrame(tweet_rows)


def create_graph_info(edge, uid_key_to_index, path):
    print('extracting graph info')
    edge = edge[edge['relation'] != 'post']
    source = edge['source_key'].map(uid_key_to_index)
    target = edge['target_key'].map(uid_key_to_index)
    keep = source.notna() & target.notna()

    edge_index = torch.tensor([source[keep].astype(int).tolist(),
                               target[keep].astype(int).tolist()], dtype=torch.long)
    # Every user-user edge (reply/retweet) is treated as one relation type.
    edge_type = torch.ones(edge_index.shape[1], dtype=torch.long)

    torch.save(edge_index, os.path.join(path, 'edge_index.pt'))
    torch.save(edge_type, os.path.join(path, 'edge_type.pt'))
    print(f'edge_index: {tuple(edge_index.shape)}, edge_type: {tuple(edge_type.shape)}')


def zscore(values, dtype=None):
    df = pd.DataFrame(values)
    if dtype is not None:
        df = df.fillna(0).astype(dtype)
    df = (df - df.mean()) / df.std()
    return torch.tensor(np.array(df), dtype=torch.float32)


def metric(public_metrics, name):
    if isinstance(public_metrics, dict) and public_metrics.get(name) is not None:
        return public_metrics[name]
    return 0


def create_num_properties(user, reference_date, path):
    print('extracting num_properties')

    followers_count = [int(metric(m, 'followers_count')) for m in user['public_metrics']]
    friends_count = [int(metric(m, 'friends_count')) for m in user['public_metrics']]
    statuses_count = [metric(m, 'statuses_count') for m in user['public_metrics']]
    screen_name_length = [len(name) if name is not None else 0 for name in user['username']]

    created_at = pd.to_datetime(user['created_at'], unit='s')
    active_days = [(reference_date - each).days for each in created_at]

    num_properties_tensor = torch.cat([zscore(followers_count),
                                       zscore(active_days, dtype=np.float32),
                                       zscore(screen_name_length),
                                       zscore(friends_count),
                                       zscore(statuses_count)], dim=1)
    torch.save(num_properties_tensor, os.path.join(path, 'num_properties_tensor.pt'))


def is_true(value):
    return value is True or value == 'True '


def create_cat_properties(user, path):
    print('extracting cat_properties')
    default_profile_image = [1 if url is None or url == DEFAULT_PROFILE_IMAGE else 0
                             for url in user['profile_image_url']]
    protected = user['protected'] if 'protected' in user.columns else [False] * len(user)

    cat_properties_tensor = torch.stack([
        torch.tensor(default_profile_image, dtype=torch.float),
        torch.tensor([int(is_true(v)) for v in protected], dtype=torch.float),
        torch.tensor([int(is_true(v)) for v in user['verified']], dtype=torch.float),
    ], dim=1)
    torch.save(cat_properties_tensor, os.path.join(path, 'cat_properties_tensor.pt'))


def get_each_user_tweets(edge, uid_key_to_index, tid_key_to_index, num_users, path):
    print('mapping users to their tweets')
    posts = edge[edge['relation'] == 'post']
    source = posts['source_key'].map(uid_key_to_index)
    target = posts['target_key'].map(tid_key_to_index)
    keep = source.notna() & target.notna()

    each_user_tweets = {i: [] for i in range(num_users)}
    for user_index, tweet_index in zip(source[keep].astype(int), target[keep].astype(int)):
        each_user_tweets[user_index].append(tweet_index)

    np.save(os.path.join(path, "each_user_tweets.npy"), each_user_tweets)
    return each_user_tweets


def mean_token_embedding(feature_extract, text):
    return torch.tensor(feature_extract(text))[0].mean(dim=0)


def embed_text(feature_extract, text):
    """Mean RoBERTa token embedding; empty or missing text becomes a zero vector."""
    if not text:
        return torch.zeros(EMBEDDING_DIM)
    with torch.no_grad():
        return mean_token_embedding(feature_extract, text)


def des_embedding(feature_extract, descriptions, path):
    print('Running description embedding')
    save_path = os.path.join(path, "des_tensor.pt")
    if os.path.exists(save_path):
        print(f'{save_path} already exists, skipping')
        return torch.load(save_path)

    des_tensor = torch.stack([embed_text(feature_extract, text) for text in tqdm(descriptions)], 0)
    torch.save(des_tensor, save_path)
    return des_tensor


def tweets_embedding(feature_extract, each_user_tweets, tweet_text, path, checkpoint_every=20000):
    """Average the embeddings of each user's first 20 tweets (empty tweets count as zero vectors).

    Progress is checkpointed every `checkpoint_every` users, so an interrupted run resumes where it stopped.
    """
    print('Running tweet embedding')
    save_path = os.path.join(path, "tweets_tensor.pt")
    chk_data = os.path.join(path, "tweets_partial.pt")
    chk_state = os.path.join(path, "tweets_state.json")
    if os.path.exists(save_path):
        print(f'{save_path} already exists, skipping')
        return torch.load(save_path)

    if os.path.exists(chk_state) and os.path.exists(chk_data):
        with open(chk_state) as f:
            start_idx = json.load(f)["start_idx"]
        user_vectors = list(torch.load(chk_data).unbind(0))
        print(f"Resuming at user {start_idx}")
    else:
        start_idx = 0
        user_vectors = []

    for i in tqdm(range(start_idx, len(each_user_tweets))):
        tweet_ids = each_user_tweets[i][:MAX_TWEETS_PER_USER]
        if tweet_ids:
            user_vectors.append(torch.stack([embed_text(feature_extract, tweet_text[tid]) for tid in tweet_ids]).mean(dim=0))
        else:
            user_vectors.append(torch.zeros(EMBEDDING_DIM))

        if (i + 1) % checkpoint_every == 0:
            # Save as one stacked tensor; a list of tensors makes the file much larger.
            torch.save(torch.stack(user_vectors), chk_data)
            with open(chk_state, "w") as f:
                json.dump({"start_idx": i + 1}, f)

    tweets_tensor = torch.stack(user_vectors)
    torch.save(tweets_tensor, save_path)
    for f in (chk_data, chk_state):
        if os.path.exists(f):
            os.remove(f)
    return tweets_tensor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--node', default='sample-data/sample-node.json', help='node.json written by clean_data.py')
    parser.add_argument('--edge', default='sample-data/sample-edge.csv', help='edge.csv written by clean_data.py')
    parser.add_argument('--output-dir', default='results/', help='where to write the tensors')
    parser.add_argument('--reference-date', default='2024-11-30',
                        help="date used to compute account age (use 2020-09-01 for TwiBot-20)")
    parser.add_argument('--checkpoint-every', type=int, default=20000,
                        help='save tweet-embedding progress every N users so an interrupted run can resume')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='only build the graph/property files, not the RoBERTa embeddings')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('loading raw data')
    # Without dtype, read_json rounds 19-digit IDs.
    node = pd.read_json(args.node, dtype={'id': str})
    edge = pd.read_csv(args.edge, dtype={'source_id': str, 'target_id': str})
    edge['source_key'] = edge['source_id'].map(id_key)
    edge['target_key'] = edge['target_id'].map(id_key)

    user, tweet = create_user_and_tweet(node)
    # A node.json merged from several clean_data.py runs can list a user more than once. Keep the
    # first entry so every tensor row and uid_to_user_index refer to the same users.
    duplicates = user['id'].duplicated()
    if duplicates.any():
        print(f'dropping {duplicates.sum()} duplicate users')
        user = user[~duplicates].reset_index(drop=True)
    uid_key_to_index = {id_key(uid): i for i, uid in enumerate(user['id'])}
    tid_key_to_index = {id_key(tid): i for i, tid in enumerate(tweet['id'])}

    # Exact IDs, so predictions can be joined back to the original accounts.
    uid_to_user_index = {uid: i for i, uid in enumerate(user['id'])}
    with open(os.path.join(args.output_dir, "uid_to_user_index.pkl"), 'wb') as f:
        pickle.dump(uid_to_user_index, f)

    create_graph_info(edge, uid_key_to_index, args.output_dir)
    create_num_properties(user, pd.Timestamp(args.reference_date, tz='UTC'), args.output_dir)
    create_cat_properties(user, args.output_dir)
    each_user_tweets = get_each_user_tweets(edge, uid_key_to_index, tid_key_to_index, len(user), args.output_dir)

    if args.skip_embeddings:
        return

    # Uses the GPU if one is available, otherwise the CPU.
    device_id = 0 if torch.cuda.is_available() else -1
    feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base',
                               device=device_id, padding=True, truncation=True, max_length=50,
                               add_special_tokens=True)

    des_embedding(feature_extract, list(user['description']), args.output_dir)
    tweets_embedding(feature_extract, each_user_tweets, list(tweet['content']), args.output_dir,
                     checkpoint_every=args.checkpoint_every)
    print('Finished')


if __name__ == "__main__":
    main()
