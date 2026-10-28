from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os
from torch_geometric.data import Data, HeteroData

def get_data_dir(server_id):
    if server_id == "206":
        return Path("/new_temp/fsb/Twibot22-baselines/datasets")
    elif server_id == "208":
        return Path("")
    elif server_id == "209":
        return Path("../datasets")
    elif server_id == "local":
        return Path(__file__).parent
    else:
        raise NotImplementedError

dataset_names = [
    'botometer-feedback-2019', 'botwiki-2019', 'celebrity-2019', 'cresci-2015', 'cresci-2017', 'cresci-rtbust-2019', 'cresci-stock-2018', 'gilani-2017', 'midterm-2018', 'political-bots-2019', 'pronbots-2019', 'vendor-purchased-2019', 'verified-2019', "Twibot-20"
]

def split_user_and_tweet(df):
    """
    split user and tweet from df, and ignore entries whose id is `None`
    """
    df = df[df.id.str.len() > 0]
    return df[df.id.str.contains("^u")], df[df.id.str.contains("^t")]
    

def fast_merge(dataset="Twibot-20", server_id="209"):
    print("fast_merge...")
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    node_info = pd.read_json(dataset_dir / "node.json")
    label = pd.read_csv(dataset_dir / "label.csv")
    split = pd.read_csv(dataset_dir / "split.csv")
    
    user, tweet = split_user_and_tweet(node_info)
    
    id_to_label = {}
    id_to_split = {}
    
    length = len(label)
    
    if len(label) != len(split):
        for i in range(length):
            id_to_label[label["id"][i]] = label["label"][i]
        
        length = len(split)
        for i in range(length):
            id_to_split[split["id"][i]] = split["split"][i]
    else:
        for i in range(length):
            id_to_label[label["id"][i]] = label["label"][i]
            id_to_split[split["id"][i]] = split["split"][i]
    
    length = len(user)
    
    user["label"] = "None"
    user["split"] = "None"
    
    for i in tqdm(range(length)):
        if user["id"][i] in id_to_label.keys():
            if i<len(label):
                user["label"][i] = id_to_label[user["id"][i]]
                user["split"][i] = id_to_split[user["id"][i]]
            else:
                user["label"][i] = np.nan
                user["split"][i] = id_to_split[user["id"][i]]

    return user, tweet


def df_to_mask(uid_with_label, uid_to_user_index, phase="train"):
    user_list = list(uid_with_label[uid_with_label.split == phase].id)
    phase_index = list(map(lambda x: uid_to_user_index[x], user_list))
    return torch.LongTensor(phase_index)
