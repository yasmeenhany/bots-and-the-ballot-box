import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os
import pathlib
import json

user,tweet=fast_merge(dataset="Twibot-20", server_id="local")

user_text=list(user['description'])
tweet_text = [text for text in tweet.text]
each_user_tweets = np.load('./processed_data/each_user_tweets.npy', allow_pickle=True).item()

device_id = 0 if torch.cuda.is_available() else -1
feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base',device=device_id,padding=True, truncation=True,max_length=50, add_special_tokens = True)

def Des_embbeding():
        print('Running feature1 embedding')
        path="./processed_data/des_tensor.pt"
        if not os.path.exists(path):
            des_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                if each is None:
                    des_vec.append(torch.zeros(768))
                else:
                    feature=torch.Tensor(feature_extract(each))
                    for (i,tensor) in enumerate(feature[0]):
                        if i==0:
                            feature_tensor=tensor
                        else:
                            feature_tensor+=tensor
                    feature_tensor/=feature.shape[1]
                    des_vec.append(feature_tensor)
                    
            des_tensor=torch.stack(des_vec,0)
            torch.save(des_tensor,path)
        else:
            des_tensor=torch.load(path)
        print('Finished')
        return des_tensor


CHK_DATA = "./processed_data/tweets_partial.pt"
CHK_STATE = "./processed_data/tweets_state.json"

def tweets_embedding(resume=True, checkpoint_every=1000):
    """
    Incrementally builds tweets_tensor.pt.
    Parameters:
    - resume: If True, resumes from the last checkpoint.
    - checkpoint_every: Saves progress every N users.
    """
    print("tweets_embedding...")

    if resume and os.path.exists(CHK_STATE):
        with open(CHK_STATE) as f:
            state = json.load(f)
        start_idx = state["start_idx"]
        tweets_list = torch.load(CHK_DATA)
        print(f"Resuming at index {start_idx} (have {len(tweets_list)} users)")
    else:
        start_idx = 0
        tweets_list = []
        print("Starting from scratch")

    for i in tqdm(range(start_idx, len(each_user_tweets))):
        if len(each_user_tweets[i]) == 0:
            total_each_person_tweets = torch.zeros(768)
        else:
            for j in range(len(each_user_tweets[i])):
                each_tweet = tweet_text[each_user_tweets[i][j]]
                if each_tweet is None:
                    total_word_tensor = torch.zeros(768)
                else:
                    each_tweet_tensor = torch.tensor(feature_extract(each_tweet))
                    for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                        if k==0:
                            total_word_tensor=each_word_tensor
                        else:
                            total_word_tensor+=each_word_tensor
                    total_word_tensor/=each_tweet_tensor.shape[1]
                if j == 0:
                    total_each_person_tweets = total_word_tensor
                elif j == 20:
                    break
                else:
                    total_each_person_tweets += total_word_tensor
            total_each_person_tweets /= 20 if j == 20 else len(each_user_tweets[i])

        tweets_list.append(total_each_person_tweets)

        if (i + 1) % checkpoint_every == 0:
            torch.save(tweets_list, CHK_DATA)
            with open(CHK_STATE, "w") as f:
                json.dump({"start_idx": i + 1}, f)
            print(f"Checkpointed at index {i+1}")

    full_tensor = torch.stack(tweets_list)
    torch.save(full_tensor, "./processed_data/tweets_tensor.pt")

    if os.path.exists(CHK_DATA):
        os.remove(pathlib.Path(CHK_DATA).resolve())
    if os.path.exists(CHK_STATE):
        os.remove(pathlib.Path(CHK_STATE).resolve())

    print("Finished. Saved tweets_tensor.pt")

Des_embbeding()
tweets_embedding()
