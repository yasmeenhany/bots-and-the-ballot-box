# Bots and the Ballot Box

Code for **"Bots and the Ballot Box: How Automated Accounts Shaped Narratives and Networks in the 2024 US Election"**
by Yasmeen Abdelhafeez, Yaroslava Boyko, and Masaki Ito (Georgia Institute of Technology).

📄 [Paper (PDF)](paper/bots_and_the_ballot_box.pdf)

We analyze 42 million election-related tweets from X (formerly Twitter), label 4.3 million accounts as bots or humans with
[BotMoE](https://arxiv.org/abs/2304.06280), and compare how bots and humans behave over time, what they talk about, and where
they sit in the network.

**Key findings**
- Bots post **three times more tweets** than humans in the peak month, and use **five to seven times more hashtags**. Activity spikes
  around the Trump assassination attempt, the party conventions, and Biden's withdrawal.
- Bots consistently amplify pro-Trump slogans and media tags (e.g. `#foxnews`). Human discussion covers more topics and dies down faster.
- Removing bots splits the reply graph into **about six times more communities**. Bots act as bridges that hide how fragmented the
  conversation really is.

## Pipeline

```
raw tweet CSVs ──► clean_data.py ──► node.json + edge.csv ──► preprocess.py ──► feature tensors
                                                                                      │
TwiBot-20 ──► twibot_20/ ──► BotMoE/main.py (train) ──► checkpoints ──► BotMoE/test.py ◄┘
                                                                              │
                                                           bot/human predictions
                                                              │            │
                                              polarization_analysis/   network_analysis/
```

| Folder | What it does |
|---|---|
| [`preprocessing/`](preprocessing/) | Converts the raw election CSVs into TwiBot-20-style `node.json`/`edge.csv`, then into the RoBERTa and profile-feature tensors BotMoE needs. |
| [`BotMoE/`](BotMoE/) | The official BotMoE code ([paper](https://arxiv.org/abs/2304.06280), [MIT licence](BotMoE/LICENSE)), with our changes for training on TwiBot-20 and labelling millions of users (`test.py`). |
| [`twibot_20/`](twibot_20/) | Preprocessing for the TwiBot-20 training set, from the [TwiBot-22 BotRGCN baseline](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotRGCN/twibot_20). |
| [`polarization_analysis/`](polarization_analysis/) | Louvain/Leiden community detection on the reply graph with and without bots (Sections 4.5, 5.2). |
| [`network_analysis/`](network_analysis/) | PageRank of bots vs. humans and bot → human interactions (Sections 4.6, 5.3). |
| [`utils/`](utils/) | Small helpers for unzipping and inspecting the raw data. |

The temporal and hashtag/topic analyses (Sections 4.3, 5.1) and the degree and eigenvector centrality results in Section 5.3
are not in this repository yet.

## Setup

Preprocessing and BotMoE need different PyTorch versions, so use separate environments (all Python 3.9):

```bash
# Preprocessing (clean_data.py, preprocess.py)
conda create -n ballot-preprocess python=3.9 && conda activate ballot-preprocess
pip install -r preprocessing/requirements.txt

# BotMoE (training and inference), the sample polarization notebook and the network analysis
conda create -n ballot-botmoe python=3.9 && conda activate ballot-botmoe
pip install -r BotMoE/requirements.txt -r polarization_analysis/requirements.txt -r network_analysis/requirements.txt
```

On a CUDA machine use `BotMoE/requirements_cuda.txt` instead of `BotMoE/requirements.txt`. On Apple Silicon, the CPU
requirements (PyTorch 1.10) need a Rosetta environment; see [`BotMoE/README.md`](BotMoE/README.md#step-0-create-environment).
The full-scale polarization notebook needs an NVIDIA GPU with [RAPIDS](https://docs.rapids.ai/install).

## Data

**We do not redistribute tweets or user profiles.** You need to download them yourself:

- **2024 US election tweets**: Balasubramanian et al., *A Public Dataset Tracking Social Media Discourse about the 2024 U.S.
  Presidential Election on Twitter/X* ([arXiv:2411.00376](https://arxiv.org/abs/2411.00376)). Follow their instructions to download
  and hydrate the tweets. We used May–November 2024.
- **TwiBot-20** (training only): see [`twibot_20/Twibot-20/README.md`](twibot_20/Twibot-20/README.md).

## Running the pipeline

### 1. Election data → `node.json` / `edge.csv`

```bash
cd preprocessing
python ../utils/unzip_all_files.py /path/to/raw-data        # extract the .csv.gz files
python clean_data.py --input-dir /path/to/raw-data --output-dir data/
```

`node.json` has one entry per user (profile fields plus their posts). `edge.csv` has `reply` (user → user), `retweet` and `post`
(user → tweet) edges.

### 2. `node.json` / `edge.csv` → feature tensors

```bash
python preprocess.py --node data/node.json --edge data/edge.csv --output-dir ../BotMoE/data/US-24/processed_data/
```

This writes the graph (`edge_index.pt`, `edge_type.pt`), numeric and categorical profile features, RoBERTa embeddings of each
user's description and first 20 tweets, and `uid_to_user_index.pkl`, which maps each user ID to its row. The embeddings run on
a GPU if one is available. Tweet embedding is checkpointed every 20,000 users (`--checkpoint-every`), so an interrupted run
resumes where it stopped. Use `--skip-embeddings` to build only the graph and profile features.

### 3. Train BotMoE on TwiBot-20

This repository doesn't include trained checkpoints, so you need to train your own. Put the TwiBot-20 tensors in
`BotMoE/data/Twibot-20/processed_data/`: build them with `python twibot_20/preprocess.py`
(then copy `twibot_20/processed_data/*` over), or download the authors' preprocessed files (see the README in that folder).
Then run:

```bash
cd BotMoE
python main.py
```

Checkpoints with validation accuracy above 0.866 are saved to `BotMoE/artifacts/models/`. We trained five models with
different random seeds on our own preprocessed TwiBot-20, following BotMoE's original settings. They reached a mean accuracy
of 87.38% and a mean F1 of 89.11% (BotMoE reports 87.76% / 89.22%).

### 4. Label the election accounts

```bash
cd BotMoE
python test.py --model artifacts/models/<checkpoint>.pth --dataset US-24 --output artifacts/predictions/test_predictions.csv
```

The output has one row per user: `user_id`, `prediction` (1 = bot, 0 = human), the raw `logits`, and the softmax `probabilities`.
In the paper we ran all five models and took a majority vote per user.

[`BotMoE/artifacts/predictions/test_predictions.csv`](BotMoE/artifacts/predictions/test_predictions.csv) contains example
predictions for a 32,477-user sample of the dataset, so the analysis code can be tried without training. They come from a
single earlier checkpoint (validation accuracy 87.36%), not from the five-model vote used in the paper.

### 5. Polarization analysis

- [`polarization.ipynb`](polarization_analysis/polarization.ipynb): the full-dataset analysis from the paper, using cuGraph
  (GPU) for Louvain and Leiden and Graphistry for visualizations. Set `PREDICTIONS` and `DATA_DIR` in the first cell, and put your
  Graphistry key in the `GRAPHISTRY_KEY_ID` / `GRAPHISTRY_SECRET` environment variables.
- [`polarization_sample.ipynb`](polarization_analysis/polarization_sample.ipynb): the same before/after comparison with
  NetworkX and python-louvain. It runs on a laptop with the sample data in `preprocessing/sample-data/`.

### 6. Network analysis

```bash
cd network_analysis
python analysis.py --labels ../BotMoE/artifacts/predictions/test_predictions.csv --edges ../preprocessing/data/edge.csv
```

This writes `results.json` (bot and human counts, average PageRank, the top 100 bots by PageRank, average human targets per bot,
and the 10 most targeted humans) and the bot → human edges of bots with more than five human targets. Add `--plot` to upload
the bot → human graph to Graphistry. This needs `KEY_ID` and `SECRET_KEY` in the environment or a `.env` file.

## Known issues

- **Screen-name length is always missing.** `clean_data.py` reads `username` from a top-level CSV column that doesn't exist in
  the raw data, so that feature is NaN for every user and `test.py` replaces it with 0. We left it unchanged so the code reproduces
  the paper. To fix it, read `username` from the `user` dictionary instead.
- **Reply-target IDs are rounded in the raw data.** The dataset stores them as floats (e.g. `1.4835964632212644e+18`). The
  pipeline matches users by comparing IDs as floats, which works as long as no two users' IDs round to the same value.
- **Inference is not fully deterministic.** `NeighborLoader` samples at most 256 neighbors per node, so logits for very
  well-connected users can change slightly between runs.

## Citation

```bibtex
@unpublished{abdelhafeez2025bots,
  title  = {Bots and the Ballot Box: How Automated Accounts Shaped Narratives and Networks in the 2024 US Election},
  author = {Abdelhafeez, Yasmeen and Boyko, Yaroslava and Ito, Masaki},
  year   = {2025},
  note   = {Working paper, Georgia Institute of Technology}
}
```

This work builds on BotMoE (Liu et al., SIGIR 2023) and TwiBot-20 (Feng et al., CIKM 2021). Please cite them as well.

## Acknowledgments

Thanks to the instructional team of Georgia Tech's OMSCS CS 8803-O23 *Modern Internet Research Methods* course, who supervised
the first draft of this paper.
