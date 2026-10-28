"""PageRank and bot -> human interaction analysis of the labelled interaction graph (paper, Section 5.3).

Inputs:  the predictions CSV from BotMoE/test.py and edge.csv from preprocessing/clean_data.py
Outputs (in --output-dir):
    results.json                           bot/human counts, average PageRank, top-100 bots by PageRank,
                                           average human targets per bot, 10 most targeted humans
    bot_to_human_interactions_multiple.csv bot -> human edges from bots with more than 5 human targets

With --plot, the bot -> human graph is uploaded to Graphistry (hub.graphistry.com). Put your
Graphistry personal key in a .env file or the environment as KEY_ID and SECRET_KEY.
"""

import argparse
import csv
import json
import os
from collections import Counter

import networkx as nx
import pandas as pd
from tqdm import tqdm


def id_key(value):
    """The raw dataset stores reply-target IDs as floats, so IDs are matched as floats."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def setup_graph(labels_path, edges_path):
    print("Loading labels...")
    df_labels = pd.read_csv(labels_path, dtype={"user_id": str})
    df_labels["label"] = df_labels["prediction"].map({0: "human", 1: "bot"})
    df_labels["key"] = df_labels["user_id"].map(id_key)
    print(f"Labels loaded: {len(df_labels)} rows")

    label_counts = df_labels["label"].value_counts()
    print(f"Number of humans: {label_counts.get('human', 0)}")
    print(f"Number of bots: {label_counts.get('bot', 0)}")
    print(f"Percentage of bots in labels: {label_counts.get('bot', 0) / label_counts.sum() * 100:.2f}%")

    print("Loading edges...")
    df_edges = pd.read_csv(edges_path, dtype={"source_id": str, "target_id": str})
    df_edges["source"] = df_edges["source_id"].map(id_key)
    df_edges["target"] = df_edges["target_id"].map(id_key)
    print(f"Edges loaded: {len(df_edges)} rows")

    print("Building graph...")
    G = nx.from_pandas_edgelist(df_edges, source="source", target="target", create_using=nx.DiGraph())
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    print("Assigning labels to graph nodes...")
    nx.set_node_attributes(G, df_labels.set_index("key")["label"].to_dict(), name="label")

    # Report exact IDs (where we have them) rather than the float keys.
    exact_id = df_labels.set_index("key")["user_id"].to_dict()
    return G, exact_id


def nodes_with_label(G, label):
    nodes = [n for n in G.nodes if G.nodes[n].get("label") == label]
    print(f"Total {label}s: {len(nodes)}")
    return nodes


def calculate_page_rank(G, bots, humans, exact_id, results):
    print("Calculating PageRank...")
    pagerank = nx.pagerank(G)
    nx.set_node_attributes(G, pagerank, name="pagerank")

    top_bots = sorted(bots, key=lambda n: pagerank.get(n, 0), reverse=True)[:100]
    results["top_bots"] = [exact_id.get(n, n) for n in top_bots]

    avg_bot = sum(pagerank[n] for n in bots) / len(bots) if bots else 0
    avg_human = sum(pagerank[n] for n in humans) / len(humans) if humans else 0
    print(f"Average bot PageRank: {avg_bot}")
    print(f"Average human PageRank: {avg_human}")
    results["avg_bot_pagerank"] = avg_bot
    results["avg_human_pagerank"] = avg_human


def bot_analysis(G, exact_id, results, output_dir, plot):
    print("Finding bot to human interactions...")
    bot_human_edges = [(source, target) for source, target in tqdm(G.edges(), desc="Processing edges")
                       if G.nodes[source].get("label") == "bot" and G.nodes[target].get("label") == "human"]

    source_counts = Counter(source for source, _ in bot_human_edges)
    average_targets_per_bot = len(bot_human_edges) / len(source_counts) if source_counts else 0
    print(f"Average number of human targets per bot: {average_targets_per_bot:.2f}")
    results["avg_human_targets_per_bot"] = average_targets_per_bot

    filtered_bot_edges = [(s, t) for s, t in bot_human_edges if source_counts[s] > 5]
    csv_path = os.path.join(output_dir, "bot_to_human_interactions_multiple.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows((exact_id.get(s, s), exact_id.get(t, t)) for s, t in filtered_bot_edges)
    print(f"Saved {len(filtered_bot_edges)} bot→human edges to {csv_path}")

    most_targeted_humans = Counter(t for _, t in bot_human_edges).most_common(10)
    print(f"Most targeted humans: {most_targeted_humans}")
    results["most_targeted_humans"] = [(exact_id.get(n, n), count) for n, count in most_targeted_humans]

    if plot:
        create_graphistry_graph(bot_human_edges)


def create_graphistry_graph(bot_human_edges):
    import graphistry
    from dotenv import load_dotenv

    load_dotenv()
    df_edges = pd.DataFrame(bot_human_edges, columns=['source', 'target'])
    bot_nodes = set(df_edges['source'])
    df_nodes = pd.DataFrame([
        {'node_id': node, 'label': 'bot' if node in bot_nodes else 'human',
         'color': '#FF0000' if node in bot_nodes else '#0617FF'}
        for node in set(df_edges['source']) | set(df_edges['target'])
    ])

    graphistry.register(api=3, protocol="https", server="hub.graphistry.com",
                        personal_key_id=os.getenv("KEY_ID"), personal_key_secret=os.getenv("SECRET_KEY"))
    graphistry.bind(source='source', destination='target', node='node_id', point_color='color') \
        .nodes(df_nodes).edges(df_edges).plot()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, help="predictions CSV from BotMoE/test.py")
    parser.add_argument("--edges", required=True, help="edge.csv from preprocessing/clean_data.py")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--plot", action="store_true", help="upload the bot -> human graph to Graphistry")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    G, exact_id = setup_graph(args.labels, args.edges)
    bots = nodes_with_label(G, "bot")
    humans = nodes_with_label(G, "human")
    results["num_bots"] = len(bots)
    results["num_humans"] = len(humans)

    calculate_page_rank(G, bots, humans, exact_id, results)
    bot_analysis(G, exact_id, results, args.output_dir, args.plot)

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
