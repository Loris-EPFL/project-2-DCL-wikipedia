import json
import pandas as pd

# ----------------------------
# Load JSON
# ----------------------------
with open("small_articles_2.json", "r", encoding="utf-8") as f:
    small_articles = json.load(f)

# Map: title -> set of linked titles
link_map = {a["title"]: set(a["links"]) for a in small_articles}

# ----------------------------
# Load similarity CSV
# ----------------------------
sim_df = pd.read_csv("small_linked_similarity2.csv")

# ----------------------------
# Evaluate
# ----------------------------
def precision_at_k(sim_df, link_map, k=5):
    precisions = []
    for title, group in sim_df.groupby("source_title"):
        top_k = group.sort_values("score", ascending=False).head(k)
        target_titles = set(top_k["target_title"])
        true_links = link_map.get(title, set())
        if not true_links:
            continue
        # Count how many of the top-k are in true links
        num_correct = len(target_titles & true_links)
        precisions.append(num_correct / k)
    return sum(precisions) / len(precisions)

def recall_at_k(sim_df, link_map, k=5):
    recalls = []
    for title, group in sim_df.groupby("source_title"):
        top_k = group.sort_values("score", ascending=False).head(k)
        target_titles = set(top_k["target_title"])
        true_links = link_map.get(title, set())
        if not true_links:
            continue
        recalls.append(len(target_titles & true_links) / len(true_links))
    return sum(recalls) / len(recalls)

p_at_5 = precision_at_k(sim_df, link_map, k=5)
r_at_5 = recall_at_k(sim_df, link_map, k=5)

print(f"Precision@5: {p_at_5:.2f}")
print(f"Recall@5:    {r_at_5:.2f}")
