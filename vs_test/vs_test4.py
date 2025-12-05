import json
import re
import urllib.parse
import random

# ---------------------------
# Load dataset
# ---------------------------
with open("../wikiextractor/articles_fr_withLinks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to dict for easy lookup
articles = {item["title"]: item["text"] for item in data}
all_titles = list(articles.keys())
title_set = set(all_titles)

# ---------------------------
# Extract internal links from one article
# ---------------------------
def extract_links(text):
    hrefs = re.findall(r'href="([^"]+)"', text)
    return [urllib.parse.unquote(h) for h in hrefs]

# ---------------------------
# Find candidate articles with ≥5 links
# ---------------------------
candidate_articles = []

for title, text in articles.items():
    links = extract_links(text)
    valid_links = [l for l in links if l in title_set]
    if len(valid_links) >= 5:
        candidate_articles.append((title, valid_links))

print(f"Found {len(candidate_articles)} articles with ≥ 5 links.")

# ---------------------------
# Sample 5 articles
# ---------------------------
random.shuffle(candidate_articles)
selected = candidate_articles[:5]

print("Selected articles:")
for title, links in selected:
    print(f"  {title} ({len(links)} links)")

# ---------------------------
# Build 5 positive pairs per article
# ---------------------------
positive_pairs = []

for title, links in selected:
    sampled = random.sample(links, 5)   # pick 5 linked articles
    for target in sampled:
        positive_pairs.append((title, target))

print(f"\nTotal positive pairs: {len(positive_pairs)}")

# ---------------------------
# Build negative pairs
# For each positive link (A → B),
# generate one negative example (A → C) with C unrelated
# ---------------------------
negative_pairs = set()

for title, _ in positive_pairs:
    while True:
        neg = random.choice(all_titles)
        if neg != title and (title, neg) not in positive_pairs:
            negative_pairs.add((title, neg))
            break

negative_pairs = list(negative_pairs)
print(f"Total negative pairs: {len(negative_pairs)}")

# ---------------------------
# Save tiny dataset
# ---------------------------
output = {
    "selected_articles": [t for t, _ in selected],
    "positive_pairs": positive_pairs,
    "negative_pairs": negative_pairs
}

with open("wiki_pairs_SMALL.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\nSaved wiki_pairs_SMALL.json")
