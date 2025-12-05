import json
import re
import urllib.parse
import random

# ---------------------------
# Load dataset
# ---------------------------
with open("../wikiextractor/articles_fr_withLinks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to dict: {title -> article}
articles = {item["title"]: item["text"] for item in data}

all_titles = list(articles.keys())
title_set = set(all_titles)

# ---------------------------
# Extract internal links
# ---------------------------
def extract_links(text):
    """
    Extracts wikipedia targets from <a href="TARGET"> forms.
    Returns decoded titles.
    """
    hrefs = re.findall(r'href="([^"]+)"', text)
    decoded = [urllib.parse.unquote(h) for h in hrefs]
    return decoded

# ---------------------------
# Build related (positive) link pairs
# ---------------------------
positive_pairs = set()

for title, text in articles.items():
    links = extract_links(text)
    for target in links:
        if target in title_set:  # only keep links pointing to an existing article
            positive_pairs.add((title, target))

positive_pairs = list(positive_pairs)

print(f"Found {len(positive_pairs)} related article pairs.")

# ---------------------------
# Build unrelated (negative) link pairs
# ---------------------------
negative_pairs = set()

while len(negative_pairs) < len(positive_pairs):
    a = random.choice(all_titles)
    b = random.choice(all_titles)
    if a != b and (a, b) not in positive_pairs:
        negative_pairs.add((a, b))

negative_pairs = list(negative_pairs)

print(f"Generated {len(negative_pairs)} unrelated article pairs.")

# ---------------------------
# Save small dataset for experiments
# ---------------------------
output = {
    "positive_pairs": positive_pairs,
    "negative_pairs": negative_pairs
}

with open("wiki_pairs_small.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Saved wiki_pairs_small.json")
