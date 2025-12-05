import json
import re
import urllib.parse
import unicodedata
from collections import defaultdict

# ----------------------
# Config
# ----------------------
BIG_WIKI = "../wikiextractor/articles_fr_withLinks.json"  # JSONL full Wikipedia dump
SMALL_OUTPUT = "small_articles_2.json"
LINKED_OUTPUT = "linked_articles_2.json"
TARGET_ARTICLES = 5
MIN_VALID_LINKS = 5
NB_ARTICLE_TO_SAVE = 20

# ----------------------
# Helpers
# ----------------------
def normalize_title(t):
    """Normalize Wikipedia titles to canonical format."""
    if not t:
        return ""
    t = urllib.parse.unquote(t)
    t = t.split("#")[0]  # remove fragments
    if t.startswith("/wiki/"):
        t = t[len("/wiki/"):]
    t = t.replace(" ", "_")
    t = unicodedata.normalize("NFC", t)
    return t

def extract_link_titles(text):
    """Extract normalized href links from article text."""
    hrefs = re.findall(r'href="([^"]+)"', text)
    titles = [normalize_title(h) for h in hrefs if h]
    return titles

# ----------------------
# Load full dataset into memory (titles -> article)
# ----------------------
print("Loading full dataset...")
all_articles = {}
with open(BIG_WIKI, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            article = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = normalize_title(article.get("title", ""))
        all_articles[title] = article
print(f"Loaded {len(all_articles)} articles.")

# ----------------------
# Find candidate articles
# ----------------------
selected = []

for title, article in all_articles.items():
    text = article.get("text", "")
    links = extract_link_titles(text)
    
    # Keep only links that exist in the dataset
    valid_links = [l for l in links if l in all_articles]
    
    if len(valid_links) >= MIN_VALID_LINKS:
        selected.append({
            "title": title,
            "links": valid_links
        })
    
    if len(selected) >= TARGET_ARTICLES:
        break

# ----------------------
# Save small_articles.json
# ----------------------
with open(SMALL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)
print(f"Saved {len(selected)} articles to {SMALL_OUTPUT}.")

# ----------------------
# Save linked_articles.json
# ----------------------
linked_titles = set()
for article in selected:
    linked_titles.update(article["links"])

linked_articles = [all_articles[t] for t in linked_titles if t in all_articles]

with open(LINKED_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(linked_articles, f, ensure_ascii=False, indent=2)
print(f"Saved {len(linked_articles)} linked articles to {LINKED_OUTPUT}.")
