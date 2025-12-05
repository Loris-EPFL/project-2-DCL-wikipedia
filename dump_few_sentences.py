import json
import re
import urllib.parse
import unicodedata
from collections import defaultdict
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# ----------------------
# Config
# ----------------------
BIG_WIKI = "../wikiextractor/articles_fr_withLinks.json"
SMALL_OUTPUT = "small_articles_2.json"
LINKED_OUTPUT = "linked_articles_2.json"
SENTENCE_LINKS_OUTPUT = "sentence_links.json"
TARGET_ARTICLES = 5
MIN_VALID_LINKS = 5

# ----------------------
# Helpers
# ----------------------
def normalize_title(t):
    if not t:
        return ""
    t = urllib.parse.unquote(t)
    t = t.split("#")[0]
    if t.startswith("/wiki/"):
        t = t[len("/wiki/"):]
    t = t.replace(" ", "_")
    t = unicodedata.normalize("NFC", t)
    return t

def extract_link_titles(text):
    hrefs = re.findall(r'href="([^"]+)"', text)
    titles = [normalize_title(h) for h in hrefs if h]
    return titles

# ----------------------
# Load full dataset
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
# Select main articles
# ----------------------
selected = []
for title, article in all_articles.items():
    text = article.get("text", "")
    links = extract_link_titles(text)
    valid_links = [l for l in links if l in all_articles]

    if len(valid_links) >= MIN_VALID_LINKS:
        selected.append({
            "title": title,
            "links": valid_links,
            "text": text
        })
    
    if len(selected) >= TARGET_ARTICLES:
        break

with open(SMALL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)
print(f"Saved {len(selected)} main articles to {SMALL_OUTPUT}.")

# ----------------------
# Save linked articles
# ----------------------
linked_titles = set()
for article in selected:
    linked_titles.update(article["links"])

linked_articles = [all_articles[t] for t in linked_titles if t in all_articles]

with open(LINKED_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(linked_articles, f, ensure_ascii=False, indent=2)
print(f"Saved {len(linked_articles)} linked articles to {LINKED_OUTPUT}.")

# ----------------------
# Extract sentence-level links
# ----------------------
sentence_links = []
for article in selected:
    source_title = article["title"]
    text = article["text"]
    sentences = sent_tokenize(text, language="french")
    valid_links = set(article["links"])
    
    for sent in sentences:
        sent_links = extract_link_titles(sent)
        for target in sent_links:
            if target in valid_links:
                sentence_links.append({
                    "source_title": source_title,
                    "sentence": sent,
                    "target_title": target
                })

                # ----------------------
# Extract sentence-level links (cleaned)
# ----------------------
import html

def remove_links(text: str) -> str:
    """
    Remove all HTML links from the text, keeping only the visible text.
    Handles escaped HTML like &lt;a href="..."&gt;text&lt;/a&gt;.
    """
    # 1. Unescape HTML
    text = html.unescape(text)

    # 2. Replace <a href="...">text</a> with just "text"
    text = re.sub(r'<a [^>]*href="[^"]+"[^>]*>(.*?)</a>', r'\1', text)

    # 3. Remove any remaining HTML tags (optional, just in case)
    text = re.sub(r'<[^>]+>', '', text)

    # 4. Clean extra whitespace
    text = " ".join(text.split())

    return text


sentence_links = []
for article in selected:
    source_title = article["title"]
    text = article["text"]
    sentences = sent_tokenize(text, language="french")
    valid_links = set(article["links"])
    
    for sent in sentences:
        sent_links = extract_link_titles(sent)
        for target in sent_links:
            if target in valid_links:
                clean_sent = remove_links(sent)
                sentence_links.append({
                    "source_title": source_title,
                    "sentence": clean_sent,
                    "target_title": target
                })


with open(SENTENCE_LINKS_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(sentence_links, f, ensure_ascii=False, indent=2)
print(f"Saved {len(sentence_links)} cleaned sentence-level links to {SENTENCE_LINKS_OUTPUT}.")

