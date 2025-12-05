
import json
import re
import urllib.parse
import unicodedata

INPUT = "../wikiextractor/articles_fr_withLinks.json"

TARGET_ARTICLES = 5
MIN_LINKS = 5

# ----------------------
# Helpers
# ----------------------

def normalize_title(t):
    """Convert an href or title into the canonical Wikipedia title format."""
    if not t:
        return ""

    # Decode %xx URL-encoding
    t = urllib.parse.unquote(t)

    # Remove fragment (#section)
    t = t.split("#")[0]

    # Remove leading /wiki/ if present
    if t.startswith("/wiki/"):
        t = t[len("/wiki/"):]

    # Spaces → underscores (Wikipedia uses underscores)
    t = t.replace(" ", "_")

    # Normalize unicode so accents compare properly
    t = unicodedata.normalize("NFC", t)

    return t


def extract_link_titles(text):
    """Extract canonical Wikipedia titles from href attributes."""
    hrefs = re.findall(r'href="([^"]+)"', text)
    titles = []

    for h in hrefs:
        t = normalize_title(h)
        if t:
            titles.append(t)

    return titles


# ----------------------
# Main logic
# ----------------------

selected = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Try reading the JSON
        try:
            article = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw_title = article.get("title", "")
        title = normalize_title(raw_title)
        text = article.get("text", "")

        links = extract_link_titles(text)

        print(f"Article: {title}   Links found: {len(links)}")

        if len(links) >= MIN_LINKS:
            selected.append({
                "title": title,
                "links": links[:MIN_LINKS]   # take only first N
            })

        if len(selected) >= TARGET_ARTICLES:
            break

print("Done!")
print("Selected articles:")
for a in selected:
    print(" -", a["title"], len(a["links"]), "links")


with open("small_articles.json", "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)
