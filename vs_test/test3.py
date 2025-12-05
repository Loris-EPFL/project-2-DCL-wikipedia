import json
import urllib.parse
import unicodedata

SMALL_INPUT = "small_articles.json"          # The file with your selected articles
BIG_WIKI = "../wikiextractor/articles_fr_withLinks.json"  # Full JSONL Wikipedia dump
OUTPUT = "linked_articles.json"


# ----------------------
# Helpers
# ----------------------

def normalize_title(t):
    """Normalize Wikipedia titles to canonical format."""
    if not t:
        return ""
    # Decode URL encoding
    t = urllib.parse.unquote(t)
    # Remove fragment
    t = t.split("#")[0]
    # Remove /wiki/ prefix
    if t.startswith("/wiki/"):
        t = t[len("/wiki/"):]
    # Replace spaces with underscores
    t = t.replace(" ", "_")
    # Normalize unicode
    t = unicodedata.normalize("NFC", t)
    return t


def load_target_titles():
    """Load the titles of all linked articles we want to extract."""
    with open(SMALL_INPUT, "r", encoding="utf-8") as f:
        small = json.load(f)

    targets = set()
    for article in small:
        for link_title in article["links"]:
            normalized_link = normalize_title(link_title)
            targets.add(normalized_link)

    print(f"Need to retrieve {len(targets)} linked articles.")
    return targets


def extract_articles_by_title(target_titles):
    """Scan the large JSONL file and extract articles whose titles match."""
    found = {}

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
            if title in target_titles:
                found[title] = article
                print(f"Found: {title} ({len(found)}/{len(target_titles)})")

                # Stop early if we found everything
                if len(found) == len(target_titles):
                    break

    return found


# ----------------------
# Main
# ----------------------

def main():
    target_titles = load_target_titles()

    print("\nHere are the target titles:")
    for title in target_titles:
        print(f" - {title}")

    found_articles = extract_articles_by_title(target_titles)

    missing = set(target_titles) - set(found_articles.keys())
    if missing:
        print("\nWe didn’t find the following articles:")
        for title in missing:
            print(f" - {title}")
    else:
        print("\nAll target articles were found!")

    print(f"\nExtracted {len(found_articles)} articles.")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(list(found_articles.values()), f, ensure_ascii=False, indent=2)

    print(f"Saved extracted articles to {OUTPUT}")


if __name__ == "__main__":
    main()
