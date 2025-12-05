import json

SMALL_INPUT = "small_articles.json"          # The file you generated
BIG_WIKI = "../wikiextractor/articles_fr_withLinks.json"      # JSONL full Wikipedia dump
OUTPUT = "linked_articles.json"


def load_target_titles():
    """Load the titles of all linked articles we want to extract."""
    with open(SMALL_INPUT, "r", encoding="utf-8") as f:
        small = json.load(f)

    targets = set()
    for article in small:
        for link_title in article["links"]:
            targets.add(link_title)

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
            except:
                continue

            title = article.get("title", "")
            if title in target_titles:
                found[title] = article
                print(f"Found: {title} ({len(found)}/{len(target_titles)})")

                if len(found) == len(target_titles):
                    break

    return found


def main():
    target_titles = load_target_titles()
    print("Here are the target titles:")
    for title in target_titles:
        print(f" - {title}")
    found_articles = extract_articles_by_title(target_titles)

    print("\nExtraction complete.")
    print("We didnt find the following articles:" \
          f"{set(target_titles) - set(found_articles.keys())}")
    

    print(f"\nExtracted {len(found_articles)} articles.")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(list(found_articles.values()), f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
