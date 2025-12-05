import json
import re
import urllib.parse

INPUT = "../wikiextractor/articles_fr_withLinks.json"

TARGET_ARTICLES = 5       # we want 5 source articles
MIN_LINKS = 5             # each must have >=5 links

def extract_links(text):
    hrefs = re.findall(r'href="([^"]+)"', text)
    return [urllib.parse.unquote(h) for h in hrefs]

selected = []

# ---- STREAMING LOAD (no full JSON in RAM) ----
with open(INPUT, "r", encoding="utf-8") as f:
    # Skip the leading '['
    f.read(1)

    buffer = ""
    depth = 0

    while True:
        c = f.read(1)
        if not c:
            break

        buffer += c

        if c == "{":
            depth += 1
        elif c == "}":
            print("DECR")
            depth -= 1
            if depth == 0:
                print("FULL OBJ")
                # we finished reading one JSON object
                obj_text = buffer.rstrip().rstrip(",")   # remove trailing comma
                buffer = ""

                print("PARSED OBJ:", obj_text)

                try:
                    article = json.loads(obj_text)
                except json.JSONDecodeError:
                    continue

                title = article.get("title", "")
                text = article.get("text", "")

                # Extract internal links
                links = extract_links(text)
                print(f"Article: {title}   Links found: {len(links)}")

                # Don't check if links exist in dataset — we only want quantity
                if len(links) >= MIN_LINKS:
                    selected.append((title, links))
                    print("Selected:", title, "with", len(links), "links")
                    break


                if len(selected) >= TARGET_ARTICLES:
                    break

# ---- STOP HERE IF NOT ENOUGH FOUND ----
if len(selected) < TARGET_ARTICLES:
    print(f"Only found {len(selected)} articles, stopping.")
else:
    print(f"Selected {len(selected)} articles:")
    for t, l in selected:
        print(f" - {t}   ({len(l)} links)")

# ---- SAVE RESULTS ----
out = [{"title": title, "links": links[:MIN_LINKS]} for title, links in selected]

with open("small_articles.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("Saved small_articles.json")
