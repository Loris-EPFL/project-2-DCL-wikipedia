import json
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config
# ----------------------------
SMALL_FILE = "small_articles.json"
LINKED_FILE = "linked_articles.json"
COLLECTION_NAME = "wikipedia_fr_small"
CHUNK_SIZE_WORDS = 256

# ----------------------------
# Load data
# ----------------------------
with open(SMALL_FILE, "r", encoding="utf-8") as f:
    small_articles = json.load(f)

with open(LINKED_FILE, "r", encoding="utf-8") as f:
    linked_articles = json.load(f)

# Combine all articles into a single dict by id (or generate new IDs if missing)
all_articles = {}
counter = 1
for article in small_articles + linked_articles:
    if "id" in article:
        article_id = int(article["id"])
    else:
        article_id = counter
        counter += 1
    all_articles[article_id] = article

print(f"Loaded {len(all_articles)} articles from both JSON files.")

# ----------------------------
# Load model
# ----------------------------
model = SentenceTransformer("models/multilingual-e5-large")
vector_size = model.get_sentence_embedding_dimension()

# ----------------------------
# Connect to Qdrant
# ----------------------------
client = QdrantClient(":memory:")  # in-memory
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance="Cosine")
)

# ----------------------------
# Helper functions
# ----------------------------
def encode_article_text(text, chunk_size_words=CHUNK_SIZE_WORDS):
    """Encode an article by chunking and mean pooling embeddings."""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
    if not chunks:
        chunks = [""]
    prefixed = [f"passage: {chunk}" for chunk in chunks]
    vecs = model.encode(prefixed, normalize_embeddings=True)
    return np.mean(vecs, axis=0)

# ----------------------------
# Ingest into Qdrant
# ----------------------------
points = []
for article_id, article in all_articles.items():
    text = article.get("text", "")
    vec = encode_article_text(text)
    points.append(
        PointStruct(
            id=int(article_id),
            vector=vec,
            payload={
                "id": int(article_id),
                "title": article.get("title", ""),
                "source": "small" if article in small_articles else "linked"
            }
        )
    )

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Upserted {len(points)} points into Qdrant.")

# ----------------------------
# Compute top-k similarity (article-to-article)
# ----------------------------
from qdrant_client.http import models as rest
import csv

def get_similar_articles(client, collection_name, source_id, k=5):
    """Find top-k nearest neighbors for a given article id."""
    results = client.recommend(collection_name=collection_name, positive=[source_id], limit=k)
    return results

# Export similarity links to CSV
CSV_OUT = "small_linked_similarity.csv"
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "source_title", "target_id", "target_title", "score"])
    for article_id, article in all_articles.items():
        source_title = article.get("title", "")
        neighbors = get_similar_articles(client, COLLECTION_NAME, article_id, k=5)
        for n in neighbors:
            target_id = n.payload.get("id", n.id)
            target_title = n.payload.get("title", "(unknown)")
            writer.writerow([article_id, source_title, target_id, target_title, f"{n.score:.6f}"])

print(f"✅ Similarity graph saved to {CSV_OUT}")
