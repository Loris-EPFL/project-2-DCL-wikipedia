#!/usr/bin/env python3
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import csv

# ----------------------------
# Config
# ----------------------------
SMALL_FILE = "small_articles.json"
LINKED_FILE = "linked_articles.json"
COLLECTION_NAME = "wikipedia_fr_small"
CHUNK_SIZE_WORDS = 256
TOP_K = 5
CSV_OUT = "small_linked_similarity.csv"

# ----------------------------
# Load data
# ----------------------------
with open(SMALL_FILE, "r", encoding="utf-8") as f:
    small_articles = json.load(f)

with open(LINKED_FILE, "r", encoding="utf-8") as f:
    linked_articles = json.load(f)

# Combine articles into dict
all_articles = {}
counter = 1
for article in small_articles + linked_articles:
    if "id" in article:
        try:
            article_id = int(article["id"])
        except Exception:
            article_id = counter
    else:
        article_id = counter

    while article_id in all_articles:
        article_id += 1
    counter = article_id + 1
    all_articles[article_id] = article

print(f"Loaded {len(all_articles)} articles.")

# ----------------------------
# Load embedding model
# ----------------------------
model = SentenceTransformer("models/multilingual-e5-large")
vector_size = model.get_sentence_embedding_dimension()
print(f"Model embedding dimension: {vector_size}")

# ----------------------------
# Helper: encode text
# ----------------------------
def encode_article_text(text, chunk_size_words=CHUNK_SIZE_WORDS):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
    if not chunks:
        chunks = [""]

    prefixed = [f"passage: {c}" for c in chunks]
    vecs = model.encode(prefixed, normalize_embeddings=True)
    pooled = np.max(vecs, axis=0)
    return pooled.astype(np.float32)

# ----------------------------
# Build vectors
# ----------------------------
article_vectors = {}
for aid, article in all_articles.items():
    text = article.get("text") or article.get("title") or ""
    article_vectors[aid] = encode_article_text(text)

print("Encoded all articles.")

# ----------------------------
# Debug first 3 vectors
# ----------------------------
sample_ids = list(all_articles.keys())[:3]
print("\n--- Debug vectors ---")
for aid in sample_ids:
    v = article_vectors[aid]
    print(aid, all_articles[aid].get("title"), v[:6], "norm=", np.linalg.norm(v))
print("-----------------------\n")

# ----------------------------
# Qdrant setup (in-memory)
# ----------------------------
client = QdrantClient(":memory:")

client.delete_collection(collection_name=COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance="Cosine")
)

# Upload
points = [
    PointStruct(
        id=int(aid),
        vector=vec.tolist(),
        payload={"title": all_articles[aid].get("title", "")}
    )
    for aid, vec in article_vectors.items()
]

client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Uploaded {len(points)} vectors to Qdrant.")

# ----------------------------
# Compute top-K similarities
# ----------------------------
with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "source_title", "target_id", "target_title", "score"])

    for aid, article in all_articles.items():
        query_vec = article_vectors[aid].tolist()

        # --- IMPORTANT: use query() instead of search() or search_points() ---
        results = client.query(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=TOP_K + 1
        )

        neighbors = [r for r in results if r.id != aid][:TOP_K]

        for r in neighbors:
            writer.writerow([
                aid,
                article.get("title", ""),
                r.id,
                r.payload.get("title", "(unknown)"),
                f"{r.score:.6f}"
            ])

print(f"✅ Saved similarity CSV to {CSV_OUT}")
