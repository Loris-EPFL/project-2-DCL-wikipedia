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

# Combine articles into dict keyed by id (create ids if missing)
all_articles = {}
counter = 1
for article in small_articles + linked_articles:
    if "id" in article:
        try:
            article_id = int(article["id"])
        except Exception:
            article_id = counter
            counter += 1
    else:
        article_id = counter
        counter += 1
    while article_id in all_articles:
        article_id = counter
        counter += 1
    all_articles[article_id] = article

print(f"Loaded {len(all_articles)} articles.")

# ----------------------------
# Load model
# ----------------------------
model = SentenceTransformer("models/multilingual-e5-large")
vector_size = model.get_sentence_embedding_dimension()
print(f"Model embedding dimension: {vector_size}")

# ----------------------------
# Helper functions
# ----------------------------
def encode_article_text(text, chunk_size_words=CHUNK_SIZE_WORDS):
    """Encode an article by chunking and max pooling embeddings."""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
    if not chunks:
        chunks = [""]  # empty chunk fallback
    prefixed = [f"passage: {chunk}" for chunk in chunks]
    vecs = model.encode(prefixed, normalize_embeddings=True)
    pooled = np.max(vecs, axis=0)
    return np.asarray(pooled, dtype=np.float32)

def cosine(a, b):
    """Compute cosine similarity between two vectors."""
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))

# ----------------------------
# Build vectors
# ----------------------------
article_vectors = {}
for article_id, article in all_articles.items():
    # fallback to title if text missing
    text = article.get("text") or article.get("title") or ""
    vec = encode_article_text(text)
    article_vectors[article_id] = vec

print("Encoded all articles (pooled vectors ready).")

# Debug: first 3 vectors
sample_ids = list(all_articles.keys())[:3]
print("\n--- Debug: first 3 article vector samples (first 6 dims) ---")
for aid in sample_ids:
    v = article_vectors[aid]
    print(f"id={aid} title={all_articles[aid].get('title','(no title)')}")
    print(" vec[:6] =", np.round(v[:6], 6), " norm=", np.linalg.norm(v))
if len(sample_ids) >= 3:
    c01 = cosine(article_vectors[sample_ids[0]], article_vectors[sample_ids[1]])
    c02 = cosine(article_vectors[sample_ids[0]], article_vectors[sample_ids[2]])
    c12 = cosine(article_vectors[sample_ids[1]], article_vectors[sample_ids[2]])
    print(f"\nPairwise cosines: {sample_ids[0]}-{sample_ids[1]}={c01:.6f}, "
          f"{sample_ids[0]}-{sample_ids[2]}={c02:.6f}, {sample_ids[1]}-{sample_ids[2]}={c12:.6f}")
print("--- end debug ---\n")

# ----------------------------
# Connect to Qdrant (in-memory)
# ----------------------------
client = QdrantClient(":memory:")

# Recreate collection
client.delete_collection(collection_name=COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance="Cosine")
)
print("Qdrant collection created.")

# ----------------------------
# Upload vectors
# ----------------------------
points = []
for article_id, vec in article_vectors.items():
    article = all_articles[article_id]
    payload = {"id": int(article_id), "title": article.get("title","")}
    points.append(PointStruct(id=int(article_id), vector=vec.tolist(), payload=payload))

BATCH = 128
for i in range(0, len(points), BATCH):
    client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+BATCH])
print(f"Upserted {len(points)} points into Qdrant.")

# ----------------------------
# Compute top-K similarity
# ----------------------------
with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "source_title", "target_id", "target_title", "score"])

    for article_id, article in all_articles.items():
        query_vec = article_vectors[article_id]

        results = client.search_points(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec.tolist(),
            limit=TOP_K + 1
        )

        neighbors = [n for n in results if n.id != article_id][:TOP_K]

        for n in neighbors:
            tgt_id = n.payload.get("id", n.id)
            tgt_title = n.payload.get("title", "(unknown)")
            writer.writerow([article_id, article.get("title",""), tgt_id, tgt_title, f"{n.score:.6f}"])

print(f"✅ Similarity graph saved to {CSV_OUT}")
