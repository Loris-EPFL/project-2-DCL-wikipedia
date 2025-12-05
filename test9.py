#!/usr/bin/env python3
import json
import numpy as np
import csv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

SMALL_FILE = "small_articles_2.json"
LINKED_FILE = "linked_articles_2.json"
CHUNK_SIZE = 256
TOP_K = 5
CSV_OUT = "small_linked_similarity.csv"
COLL = "wikipedia_small"

# ------------------------
# Load articles
# ------------------------
with open(SMALL_FILE, "r") as f:
    small = json.load(f)
with open(LINKED_FILE, "r") as f:
    linked = json.load(f)

articles = {}
next_id = 1
for a in small + linked:
    if "id" in a:
        try:
            aid = int(a["id"])
        except:
            aid = next_id
    else:
        aid = next_id

    while aid in articles:
        aid += 1

    articles[aid] = a
    next_id = aid + 1

print(f"Loaded {len(articles)} articles.")

# ------------------------
# Encoder
# ------------------------
model = SentenceTransformer("models/multilingual-e5-large")
dim = model.get_sentence_embedding_dimension()
print("Embedding dimension:", dim)

def encode(text):
    words = text.split()
    chunks = [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
    if not chunks:
        chunks = [""]

    prefixed = [f"passage: {c}" for c in chunks]
    vecs = model.encode(prefixed, normalize_embeddings=True)
    pooled = np.max(vecs, axis=0)
    return pooled.astype(np.float32)

# ------------------------
# Encode all
# ------------------------
vecs = {aid: encode(a.get("text", "")) for aid, a in articles.items()}
print("Encoded all articles.\n")

# Debug (first 3)
for i, aid in enumerate(vecs):
    print(aid, articles[aid].get("title"), vecs[aid][:6], "norm=", np.linalg.norm(vecs[aid]))
    if i == 2:
        break

# ------------------------
# Setup Qdrant
# ------------------------
client = QdrantClient(":memory:")

try:
    client.delete_collection(COLL)
except:
    pass

client.create_collection(
    collection_name=COLL,
    vectors_config=VectorParams(size=dim, distance="Cosine")
)

points = [
    PointStruct(
        id=aid,
        vector=vec.tolist(),
        payload={"title": articles[aid].get("title", "")}
    )
    for aid, vec in vecs.items()
]

if points:
    client.upsert(collection_name=COLL, points=points, wait=True)
    points.clear()
    print(f"Uploaded {len(points)} vectors to Qdrant.\n")

# ------------------------
# SEARCH USING MODERN API
# ------------------------
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source_id", "source_title", "target_id", "target_title", "score"])

    for aid, article in articles.items():
        qvec = vecs[aid].tolist()

        # ✔️ modern qdrant-client syntax
        results = client.search(
            collection_name=COLL,
            query_vector=qvec,
            limit=TOP_K + 1,
            with_payload=True
        )

        # remove self
        neighbors = [r for r in results if r.id != aid][:TOP_K]

        for r in neighbors:
            w.writerow([
                aid,
                article.get("title", ""),
                r.id,
                r.payload.get("title", ""),
                f"{r.score:.6f}"
            ])

print("✅ CSV saved:", CSV_OUT)
