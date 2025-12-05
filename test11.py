#!/usr/bin/env python3
"""
fast_similarity_fixed.py

Parallel embedding (ThreadPool), batch upsert to Qdrant, search, write CSV.
Designed to be robust across environments and avoid Polars dtype issues.
"""
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
SMALL_FILE = "small_articles_2.json"
LINKED_FILE = "linked_articles_2.json"
CSV_OUT = "small_linked_similarity.csv"
COLLECTION = "wikipedia_small"
CHUNK_SIZE = 256
TOP_K = 5
UPSERT_BATCH = 128
EMB_MODEL_PATH = "models/multilingual-e5-large"  # or "intfloat/multilingual-e5-large"
MAX_WORKERS = 6  # adjust for your machine (CPU cores)
# ----------------------------

# Load data
with open(SMALL_FILE, "r", encoding="utf-8") as f:
    small = json.load(f)
with open(LINKED_FILE, "r", encoding="utf-8") as f:
    linked = json.load(f)

all_articles_list = small + linked

# Normalize and assign integer IDs (avoid collisions)
articles: Dict[int, dict] = {}
next_id = 1
for a in all_articles_list:
    if "id" in a:
        try:
            aid = int(a["id"])
        except Exception:
            aid = next_id
    else:
        aid = next_id
    while aid in articles:
        aid += 1
    articles[aid] = a
    next_id = aid + 1

print(f"Loaded {len(articles)} articles.")

# Load model
print("Loading sentence-transformers model...")
model = SentenceTransformer(EMB_MODEL_PATH, device="cpu")
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dim = {embedding_dim}")

# Encoder helper (chunk, encode, mean-pool, normalize)
def encode_text_to_vector(text: str, chunk_size_words: int = CHUNK_SIZE) -> np.ndarray:
    if text is None:
        text = ""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
    if not chunks:
        chunks = [""]
    prefixed = [f"passage: {c}" for c in chunks]
    vecs = model.encode(prefixed, normalize_embeddings=False)  # get raw vectors
    # mean pool, then normalize once
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = (mean_vec / norm).astype(np.float32)
    else:
        mean_vec = mean_vec.astype(np.float32)
    return mean_vec

# Parallel encode all articles
print("Encoding articles in parallel...")
vectors: Dict[int, np.ndarray] = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {}
    for aid, art in articles.items():
        text = art.get("text") or art.get("title") or ""
        futures[ex.submit(encode_text_to_vector, text)] = aid

    for fut in tqdm(as_completed(futures), total=len(futures), desc="Encoding"):
        aid = futures[fut]
        try:
            vec = fut.result()
        except Exception as e:
            print(f"Error encoding article id={aid}: {e}")
            vec = np.zeros(embedding_dim, dtype=np.float32)
        vectors[aid] = vec

print("Encoding complete.\n")

# Quick debug of first 3 vectors
some_ids = list(vectors.keys())[:3]
print("--- debug vectors ---")
for aid in some_ids:
    print(aid, articles[aid].get("title"), vectors[aid][:6], "norm=", np.linalg.norm(vectors[aid]))
print("---------------------\n")

# Setup Qdrant in-memory
client = QdrantClient(":memory:")

# Recreate collection safely
try:
    client.delete_collection(collection_name=COLLECTION)
except Exception:
    pass

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=embedding_dim, distance="Cosine")
)
print("Qdrant collection created.")

# Upsert in batches
print("Upserting vectors to Qdrant in batches...")
buffer: List[PointStruct] = []
count = 0
for aid, vec in tqdm(vectors.items(), desc="Preparing points"):
    payload = {"title": articles[aid].get("title", "")}
    buffer.append(PointStruct(id=int(aid), vector=vec.tolist(), payload=payload))
    if len(buffer) >= UPSERT_BATCH:
        client.upsert(collection_name=COLLECTION, points=buffer, wait=True)
        count += len(buffer)
        buffer.clear()
# flush
if buffer:
    client.upsert(collection_name=COLLECTION, points=buffer, wait=True)
    count += len(buffer)
print(f"Upserted {count} points to Qdrant.\n")

# Use client.search to get top-k (modern qdrant-client)
print("Searching nearest neighbors and writing CSV...")
with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "source_title", "target_id", "target_title", "score"])

    for aid, vec in tqdm(vectors.items(), desc="Searching"):
        results = client.search(
            collection_name=COLLECTION,
            query_vector=vec.tolist(),
            limit=TOP_K + 1,
            with_payload=True
        )
        # filter self and keep top K
        neighbors = [r for r in results if r.id != aid][:TOP_K]
        for r in neighbors:
            writer.writerow([
                aid,
                articles[aid].get("title", ""),
                r.id,
                r.payload.get("title", ""),
                f"{r.score:.6f}"
            ])

print("✅ Done. CSV written to", CSV_OUT)
