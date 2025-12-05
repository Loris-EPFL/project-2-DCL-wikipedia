#!/usr/bin/env python3
"""
fast_similarity_gpu.py

Compute top-K similarities between main articles and all articles using GPU embeddings.
"""
import json
import numpy as np
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
SMALL_FILE = "small_articles_2.json"    # main articles
LINKED_FILE = "linked_articles_2.json"  # target corpus
CSV_OUT = "small_linked_similarity2.csv"
CHUNK_SIZE = 256
TOP_K = 100
BATCH_SIZE = 32          # GPU batch size for encoding
MODEL_PATH = "models/multilingual-e5-large"
# ----------------------------

print("📥 Loading JSON files...")
with open(SMALL_FILE, "r", encoding="utf-8") as f:
    small_articles = json.load(f)
with open(LINKED_FILE, "r", encoding="utf-8") as f:
    linked_articles = json.load(f)

# Combine into all articles dict (id -> article), assign integer IDs
articles = {}
next_id = 1
for art in small_articles + linked_articles:
    if "id" in art:
        try:
            aid = int(art["id"])
        except:
            aid = next_id
    else:
        aid = next_id
    while aid in articles:
        aid += 1
    articles[aid] = art
    next_id = aid + 1

# Keep track of main article IDs
main_ids = list(range(1, len(small_articles)+1))
print(f"Loaded {len(articles)} articles ({len(main_ids)} main).")

# ---------- Load model ----------
print("🧠 Loading model on GPU...")
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
model = SentenceTransformer(MODEL_PATH, device=device)

embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dim = {embedding_dim}")

# ---------- Helper: encode text to vector ----------
def encode_text(text: str) -> np.ndarray:
    if not text:
        text = ""
    words = text.split()
    chunks = [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
    if not chunks:
        chunks = [""]
    prefixed = [f"passage: {c}" for c in chunks]
    vecs = model.encode(prefixed, normalize_embeddings=False)
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = (mean_vec / norm).astype(np.float32)
    else:
        mean_vec = mean_vec.astype(np.float32)
    return mean_vec

# ---------- Encode all articles in batches ----------
print("⚡ Encoding all articles on GPU...")
all_ids = list(articles.keys())
all_texts = [articles[i].get("text", "") or articles[i].get("title", "") for i in all_ids]

vectors = {}
for start in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Batches"):
    batch_texts = all_texts[start:start+BATCH_SIZE]
    batch_ids = all_ids[start:start+BATCH_SIZE]
    batch_vecs = model.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=False)
    # normalize vectors
    batch_vecs = batch_vecs.astype(np.float32)
    norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
    batch_vecs = batch_vecs / np.maximum(norms, 1e-8)
    for aid, vec in zip(batch_ids, batch_vecs):
        vectors[aid] = vec

print("✅ Encoding complete.\n")

# ---------- Compute similarities ----------
print("📝 Computing top-K similarities...")
# Prepare arrays for vectorized computation
all_ids_array = np.array(all_ids)
all_vectors = np.stack([vectors[i] for i in all_ids])
main_vectors = np.stack([vectors[i] for i in main_ids])

with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "source_title", "target_id", "target_title", "score"])

    for idx, aid in enumerate(tqdm(main_ids, desc="Main articles")):
        src_vec = main_vectors[idx:idx+1]  # shape (1, dim)
        # cosine similarity: dot product since vectors are normalized
        sims = np.dot(all_vectors, src_vec.T).squeeze()  # shape (len(all),)
        # Get top-K (excluding self if present)
        top_idx = np.argsort(-sims)  # descending
        top_ids = []
        top_scores = []
        for tidx in top_idx:
            target_id = all_ids_array[tidx]
            if target_id == aid:
                continue
            top_ids.append(target_id)
            top_scores.append(sims[tidx])
            if len(top_ids) >= TOP_K:
                break
        # write rows
        src_title = articles[aid].get("title", "")
        for t_id, score in zip(top_ids, top_scores):
            tgt_title = articles[t_id].get("title", "")
            writer.writerow([aid, src_title, t_id, tgt_title, f"{score:.6f}"])

print(f"✅ CSV saved to {CSV_OUT}")
