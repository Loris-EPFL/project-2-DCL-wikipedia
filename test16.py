#!/usr/bin/env python3
"""
sentence_to_article_similarity.py

Compute top-K similarities between each sentence with a target corpus
(split into chunks) using GPU embeddings.
"""
import json
import numpy as np
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
SMALL_FILE = "small_articles_2.json"    # main articles
LINKED_FILE = "linked_articles_2.json"  # target corpus
SENTENCE_LINKS_FILE = "sentence_links.json"
CSV_OUT = "sentence_to_article_topk.csv"

CHUNK_SIZE = 256     # words per target article chunk
TOP_K = 10
BATCH_SIZE = 32      # GPU batch size
MODEL_PATH = "models/multilingual-e5-large"
# ----------------------------

# ---------- Load JSON files ----------
print("📥 Loading JSON files...")
with open(SMALL_FILE, "r", encoding="utf-8") as f:
    small_articles = json.load(f)
with open(LINKED_FILE, "r", encoding="utf-8") as f:
    linked_articles = json.load(f)
with open(SENTENCE_LINKS_FILE, "r", encoding="utf-8") as f:
    sentence_links = json.load(f)

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

print(f"Loaded {len(articles)} articles.")

# ---------- Load model ----------
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("🧠 Loading model on", device)
model = SentenceTransformer(MODEL_PATH, device=device)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dim = {embedding_dim}")

# ---------- Helper functions ----------
def encode_text(texts):
    """Encode list of texts to normalized vectors."""
    vecs = model.encode(texts, batch_size=len(texts), normalize_embeddings=False)
    vecs = vecs.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    return vecs

def chunk_text(text, chunk_size_words=CHUNK_SIZE):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
    if not chunks:
        chunks = [text]
    return chunks

# ---------- Encode target article chunks ----------
print("⚡ Preparing and encoding target article chunks...")
target_chunks = []  # list of dicts: {"article_id": int, "chunk": str}
for aid, art in articles.items():
    text = art.get("text", "") or art.get("title", "")
    chunks = chunk_text(text, CHUNK_SIZE)
    for chunk in chunks:
        target_chunks.append({"article_id": aid, "chunk": chunk})

# Encode all chunks in batches
all_chunk_texts = [c["chunk"] for c in target_chunks]
chunk_vectors = []
for start in tqdm(range(0, len(all_chunk_texts), BATCH_SIZE), desc="Chunk batches"):
    batch_texts = all_chunk_texts[start:start+BATCH_SIZE]
    batch_vecs = encode_text(batch_texts)
    chunk_vectors.append(batch_vecs)
chunk_vectors = np.vstack(chunk_vectors)
print(f"✅ Encoded {len(chunk_vectors)} chunks from {len(target_chunks)} articles.\n")

# ---------- Encode sentences ----------
unique_sentences = list({s["sentence"] for s in sentence_links})
print(f"⚡ Encoding {len(unique_sentences)} unique sentences...")
sentence_vectors = []
for start in tqdm(range(0, len(unique_sentences), BATCH_SIZE), desc="Sentence batches"):
    batch_texts = unique_sentences[start:start+BATCH_SIZE]
    batch_vecs = encode_text(batch_texts)
    sentence_vectors.append(batch_vecs)
sentence_vectors = np.vstack(sentence_vectors)
print("✅ Sentence encoding complete.\n")

# ---------- Compute top-K similarities ----------
print("📝 Computing top-K similarities...")
target_article_ids = np.array([c["article_id"] for c in target_chunks])
results_seen = set()  # avoid duplicate sentence->article rows

with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence", "source_title", "target_article_id", "target_title", "score"])

    for idx, sent in enumerate(tqdm(unique_sentences, desc="Sentences")):
        sent_vec = sentence_vectors[idx:idx+1]  # shape (1, dim)
        # cosine similarity with all chunks
        sims = np.dot(chunk_vectors, sent_vec.T).squeeze()  # shape=(num_chunks,)
        # compute max similarity per article
        article_max_sims = {}
        for chunk_idx, sim in enumerate(sims):
            aid = target_article_ids[chunk_idx]
            article_max_sims[aid] = max(sim, article_max_sims.get(aid, -np.inf))
        # top-K articles
        top_articles = sorted(article_max_sims.items(), key=lambda x: -x[1])[:TOP_K]
        for aid, score in top_articles:
            row_key = (sent, aid)
            if row_key in results_seen:
                continue
            results_seen.add(row_key)
            tgt_title = articles[aid].get("title", "")
            # find source_title from sentence_links
            source_title = next((s["source_title"] for s in sentence_links if s["sentence"] == sent), "")
            writer.writerow([sent, source_title, aid, tgt_title, f"{score:.6f}"])

print(f"✅ CSV saved to {CSV_OUT}")
