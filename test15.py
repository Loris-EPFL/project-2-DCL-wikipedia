#!/usr/bin/env python3
"""
sentence_to_articles_similarity.py

Compute top-K similarities between sentences (with links) and all articles.
Uses GPU if available and SentenceTransformers for embeddings.
"""
import json
import numpy as np
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ---------- CONFIG ----------
SENTENCE_LINKS_FILE = "sentence_links.json"  # cleaned sentences with target_title
ALL_ARTICLES_FILE = "small_articles_2.json"  # main articles
LINKED_ARTICLES_FILE = "linked_articles_2.json"  # target corpus
CSV_OUT = "sentence_topk_similarity.csv"
CHUNK_SIZE = 256
TOP_K = 20
BATCH_SIZE = 32
MODEL_PATH = "models/multilingual-e5-large"
# ----------------------------

print("📥 Loading JSON files...")
with open(SENTENCE_LINKS_FILE, "r", encoding="utf-8") as f:
    sentence_links = json.load(f)

with open(ALL_ARTICLES_FILE, "r", encoding="utf-8") as f:
    main_articles = json.load(f)

with open(LINKED_ARTICLES_FILE, "r", encoding="utf-8") as f:
    linked_articles = json.load(f)

# Combine all articles (id -> article)
articles = {}
next_id = 1
for art in main_articles + linked_articles:
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

all_ids = list(articles.keys())
all_texts = [articles[i].get("text", "") or articles[i].get("title", "") for i in all_ids]

print(f"Loaded {len(articles)} articles and {len(sentence_links)} sentence links.")

# ---------- Load model ----------
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("🧠 Loading model on", device)
model = SentenceTransformer(MODEL_PATH, device=device)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dim = {embedding_dim}")

# ---------- Encode all articles ----------
print("⚡ Encoding all articles on GPU...")
article_vectors = {}
for start in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Article batches"):
    batch_texts = all_texts[start:start+BATCH_SIZE]
    batch_ids = all_ids[start:start+BATCH_SIZE]
    batch_vecs = model.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=True)
    for aid, vec in zip(batch_ids, batch_vecs):
        article_vectors[aid] = vec.astype(np.float32)

# Prepare arrays for fast similarity computation
all_vectors = np.stack([article_vectors[i] for i in all_ids])
all_ids_array = np.array(all_ids)

# ---------- Deduplicate sentences ----------
unique_sentences = {}
for s in sentence_links:
    text = s["sentence"].strip()
    if text not in unique_sentences:
        unique_sentences[text] = s

sentences_list = list(unique_sentences.values())
print(f"{len(sentences_list)} unique sentences to process.")

# ---------- Encode sentences ----------
print("⚡ Encoding sentences on GPU...")
sentence_vectors = {}
sent_texts = [s["sentence"] for s in sentences_list]

for start in tqdm(range(0, len(sent_texts), BATCH_SIZE), desc="Sentence batches"):
    batch_texts = sent_texts[start:start+BATCH_SIZE]
    batch_vecs = model.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=True)
    for idx, vec in enumerate(batch_vecs):
        sentence_vectors[start + idx] = vec.astype(np.float32)

# ---------- Compute top-K similarities ----------
print("📝 Computing top-K similarities...")
with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence", "source_title", "target_id", "target_title", "score"])

    for idx, s in enumerate(tqdm(sentences_list, desc="Sentences")):
        sent_vec = sentence_vectors[idx].reshape(1, -1)  # ✅ reshape to (1, dim)
        # cosine similarity
        sims = np.dot(all_vectors, sent_vec.T).squeeze()
        top_idx = np.argsort(-sims)[:TOP_K]
        for tidx in top_idx:
            target_id = all_ids_array[tidx]
            target_title = articles[target_id].get("title", "")
            writer.writerow([s["sentence"], s["source_title"], target_id, target_title, f"{sims[tidx]:.6f}"])


print(f"✅ CSV saved to {CSV_OUT}")
