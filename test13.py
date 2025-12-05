#!/usr/bin/env python3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SMALL_FILE = "small_articles_2.json"
LINKED_FILE = "linked_articles_2.json"
EMB_MODEL_PATH = "models/multilingual-e5-large"
DEVICE = "mps"  # Apple GPU
CHUNK_SIZE = 256  # only needed for very long texts

# ------------------------
# Load articles
# ------------------------
with open(SMALL_FILE, "r", encoding="utf-8") as f:
    small = json.load(f)
with open(LINKED_FILE, "r", encoding="utf-8") as f:
    linked = json.load(f)

all_articles = small + linked
article_texts = [a.get("text") or a.get("title") or "" for a in all_articles]
article_ids = list(range(1, len(all_articles) + 1))

print(f"Loaded {len(all_articles)} articles.")

# ------------------------
# Load model on GPU
# ------------------------
model = SentenceTransformer(EMB_MODEL_PATH, device=DEVICE)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dim = {embedding_dim}")

# ------------------------
# Encode all articles at once (batch on GPU)
# ------------------------
print("Encoding all articles on GPU...")
embeddings = model.encode(
    article_texts,
    batch_size=32,             # adjust batch size if memory issues
    normalize_embeddings=True,  # cosine ready
    show_progress_bar=True,
)

# Convert to float32 numpy arrays
vectors = {aid: np.array(vec, dtype=np.float32) for aid, vec in zip(article_ids, embeddings)}

# Quick debug
for aid in article_ids[:3]:
    print(aid, all_articles[aid-1].get("title"), vectors[aid][:6], "norm=", np.linalg.norm(vectors[aid]))
