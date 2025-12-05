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
COLLECTION_NAME = "debug_collection"
CHUNK_SIZE_WORDS = 256

# ----------------------------
# Load data
# ----------------------------
with open(SMALL_FILE, "r", encoding="utf-8") as f:
    small_articles = json.load(f)

with open(LINKED_FILE, "r", encoding="utf-8") as f:
    linked_articles = json.load(f)

all_articles = {}
counter = 1
for article in small_articles + linked_articles:
    if "id" in article:
        article_id = int(article["id"])
    else:
        article_id = counter
        counter += 1
    all_articles[article_id] = article

print(f"\n📌 Loaded {len(all_articles)} articles\n")

# ----------------------------
# Load model
# ----------------------------
print("📦 Loading model…")
model = SentenceTransformer("intfloat/multilingual-e5-large")
vector_size = model.get_sentence_embedding_dimension()
print(f"✅ Model loaded. Embedding size = {vector_size}\n")

# ----------------------------
# Helper: encode with debugging
# ----------------------------
def encode_article_text(text, chunk_size_words=CHUNK_SIZE_WORDS):
    words = text.split()

    chunks = [" ".join(words[i:i+chunk_size_words]) 
              for i in range(0, len(words), chunk_size_words)]
    if not chunks:
        chunks = [""]

    print(f"   - Chunk count: {len(chunks)}")

    prefixed = [f"passage: {chunk}" for chunk in chunks]

    vecs = model.encode(prefixed, normalize_embeddings=False)
    print(f"   - Raw vecs shape: {vecs.shape}")

    # mean-pool then normalize manually (correct way)
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)

    if norm == 0:
        print("   ⚠ WARNING: mean vector norm = 0 (collapsed embedding!)")

    mean_vec = mean_vec / (norm + 1e-12)

    print(f"   - Final vector norm: {np.linalg.norm(mean_vec):.4f}")

    return mean_vec

# ----------------------------
# Encode all articles
# ----------------------------
print("🚀 Encoding all articles…\n")

points = []
vectors_debug = []

for article_id, article in list(all_articles.items())[:20]:
    # only show debugging for first 20 articles so logs don't explode
    print(f"Encoding article {article_id}: {article.get('title','(no title)')}")

    text = article.get("text", "")
    vec = encode_article_text(text)

    vectors_debug.append(vec)

    points.append(
        PointStruct(
            id=int(article_id),
            vector=vec.tolist(),
            payload={"id": int(article_id), "title": article.get("title", "")}
        )
    )
    print("")

# ----------------------------
# VECTOR DIAGNOSTICS
# ----------------------------
vectors_debug = np.array(vectors_debug)

print("\n🧪 ===== VECTOR DIAGNOSTICS =====")

print("Vector matrix shape:", vectors_debug.shape)
print("Global mean value:", vectors_debug.mean())
print("Global std deviation:", vectors_debug.std())

# L2 distance between consecutive vectors
dists = [np.linalg.norm(vectors_debug[i] - vectors_debug[i+1])
         for i in range(len(vectors_debug)-1)]

print("Avg L2 distance between article vectors:", np.mean(dists))
print("Min L2 distance:", np.min(dists))
print("Max L2 distance:", np.max(dists))

# Print first 2 vectors for visual inspection
print("\nFirst vector (first 10 dims):", vectors_debug[0][:10])
print("Second vector (first 10 dims):", vectors_debug[1][:10])
print("\n================================\n")

# ----------------------------
# Create Qdrant collection
# ----------------------------
print("🔧 Creating Qdrant collection…")

client = QdrantClient(":memory:")

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance="Cosine"),
)

print("✅ Qdrant ready.\n")

# ----------------------------
# Upsert
# ----------------------------
client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"📌 Inserted {len(points)} debug vectors.\n")
