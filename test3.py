import json
import numpy as np
from pathlib import Path
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

# ----------------------------
# Load data
# ----------------------------
with open(SMALL_FILE, "r", encoding="utf-8") as f:
    small_articles = json.load(f)

with open(LINKED_FILE, "r", encoding="utf-8") as f:
    linked_articles = json.load(f)

# Combine articles
all_articles = {}
counter = 1
for article in small_articles + linked_articles:
    if "id" in article:
        article_id = int(article["id"])
    else:
        article_id = counter
        counter += 1
    all_articles[article_id] = article

print(f"Loaded {len(all_articles)} articles.")

# ----------------------------
# Load model
# ----------------------------
model = SentenceTransformer("models/multilingual-e5-large")
vector_size = model.get_sentence_embedding_dimension()

# ----------------------------
# Helper functions
# ----------------------------
def encode_article_text(text, chunk_size_words=CHUNK_SIZE_WORDS):
    """Encode an article by chunking and MAX pooling embeddings."""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size_words]) 
              for i in range(0, len(words), chunk_size_words)]
    
    if not chunks:
        chunks = ["empty"]

    # E5 requires prefix `passage:`
    prefixed = [f"passage: {chunk}" for chunk in chunks]

    vecs = model.encode(prefixed, normalize_embeddings=True)

    # MAX POOL instead of mean-pooling to avoid collapsing vectors
    return np.max(vecs, axis=0)

# ----------------------------
# Build vectors
# ----------------------------
article_vectors = {}

for article_id, article in all_articles.items():
    text = article.get("text", "")
    vec = encode_article_text(text)
    article_vectors[article_id] = vec

print("Encoded all articles.")

# ----------------------------
# Connect to Qdrant
# ----------------------------
client = QdrantClient(":memory:")

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance="cosine")
)

# ----------------------------
# Upload vectors
# ----------------------------
points = []
for article_id, vec in article_vectors.items():
    article = all_articles[article_id]
    points.append(
        PointStruct(
            id=int(article_id),
            vector=vec,
            payload={
                "id": int(article_id),
                "title": article.get("title", ""),
            }
        )
    )

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Upserted {len(points)} points.")

# ----------------------------
# Similarity computation (REAL version)
# ----------------------------
CSV_OUT = "small_linked_similarity.csv"

with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "source_title", "target_id", "target_title", "score"])

    for article_id, article in all_articles.items():

        vec = article_vectors[article_id]  # REAL query vector

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=TOP_K + 1  # +1 because we will remove the self-match
        )

        # Filter out itself
        neighbors = [n for n in results if n.id != article_id][:TOP_K]

        for n in neighbors:
            writer.writerow([
                article_id,
                article.get("title", ""),
                n.payload["id"],
                n.payload.get("title", "(unknown)"),
                f"{n.score:.6f}"
            ])

print(f"✅ Fixed similarity graph saved to {CSV_OUT}")
