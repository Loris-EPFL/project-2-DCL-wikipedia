# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Wikipedia Link Reconstruction Pipeline
#
# This notebook implements a complete machine learning pipeline for French Wikipedia articles:
#
# 1. Load and preprocess Wikipedia articles (JSON/Parquet)
# 2. Generate semantic embeddings using multilingual-e5-large
# 3. Store embeddings in Qdrant vector database
# 4. Perform semantic search and similarity analysis
# 5. Reconstruct Wikipedia links using vector similarity
#
# **Architecture:**
# - **Embedding Model:** intfloat/multilingual-e5-large (1024-dimensional vectors)
# - **Vector Store:** Qdrant (local Docker or in-memory)
# - **Data Processing:** Polars for efficient DataFrame operations

# %% [markdown]
# ## 1. Imports and Dependencies

# %%
# Standard library
from pathlib import Path
from typing import Optional
from urllib.parse import unquote
import html
import json
import re

# Third-party data processing
import numpy as np
import polars as pl
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# Machine learning
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import PointStruct, VectorParams

# %% [markdown]
# ## 2. Configuration
#
# Adjust these parameters according to your setup and requirements.

# %%
# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
DATA_JSON_PATH = Path("merged_wikipedia_withoutLinks_FR.json")
DATA_PARQUET_PATH = Path("merged_wikipedia_withoutLinks_FR.parquet")
ARTICLES_WITH_LINKS_PARQUET = Path("articles_fr_withLinks.parquet")
ARTICLES_CSV_PATH = Path("articles.csv")
ARTICLES_STREAM_CSV_PATH = Path("articles_stream.csv")
LINKS_PARQUET_PATH = Path("per_article_links.parquet")

# ---------------------------------------------------------------------------
# Qdrant Configuration
# ---------------------------------------------------------------------------
QDRANT_COLLECTION_NAME = "wikipedia_fr"
USE_IN_MEMORY_QDRANT = True  # False = Docker Qdrant with persistence
QDRANT_HOST = "localhost"
QDRANT_HTTP_PORT = 6333
QDRANT_GRPC_PORT = 6334

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
# Using static-similarity-mrl: 1000x faster than E5, better separation for similarity tasks
MODEL_NAME = "sentence-transformers/static-similarity-mrl-multilingual-v1"
MODEL_LOCAL_DIR = Path("models/static-similarity-mrl-multilingual-v1")
SAVE_DOWNLOADED_MODEL = True  # Cache model locally after first download

# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Processing Parameters
# ---------------------------------------------------------------------------
# With static-similarity-mrl (1000x faster), we can process much more data
ROWGROUP_BATCH = 512       # Rows per batch from Parquet row groups
ENCODE_BATCH = 128         # Batch size for embedding generation (fast model handles larger batches)
ARTICLES_NUMBER = 2000     # Limit articles (None = process all 4.5M)
INGEST_FOR_SMOKE_TEST = 100  # Articles for initial test (50k takes ~2-3 min with new model)

# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------
BUILD_LINK_INDEX = False   # Extract and index article links

# %% [markdown]
# ## 3. Qdrant Vector Database Setup
#
# Initialize connection to Qdrant vector database. Supports two modes:
# - **Docker mode:** Persistent storage, requires Docker container
# - **In-memory mode:** Ephemeral, no Docker needed (for testing)
#
# **Docker setup:**
# ```bash
# mkdir -p ~/qdrant_storage
# docker run -d --name qdrant \
#   -p 6333:6333 -p 6334:6334 \
#   -v ~/qdrant_storage:/qdrant/storage \
#   qdrant/qdrant:latest
# ```

# %%
if USE_IN_MEMORY_QDRANT:
    client = QdrantClient(":memory:")
    print("Connected to in-memory Qdrant instance")
else:
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        prefer_grpc=True,
    )
    print(f"Connected to Qdrant at http://{QDRANT_HOST}:{QDRANT_HTTP_PORT}")

# %% [markdown]
# ## 4. Embedding Model Initialization
#
# Load the static-similarity-mrl model for generating semantic embeddings.
# The model is cached locally to avoid repeated downloads.
#
# **Model details:**
# - 1024-dimensional embeddings
# - **1000x faster than E5-large on CPU** (634 vs 0.6 texts/sec)
# - **5x better separation** for distinguishing similar vs random pairs (0.526 vs 0.106 gap)
# - Optimized for semantic textual similarity tasks
# - Supports 100+ languages including French

# %%
def load_sentence_transformer() -> SentenceTransformer:
    """
    Load static-similarity-mrl embedding model.

    This model is optimized for semantic similarity (not retrieval), making it
    ideal for finding related Wikipedia articles. It's 1000x faster than E5-large
    and has much better separation between similar and random pairs.

    Returns:
        Initialized SentenceTransformer model on configured device
    """
    if MODEL_LOCAL_DIR.exists():
        print(f"Loading model from local directory: {MODEL_LOCAL_DIR}")
        return SentenceTransformer(str(MODEL_LOCAL_DIR), device=DEVICE)

    print(f"Downloading model from Hugging Face: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    if SAVE_DOWNLOADED_MODEL:
        MODEL_LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {MODEL_LOCAL_DIR}")
        model.save(str(MODEL_LOCAL_DIR))

    return model


model = load_sentence_transformer()
VECTOR_SIZE = model.get_sentence_embedding_dimension()
print(f"Model loaded successfully | Vector dimension: {VECTOR_SIZE}")

# %% [markdown]
# ## 5. Data Loading Functions
#
# Utilities for loading Wikipedia articles from JSON and Parquet formats.
# Implements automatic format conversion and caching for efficiency.

# %%
def clean_text(text: str) -> str:
    """
    Clean article text for better embedding quality.

    Removes:
    - HTML entities and tags
    - Multiple consecutive spaces
    - Leading/trailing whitespace

    Args:
        text: Raw article text

    Returns:
        Cleaned text suitable for embedding (minimum 10 chars)
    """
    if not text:
        return ""

    # Remove HTML entities
    text = html.unescape(text)

    # Remove common HTML tags that might remain
    text = re.sub(r'&lt;[^&]*?&gt;', ' ', text)
    text = re.sub(r'<[^>]*?>', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Trim
    text = text.strip()

    # Return empty string for very short articles (causes 1.000 similarity)
    if len(text) < 10:
        return ""

    # Truncate very long texts (model has token limits)
    # Keeping first ~8000 chars (~2000 tokens) per article
    if len(text) > 8000:
        text = text[:8000]

    return text


def load_jsonl_to_df(path: Path, max_rows: Optional[int] = None) -> pl.DataFrame:
    """
    Load newline-delimited JSON into a Polars DataFrame.

    Args:
        path: Path to JSONL file
        max_rows: Optional limit on rows to load

    Returns:
        Polars DataFrame with normalized schema (id, title, url, text)
    """
    if max_rows is None:
        try:
            df = pl.read_ndjson(str(path))
        except Exception:
            # Fallback for malformed lines
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            df = pl.DataFrame(rows)
    else:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        df = pl.DataFrame(rows)

    # Normalize schema
    cols = set(df.columns)

    if "id" not in cols:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("id"))
    else:
        df = df.with_columns(pl.col("id").cast(pl.Int64))

    for col, default in [("title", ""), ("url", ""), ("text", "")]:
        if col not in cols:
            df = df.with_columns(pl.lit(default).alias(col))
        else:
            df = df.with_columns(pl.col(col).cast(pl.Utf8).fill_null(default))

    return df

# %% [markdown]
# ### Load Wikipedia Dataset
#
# Load articles with automatic caching:
# - First run: loads JSON and saves as Parquet
# - Subsequent runs: loads from Parquet (much faster)

# %%
if DATA_PARQUET_PATH.exists():
    print(f"Loading from cached Parquet: {DATA_PARQUET_PATH}")
    df = pl.read_parquet(DATA_PARQUET_PATH)
    if ARTICLES_NUMBER is not None and df.height > ARTICLES_NUMBER:
        df = df.slice(0, ARTICLES_NUMBER)
    print(f"Loaded {df.height:,} articles from Parquet")
else:
    print(f"Loading from JSON: {DATA_JSON_PATH}")
    df = load_jsonl_to_df(DATA_JSON_PATH, max_rows=ARTICLES_NUMBER)
    print(f"Loaded {df.height:,} articles, saving to Parquet...")
    df.write_parquet(DATA_PARQUET_PATH)
    print(f"Saved to {DATA_PARQUET_PATH}")

# %% [markdown]
# ## 6. Link Extraction (Optional)
#
# Extract hyperlinks from Wikipedia article HTML for link reconstruction tasks.
# This builds an index of links per article with positions and metadata.

# %%
BASE_URL = "https://fr.wikipedia.org/wiki/"
ANCHOR_RE = re.compile(
    r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

LINK_ITEM = pl.Struct([
    pl.Field("full_url", pl.Utf8),
    pl.Field("start_idx", pl.Int64),
    pl.Field("anchor", pl.Utf8),
    pl.Field("href_raw", pl.Utf8),
    pl.Field("href_decoded", pl.Utf8),
])
LINK_LIST = pl.List(LINK_ITEM)


def extract_links_with_pos(text: str) -> list[dict]:
    """
    Extract links from HTML-escaped anchor tags.

    Args:
        text: HTML-escaped article text

    Returns:
        List of dicts with link metadata (URL, position, anchor text)
    """
    if not text:
        return []

    unescaped = html.unescape(text)
    links = []
    search_from = 0

    for match in ANCHOR_RE.finditer(unescaped):
        href_raw = match.group(1)
        anchor = html.unescape(match.group(2))

        # Find position in original escaped text
        needle = f'&lt;a href="{href_raw}"'
        pos = text.find(needle, search_from)
        if pos == -1:
            pos = text.find("&lt;a ", search_from)
        if pos == -1:
            pos = 0
        search_from = pos + 1

        links.append({
            "full_url": BASE_URL + href_raw,
            "start_idx": pos,
            "anchor": anchor,
            "href_raw": href_raw,
            "href_decoded": unquote(href_raw),
        })

    return links


def build_per_article_links(df: pl.DataFrame, max_rows: Optional[int] = None) -> pl.DataFrame:
    """
    Build per-article link index with positions and counts.

    Args:
        df: DataFrame with article texts
        max_rows: Optional limit on articles to process

    Returns:
        DataFrame with (id, title, links, link_count)
    """
    if max_rows is not None:
        df = df.slice(0, min(max_rows, df.height))

    return (
        df.select(["id", "title", "text"])
        .with_columns(pl.col("text").cast(pl.Utf8).fill_null(""))
        .with_columns(
            links=pl.col("text").map_elements(
                extract_links_with_pos,
                return_dtype=LINK_LIST,
            ),
        )
        .drop("text")
        .with_columns(
            links=pl.when(pl.col("links").is_null())
            .then(pl.lit([]).cast(LINK_LIST))
            .otherwise(pl.col("links")),
            link_count=pl.col("links").list.len(),
        )
    )


# Build or load link index
if BUILD_LINK_INDEX:
    if LINKS_PARQUET_PATH.exists():
        print(f"Loading cached link index: {LINKS_PARQUET_PATH}")
        per_article_links = pl.read_parquet(LINKS_PARQUET_PATH)
    else:
        print("Building per-article link index (may take a while)...")
        per_article_links = build_per_article_links(df, max_rows=10_000)
        per_article_links.write_parquet(LINKS_PARQUET_PATH)
        print(f"Saved link index to {LINKS_PARQUET_PATH}")
    print(per_article_links.select(["id", "title", "link_count"]).head())
else:
    print("Skipping link index build (BUILD_LINK_INDEX=False)")

# %% [markdown]
# ## 7. Qdrant Collection Management
#
# Functions for creating and managing the Qdrant vector collection.

# %%
def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create Qdrant collection if it doesn't exist.

    Args:
        client: Qdrant client instance
        collection_name: Name for the collection
        vector_size: Embedding dimension
    """
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine"),
    )
    print(f"Created collection '{collection_name}'")


def count_points(client: QdrantClient, collection_name: str) -> int:
    """
    Count total points in a collection.

    Returns:
        Number of vectors stored in the collection
    """
    total = client.count(collection_name=collection_name, exact=True).count
    print(f"Total points in '{collection_name}': {total:,}")
    return total


def recover_snapshot(collection_name: str, snapshot_url: str):
    """
    Recover collection from a snapshot URL.

    Args:
        collection_name: Target collection name
        snapshot_url: URL to snapshot file

    Example:
        recover_snapshot("wikipedia_fr",
                        "http://localhost:6333/collections/wikipedia_fr/snapshots/snapshot.tar")
    """
    client.recover_snapshot(collection_name=collection_name, location=snapshot_url)
    print(f"Recovered snapshot for '{collection_name}' from {snapshot_url}")


# Initialize collection
ensure_collection(client, QDRANT_COLLECTION_NAME, VECTOR_SIZE)
count_points(client, QDRANT_COLLECTION_NAME)

# %% [markdown]
# ## 8. Data Ingestion Pipeline
#
# Ingest Wikipedia articles into Qdrant with semantic embeddings.
# Implements efficient batch processing and duplicate detection.

# %%
def ingest_parquet_to_qdrant(
    parquet_path: Path,
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    rowgroup_batch: int = 512,
    encode_batch: int = 64,
    max_rows: Optional[int] = None,
):
    """
    Stream Parquet file into Qdrant with progress tracking.

    Skips articles already present in the collection (by ID).

    Args:
        parquet_path: Path to source Parquet file
        client: Qdrant client instance
        collection_name: Target collection name
        model: SentenceTransformer for encoding
        rowgroup_batch: Rows to process per batch
        encode_batch: Batch size for embedding generation
        max_rows: Optional limit on total rows to ingest
    """
    pf = pq.ParquetFile(str(parquet_path))

    total_rows = pf.metadata.num_rows
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)

    processed = 0
    written = 0
    pbar = tqdm(total=total_rows, desc="Indexing articles", unit="rows")

    for rg in range(pf.num_row_groups):
        if max_rows is not None and processed >= max_rows:
            break

        # Read row group
        table = pf.read_row_group(rg, columns=["id", "title", "url", "text"])
        df_rg = (
            pl.from_arrow(table)
            .with_columns(
                pl.col("id").cast(pl.Int64),
                pl.col("title").fill_null(""),
                pl.col("url").fill_null(""),
                pl.col("text").fill_null(""),
            )
        )

        # Apply max_rows limit
        n = df_rg.height
        remaining = None if max_rows is None else max_rows - processed
        if remaining is not None and n > remaining:
            n = remaining
            df_rg = df_rg.slice(0, n)

        # Process in batches
        for start in range(0, n, rowgroup_batch):
            length = min(rowgroup_batch, n - start)
            sub = df_rg.slice(start, length)

            ids = sub["id"].to_list()
            texts = sub["text"].to_list()
            titles = sub["title"].to_list()
            urls = sub["url"].to_list()

            # Check for existing points
            existing = client.retrieve(
                collection_name=collection_name,
                ids=[int(i) for i in ids],
                with_payload=False,
                with_vectors=False,
            )
            existing_ids = {p.id for p in existing}
            missing_idx = [i for i, pid in enumerate(ids) if int(pid) not in existing_ids]

            # Encode and insert only missing points
            if missing_idx:
                ids_new = [ids[i] for i in missing_idx]
                texts_new = [texts[i] for i in missing_idx]
                titles_new = [titles[i] for i in missing_idx]
                urls_new = [urls[i] for i in missing_idx]

                # Clean and preprocess texts
                texts_cleaned = [clean_text(text) for text in texts_new]

                # E5 model: Use "query:" for symmetric tasks (semantic similarity)
                # "passage:" is for asymmetric retrieval (query→document matching)
                texts_with_prefix = [f"query: {text}" for text in texts_cleaned]

                vectors = model.encode(
                    texts_with_prefix,
                    batch_size=encode_batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                points = [
                    PointStruct(
                        id=int(ids_new[i]),
                        vector=vectors[i].tolist(),
                        payload={
                            "id": int(ids_new[i]),
                            "title": titles_new[i],
                            "url": urls_new[i],
                            "text": texts_new[i],
                        },
                    )
                    for i in range(len(ids_new))
                ]

                client.upsert(collection_name=collection_name, points=points, wait=True)
                written += len(ids_new)

            processed += length
            pbar.update(length)
            pbar.set_postfix(written=written)

            if max_rows is not None and processed >= max_rows:
                break

    pbar.close()
    print(f"Ingestion complete | Scanned: {processed:,} | Written: {written:,}")

# %% [markdown]
# ## 9. Similarity Search Functions
#
# Perform semantic search and find similar articles using vector similarity.

# %%
def search_by_title_or_text(
    df: pl.DataFrame,
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    query_title: str,
    top_k: int = 5,
):
    """
    Search for similar articles by title or text query.

    If an article with the exact title exists, uses its full text as query.
    Otherwise, uses the title string directly.

    Args:
        df: DataFrame with article data
        client: Qdrant client
        collection_name: Target collection
        model: SentenceTransformer for encoding
        query_title: Article title or search query
        top_k: Number of results to return
    """
    # Try to find exact title match
    matches = df.filter(pl.col("title").str.to_lowercase() == query_title.lower())
    query_text = matches.select("text").to_series()[0] if matches.height > 0 else query_title

    # Clean and prepare query
    query_text = clean_text(query_text)

    # E5 model requires "query:" prefix for search queries
    query_vector = model.encode([f"query: {query_text}"], normalize_embeddings=True)[0]

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )

    print(f"\nTop {top_k} matches for '{query_title}':")
    for r in results:
        print(f"  {r.payload.get('title')} (score={r.score:.3f})")


def encode_article_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Encode long article by chunking and averaging embeddings.

    Splits text into 256-word chunks, encodes each with 'passage:' prefix,
    and averages the resulting vectors for a single representation.

    Args:
        text: Full article text
        model: SentenceTransformer instance

    Returns:
        Single averaged embedding vector
    """
    words = text.split()
    if not words:
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)

    chunks = [" ".join(words[i : i + 256]) for i in range(0, len(words), 256)]
    # Use "query:" for symmetric similarity (not "passage:")
    vecs = model.encode(
        [f"query: {c}" for c in chunks],
        normalize_embeddings=True,
    )
    return np.mean(vecs, axis=0)


def find_similar_articles(
    df: pl.DataFrame,
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    article_ids: list[int],
    top_k: int = 5,
):
    """
    Find similar articles for given article IDs.

    For articles in Qdrant: uses recommendation API
    For articles not in Qdrant: encodes text and performs search

    Args:
        df: DataFrame with article data
        client: Qdrant client
        collection_name: Target collection
        model: SentenceTransformer for encoding
        article_ids: List of article IDs to find similarities for
        top_k: Number of similar articles to return per ID
    """
    valid_ids = set(df.select(pl.col("id").cast(pl.Int64)).to_series().to_list())
    article_ids = [aid for aid in article_ids if aid in valid_ids]

    for article_id in article_ids:
        title_row = df.filter(pl.col("id") == article_id).select("title")
        title = title_row.to_series()[0] if title_row.height > 0 else "(unknown)"

        try:
            # Try to use recommendation API if article is in Qdrant
            points = client.retrieve(
                collection_name=collection_name,
                ids=[int(article_id)],
                with_payload=True,
                with_vectors=False,
            )
            if points:
                results = client.recommend(
                    collection_name=collection_name,
                    positive=[int(article_id)],
                    limit=top_k,
                )
            else:
                raise ValueError("not_found")
        except Exception:
            # Fallback: encode article text and search
            row = df.filter(pl.col("id") == article_id)
            if row.height == 0:
                continue
            text = row.select(pl.col("text").fill_null("")).to_series()[0]
            vec = encode_article_text(text, model)
            results = client.search(
                collection_name=collection_name,
                query_vector=vec,
                limit=top_k,
            )

        print(f"\nSimilar articles for: {title} (id={article_id})")
        for r in results:
            print(f"  {r.payload.get('title')} (id={r.payload.get('id')}, score={r.score:.3f})")

# %% [markdown]
# ## 10. Data Export Utilities
#
# Functions for exporting data from Qdrant and converting between formats.

# %%
def rebuild_df_from_qdrant(
    client: QdrantClient,
    collection_name: str,
    max_rows: Optional[int] = 1000,
    batch_limit: int = 1000,
) -> pl.DataFrame:
    """
    Reconstruct DataFrame from Qdrant collection using scroll API.

    Args:
        client: Qdrant client
        collection_name: Source collection
        max_rows: Maximum rows to retrieve
        batch_limit: Batch size for scrolling

    Returns:
        Polars DataFrame with (id, title, url, text)
    """
    rows = []
    next_page = None
    selector = rest.PayloadSelectorInclude(include=["id", "title", "url", "text"])

    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            limit=batch_limit,
            with_payload=True,
            with_vectors=False,
            payload_selector=selector,
            offset=next_page,
        )
        if not points:
            break

        for p in points:
            pid = p.payload.get("id", p.id)
            rows.append({
                "id": pid,
                "title": p.payload.get("title", ""),
                "url": p.payload.get("url", ""),
                "text": p.payload.get("text", ""),
            })
            if max_rows is not None and len(rows) >= max_rows:
                break

        if max_rows is not None and len(rows) >= max_rows:
            break
        if next_page is None:
            break

    df = pl.DataFrame(rows).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col("title").cast(pl.Utf8).fill_null(""),
        pl.col("url").cast(pl.Utf8).fill_null(""),
        pl.col("text").cast(pl.Utf8).fill_null(""),
    )
    print(f"Rebuilt DataFrame from Qdrant: {df.height:,} rows")
    return df


def export_df_to_csv(df: pl.DataFrame, path: Path, cols: Optional[list[str]] = None):
    """
    Export DataFrame to CSV.

    Args:
        df: Source DataFrame
        path: Output CSV path
        cols: Columns to export (default: ["id", "title", "url", "text"])
    """
    if cols is None:
        cols = ["id", "title", "url", "text"]
    df.select(cols).write_csv(path)
    print(f"Exported {df.height:,} rows to {path}")


def stream_parquet_to_csv(
    parquet_path: Path,
    output_csv: Path,
    cols: Optional[list[str]] = None,
    max_rows: Optional[int] = None,
):
    """
    Efficiently stream large Parquet to CSV without loading all into memory.

    Args:
        parquet_path: Source Parquet file
        output_csv: Output CSV path
        cols: Columns to export (default: ["id", "title", "url", "text"])
        max_rows: Optional limit on rows to export
    """
    if cols is None:
        cols = ["id", "title", "url", "text"]

    pf = pq.ParquetFile(str(parquet_path))
    processed = 0
    first = True

    for rg in range(pf.num_row_groups):
        if max_rows is not None and processed >= max_rows:
            break

        table = pf.read_row_group(rg, columns=cols)
        df_rg = (
            pl.from_arrow(table)
            .with_columns(
                pl.col("id").cast(pl.Int64),
                pl.col("title").fill_null(""),
                pl.col("url").fill_null(""),
                pl.col("text").fill_null(""),
            )
        )

        remaining = None if max_rows is None else max_rows - processed
        if remaining is not None and df_rg.height > remaining:
            df_rg = df_rg.slice(0, remaining)

        mode = "wb" if first else "ab"
        with output_csv.open(mode) as f:
            pacsv.write_csv(
                df_rg.to_arrow(),
                f,
                write_options=pacsv.WriteOptions(include_header=first),
            )

        processed += df_rg.height
        first = False

    print(f"Streamed {processed:,} rows from {parquet_path} to {output_csv}")

# %% [markdown]
# **Note:** For ingesting data to Qdrant, use `ingest_parquet_to_qdrant()` (Section 8).
# It includes text cleaning, deduplication, and progress tracking.

# %% [markdown]
# ## 11. Model Benchmarking
#
# Compare different embedding models for speed and quality on your Wikipedia data.

# %%
import time
from typing import Dict, List, Tuple
import pandas as pd


# Candidate models for benchmarking
BENCHMARK_MODELS = {
    "multilingual-e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "prefix": "query:",
        "description": "Former model (1024-dim, very slow)",
    },
    "static-similarity-mrl": {
        "name": "sentence-transformers/static-similarity-mrl-multilingual-v1",
        "prefix": "query:",
        "description": "Current! 1000x faster (1024-dim, optimal)",
    },
    "paraphrase-mpnet": {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "prefix": "",  # No prefix needed
        "description": "Alternative (768-dim, good quality)",
    },
    "paraphrase-minilm": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prefix": "",  # No prefix needed
        "description": "Fast & small (384-dim)",
    },
}


def benchmark_model_speed(
    df: pl.DataFrame,
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    encode_batch: int = 64,
    upsert_batch: int = 512,
    max_rows: Optional[int] = None,
):
    """
    Encode and upsert DataFrame directly to Qdrant.

    Args:
        df: Source DataFrame
        client: Qdrant client
        collection_name: Target collection
        model: SentenceTransformer for encoding
        encode_batch: Batch size for encoding
        upsert_batch: Batch size for upserting
        max_rows: Optional limit on rows to export
    """
    n = df.height if max_rows is None else min(df.height, max_rows)
    processed = 0
    pbar = tqdm(total=n, desc="Exporting DataFrame to Qdrant", unit="rows")

    while processed < n:
        length = min(upsert_batch, n - processed)
        sub = df.slice(processed, length)

        ids = sub["id"].cast(pl.Int64).to_list()
        titles = sub["title"].fill_null("").to_list()
        urls = sub["url"].fill_null("").to_list()
        texts = sub["text"].fill_null("").to_list()

        # E5 model: Use "query:" for symmetric tasks (semantic similarity)
        texts_with_prefix = [f"query: {text}" for text in texts]

        vectors = model.encode(
            texts_with_prefix,
            batch_size=encode_batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        points = [
            PointStruct(
                id=int(ids[i]),
                vector=vectors[i].tolist(),
                payload={
                    "id": int(ids[i]),
                    "title": titles[i],
                    "url": urls[i],
                    "text": texts[i],
                },
            )
            for i in range(len(ids))
        ]

        client.upsert(collection_name=collection_name, points=points, wait=True)
        processed += length
        pbar.update(length)

    pbar.close()
    print(f"Exported {n:,} rows to Qdrant")


def export_parquet_to_qdrant(
    parquet_path: Path,
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    encode_batch: int = 64,
    upsert_batch: int = 512,
    max_rows: Optional[int] = None,
):
    """
    Encode and upsert Parquet file directly to Qdrant.

    Args:
        parquet_path: Source Parquet file
        client: Qdrant client
        collection_name: Target collection
        model: SentenceTransformer for encoding
        encode_batch: Batch size for encoding
        upsert_batch: Batch size for upserting
        max_rows: Optional limit on rows to export
    """
    pf = pq.ParquetFile(str(parquet_path))
    total_rows = pf.metadata.num_rows
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)

    processed = 0
    pbar = tqdm(total=total_rows, desc="Exporting Parquet to Qdrant", unit="rows")

    for rg in range(pf.num_row_groups):
        if max_rows is not None and processed >= max_rows:
            break

        table = pf.read_row_group(rg, columns=["id", "title", "url", "text"])
        df_rg = (
            pl.from_arrow(table)
            .with_columns(
                pl.col("id").cast(pl.Int64),
                pl.col("title").fill_null(""),
                pl.col("url").fill_null(""),
                pl.col("text").fill_null(""),
            )
        )

        remaining = None if max_rows is None else max_rows - processed
        n = df_rg.height if remaining is None else min(df_rg.height, remaining)

        inner = 0
        while inner < n:
            length = min(upsert_batch, n - inner)
            sub = df_rg.slice(inner, length)

            ids = sub["id"].to_list()
            titles = sub["title"].to_list()
            urls = sub["url"].to_list()
            texts = sub["text"].to_list()

            # E5 model: Use "query:" for symmetric tasks (semantic similarity)
            texts_with_prefix = [f"query: {text}" for text in texts]

            vectors = model.encode(
                texts_with_prefix,
                batch_size=encode_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            points = [
                PointStruct(
                    id=int(ids[i]),
                    vector=vectors[i].tolist(),
                    payload={
                        "id": int(ids[i]),
                        "title": titles[i],
                        "url": urls[i],
                        "text": texts[i],
                    },
                )
                for i in range(len(ids))
            ]

            client.upsert(collection_name=collection_name, points=points, wait=True)
            processed += length
            inner += length
            pbar.update(length)

    pbar.close()
    print(f"Exported {processed:,} rows from {parquet_path} to Qdrant")

# %% [markdown]
# ## 12. Model Benchmarking
#
# Compare different embedding models for speed and quality on your Wikipedia data.

# %%
import time
from typing import Dict, List, Tuple
import pandas as pd


# Candidate models for benchmarking
BENCHMARK_MODELS = {
    "multilingual-e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "prefix": "query:",
        "description": "Current model (1024-dim, slower)",
    },
    "static-similarity-mrl": {
        "name": "sentence-transformers/static-similarity-mrl-multilingual-v1",
        "prefix": "query:",
        "description": "125x faster! (256-dim, optimized for STS)",
    },
    "paraphrase-mpnet": {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "prefix": "",  # No prefix needed
        "description": "Best for French similarity (768-dim)",
    },
    "paraphrase-minilm": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prefix": "",  # No prefix needed
        "description": "Fast & balanced (384-dim)",
    },
}


def benchmark_model_speed(
    model_name: str,
    prefix: str,
    sample_texts: List[str],
    batch_size: int = 32,
    num_runs: int = 3,
) -> Dict:
    """
    Benchmark a model's encoding speed.

    Args:
        model_name: HuggingFace model identifier
        prefix: Prefix to add to texts (e.g., "query:")
        sample_texts: List of texts to encode
        batch_size: Batch size for encoding
        num_runs: Number of runs for averaging

    Returns:
        Dict with timing statistics and model info
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    # Load model
    start_load = time.time()
    test_model = SentenceTransformer(model_name, device=DEVICE)
    load_time = time.time() - start_load
    vector_dim = test_model.get_sentence_embedding_dimension()

    print(f"✓ Model loaded in {load_time:.2f}s | Dimension: {vector_dim}")

    # Prepare texts with prefix
    texts_with_prefix = [f"{prefix} {text}".strip() for text in sample_texts]

    # Warm-up run
    _ = test_model.encode(texts_with_prefix[:10], batch_size=batch_size, show_progress_bar=False)

    # Benchmark encoding
    encode_times = []
    for run in range(num_runs):
        start = time.time()
        embeddings = test_model.encode(
            texts_with_prefix,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.time() - start
        encode_times.append(elapsed)
        print(f"  Run {run+1}/{num_runs}: {elapsed:.3f}s ({len(sample_texts)/elapsed:.1f} texts/sec)")

    avg_time = np.mean(encode_times)
    texts_per_sec = len(sample_texts) / avg_time

    return {
        "model_name": model_name,
        "vector_dim": vector_dim,
        "load_time_sec": load_time,
        "avg_encode_time_sec": avg_time,
        "texts_per_sec": texts_per_sec,
        "sample_size": len(sample_texts),
        "embeddings": embeddings,
    }


def benchmark_model_quality(
    embeddings: np.ndarray,
    sample_titles: List[str],
    ground_truth_pairs: List[Tuple[int, int]],
) -> Dict:
    """
    Benchmark embedding quality using known similar article pairs.

    Args:
        embeddings: Generated embeddings (N x D)
        sample_titles: Article titles
        ground_truth_pairs: List of (idx1, idx2) tuples of known similar articles

    Returns:
        Dict with quality metrics
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Calculate metrics on ground truth pairs
    similarities_positive = []
    for idx1, idx2 in ground_truth_pairs:
        similarities_positive.append(sim_matrix[idx1, idx2])

    # Random negative pairs
    np.random.seed(42)
    n_samples = len(embeddings)
    negative_pairs = [(i, j) for i, j in zip(
        np.random.randint(0, n_samples, 100),
        np.random.randint(0, n_samples, 100)
    ) if i != j]

    similarities_negative = []
    for idx1, idx2 in negative_pairs:
        similarities_negative.append(sim_matrix[idx1, idx2])

    return {
        "avg_similarity_related": np.mean(similarities_positive),
        "avg_similarity_random": np.mean(similarities_negative),
        "similarity_gap": np.mean(similarities_positive) - np.mean(similarities_negative),
        "score_range": (np.min(sim_matrix), np.max(sim_matrix)),
    }


def run_full_benchmark(
    sample_size: int = 500,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Run comprehensive benchmark on all candidate models.

    Args:
        sample_size: Number of Wikipedia articles to test
        batch_size: Batch size for encoding

    Returns:
        DataFrame with benchmark results
    """
    print(f"\n{'#'*60}")
    print(f"# MODEL BENCHMARK - Wikipedia Link Reconstruction")
    print(f"# Sample size: {sample_size:,} articles")
    print(f"# Device: {DEVICE}")
    print(f"{'#'*60}\n")

    # Get sample articles
    sample_df = df.head(sample_size)
    sample_texts = [clean_text(text) for text in sample_df["text"].to_list()]
    sample_titles = sample_df["title"].to_list()

    # Create ground truth pairs (articles with similar titles/topics)
    # For demo: pairs of articles where titles share common words
    ground_truth_pairs = []
    for i in range(min(50, sample_size)):
        for j in range(i+1, min(i+10, sample_size)):
            # Simple heuristic: similar if titles share 2+ words
            words_i = set(sample_titles[i].lower().split())
            words_j = set(sample_titles[j].lower().split())
            if len(words_i & words_j) >= 2:
                ground_truth_pairs.append((i, j))
                if len(ground_truth_pairs) >= 20:
                    break
        if len(ground_truth_pairs) >= 20:
            break

    print(f"✓ Loaded {len(sample_texts):,} sample texts")
    print(f"✓ Created {len(ground_truth_pairs)} ground truth similar pairs\n")

    results = []

    for model_id, model_info in BENCHMARK_MODELS.items():
        try:
            # Speed benchmark
            speed_results = benchmark_model_speed(
                model_info["name"],
                model_info["prefix"],
                sample_texts,
                batch_size=batch_size,
            )

            # Quality benchmark
            quality_results = benchmark_model_quality(
                speed_results["embeddings"],
                sample_titles,
                ground_truth_pairs,
            )

            # Combine results
            results.append({
                "Model": model_id,
                "Description": model_info["description"],
                "Dimensions": speed_results["vector_dim"],
                "Load Time (s)": f"{speed_results['load_time_sec']:.2f}",
                "Encode Time (s)": f"{speed_results['avg_encode_time_sec']:.2f}",
                "Speed (texts/sec)": f"{speed_results['texts_per_sec']:.1f}",
                "Avg Similar Score": f"{quality_results['avg_similarity_related']:.3f}",
                "Avg Random Score": f"{quality_results['avg_similarity_random']:.3f}",
                "Separation Gap": f"{quality_results['similarity_gap']:.3f}",
            })

        except Exception as e:
            print(f"❌ Error benchmarking {model_id}: {e}\n")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print("- Higher 'Speed (texts/sec)' = Faster model")
    print("- Higher 'Separation Gap' = Better at distinguishing similar vs random pairs")
    print("- Lower 'Avg Random Score' = Better separation baseline")
    print(f"{'='*80}\n")

    return results_df


# %% [markdown]
# ### Run Benchmark
#
# Uncomment to benchmark all models:
# ```python
# benchmark_results = run_full_benchmark(sample_size=500, batch_size=32)
# ```

# %% [markdown]
# ## 12. Example Usage
#
# Run the cells below to test the pipeline with example queries.

# %% [markdown]
# ### Example: Ingest Articles
#
# Uncomment and run to ingest articles into Qdrant:
# ```python
# ingest_parquet_to_qdrant(
#     DATA_PARQUET_PATH,
#     client,
#     QDRANT_COLLECTION_NAME,
#     model,
#     rowgroup_batch=ROWGROUP_BATCH,
#     encode_batch=ENCODE_BATCH,
#     max_rows=ARTICLES_NUMBER,
# )
# ```

# %% [markdown]
# ### Example: Search by Title
#
# Uncomment and run to search for similar articles:
# ```python
# search_by_title_or_text(
#     df,
#     client,
#     QDRANT_COLLECTION_NAME,
#     model,
#     query_title="Math",
#     top_k=5
# )
# ```

# %% [markdown]
# ### Example: Find Similar Articles
#
# Uncomment and run to find articles similar to specific IDs:
# ```python
# find_similar_articles(
#     df,
#     client,
#     QDRANT_COLLECTION_NAME,
#     model,
#     article_ids=[189, 205],
#     top_k=5
# )
# ```

# %% [markdown]
# ### Example: Export to CSV
#
# Uncomment and run to export data:
# ```python
# export_df_to_csv(df, ARTICLES_CSV_PATH)
# ```

# %% [markdown]
# ## 13. Link Prediction & Chunking System
#
# Core functionality for reconstructing Wikipedia links using semantic similarity.

# %%
from dataclasses import dataclass
from typing import Any, List, Set, Tuple, Dict
import json


@dataclass
class ArticleChunk:
    """Represents a chunk of an article with metadata."""
    article_id: int
    article_title: str
    chunk_id: int  # 0-indexed within article
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] | None = None


def chunk_article(
    article_id: int,
    article_title: str,
    text: str,
    chunk_size: int = 256,  # words
    overlap: int = 64,      # words
) -> List[ArticleChunk]:
    """
    Split article into overlapping chunks by words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    stride = chunk_size - overlap
    if stride <= 0:
        stride = chunk_size

    for i in range(0, len(words), stride):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 50:  # Skip very small chunks at the end
            continue

        chunk_text = " ".join(chunk_words)
        chunks.append(ArticleChunk(
            article_id=article_id,
            article_title=article_title,
            chunk_id=len(chunks),
            text=chunk_text,
            start_pos=i,
            end_pos=i + len(chunk_words),
        ))

    return chunks


def split_sentences_with_positions(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences and track word start/end positions.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    positioned = []
    cursor = 0
    for sentence in sentences:
        cleaned = clean_text(sentence)
        words = cleaned.split()
        if not words:
            continue
        start = cursor
        end = cursor + len(words)
        positioned.append((sentence, start, end))
        cursor = end
    return positioned


def chunk_article_by_sentences(
    article_id: int,
    article_title: str,
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
    **kwargs,
) -> List[ArticleChunk]:
    """
    Split article into overlapping chunks based on sentences.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    stride = sentences_per_chunk - overlap_sentences
    if stride <= 0:
        stride = 1

    word_position = 0
    for i in range(0, len(sentences), stride):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = " ".join(chunk_sentences)
        if len(chunk_text.split()) < 10:
            continue

        chunks.append(ArticleChunk(
            article_id=article_id,
            article_title=article_title,
            chunk_id=len(chunks),
            text=chunk_text,
            start_pos=word_position,
            end_pos=word_position + len(chunk_text.split()),
        ))
        word_position += len(" ".join(chunk_sentences[:stride]).split())

    return chunks

def chunk_article_by_linked_sentences(
    article_id: int,
    article_title: str,
    text: str,
    include_neighbor_sentences: int = 1,
    min_words: int = 5,
    **kwargs, # to accept unused arguments
) -> List[ArticleChunk]:
    """
    Splits an article into chunks, where each chunk is a sentence containing a link.
    Optionally includes neighboring sentences for a bit more context.
    """
    sentences = split_sentences_with_positions(text)
    if not sentences:
        return []

    chunks = []
    for idx, (sentence, start, end) in enumerate(sentences):
        if "&lt;a href=" not in sentence:
            continue

        window_start = max(0, idx - include_neighbor_sentences)
        window_end = min(len(sentences), idx + include_neighbor_sentences + 1)
        window_sentences = [clean_text(sentences[j][0]) for j in range(window_start, window_end)]
        chunk_text = " ".join(s for s in window_sentences if s)

        if len(chunk_text.split()) < min_words:
            continue

        chunks.append(ArticleChunk(
            article_id=article_id,
            article_title=article_title,
            chunk_id=len(chunks),
            text=chunk_text,
            start_pos=sentences[window_start][1],
            end_pos=sentences[window_end - 1][2],
            metadata={"chunking_method": "linked_sentences", "source_sentence_index": idx},
        ))
    return chunks


def link_likelihood_score(sentence: str) -> float:
    """
    Heuristic score for how likely a sentence contains a link.
    """
    tokens = sentence.split()
    length = len(tokens)
    if length == 0:
        return 0.0

    caps = sum(1 for t in tokens if t[:1].isupper())
    cap_ratio = caps / length
    link_tag = 1.0 if "&lt;a href=" in sentence else 0.0
    punctuation_bonus = 0.1 if any(p in sentence for p in ("(", ")", ":", ";")) else 0.0
    long_bonus = 0.1 if length > 12 else 0.0

    score = 0.5 * cap_ratio + 0.3 * link_tag + punctuation_bonus + long_bonus
    return min(score, 1.0)


def chunk_article_by_link_likelihood(
    article_id: int,
    article_title: str,
    text: str,
    min_score: float = 0.35,
    include_neighbor_sentences: int = 0,
    min_words: int = 6,
    **kwargs,
) -> List[ArticleChunk]:
    """
    Chunk sentences that look link-heavy using a lightweight heuristic scorer.
    """
    sentences = split_sentences_with_positions(text)
    if not sentences:
        return []

    chunks = []
    for idx, (sentence, start, end) in enumerate(sentences):
        score = link_likelihood_score(sentence)
        if score < min_score:
            continue

        window_start = max(0, idx - include_neighbor_sentences)
        window_end = min(len(sentences), idx + include_neighbor_sentences + 1)
        window_sentences = [clean_text(sentences[j][0]) for j in range(window_start, window_end)]
        chunk_text = " ".join(s for s in window_sentences if s)

        if len(chunk_text.split()) < min_words:
            continue

        chunks.append(ArticleChunk(
            article_id=article_id,
            article_title=article_title,
            chunk_id=len(chunks),
            text=chunk_text,
            start_pos=sentences[window_start][1],
            end_pos=sentences[window_end - 1][2],
            metadata={
                "chunking_method": "link_likelihood",
                "source_sentence_index": idx,
                "link_likelihood": round(score, 3),
            },
        ))

    return chunks


def chunk_article_by_clauses(
    article_id: int,
    article_title: str,
    text: str,
    max_clause_words: int = 40,
    min_clause_words: int = 8,
    join_adjacent: int = 1,
    **kwargs,
) -> List[ArticleChunk]:
    """
    Create micro-chunks by splitting sentences into clauses using punctuation.
    """
    import re
    clauses = re.split(r'(?<=[.!?;:])\s+|\s*,\s*', text)
    clauses = [clean_text(c) for c in clauses if c.strip()]

    chunks = []
    word_cursor = 0

    for i in range(len(clauses)):
        window = clauses[i:i + join_adjacent + 1]
        combined = " ".join(window).strip()
        words = combined.split()

        if len(words) < min_clause_words:
            word_cursor += len(words)
            continue

        if len(words) > max_clause_words:
            combined = " ".join(words[:max_clause_words])
            words = combined.split()

        chunks.append(ArticleChunk(
            article_id=article_id,
            article_title=article_title,
            chunk_id=len(chunks),
            text=combined,
            start_pos=word_cursor,
            end_pos=word_cursor + len(words),
            metadata={"chunking_method": "clauses"},
        ))

        word_cursor += len(words)

    return chunks


def chunk_article_whole(
    article_id: int,
    article_title: str,
    text: str,
    **kwargs, # to accept unused arguments
) -> List[ArticleChunk]:
    """
    Treats the whole article as a single chunk.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []
    
    return [ArticleChunk(
        article_id=article_id,
        article_title=article_title,
        chunk_id=0,
        text=cleaned_text,
        start_pos=0,
        end_pos=len(cleaned_text.split()),
    )]


def ingest_chunks_to_qdrant(
    df: pl.DataFrame,
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    encode_batch: int = 128,
    max_articles: Optional[int] = None,
    chunking_method: str = "words",
    timeout_seconds: Optional[float] = None,
    start_time: Optional[float] = None,
    **chunking_options,
):
    """
    Chunk articles and ingest into Qdrant with metadata.
    """
    n_articles = df.height if max_articles is None else min(df.height, max_articles)

    all_chunks = []
    print(f"Chunking {n_articles:,} articles using '{chunking_method}' method...")

    chunking_functions = {
        "words": chunk_article,
        "sentences": chunk_article_by_sentences,
        "linked_sentences": chunk_article_by_linked_sentences,
        "whole_article": chunk_article_whole,
        "link_likelihood": chunk_article_by_link_likelihood,
        "clauses": chunk_article_by_clauses,
    }

    if chunking_method not in chunking_functions:
        raise ValueError(f"Unknown chunking method: {chunking_method}")

    chunk_fn = chunking_functions[chunking_method]

    for i in tqdm(range(n_articles), desc="Chunking articles"):
        if timeout_seconds and start_time and (time.time() - start_time) > timeout_seconds:
            raise TimeoutError(f"Chunking exceeded {timeout_seconds}s")

        row = df[i]
        article_id = int(row["id"][0])
        title = row["title"][0]
        # For linked_sentences, we need the raw text with HTML tags
        if chunking_method in {"linked_sentences", "link_likelihood"}:
            text = row["text"][0]
        else:
            text = clean_text(row["text"][0])

        chunks = chunk_fn(article_id, title, text, **chunking_options)
        all_chunks.extend(chunks)

    if n_articles > 0:
        print(f"✓ Created {len(all_chunks):,} chunks from {n_articles:,} articles")
        print(f"  Average: {len(all_chunks) / n_articles:.1f} chunks/article")
    else:
        print(f"✓ Created 0 chunks from 0 articles")


    # Encode and ingest in batches
    if all_chunks:
        print("Encoding and ingesting chunks...")
        for i in tqdm(range(0, len(all_chunks), encode_batch), desc="Ingesting"):
            if timeout_seconds and start_time and (time.time() - start_time) > timeout_seconds:
                raise TimeoutError(f"Ingestion exceeded {timeout_seconds}s")

            batch = all_chunks[i:i + encode_batch]

            texts = [f"query: {chunk.text}" for chunk in batch]

            vectors = model.encode(
                texts,
                batch_size=min(len(texts), encode_batch),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            points = []
            for j in range(len(batch)):
                payload = {
                    "article_id": batch[j].article_id,
                    "chunk_id": batch[j].chunk_id,
                    "title": batch[j].article_title,
                    "text": batch[j].text,
                }
                if batch[j].metadata:
                    payload.update(batch[j].metadata)

                points.append(
                    PointStruct(
                        id=batch[j].article_id * 10000 + batch[j].chunk_id,
                        vector=vectors[j].tolist(),
                        payload=payload,
                    )
                )

            client.upsert(collection_name=collection_name, points=points, wait=True)
        print(f"✓ Ingested {len(all_chunks):,} chunks")
    else:
        print("No chunks to ingest.")


def predict_article_links(
    client: QdrantClient,
    collection_name: str,
    article_id: int,
    top_k: int = 20,
    min_chunks: int = 2,
    exclude_same_article: bool = True,
) -> List[Tuple[int, float]]:
    """
    Predict which articles should link from a given article.

    Strategy:
    1. Retrieve all chunks for the source article
    2. For each chunk, find top-k similar chunks
    3. Aggregate by target article_id
    4. Rank by: (# of chunk matches) * (avg similarity)

    Args:
        client: Qdrant client
        collection_name: Collection name
        article_id: Source article ID
        top_k: Top chunks to retrieve per source chunk
        min_chunks: Minimum chunks matched to consider a link
        exclude_same_article: Filter out same-article links

    Returns:
        List of (target_article_id, confidence_score) sorted by score
    """
    # Get all chunks for source article
    source_chunks = client.scroll(
        collection_name=collection_name,
        scroll_filter=rest.Filter(
            must=[rest.FieldCondition(
                key="article_id",
                match=rest.MatchValue(value=article_id)
            )]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=True,
    )[0]

    if not source_chunks:
        return []

    # For each source chunk, find similar chunks
    target_matches = {}  # {target_article_id: [similarities]}

    for chunk in source_chunks:
        results = client.search(
            collection_name=collection_name,
            query_vector=chunk.vector,
            limit=top_k,
            with_payload=True,
        )

        for result in results:
            target_id = result.payload["article_id"]

            # Skip same article if requested
            if exclude_same_article and target_id == article_id:
                continue

            if target_id not in target_matches:
                target_matches[target_id] = []
            target_matches[target_id].append(result.score)

    # Aggregate and rank
    predictions = []
    for target_id, similarities in target_matches.items():
        if len(similarities) < min_chunks:
            continue

        # Score = (# chunks matched) * (average similarity)
        score = len(similarities) * np.mean(similarities)
        predictions.append((target_id, score))

    # Sort by score descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions


def load_ground_truth_links(wikilink_file: Path) -> Dict[int, Set[int]]:
    """
    Load ground truth links from WikiLinkGraphs dataset.

    Expected format: JSONL with {source_id, target_ids, ...}

    Args:
        wikilink_file: Path to ground truth file

    Returns:
        Dict mapping source_article_id -> set of target_article_ids
    """
    ground_truth = {}

    if not wikilink_file.exists():
        print(f"⚠️  Ground truth file not found: {wikilink_file}")
        return ground_truth

    with wikilink_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                source_id = data.get("source_id")
                target_ids = data.get("target_ids", [])

                if source_id and target_ids:
                    ground_truth[source_id] = set(target_ids)
            except json.JSONDecodeError:
                continue

    print(f"✓ Loaded ground truth: {len(ground_truth):,} articles with links")
    return ground_truth


def evaluate_link_predictions(
    client: QdrantClient,
    collection_name: str,
    df: pl.DataFrame,
    ground_truth_links: Dict[int, Set[int]],
    article_ids: List[int],
    top_k_list: List[int] = [5, 10, 20],
) -> Dict:
    """
    Evaluate link prediction performance against ground truth.

    Metrics:
    - Precision@k: % of predicted links that are correct
    - Recall@k: % of true links that were found
    - F1@k: Harmonic mean of precision and recall
    - MRR: Mean Reciprocal Rank of first correct prediction

    Args:
        client: Qdrant client
        collection_name: Collection name
        df: Articles DataFrame
        ground_truth_links: Dict mapping article_id -> set of linked article_ids
        article_ids: List of articles to evaluate
        top_k_list: List of k values for metrics

    Returns:
        Dict with evaluation metrics
    """
    results = {k: {"precision": [], "recall": [], "f1": []} for k in top_k_list}
    mrr_scores = []

    print(f"\nEvaluating {len(article_ids)} articles...")

    for article_id in tqdm(article_ids, desc="Evaluating"):
        if article_id not in ground_truth_links:
            continue

        true_links = ground_truth_links[article_id]
        if not true_links:
            continue

        # Predict links
        predictions = predict_article_links(
            client, collection_name, article_id,
            top_k=50,  # Get more for evaluation
            min_chunks=2,
        )

        predicted_ids = [pred[0] for pred in predictions]

        # Calculate MRR
        for rank, pred_id in enumerate(predicted_ids, 1):
            if pred_id in true_links:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

        # Calculate P/R/F1 at different k
        for k in top_k_list:
            top_k_preds = set(predicted_ids[:k])

            true_positives = len(top_k_preds & true_links)
            precision = true_positives / k if k > 0 else 0
            recall = true_positives / len(true_links) if true_links else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[k]["precision"].append(precision)
            results[k]["recall"].append(recall)
            results[k]["f1"].append(f1)

    # Aggregate results
    metrics = {}
    for k in top_k_list:
        metrics[f"P@{k}"] = np.mean(results[k]["precision"])
        metrics[f"R@{k}"] = np.mean(results[k]["recall"])
        metrics[f"F1@{k}"] = np.mean(results[k]["f1"])
    metrics["MRR"] = np.mean(mrr_scores)

    return metrics


# %% [markdown]
# ## 14. Chunking Experimentation Framework
#
# Systematic framework for testing different chunking strategies to optimize
# Wikipedia link prediction performance.

# %%
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class ChunkingExperiment:
    """
    Stores metadata and results for a single chunking experiment.

    Attributes:
        experiment_id: Unique identifier (timestamp-based)
        chunking_method: Strategy name (words, sentences, linked_sentences, clauses, link_likelihood)
        chunking_options: Extra options forwarded to chunker
        chunk_size: Number of words per chunk
        overlap: Number of words overlapping between chunks
        min_chunks: Minimum chunk matches required for link prediction
        max_articles: Total articles used in experiment
        test_articles: Number of articles evaluated
        collection_name: Qdrant collection name for this experiment
        metrics: Performance metrics (P@k, R@k, F1@k, MRR)
        timing: Execution time statistics
        status: Experiment status (running, completed, failed)
        error: Error message if failed
    """
    experiment_id: str
    chunking_method: str
    chunking_options: Dict[str, Any]
    chunk_size: int
    overlap: int
    min_chunks: int
    max_articles: int
    test_articles: int
    collection_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    status: str = "initialized"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary for JSON serialization."""
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary of experiment."""
        lines = [
            f"Experiment ID: {self.experiment_id}",
            f"Status: {self.status}",
            f"Configuration:",
            f"  - Chunking method: {self.chunking_method}",
            f"  - Chunk size: {self.chunk_size} words",
            f"  - Overlap: {self.overlap} words",
            f"  - Min chunks threshold: {self.min_chunks}",
            f"  - Articles: {self.max_articles} (tested: {self.test_articles})",
        ]
        if self.chunking_options:
            lines.append(f"  - Options: {self.chunking_options}")

        if self.metrics:
            lines.append("Metrics:")
            for metric_name, value in sorted(self.metrics.items()):
                lines.append(f"  - {metric_name}: {value:.4f}")

        if self.timing:
            total_time = self.timing.get("total_seconds", 0)
            lines.append(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")

        if self.error:
            lines.append(f"Error: {self.error}")

        return "\n".join(lines)


def run_chunking_experiment(
    df: pl.DataFrame,
    client: QdrantClient,
    model: SentenceTransformer,
    ground_truth_links: Dict[int, Set[int]],
    chunking_method: str = "words",
    chunking_options: Optional[Dict[str, Any]] = None,
    chunk_size: int = 256,
    overlap: int = 64,
    min_chunks: int = 2,
    max_articles: int = 1000,
    test_articles: int = 100,
    encode_batch: int = 128,
    cleanup: bool = True,
    timeout_seconds: Optional[float] = None,
) -> ChunkingExperiment:
    """
    Run a complete chunking experiment with specified parameters.

    This function:
    1. Creates a temporary Qdrant collection
    2. Chunks articles with given parameters
    3. Ingests chunks into Qdrant
    4. Evaluates link prediction on test set
    5. Cleans up collection (optional)

    Args:
        df: Articles DataFrame (id, title, text, url)
        client: Qdrant client instance
        model: SentenceTransformer for encoding
        ground_truth_links: Dict mapping article_id -> set of linked article_ids
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        min_chunks: Minimum chunk matches for link prediction
        max_articles: Total articles to ingest
        test_articles: Number of articles to evaluate (from those with ground truth)
        encode_batch: Batch size for embedding generation
        cleanup: Whether to delete collection after experiment

    Returns:
        ChunkingExperiment object with results and metrics

    Example:
        >>> experiment = run_chunking_experiment(
        ...     df=df,
        ...     client=client,
        ...     model=model,
        ...     ground_truth_links=ground_truth,
        ...     chunk_size=256,
        ...     overlap=64,
        ...     max_articles=1000,
        ...     test_articles=100,
        ... )
        >>> print(experiment.summary())
    """
    import time

    chunking_options = chunking_options or {}

    # Create experiment metadata
    experiment_id = f"{chunking_method}_chunk_{chunk_size}_overlap_{overlap}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    collection_name = f"exp_{experiment_id}"

    experiment = ChunkingExperiment(
        experiment_id=experiment_id,
        chunking_method=chunking_method,
        chunking_options=chunking_options,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunks=min_chunks,
        max_articles=max_articles,
        test_articles=test_articles,
        collection_name=collection_name,
    )

    try:
        experiment.status = "running"
        start_time = time.time()
        def check_timeout(stage: str):
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                raise TimeoutError(f"Exceeded timeout ({timeout_seconds}s) during {stage}")

        # Step 1: Create collection
        print(f"\n{'='*70}")
        print(f"Experiment: {experiment_id}")
        print(f"{'='*70}")
        print(f"Configuration: chunk_size={chunk_size}, overlap={overlap}, min_chunks={min_chunks}")
        print(f"Chunking method: {chunking_method} | options: {chunking_options}")
        print(f"Articles: {max_articles} (test: {test_articles})")

        collection_start = time.time()
        vector_size = model.get_sentence_embedding_dimension()

        # Delete if exists (from previous failed run)
        if client.collection_exists(collection_name):
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine"),
        )
        print(f"✓ Created collection: {collection_name}")
        experiment.timing["collection_creation_seconds"] = time.time() - collection_start

        # Step 2: Chunk and ingest articles
        ingest_start = time.time()
        check_timeout("collection setup")
        ingest_chunks_to_qdrant(
            df=df,
            client=client,
            collection_name=collection_name,
            model=model,
            chunk_size=chunk_size,
            overlap=overlap,
            encode_batch=encode_batch,
            max_articles=max_articles,
            chunking_method=chunking_method,
            timeout_seconds=timeout_seconds,
            start_time=start_time,
            **chunking_options,
        )
        experiment.timing["ingestion_seconds"] = time.time() - ingest_start

        # Verify ingestion
        total_chunks = client.count(collection_name=collection_name, exact=True).count
        print(f"✓ Ingested {total_chunks:,} chunks")

        # Step 3: Select test articles (those with ground truth links)
        eval_start = time.time()
        check_timeout("ingestion")
        available_article_ids = [
            aid for aid in ground_truth_links.keys()
            if aid < max_articles  # Only test articles that were ingested
        ]

        if not available_article_ids:
            raise ValueError("No articles with ground truth found in ingested range")

        # Sample test articles
        import random
        random.seed(42)
        test_article_ids = random.sample(
            available_article_ids,
            min(test_articles, len(available_article_ids))
        )

        print(f"✓ Selected {len(test_article_ids)} test articles from {len(available_article_ids)} available")

        # Step 4: Evaluate link prediction
        print("Evaluating link predictions...")
        metrics = evaluate_link_predictions(
            client=client,
            collection_name=collection_name,
            df=df,
            ground_truth_links=ground_truth_links,
            article_ids=test_article_ids,
            top_k_list=[5, 10, 20],
        )
        check_timeout("evaluation")

        experiment.metrics = metrics
        experiment.timing["evaluation_seconds"] = time.time() - eval_start
        experiment.timing["total_seconds"] = time.time() - start_time
        experiment.status = "completed"

        # Display results
        print(f"\n{'='*70}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*70}")
        print(f"Precision@5:  {metrics.get('P@5', 0):.4f}")
        print(f"Precision@10: {metrics.get('P@10', 0):.4f}")
        print(f"Recall@5:     {metrics.get('R@5', 0):.4f}")
        print(f"Recall@10:    {metrics.get('R@10', 0):.4f}")
        print(f"F1@5:         {metrics.get('F1@5', 0):.4f}")
        print(f"F1@10:        {metrics.get('F1@10', 0):.4f}")
        print(f"MRR:          {metrics.get('MRR', 0):.4f}")
        print(f"Total time:   {experiment.timing['total_seconds']:.2f}s")
        print(f"{'='*70}\n")

    except Exception as e:
        experiment.status = "failed"
        experiment.error = str(e)
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup collection if requested
        if cleanup and client.collection_exists(collection_name):
            print(f"Cleaning up collection: {collection_name}")
            client.delete_collection(collection_name)
            print("✓ Collection deleted")

    return experiment


def compare_chunking_strategies(
    df: pl.DataFrame,
    client: QdrantClient,
    model: SentenceTransformer,
    ground_truth_links: Dict[int, Set[int]],
    max_articles: int = 1000,
    test_articles: int = 100,
    output_file: str = "chunking_experiments.json",
    parallel: bool = False,
    timeout_seconds_per_experiment: Optional[float] = None,
    target_f1_at_10: Optional[float] = None,
    configurations: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Compare multiple chunking strategies and identify the best configuration.

    Tests a mix of baselines and link-focused variants by default:
    - Word windows (small/medium/large/no-overlap/tiny)
    - Whole-article single chunk
    - Sentence-only chunks
    - Linked sentences with context
    - Clause-level micro-chunks
    - Heuristic link-likelihood filtering

    Args:
        df: Articles DataFrame
        client: Qdrant client instance
        model: SentenceTransformer for encoding
        ground_truth_links: Ground truth links for evaluation
        max_articles: Total articles to ingest per experiment
        test_articles: Number of articles to evaluate
        output_file: JSON file to save results
        parallel: Run experiments in parallel (not implemented yet)
        timeout_seconds_per_experiment: Optional timeout guard
        target_f1_at_10: Stop sweep once F1@10 meets/exceeds this threshold
        configurations: Optional override for custom experiment grid

    Returns:
        DataFrame with comparison results

    Example:
        >>> results_df = compare_chunking_strategies(
        ...     df=df,
        ...     client=client,
        ...     model=model,
        ...     ground_truth_links=ground_truth,
        ...     max_articles=1000,
        ...     test_articles=100,
        ... )
        >>> print(results_df)
    """
    print("\n" + "="*80)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*80)
    print(f"Dataset: {max_articles} articles | Test set: {test_articles} articles")
    print(f"Ground truth: {len(ground_truth_links)} articles with links")
    print("="*80 + "\n")

    # Define chunking configurations to test
    if configurations is None:
        configurations = [
            {"name": "Small chunks", "chunking_method": "words", "chunk_size": 128, "overlap": 32, "min_chunks": 2},
            {"name": "Medium chunks", "chunking_method": "words", "chunk_size": 256, "overlap": 64, "min_chunks": 2},
            {"name": "Large chunks", "chunking_method": "words", "chunk_size": 512, "overlap": 128, "min_chunks": 2},
            {"name": "No overlap", "chunking_method": "words", "chunk_size": 256, "overlap": 0, "min_chunks": 2},
            {"name": "Sentence links + context", "chunking_method": "linked_sentences", "chunk_size": 64, "overlap": 0, "min_chunks": 1, "chunking_options": {"include_neighbor_sentences": 1, "min_words": 6}},
            {"name": "Single sentences", "chunking_method": "sentences", "chunk_size": 0, "overlap": 0, "min_chunks": 1, "chunking_options": {"sentences_per_chunk": 1, "overlap_sentences": 0}},
            {"name": "Clause micro-chunks", "chunking_method": "clauses", "chunk_size": 64, "overlap": 0, "min_chunks": 1, "chunking_options": {"max_clause_words": 40, "min_clause_words": 8, "join_adjacent": 1}},
            {"name": "Link-likelihood sentences", "chunking_method": "link_likelihood", "chunk_size": 64, "overlap": 0, "min_chunks": 1, "chunking_options": {"min_score": 0.35, "include_neighbor_sentences": 1, "min_words": 6}},
            {"name": "Whole article", "chunking_method": "whole_article", "chunk_size": 0, "overlap": 0, "min_chunks": 1},
            {"name": "Tiny windows", "chunking_method": "words", "chunk_size": 64, "overlap": 8, "min_chunks": 1},
        ]

    experiments = []

    # Run experiments sequentially (parallel not implemented for safety)
    for i, config in enumerate(configurations, 1):
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(configurations)}: {config['name']}")
        print(f"{'#'*80}\n")

        experiment = run_chunking_experiment(
            df=df,
            client=client,
            model=model,
            ground_truth_links=ground_truth_links,
            chunking_method=config.get("chunking_method", "words"),
            chunking_options=config.get("chunking_options", {}),
            chunk_size=config.get("chunk_size", 256),
            overlap=config.get("overlap", 64),
            min_chunks=config.get("min_chunks", 2),
            max_articles=max_articles,
            test_articles=test_articles,
            cleanup=True,  # Clean up after each experiment
            timeout_seconds=timeout_seconds_per_experiment,
        )

        experiments.append({
            "Configuration": config["name"],
            "Chunking Method": config.get("chunking_method", "words"),
            "Chunk Size": config["chunk_size"],
            "Overlap": config["overlap"],
            "Min Chunks": config["min_chunks"],
            "Options": config.get("chunking_options", {}),
            "P@5": experiment.metrics.get("P@5", 0.0),
            "P@10": experiment.metrics.get("P@10", 0.0),
            "R@5": experiment.metrics.get("R@5", 0.0),
            "R@10": experiment.metrics.get("R@10", 0.0),
            "F1@5": experiment.metrics.get("F1@5", 0.0),
            "F1@10": experiment.metrics.get("F1@10", 0.0),
            "MRR": experiment.metrics.get("MRR", 0.0),
            "Total Time (s)": experiment.timing.get("total_seconds", 0.0),
            "Status": experiment.status,
        })

        if target_f1_at_10 is not None and experiment.metrics.get("F1@10", 0.0) >= target_f1_at_10:
            print(f"Stopping early: target F1@10 {target_f1_at_10} reached by '{config['name']}'")
            break

    # Create comparison DataFrame
    results_df = pd.DataFrame(experiments)

    # Display comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80 + "\n")
    print(results_df.to_string(index=False))

    # Identify best configuration
    if len(results_df) > 0:
        best_f1_idx = results_df["F1@10"].idxmax()
        best_mrr_idx = results_df["MRR"].idxmax()

        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        print(f"Best F1@10: {results_df.loc[best_f1_idx, 'Configuration']}")
        print(f"  - F1@10: {results_df.loc[best_f1_idx, 'F1@10']:.4f}")
        print(f"  - Chunk size: {results_df.loc[best_f1_idx, 'Chunk Size']}, Overlap: {results_df.loc[best_f1_idx, 'Overlap']}")
        print()
        print(f"Best MRR: {results_df.loc[best_mrr_idx, 'Configuration']}")
        print(f"  - MRR: {results_df.loc[best_mrr_idx, 'MRR']:.4f}")
        print(f"  - Chunk size: {results_df.loc[best_mrr_idx, 'Chunk Size']}, Overlap: {results_df.loc[best_mrr_idx, 'Overlap']}")
        print(f"{'='*80}\n")

    # Save results to JSON
    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "max_articles": max_articles,
            "test_articles": test_articles,
            "experiments": experiments,
        }, f, indent=2)

    print(f"✓ Results saved to: {output_path.absolute()}")

    return results_df


# %% [markdown]
# ### Example: Run Chunking Experiments
#
# ```python
# # Load ground truth links
# ground_truth = load_ground_truth_links(Path("wikilink_graphs_fr.jsonl"))
#
# # Run single experiment
# experiment = run_chunking_experiment(
#     df=df,
#     client=client,
#     model=model,
#     ground_truth_links=ground_truth,
#     chunk_size=256,
#     overlap=64,
#     max_articles=1000,
#     test_articles=100,
# )
# print(experiment.summary())
#
# # Compare multiple strategies
# results = compare_chunking_strategies(
#     df=df,
#     client=client,
#     model=model,
#     ground_truth_links=ground_truth,
#     max_articles=1000,
#     test_articles=100,
# )
# ```


# %% [markdown]
# ## 15. Smoke Test
#
# Automated test to verify the pipeline works end-to-end.

# %%
def main():
    """
    Smoke test for the complete pipeline:
    1. Ensure Qdrant collection exists
    2. Ingest sample articles if collection is empty
    3. Run a test query to verify search functionality
    """
    print("\n" + "="*60)
    print("Smoke Test: Qdrant + Embeddings Pipeline")
    print("="*60 + "\n")

    # Ensure collection exists
    ensure_collection(client, QDRANT_COLLECTION_NAME, VECTOR_SIZE)

    # Check current state
    total_before = count_points(client, QDRANT_COLLECTION_NAME)

    # Ingest sample data if needed
    if total_before <= INGEST_FOR_SMOKE_TEST:
        print(f"\nIngesting {INGEST_FOR_SMOKE_TEST:,} articles for testing...")
        ingest_parquet_to_qdrant(
            DATA_PARQUET_PATH,
            client,
            QDRANT_COLLECTION_NAME,
            model,
            rowgroup_batch=ROWGROUP_BATCH,
            encode_batch=ENCODE_BATCH,
            max_rows=INGEST_FOR_SMOKE_TEST,
        )
        total_after = count_points(client, QDRANT_COLLECTION_NAME)
        print(f"Ingestion complete: {total_after:,} total points")
    else:
        print(f"Collection already populated ({total_before:,} points), skipping ingestion")

    # Test search functionality
    print("\nRunning test query for 'trottoir'...")
    search_by_title_or_text(
        df=df,
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        model=model,
        query_title="math",
        top_k=5,
    )

    print("\n" + "="*60)
    print("Smoke test completed successfully")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

# %%
