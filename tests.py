#!/usr/bin/env python3
"""
Extreme Boosting Optimizer - Maximize Recall & Precision
==========================================================

Uses extreme boosting factors and multiple signals to maximize F1.
"""

import gc, html, json, re, time
from pathlib import Path
from collections import defaultdict, Counter
from urllib.parse import unquote
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from qdrant import client, df, clean_text, model, VECTOR_SIZE
from qdrant_client.models import VectorParams, PointStruct

# Config
COLL = "extreme_v1"
MAX_ART = 10000
TEST_ART = 1000
BATCH = 512
TOP_K = 60

def extract_links(text):
    links = set()
    for m in re.finditer(r'<a\s+href="([^"]+)"', html.unescape(text)):
        h = unquote(m.group(1)).replace("_", " ").split("#")[0].strip()
        if h and len(h) > 1:
            links.add(h)
    return links

def load_gt(f, max_a):
    gt, t2id = {}, {}
    with open(f, 'r', encoding='utf-8') as file:
        for i, line in enumerate(tqdm(file, desc="Loading", total=max_a)):
            if i >= max_a:
                break
            try:
                d = json.loads(line)
                aid, title, text = int(d.get("id", i)), d.get("title", "").strip(), d.get("text", "")
                links = extract_links(text)
                if links:
                    gt[aid] = links
                if title:
                    t2id[title] = aid
            except:
                pass
    print(f"✓ {len(gt)} articles with links")
    return gt, t2id

def entities(text):
    return set(re.findall(r'\b[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][a-zàâäçéèêëîïôöùûüÿ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][a-zàâäçéèêëîïôöùûüÿ]+)*', text))

def extreme_boost(src_title, src_text, preds, titles):
    """Extreme boosting with multiple signals."""
    src_ent = entities(src_text)
    src_words = set(src_title.lower().split())
    src_lower = src_text.lower()

    boosted = []
    for aid, score in preds:
        title = titles.get(aid, "")

        # Title exact match in text (10x boost!)
        if len(title) > 3 and title.lower() in src_lower:
            score *= 10.0

        # Title partial match (5x)
        title_words = title.lower().split()
        if any(w in src_lower for w in title_words if len(w) > 4):
            score *= 5.0

        # Entity overlap (exponential boost)
        ent_overlap = len(src_ent & entities(title))
        if ent_overlap > 0:
            score *= (3.0 ** ent_overlap)

        # Word overlap (multiplicative)
        word_overlap = len(src_words & set(title.lower().split()))
        if word_overlap > 0:
            score *= (2.0 * word_overlap)

        # Capitalization signal
        if title and title[0].isupper() and title[0] in src_text:
            score *= 1.5

        # Length bonus (longer titles = more specific)
        if len(title.split()) >= 2:
            score *= 1.3

        # Numeric overlap (dates, years)
        src_nums = set(re.findall(r'\b\d{3,4}\b', src_text))
        title_nums = set(re.findall(r'\b\d{3,4}\b', title))
        if src_nums & title_nums:
            score *= 2.5

        boosted.append((aid, score))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted

def chunk_extreme(aid, title, text):
    """Extreme chunking: title everywhere."""
    chunks = []

    # Whole article with title
    chunks.append({"id": aid*10000, "aid": aid, "title": title, "text": f"{title}. {text[:1000]}"})

    # Every sentence gets title prepended
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip().split()) >= 5]

    for i, sent in enumerate(sents[:20]):  # Max 20 sentences
        chunks.append({
            "id": aid*10000 + i + 1,
            "aid": aid,
            "title": title,
            "text": f"{title}. {sent}"  # Title + sentence
        })

    return chunks

def run():
    print("\n" + "="*80)
    print("EXTREME BOOSTING OPTIMIZER")
    print("="*80)
    print(f"Target: F1@10 > 0.30")
    print("="*80 + "\n")

    start = time.time()

    # Load
    gt, t2id = load_gt(Path("articles_fr_withLinks.json"), MAX_ART)
    if not gt:
        return

    # Prep
    print("Preparing...")
    subset = df.head(MAX_ART)
    titles = {int(subset[i]["id"][0]): subset[i]["title"][0] for i in range(len(subset))}

    # Collection
    print("Creating collection...")
    if client.collection_exists(COLL):
        client.delete_collection(COLL)
    client.create_collection(COLL, vectors_config=VectorParams(size=VECTOR_SIZE, distance="Cosine"))

    # Chunk
    print("Chunking...")
    all_c = []
    c2a = {}

    for i in tqdm(range(len(subset)), desc="Chunk"):
        row = subset[i]
        aid, title, text = int(row["id"][0]), row["title"][0], clean_text(row["text"][0])
        for c in chunk_extreme(aid, title, text):
            all_c.append(c)
            c2a[c["id"]] = aid

    print(f"✓ {len(all_c)} chunks ({len(all_c)/len(subset):.1f}/article)")

    # Ingest
    for i in tqdm(range(0, len(all_c), BATCH), desc="Ingest"):
        batch = all_c[i:i+BATCH]
        texts = [f"query: {c['text']}" for c in batch]
        vecs = model.encode(texts, batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        points = [PointStruct(id=batch[j]["id"], vector=vecs[j].tolist(),
                             payload={"article_id": batch[j]["aid"], "title": batch[j]["title"], "text": batch[j]["text"]})
                 for j in range(len(batch))]
        client.upsert(collection_name=COLL, points=points, wait=True)

    print(f"✓ Ingested")
    del all_c
    gc.collect()

    # Predict
    print("Predicting...")
    test_ids = [aid for aid in gt.keys() if aid < MAX_ART][:TEST_ART]

    all_p, all_gt = {}, {}

    for idx, aid in enumerate(tqdm(test_ids, desc="Predict")):
        row = subset.filter(pl.col("id") == aid)
        if len(row) == 0:
            continue

        title, text = row["title"][0], clean_text(row["text"][0])

        # Query with title
        qv = model.encode([f"query: {title}. {text[:800]}"], normalize_embeddings=True, show_progress_bar=False)[0]

        # Search
        res = client.search(COLL, query_vector=qv.tolist(), limit=TOP_K*3)

        # Aggregate
        ascores = defaultdict(lambda: {"max": 0.0, "sum": 0.0, "cnt": 0})
        for r in res:
            caid = r.payload["article_id"]
            if caid == aid:
                continue
            ascores[caid]["max"] = max(ascores[caid]["max"], r.score)
            ascores[caid]["sum"] += r.score
            ascores[caid]["cnt"] += 1

        # Score: weighted combination
        preds = [(caid, s["max"] * s["cnt"] ** 1.5 + s["sum"]) for caid, s in ascores.items()]

        # EXTREME BOOST
        preds = extreme_boost(title, text, preds, titles)
        all_p[aid] = preds[:TOP_K]

        # GT
        gt_t = gt.get(aid, set())
        all_gt[aid] = {t2id[t] for t in gt_t if t in t2id}

        if idx % 20 == 0:
            gc.collect()

    # Eval
    print("Evaluating...")
    metrics = {k: [] for k in ["P@5", "P@10", "P@20", "R@5", "R@10", "R@20", "F1@5", "F1@10", "F1@20", "MRR"]}

    for aid, preds in all_p.items():
        gt_ids = all_gt.get(aid, set())
        if not gt_ids:
            continue

        pred_ids = [pid for pid, _ in preds]

        # MRR
        mrr = 0.0
        for rank, pid in enumerate(pred_ids, 1):
            if pid in gt_ids:
                mrr = 1.0 / rank
                break
        metrics["MRR"].append(mrr)

        # P/R/F1
        for k in [5, 10, 20]:
            topk = set(pred_ids[:k])
            tp = len(topk & gt_ids)
            p = tp / k if k else 0
            r = tp / len(gt_ids) if gt_ids else 0
            f1 = 2*p*r/(p+r) if p+r else 0
            metrics[f"P@{k}"].append(p)
            metrics[f"R@{k}"].append(r)
            metrics[f"F1@{k}"].append(f1)

    final = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
    total = time.time() - start

    # Report
    print("\n" + "="*80)
    print("EXTREME BOOSTING RESULTS")
    print("="*80)
    for k in [5, 10, 20]:
        print(f"P@{k}: {final[f'P@{k}']:.4f} | R@{k}: {final[f'R@{k}']:.4f} | F1@{k}: {final[f'F1@{k}']:.4f}")
    print(f"MRR: {final['MRR']:.4f} | Time: {total:.1f}s")
    print("="*80 + "\n")

    # Save
    with open("extreme_boost_results.json", 'w') as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": final, "time": total}, f, indent=2)

    print("✓ Saved to extreme_boost_results.json")

    # Cleanup
    if client.collection_exists(COLL):
        client.delete_collection(COLL)
    gc.collect()

    print(f"\n🎯 F1@10: {final['F1@10']:.4f}")
    if final['F1@10'] >= 0.3:
        print("✅ TARGET ACHIEVED!")
    else:
        print(f"Gap: {0.3 - final['F1@10']:.4f}")

    return final

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"\n❌ {e}")
        import traceback
        traceback.print_exc()
