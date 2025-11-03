# project-2-DCL-wikipedia

Dataset links : 

English wiki:
https://dumps.wikimedia.org/enwiki/20251020/enwiki-20251020-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/enwiki/20251020/enwiki-20251020-pages-articles-multistream-index.txt.bz2

from https://dumps.wikimedia.org/enwiki/20251020/

French wiki:

https://dumps.wikimedia.org/frwiki/20251101/frwiki-20251101-pages-articles-multistream.xml.bz2
https://dumps.wikimedia.org/frwiki/20251101/frwiki-20251101-pages-articles-multistream-index.txt.bz2

From : https://dumps.wikimedia.org/frwiki/20251101/


https://arxiv.org/abs/1902.04298 : Complete Links datasets ?

=> https://zenodo.org/records/2539424


Tools :

Reformat xml dumps into plaintext:
https://github.com/wolfgarbe/WikipediaExport


````markdown
# 🧠 Wikipedia Link Reconstruction via Semantic Retrieval (RAG-style)

## 📘 Project Overview
This project aims to reconstruct the **Wikipedia internal link graph** by leveraging modern **semantic embeddings** and **retrieval-based machine learning**.

We build a fully **local, deterministic pipeline** that:
1. Extracts Wikipedia article text from official dumps.
2. Removes internal links (`[[Target|Anchor]]`).
3. Embeds articles into a semantic vector space.
4. Uses vector similarity to predict and reconstruct links between related articles.
5. Evaluates predicted links against the real Wikipedia link graph (WikiLinkGraphs dataset).

---

## 🧩 1. Data Sources

| Dataset | Description | Language | Source |
|----------|--------------|-----------|---------|
| **Wikipedia dumps** | Full article content in Wikitext format | English & French | [https://dumps.wikimedia.org/enwiki/](https://dumps.wikimedia.org/enwiki/) and [https://dumps.wikimedia.org/frwiki/](https://dumps.wikimedia.org/frwiki/) |
| **WikiLinkGraphs** | Clean article-to-article link graph (ground truth) | English & French | [https://zenodo.org/record/2539424](https://zenodo.org/record/2539424) |

---

## ⚙️ 2. Preprocessing Pipeline

### Step 1 — Extract Articles
- Download the latest `*-pages-articles-multistream.xml.bz2` dump.
- Parse Wikitext using [`mwparserfromhell`](https://github.com/earwig/mwparserfromhell).
- Keep only **main namespace** articles.

### Step 2 — Clean Text
- Remove all MediaWiki markup, templates, and categories.
- Strip internal links:
  ```text
  [[New York City|The Big Apple]] → The Big Apple
  [[France]] → France
````

* Remove references (`<ref>...</ref>`) and other non-content markup.

### Step 3 — Store Metadata

Save each article as JSON:

```json
{
  "id": "12345",
  "title": "Paris",
  "text": "Paris is the capital and most populous city of France.",
  "links": ["France", "Seine"]
}
```

---

## 🧠 3. Embedding Model

We use a **local, open-source multilingual embedding model** from [Sentence-Transformers](https://www.sbert.net/):

* Model: `intfloat/e5-multilingual-large` or `sentence-transformers/all-mpnet-base-v2`
* Dimension: 768–1024
* Framework: PyTorch
* Output: Dense vector representation of article semantics

### Example

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/e5-multilingual-large")
embedding = model.encode(article_text, normalize_embeddings=True)
```

---

## 📦 4. Vector Index and Retrieval

We use **FAISS (Facebook AI Similarity Search)** to index and retrieve nearest articles locally.

### Step 1 — Build Index

```python
import faiss
import numpy as np

index = faiss.IndexFlatIP(768)  # cosine similarity via normalized vectors
index.add(np.array(article_embeddings))
```

### Step 2 — Query

For each article embedding, find top-K similar articles:

```python
D, I = index.search(np.array([query_vector]), k=10)
```

These nearest neighbors represent **predicted link targets**.

---

## 🕸️ 5. Link Graph Reconstruction

For each article:

1. Retrieve its top-K nearest articles.
2. Create directed edges `(source → target)` for these neighbors.
3. Save edges as CSV or NetworkX graph.

Example output:

```
source_title,target_title,similarity
Paris,France,0.89
Paris,Seine,0.85
```

---

## 🧪 6. Evaluation

To measure reconstruction quality, compare predicted links with the **WikiLinkGraphs** ground-truth edges.

### Metrics:

* Precision@K
* Recall@K
* F1-score
* Mean Average Precision (MAP)

Each article’s predicted links are validated against its known links in WikiLinkGraphs.

---

## 🔧 7. Implementation Details

| Component     | Library                                       | Purpose                              |
| ------------- | --------------------------------------------- | ------------------------------------ |
| Text parsing  | `mwparserfromhell`                            | Extract & clean Wikitext             |
| Embedding     | `sentence-transformers`                       | Convert articles to semantic vectors |
| Vector search | `faiss`                                       | Nearest-neighbor retrieval           |
| Evaluation    | `pandas`, `numpy`, `networkx`, `scikit-learn` | Metrics and graph handling           |

---

## 📈 8. Possible Extensions

* **Temporal graph reconstruction:** Repeat process on multiple yearly snapshots.
* **Multilingual graph merging:** Compare link structures between English ↔ French articles.
* **Fine-tuned model:** Train a small transformer to rank retrieved articles as "should link" vs. "should not link".
* **RAG generation:** Add an LLM layer (local LLaMA / Mistral) to generate natural link anchors.

---

## 🧰 9. Project Structure

```
wikipedia-link-reconstruction/
│
├── data/
│   ├── dumps/                # raw Wikipedia dumps (.xml.bz2)
│   ├── wikilinkgraphs/       # ground truth link graphs
│   ├── processed/            # cleaned text + embeddings
│
├── src/
│   ├── 01_parse_wikitext.py
│   ├── 02_clean_text.py
│   ├── 03_embed_articles.py
│   ├── 04_build_faiss_index.py
│   ├── 05_reconstruct_links.py
│   ├── 06_evaluate_links.py
│
├── notebooks/
│   └── exploration.ipynb     # for analysis and visualization
│
├── requirements.txt
└── README.md
```

---

## 💡 10. Summary

**Goal:**
Rebuild Wikipedia’s internal link structure using embeddings and retrieval.

**Pipeline:**

1. Parse → Clean → Embed → Index → Retrieve → Evaluate

**Key characteristics:**

* 100% local (no API calls)
* Fully reproducible academic setup
* Scalable to multiple languages
* Compatible with both semantic retrieval and RAG generation extensions

---

## 🧾 References

* **WikiLinkGraphs Dataset:**
  Piccardi, T. et al. (2018). *WikiLinkGraphs: A Complete, Large-Scale and Multilingual Link Dataset for Wikipedia.*
  [https://zenodo.org/record/2539424](https://zenodo.org/record/2539424)

* **Wikipedia Dumps:**
  [https://dumps.wikimedia.org](https://dumps.wikimedia.org)

* **FAISS:**
  Johnson et al., 2019. *Billion-scale similarity search with GPUs.*

* **Sentence-Transformers:**
  Reimers & Gurevych, 2019. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*


