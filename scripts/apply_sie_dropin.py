#!/usr/bin/env python3
import io
import os
import re
import sys

TARGET = "docs/sie blog/optimizing-rag-with-hybrid-search-reranking.md"

REPLACEMENT = """## Taking Hybrid Search Further with Advanced Frameworks

SIE (Superlinked Inference Engine) is a production-grade Python inference server for embedding, reranking, and extraction. It provides the three primitives `encode`, `score`, and `extract`, plus multi-model serving, memory management, and hot reload, so retrieval and reranking can be expressed as a single inference flow.

A typical SIE-powered hybrid pipeline looks like this:

1. `encode` the query (dense, sparse, or both)
2. retrieve candidates from your vector store using the outputs
3. `score` the candidate set with a cross-encoder reranker
4. optionally `extract` structured context for generation or enrichment

This pattern is the recommended approach in the SIE docs and matches the RAG + reranking examples.

### Compact example

```python
from sie_sdk import SIEClient, Item
from qdrant_client import QdrantClient

sie = SIEClient("http://localhost:8080")
qdrant = QdrantClient(url="http://localhost:6333")

q_vec = sie.encode("sentence-transformers/all-mpnet-base-v2", Item(text="How does hybrid search improve RAG systems?"), output_types=["dense"])
hits = qdrant.search(collection_name="docs", query_vector=q_vec.dense, limit=100)

candidate_texts = [h.payload["text"] for h in hits]
scores = sie.score(model="cross-encoder/ms-marco-MiniLM-L-6-v2", query="How does hybrid search improve RAG systems?", documents=candidate_texts, top_k=10)

ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:10]
for hit, score in ranked:
    print(f"{score:.3f}\t{hit.payload.get('title','')}\t{hit.payload.get('text')[:180]}")
```

### Quick notes

SIE can return multiple outputs in one encode call, enabling single-call hybrid indexing.

For fusion workflows, encode dense and sparse with SIE, fuse candidates in the vector DB, then rerank with score.

### Get started

Book a demo to see SIE in action with your data, or explore the SIE open source repo on GitHub to try it yourself.
"""


def read(path):
    with io.open(path, "r", encoding="utf8") as f:
        return f.read()


def write(path, text):
    with io.open(path, "w", encoding="utf8") as f:
        f.write(text)


def find_bounds(text, header_line):
    pat = re.compile(r"(?m)^" + re.escape(header_line) + r"\s*$")
    m = pat.search(text)
    if not m:
        return None, None
    start = m.start()
    next_pat = re.compile(r"(?m)^##\s+")
    nm = next_pat.search(text, m.end())
    if nm:
        end = nm.start()
    else:
        end = len(text)
    return start, end


def main():
    if not os.path.exists(TARGET):
        print("ERROR: target file not found:", TARGET, file=sys.stderr)
        sys.exit(1)
    orig = read(TARGET)
    header = "## Taking Hybrid Search Further with Advanced Frameworks"
    start, end = find_bounds(orig, header)
    if start is None:
        print("ERROR: header not found in target file. Aborting.", file=sys.stderr)
        sys.exit(1)
    new_text = orig[:start] + REPLACEMENT + "\n\n" + orig[end:]
    write(TARGET, new_text)
    print("Rewrote", TARGET, "with SIE drop-in replacement.")
    if "bge-m3" in new_text:
        print("WARNING: 'bge-m3' found in output. Check replacement.", file=sys.stderr)


if __name__ == "__main__":
    main()
