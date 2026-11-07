import json
import math
import time
from pathlib import Path

import pandas as pd

from modules.inverted_index import (
    build_inverted_index,
    get_postings,
    search_and
)
from modules.tfidf import build_tfidf, vector_norm
from modules.similarity import (
    cosine_similarity,
    top_k_similar,
    top_k_for_query
)


# --------------------------------------------------
# 1. Small hand-checkable example
# --------------------------------------------------

toy_documents = [
    ["stock", "market", "rise", "market"],
    ["football", "match", "goal"],
    ["stock", "price", "fall"],
]

index = build_inverted_index(toy_documents)

print("Inverted index:")
for term in sorted(index):
    print(f"  {term:10} -> {get_postings(index, term)}")

assert index["market"] == {0: 2}
assert index["stock"] == {0: 1, 2: 1}
assert search_and(index, ["stock", "fall"]) == [2]

model = build_tfidf(toy_documents, tf_scheme="log", normalize=False)

# market: tf = 1 + log10(2), idf = log10(3 / 1)
market = model.term_to_index["market"]
expected = (1 + math.log10(2)) * math.log10(3)
assert abs(model.vectors[0][market] - expected) < 1e-12

# stock: tf = 1, idf = log10(3 / 2)
stock = model.term_to_index["stock"]
assert abs(model.vectors[0][stock] - math.log10(1.5)) < 1e-12

normalized = build_tfidf(toy_documents)

for vector in normalized.vectors:
    assert abs(vector_norm(vector) - 1) < 1e-9

# Docs 0 and 2 share "stock"; doc 1 shares nothing.
assert cosine_similarity(normalized.vectors[0], normalized.vectors[1]) == 0
assert cosine_similarity(normalized.vectors[0], normalized.vectors[2]) > 0
assert top_k_similar(normalized, 0, k=1)[0][0] == 2

print("\nToy example checks passed.")


# --------------------------------------------------
# 2. Shared BBC dataset
# --------------------------------------------------

DATA_PATH = Path("data/processed_dataset.csv")

if not DATA_PATH.exists():
    print(f"\n{DATA_PATH} not found; skipping corpus test.")
    raise SystemExit

df = pd.read_csv(DATA_PATH)
token_documents = [json.loads(tokens) for tokens in df["tokens"]]

start = time.time()
model = build_tfidf(token_documents)
elapsed = time.time() - start

print(f"\nBuilt TF-IDF for {model.num_documents} documents "
      f"in {elapsed:.2f}s")
print(f"Vocabulary size: {model.num_features:,}")
print(f"Non-zero entries: {model.non_zero_entries():,}")
print(f"Sparsity: {model.sparsity():.4%}")

assert model.num_documents == len(df)

norms = [vector_norm(vector) for vector in model.vectors]
print(f"Vector length range after normalization: "
      f"{min(norms):.6f} - {max(norms):.6f}")

print(f"\nTop terms of {df.loc[0, 'document_id']} "
      f"({df.loc[0, 'title']}):")
for term, weight in model.top_terms(0, n=8):
    print(f"  {term:15} {weight:.4f}")

start = time.time()
neighbours = top_k_similar(model, 0, k=5)
print(f"\nTop-5 similar to document 0 "
      f"({time.time() - start:.3f}s):")
for doc_index, score in neighbours:
    print(f"  {score:.4f}  {df.loc[doc_index, 'category']:13} "
          f"{df.loc[doc_index, 'title']}")

# Nearest neighbours should mostly share the category.
same_category = 0
sample = range(0, len(df), 25)
for doc_index in sample:
    best = top_k_similar(model, doc_index, k=1)
    if best and df.loc[best[0][0], "category"] == df.loc[doc_index, "category"]:
        same_category += 1
print(f"\n1-NN category agreement on {len(sample)} sampled docs: "
      f"{same_category / len(sample):.2%}")

print("\nQuery 'mobile phone broadband':")
for doc_index, score in top_k_for_query(
    model, ["mobile", "phone", "broadband"], k=3
):
    print(f"  {score:.4f}  {df.loc[doc_index, 'category']:13} "
          f"{df.loc[doc_index, 'title']}")

dense = model.to_dense(columns=model.top_columns(3000))
print(f"\nDense matrix for K-Means (top 3000 terms): {dense.shape}, "
      f"{dense.nbytes / 1e6:.1f} MB")
