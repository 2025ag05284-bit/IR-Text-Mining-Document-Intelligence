import os
from pathlib import Path
import pandas as pd

from modules.kmeans import ManualKMeans, build_tfidf_from_repo, run_kmeans
from modules.clustering_metrics import (
    compute_purity,
    compute_rand_index,
    compute_pair_confusion_counts,
    build_contingency_matrix
)


def main():
    dataset_path = None
    for candidate in ["data/processed_dataset.csv", "data/cleaned_dataset.csv", "data/raw_dataset.csv"]:
        if Path(candidate).exists():
            dataset_path = candidate
            break

    if dataset_path is None:
        raise FileNotFoundError("No dataset found in data/")

    df = pd.read_csv(dataset_path)
    print(f"[1/5] Loaded {dataset_path}: {len(df)} documents")

    X, vocab, _ = build_tfidf_from_repo(df, max_features=3000)
    print(f"[2/5] Built dense TF-IDF matrix: shape={X.shape}, vocab={len(vocab)}")

    k = int(df["category"].nunique()) if "category" in df.columns else 5
    kmeans = ManualKMeans(n_clusters=k, max_iter=100, tol=1e-4, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    contract_labels = run_kmeans(X, n_clusters=k, max_iter=100, random_state=42)
    assert len(contract_labels) == len(df)

    print(f"[3/5] K-Means (k={k}) converged in {kmeans.iterations_run} iterations | Inertia={kmeans.inertia_:.4f}")

    if "category" in df.columns:
        y_true = df["category"].astype(str).tolist()
        purity = compute_purity(y_true, labels)
        rand_idx = compute_rand_index(y_true, labels)
        pairs = compute_pair_confusion_counts(y_true, labels)

        print(f"[4/5] Purity: {purity:.4f} | Rand Index: {rand_idx:.4f}")
        print(f"      Pair Counts: {pairs}")
        print("\nContingency Matrix:")
        print(pd.DataFrame(build_contingency_matrix(y_true, labels)).T)

    os.makedirs("results", exist_ok=True)
    df_out = df.copy()
    df_out["cluster"] = labels
    export_cols = [c for c in ["document_id", "title", "category", "cluster"] if c in df_out.columns]
    df_out[export_cols].to_csv("results/clustering_results.csv", index=False)

    exp_rows = []
    for k_test in range(2, 9):
        km_t = ManualKMeans(n_clusters=k_test, max_iter=100, random_state=42).fit(X)
        row = {"k": k_test, "iterations": km_t.iterations_run, "inertia": round(km_t.inertia_, 4)}
        if "category" in df.columns:
            row["purity"] = round(compute_purity(df["category"].astype(str).tolist(), km_t.labels_), 4)
            row["rand_index"] = round(compute_rand_index(df["category"].astype(str).tolist(), km_t.labels_), 4)
        exp_rows.append(row)
    pd.DataFrame(exp_rows).to_csv("results/clustering_k_experiments.csv", index=False)

    print("\n[5/5] Saved results/clustering_results.csv and results/clustering_k_experiments.csv")


if __name__ == "__main__":
    main()