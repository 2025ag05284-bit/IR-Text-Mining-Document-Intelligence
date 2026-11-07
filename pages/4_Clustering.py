import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from modules.kmeans import (
    ManualKMeans,
    extract_dense_from_model,
    build_tfidf_from_repo
)
from modules.clustering_metrics import (
    compute_purity,
    compute_rand_index,
    compute_pair_confusion_counts,
    build_contingency_matrix
)

st.set_page_config(page_title="Clustering | K-Means", page_icon="🎯", layout="wide")
st.title("🎯 Unsupervised Document Clustering (Manual K-Means)")
st.markdown(
    "Manual K-Means clustering on length-normalized TF-IDF vectors, "
    "evaluated with manual Purity and Rand Index."
)
st.divider()

PROCESSED_PATH = Path("data/processed_dataset.csv")
CLEANED_PATH = Path("data/cleaned_dataset.csv")
RAW_PATH = Path("data/raw_dataset.csv")


@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)


def load_data():
    if PROCESSED_PATH.exists():
        return read_csv(str(PROCESSED_PATH))
    if "processed_df" in st.session_state:
        return st.session_state["processed_df"].copy()
    for p in (CLEANED_PATH, RAW_PATH):
        if p.exists():
            return read_csv(str(p))
    return None


df = load_data()
if df is None:
    st.warning("No dataset found in data/. Please run export_team_dataset.py or upload a dataset.")
    st.stop()

df = df.reset_index(drop=True)

st.sidebar.header("K-Means Configuration")
max_features = st.sidebar.slider("Top TF-IDF Features", min_value=500, max_value=5000, value=3000, step=500)
default_k = int(df["category"].nunique()) if "category" in df.columns else 5
k_val = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=min(default_k, 10))
max_iter = st.sidebar.slider("Max Iterations", min_value=10, max_value=200, value=100, step=10)
tol_val = st.sidebar.select_slider("Convergence Tolerance", options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4)
seed_val = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

if "tfidf_model" in st.session_state and st.session_state["tfidf_model"] is not None:
    model = st.session_state["tfidf_model"]
    X, vocab = extract_dense_from_model(model, max_features=max_features)
else:
    with st.spinner("Building TF-IDF vectors..."):
        X, vocab, model = build_tfidf_from_repo(df, max_features=max_features)
        st.session_state["tfidf_model"] = model

st.info(f"Documents: `{X.shape[0]:,}` | Active TF-IDF Features: `{X.shape[1]:,}`")

if st.button("Run Manual K-Means Clustering", type="primary"):
    with st.spinner(f"Running Manual K-Means (k={k_val})..."):
        kmeans = ManualKMeans(
            n_clusters=k_val,
            max_iter=max_iter,
            tol=tol_val,
            random_state=seed_val
        )
        kmeans.fit(X)
        labels = kmeans.labels_

        st.session_state["cluster_labels"] = labels
        st.session_state["kmeans_model"] = kmeans
        df["cluster"] = labels

        os.makedirs("results", exist_ok=True)
        export_cols = [c for c in ["document_id", "title", "category", "cluster"] if c in df.columns]
        df[export_cols].to_csv("results/clustering_results.csv", index=False)

    st.success(
        f"Converged in {kmeans.iterations_run} iterations "
        f"| Final Inertia (SSD): {kmeans.inertia_:.4f}"
    )

    if "category" in df.columns:
        y_true = df["category"].astype(str).tolist()
        purity = compute_purity(y_true, labels)
        rand_idx = compute_rand_index(y_true, labels)
        pair_counts = compute_pair_confusion_counts(y_true, labels)

        st.subheader("1. Evaluation Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Purity", f"{purity:.4f}")
        c2.metric("Rand Index", f"{rand_idx:.4f}")
        c3.metric("Iterations", f"{kmeans.iterations_run}")
        c4.metric("Inertia (SSD)", f"{kmeans.inertia_:.2f}")

        with st.expander("Pairwise Rand Index Counts (TP, TN, FP, FN)"):
            st.json(pair_counts)

        st.subheader("2. Contingency Matrix (Cluster vs. Ground Truth Category)")
        contingency_df = pd.DataFrame(build_contingency_matrix(y_true, labels)).T
        contingency_df["Total"] = contingency_df.sum(axis=1)
        contingency_df["Majority Category"] = contingency_df.drop(columns=["Total"]).idxmax(axis=1)
        st.dataframe(contingency_df, use_container_width=True)

    st.subheader("3. 2D PCA Cluster Visualization")
    pca = PCA(n_components=2, random_state=seed_val)
    coords_2d = pca.fit_transform(X)
    centroids_2d = pca.transform(kmeans.centroids)

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    scatter = ax_pca.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.75,
        edgecolors="k",
        linewidth=0.3,
        s=35
    )
    ax_pca.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c=range(k_val),
        cmap="tab10",
        marker="X",
        s=220,
        edgecolors="black",
        linewidth=1.5,
        label="Centroids"
    )
    ax_pca.set_title(f"K-Means Clusters in 2D PCA Space (k={k_val})")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend()
    fig_pca.colorbar(scatter, ax=ax_pca, label="Cluster ID")
    st.pyplot(fig_pca)
    plt.close(fig_pca)

    st.subheader("4. Convergence Trace")
    history_df = pd.DataFrame(kmeans.history)
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    with col_t2:
        fig_conv, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        ax1.plot(history_df["iteration"], history_df["inertia"], marker="o")
        ax1.set_title("Inertia per Iteration")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Inertia")

        ax2.plot(history_df["iteration"], history_df["centroid_shift"], marker="s", color="orange")
        ax2.set_title("Centroid Shift (L2 Norm)")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Shift")
        plt.tight_layout()
        st.pyplot(fig_conv)
        plt.close(fig_conv)

    st.subheader("5. Top Centroid Terms per Cluster")
    top_terms = kmeans.get_top_terms_per_cluster(vocab, top_n=8)
    profile_rows = []
    for c_id in range(k_val):
        count = int(np.sum(labels == c_id))
        terms_str = ", ".join([f"{w} ({score:.3f})" for w, score in top_terms[c_id]])
        profile_rows.append({
            "Cluster": f"Cluster {c_id}",
            "Documents": count,
            "Share": f"{(count / len(labels)) * 100:.1f}%",
            "Top Terms": terms_str
        })
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

st.divider()
st.subheader("6. Multi-K Experiment (k = 2 to 8)")
if st.button("Run Multi-K Comparison"):
    exp_records = []
    y_true = df["category"].astype(str).tolist() if "category" in df.columns else None

    with st.spinner("Running K-Means for k = 2..8..."):
        for k_test in range(2, 9):
            km_exp = ManualKMeans(n_clusters=k_test, max_iter=max_iter, tol=tol_val, random_state=seed_val)
            km_exp.fit(X)
            rec = {
                "k": k_test,
                "Iterations": km_exp.iterations_run,
                "Inertia": round(km_exp.inertia_, 4),
            }
            if y_true is not None:
                rec["Purity"] = round(compute_purity(y_true, km_exp.labels_), 4)
                rec["Rand Index"] = round(compute_rand_index(y_true, km_exp.labels_), 4)
            exp_records.append(rec)

    exp_df = pd.DataFrame(exp_records)
    os.makedirs("results", exist_ok=True)
    exp_df.to_csv("results/clustering_k_experiments.csv", index=False)
    st.dataframe(exp_df, use_container_width=True, hide_index=True)
    st.success("Saved results/clustering_k_experiments.csv")