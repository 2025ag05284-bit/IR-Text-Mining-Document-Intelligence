import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.preprocessing import preprocess_documents, preprocess_text
from modules.naive_bayes import ManualMultinomialNB, train_test_split_manual
from modules.classification_metrics import (
    compute_confusion_matrix,
    compute_accuracy,
    compute_macro_metrics,
    compute_per_class_metrics,
    classification_report_table
)
from modules.clustering_metrics import compute_purity, compute_rand_index

st.set_page_config(
    page_title="Classification | Naive Bayes",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Supervised Document Classification (Manual Naive Bayes)")
st.markdown(
    "Manual Multinomial Naive Bayes text classification with Laplace smoothing, "
    "pure Python evaluation metrics, and comparison against unsupervised clustering."
)
st.divider()

PROCESSED_PATH = Path("data/processed_dataset.csv")
CLEANED_PATH = Path("data/cleaned_dataset.csv")
RAW_PATH = Path("data/raw_dataset.csv")


@st.cache_data(show_spinner=False)
def load_csv(path_str):
    return pd.read_csv(path_str)


def get_dataset():
    if PROCESSED_PATH.exists():
        return load_csv(str(PROCESSED_PATH))
    if "processed_df" in st.session_state:
        return st.session_state["processed_df"].copy()
    for p in (CLEANED_PATH, RAW_PATH):
        if p.exists():
            return load_csv(str(p))
    return None


df = get_dataset()
if df is None:
    st.warning("No dataset found in data/. Please place raw_dataset.csv or cleaned_dataset.csv in data/.")
    st.stop()

df = df.reset_index(drop=True)

if "category" not in df.columns:
    st.error("Dataset must contain a 'category' column for classification.")
    st.stop()

# --------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------
st.sidebar.header("Naive Bayes Configuration")
test_size_pct = st.sidebar.slider("Test Split Size (%)", min_value=10, max_value=40, value=20, step=5)
alpha_val = st.sidebar.slider("Laplace Smoothing (α)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
seed_val = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)
stratify_toggle = st.sidebar.checkbox("Stratified Train/Test Split", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing Options")
opt_lowercase = st.sidebar.checkbox("Lowercase", value=True)
opt_stopwords = st.sidebar.checkbox("Remove Stopwords", value=True)
stem_lem_choice = st.sidebar.radio("Stemming / Lemmatization", ["None", "Porter Stemming", "WordNet Lemmatization"], index=0)

use_stemming = (stem_lem_choice == "Porter Stemming")
use_lemmatization = (stem_lem_choice == "WordNet Lemmatization")

# --------------------------------------------------
# PREPARE TOKENS
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_tokens(data_frame, lowercase, stopwords, stemming, lemmatization):
    # If standard preprocessing is chosen and precomputed tokens exist, use them
    if (lowercase and stopwords and not stemming and not lemmatization) and ("tokens" in data_frame.columns):
        sample = data_frame["tokens"].iloc[0]
        if isinstance(sample, list):
            return data_frame["tokens"].tolist()
        try:
            return [json.loads(t) for t in data_frame["tokens"]]
        except Exception:
            pass

    docs = (data_frame["title"].fillna("") + " " + data_frame["text"].fillna("")).tolist()
    return preprocess_documents(
        docs,
        lowercase=lowercase,
        remove_stopword=stopwords,
        stemming=stemming,
        lemmatization=lemmatization
    )


with st.spinner("Preparing corpus tokens..."):
    token_documents = extract_tokens(df, opt_lowercase, opt_stopwords, use_stemming, use_lemmatization)

labels = df["category"].astype(str).tolist()
categories = sorted(list(set(labels)))

# --------------------------------------------------
# TRAIN / TEST SPLIT & FIT
# --------------------------------------------------
test_ratio = test_size_pct / 100.0

X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_manual(
    token_documents,
    labels,
    test_size=test_ratio,
    random_state=seed_val,
    stratify=stratify_toggle
)

# Cache model in session state
model_key = f"nb_model_{alpha_val}_{test_size_pct}_{seed_val}_{stem_lem_choice}_{opt_stopwords}"

if model_key not in st.session_state:
    with st.spinner("Training Manual Multinomial Naive Bayes..."):
        nb = ManualMultinomialNB(alpha=alpha_val)
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        test_probs = nb.predict_proba(X_test)

        macro_res = compute_macro_metrics(y_test, y_pred, labels=categories)
        conf_mat, ordered_labels = compute_confusion_matrix(y_test, y_pred, labels=categories)
        report_data = classification_report_table(y_test, y_pred, labels=categories)

        st.session_state[model_key] = {
            "model": nb,
            "y_pred": y_pred,
            "test_probs": test_probs,
            "macro_res": macro_res,
            "conf_mat": conf_mat,
            "ordered_labels": ordered_labels,
            "report_data": report_data,
            "train_idx": train_idx,
            "test_idx": test_idx
        }

res = st.session_state[model_key]
nb = res["model"]
y_pred = res["y_pred"]
test_probs = res["test_probs"]
macro_res = res["macro_res"]
conf_mat = res["conf_mat"]
ordered_labels = res["ordered_labels"]
report_data = res["report_data"]

# --------------------------------------------------
# SECTION 1: CORPUS & SPLIT SUMMARY
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Documents", f"{len(df):,}")
col2.metric("Training Samples", f"{len(X_train):,} ({100-test_size_pct}%)")
col3.metric("Test Samples", f"{len(X_test):,} ({test_size_pct}%)")
col4.metric("Vocabulary Size", f"{nb.vocab_size_:,} terms")

st.markdown("---")

# --------------------------------------------------
# SECTION 2: CLASSIFICATION PERFORMANCE METRICS
# --------------------------------------------------
st.subheader("📊 Model Evaluation Metrics (Manual Implementation)")
st.caption("Calculated strictly via manual pure-Python functions without using scikit-learn.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{macro_res['accuracy'] * 100:.2f}%")
m2.metric("Macro Precision", f"{macro_res['macro_precision'] * 100:.2f}%")
m3.metric("Macro Recall", f"{macro_res['macro_recall'] * 100:.2f}%")
m4.metric("Macro F1-Score", f"{macro_res['macro_f1'] * 100:.2f}%")

st.markdown("#### Detailed Classification Report")
report_df = pd.DataFrame(report_data)
st.dataframe(
    report_df.style.format({
        "Precision": lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v),
        "Recall": lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v),
        "F1-Score": lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v),
        "Support": "{:,}"
    }),
    use_container_width=True
)

st.markdown("---")

# --------------------------------------------------
# SECTION 3: VISUALIZATIONS
# --------------------------------------------------
st.subheader("📈 Classification Visualizations")

tab_conf, tab_dist, tab_feats = st.tabs([
    "Confusion Matrix",
    "Class Distribution (Actual vs Predicted)",
    "Top Discriminative Terms"
])

with tab_conf:
    st.markdown("##### Confusion Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(conf_mat, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(ordered_labels)))
    ax.set_yticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=45, ha="left")
    ax.set_yticklabels(ordered_labels)
    ax.set_xlabel("Predicted Category", fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual Category", fontweight="bold")

    for i in range(len(ordered_labels)):
        for j in range(len(ordered_labels)):
            val = conf_mat[i][j]
            color = "white" if val > np.max(conf_mat) / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab_dist:
    st.markdown("##### Test Set Distribution: Ground-Truth vs Predictions")
    actual_counts = pd.Series(y_test).value_counts().reindex(ordered_labels, fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(ordered_labels, fill_value=0)

    dist_df = pd.DataFrame({
        "Category": ordered_labels,
        "Actual (Ground Truth)": actual_counts.values,
        "Predicted (Naive Bayes)": pred_counts.values
    }).set_index("Category")

    st.bar_chart(dist_df)

with tab_feats:
    st.markdown("##### Top Informative Terms per Category (Log-Odds Ratio)")
    st.caption("Identifies terms with the highest relative salience: log P(w | c) - log P(w | ¬c).")
    top_terms_dict = nb.get_top_features_per_class(top_n=8, method="log_odds")

    cols = st.columns(len(ordered_labels))
    for idx, c in enumerate(ordered_labels):
        with cols[idx]:
            st.markdown(f"**`{c}`**")
            for rank, (w, score) in enumerate(top_terms_dict.get(c, []), 1):
                st.write(f"{rank}. **{w}** (`{score:+.2f}`)")

st.markdown("---")

# --------------------------------------------------
# SECTION 4: INTERACTIVE DOCUMENT PREDICTOR
# --------------------------------------------------
st.subheader("🔍 Interactive Document Prediction Sandbox")
st.markdown("Test the trained Naive Bayes classifier with existing test documents or any custom text.")

pred_mode = st.radio("Input Source", ["Select from Test Set", "Custom Text Input"], horizontal=True)

if pred_mode == "Select from Test Set":
    test_sub_df = df.iloc[res["test_idx"]].reset_index(drop=True)
    selected_idx = st.selectbox(
        "Choose a test document:",
        options=range(len(test_sub_df)),
        format_func=lambda i: f"[{test_sub_df.iloc[i].get('document_id', f'Doc #{i}')}] - {test_sub_df.iloc[i].get('title', 'Untitled')[:70]}"
    )
    doc_row = test_sub_df.iloc[selected_idx]
    input_text = f"{doc_row.get('title', '')} {doc_row.get('text', '')}"
    true_label = doc_row.get("category", "Unknown")
    st.info(f"**True Category:** `{true_label}`")
else:
    input_text = st.text_area(
        "Enter document text to classify:",
        value="Apple unveiled its latest computer processor today, boosting stock market performance and tech investor confidence.",
        height=100
    )
    true_label = None

if input_text.strip():
    custom_tokens = preprocess_text(
        input_text,
        lowercase=opt_lowercase,
        remove_stopword=opt_stopwords,
        stemming=use_stemming,
        lemmatization=use_lemmatization
    )
    pred_label = nb.predict([custom_tokens])[0]
    prob_dict = nb.predict_proba([custom_tokens])[0]

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown("#### Classification Output")
        st.success(f"**Predicted Category:** `{pred_label}`")
        confidence_val = prob_dict[pred_label] * 100
        st.metric("Prediction Confidence", f"{confidence_val:.2f}%")
        if true_label:
            match = (pred_label == true_label)
            st.markdown(f"Match with Ground Truth: **{'✅ Correct' if match else '❌ Incorrect'}**")

    with res_col2:
        st.markdown("#### Class Probability Distribution")
        prob_series = pd.Series(prob_dict) * 100
        st.bar_chart(prob_series)

st.markdown("---")

# --------------------------------------------------
# SECTION 5: SUPERVISED VS UNSUPERVISED COMPARISON
# --------------------------------------------------
st.subheader("⚖️ Comparative Analysis: Supervised (Naive Bayes) vs Unsupervised (K-Means)")
st.markdown(
    "Assignment Requirement: Compare Naive Bayes classification against K-Means clustering "
    "using appropriate evaluation metrics (Accuracy & F1 vs Purity & Rand Index)."
)

# Check for Person 4 clustering results
cluster_csv_path = Path("results/clustering_results.csv")
kmeans_available = False
purity_val = None
rand_idx_val = None

if cluster_csv_path.exists():
    try:
        cluster_df = pd.read_csv(cluster_csv_path)
        if "category" in cluster_df.columns and "cluster" in cluster_df.columns:
            purity_val = compute_purity(cluster_df["category"], cluster_df["cluster"])
            rand_idx_val = compute_rand_index(cluster_df["category"], cluster_df["cluster"])
            kmeans_available = True
    except Exception:
        pass

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.markdown("#### 🎯 Supervised Learning: Naive Bayes")
    st.write(
        "Learns category-conditional term distributions with direct supervision "
        "from human ground-truth labels."
    )
    st.metric("Classification Accuracy", f"{macro_res['accuracy'] * 100:.2f}%")
    st.metric("Macro F1-Score", f"{macro_res['macro_f1'] * 100:.2f}%")
    st.caption("Evaluated on unseen test set (out-of-sample generalization).")

with comp_col2:
    st.markdown("#### 🔍 Unsupervised Learning: K-Means (Person 4)")
    st.write(
        "Discovers geometric groupings based solely on document vector proximity "
        "(Cosine distance on TF-IDF) without knowing document labels."
    )
    if kmeans_available:
        st.metric("Cluster Purity", f"{purity_val * 100:.2f}%")
        st.metric("Rand Index", f"{rand_idx_val * 100:.2f}%")
        st.caption("Computed from Person 4's saved results in `results/clustering_results.csv`.")
    else:
        st.info("Run Person 4's clustering (`pages/4_Clustering.py` or `test_kmeans.py`) to generate clustering artifacts.")

st.markdown("#### 📝 Methodological Discussion & Inferences")
st.markdown("""
| Dimension | Supervised Naive Bayes | Unsupervised K-Means |
| :--- | :--- | :--- |
| **Ground-Truth Labels** | **Required during training** to estimate priors $P(c)$ and likelihoods $P(w \\mid c)$. | **Not used during training**; clusters are formed purely from vector distances in TF-IDF space. |
| **Decision Boundaries** | Optimizes log-posterior probability planes between predefined discrete classes. | Partitions data space into Voronoi cells around centroid means based on Cosine similarity. |
| **Metric Interpretation** | **Accuracy & F1-Score**: Measures exact correctness against known categories. | **Purity & Rand Index**: Measures internal cluster class homogeneity and pairwise clustering agreement. |
| **Strengths** | Fast closed-form training, highly accurate when labels exist, resilient to high dimensionality. | Discovers latent subtopics and groupings without requiring costly human annotations. |
| **Limitations** | Conditional independence assumption ($P(w_1, w_2 \\mid c) = P(w_1 \\mid c)P(w_2 \\mid c)$); cannot discover novel categories. | Sensitive to initialization seeds, spherical cluster assumption, and choice of $k$. |
""")

# Save outputs to results if requested
if st.button("Export Classification Artifacts to results/", type="secondary"):
    os.makedirs("results", exist_ok=True)
    test_export = df.iloc[res["test_idx"]].copy().reset_index(drop=True)
    test_export["true_category"] = y_test
    test_export["predicted_category"] = y_pred
    test_export["confidence"] = [p[yp] for p, yp in zip(test_probs, y_pred)]
    cols_to_save = [c for c in ["document_id", "title", "true_category", "predicted_category", "confidence"] if c in test_export.columns]
    test_export[cols_to_save].to_csv("results/classification_results.csv", index=False)
    pd.DataFrame([macro_res]).to_csv("results/classification_metrics.csv", index=False)
    pd.DataFrame(conf_mat, index=ordered_labels, columns=ordered_labels).to_csv("results/confusion_matrix.csv")
    st.success("Successfully exported classification artifacts to `results/`!")

