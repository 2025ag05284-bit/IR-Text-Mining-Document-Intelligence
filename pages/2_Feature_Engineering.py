import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.preprocessing import preprocess_documents
from modules.inverted_index import (
    get_postings,
    index_statistics,
    search_and
)
from modules.tfidf import build_tfidf
from modules.similarity import (
    cosine_similarity,
    similarity_matrix,
    top_k_similar
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Feature Engineering | TF-IDF",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Feature Engineering: Inverted Index & TF-IDF")
st.markdown(
    "Manual inverted index, TF-IDF weighting with length "
    "normalization, and cosine similarity. No library TF-IDF "
    "implementation is used."
)

st.divider()


PROCESSED_PATH = Path("data/processed_dataset.csv")
RAW_PATH = Path("data/raw_dataset.csv")
CLEANED_PATH = Path("data/cleaned_dataset.csv")


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)


def load_data():
    """Prefer the shared processed dataset, then the app session."""

    if PROCESSED_PATH.exists():
        return read_csv(str(PROCESSED_PATH)), "data/processed_dataset.csv"

    if "processed_df" in st.session_state:
        return st.session_state["processed_df"], "Text Preprocessing page"

    for path in (RAW_PATH, CLEANED_PATH):
        if path.exists():
            return read_csv(str(path)), str(path)

    uploaded_file = st.sidebar.file_uploader(
        "Upload processed_dataset.csv",
        type=["csv"]
    )

    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "uploaded file"

    return None, None


df, data_source = load_data()

if df is None:
    st.warning(
        "No dataset found. Place processed_dataset.csv in the data "
        "folder, run the Text Preprocessing page, or upload a CSV."
    )
    st.stop()

missing_columns = {"document_id", "title", "text", "category"} - set(df.columns)

if missing_columns:
    st.error(f"Dataset is missing columns: {sorted(missing_columns)}")
    st.stop()

df = df.reset_index(drop=True)

for column in ("document_id", "title", "text", "category"):
    df[column] = df[column].fillna("").astype(str)

documents = (df["title"] + " " + df["text"]).tolist()


# --------------------------------------------------
# 3. CONFIGURATION
# --------------------------------------------------

st.sidebar.header("TF-IDF Configuration")

token_source_options = ["Custom preprocessing"]
if "tokens" in df.columns:
    token_source_options.insert(0, "Shared tokens (dataset)")

token_source = st.sidebar.radio(
    "Token source",
    token_source_options,
    help=(
        "Shared tokens use the team's processed_dataset.csv "
        "(lowercase + stop-word removal). Custom preprocessing lets "
        "you compare stemming / lemmatization for the report."
    )
)

remove_stopwords = True
normalization_method = "None"

if token_source == "Custom preprocessing":
    remove_stopwords = st.sidebar.checkbox("Remove stop words", value=True)
    normalization_method = st.sidebar.selectbox(
        "Word normalization",
        ["None", "Stemming", "Lemmatization"]
    )

tf_scheme = st.sidebar.selectbox(
    "TF weighting",
    ["log", "raw", "relative"],
    format_func={
        "log": "Log: 1 + log10(tf)",
        "raw": "Raw count",
        "relative": "Relative: tf / doc length"
    }.get
)

idf_scheme = st.sidebar.selectbox(
    "IDF weighting",
    ["standard", "smooth"],
    format_func={
        "standard": "log10(N / df)",
        "smooth": "log10((1+N)/(1+df)) + 1"
    }.get
)

normalize = st.sidebar.checkbox(
    "Length normalization (L2)",
    value=True
)

min_df = st.sidebar.number_input(
    "Minimum document frequency",
    min_value=1,
    max_value=50,
    value=1
)

max_df_ratio = st.sidebar.slider(
    "Maximum document frequency (fraction of corpus)",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.05
)

st.sidebar.caption(f"Data source: {data_source}")


# --------------------------------------------------
# 4. BUILD FEATURES (cached in session)
# --------------------------------------------------

data_signature = (data_source, len(df), df["document_id"].iloc[0])


def get_tokens():
    """Return token lists for the selected source, cached per session."""

    key = ("tokens", data_signature, token_source,
           remove_stopwords, normalization_method)

    cache = st.session_state.setdefault("fe_token_cache", {})

    if key not in cache:

        if token_source == "Shared tokens (dataset)":
            cache[key] = [json.loads(tokens) for tokens in df["tokens"]]

        else:
            with st.spinner("Preprocessing documents..."):
                cache[key] = preprocess_documents(
                    documents,
                    lowercase=True,
                    remove_stopword=remove_stopwords,
                    stemming=normalization_method == "Stemming",
                    lemmatization=normalization_method == "Lemmatization"
                )

    return cache[key]


def get_model(token_documents):
    """Build or reuse the TF-IDF model for the current configuration."""

    key = ("model", data_signature, token_source, remove_stopwords,
           normalization_method, tf_scheme, idf_scheme, normalize,
           min_df, max_df_ratio)

    cache = st.session_state.setdefault("fe_model_cache", {})

    if key not in cache:
        with st.spinner("Building inverted index and TF-IDF vectors..."):
            cache.clear()
            cache[key] = build_tfidf(
                token_documents,
                tf_scheme=tf_scheme,
                idf_scheme=idf_scheme,
                normalize=normalize,
                min_df=int(min_df),
                max_df_ratio=max_df_ratio
            )

    return cache[key]


try:
    token_documents = get_tokens()
except LookupError as error:
    st.error(str(error))
    st.stop()

model = get_model(token_documents)

# Shared with the classification, clustering and recommendation pages.
st.session_state["tfidf_model"] = model
st.session_state["tfidf_documents"] = df[
    ["document_id", "title", "category"]
]

if model.num_features == 0:
    st.error("The vocabulary is empty. Relax the document frequency filters.")
    st.stop()


def document_label(doc_index):
    row = df.iloc[doc_index]
    return f"{row['document_id']} | {row['title']}"


# --------------------------------------------------
# 5. OVERVIEW METRICS
# --------------------------------------------------

stats = index_statistics(model.index, model.num_documents)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Documents", f"{model.num_documents:,}")
col2.metric("Vocabulary (features)", f"{model.num_features:,}")
col3.metric("Postings", f"{stats['total_postings']:,}")
col4.metric("Matrix sparsity", f"{model.sparsity():.2%}")

st.caption(
    f"TF = {tf_scheme}, IDF = {idf_scheme}, "
    f"L2 normalization = {'on' if normalize else 'off'}, "
    f"min_df = {min_df}, max_df = {max_df_ratio:.0%} of documents."
)

tabs = st.tabs([
    "Inverted Index",
    "TF-IDF Weights",
    "Length Normalization",
    "Category Terms",
    "Cosine Similarity",
    "Before vs After Preprocessing"
])


# --------------------------------------------------
# 6. INVERTED INDEX
# --------------------------------------------------

with tabs[0]:

    st.subheader("Inverted Index Explorer")

    st.write(
        "Each term maps to a postings list of the documents that "
        "contain it, with the raw term frequency in each document."
    )

    default_term = "economy" if "economy" in model.term_to_index else model.vocabulary[0]

    term = st.text_input(
        "Look up a term (after preprocessing)",
        value=default_term
    ).strip().lower()

    if term in model.term_to_index:

        column = model.term_to_index[term]
        postings = get_postings(model.index, term)

        col1, col2, col3 = st.columns(3)
        col1.metric("Document frequency (df)", model.document_frequency[column])
        col2.metric("Collection frequency", sum(tf for _, tf in postings))
        col3.metric("IDF", f"{model.idf[column]:.4f}")

        postings_df = pd.DataFrame([
            {
                "doc_index": doc_index,
                "document_id": df.iloc[doc_index]["document_id"],
                "title": df.iloc[doc_index]["title"],
                "category": df.iloc[doc_index]["category"],
                "tf": tf,
                "tfidf_weight": round(
                    model.vectors[doc_index].get(column, 0.0), 4
                )
            }
            for doc_index, tf in postings
        ]).sort_values("tfidf_weight", ascending=False)

        st.write(f"Postings list for **{term}** ({len(postings)} documents)")
        st.dataframe(postings_df, use_container_width=True, hide_index=True)

        category_counts = postings_df["category"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 3))
        category_counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Documents containing '{term}' by category")
        ax.set_ylabel("Documents")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    elif term:
        st.info(f"'{term}' is not in the vocabulary.")

    st.divider()

    st.subheader("Boolean AND Query")

    query = st.text_input(
        "Terms (space separated)",
        value="oil price"
    )

    query_terms = query.lower().split()

    if query_terms:
        matches = search_and(model.index, query_terms)

        st.write(
            f"{len(matches)} documents contain all of: "
            f"{', '.join(query_terms)}"
        )

        if matches:
            st.dataframe(
                df.iloc[matches][["document_id", "title", "category"]].head(50),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    st.subheader("Index Sample")

    sample_terms = model.vocabulary[:: max(1, model.num_features // 15)][:15]

    st.dataframe(
        pd.DataFrame([
            {
                "term": sample_term,
                "df": len(model.index[sample_term]),
                "postings (doc_index: tf)": ", ".join(
                    f"{d}:{tf}"
                    for d, tf in get_postings(model.index, sample_term)[:8]
                ) + (" ..." if len(model.index[sample_term]) > 8 else "")
            }
            for sample_term in sample_terms
        ]),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 7. TF-IDF WEIGHTS
# --------------------------------------------------

with tabs[1]:

    st.subheader("Document Frequency and IDF Distributions")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(model.idf, bins=40)
        ax.set_xlabel("IDF")
        ax.set_ylabel("Number of terms")
        ax.set_title("IDF distribution")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        ranked_df = sorted(model.document_frequency, reverse=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(range(1, len(ranked_df) + 1), ranked_df)
        ax.set_xlabel("Term rank")
        ax.set_ylabel("Document frequency")
        ax.set_title("Document frequency by rank (log-log)")
        st.pyplot(fig)
        plt.close(fig)

    singleton_terms = sum(1 for d in model.document_frequency if d == 1)

    st.caption(
        f"{singleton_terms:,} terms ({singleton_terms / model.num_features:.1%}) "
        "appear in only one document and receive the maximum IDF."
    )

    idf_df = pd.DataFrame({
        "term": model.vocabulary,
        "df": model.document_frequency,
        "idf": [round(value, 4) for value in model.idf]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.write("Lowest IDF (most common) terms")
        st.dataframe(
            idf_df.sort_values(["idf", "term"]).head(15),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.write("Highest IDF terms with df ≥ 3")
        st.dataframe(
            idf_df[idf_df["df"] >= 3]
            .sort_values(["idf", "term"], ascending=[False, True])
            .head(15),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("Top TF-IDF Terms for a Document")

    doc_index = st.selectbox(
        "Document",
        range(model.num_documents),
        format_func=document_label,
        key="weights_doc"
    )

    top_n = st.slider("Number of terms", 5, 30, 15)

    top_terms = model.top_terms(doc_index, n=top_n)

    if top_terms:
        terms_df = pd.DataFrame(top_terms, columns=["term", "tfidf_weight"])

        fig, ax = plt.subplots(figsize=(9, 0.3 * len(terms_df) + 1))
        ax.barh(terms_df["term"], terms_df["tfidf_weight"])
        ax.invert_yaxis()
        ax.set_xlabel("TF-IDF weight")
        ax.set_title(f"Top terms: {df.iloc[doc_index]['title']}")
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Document text"):
            st.write(df.iloc[doc_index]["text"])
    else:
        st.info("This document has no terms in the current vocabulary.")


# --------------------------------------------------
# 8. LENGTH NORMALIZATION
# --------------------------------------------------

with tabs[2]:

    st.subheader("Effect of Length Normalization")

    st.latex(
        r"w'_{t,d} = \frac{w_{t,d}}{\sqrt{\sum_{t'} w_{t',d}^2}}"
    )

    st.write(
        "Without normalization, long documents have larger vectors and "
        "dominate dot-product similarity. Dividing by the vector length "
        "puts every document on the unit sphere, so cosine similarity "
        "reduces to a dot product."
    )

    doc_lengths = [len(tokens) for tokens in token_documents]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(doc_lengths, model.raw_norms, s=6, alpha=0.5)
        ax.set_xlabel("Document length (tokens)")
        ax.set_ylabel("Vector length before normalization")
        ax.set_title("Raw TF-IDF vector length vs document length")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        final_norms = [
            sum(w * w for w in vector.values()) ** 0.5
            for vector in model.vectors
        ]
        # Shared bin edges: after normalization every length is 1.0,
        # which a 40-bin histogram of its own cannot represent.
        low = min(model.raw_norms + final_norms)
        high = max(model.raw_norms + final_norms)
        bins = (
            [low + (high - low) * i / 40 for i in range(41)]
            if high > low else 1
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(model.raw_norms, bins=bins, alpha=0.7, label="Before")
        ax.hist(final_norms, bins=bins, alpha=0.7, label="After")
        ax.set_xlabel("Vector length")
        ax.set_ylabel("Documents")
        ax.set_title("Vector lengths before and after")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    if not normalize:
        st.warning(
            "Length normalization is switched off in the sidebar, so the "
            "'after' lengths equal the raw lengths."
        )

    doc_index = st.selectbox(
        "Inspect a document",
        range(model.num_documents),
        format_func=document_label,
        key="norm_doc"
    )

    raw_norm = model.raw_norms[doc_index]

    rows = []
    for term, weight in model.top_terms(doc_index, n=10):
        raw_weight = weight * raw_norm if normalize else weight
        rows.append({
            "term": term,
            "tf": model.index[term][doc_index],
            "idf": round(model.idf[model.term_to_index[term]], 4),
            "tf-idf (raw)": round(raw_weight, 4),
            "tf-idf (normalized)": round(
                raw_weight / raw_norm if raw_norm else 0.0, 4
            )
        })

    st.write(
        f"Tokens: {doc_lengths[doc_index]:,} | "
        f"Raw vector length: {raw_norm:.4f}"
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --------------------------------------------------
# 9. CATEGORY TERMS
# --------------------------------------------------

with tabs[3]:

    st.subheader("Most Characteristic Terms per Category")

    st.write(
        "Average TF-IDF weight of each term across the documents of a "
        "category. These terms explain why classification and "
        "clustering can separate the categories."
    )

    categories = sorted(df["category"].unique())

    category_tables = {}

    for category in categories:

        doc_indexes = df.index[df["category"] == category].tolist()

        totals = Counter()
        for doc_index in doc_indexes:
            totals.update(model.vectors[doc_index])

        category_tables[category] = [
            (model.vocabulary[column], total / len(doc_indexes))
            for column, total in totals.most_common(10)
        ]

    columns = st.columns(min(len(categories), 5))

    for position, category in enumerate(categories):
        with columns[position % len(columns)]:
            st.write(f"**{category}**")
            st.dataframe(
                pd.DataFrame(
                    category_tables[category],
                    columns=["term", "avg weight"]
                ).round(4),
                use_container_width=True,
                hide_index=True
            )


# --------------------------------------------------
# 10. COSINE SIMILARITY
# --------------------------------------------------

with tabs[4]:

    st.subheader("Cosine Similarity Between Two Documents")

    st.latex(
        r"\cos(\vec{a}, \vec{b}) = "
        r"\frac{\vec{a} \cdot \vec{b}}{\lVert\vec{a}\rVert\,\lVert\vec{b}\rVert}"
    )

    col1, col2 = st.columns(2)

    with col1:
        doc_a = st.selectbox(
            "Document A",
            range(model.num_documents),
            format_func=document_label,
            key="sim_a"
        )

    with col2:
        doc_b = st.selectbox(
            "Document B",
            range(model.num_documents),
            index=min(1, model.num_documents - 1),
            format_func=document_label,
            key="sim_b"
        )

    vector_a = model.vectors[doc_a]
    vector_b = model.vectors[doc_b]

    st.metric(
        "Cosine similarity",
        f"{cosine_similarity(vector_a, vector_b):.4f}"
    )

    norm_a = sum(w * w for w in vector_a.values()) ** 0.5
    norm_b = sum(w * w for w in vector_b.values()) ** 0.5

    shared = [
        {
            "term": model.vocabulary[column],
            "weight A": round(vector_a[column], 4),
            "weight B": round(vector_b[column], 4),
            "contribution": round(
                vector_a[column] * vector_b[column] / (norm_a * norm_b), 4
            )
        }
        for column in set(vector_a) & set(vector_b)
    ]

    if shared:
        st.write(f"{len(shared)} shared terms (top contributors):")
        st.dataframe(
            pd.DataFrame(shared)
            .sort_values("contribution", ascending=False)
            .head(15),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("The documents share no terms, so their similarity is 0.")

    st.divider()

    st.subheader("Most Similar Documents")

    query_doc = st.selectbox(
        "Query document",
        range(model.num_documents),
        format_func=document_label,
        key="sim_query"
    )

    k = st.slider("Top-K", 3, 20, 5)

    neighbours = top_k_similar(model, query_doc, k=k)

    st.dataframe(
        pd.DataFrame([
            {
                "rank": rank,
                "document_id": df.iloc[doc_index]["document_id"],
                "title": df.iloc[doc_index]["title"],
                "category": df.iloc[doc_index]["category"],
                "cosine": round(score, 4)
            }
            for rank, (doc_index, score) in enumerate(neighbours, start=1)
        ]),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Similarity Heatmap (sample grouped by category)")

    per_category = st.slider("Documents per category", 3, 20, 8)

    sample_indexes = []
    for category in sorted(df["category"].unique()):
        sample_indexes.extend(
            df.index[df["category"] == category][:per_category].tolist()
        )

    matrix = similarity_matrix([model.vectors[i] for i in sample_indexes])

    # Scale colours to off-diagonal scores; the diagonal (1.0) would
    # otherwise wash out the category blocks.
    off_diagonal = sorted(
        matrix[i][j]
        for i in range(len(matrix))
        for j in range(len(matrix))
        if i != j
    )
    color_max = off_diagonal[int(0.99 * (len(off_diagonal) - 1))] if off_diagonal else 1.0

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(color_max, 0.01))
    fig.colorbar(image, ax=ax, label="Cosine similarity")

    labels = df.iloc[sample_indexes]["category"].tolist()
    boundaries = [
        position for position in range(1, len(labels))
        if labels[position] != labels[position - 1]
    ]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.8)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.8)

    tick_positions = [0] + boundaries
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels[p] for p in tick_positions], rotation=45, ha="left")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([labels[p] for p in tick_positions])
    ax.set_title("Pairwise cosine similarity")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Brighter blocks on the diagonal mean documents of the same "
        "category are more similar to each other than to other categories."
    )


# --------------------------------------------------
# 11. BEFORE VS AFTER PREPROCESSING
# --------------------------------------------------

with tabs[5]:

    st.subheader("Vocabulary Before and After Preprocessing")

    st.write(
        "Before: lowercase tokenization only. After: the token source "
        "selected in the sidebar."
    )

    baseline_key = ("baseline", data_signature)
    token_cache = st.session_state.setdefault("fe_token_cache", {})

    if baseline_key not in token_cache:
        with st.spinner("Tokenizing raw text..."):
            token_cache[baseline_key] = preprocess_documents(
                documents,
                lowercase=True,
                remove_stopword=False
            )

    baseline_tokens = token_cache[baseline_key]

    baseline_counts = Counter(
        token for tokens in baseline_tokens for token in tokens
    )
    processed_counts = Counter(
        token for tokens in token_documents for token in tokens
    )

    comparison = pd.DataFrame([
        {
            "stage": "Before",
            "total_tokens": sum(baseline_counts.values()),
            "vocabulary_size": len(baseline_counts),
            "avg_tokens_per_doc": round(
                sum(baseline_counts.values()) / len(baseline_tokens), 2
            )
        },
        {
            "stage": "After",
            "total_tokens": sum(processed_counts.values()),
            "vocabulary_size": len(processed_counts),
            "avg_tokens_per_doc": round(
                sum(processed_counts.values()) / len(token_documents), 2
            )
        }
    ])

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    for column, title, counts in (
        (col1, "Top 20 terms before", baseline_counts),
        (col2, "Top 20 terms after", processed_counts)
    ):
        with column:
            top = counts.most_common(20)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.barh([t for t, _ in top], [c for _, c in top])
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            ax.set_title(title)
            st.pyplot(fig)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, counts in (("Before", baseline_counts), ("After", processed_counts)):
        frequencies = sorted(counts.values(), reverse=True)
        ax.loglog(range(1, len(frequencies) + 1), frequencies, label=label)
    ax.set_xlabel("Term rank")
    ax.set_ylabel("Collection frequency")
    ax.set_title("Term-frequency distribution (Zipf plot)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Feature Engineering | Manual inverted index, TF-IDF with L2 length "
    "normalization and cosine similarity (modules/inverted_index.py, "
    "modules/tfidf.py, modules/similarity.py)"
)
