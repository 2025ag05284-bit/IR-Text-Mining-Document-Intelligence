
import streamlit as st
import pandas as pd

from modules.recommendation import recommend_documents


st.set_page_config(
    page_title="Document Recommendations",
    page_icon="📑",
    layout="wide"
)

st.title("📑 Document Recommendation System")

st.markdown(
    "Discover related news articles using "
    "TF-IDF and cosine similarity."
)

st.divider()

# Retrieve Person 2's shared TF-IDF model.
model = st.session_state.get("tfidf_model")
metadata = st.session_state.get("tfidf_documents")

if model is None or metadata is None:
    st.warning(
        "Please open the Feature Engineering page first "
        "to initialize the TF-IDF model."
    )
    st.stop()

metadata = metadata.reset_index(drop=True)

if model.num_documents < 2:
    st.warning("At least two documents are required.")
    st.stop()

# Document selection.
st.subheader("Select a News Article")

selected_index = st.selectbox(
    "Choose a document",
    range(model.num_documents),
    format_func=lambda i: (
        f"{metadata.iloc[i]['document_id']} | "
        f"{metadata.iloc[i]['title']}"
    )
)

selected_document = metadata.iloc[selected_index]

col1, col2 = st.columns(2)

col1.metric(
    "Document ID",
    selected_document["document_id"]
)

col2.metric(
    "Category",
    selected_document["category"]
)

st.write(f"**Selected Article:** {selected_document['title']}")

st.divider()

# Recommendation configuration.
st.subheader("Recommendation Settings")

top_k = st.slider(
    "Number of similar documents",
    min_value=1,
    max_value=min(10, model.num_documents - 1),
    value=min(5, model.num_documents - 1)
)

if st.button("Find Similar Documents", type="primary"):

    with st.spinner("Finding similar articles..."):

        results = recommend_documents(
            model=model,
            metadata=metadata,
            doc_index=selected_index,
            k=top_k
        )

    st.subheader("Recommended Articles")

    if not results:
        st.info("No similar documents found.")

    else:
        for rank, result in enumerate(results, start=1):

            with st.container(border=True):

                st.markdown(
                    f"**{rank}. {result['title']}**"
                )

                st.caption(
                    f"Document ID: {result['document_id']} "
                    f"| Category: {result['category']}"
                )

                st.progress(
                    min(1.0, max(0.0, result["similarity_score"]))
                )

                st.write(
                    f"Cosine Similarity: "
                    f"{result['similarity_score']:.4f}"
                )

        st.success(
            f"Successfully retrieved {len(results)} recommendations."
        )

        results_df = pd.DataFrame(results)

        st.download_button(
            "Download Recommendations",
            data=results_df.to_csv(index=False),
            file_name="document_recommendations.csv",
            mime="text/csv"
        )
