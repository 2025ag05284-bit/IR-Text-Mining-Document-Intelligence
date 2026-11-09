
import re
from modules.similarity import top_k_similar

TEXT_COLUMNS = [
    "text",
    "content",
    "article",
    "body",
    "raw_text",
    "document_text",
    "cleaned_text",
    "processed_text"
]


def normalize_text(value):
    """Normalize text for duplicate comparison."""
    if value is None:
        return ""

    text = str(value).casefold().strip()
    return re.sub(r"\s+", " ", text)


def get_article_key(row):
    """
    Identify an article using its complete text.
    If article text is unavailable, use its title.
    """

    for column in TEXT_COLUMNS:
        if column in row.index:
            content = normalize_text(row[column])

            if content:
                return ("content", content)

    title = normalize_text(row.get("title", ""))

    if title:
        return ("title", title)

    return None


def recommend_documents(model, metadata, doc_index, k=5):

    if not isinstance(doc_index, int):
        raise TypeError("Document index must be an integer.")

    if doc_index < 0 or doc_index >= model.num_documents:
        raise IndexError("Invalid document index.")

    if k < 1:
        raise ValueError("K must be at least 1.")

    if len(metadata) != model.num_documents:
        raise ValueError("Metadata and model size mismatch.")

    selected = metadata.iloc[doc_index]

    selected_id = normalize_text(selected["document_id"])

    # Exclude the selected article and its duplicates.
    seen_ids = {selected_id}

    selected_key = get_article_key(selected)
    seen_articles = set()

    if selected_key:
        seen_articles.add(selected_key)

    recommendations = []

    candidates = top_k_similar(
        model,
        doc_index=doc_index,
        k=model.num_documents - 1
    )

    for index, score in candidates:

        if index == doc_index:
            continue

        row = metadata.iloc[index]

        doc_id = normalize_text(row["document_id"])

        if doc_id in seen_ids:
            continue

        article_key = get_article_key(row)

        # Exclude duplicate article content.
        if article_key and article_key in seen_articles:
            continue

        seen_ids.add(doc_id)

        if article_key:
            seen_articles.add(article_key)

        recommendations.append({
            "document_id": row["document_id"],
            "title": row["title"],
            "category": row["category"],
            "similarity_score": round(float(score), 4)
        })

        if len(recommendations) == k:
            break

    return recommendations
