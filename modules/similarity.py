"""Manual cosine similarity over sparse TF-IDF vectors.

    cos(a, b) = (a . b) / (|a| * |b|)

Vectors are dicts {column_index: weight} as produced by
modules/tfidf.py. When the vectors are L2-normalized the denominator
is 1, but the full formula is used so un-normalized vectors also work.
"""

import math


def dot_product(vector_a, vector_b):
    """Dot product of two sparse vectors."""

    # Iterate over the shorter vector.
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a

    return sum(
        weight * vector_b.get(column, 0.0)
        for column, weight in vector_a.items()
    )


def cosine_similarity(vector_a, vector_b):
    """Cosine similarity of two sparse vectors (0 if either is empty)."""

    norm_a = math.sqrt(sum(w * w for w in vector_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vector_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product(vector_a, vector_b) / (norm_a * norm_b)


def similarity_matrix(vectors):
    """Pairwise cosine similarity for a (small) list of vectors."""

    size = len(vectors)

    matrix = [[0.0] * size for _ in range(size)]

    for i in range(size):

        matrix[i][i] = 1.0 if vectors[i] else 0.0

        for j in range(i + 1, size):
            score = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = score
            matrix[j][i] = score

    return matrix


def score_against_corpus(model, query_vector):
    """Cosine score of a query vector against every corpus document.

    Uses the inverted index (term-at-a-time accumulation), so only
    documents sharing at least one term with the query are touched.
    Returns {doc_index: score}.
    """

    query_norm = math.sqrt(
        sum(w * w for w in query_vector.values())
    )

    if query_norm == 0:
        return {}

    scores = {}

    for column, query_weight in query_vector.items():

        term = model.vocabulary[column]

        for doc_index in model.index.get(term, {}):

            doc_weight = model.vectors[doc_index].get(column, 0.0)

            scores[doc_index] = (
                scores.get(doc_index, 0.0) + query_weight * doc_weight
            )

    for doc_index in scores:

        doc_norm = math.sqrt(
            sum(w * w for w in model.vectors[doc_index].values())
        )

        scores[doc_index] = (
            scores[doc_index] / (query_norm * doc_norm)
            if doc_norm else 0.0
        )

    return scores


def top_k_similar(model, doc_index, k=5):
    """Top-k corpus documents most similar to document doc_index.

    Returns a list of (doc_index, score), highest score first,
    excluding the document itself.
    """

    scores = score_against_corpus(model, model.vectors[doc_index])

    scores.pop(doc_index, None)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )[:k]


def top_k_for_query(model, query_tokens, k=5):
    """Top-k corpus documents for a free-text query (preprocessed tokens)."""

    query_vector = model.transform([query_tokens])[0]

    scores = score_against_corpus(model, query_vector)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )[:k]
