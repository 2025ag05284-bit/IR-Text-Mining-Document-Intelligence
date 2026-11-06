"""Manual TF-IDF feature engineering with length normalization.

No library TF-IDF implementation is used. Weights are computed from the
inverted index built in modules/inverted_index.py:

    tf(t, d)    raw      : count of t in d
                log      : 1 + log10(count)            (default)
                relative : count / len(d)

    idf(t)      standard : log10(N / df(t))            (default)
                smooth   : log10((1 + N) / (1 + df(t))) + 1

    w(t, d)     tf(t, d) * idf(t)

    Length normalization (cosine / L2):
                w'(t, d) = w(t, d) / sqrt(sum over t of w(t, d)^2)

After normalization every non-empty document vector has length 1, so the
cosine similarity of two documents is just their dot product.

Vectors are stored sparsely as {column_index: weight} because the BBC
corpus has ~28k terms and 2,225 documents (a dense matrix would be
~500 MB). Use to_dense() with max_features when a module needs a matrix.

Shared contract for downstream modules (Naive Bayes, K-Means,
recommendation): column i of every vector is model.vocabulary[i], and
row i is document i of the dataset, in the original document_id order.
"""

import math

from modules.inverted_index import build_inverted_index


TF_SCHEMES = ("raw", "log", "relative")
IDF_SCHEMES = ("standard", "smooth")


def compute_tf(count, document_length, scheme="log"):
    """Term frequency weight for a term occurring count times."""

    if count <= 0:
        return 0.0

    if scheme == "raw":
        return float(count)

    if scheme == "log":
        return 1.0 + math.log10(count)

    if scheme == "relative":
        return count / document_length if document_length else 0.0

    raise ValueError(f"Unknown TF scheme: {scheme}")


def compute_idf(document_frequency, total_documents, scheme="standard"):
    """Inverse document frequency for a term."""

    if scheme == "standard":
        if document_frequency == 0:
            return 0.0
        return math.log10(total_documents / document_frequency)

    if scheme == "smooth":
        return math.log10(
            (1 + total_documents) / (1 + document_frequency)
        ) + 1.0

    raise ValueError(f"Unknown IDF scheme: {scheme}")


def vector_norm(vector):
    """Euclidean (L2) length of a sparse vector."""

    return math.sqrt(
        sum(weight * weight for weight in vector.values())
    )


def normalize_vector(vector):
    """Divide every weight by the vector's L2 length."""

    norm = vector_norm(vector)

    if norm == 0:
        return dict(vector)

    return {
        column: weight / norm
        for column, weight in vector.items()
    }


def select_vocabulary(
    index,
    total_documents,
    min_df=1,
    max_df_ratio=1.0,
    max_features=None
):
    """Choose which terms become feature columns.

    Terms rarer than min_df documents or more common than
    max_df_ratio * N documents are dropped. If max_features is set,
    the terms with the highest document frequency are kept.
    The returned vocabulary is sorted alphabetically so the column
    order is deterministic.
    """

    max_df = max_df_ratio * total_documents

    candidates = [
        (term, len(postings))
        for term, postings in index.items()
        if min_df <= len(postings) <= max_df
    ]

    if max_features is not None and len(candidates) > max_features:
        candidates.sort(key=lambda item: (-item[1], item[0]))
        candidates = candidates[:max_features]

    return sorted(term for term, _ in candidates)


class TfidfModel:
    """Fitted TF-IDF representation of a corpus."""

    def __init__(
        self,
        vocabulary,
        idf,
        document_frequency,
        vectors,
        raw_norms,
        index,
        tf_scheme,
        idf_scheme,
        normalize
    ):
        self.vocabulary = vocabulary
        self.term_to_index = {
            term: column
            for column, term in enumerate(vocabulary)
        }
        self.idf = idf
        self.document_frequency = document_frequency
        self.vectors = vectors
        self.raw_norms = raw_norms
        self.index = index
        self.tf_scheme = tf_scheme
        self.idf_scheme = idf_scheme
        self.normalize = normalize

    @property
    def num_documents(self):
        return len(self.vectors)

    @property
    def num_features(self):
        return len(self.vocabulary)

    def non_zero_entries(self):
        return sum(len(vector) for vector in self.vectors)

    def sparsity(self):
        cells = self.num_documents * self.num_features
        return 1 - self.non_zero_entries() / cells if cells else 0.0

    def transform(self, token_documents):
        """Vectorize new documents using the fitted vocabulary and IDF.

        Unknown terms are ignored. Use this for test documents or a
        free-text query so they share the training feature space.
        """

        vectors = []

        for tokens in token_documents:

            counts = {}
            for term in tokens:
                column = self.term_to_index.get(term)
                if column is not None:
                    counts[column] = counts.get(column, 0) + 1

            vector = {
                column: compute_tf(count, len(tokens), self.tf_scheme)
                * self.idf[column]
                for column, count in counts.items()
            }

            vector = {
                column: weight
                for column, weight in vector.items()
                if weight != 0
            }

            if self.normalize:
                vector = normalize_vector(vector)

            vectors.append(vector)

        return vectors

    def top_terms(self, doc_index, n=10):
        """Highest-weighted terms of one document as (term, weight)."""

        vector = self.vectors[doc_index]

        ranked = sorted(
            vector.items(),
            key=lambda item: item[1],
            reverse=True
        )[:n]

        return [
            (self.vocabulary[column], weight)
            for column, weight in ranked
        ]

    def to_dense(self, doc_indexes=None, columns=None, dtype="float32"):
        """Convert (a subset of) the sparse vectors to a NumPy matrix.

        columns is an optional list of column indexes to keep, e.g.
        the result of model.top_columns(3000) for K-Means.
        """

        import numpy as np

        if doc_indexes is None:
            doc_indexes = range(self.num_documents)

        doc_indexes = list(doc_indexes)

        if columns is None:
            column_map = None
            width = self.num_features
        else:
            column_map = {
                column: position
                for position, column in enumerate(columns)
            }
            width = len(columns)

        matrix = np.zeros((len(doc_indexes), width), dtype=dtype)

        for row, doc_index in enumerate(doc_indexes):
            for column, weight in self.vectors[doc_index].items():
                if column_map is None:
                    matrix[row, column] = weight
                elif column in column_map:
                    matrix[row, column_map[column]] = weight

        return matrix

    def top_columns(self, n):
        """Column indexes of the n terms with the highest total weight."""

        totals = {}

        for vector in self.vectors:
            for column, weight in vector.items():
                totals[column] = totals.get(column, 0.0) + weight

        ranked = sorted(
            totals.items(),
            key=lambda item: item[1],
            reverse=True
        )[:n]

        return sorted(column for column, _ in ranked)


def build_tfidf(
    token_documents,
    tf_scheme="log",
    idf_scheme="standard",
    normalize=True,
    min_df=1,
    max_df_ratio=1.0,
    max_features=None
):
    """Build manual TF-IDF vectors from preprocessed token lists.

    token_documents: list of token lists, one per document, in dataset
    order (e.g. json.loads of the tokens column).

    Returns a TfidfModel whose .vectors[i] is the sparse vector of
    document i.
    """

    if tf_scheme not in TF_SCHEMES:
        raise ValueError(f"tf_scheme must be one of {TF_SCHEMES}")

    if idf_scheme not in IDF_SCHEMES:
        raise ValueError(f"idf_scheme must be one of {IDF_SCHEMES}")

    total_documents = len(token_documents)

    full_index = build_inverted_index(token_documents)

    vocabulary = select_vocabulary(
        full_index,
        total_documents,
        min_df=min_df,
        max_df_ratio=max_df_ratio,
        max_features=max_features
    )

    index = {
        term: full_index[term]
        for term in vocabulary
    }

    document_frequency = [
        len(index[term])
        for term in vocabulary
    ]

    idf = [
        compute_idf(df, total_documents, idf_scheme)
        for df in document_frequency
    ]

    document_lengths = [
        len(tokens)
        for tokens in token_documents
    ]

    # Walk the postings lists to fill each document's vector.
    vectors = [{} for _ in range(total_documents)]

    for column, term in enumerate(vocabulary):

        term_idf = idf[column]

        if term_idf == 0:
            continue

        for doc_index, count in index[term].items():

            vectors[doc_index][column] = compute_tf(
                count,
                document_lengths[doc_index],
                tf_scheme
            ) * term_idf

    raw_norms = [
        vector_norm(vector)
        for vector in vectors
    ]

    if normalize:
        vectors = [
            normalize_vector(vector)
            for vector in vectors
        ]

    return TfidfModel(
        vocabulary=vocabulary,
        idf=idf,
        document_frequency=document_frequency,
        vectors=vectors,
        raw_norms=raw_norms,
        index=index,
        tf_scheme=tf_scheme,
        idf_scheme=idf_scheme,
        normalize=normalize
    )
