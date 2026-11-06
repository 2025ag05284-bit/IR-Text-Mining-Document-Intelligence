"""Manual inverted index construction.

The index maps every term to a postings list of the documents that
contain it, together with the raw term frequency in each document:

    term -> {doc_index: term_count, ...}

Document indexes are row positions in the shared dataset, so they line
up with document_id order from data/processed_dataset.csv.
"""


def build_inverted_index(token_documents):
    """Build an inverted index from preprocessed token lists.

    Returns a dict: term -> {doc_index: term_frequency}.
    Postings are inserted in increasing doc_index order.
    """

    index = {}

    for doc_index, tokens in enumerate(token_documents):

        for term in tokens:

            postings = index.get(term)

            if postings is None:
                postings = {}
                index[term] = postings

            postings[doc_index] = postings.get(doc_index, 0) + 1

    return index


def document_frequencies(index):
    """Return term -> number of documents containing the term."""

    return {
        term: len(postings)
        for term, postings in index.items()
    }


def collection_frequencies(index):
    """Return term -> total occurrences across the corpus."""

    return {
        term: sum(postings.values())
        for term, postings in index.items()
    }


def get_postings(index, term):
    """Return the postings list for a term as sorted (doc_index, tf) pairs."""

    postings = index.get(term, {})

    return sorted(postings.items())


def search_and(index, terms):
    """Boolean AND query: documents containing every term."""

    if not terms:
        return []

    postings_sets = [
        set(index.get(term, {}))
        for term in terms
    ]

    # Intersect the shortest lists first.
    postings_sets.sort(key=len)

    result = postings_sets[0]

    for postings in postings_sets[1:]:
        result = result & postings

        if not result:
            break

    return sorted(result)


def search_or(index, terms):
    """Boolean OR query: documents containing at least one term."""

    result = set()

    for term in terms:
        result.update(index.get(term, {}))

    return sorted(result)


def index_statistics(index, total_documents):
    """Summarize the size and sparsity of the inverted index."""

    vocabulary_size = len(index)

    total_postings = sum(
        len(postings)
        for postings in index.values()
    )

    possible_entries = vocabulary_size * total_documents

    return {
        "vocabulary_size": vocabulary_size,
        "total_postings": total_postings,
        "average_postings_per_term": round(
            total_postings / vocabulary_size, 2
        ) if vocabulary_size else 0,
        "sparsity": round(
            1 - total_postings / possible_entries, 6
        ) if possible_entries else 0
    }
