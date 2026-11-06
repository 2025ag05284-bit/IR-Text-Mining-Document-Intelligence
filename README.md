# IR-Text-Mining-Document-Intelligence
An end-to-end Streamlit-based Text Mining and Document Intelligence System using manual TF-IDF, Naive Bayes, K-Means, and content-based document recommendation.

## Feature Engineering (Person 2): Inverted Index, TF-IDF, Cosine Similarity

Modules: `modules/inverted_index.py`, `modules/tfidf.py`, `modules/similarity.py`
Streamlit page: `pages/2_Feature_Engineering.py` (appears in the sidebar when running `streamlit run app.py`)
Self-test: `python test_tfidf.py`

All weights are computed manually (no scikit-learn):

- TF: `1 + log10(tf)` (default), raw count, or `tf / doc_length`
- IDF: `log10(N / df)` (default) or smoothed `log10((1+N)/(1+df)) + 1`
- L2 length normalization, so cosine similarity = dot product

### Using it in your module

```python
import json
import pandas as pd
from modules.tfidf import build_tfidf
from modules.similarity import top_k_similar, cosine_similarity

df = pd.read_csv("data/processed_dataset.csv")
tokens = [json.loads(t) for t in df["tokens"]]

model = build_tfidf(tokens)          # same defaults as the Streamlit page

model.vectors[i]       # sparse dict {column_index: weight} for document i (row order = df order)
model.vocabulary[c]    # term for column c (sorted alphabetically, same for everyone)
model.transform([new_tokens])        # vectorize unseen/test documents with the fitted IDF
model.to_dense(columns=model.top_columns(3000))  # NumPy matrix (2225 x 3000, ~27 MB) for K-Means
top_k_similar(model, doc_index=0, k=5)           # [(doc_index, cosine), ...] for recommendation
```

The full vocabulary is ~28k terms, so a dense 2225 x 28k matrix would be ~250 MB;
use `top_columns(n)` or `build_tfidf(tokens, min_df=2)` when you need a matrix.
The page also stores the fitted model in `st.session_state["tfidf_model"]`.
