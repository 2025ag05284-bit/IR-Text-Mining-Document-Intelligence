import json
import numpy as np
import pandas as pd

from modules.preprocessing import preprocess_documents
from modules.tfidf import TfidfModel, build_tfidf


def extract_dense_from_model(tfidf_input, max_features=3000):
    if isinstance(tfidf_input, TfidfModel):
        if max_features and tfidf_input.num_features > max_features:
            cols = tfidf_input.top_columns(max_features)
            X = tfidf_input.to_dense(columns=cols, dtype="float64")
            vocab = [tfidf_input.vocabulary[c] for c in cols]
        else:
            X = tfidf_input.to_dense(dtype="float64")
            vocab = list(tfidf_input.vocabulary)
        return X, vocab

    if isinstance(tfidf_input, tuple) and len(tfidf_input) >= 2:
        return np.asarray(tfidf_input[0], dtype="float64"), list(tfidf_input[1])

    if isinstance(tfidf_input, pd.DataFrame):
        return tfidf_input.to_numpy(dtype="float64"), list(tfidf_input.columns)

    X = np.asarray(tfidf_input, dtype="float64")
    vocab = [f"term_{i}" for i in range(X.shape[1])]
    return X, vocab


def build_tfidf_from_repo(df, max_features=3000):
    if "tokens" in df.columns:
        sample = df["tokens"].iloc[0]
        if isinstance(sample, list):
            token_documents = df["tokens"].tolist()
        else:
            token_documents = [json.loads(t) for t in df["tokens"]]
    else:
        documents = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()
        token_documents = preprocess_documents(
            documents,
            lowercase=True,
            remove_stopword=True,
            stemming=False,
            lemmatization=False
        )

    model = build_tfidf(
        token_documents,
        tf_scheme="log",
        idf_scheme="standard",
        normalize=True
    )
    X, vocab = extract_dense_from_model(model, max_features=max_features)
    return X, vocab, model


class ManualKMeans:
    def __init__(self, n_clusters=5, max_iter=100, tol=1e-4, random_state=42):
        self.k = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.centroids = None
        self.initial_centroids = None
        self.labels_ = None
        self.inertia_ = 0.0
        self.iterations_run = 0
        self.history = []

    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        if self.k > n_samples:
            raise ValueError(f"k={self.k} cannot exceed sample size N={n_samples}")
        indices = rng.choice(n_samples, size=self.k, replace=False)
        return X[indices].copy()

    def _compute_distances(self, X, centroids):
        x_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
        c_norm_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
        dist_sq = x_norm_sq + c_norm_sq - 2.0 * np.dot(X, centroids.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.sqrt(dist_sq)

    def fit(self, X):
        X_arr, _ = extract_dense_from_model(X)
        n_samples, _ = X_arr.shape

        row_norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        X_norm = X_arr / row_norms

        self.centroids = self._init_centroids(X_norm)
        self.initial_centroids = self.centroids.copy()
        self.history = []

        for iteration in range(self.max_iter):
            self.iterations_run = iteration + 1

            distances = self._compute_distances(X_norm, self.centroids)
            new_labels = np.argmin(distances, axis=1)

            min_dists = np.min(distances, axis=1)
            iter_inertia = float(np.sum(min_dists ** 2))

            new_centroids = np.zeros_like(self.centroids)
            for cluster_idx in range(self.k):
                cluster_points = X_norm[new_labels == cluster_idx]
                if len(cluster_points) > 0:
                    mean_vec = np.mean(cluster_points, axis=0)
                    norm_val = np.linalg.norm(mean_vec)
                    new_centroids[cluster_idx] = mean_vec / norm_val if norm_val > 0 else mean_vec
                else:
                    rng = np.random.RandomState(self.random_state + iteration + cluster_idx)
                    new_centroids[cluster_idx] = X_norm[rng.choice(n_samples)]

            shift = float(np.linalg.norm(self.centroids - new_centroids))
            self.history.append({
                "iteration": self.iterations_run,
                "inertia": round(iter_inertia, 4),
                "centroid_shift": round(shift, 6)
            })

            self.centroids = new_centroids
            self.labels_ = new_labels

            if shift < self.tol:
                break

        final_dists = self._compute_distances(X_norm, self.centroids)
        self.inertia_ = float(np.sum(np.min(final_dists, axis=1) ** 2))
        return self

    def predict(self, X):
        X_arr, _ = extract_dense_from_model(X)
        row_norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        X_norm = X_arr / row_norms
        distances = self._compute_distances(X_norm, self.centroids)
        return np.argmin(distances, axis=1)

    def get_top_terms_per_cluster(self, feature_names, top_n=8):
        top_terms = {}
        for c_id in range(self.k):
            centroid = self.centroids[c_id]
            top_idx = np.argsort(centroid)[::-1][:top_n]
            top_terms[c_id] = [
                (feature_names[i] if i < len(feature_names) else f"term_{i}", float(centroid[i]))
                for i in top_idx
            ]
        return top_terms


def run_kmeans(tfidf_vectors, n_clusters=5, max_iter=100, random_state=42):
    model = ManualKMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
    model.fit(tfidf_vectors)
    return model.labels_