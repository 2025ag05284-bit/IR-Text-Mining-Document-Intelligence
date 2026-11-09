import math
import random
import re
from collections import Counter, defaultdict
import numpy as np


def default_tokenize(text):
    """Fallback tokenizer: extracts lowercase alphabetic words."""
    return re.findall(r"\b[a-zA-Z]+\b", str(text).lower())


def train_test_split_manual(documents, labels, test_size=0.2, random_state=42, stratify=True):
    """
    Manually split documents and labels into train and test sets without using sklearn.
    
    Parameters
    ----------
    documents : list
        List of document token lists, strings, or records.
    labels : list
        Corresponding ground-truth class labels.
    test_size : float
        Proportion of dataset to include in the test split (default 0.2).
    random_state : int
        Seed for reproducibility.
    stratify : bool
        Whether to preserve class proportions in train and test splits (default True).
        
    Returns
    -------
    X_train, X_test, y_train, y_test, train_indices, test_indices
    """
    rng = random.Random(random_state)
    n = len(documents)
    if n != len(labels):
        raise ValueError(f"Length mismatch: {n} documents vs {len(labels)} labels")

    indices = list(range(n))

    if not stratify:
        rng.shuffle(indices)
        split_idx = int(n * (1.0 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
    else:
        # Group indices by class label to stratify
        class_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            class_to_indices[lbl].append(idx)

        train_idx = []
        test_idx = []

        for lbl in sorted(class_to_indices.keys()):
            lbl_indices = class_to_indices[lbl].copy()
            rng.shuffle(lbl_indices)
            k_test = int(round(len(lbl_indices) * test_size))
            k_test = max(1, k_test) if len(lbl_indices) > 1 and test_size > 0 else 0
            k_test = min(k_test, len(lbl_indices) - 1)  # ensure at least 1 in train

            test_idx.extend(lbl_indices[:k_test])
            train_idx.extend(lbl_indices[k_test:])

        # Shuffle indices to mix classes
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    X_train = [documents[i] for i in train_idx]
    X_test = [documents[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    return X_train, X_test, y_train, y_test, train_idx, test_idx


class ManualMultinomialNB:
    """
    Multinomial Naive Bayes text classifier implemented entirely from scratch.
    
    Mathematical Formulation:
      - Prior probability:
          P(c) = N_c / N
          log P(c) = ln(N_c) - ln(N)
          
      - Likelihood with Laplace / Lidstone smoothing (alpha):
          P(w | c) = (count(w, c) + alpha) / (sum_w' count(w', c) + alpha * |V|)
          log P(w | c) = ln(count(w, c) + alpha) - ln(total_tokens_c + alpha * |V|)
          
      - Posterior score for document d = (w_1, ..., w_m):
          log P(c | d) = log P(c) + sum_{w in d cap V} count(w, d) * log P(w | c)
          
      - Prediction:
          c* = argmax_c log P(c | d)
          
      - Calibrated probabilities using Softmax (Log-Sum-Exp trick):
          P(c | d) = exp(s_c - max(s)) / sum_k exp(s_k - max(s))
    """

    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError("Laplace smoothing parameter alpha must be non-negative.")
        self.alpha = float(alpha)
        self.classes_ = []
        self.class_priors_ = {}          # c -> P(c)
        self.class_log_priors_ = {}      # c -> log P(c)
        self.class_doc_counts_ = {}      # c -> doc count
        self.feature_counts_ = {}        # c -> Counter(word -> count)
        self.class_total_tokens_ = {}    # c -> total word tokens
        self.vocabulary_ = set()         # unique words across all classes
        self.vocab_size_ = 0
        self.log_likelihoods_ = {}       # c -> dict(word -> log P(w|c))
        self.default_log_likelihood_ = {}# c -> log P(unseen | c)

    def _normalize_doc(self, doc):
        """Converts text or token list to a list of token strings."""
        if isinstance(doc, list):
            return [str(w).lower() for w in doc]
        return default_tokenize(doc)

    def fit(self, documents, y):
        """
        Train the Naive Bayes classifier on the given training documents and labels.
        
        Parameters
        ----------
        documents : list of (list of str or str)
            Tokenized or raw text documents.
        y : list of str
            Corresponding class labels.
        """
        n_samples = len(documents)
        if n_samples != len(y):
            raise ValueError(f"Mismatch: {n_samples} documents vs {len(y)} labels.")
        if n_samples == 0:
            raise ValueError("Cannot train on an empty dataset.")

        self.classes_ = sorted(list(set(y)))
        self.class_doc_counts_ = Counter(y)
        self.class_priors_ = {}
        self.class_log_priors_ = {}
        self.feature_counts_ = {c: Counter() for c in self.classes_}
        self.class_total_tokens_ = {c: 0 for c in self.classes_}
        self.vocabulary_ = set()

        # 1. Compute Class Priors & Word Frequencies
        for doc, label in zip(documents, y):
            tokens = self._normalize_doc(doc)
            self.feature_counts_[label].update(tokens)
            self.class_total_tokens_[label] += len(tokens)
            self.vocabulary_.update(tokens)

        self.vocab_size_ = len(self.vocabulary_)

        for c in self.classes_:
            p_c = self.class_doc_counts_[c] / n_samples
            self.class_priors_[c] = p_c
            self.class_log_priors_[c] = math.log(p_c)

        # 2. Compute Class-Conditional Likelihoods with Laplace Smoothing
        self.log_likelihoods_ = {c: {} for c in self.classes_}
        self.default_log_likelihood_ = {}

        for c in self.classes_:
            # Denominator: total tokens in class c + alpha * |V|
            denom = self.class_total_tokens_[c] + (self.alpha * self.vocab_size_)
            if denom == 0:
                denom = 1e-9
            log_denom = math.log(denom)

            # Precompute log-probabilities for all vocabulary words in this class
            for word, count in self.feature_counts_[c].items():
                self.log_likelihoods_[c][word] = math.log(count + self.alpha) - log_denom

            # Default log-likelihood for unseen words in this class (count = 0)
            if self.alpha > 0:
                self.default_log_likelihood_[c] = math.log(self.alpha) - log_denom
            else:
                self.default_log_likelihood_[c] = -1e9

        return self

    def _get_doc_log_posteriors(self, tokens):
        """Compute unnormalized log posterior scores for a single document."""
        token_counts = Counter(tokens)
        log_scores = {}

        for c in self.classes_:
            score = self.class_log_priors_[c]
            c_log_likes = self.log_likelihoods_[c]
            c_default = self.default_log_likelihood_[c]

            for word, cnt in token_counts.items():
                if word in self.vocabulary_:
                    # Known word in training vocabulary
                    word_log_p = c_log_likes.get(word, c_default)
                    score += cnt * word_log_p
                # Out-of-vocabulary words are ignored (standard text mining practice)
            log_scores[c] = score

        return log_scores

    def predict(self, documents):
        """Predict class labels for given documents."""
        predictions = []
        for doc in documents:
            tokens = self._normalize_doc(doc)
            scores = self._get_doc_log_posteriors(tokens)
            # Find class with maximum log posterior
            best_class = max(scores.keys(), key=lambda c: scores[c])
            predictions.append(best_class)
        return predictions

    def predict_proba(self, documents):
        """
        Predict class probabilities for each document using Softmax (Log-Sum-Exp).
        
        Returns
        -------
        list of dict: Each dict maps class_name -> probability in [0, 1].
        """
        all_probs = []
        for doc in documents:
            tokens = self._normalize_doc(doc)
            scores = self._get_doc_log_posteriors(tokens)

            # Log-sum-exp trick for numerical stability
            max_s = max(scores.values())
            exp_scores = {c: math.exp(s - max_s) for c, s in scores.items()}
            sum_exp = sum(exp_scores.values())
            if sum_exp == 0:
                sum_exp = 1e-9

            probs = {c: exp_scores[c] / sum_exp for c in self.classes_}
            all_probs.append(probs)

        return all_probs

    def predict_log_proba(self, documents):
        """Predict unnormalized log-posterior scores for each document."""
        return [self._get_doc_log_posteriors(self._normalize_doc(doc)) for doc in documents]

    def get_top_features_per_class(self, top_n=10, method="log_odds"):
        """
        Extract the most representative terms for each class.
        
        Parameters
        ----------
        top_n : int
            Number of terms to return per class.
        method : str, {'log_odds', 'probability'}
            - 'log_odds': compares log P(w|c) - log P(w|~c) to find distinctive words.
            - 'probability': ranks directly by P(w|c).
            
        Returns
        -------
        dict: {class_label: [(word, score), ...]}
        """
        top_features = {}

        if method == "probability":
            for c in self.classes_:
                c_likes = self.log_likelihoods_[c]
                sorted_words = sorted(
                    [(w, c_likes.get(w, self.default_log_likelihood_[c])) for w in self.vocabulary_],
                    key=lambda item: item[1],
                    reverse=True
                )
                top_features[c] = sorted_words[:top_n]
        else:
            # Log-odds ratio: salience of word in class c vs all other classes combined
            total_other_tokens = {}
            for c in self.classes_:
                total_other_tokens[c] = sum(
                    self.class_total_tokens_[other] for other in self.classes_ if other != c
                )

            for c in self.classes_:
                c_likes = self.log_likelihoods_[c]
                denom_other = total_other_tokens[c] + (self.alpha * self.vocab_size_)
                log_denom_other = math.log(max(denom_other, 1e-9))

                word_scores = []
                for w in self.vocabulary_:
                    # word count in class c
                    count_c = self.feature_counts_[c][w]
                    if count_c == 0:
                        continue  # not in this class

                    # word count in other classes
                    count_other = sum(
                        self.feature_counts_[other][w] for other in self.classes_ if other != c
                    )

                    log_p_c = c_likes.get(w, self.default_log_likelihood_[c])
                    log_p_other = math.log(count_other + self.alpha) - log_denom_other

                    log_odds = log_p_c - log_p_other
                    word_scores.append((w, round(log_odds, 4)))

                word_scores.sort(key=lambda x: x[1], reverse=True)
                top_features[c] = word_scores[:top_n]

        return top_features

