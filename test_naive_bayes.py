import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

from modules.preprocessing import preprocess_documents
from modules.naive_bayes import ManualMultinomialNB, train_test_split_manual
from modules.classification_metrics import (
    compute_confusion_matrix,
    compute_accuracy,
    compute_macro_metrics,
    compute_per_class_metrics,
    classification_report_table
)
from modules.clustering_metrics import compute_purity, compute_rand_index


def find_dataset():
    candidates = [
        Path("data/processed_dataset.csv"),
        Path("data/cleaned_dataset.csv"),
        Path("data/raw_dataset.csv")
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    print("=" * 70)
    print("PERSON 3: NAIVE BAYES CLASSIFICATION & EVALUATION VERIFICATION")
    print("=" * 70)

    dataset_path = find_dataset()
    if not dataset_path:
        raise FileNotFoundError("No dataset found in data/")

    print(f"\n[1/6] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"      Total documents loaded: {len(df)}")
    if "category" not in df.columns:
        raise ValueError("Missing 'category' column in dataset!")
    print(f"      Categories: {sorted(df['category'].unique().tolist())}")
    print(f"      Distribution:\n{df['category'].value_counts().to_string()}")

    # 2. Extract or preprocess tokens
    print("\n[2/6] Extracting token representations...")
    if "tokens" in df.columns and pd.notna(df["tokens"].iloc[0]):
        sample = df["tokens"].iloc[0]
        if isinstance(sample, list):
            token_documents = df["tokens"].tolist()
        else:
            token_documents = [json.loads(t) for t in df["tokens"]]
        print("      Loaded precomputed tokens from dataset.")
    else:
        documents = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()
        print("      Preprocessing raw text (lowercase, stopword removal)...")
        token_documents = preprocess_documents(
            documents,
            lowercase=True,
            remove_stopword=True,
            stemming=False,
            lemmatization=False
        )
        print(f"      Tokenized {len(token_documents)} documents.")

    labels = df["category"].astype(str).tolist()

    # 3. Train / Test Split
    print("\n[3/6] Performing manual stratified Train/Test split (80/20)...")
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_manual(
        token_documents,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=True
    )
    print(f"      Training samples: {len(X_train)} ({len(X_train)/len(labels)*100:.1f}%)")
    print(f"      Testing samples:  {len(X_test)} ({len(X_test)/len(labels)*100:.1f}%)")

    # 4. Fit Manual Naive Bayes
    print("\n[4/6] Training Manual Multinomial Naive Bayes (alpha=1.0)...")
    nb = ManualMultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    print(f"      Vocabulary size: {nb.vocab_size_:,} unique terms")
    print("      Class priors:")
    for c, p in nb.class_priors_.items():
        print(f"        - {c:15s}: {p:.4f} (log={nb.class_log_priors_[c]:.4f})")

    # 5. Evaluate on Test Set
    print("\n[5/6] Predicting test documents & calculating manual metrics...")
    y_pred = nb.predict(X_test)
    probs = nb.predict_proba(X_test)

    # Metrics
    acc = compute_accuracy(y_test, y_pred)
    macro = compute_macro_metrics(y_test, y_pred)
    conf_mat, ordered_labels = compute_confusion_matrix(y_test, y_pred)
    report_rows = classification_report_table(y_test, y_pred, labels=ordered_labels)

    print(f"\n      >>> Accuracy:        {acc * 100:.2f}%")
    print(f"      >>> Macro Precision: {macro['macro_precision'] * 100:.2f}%")
    print(f"      >>> Macro Recall:    {macro['macro_recall'] * 100:.2f}%")
    print(f"      >>> Macro F1-Score:  {macro['macro_f1'] * 100:.2f}%")

    print("\nClassification Report (Manual):")
    report_df = pd.DataFrame(report_rows)
    print(report_df.to_string(index=False))

    print("\nConfusion Matrix (Rows: True, Cols: Predicted):")
    conf_df = pd.DataFrame(conf_mat, index=ordered_labels, columns=ordered_labels)
    print(conf_df)

    # Top discriminative terms per category
    print("\nTop 5 Distinctive Words per Category (Log-Odds Ratio):")
    top_terms = nb.get_top_features_per_class(top_n=5, method="log_odds")
    for c, terms in top_terms.items():
        words_str = ", ".join([f"{w} ({score})" for w, score in terms])
        print(f"  {c:15s}: {words_str}")

    # 6. Save Artifacts to results/
    print("\n[6/6] Exporting results to results/ directory...")
    os.makedirs("results", exist_ok=True)

    # Save detailed predictions for test set
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    test_df["true_category"] = y_test
    test_df["predicted_category"] = y_pred
    test_df["confidence"] = [p[yp] for p, yp in zip(probs, y_pred)]

    out_cols = [c for c in ["document_id", "title", "true_category", "predicted_category", "confidence"] if c in test_df.columns]
    test_df[out_cols].to_csv("results/classification_results.csv", index=False)
    print("      Saved results/classification_results.csv")

    # Save summary metrics
    pd.DataFrame([macro]).to_csv("results/classification_metrics.csv", index=False)
    print("      Saved results/classification_metrics.csv")

    # Save confusion matrix
    conf_df.to_csv("results/confusion_matrix.csv")
    print("      Saved results/confusion_matrix.csv")

    # Check for Person 4 clustering results to compare
    cluster_res_path = Path("results/clustering_results.csv")
    if cluster_res_path.exists():
        cdf = pd.read_csv(cluster_res_path)
        if "cluster" in cdf.columns and "category" in cdf.columns:
            purity = compute_purity(cdf["category"], cdf["cluster"])
            rand_idx = compute_rand_index(cdf["category"], cdf["cluster"])
            print("\n" + "=" * 70)
            print("COMPARATIVE ANALYSIS: SUPERVISED (NB) VS UNSUPERVISED (K-MEANS)")
            print("=" * 70)
            print(f"  Supervised Naive Bayes:")
            print(f"    - Accuracy:         {acc * 100:.2f}%")
            print(f"    - Macro F1-Score:   {macro['macro_f1'] * 100:.2f}%")
            print(f"  Unsupervised K-Means:")
            print(f"    - Purity:           {purity * 100:.2f}%")
            print(f"    - Rand Index:       {rand_idx * 100:.2f}%")
            print("=" * 70)

    print("\nVerification completed successfully!")


if __name__ == "__main__":
    main()

