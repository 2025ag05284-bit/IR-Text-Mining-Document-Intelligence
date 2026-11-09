from collections import Counter
import numpy as np


def compute_confusion_matrix(y_true, y_pred, labels=None):
    """
    Computes a confusion matrix manually without using sklearn.
    
    Parameters
    ----------
    y_true : list
        True ground-truth class labels.
    y_pred : list
        Predicted class labels.
    labels : list, optional
        Order of unique labels. If None, uses sorted unique labels from y_true and y_pred.
        
    Returns
    -------
    matrix : list of list of int
        Row index corresponds to True class, Column index corresponds to Predicted class.
    labels : list of str
        The class label ordering.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    k = len(labels)
    matrix = [[0] * k for _ in range(k)]

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            r = label_to_idx[yt]
            c = label_to_idx[yp]
            matrix[r][c] += 1

    return matrix, labels


def compute_accuracy(y_true, y_pred):
    """Computes overall classification accuracy."""
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return float(correct / len(y_true))


def compute_per_class_metrics(y_true, y_pred, labels=None):
    """
    Computes per-class TP, FP, FN, TN, Precision, Recall, and F1-score manually.
    
    Returns
    -------
    dict: {label: {'precision': float, 'recall': float, 'f1': float, 'support': int, ...}}
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    matrix, ordered_labels = compute_confusion_matrix(y_true, y_pred, labels=labels)
    k = len(ordered_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(ordered_labels)}
    total_samples = len(y_true)

    metrics = {}

    for lbl in ordered_labels:
        i = label_to_idx[lbl]
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(k) if r != i)
        fn = sum(matrix[i][c] for c in range(k) if c != i)
        tn = total_samples - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        metrics[lbl] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support)
        }

    return metrics


def compute_macro_metrics(y_true, y_pred, labels=None):
    """
    Computes Macro and Weighted Precision, Recall, and F1-scores.
    
    Returns
    -------
    dict: {'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'}
    """
    per_class = compute_per_class_metrics(y_true, y_pred, labels=labels)
    acc = compute_accuracy(y_true, y_pred)

    n_classes = len(per_class)
    if n_classes == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0
        }

    macro_p = sum(m["precision"] for m in per_class.values()) / n_classes
    macro_r = sum(m["recall"] for m in per_class.values()) / n_classes
    macro_f1 = sum(m["f1"] for m in per_class.values()) / n_classes

    total_support = sum(m["support"] for m in per_class.values())
    if total_support > 0:
        weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / total_support
        weighted_p = sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
        weighted_r = sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
    else:
        weighted_f1 = 0.0
        weighted_p = 0.0
        weighted_r = 0.0

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1)
    }


def classification_report_table(y_true, y_pred, labels=None):
    """
    Builds a summary list of dictionaries formatted like classification_report.
    Suitable for direct conversion to a pandas DataFrame.
    """
    per_class = compute_per_class_metrics(y_true, y_pred, labels=labels)
    macro = compute_macro_metrics(y_true, y_pred, labels=labels)
    total_support = len(y_true)

    rows = []
    for lbl, m in per_class.items():
        rows.append({
            "Class": lbl,
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1"], 4),
            "Support": m["support"]
        })

    # Summary rows
    rows.append({
        "Class": "Accuracy",
        "Precision": "-",
        "Recall": "-",
        "F1-Score": round(macro["accuracy"], 4),
        "Support": total_support
    })
    rows.append({
        "Class": "Macro Avg",
        "Precision": round(macro["macro_precision"], 4),
        "Recall": round(macro["macro_recall"], 4),
        "F1-Score": round(macro["macro_f1"], 4),
        "Support": total_support
    })
    rows.append({
        "Class": "Weighted Avg",
        "Precision": round(macro["weighted_precision"], 4),
        "Recall": round(macro["weighted_recall"], 4),
        "F1-Score": round(macro["weighted_f1"], 4),
        "Support": total_support
    })

    return rows

