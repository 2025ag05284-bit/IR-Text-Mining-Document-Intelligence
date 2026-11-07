from collections import Counter
import numpy as np


def compute_purity(y_true, cluster_labels):
    y_true_arr = np.asarray(y_true)
    labels_arr = np.asarray(cluster_labels)
    n = len(y_true_arr)

    if n == 0:
        return 0.0

    unique_clusters = np.unique(labels_arr)
    majority_sum = 0

    for cluster_id in unique_clusters:
        members = y_true_arr[labels_arr == cluster_id]
        if len(members) > 0:
            most_common_count = Counter(members).most_common(1)[0][1]
            majority_sum += most_common_count

    return float(majority_sum / n)


def compute_rand_index(y_true, cluster_labels):
    counts = compute_pair_confusion_counts(y_true, cluster_labels)
    total = counts["Total Pairs"]
    if total == 0:
        return 1.0
    return float((counts["TP"] + counts["TN"]) / total)


def compute_pair_confusion_counts(y_true, cluster_labels):
    y_true_arr = np.asarray(y_true)
    labels_arr = np.asarray(cluster_labels)
    n = len(y_true_arr)

    if n < 2:
        return {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "Total Pairs": 0}

    same_class = y_true_arr[:, None] == y_true_arr[None, :]
    same_cluster = labels_arr[:, None] == labels_arr[None, :]
    triu_idx = np.triu_indices(n, k=1)

    sc = same_class[triu_idx]
    sk = same_cluster[triu_idx]

    tp = int(np.sum(sc & sk))
    tn = int(np.sum((~sc) & (~sk)))
    fp = int(np.sum((~sc) & sk))
    fn = int(np.sum(sc & (~sk)))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "Total Pairs": tp + tn + fp + fn}


def build_contingency_matrix(y_true, cluster_labels):
    categories = sorted([str(c) for c in set(y_true)])
    clusters = sorted([int(c) for c in set(cluster_labels)])

    matrix = {f"Cluster {cl}": {cat: 0 for cat in categories} for cl in clusters}
    for yt, cl in zip(y_true, cluster_labels):
        matrix[f"Cluster {int(cl)}"][str(yt)] += 1

    return matrix