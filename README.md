# Person 4: Unsupervised Document Clustering & Evaluation Module

This directory and module set contains the complete implementation for **Person 4** of the Information Retrieval – Assignment 2 project. All requirements—including manual algorithms, evaluation metrics, and interactive Streamlit UI integration—have been implemented from scratch without relying on built-in machine learning libraries (such as `sklearn.cluster.KMeans` or `sklearn.metrics.rand_score`).

## 📂 Deliverables & File Structure

```
├── modules/
│   ├── kmeans.py               # Manual K-Means implementation & universal TF-IDF adapter
│   └── clustering_metrics.py   # Manual Purity, Rand Index, and Contingency Matrix calculators
├── pages/
│   └── 4_Clustering.py         # Streamlit dashboard page for unsupervised exploration
├── test_kmeans.py              # Standalone test runner and verification script
└── results/
    ├── clustering_results.csv       # Document-to-cluster assignment outputs
    └── clustering_k_experiments.csv # Multi-k experimental evaluation metrics ($k=2 \dots 8$)

```

## ⚙️ Technical Component Breakdown

### 1. Manual K-Means Clustering (`modules/kmeans.py`)

* **Centroid Initialization:** Randomly selects $k$ unique document vectors as initial seeds using a configurable random state.

* **Distance Metric:** Computes vector distances optimized via vectorized NumPy operations ($\vert{}\vert{}x - c\vert{}\vert{}^2 = \vert{}\vert{}x\vert{}\vert{}^2 + \vert{}\vert{}c\vert{}\vert{}^2 - 2(x \cdot c)$). Rows are length-normalized ($L_2 = 1$) so Euclidean distance corresponds directly to Cosine distance.

* **Convergence & Stopping Criteria:** Tracks centroid shift magnitude ($\Delta c$) against tolerance $\epsilon$ and enforces maximum iteration caps.

* **Iteration Trace Logging:** Records inertia (Sum of Squared Distances) and centroid shift per iteration for UI convergence charts.

* **Cluster Profiling:** Extracts the highest-weighted TF-IDF terms per centroid to explain cluster themes.

### 2. Manual Evaluation Metrics (`modules/clustering_metrics.py`)

* **Purity:** Measures the extent to which clusters contain a single class:
  

  $$
  \text{Purity}(\Omega, \mathbb{C}) = \frac{1}{N} \sum_{k} \max_{j} \vert{}\omega_k \cap c_j\vert{}
  $$

* **Rand Index:** Evaluates pairwise agreement between ground-truth labels and cluster partitions across all $\binom{N}{2}$ document pairs ($TP, TN, FP, FN$):
  

  $$
  \text{Rand Index} = \frac{TP + TN}{TP + TN + FP + FN}
  $$

* **Contingency Matrix:** Builds a cross-tabulation table mapping cluster IDs to true news categories.

### 3. Streamlit Dashboard Page (`pages/4_Clustering.py`)

* Provides interactive sidebar controls for hyperparameter tuning ($k$, max iterations, tolerance, random seed).

* Displays live metrics cards, contingency matrices, step-by-step convergence traces, and 2D scatter plots projected via PCA.

* Includes an automated multi-$k$ experiment runner ($k = 2$ to $8$) generating elbow curves and performance graphs.

## 🚀 How to Run and Test

1. **Ensure dependencies are installed** in your active virtual environment:

   ```
   pip install -r requirements.txt
   
   ```

2. **Run the standalone verification test script**:

   ```
   python test_kmeans.py
   
   ```

   *This script verifies dataset loading, tensor conversion via Person 2's TF-IDF model, K-Means convergence, manual purity/rand index calculation, and automatically saves output artifacts to the `results/` folder.*

3. **Launch the full Streamlit dashboard**:

   ```
   streamlit run app.py
   
   ```

   *Navigate to **4_Clustering** from the sidebar to interact with the clustering UI.*