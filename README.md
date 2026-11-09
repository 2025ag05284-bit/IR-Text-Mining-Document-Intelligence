# IR-Text-Mining-Document-Intelligence
An end-to-end Streamlit-based Text Mining and Document Intelligence System for the BBC News corpus implementing manual preprocessing, manual TF-IDF feature extraction, manual Naive Bayes classification, manual K-Means document clustering, and content-based recommendation.

---

## 👥 Module Ownership & Deliverables

| Role | Domain / Focus | Key Modules | Streamlit Page | Verification Script |
| :--- | :--- | :--- | :--- | :--- |
| **Person 1** | Preprocessing & Corpus Analytics | `modules/preprocessing.py`<br>`modules/corpus_analytics.py`<br>`modules/duplicates_detection.py` | `app.py` | `test_preprocessing.py`<br>`test_dataset.py` |
| **Person 2** | Feature Engineering | `modules/inverted_index.py`<br>`modules/tfidf.py`<br>`modules/similarity.py` | `pages/2_Feature_Engineering.py` | `test_tfidf.py` |
| **Person 3** | Supervised Classification | `modules/naive_bayes.py`<br>`modules/classification_metrics.py` | `pages/3_Classification.py` | `test_naive_bayes.py` |
| **Person 4** | Unsupervised Clustering | `modules/kmeans.py`<br>`modules/clustering_metrics.py` | `pages/4_Clustering.py` | `test_kmeans.py` |

---

## ⚖️ Person 3: Supervised Document Classification (Naive Bayes)

This module contains the complete manual implementation for **Person 3** of the Information Retrieval – Assignment 2 project. All algorithms, feature extractions, log-posterior inferences, and classification evaluation metrics are written completely from scratch without importing machine learning libraries such as `sklearn.naive_bayes` or `sklearn.metrics`.

### 📂 File Structure

```
├── modules/
│   ├── naive_bayes.py              # Pure-Python Multinomial Naive Bayes classifier & stratified split
│   └── classification_metrics.py   # Pure-Python Accuracy, Precision, Recall, F1, and Confusion Matrix
├── pages/
│   └── 3_Classification.py         # Streamlit dashboard page for classification and comparative analysis
├── test_naive_bayes.py             # Standalone test runner and verification script
└── results/
    ├── classification_results.csv  # Test set document-level predictions and confidence scores
    ├── classification_metrics.csv  # Macro and weighted summary evaluation metrics
    └── confusion_matrix.csv        # 5x5 category confusion matrix
```

### ⚙️ Technical Component Breakdown

#### 1. Manual Multinomial Naive Bayes (`modules/naive_bayes.py`)
* **Class Prior Probabilities:**
  $$P(c) = \frac{N_c}{N}, \quad \log P(c) = \ln(N_c) - \ln(N)$$
* **Class-Conditional Word Likelihoods with Laplace Smoothing ($\alpha = 1.0$):**
  $$P(w \mid c) = \frac{\text{count}(w, c) + \alpha}{\sum_{w' \in V} \text{count}(w', c) + \alpha \cdot |V|}$$
* **Log-Posterior Inference:**
  $$\log P(c \mid d) = \log P(c) + \sum_{w \in d \cap V} \text{count}(w, d) \cdot \log P(w \mid c)$$
* **Calibrated Probabilities via Softmax (Log-Sum-Exp Trick):**
  $$P(c \mid d) = \frac{\exp(s_c - \max_k s_k)}{\sum_j \exp(s_j - \max_k s_k)}$$
* **Discriminative Feature Salience (Log-Odds Ratio):**
  $$\text{score}(w, c) = \log P(w \mid c) - \log P(w \mid \neg c)$$
* **Manual Stratified Train/Test Split:** Partitions the corpus into 80% train and 20% test while strictly preserving class proportions.

#### 2. Manual Evaluation Metrics (`modules/classification_metrics.py`)
* **Accuracy:** Overall ratio of correctly classified test documents.
* **Per-Class Metrics:** Computes $TP$, $FP$, $FN$, and $TN$ to derive Precision, Recall, and $F_1$-score for each category.
* **Macro & Weighted Averages:** Unweighted and support-weighted arithmetic means across all classes.
* **Confusion Matrix:** 2D contingency table mapping ground-truth categories to predicted categories.

#### 3. Streamlit Classification Dashboard (`pages/3_Classification.py`)
* **Sidebar Controls:** Interactive sliders for test split ratio (10%–40%), Laplace smoothing ($\alpha$), random seed, and preprocessing toggles.
* **Evaluation Metrics:** KPI cards displaying Accuracy, Macro Precision, Macro Recall, and Macro F1 alongside a styled per-category report table.
* **Interactive Visualizations:**
  1. Annotated Confusion Matrix Heatmap.
  2. Grouped bar chart comparing Actual vs. Predicted class counts.
  3. Top discriminative terms per category ranked by log-odds ratio.
* **Interactive Document Sandbox:** Allows testing with pre-selected test set documents or typing arbitrary custom text with live class probability distribution charts.
* **Supervised vs. Unsupervised Comparative Analysis:** Directly links to Person 4's K-Means clustering outputs (`results/clustering_results.csv`), comparing classification metrics (Accuracy & F1) against clustering metrics (Purity & Rand Index).

---

## 🎯 Person 4: Unsupervised Document Clustering (K-Means)

Contains manual K-Means clustering, $L_2$-normalized vector distance computation, iteration convergence logging, and manual Purity and Rand Index evaluation metrics.

* **Modules:** `modules/kmeans.py`, `modules/clustering_metrics.py`
* **Streamlit Page:** `pages/4_Clustering.py`
* **Verification Script:** `python test_kmeans.py`

---

## 🚀 How to Run and Test

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Person 3 Standalone Verification (Naive Bayes)**:
   ```bash
   python test_naive_bayes.py
   ```
   *Expected output on the BBC corpus (80/20 stratified split):*
   * **Accuracy:** `96.62%`
   * **Macro F1-Score:** `96.48%`
   * *Automatically exports results to `results/classification_results.csv` and `results/classification_metrics.csv`.*

3. **Run Person 4 Standalone Verification (K-Means)**:
   ```bash
   python test_kmeans.py
   ```

4. **Launch the Full Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   *In the sidebar, navigate between:*
   * `app.py` - Dataset Overview & Preprocessing (Person 1)
   * `2_Feature_Engineering` - Inverted Index & TF-IDF (Person 2)
   * `3_Classification` - Naive Bayes Classifier (Person 3)
   * `4_Clustering` - K-Means Clustering (Person 4)
