# Multi-Class Classification Benchmark: PyTorch MLP vs. KNN vs. K-Means

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end comparative study benchmarking a custom **PyTorch Multi-Layer Perceptron (MLP)** against classical supervised learning (**K-Nearest Neighbors**) and unsupervised clustering (**K-Means**) on 2D synthetic multi-cluster datasets (`make_blobs`).

---

## 📌 Overview

When tackling multi-class classification, practitioners have multiple algorithmic paradigms at their disposal:
1. **Deep Learning (Supervised):** Parametric neural networks that learn continuous decision boundaries through backpropagation.
2. **Classical Machine Learning (Supervised):** Non-parametric instance-based algorithms that classify based on local feature space proximity.
3. **Unsupervised Learning:** Centroid-based clustering that groups data purely based on geometric variance without supervision.

This project implements, evaluates, and compares all three approaches on 2D synthetic Gaussian clusters, visualizing decision boundaries, cluster assignments, and classification metrics.

---

## 📊 Benchmark Results

| Model | Paradigm | Primary Evaluation Metric | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **PyTorch MLP** | Supervised (Deep Learning) | **Test Accuracy: 98.25%**<br>*(Test Loss: 0.060)* | Achieved the highest accuracy by learning smooth, optimal linear/non-linear decision boundaries through gradient descent. |
| **K-Nearest Neighbors ($k=5$)** | Supervised (Classical ML) | **Test Accuracy: 91.75%** | Robust non-parametric baseline; fast to train, but misclassifications occur along overlapping cluster borders. |
| **K-Means Clustering ($k=5$)** | Unsupervised Learning | **Adjusted Rand Index: 0.82**<br>*(Silhouette Score: 0.52)* | Reconstructed the underlying cluster structure without ground truth labels, demonstrating strong unsupervised grouping. |

---

## 🧠 Methodology & Implementations

### 1. PyTorch Multi-Layer Perceptron (`BlobModel`)
* **Architecture:** Custom `nn.Module` subclass with a 3-layer `linear_layer_stack` (2 input features $\to$ 8 hidden $\to$ 8 hidden $\to$ output classes).
* **Optimization:** `nn.CrossEntropyLoss` with Stochastic Gradient Descent (`torch.optim.SGD`, $\text{lr}=0.1$).
* **Device Agnostic:** Seamless execution across CUDA GPU and CPU runtimes.
* **Evaluation & Visualization:** Decision boundary contour plots mapped over training and test distributions.

```python
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
```

### 2. K-Means Clustering (Unsupervised)
* **Configuration:** $k=5$ clusters initialized with `k-means++` over 10 random restarts (`n_init=10`).
* **Cluster Quality Metrics:**
  * **Adjusted Rand Index (ARI = 0.82):** Measures similarity between predicted cluster assignments and true labels, adjusted for chance.
  * **Silhouette Score (0.52):** Evaluates intra-cluster tightness against inter-cluster separation.
* **Visualizations:** Side-by-side scatter plots comparing ground-truth distributions against partitioned clusters with marked centroid locations.

### 3. K-Nearest Neighbors (Classical Supervised)
* **Configuration:** $k=5$ neighbors with Euclidean distance weighting.
* **Evaluation:**
  * **Test Accuracy:** 91.75%
  * **Confusion Matrix & Classification Report:** Comprehensive breakdown of precision, recall, and F1-score across all classes.
  * **Prediction Scatter Plot:** Visual mapping of test sample predictions across the 2D plane.

---

## 📁 Repository Structure

```text
├── blob_classification_benchmark.ipynb   # Main Jupyter/Colab notebook
├── helper_functions.py                  # Plotting & decision boundary utilities
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation & results
```

---

## 🚀 Getting Started

### Prerequisites & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RishiRaj1112/PyTorch-Blobs-Classification-Benchmark.git
   cd PyTorch-Blobs-Classification-Benchmark
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/macOS
   .\venv\Scripts\activate        # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the notebook:**
   ```bash
   jupyter notebook blob_classification_benchmark.ipynb
   ```

---

## 🙏 Acknowledgments & References

* Foundational PyTorch workflow, device agnostic setup, and decision boundary helper utilities adapted from **Daniel Bourke's** [PyTorch Deep Learning Course](https://github.com/mrdbourke/pytorch-deep-learning) (FreeCodeCamp).
* Extended with custom comparative evaluations across classical supervised learning (KNN) and unsupervised clustering (K-Means).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
