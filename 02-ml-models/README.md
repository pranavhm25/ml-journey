# 📘 Topic 2 — Basic ML Models

---

## 1. Classification

**Classification** is a supervised learning task where the output is a **discrete class label** (not a continuous number).

| Type | Output | Example |
|------|--------|---------|
| **Binary** | 2 classes | Spam / Not Spam |
| **Multiclass** | 3+ classes | Cat / Dog / Bird |
| **Multilabel** | Multiple labels per sample | Movie genres |

---

## 2. Logistic Regression

Despite the name, this is a **classification** algorithm, not regression.

### How it works:
1. Compute a linear combination: `z = β₀ + β₁x₁ + ... + βₙxₙ`
2. Pass through a **sigmoid function** to get a probability:
```
σ(z) = 1 / (1 + e^(-z))        # output ∈ (0, 1)
```
3. Threshold at 0.5: if `σ(z) ≥ 0.5` → class 1, else → class 0

### Loss function (Binary Cross-Entropy):
```
Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

### Key points:
- Outputs probabilities, not just hard labels
- Decision boundary is **linear** (hyperplane)
- For multiclass: use **One-vs-Rest (OvR)** or **Softmax (Multinomial LR)**

---

## 3. Support Vector Machine (SVM)

SVM finds the **optimal hyperplane** that maximally separates two classes.

### Key concepts:

| Term | Meaning |
|------|---------|
| **Hyperplane** | Decision boundary (line in 2D, plane in 3D) |
| **Support Vectors** | Data points closest to the hyperplane |
| **Margin** | Distance between the hyperplane and nearest support vectors |
| **Hard Margin** | Perfectly separable data — no misclassification allowed |
| **Soft Margin (C param)** | Allows some misclassification for noisy data |

### The objective:
```
Maximize margin = 2 / ||w||
Subject to: yᵢ(w·xᵢ + b) ≥ 1
```

### Kernel trick:
When data is not linearly separable, map to higher dimensions using a **kernel function** without actually computing the high-dimensional coordinates.

Common kernels:
| Kernel | Formula | Use case |
|--------|---------|----------|
| **Linear** | `K(x,z) = x·z` | Linearly separable |
| **Polynomial** | `K(x,z) = (x·z + c)^d` | Polynomial boundaries |
| **RBF (Gaussian)** | `K(x,z) = exp(-γ||x-z||²)` | General non-linear |
| **Sigmoid** | `K(x,z) = tanh(αx·z + c)` | Neural-net-like |

---

## 4. Decision Trees

A tree-structured model that splits data based on feature thresholds.

### How it works:
```
Root
├── Feature A ≤ 2.5?
│   ├── YES → Feature B ≤ 1.0?
│   │           ├── YES → Class 0
│   │           └── NO  → Class 1
│   └── NO  → Class 2
```

### Splitting criteria:

| Criterion | Formula | Used for |
|-----------|---------|----------|
| **Gini Impurity** | `1 - Σpᵢ²` | Classification (CART) |
| **Entropy / Info Gain** | `-Σpᵢ log₂(pᵢ)` | Classification (ID3, C4.5) |
| **MSE** | `Σ(yᵢ - ȳ)²` | Regression |

### Hyperparameters:
- `max_depth` — limits tree depth (prevents overfitting)
- `min_samples_split` — minimum samples needed to split a node
- `min_samples_leaf` — minimum samples in a leaf node

### Pros & Cons:
| Pros | Cons |
|------|------|
| Interpretable | Prone to overfitting |
| Handles mixed data types | Unstable (small changes → different tree) |
| No scaling needed | Biased toward features with many levels |

---

## 5. Bias

**Bias** is the error from wrong assumptions in the learning algorithm.

- High bias = model is **too simple**, underfits the data
- Example: fitting a straight line to clearly curved data

```
High Bias → Underfitting → High training error AND high test error
```

---

## 6. Bias-Variance Tradeoff

The ultimate goal of training a model is to minimize **Total Error**, which is driven by two competing forces as model complexity changes:


* **Bias² (Decreases with Complexity):** Error introduced by underestimating the data's complexity. Simple models have high bias because they cannot capture non-linear relationships, leading to **underfitting**.

* **Variance (Increases with Complexity):** Error introduced by overestimating the significance of random noise. Highly flexible models have high variance because they radically alter their predictions based on minor changes in training data, leading to **overfitting**.

* **The Total Error Curve:** Forms a **U-shape**. Total error is high when complexity is too low (dominated by bias) and high when complexity is too high (dominated by variance). The ideal model architecture sits at the lowest point of this curve.

| | Low Complexity Model | High Complexity Model |
|--|---------------------|-----------------------|
| **Bias** | High (underfits) | Low |
| **Variance** | Low | High (overfits) |
| **Training Error** | High | Very low |
| **Test Error** | High | High |

### The sweet spot:
```
Total Error = Bias² + Variance + Irreducible Noise
```
You want the model complexity where `Bias² + Variance` is minimized.

### Visual intuition:
```
Error
  |          \      /
  |           \    /    ← Total Error
  |    Bias²   \  /
  |             \/
  |              \  /← Variance
  |_______________\/______
                  Model Complexity
```

---

## 7. Boosting

**Boosting** is an ensemble technique that trains models **sequentially**, where each new model focuses on the mistakes of the previous ones.

> Analogy: A student who keeps reviewing only the problems they got wrong.

### Types of Boosting:

#### 7.1 AdaBoost (Adaptive Boosting)
- Trains weak classifiers (usually stumps — 1-level trees) sequentially
- Misclassified samples get **higher weights** in the next round
- Final prediction = weighted vote of all classifiers

```
Final output = sign( Σ αₜ * hₜ(x) )
```
where αₜ = weight of classifier t based on its accuracy.

#### 7.2 Gradient Boosting (GBM)
- Each new model fits the **residual errors** (gradient of the loss) of the previous ensemble
- More general than AdaBoost — works with any differentiable loss function

```
F_m(x) = F_{m-1}(x) + η * hₘ(x)
```
where η = learning rate, hₘ = new weak learner fitted to residuals

#### 7.3 XGBoost (Extreme Gradient Boosting)
- Faster and regularized version of GBM
- Uses **second-order gradient** (Hessian) for better optimization
- Built-in L1/L2 regularization
- Handles missing values natively
- **Industry standard for tabular data competitions**

#### 7.4 LightGBM
- Uses **Leaf-wise** tree growth (vs Level-wise in XGBoost)
- Faster on large datasets
- Better for high-dimensional sparse data

#### 7.5 CatBoost
- Designed for **categorical features** — no need for manual encoding
- Uses ordered boosting to avoid target leakage
- Often best out-of-the-box without tuning

### Comparison:

| | AdaBoost | GBM | XGBoost | LightGBM | CatBoost |
|--|----------|-----|---------|----------|----------|
| **Speed** | Fast | Slow | Fast | Fastest | Moderate |
| **Regularization** | No | No | Yes | Yes | Yes |
| **Categorical** | Manual | Manual | Manual | Manual | Built-in |
| **Best for** | Simple problems | General | Competitions | Large data | Categorical |

---

## 8. Extra Depth

### 8.1 Bagging vs Boosting
| | Bagging | Boosting |
|--|---------|---------|
| **Training** | Parallel (independent models) | Sequential (each fixes previous) |
| **Goal** | Reduce variance | Reduce bias |
| **Example** | Random Forest | XGBoost |

### 8.2 Random Forest
- An ensemble of Decision Trees trained on **random subsets** of data and features
- Combines them by majority voting (classification) or averaging (regression)
- More robust than a single tree — reduces overfitting

### 8.3 Cross-Validation
Instead of a single train/test split:

**k-Fold Cross-Validation:**
```
Split data into k folds
For each fold i:
    Train on all folds except i
    Test on fold i
Final score = average of k scores
```

Standard: `k=5` or `k=10`

### 8.4 Evaluation Metrics for Classification
| Metric | Formula | When to use |
|--------|---------|-------------|
| **Accuracy** | Correct / Total | Balanced classes |
| **Precision** | TP / (TP + FP) | When false positives are costly |
| **Recall** | TP / (TP + FN) | When false negatives are costly |
| **F1 Score** | 2 * P*R / (P+R) | Imbalanced classes |
| **AUC-ROC** | Area under ROC curve | Ranking quality |
| **Confusion Matrix** | Full breakdown | Always show this |

---

## 🔗 Resources
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [StatQuest: Gradient Boosting](https://www.youtube.com/watch?v=3CC4N4z3GJc)
- [Towards DS: Bias-Variance Tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
