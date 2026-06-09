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

**Boosting** is an ensemble learning technique that transforms a collection of weak learners (models that perform only slightly better than random guessing) into a single strong learner. Unlike Bagging (e.g., Random Forest), which trains models in parallel, Boosting trains models **sequentially**. Each new model is explicitly trained to correct the errors made by the models that came before it.

> **Analogy:** Imagine a student preparing for an exam. Instead of re-reading the entire textbook equally, they take a practice test, identify the exact problems they got wrong, and spend the next hour studying *only* those specific weak areas. They repeat this cycle until they minimize their mistakes.



---

### Types of Boosting

#### 7.1 AdaBoost (Adaptive Boosting)
AdaBoost is the foundational boosting algorithm. It typically uses "decision stumps"—decision trees with a depth of exactly one layer (a single split)—as its weak learners.

* **The Core Mechanism:** It adjusts data weights dynamically. Every sample starts with equal weight. After a tree is trained, the algorithm increases the weights of misclassified samples and decreases the weights of correctly classified ones. The next tree is forced to focus its attention on the high-weight, difficult samples.
* **Mathematical Intuition:** The final prediction is a weighted majority vote, where more accurate trees carry higher voting power ($\alpha_t$):
  $$\text{Final Output} = \text{sign}\left( \sum_{t=1}^{T} \alpha_t h_t(x) \right)$$
  Where $h_t(x)$ is the prediction of stump $t$, and $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$ (derived from the error rate $\epsilon_t$).

#### 7.2 Gradient Boosting Machine (GBM)
While AdaBoost adjusts sample weights based on classification errors, Gradient Boosting optimizes a differentiable loss function (like Mean Squared Error or Log Loss) by fitting new models directly to the **residuals** (the errors) of the previous ensemble.

* **The Core Mechanism:** Instead of changing data weights, the next weak learner (typically a shallow decision tree) predicts the *leftover error* made by the existing combination of trees. 
* **Mathematical Intuition:** Each step moves the ensemble's predictions down the gradient of the loss function (gradient descent):
  $$F_m(x) = F_{m-1}(x) + \eta h_m(x)$$
  Where $F_{m-1}(x)$ is the current ensemble prediction, $h_m(x)$ is the new tree trained on the residuals, and $\eta$ (eta) is the learning rate/shrinkage factor used to control step size and prevent overfitting.

#### 7.3 XGBoost (Extreme Gradient Boosting)
XGBoost is a highly optimized, scalable implementation of Gradient Boosting designed for maximum computational efficiency and predictive power.

* **The Core Mechanism:** * **Second-Order Optimization:** While standard GBM uses only the first derivative (gradient), XGBoost uses a Taylor expansion up to the **second derivative** (Hessian matrix) of the loss function, allowing for faster and more precise convergence.
  * **Built-in Regularization:** Adds L1 (Lasso) and L2 (Ridge) penalties directly to the tree-building objective function to control tree complexity and prevent overfitting.
  * **System Features:** Supports parallel tree building, block structure caching, and handles missing values natively by automatically learning the best default splitting direction for empty cells.
* **Industry Status:** The historical standard for winning tabular data competitions (Kaggle).

#### 7.4 LightGBM (Light Gradient Boosting Machine)
Developed by Microsoft, LightGBM was engineered to handle massive datasets with faster training speeds and lower memory consumption than standard XGBoost.

* **The Core Mechanism:**
  * **Leaf-wise Growth:** Most boosting frameworks grow trees level-by-level (horizontally). LightGBM grows trees leaf-wise (vertically). It chooses the specific leaf that will reduce the overall loss the most, resulting in deeper, asymmetric trees that achieve higher accuracy faster (though it requires careful tuning via max_depth to avoid overfitting).
  * **GOSS & EFB:** Uses *Gradient-based One-Side Sampling* (keeping instances with large gradients and downsampling those with small gradients) and *Exclusive Feature Bundling* to drastically reduce the number of data points and features processed during splits.



#### 7.5 CatBoost (Categorical Boosting)
Developed by Yandex, CatBoost is optimized out of the box for datasets that contain heavily categorical text or structural features.

* **The Core Mechanism:**
  * **Symmetric Trees:** It builds oblivious/symmetric trees where the same splitting criteria is used across the entire level of the tree. This acts as a regularizer and makes execution during prediction incredibly fast.
  * **Ordered Boosting:** Traditional boosting suffers from target leakage because the residuals used at a step are computed using the same target values. CatBoost uses a permutation-based approach to compute residuals without leaking future target data.
  * **Native Categorical Support:** Automatically handles categorical features using advanced target statistics Encodings, completely eliminating the need for manual One-Hot or Label Encoding before training.

---

### Structural Comparison

| Feature / Metric | AdaBoost | GBM | XGBoost | LightGBM | CatBoost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Weak Learner Split** | Sample Weights | Residuals (Gradient) | Residuals (Grad + Hessian) | Residuals (Grad + Hessian) | Residuals (Permutations) |
| **Tree Growth Strategy** | Stumps (1-Level) | Level-wise | Level-wise | Leaf-wise (Asymmetric) | Level-wise (Symmetric) |
| **Speed & Scaling** | Fast (Simple) | Slow (Sequential) | Fast (Parallelized) | **Fastest** (GOSS/EFB) | Moderate (Fast inference) |
| **Overfitting Risk** | Low-Moderate | High (If untuned) | Low-Moderate (Regularized) | High (Requires depth limits) | **Lowest** (Ordered boosting) |
| **Categorical Handling** | Manual | Manual | Manual / Basic | Native Integer Support | **Advanced Native** |## 7. Boosting

**Boosting** is an ensemble learning technique that transforms a collection of weak learners (models that perform only slightly better than random guessing) into a single strong learner. Unlike Bagging (e.g., Random Forest), which trains models in parallel, Boosting trains models **sequentially**. Each new model is explicitly trained to correct the errors made by the models that came before it.

> **Analogy:** Imagine a student preparing for an exam. Instead of re-reading the entire textbook equally, they take a practice test, identify the exact problems they got wrong, and spend the next hour studying *only* those specific weak areas. They repeat this cycle until they minimize their mistakes.



---

### Types of Boosting

#### 7.1 AdaBoost (Adaptive Boosting)
AdaBoost is the foundational boosting algorithm. It typically uses "decision stumps"—decision trees with a depth of exactly one layer (a single split)—as its weak learners.

* **The Core Mechanism:** It adjusts data weights dynamically. Every sample starts with equal weight. After a tree is trained, the algorithm increases the weights of misclassified samples and decreases the weights of correctly classified ones. The next tree is forced to focus its attention on the high-weight, difficult samples.
* **Mathematical Intuition:** The final prediction is a weighted majority vote, where more accurate trees carry higher voting power ($\alpha_t$):
  $$\text{Final Output} = \text{sign}\left( \sum_{t=1}^{T} \alpha_t h_t(x) \right)$$
  Where $h_t(x)$ is the prediction of stump $t$, and $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$ (derived from the error rate $\epsilon_t$).

#### 7.2 Gradient Boosting Machine (GBM)
While AdaBoost adjusts sample weights based on classification errors, Gradient Boosting optimizes a differentiable loss function (like Mean Squared Error or Log Loss) by fitting new models directly to the **residuals** (the errors) of the previous ensemble.

* **The Core Mechanism:** Instead of changing data weights, the next weak learner (typically a shallow decision tree) predicts the *leftover error* made by the existing combination of trees. 
* **Mathematical Intuition:** Each step moves the ensemble's predictions down the gradient of the loss function (gradient descent):
  $$F_m(x) = F_{m-1}(x) + \eta h_m(x)$$
  Where $F_{m-1}(x)$ is the current ensemble prediction, $h_m(x)$ is the new tree trained on the residuals, and $\eta$ (eta) is the learning rate/shrinkage factor used to control step size and prevent overfitting.

#### 7.3 XGBoost (Extreme Gradient Boosting)
XGBoost is a highly optimized, scalable implementation of Gradient Boosting designed for maximum computational efficiency and predictive power.

* **The Core Mechanism:** * **Second-Order Optimization:** While standard GBM uses only the first derivative (gradient), XGBoost uses a Taylor expansion up to the **second derivative** (Hessian matrix) of the loss function, allowing for faster and more precise convergence.
  * **Built-in Regularization:** Adds L1 (Lasso) and L2 (Ridge) penalties directly to the tree-building objective function to control tree complexity and prevent overfitting.
  * **System Features:** Supports parallel tree building, block structure caching, and handles missing values natively by automatically learning the best default splitting direction for empty cells.
* **Industry Status:** The historical standard for winning tabular data competitions (Kaggle).

#### 7.4 LightGBM (Light Gradient Boosting Machine)
Developed by Microsoft, LightGBM was engineered to handle massive datasets with faster training speeds and lower memory consumption than standard XGBoost.

* **The Core Mechanism:**
  * **Leaf-wise Growth:** Most boosting frameworks grow trees level-by-level (horizontally). LightGBM grows trees leaf-wise (vertically). It chooses the specific leaf that will reduce the overall loss the most, resulting in deeper, asymmetric trees that achieve higher accuracy faster (though it requires careful tuning via max_depth to avoid overfitting).
  * **GOSS & EFB:** Uses *Gradient-based One-Side Sampling* (keeping instances with large gradients and downsampling those with small gradients) and *Exclusive Feature Bundling* to drastically reduce the number of data points and features processed during splits.



#### 7.5 CatBoost (Categorical Boosting)
Developed by Yandex, CatBoost is optimized out of the box for datasets that contain heavily categorical text or structural features.

* **The Core Mechanism:**
  * **Symmetric Trees:** It builds oblivious/symmetric trees where the same splitting criteria is used across the entire level of the tree. This acts as a regularizer and makes execution during prediction incredibly fast.
  * **Ordered Boosting:** Traditional boosting suffers from target leakage because the residuals used at a step are computed using the same target values. CatBoost uses a permutation-based approach to compute residuals without leaking future target data.
  * **Native Categorical Support:** Automatically handles categorical features using advanced target statistics Encodings, completely eliminating the need for manual One-Hot or Label Encoding before training.

---

### Structural Comparison

| Feature / Metric | AdaBoost | GBM | XGBoost | LightGBM | CatBoost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Weak Learner Split** | Sample Weights | Residuals (Gradient) | Residuals (Grad + Hessian) | Residuals (Grad + Hessian) | Residuals (Permutations) |
| **Tree Growth Strategy** | Stumps (1-Level) | Level-wise | Level-wise | Leaf-wise (Asymmetric) | Level-wise (Symmetric) |
| **Speed & Scaling** | Fast (Simple) | Slow (Sequential) | Fast (Parallelized) | **Fastest** (GOSS/EFB) | Moderate (Fast inference) |
| **Overfitting Risk** | Low-Moderate | High (If untuned) | Low-Moderate (Regularized) | High (Requires depth limits) | **Lowest** (Ordered boosting) |
| **Categorical Handling** | Manual | Manual | Manual / Basic | Native Integer Support | **Advanced Native** |

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
