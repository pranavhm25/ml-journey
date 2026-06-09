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
| **Categorical Handling** | Manual | Manual | Manual / Basic | Native Integer Support | **Advanced Native** |

## 8. Extra Depth

### 8.1 Bagging vs Boosting

Ensemble methods combine multiple machine learning models to create a single, superior predictive model. The two foundational branches of ensemble learning handle errors differently:



* **Bagging (Bootstrap Aggregating):** Generates multiple independent datasets using random sampling with replacement (bootstrapping) from the training pool. A base model is trained on each subset completely independently.
* **Boosting:** Fits a series of weak learners iteratively. Each subsequent model is passed a modified version of the dataset where previously misclassified or poorly predicted instances carry greater weight or form the training residuals.

| Dimension | Bagging | Boosting |
| :--- | :--- | :--- |
| **Training** | **Parallel** (Models are independent) | **Sequential** (Each model depends on previous) |
| **Data Weighting** | Equal probability sampling | Dynamic adjusting based on prediction error |
| **Primary Goal** | **Reduce Variance** (Prevents overfitting) | **Reduce Bias** (Fixes underfitting) |
| **Aggregation Method** | Voting (Classification) or Averaging (Regression) | Weighted sum of all model outputs |
| **Base Learner Bias** | Works best with high-variance, deep models | Works best with low-variance, weak models (stumps) |
| **Representative Example**| Random Forest | XGBoost, LightGBM, CatBoost |

---

### 8.2 Random Forest

A **Random Forest** is an ensemble of unpruned Decision Trees. It relies on the concept of *the wisdom of crowds*—averaging many highly uncorrelated trees to yield a model with significantly lower variance than any single tree.

#### The Core Variance-Reduction Mechanisms
1. **Bagging (Row Subsampling):** Each tree is trained on a unique bootstrap sample containing roughly 63.2% of the original training instances. The remaining 36.8% form the **Out-Of-Bag (OOB) error** pool, which is used for built-in validation without needing a separate validation set.
2. **Feature Randomness (Column Subsampling):** When splitting a node inside a tree, the algorithm does not evaluate all available features. Instead, it picks a random subset of size $m$ (typically $m = \sqrt{p}$ for classification and $m = p/3$ for regression, where $p$ is the total number of features). This decorrelates the trees; otherwise, a single highly dominant feature would be chosen as the top split across every single tree in the forest.

---

### 8.3 Cross-Validation

Relying on a single, static train/validation split can introduce statistical volatility, particularly on smaller datasets. **K-Fold Cross-Validation** provides a robust statistical alternative by ensuring every single observation is used for both training and validation exactly once.

[=========================== Training Data Pool ===========================]<br>
Iteration 1: [ VAL ] [ TRAIN ] [ TRAIN ] [ TRAIN ] [ TRAIN ] -> Score 1<br>
Iteration 2: [ TRAIN ] [ VAL ] [ TRAIN ] [ TRAIN ] [ TRAIN ] -> Score 2<br>
Iteration 3: [ TRAIN ] [ TRAIN ] [ VAL ] [ TRAIN ] [ TRAIN ] -> Score 3<br>
.<br>
.<br>
.<br>
Iteration K: [ TRAIN ] [ TRAIN ] [ TRAIN ] [ TRAIN ] [ VAL ] -> Score K<br>
```
Final Cross-Validation Metric = Average(Score 1 + Score 2 + ... + Score K)
```

#### Advanced CV Strategies
* **Stratified K-Fold:** Crucial for highly imbalanced classification datasets. It ensures each fold contains approximately the same percentage of target class labels as the complete dataset (e.g., maintaining a strict 95:5 ratio of negative to positive classes across all folds).

* **Time-Series Split (Walk-Forward Validation):** Essential for sequential or time-dependent data. Randomly shuffling rows leaks future information into past predictions. Instead, the training window dynamically expands or rolls forward chronologically, evaluating models strictly on subsequent time frames.

---

### 8.4 Evaluation Metrics for Classification

Evaluating a classification model using only a single metric like **Accuracy** can be highly misleading when classes are imbalanced (e.g., a fraud detection model that predicts "Not Fraud" for 100% of cases achieves 99% accuracy on a dataset with 1% fraud, but is completely useless).


#### The Mathematical Metrics

| Metric | Mathematical Formula | Context & When to Use |
| :--- | :--- | :--- |
| **Accuracy** | $$\frac{TP + TN}{TP + TN + FP + FN}$$ | Used when class distributions are highly **balanced** and the cost of False Positives and False Negatives is roughly equal. |
| **Precision** <br>(Positive Predictive Value) | $$\frac{TP}{TP + FP}$$ | **Minimize False Positives.** Use when a false alarm carries high financial or operational costs (e.g., Email Spam Filters—you don't want an urgent invoice sent to the spam folder). |
| **Recall** <br>(Sensitivity / Hit Rate) | $$\frac{TP}{TP + FN}$$ | **Minimize False Negatives.** Use when missing a positive case is critical or life-threatening (e.g., Medical Diagnostics or Fraud Detection—missing a sick patient or a stolen card is disastrous). |
| **F1 Score** | $$2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$ | The harmonic mean of Precision and Recall. Use when seeking an optimal balance between the two on heavily **imbalanced datasets**. |

> **Key:** `TP` = True Positive | `TN` = True Negative | `FP` = False Positive | `FN` = False Negative

#### Advanced Continuous Metrics
* **Confusion Matrix:** A full contingency breakdown of predicted versus actual classifications. You should examine this matrix before selecting a metric to see exactly where the misclassifications occur.
* **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** Measures a classifier's ability to rank-order predictions. It plots the **True Positive Rate (Recall)** against the **False Positive Rate** across every possible classification threshold from 0 to 1. 
  * An $\text{AUC} = 1.0$ indicates perfect separation.
  * An $\text{AUC} = 0.5$ indicates performance equivalent to random guessing.
  * *Advantage:* It evaluates the model independently of any specific probability decision threshold.

---

## 🔗 Resources
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [StatQuest: Gradient Boosting](https://www.youtube.com/watch?v=3CC4N4z3GJc)
- [Towards DS: Bias-Variance Tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)