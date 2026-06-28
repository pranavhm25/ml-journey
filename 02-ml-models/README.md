# 📘 Topic 2 — Basic ML Models

---

## 1. Classification

**Classification** is a supervised learning task where the target variable is categorical, meaning the goal is to map input features $X$ to a **discrete class label** $Y$. Rather than predicting a continuous value, classification algorithms output either a hard label or the conditional probability of an input instance belonging to each possible class.



### Core Taxonomies of Classification

The objective and structural configuration of a classification model depend heavily on the nature of its target labels:

| Classification Type | Target Cardinality ($Y$) | Probabilistic Constraint / Output Activation | Real-World Engineering Example |
| :--- | :--- | :--- | :--- |
| **Binary Classification** | Exactly **2** mutually exclusive classes <br>(e.g., $Y \in \{0, 1\}$) | Often uses a **Sigmoid** activation function to output a single probability $P(Y=1 \vert X)$. | • **Spam Filtering:** Spam vs. Ham.<br>• **Anomaly Detection:** Fraud vs. Legitimate.<br>• **Diagnostics:** Disease Present vs. Absent. |
| **Multiclass Classification** | **3 or more** mutually exclusive classes <br>(e.g., $Y \in \{1, 2, \dots, K\}$) | Often uses a **Softmax** activation function to output a probability distribution across all $K$ classes where $\sum P(Y_k \vert X) = 1$. | • **Computer Vision:** Classifying an image explicitly as *either* a Cat, Dog, or Bird.<br>• **Optical Character Recognition (OCR):** Identifying handwritten digits ($0-9$). |
| **Multilabel Classification** | **Multiple** non-mutually exclusive labels per instance | Often uses multiple independent **Sigmoid** activations. A single data point can simultaneously belong to zero, one, or all classes. | • **Document Tagging:** Assigning multiple genres (e.g., *Sci-Fi*, *Action*, and *Thrillar*) to a single movie.<br>• **Image Tagging:** Identifying all objects present in a photo (e.g., `['car', 'tree', 'pedestrian']`). |

---

### The Decision Boundary

At its core, training a classification model is the process of learning a geometric **Decision Boundary** that partition the high-dimensional feature space into regions assigned to distinct classes. 

* **Linear Classifiers (e.g., Logistic Regression, Linear SVM):** Construct flat hyperplanes as boundaries to separate classes.
* **Non-Linear Classifiers (e.g., Random Forest, KNN, RBF-SVM):** Learn complex, curved, or jagged boundaries capable of enclosing intricately clustered datasets.

---

## 2. Logistic Regression

Despite containing "Regression" in its name, Logistic Regression is a foundational **supervised classification** algorithm used to predict the probability of a discrete target variable.



### The Core Mechanism

Logistic Regression operates by taking a standard linear regression equation and compressing its infinite real-value output into a bounded probability space.

1. **Linear Combination ($z$):** Compute the log-odds (logit) score by taking a linear combination of input features:
   $$z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$$

2. **The Sigmoid (Logistic) Activation:** To map $z$ (which ranges from $-\infty$ to $+\infty$) into a valid probability space bounded strictly between $0$ and $1$, pass it through the **Sigmoid function**:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   Where $\hat{y} = \sigma(z) = P(Y=1 \vert X)$.

3. **Classification Thresholding:** Convert the continuous probability output $\hat{y}$ into a hard discrete class label by applying a decision threshold (default is $0.5$):
   $$\text{Predicted Class} = \begin{cases} 1 & \text{if } \sigma(z) \ge 0.5 \quad (z \ge 0) \\ 0 & \text{if } \sigma(z) < 0.5 \quad (z < 0) \end{cases}$$

---

### Loss Function: Binary Cross-Entropy (Log Loss)

Logistic Regression cannot use Ordinary Least Squares (OLS) because the Sigmoid function turns the resulting loss surface highly non-convex, introducing numerous local minima. Instead, it is trained using **Maximum Likelihood Estimation (MLE)**, which mathematically simplifies to minimizing **Binary Cross-Entropy Loss**:

$$\mathcal{L}(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

#### Why this mathematical structure works:
* **When Actual $y = 1$:** The right half of the equation $(1-y)$ becomes $0$, leaving $\mathcal{L} = -\log(\hat{y})$. If the model correctly predicts $\hat{y} \to 1$, the loss approaches $0$. If it incorrectly predicts $\hat{y} \to 0$, the loss penalizes heavily by approaching $\infty$.
* **When Actual $y = 0$:** The left half of the equation becomes $0$, leaving $\mathcal{L} = -\log(1 - \hat{y})$. If the model correctly predicts $\hat{y} \to 0$, the loss approaches $0$. If it incorrectly predicts $\hat{y} \to 1$, the loss approaches $\infty$.

---

### Architectural Characteristics

* **Output Interpretability:** Unlike algorithms that only output hard predictions, Logistic Regression provides a well-calibrated conditional probability score, making it highly valuable in risk assessment (e.g., credit scoring).
* **Linear Decision Boundary:** Even though the activation function is non-linear, the decision boundary is defined by the plane where $z = 0$. This means the geometric boundary separating classes remains a completely **linear hyperplane**.
* **Multiclass Extensions:** To scale beyond binary classification, two primary strategies are used:
  * **One-vs-Rest (OvR):** Trains $K$ independent binary classifiers (one per class) and selects the class with the highest probability.
  * **Multinomial Logistic Regression (Softmax):** Replaces the Sigmoid function with a **Softmax function** to estimate a unified probability distribution across all $K$ classes simultaneously.
  
---

## 3. Support Vector Machine (SVM)

Support Vector Machine (SVM) is a powerful supervised learning algorithm used primarily for classification. The fundamental objective of an SVM is to find an **optimal separating hyperplane** in an $N$-dimensional space that maximizes the geometric distance (margin) between distinct data classes.



### Core Architectural Concepts

| Structural Component | Geometrical & Mathematical Meaning | Operational Role |
| :--- | :--- | :--- |
| **Hyperplane** | A subspace of dimension $N-1$ defined by the linear equation:<br>$$w^T x + b = 0$$ | Serves as the sharp decision boundary separating the target classes. |
| **Support Vectors** | The data instances that lie exactly on the marginal boundaries satisfying:<br>$$\vert w^T x_i + b \vert = 1$$ | **Critical:** These are the only data points that determine the orientation and position of the hyperplane. Moving any other data points has zero effect on the model. |
| **The Margin** | The perpendicular distance between the central separating hyperplane and the closest support vectors. | Measures the model's confidence. Maximizing this distance minimizes generalization error. |

---

### Mathematical Optimization Objective

SVM converts classification into a constrained quadratic programming optimization problem.

#### 1. Hard Margin SVM (Strict Separation)
Used when the data is perfectly linearly separable. It allows absolutely zero misclassifications inside the training set:
$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \ge 1 \quad \forall i$$
* *Note:* Maximizing the geometric margin $\frac{2}{\|w\|}$ is mathematically equivalent to minimizing $\frac{1}{2}\|w\|^2$.

#### 2. Soft Margin SVM (Slack Variables & Parameter $C$)
Real-world data is rarely perfectly separable. Soft Margin SVM introduces slack variables ($\xi_i \ge 0$) to allow controlled margin violations or misclassifications:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{subject to} \quad y_i(w^T x_i + b) \ge 1 - \xi_i$$

* **The Regularization Parameter $C$:**
  * **Large $C$:** Heavy penalty for misclassification. The model focuses on correctly classifying every single point, leading to a narrower margin (**Risk of Overfitting / High Variance**).
  * **Small $C$:** High tolerance for misclassification. The model prioritizes a wider, safer margin over absolute accuracy on the training set (**Risk of Underfitting / High Bias**).

---

### The Kernel Trick

When a dataset cannot be separated by a straight line or flat hyperplane in its current configuration, the **Kernel Trick** is deployed. Instead of manually transforming data points into a computationally expensive high-dimensional feature space, a **Kernel Function** computes the inner product (similarity score) of vectors *as if* they were already transformed.



#### Standard Mathematical Kernels

| Kernel Type | Mathematical Formula | Hyperparameters & Intuition |
| :--- | :--- | :--- |
| **Linear** | $$K(x, z) = x^T z$$ | No hyperparameter tuning. Best for text classification or high-dimensional sparse datasets. |
| **Polynomial** | $$K(x, z) = (x^T z + c)^d$$ | • $d$: Degree of the polynomial.<br>• $c$: Free parameter trading off higher vs lower order terms. |
| **RBF (Radial Basis Function / Gaussian)** | $$K(x, z) = \exp\left(-\gamma \|x - z\|^2\right)$$ | Maps data into an infinite-dimensional space. **The parameter $\gamma$ (gamma)** controls the radius of influence of individual support vectors:<br>• **High $\gamma$:** Tight, localized decision boundaries around individual points (**Overfitting**).<br>• **Low $\gamma$:** Smooth, sweeping decision boundaries (**Underfitting**). |
| **Sigmoid** | $$K(x, z) = \tanh\left(\alpha x^T z + c\right)$$ | Originating from neural network architectures, behaves similarly to a multi-layer perceptron. |

---

## 4. Decision Trees

A Decision Tree is a non-parametric supervised learning algorithm that decomposes a dataset into smaller, increasingly homogeneous subsets using a hierarchical, tree-like structure of binary decisions. 


### Core Architectural Structure & How it Works
* **Root Node:** Represents the top-level choice that initiates the first data split based on the most informative feature.
* **Internal Nodes:** Represent subsequent attribute tests or intermediate decision points.
* **Leaf Nodes:** Terminal nodes that contain no further splits and output the final hard discrete class prediction (classification) or sample mean value (regression).

```text
                [ Root Node: Feature A ≤ 2.5? ]
                    /                     \
             (Yes) /                       \ (No)
                  /                         \
    [ Internal Node: Feature B ≤ 1.0? ]   [ Leaf Node: Class 2 ]
          /                 \
   (Yes) /                   \ (No)
        /                     \
[ Leaf Node: Class 0 ]    [ Leaf Node: Class 1 ]
```

---

### Splitting Criteria

Decision Trees construct splits by selecting the specific feature and threshold value that maximizes the homogeneity (purity) of the resulting child nodes.

| Criterion | Mathematical Formula | Algorithmic Context | Operational Intuition |
| :--- | :--- | :--- | :--- |
| **Gini Impurity** | $$I_G(p) = 1 - \sum_{i=1}^{K} p_i^2$$ | Classification <br>(Used natively by **CART**) | Measures the probability of a randomly chosen element being incorrectly labeled if it were randomly classified according to the distribution of labels in the subset. Ranges from $0$ (pure) to $0.5$ (max impurity for binary). |
| **Entropy** | $$H(X) = -\sum_{i=1}^{K} p_i \log_2(p_i)$$ | Classification <br>(Used by **ID3**, **C4.5**) | Derived from Shannon Information Theory; measures the level of disorder or uncertainty within a node. Ranges from $0$ (pure) to $1.0$ (max uncertainty). Used to compute **Information Gain**: <br>$$\text{IG} = H(\text{Parent}) - \sum \frac{N_{\text{Child}}}{N_{\text{Parent}}} H(\text{Child})$$ |
| **Mean Squared Error (MSE)** | $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2$$ | Regression <br>(Continuous targets) | Minimizes the variance of the target values within each resulting partition. Each leaf node predicts the mathematical mean ($\bar{y}$) of the continuous samples trapped within that leaf. |

*(Where $p_i$ is the empirical probability of an instance belonging to class $i$ out of $K$ total classes).*

---

### Regularization Hyperparameters

By default, an unconstrained decision tree will grow until every single leaf node is entirely pure, which results in extreme **Overfitting (High Variance)**. To enforce generalization, regularization boundaries must be locked in:

* `max_depth`: Restricts the maximum vertical levels the tree can split down. Lowering this value curtails overfitting by stopping the tree from finding hyper-specific patterns.
* `min_samples_split`: The minimum number of data samples that must reside inside an internal node for it to be legally allowed to split. If a node holds fewer instances than this value, it is forced to become a terminal leaf.
* `min_samples_leaf`: The minimum number of data samples that a terminal leaf node is required to contain. This smooths predictions out by preventing leaves from isolating individual noise or outlier rows.

---

### Tradeoffs & Operational Dynamics

| Advantages (Pros) | Architectural Limitations (Cons) |
| :--- | :--- |
| **High Interpretability:** Follows a clear, white-box Boolean decision path that humans can visually audit easily. | **Extreme Instability:** Exhibits high variance. Small changes or minor noise in the training data can alter the root split, yielding an entirely different tree structure. |
| **Zero Data Preprocessing:** Requires no feature scaling (Normalization/Standardization) or centering because thresholds are independent across axes. | **Prone to Overfitting:** Easily creates highly complex, jagged non-linear decision boundaries that fail to generalize well without rigid hyperparameters. |
| **Heterogeneous Handling:** Natively accommodates both numerical features and raw categorical elements without multi-dimensional expansion. | **High-Cardinality Bias:** Splitting criteria are mathematically biased toward picking features containing numerous unique categories or splitting intervals, even if they aren't truly informative. |

---

## 5. Bias

**Bias** measures how far off a model's average predictions are from the true underlying ground-truth values. It represents the systematic error introduced by making oversimplified assumptions about the nature of the data.



### Mathematical Intuition

Statistically, bias is the difference between the expected value (average) of our model's predictions and the true value we are trying to predict:

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$$

Where:
* $f(x)$ = The true, underlying target function (ground truth).
* $\hat{f}(x)$ = The model's estimated function trained on a specific dataset.
* $\mathbb{E}[\hat{f}(x)]$ = The average prediction of the model if it were trained repeatedly on different variations of the data.

---

### Structural Characteristics

* **High Bias (Underfitting):** Occurs when an algorithm is too rigid or lacks the parameters needed to capture the true distribution. It completely misses the underlying trend.
  * **The Visual:** Attempting to fit a straight linear regression line ($y = mx + b$) to a dataset that follows a clear, non-linear parabolic curve.
  * **The Metric Signal:** High training error **AND** high validation/testing error. The model fails right out of the gate.
* **Low Bias:** Means the model makes very few restrictive assumptions about the data shape, allowing it to adapt dynamically to complex, non-linear patterns (e.g., K-Nearest Neighbors with a low $k$, or Deep Decision Trees).

```
High Bias ──> Underfitting ──> High Training Error + High Test Error
```

---

#### The Mathematical Decomposition of Error
Bias cannot be analyzed in a vacuum because it is explicitly tied to a model's total generalization risk. The total expected error at a given point $x$ is broken down as:

$$\text{Total Expected Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

*(Where $\sigma^2$ is the irreducible error caused by inherent noise in the data collection process).*

---

## 6. Bias-Variance Tradeoff

The ultimate goal of training a machine learning model is to achieve strong generalization—minimizing the **Total Expected Error** on completely unseen data. This is driven by managing two fundamentally competing forces that shift as model complexity scales.



### 6.1 The Core Components

* **Bias² (Decreases with Complexity):** The error introduced by underestimating the true underlying functions of the data. Simple models make sweeping, rigid assumptions. They exhibit high bias because they cannot capture non-linear relationships, leading directly to **underfitting**.
* **Variance (Increases with Complexity):** The error introduced by overestimating the significance of random fluctuations in the training dataset. Highly flexible models adapt aggressively to the specific data points they see. They exhibit high variance because they radically alter their parameter weights based on minor variations in training samples, leading directly to **overfitting**.
* **The Total Error Curve:** Forms a distinctive **U-shape**. Total error is high when complexity is too low (dominated by squared bias) and high when complexity is too high (dominated by variance). The ideal model architecture sits at the inflection point at the absolute lowest valley of this curve.

---

### 6.2 Structural Complexity Matrix

| Dimension / Metric | Low Complexity Model (e.g., Linear) | High Complexity Model (e.g., Deep Tree) |
| :--- | :--- | :--- |
| **Statistical State** | **High Bias**, Low Variance | Low Bias, **High Variance** |
| **Fitting Paradigm** | Underfits the data | Overfits the data |
| **Training Error** | High | Extremely Low (approaching zero) |
| **Test Error** | High | High (wide generalization gap) |
| **Sensitivity to Noise** | Immune to minor data fluctuations | Extremely sensitive to noise and outliers |

---

### 6.3 The Mathematical Sweet Spot

The total expected prediction error at any given validation point $x$ can be decomposed into three mathematically distinct terms:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

Where $\sigma^2$ is the **Irreducible Noise** (inherent randomness in the data collection process itself, such as measurement limits, which no algorithm can ever mathematically eliminate). The objective of hyperparameter tuning is to find the exact model capacity where the sum of $\text{Bias}^2 + \text{Variance}$ hits its global minimum.

#### 6.3.1 Clean ASCII Visual Intuition

```
Error
  |          \      /
  |           \    /    ← Total Error
  |    Bias²   \  /
  |             \/
  |              \  /← Variance
  |_______________\/______
            Model Complexity
              (Sweet Spot)
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

* **The Core Mechanism:**
  * **Second-Order Optimization:** While standard GBM uses only the first derivative (gradient), XGBoost uses a Taylor expansion up to the **second derivative** (Hessian matrix) of the loss function, allowing for faster and more precise convergence.
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