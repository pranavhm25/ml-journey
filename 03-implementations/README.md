# 📘 Topic 3 — Implementations with scikit-learn

---

## 1. Overview

This module covers foundational supervised machine learning models implemented via `scikit-learn`. Models sharing identical data domains are consolidated into individual executable scripts to allow direct, un-biased comparisons of decision boundaries and loss performance.

| Model | Dataset | Task | File |
|-------|---------|------|------|
| Logistic Regression | Iris (petal features) | Multiclass Classification | `iris_log_linsvm_dec.py` |
| SVM (Linear kernel) | Iris (petal features) | Multiclass Classification | `iris_log_linsvm_dec.py` |
| Decision Tree | Iris (petal features) | Multiclass Classification | `iris_log_linsvm_dec.py` |
| Linear Regression | California Housing (median income) | Regression | `housing_linsvr_linreg.py` |
| Linear SVR | California Housing (median income) | Regression | `housing_linsvr_linreg.py` |

---

## 2. Dataset

### 🌸 The Iris Flower Dataset

* **API Ingestion:** `from sklearn import datasets`<br>`iris = datasets.load_iris()`
* **Dataset Scale:** $150$ total samples ($50$ instances distributed evenly per class).
* **Feature Subsampling:** Sliced to extract **Petal Length** and **Petal Width** exclusively (columns index $2$ and $3$). 
  * *Engineering Rationale:* Petal metrics exhibit significantly higher class separability and lower intra-class variance compared to sepal metrics. Using them ensures the multi-class decision boundaries are visually distinct and clean when plotted in a 2D feature space.
* **Target Vector ($Y$):** Three distinct botanical classes: *Setosa* ($0$), *Versicolor* ($1$), and *Virginica* ($2$).
* **Task Objective:** Non-linearly separable multiclass classification.

### 🏠 The California Housing Dataset

* **API Ingestion:** `from sklearn.datasets import fetch_california_housing`
* **Feature Subsampling:** Isolated down to **Median Income** as the single independent predictor variable.
  * *Engineering Rationale:* Restricting the independent input matrix to a single dimension ($X \in \mathbb{R}^1$) keeps the feature space strictly 2D ($X$ vs $Y$). This allows clear geometric plotting of the regression lines against actual data points.
* **Target Vector ($Y$):** Median house value for California districts (quantified in units of $\$100,000$).
* **Task Objective:** Continuous target estimation (Regression).

---

## 3. Mathematical Intuition for Regression Models

To understand the core behavioral differences between the two regression models executed in `housing_linsvr_linreg.py`, consider their competing error-minimization strategies:

### 3.1 Linear Regression (Ordinary Least Squares)
Minimizes the sum of all squared residual errors. Every single outlier point exerts a pull on the regression line proportional to the square of its distance from the line:

$$\min_{w, b} \frac{1}{n} \sum_{i=1}^{n} (y_i - (w x_i + b))^2$$

### 3.2 Linear Support Vector Regression (SVR)
Unlike OLS, SVR introduces an **$\epsilon$-insensitive tube** (epsilon tube) around the regression line. Any residual error smaller than $\epsilon$ incurs **zero penalty**. The model ignores points close to the trend line and focuses its optimization strictly on points falling outside this margin:

$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max(0, \vert y_i - (w x_i + b) \vert - \epsilon)$$

* **$\epsilon$ (Epsilon):** Dictates the width of the tube. A larger tube means fewer points violate the margin, leading to a sparser selection of support vectors.
* **$C$:** Controls the penalty weight assigned to points that fall outside the epsilon boundary.

---

## 4. What is a Decision Boundary?

A geometric **Decision Boundary** is a surface or partition in an $N$-dimensional feature space that splits different target classes. The trained machine learning model utilizes this boundary as its classification rule: any input instance falling on one side is assigned to a specific discrete class, while an instance falling on the opposite side is assigned to another.

#### Visual Intuition (2D Feature Space)

```text
Petal Width
   │      . . .│x x x
   │   . . .   │x x x
   │  . . .   /  x x
   │         /        ← Decision Boundary (Hyperplane)
   │________/________________ Petal Length
```

### 4.1 Geometric Properties Across Algorithms

The mathematical nature of the learning algorithm dictates the structural shape and flexibility of the resulting boundary:

* Linear Classifiers (Logistic Regression, Linear SVM): Compute flat boundaries. In a 2D plot, this is a completely straight line; in 3D, it is a flat plane; and in $N$-dimensions, it is an $(N-1)$-dimensional hyperplane defined by the linear combination $w^T x + b = 0$.
* Decision Tree Classifiers: Generate highly characteristic stepped or rectangular (orthogonal) boundaries. Because decision trees evaluate individual features one at a time via primitive threshold inequalities (e.g., $x_j \le \text{threshold}$), every single partition line is structurally forced to be axis-aligned (parallel to the feature axes).

### 4.2 Visualization and Code Metrics

* Dimensional Constraints: To properly visualize these boundaries, data inputs are explicitly restricted to two features at a time (e.g., Petal Length vs. Petal Width). It is geometrically impossible to render a multi-dimensional hyperplane spanning a 4D feature space directly on a 2D screen.
* Evaluation Diagnostics: In the consolidated evaluation script `iris_log_linsvm_dec.py`, data partitions are visually explicitly decoupled on the generated scatter plots:
  * Circles ($\bullet$): Represent the Training Set points—the coordinates the model actively used to adjust parameters and orient the decision boundgit pary.
  * X Markers ($\times$ with red outlines): Represent the completely held-out Test Set points. This allows you to immediately audit whether the model generalizes well or if the decision boundary has warped around noise near the evaluation zones.

---

## 5. Files

| File | Models Inside | Output Plot |
|------|---------------|-------------|
| `iris_log_linsvm_dec.py` | Logistic Regression, SVM (Linear), Decision Tree | `plots/iris_all_models.png` |
| `housing_linsvr_linreg.py` | Linear Regression, Linear SVR | `plots/housing_lr_svr.png` |

---

## 6. How to Run

```bash
# Install dependencies (one-time)
pip install scikit-learn matplotlib numpy

# Run Iris models (produces plots/iris_all_models.png)
python iris_log_linsvm_dec.py

# Run Housing models (produces plots/housing_lr_svr.png)
python housing_linsvr_linreg.py
```

> Plots are saved to the `plots/` folder automatically. You do **not** need a display or GUI — `matplotlib.use('Agg')` handles this.

---

## 7. The Training Process

The execution logic within the scikit-learn scripts follows strict machine learning best practices to ensure unbiased evaluation and prevent data leakage.

### 7.1 Pipeline Breakdown: `iris_log_linsvm_dec.py`

This script executes a multiclass classification pipeline across three distinct models to contrast their geometric limits:

1. **Ingestion & Feature Slicing:** Loads the Iris dataset and subsets the design matrix to features $2$ and $3$ (Petal Length and Petal Width).
2. **Double-Fold Stratified Split:** Splits the data chronologically into an $80/10/10$ ratio:
   * **Step A:** Split into $80\%$ Training and $20\%$ Temporary validation/testing data.
   * **Step B:** Split the temporary pool exactly in half to yield $10\%$ Validation and $10\%$ Testing sets.
   * *Mechanism:* The splitting utilizes `stratify=y` to preserve strict class proportions across all three subsets.
3. **Isolated Preprocessing:** Fits a standard $Z$-score normalizer (`StandardScaler`) on the Training partition only. The fitted transformation parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are then downstream applied to the Validation and Test sets to prevent data leakage.
4. **Execution Loop:** Iterates sequentially through Logistic Regression, Linear SVM, and the Decision Tree Classifier:
   * Invokes `.fit()` on the training split.
   * Computes multi-class evaluation metrics across all partitions.
   * Generates a structural meshgrid to map and overlay the decision boundaries.
5. **Reporting:** Prints a comprehensive classification report (Precision, Recall, F1-Score) and exports a unified sub-plot graphic.

### 7.2 Pipeline Breakdown: `housing_linsvr_linreg.py`

This script executes a continuous estimation workflow contrasting a residual-squared baseline with a marginal-insensitive loss model:

1. **Ingestion & Dimensional Restriction:** Ingests the California Housing data, trapping the design matrix to the Median Income column.
2. **Partitioning:** Segmented into an $80\%$ Train, $10\%$ Validation, and $10\%$ Test configuration using an identical double-split routine.
3. **Dual-Target Preprocessing:** Fits a `StandardScaler` to the feature vector $X$ (training only). Crucially, a separate instance of `StandardScaler` is fitted to the **target vector $y$** (training only). 
4. **Parameter Inversion:** Predictions generated in the scaled vector space ($\hat{y}_{\text{scaled}}$) are passed back through the inverse scale transform (`scaler_y.inverse_transform()`) to restore native dollar values before any evaluation metrics are computed.
5. **Model Fitting:** Simultaneously trains Ordinary Least Squares (OLS) Linear Regression and Linear Support Vector Regression (SVR).
6. **Geometric Visualization:** Generates a side-by-side scatter plot overlaying the OLS regression line against the SVR regression line. The SVR plot explicitly highlights the **$\epsilon$-insensitive tube** (a shaded margin band representing the dead-zone boundary where prediction errors are assigned a loss penalty of exactly $0$).

---

## 8. Preprocessing Notes

### 8.1 The Rationale Behind Scaling Target Vector $y$ in SVR
Unlike Ordinary Least Squares (OLS) regression, which adapts scale-invariantly due to its unconstrained closed-form derivative equations, Support Vector Regression (SVR) is highly sensitive to the magnitude of the target variable $y$. 

SVR measures error relative to its optimization hyperparameters: the insensitive tube width $\epsilon$ and the constraint penalty $C$. If house values are expressed in raw native digits (e.g., $\$350,000$), a hardcoded tube width like $\epsilon = 0.1$ becomes infinitesimally narrow, rendering the margin mechanic useless. Scaling the target vector $y$ to zero mean and unit variance ensures that $\epsilon$ and $C$ act predictably as relative proportions of standard deviations, regardless of the target's underlying unit scale. 

* *Note:* OLS Linear Regression does not mathematically require a scaled target vector $y$, but it is subjected to the same pipeline here to enforce a completely fair benchmark comparison.

### 8.2 The Necessity of Stratified Sampling (`stratify=y`)
In discrete classification tasks, relying on basic random distribution splits introduces high statistical risk, particularly on small or skewed datasets. A completely random assignment could result in an irregular distribution where one target class is overrepresented in the training pool and entirely absent from the testing subset. 

By passing the label array to the splitting routine (`stratify=y`), the algorithm calculates the base class distribution ratio of the global dataset (e.g., a perfect $1:1:1$ split for Iris) and forces every resulting partition to duplicate that exact relative proportion. This ensures that the model is evaluated on data that accurately reflects the training environment.

### 8.3 Feature Choice: Petals vs. Sepals
The morphological dimensions of an iris flower differ significantly in their descriptive power:

* **Sepal Subspace:** High geometric overlap. The data boundaries are deeply intertwined across classes, which forces models to either underfit or learn highly complex, noisy boundaries.
* **Petal Subspace:** High spatial separation. The three flower classes naturally isolate into distinct geometric zones. This clear separation makes it easy to observe the operational behavior of each model, visually highlighting how a linear model constructs straight hyperplanes while a decision tree builds axis-aligned, stepped bounding boxes.

---

## 9. Comprehensive Evaluation Metrics

The scripts evaluate model performance using distinct statistical metrics tailored to discrete classification and continuous regression domains.

### 9.1 Classification Metrics (`iris_log_linsvm_dec.py`)

Evaluating multiclass models requires analyzing the balance between precision and recall across all label variants.

| Classification Metric | Mathematical Formula | Analytical Objective & Intuition |
| :--- | :--- | :--- |
| **Accuracy** | $$\frac{TP + TN}{TP + TN + FP + FN}$$ | Measures the overall percentage of correctly predicted instances across the entire dataset. Ideal primarily for completely balanced datasets. |
| **Precision** <br>(Positive Predictive Value) | $$\frac{TP}{TP + FP}$$ | Measures quality: out of all samples predicted as positive, what fraction were truly positive? Minimizing False Positives avoids false alarms. |
| **Recall** <br>(Sensitivity / Hit Rate) | $$\frac{TP}{TP + FN}$$ | Measures quantity: out of all actual positive samples in the data, what fraction did the model successfully capture? Minimizing False Negatives avoids missed targets. |
| **F1-Score** | $$2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$ | The harmonic mean of Precision and Recall. Provides a unified metric that balance the two forces, especially useful on imbalanced sets. |
| **Classification Report** | *Per-class compilation* | A complete scikit-learn diagnostic printout displaying precision, recall, and F1-score for each discrete target label individually. |

* **Multiclass Aggregation Mechanic (`average='weighted'`):** To scale binary metrics to the 3-class Iris problem, scikit-learn calculates metrics for each class independently and computes their average, weighting each class score by its **support** (the number of true instances belonging to that specific class). This prevents minor classes from skewing the final evaluation.

---

### 9.2 Regression Metrics (`housing_linsvr_linreg.py`)

Continuous metrics track the geometric distance between prediction coordinates ($\hat{y}$) and true observations ($y$).

| Regression Metric | Mathematical Formula | Error Tracking Behavior |
| :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | $$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ | Computes the average squared residual distance. Because errors are squared, large outliers are penalized far more aggressively than minor deviations. |
| **Root Mean Squared Error (RMSE)** | $$\sqrt{\text{MSE}}$$ | Takes the square root of MSE to map the error scale directly back to the native units of the target variable (e.g., actual dollar value), making it highly interpretable. |
| **Mean Absolute Error (MAE)** | $$\frac{1}{n} \sum_{i=1}^{n} \vert y_i - \hat{y}_i \vert$$ | Computes the simple linear average of absolute errors. Unlike MSE, it treats all deviations linearly, making it structurally **robust to outliers**. |
| **Coefficient of Determination ($R^2$)** | $$1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$ | Measures the proportion of variance in the target variable that is predictable from the input features. <br>• $1.0$ = Perfect fit.<br>• $0.0$ = Baseline model that simply predicts the dataset mean ($\bar{y}$).<br>• Negative values = Model performs worse than predicting the mean. |

---

## 10. Quantitative Results Summary

### 10.1 Classification Performance Benchmark (Iris Dataset)

| Trained Model Algorithm | Training Accuracy | Validation Accuracy | Testing Accuracy | Test Weighted F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | *0.9667* | *0.9333* | *0.9333* | *0.9327* |
| **Support Vector Machine (Linear)** | *0.9667* | *0.9333* | *0.9333* | *0.9327* |
| **Decision Tree Classifier** | *0.9833* | *0.9333* | *1.0000* | *1.0000* |

### 10.2 Regression Performance Benchmark (California Housing)

| Trained Model Algorithm | Training Coefficient ($R^2$) | Validation Coefficient ($R^2$) | Testing Coefficient ($R^2$) | Test RMSE (Native Scale) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | *0.4770* | *0.4610* | *0.4564* | *0.8417* |
| **Linear Support Vector Regressor (SVR)**| *0.4570* | *0.4427* | *0.4299* | *0.8620* |

---

## 11. What to Observe in the Plots

### 11.1 `plots/iris_all_models.png`
- All three models side by side on the same petal feature space
- **Logistic Regression:** Smooth straight-line boundary
- **SVM (Linear):** Also straight but positioned to maximize margin between classes
- **Decision Tree:** Stepped/rectangular boundaries — always parallel to axes because splits are binary threshold decisions on one feature at a time
- Red X markers = test points → check if they fall in the correct colored region

### 11.2 `plots/housing_lr_svr.png`
- Both models plotted against the same test data
- **Linear Regression:** Single best-fit line minimizing squared error across all points
- **Linear SVR:** Similar line but ignores points inside the ε-tube — only support vectors (points outside the tube) influence the fit
- The shaded green band on the SVR plot is the ε-tube — any prediction within this band incurs zero loss

---

## 12. Extra Depth

### 12.1 Scikit-Learn Pipelines (`sklearn.pipeline.Pipeline`)

In standard machine learning workflows, manual preprocessing is prone to operational slip-ups. Scikit-Learn pipelines fix this by chaining sequential data transformation steps and a final estimator into a single, cohesive object.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Constructing the pipeline sequence
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Training: Fits the scaler on X_train, transforms X_train, then fits the model
pipe.fit(X_train, y_train)   

# Inference: Automatically transforms X_test using the *train* parameters, then predicts
predictions = pipe.predict(X_test)     
```

#### The Architecture of Data Leakage Prevention

Manual transformation often leads to accidentally applying .fit_transform() on the global dataset or the test partition. This leaks validation/test parameters (like the mean $\mu$ or standard deviation $\sigma$) into the training process, leading to artificial validation scores.

Pipelines enforce an operational firewall: when calling .fit(), the pipeline calls .fit_transform() strictly on the training folds. When calling .predict(), it calls .transform() using the locked-in training distribution parameters, guaranteeing that the model remains completely blind to test set parameters.

### 12.2 Hyperparameter Optimization via `GridSearchCV`

Finding the absolute best hyperparameters (like the optimal regularization strength $C$) manually is tedious. `GridSearchCV` automates this search by evaluating every parameter permutation across an exhaustive grid using cross-validation.

```python
from sklearn.model_selection import GridSearchCV

# Note the syntax: 'stepName__parameterName' to target nested pipeline elements
param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2']
}

# Configures an exhaustive search evaluated via 5-Fold Cross-Validation
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Extracting the optimal configuration
print(f"Optimal Hyperparameters: {grid.best_params_}") 
print(f"Top Validation Accuracy: {grid.best_score_:.4f}")
```

### 12.3 Quantifying Feature Importance in Decision Trees

Unlike black-box models, Decision Trees explicitly calculate Feature Importance (often termed Gini Importance or Mean Decrease in Impurity). This value measures the total drop in impurity (Gini or Entropy) brought by a specific feature across all splits in the tree.

```python
# Extracting structural importance weights
importances = clf.feature_importances_

# Typical empirical breakdown for Iris Petal Subspace:
# Feature Index 2 (Petal Length): ~0.45
# Feature Index 3 (Petal Width):  ~0.55
```

#### Interpretation Mechanics

* Normalization: The feature importance array is scaled to sum up to exactly $1.0$.
* Implication: A higher value signifies that a feature splits data nodes into pure clusters more frequently and closer to the root of the tree, making it highly influential for predictions.

### 12.4 Mathematical Intuition of the $\epsilon$-Insensitive Tube in SVR

Unlike Ordinary Least Squares (OLS) Linear Regression, which applies a squared loss penalty to every single deviation no matter how miniscule, Support Vector Regression (SVR) utilizes an $\epsilon$-insensitive loss function.

The model builds an envelope (a tube) of radius $\epsilon$ (epsilon) around the structural regression line. The mathematical loss penalty is defined as:

$$\mathcal{L}_{\epsilon}(y, \hat{y}) = \max(0, \vert y - \hat{y} \vert - \epsilon)$$

#### Why This Works Better in Noisy Environments

* Zero Penalty Zone: Any training point that falls inside the boundaries of the tube ($\vert y - \hat{y} \vert \le \epsilon$) incurs a loss penalty of exactly $0$. The model completely ignores minor noise close to the prediction trend.
* Support Vectors: The optimization algorithm completely ignores the points sitting inside the tube and focuses its mathematical attention entirely on instances lying outside or on the boundaries. These outliers become the Support Vectors that anchor the regression path.
* The Result: SVR becomes highly robust to minor systemic noise and variance fluctuations, whereas OLS regression paths get easily warped by outliers.