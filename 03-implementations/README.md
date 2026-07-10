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

## 9. Evaluation Metrics

### Classification — `iris_log_linsvm_dec.py`

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **Accuracy** | Correct / Total | Overall % right |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many were actually positive |
| **Recall** | TP / (TP + FN) | Of actual positives, how many did we catch |
| **F1 Score** | 2 * P*R / (P+R) | Balance between precision and recall |
| **Classification Report** | Per-class breakdown | Full picture per class |

All classification metrics use `average='weighted'` — accounts for class imbalance.

### Regression — `housing_linsvr_linreg.py`

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **MSE** | Σ(y - ŷ)² / n | Average squared error (penalizes large errors heavily) |
| **RMSE** | √MSE | Same units as target — more interpretable than MSE |
| **MAE** | Σ\|y - ŷ\| / n | Average absolute error (robust to outliers) |
| **R²** | 1 - SS_res/SS_tot | 1.0 = perfect, 0 = same as predicting the mean |

---

## 9. Results Summary

### Classification (Iris)

| Model | Train Acc | Val Acc | Test Acc | Test F1 |
|-------|-----------|---------|----------|---------|
| Logistic Regression | — | — | — | — |
| SVM (Linear) | — | — | — | — |
| Decision Tree | — | — | — | — |

### Regression (California Housing)

| Model | Train R² | Val R² | Test R² | Test RMSE |
|-------|----------|--------|---------|-----------|
| Linear Regression | — | — | — | — |
| Linear SVR | — | — | — | — |

---

## 10. What to Observe in the Plots

### `plots/iris_all_models.png`
- All three models side by side on the same petal feature space
- **Logistic Regression:** Smooth straight-line boundary
- **SVM (Linear):** Also straight but positioned to maximize margin between classes
- **Decision Tree:** Stepped/rectangular boundaries — always parallel to axes because splits are binary threshold decisions on one feature at a time
- Red X markers = test points → check if they fall in the correct colored region

### `plots/housing_lr_svr.png`
- Both models plotted against the same test data
- **Linear Regression:** Single best-fit line minimizing squared error across all points
- **Linear SVR:** Similar line but ignores points inside the ε-tube — only support vectors (points outside the tube) influence the fit
- The shaded green band on the SVR plot is the ε-tube — any prediction within this band incurs zero loss

---

## 11. Extra Depth

### 11.1 sklearn Pipelines
Instead of manually scaling then fitting, chain steps together:
```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)   # scaling and fitting happen together, safely
pipe.predict(X_test)          # test data is scaled using train parameters automatically
```
Pipelines prevent data leakage — there's no way to accidentally fit the scaler on test data.

### 11.2 Hyperparameter Tuning with GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'model__C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)   # → {'model__C': 1}
```

### 11.3 Feature Importance from Decision Tree
```python
importances = clf.feature_importances_
# For Iris petal features:
# petal length → usually ~0.45
# petal width  → usually ~0.55
# Higher = more useful for splitting
```

### 11.4 What the ε-tube in SVR means
SVR does not try to minimize error for every single training point. Instead, it defines a tube of width ε around the regression line and says: *"I don't care about errors smaller than ε — only penalize points outside the tube."*

This makes SVR more robust to small noise in the data compared to Linear Regression which penalizes every deviation no matter how small.