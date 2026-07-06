# 📘 Topic 3 — Implementations with scikit-learn

---

## Overview

| Model | Dataset | Task | Graph |
|-------|---------|------|-------|
| Linear Regression | House Price Prediction | Regression | Actual vs Predicted, Residual plot |
| Logistic Regression | Iris | Multiclass Classification | Decision boundary |
| SVM (RBF + Linear) | Iris + House Price | Both | Decision boundary |
| Decision Tree | Iris | Classification | Decision boundary + Tree visualization |

---

## Datasets

### 🌸 Iris Dataset
- **Source:** Built into sklearn (`sklearn.datasets.load_iris()`)
- **Samples:** 150 (50 per class)
- **Features:** Sepal length, Sepal width, Petal length, Petal width
- **Classes:** Setosa, Versicolor, Virginica
- **Task:** Multiclass classification

### 🏠 House Price Dataset
- **Source:** [Kaggle - House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
  Or use sklearn's built-in California Housing: `sklearn.datasets.fetch_california_housing()`
- **Task:** Regression (predict sale price)

---

## What is a Decision Boundary?

A decision boundary is the line (or surface) that separates different classes in the feature space. Points on one side are predicted as one class; points on the other side as another.

```
Feature 2
   |    oooo  xxxx
   |   ooo  /  xxx
   |  oo   /   xx
   |      /         ← Decision Boundary
   |_________________ Feature 1
```

- **Linear models** (Logistic Regression, Linear SVM): straight line
- **Non-linear models** (RBF SVM, Decision Tree): curved or stepped boundaries

To visualize, we typically plot only 2 features at a time (since we can't visualize 4D+).

---

## Training Process Overview

### Pre-processing steps:
1. Load and inspect data
2. Handle missing values (if any)
3. Feature scaling (StandardScaler for SVM, LR; not needed for trees)
4. Train/test/validation split

### Training:
5. Instantiate model with hyperparameters
6. Fit on training data

### Post-processing:
7. Predict on test set
8. Compute evaluation metrics
9. Plot decision boundaries and graphs

---

## Evaluation Metrics Used

### For Classification (Iris):
| Metric | Description |
|--------|-------------|
| **Accuracy** | % of correct predictions |
| **Precision** | TP / (TP + FP) per class |
| **Recall** | TP / (TP + FN) per class |
| **F1 Score** | Harmonic mean of precision & recall |
| **Confusion Matrix** | Visual breakdown of TP/FP/TN/FN |
| **Classification Report** | All of the above per class |

### For Regression (House Prices):
| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **R²** | Coefficient of determination (1 = perfect) |

---

## Results Summary

> *(Fill this in after running the scripts)*

| Model | Dataset | Train Acc | Val Acc | Test Acc |
|-------|---------|-----------|---------|----------|
| Logistic Regression | Iris | — | — | — |
| SVM (RBF) | Iris | — | — | — |
| Decision Tree | Iris | — | — | — |
| Linear Regression | House Price | R²=— | R²=— | R²=— |
| Linear SVM | House Price | R²=— | R²=— | R²=— |

---

## Files

| File | Description |
|------|-------------|
| `linear_regression_sklearn.py` | Regression on house price data |
| `logistic_regression_iris.py` | Multiclass classification on Iris |
| `svm_iris.py` | SVM (RBF) on Iris |
| `decision_tree_iris.py` | Decision Tree on Iris |
| `plots/` | Decision boundary and evaluation plots |

---

## How to Run All Models

```bash
# Install dependencies (one-time)
pip install scikit-learn matplotlib seaborn numpy pandas

# Run each model
python linear_regression_sklearn.py
python logistic_regression_iris.py
python svm_iris.py
python decision_tree_iris.py
```

---

## Things Your Senior Missed (Extra Depth)

### Pipelines
Instead of manually scaling → fitting, use sklearn Pipelines:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])
pipe.fit(X_train, y_train)
```
Pipelines prevent **data leakage** (fitting scaler on test data by accident).

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 'auto', 0.1]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
```

### Learning Curves
Plot training vs validation accuracy as a function of training set size — useful for diagnosing overfitting/underfitting.

### Feature Importance (Decision Trees)
```python
importances = clf.feature_importances_
# Plot as bar chart to understand which features matter most
```