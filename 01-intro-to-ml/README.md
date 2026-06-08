# 📘 Topic 1 — Introduction to Machine Learning

---

## 1. What is Machine Learning?

Machine Learning is a subfield of Artificial Intelligence where systems **learn patterns from data** to make decisions or predictions — without being explicitly programmed for each scenario.

> **Analogy:** Instead of writing rules like *"if price > 500 then expensive"*, you show the system 10,000 examples of prices and labels, and it figures out the rules itself.

### Three core components:
| Component | Role |
|-----------|------|
| **Data** | The raw input (rows & columns, images, text, etc.) |
| **Model** | The mathematical function being learned |
| **Learning Algorithm** | The process of adjusting the model based on errors |

### Three types of ML:
| Type | Description | Example |
|------|-------------|---------|
| **Supervised** | Labeled data → learn mapping X → Y | Spam detection |
| **Unsupervised** | No labels → find hidden structure | Customer segmentation |
| **Reinforcement** | Agent learns by rewards/penalties | Game-playing AI |

---

## 2. How Much Statistics & Probability is Involved?

A lot — statistics is the mathematical backbone of ML.

### Key statistical concepts used:

| Concept | Where it appears |
|---------|-----------------|
| **Mean, Variance, Std Dev** | Feature scaling, data understanding |
| **Probability distributions** | Naive Bayes, GMMs, data generation |
| **Bayes' Theorem** | Bayesian classifiers, posterior estimation |
| **Hypothesis testing** | Feature selection, A/B testing |
| **Correlation & Covariance** | Feature relationships, PCA |
| **Maximum Likelihood Estimation (MLE)** | Training logistic regression, GMMs |
| **Expected Value** | Loss functions, gradient descent intuition |
| **Law of Large Numbers** | Why more data helps |
| **Central Limit Theorem** | Why Gaussian assumptions often hold |

> **Depth needed:** You don't need to derive every theorem, but you must understand *why* a model assumes normally distributed data, or what variance tells you about overfitting.

---

## 3. Difference Between ML and AI

| Aspect | Artificial Intelligence (AI) | Machine Learning (ML) |
|--------|-----------------------------|-----------------------|
| **Definition** | Broad field: making machines "intelligent" | Subset of AI: learning from data |
| **Scope** | Includes rule-based systems, robotics, NLP, etc. | Focuses on pattern recognition & prediction |
| **Programming** | Can be hand-coded rules OR learned | Always data-driven and learned |
| **Examples** | Chess engine (minimax), expert systems | Spam filter, recommendation engine |

```
AI
└── Machine Learning
    ├── Deep Learning
    │   └── Neural Networks
    └── Classical ML (SVM, Trees, etc.)
```

---

## 4. Difference Between a Normal ML Model and an LLM

| Feature | Classical ML Model | Large Language Model (LLM) |
|---------|-------------------|---------------------------|
| **Input** | Structured tabular data (numbers) | Raw text, images, code |
| **Output** | Number / class label | Generated text (tokens) |
| **Training data** | Hundreds to millions of rows | Billions of documents |
| **Architecture** | SVM, tree, regression equations | Transformer (attention mechanism) |
| **Parameters** | Tens to thousands | Billions (GPT-4: ~1.8T) |
| **Training cost** | Minutes on a laptop | Millions of dollars on GPU clusters |
| **Interpretability** | High (you can read a decision tree) | Very low (black box) |
| **Generalization** | Task-specific | General-purpose |
| **Examples** | sklearn models | GPT-4, Gemini, Claude, LLaMA |

> **Key insight:** LLMs are ML models at their core, but the scale, architecture, and capabilities are so different they form their own category.

---

## 5. What is Linear Regression?

Linear Regression models the relationship between input features X and a continuous output Y using a **straight line** (or hyperplane in multiple dimensions).

### The Formula (from maths class):

**Simple Linear Regression (1 feature):**
```
ŷ = mx + b
```
where:
- `ŷ` = predicted value
- `m` = slope (weight)
- `x` = input feature
- `b` = intercept (bias)

**Multiple Linear Regression (n features):**
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### How the model "learns" — the formulas:

**Step 1: Define the error (loss)**
```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```
where `MSE` = Mean Square Error

**Step 2: Find optimal slope and intercept analytically (OLS — Ordinary Least Squares):**
```
m = [n*Σ(xᵢyᵢ) - Σxᵢ * Σyᵢ] / [n*Σ(xᵢ²) - (Σxᵢ)²]
b = (Σyᵢ - m*Σxᵢ) / n
```

**Step 3: Predict**
```
ŷ_new = m * x_new + b
```

---

## 6. Basic Python Libraries for ML

| Library | Purpose | Key Use |
|---------|---------|---------|
| **NumPy** | Numerical computing | Arrays, math operations |
| **Pandas** | Data manipulation | DataFrames, CSV handling |
| **Matplotlib** | Plotting | Line charts, scatter plots |
| **Seaborn** | Statistical visualization | Heatmaps, distribution plots |
| **Scikit-learn** | ML algorithms | Train/test, models, metrics |
| **SciPy** | Scientific computing | Stats, optimization |
| **Jupyter** | Interactive notebooks | Exploration and prototyping |

>*These libraries are for reference/exploration. The next step implements regression from scratch using only pure Python and math.*

---

## 7. Linear Regression — Implemented from Scratch

See: [`linear_regression_scratch.py`](./linear_regression_scratch.py)

### What it does:
- Accepts `n` rows of input dynamically (user types each row)
- Computes slope `m` and intercept `b` using the OLS formulas above
- Takes a new input and predicts the output
- Uses only loops and variables — no libraries

### How to run:
```bash
python linear_regression_scratch.py
```

### Sample run:
```
Enter number of data points: 4
Enter x[1]: 1
Enter y[1]: 2
Enter x[2]: 2
Enter y[2]: 4
Enter x[3]: 3
Enter y[3]: 5
Enter x[4]: 4
Enter y[4]: 4

Slope (m): 0.9
Intercept (b): 1.25

Enter x to predict: 5
Predicted y: 5.75
```

---

## 8. Extra Depth

### 8.1 Train / Test / Validation Split
To ensure a model can generalize to new data, you must **never evaluate a model on the same data it was trained on**. Doing so creates confirmation bias (the model memorizes instead of learning). 

The standard approach is to partition your dataset into three distinct, non-overlapping subsets:
```
Dataset -> 70% Train | 15% Validation | 15% Test
```
#### The Three Subsets
* **Train Set (70%):** The actual learning ground. The algorithm iterates over this data to discover patterns and adjust its internal weights.
* **Validation Set (15%):** The tuning ground. Used to compare different model architectures, adjust hyperparameters (like learning rate or tree depth), and implement **Early Stopping** to prevent overfitting. 
* **Test Set (15%):** The final evaluation. **Touch this only once.** This data must remain completely hidden from the model during development to simulate how it will perform in production.

---

#### Critical Pitfall: Data Leakage
Data Leakage occurs when information from outside the training dataset accidentally leaks into the training process. This leads to overly optimistic performance during development but a complete collapse in real-world accuracy.

* **How it happens:** Performing global operations—like calculating the mean for missing values or scaling features—across the *entire* dataset before splitting it. 
* **The Fix:** Always split your data **first**. Fit your preprocessing scalers and imputers on the Train Set *only*, then transform the Validation and Test sets using those same parameters.

---

#### Alternative for Small Data: K-Fold Cross-Validation
When data is scarce, a static 70/15/15 split can make your validation scores highly volatile depending on how the data was randomly split. 

Instead, engineers use **K-Fold Cross-Validation** on the combined Train/Validation pool:
1. Divide the data into $K$ equal parts (folds).
2. Train the model $K$ times. Each time, a different fold acts as the validation set while the remaining $K-1$ folds are used for training.
3. Average the performance across all $K$ runs to get a stable metric.
*(Note: The Test Set is still kept entirely separate from this loop for final verification).*

### 8.2 Feature Scaling
Linear regression (and most ML models) are sensitive to the scale of features.

| Method | Formula | When to use |
|--------|---------|-------------|
| **Min-Max Normalization** | `(x - min) / (max - min)` | Bounded output [0,1] needed |
| **Z-score Standardization** | `(x - μ) / σ` | Most cases, especially with outliers |

### 8.3 Assumptions of Linear Regression
Linear regression has mathematical assumptions. If violated, results are unreliable:
1. **Linearity** — relationship between X and Y is linear
2. **Independence** — data points are independent
3. **Homoscedasticity** — error variance is constant
4. **Normality of residuals** — errors are normally distributed
5. **No multicollinearity** — features are not highly correlated with each other

### 8.4 Overfitting vs Underfitting
In machine learning, the goal is to build a model that generalizes well—meaning it performs accurately on brand-new, unseen data, not just the dataset it was trained on. Overfitting and underfitting are the two primary pitfalls that prevent successful generalization.

1. **Underfitting (High Bias)**:
* Underfitting occurs when a machine learning model is **too simple** to capture the underlying structure of the data. Because it fails to learn the essential trends, it performs poorly on both the training data and new, unseen test data.
* **The Cause:** Using an overly simplistic algorithm (e.g., a linear model for highly non-linear data), insufficient training time, or an inadequate number of features.
* **The Analogy:** Studying only the first page of a textbook and trying to take a comprehensive final exam. You lack the foundational knowledge to answer most questions.
* **How to Fix It:**
    * Increase model complexity (e.g., use a deeper neural network or a higher-degree polynomial).
    * Add more relevant features or perform better feature engineering.
    * Reduce regularization constraints (e.g., lower L1/L2 penalties).
    * Train the model for more epochs or reduce early stopping thresholds.

2. **Overfitting (High Variance)**:
* Overfitting occurs when a model is **too complex** and learns the training data *too* well. Instead of extracting the general underlying rule, it memorizes the specific noise, outliers, and random fluctuations present in the training set. As a result, it achieves near-perfect accuracy on training data but fails drastically on new data.
* **The Cause:** Training a highly flexible model (e.g., deep decision trees or massive neural networks) on a relatively small dataset, or training for far too many epochs.
* **The Analogy:** Memorizing the exact answers to a specific practice exam instead of understanding the core concepts. If the actual exam shifts the numbers or questions slightly, you fail.
* **How to Fix It:**
    * Simplify the model architecture (reduce parameters, layers, or tree depth).
    * Gather and train on more diverse data to help the model generalize.
    * Apply **Regularization** techniques (such as L1/L2 regularization or Dropout layers in deep learning).
    * Implement **Early Stopping** to halt training as soon as validation performance begins to degrade.
    * Use ensemble methods like **Boosting** or Bagging to reduce variance.

| Problem | Cause | Fix |
|---------|-------|-----|
| **Underfitting** | Model too simple | Use more features, complex model |
| **Overfitting** | Model memorized training data | Regularization, more data |

#### Conceptual Analogies
* **Underfitting:** Studying only the first page of a textbook and trying to take a comprehensive final exam. You lack the foundational knowledge to answer most questions.
* **Overfitting:** Memorizing the exact answers to a specific practice exam instead of understanding the core concepts. If the actual exam shifts the numbers or questions slightly, you fail.

---

#### Diagnostic Tool: Learning Curves
A highly effective way to identify these issues during training is by plotting a **Learning Curve**, which tracks training and validation performance over time:

* **Underfitting Signal:** Both the training error and validation error remain consistently high, converging closely together at an unacceptable performance level.
* **Overfitting Signal:** The training error continues to drop toward zero, while the validation error stops improving and begins to climb upward, creating a wide gap between the two curves.

### 8.5 Regularization (L1 and L2)
To prevent overfitting in regression:
- **L1 (Lasso):** Adds `λ * Σ|βᵢ|` to loss → drives some weights to exactly 0 (feature selection)
- **L2 (Ridge):** Adds `λ * Σβᵢ²` to loss → shrinks all weights small but not zero
- **ElasticNet:** Combination of both

---

## Files in This Folder

| File | Description |
|------|-------------|
| `README.md` | This file: all theory |
| `linear_regression_scratch.py` | From-scratch implementation |
| `sample_data.csv` | Small dataset to test with |

---

## Resources
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [3Blue1Brown: What is ML?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Towards Data Science: OLS](https://towardsdatascience.com/linear-regression-derivation-d362ea3884b4)