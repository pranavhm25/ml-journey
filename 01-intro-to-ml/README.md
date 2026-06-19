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

### 1.1 The Three Core Paradigms of Machine Learning

Machine Learning algorithms are broadly categorized into three distinct paradigms based on the nature of the learning signal and the feedback loop available to the model during training:



| Machine Learning Type | Mathematical Objective | Underlying Core Mechanism | Industry Use-Cases & Applications |
| :--- | :--- | :--- | :--- |
| **Supervised Learning** | Learn a mapping function $f(x)$ to predict target $Y$ given feature matrix $X$: <br>$$\hat{Y} = f(X)$$ | The model optimizes its parameters by minimizing a loss function (e.g., Mean Squared Error or Cross-Entropy) that measures the discrepancy between predicted labels and actual ground-truth labels. | • **Classification:** Spam detection, medical diagnosis, sentiment analysis. <br>• **Regression:** House price prediction, stock forecasting, demand estimation. |
| **Unsupervised Learning** | Model the underlying probability distribution or structure of the unlabeled input space $X$: <br>$$P(X)$$ | The algorithm discovers intrinsic geometric structures, clusters, or patterns within the feature space entirely without human-annotated targets or external feedback. | • **Clustering:** Customer segmentation, anomaly/fraud detection. <br>• **Dimensionality Reduction:** Principal Component Analysis (PCA) for data visualization and compression. |
| **Reinforcement Learning** | Maximize a cumulative scalar reward signal over sequential time steps: <br>$$\max \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_t \right]$$ | An autonomous **Agent** interacts with a dynamic **Environment**. It perceives the current **State** ($S$), executes an **Action** ($A$), transitions to a new state, and receives a positive or negative **Reward** ($R$) to optimize its policy. | • **Robotics:** Autonomous trajectory planning and control. <br>• **Gaming:** Human-level game-playing AI (e.g., AlphaGo, chess engines). <br>• **Systems:** Resource allocation and cloud server load balancing. |

---

## 2. How Much Statistics & Probability is Involved?

A lot — statistics is the mathematical backbone of ML.

### Key Statistical Concepts in Machine Learning

Machine Learning is fundamentally built on top of statistical foundations. Below is a deep dive into the mathematical mechanisms of these core concepts and exactly how they impact model development:

| Statistical Concept | Core Mathematical Definition / Intuition | Explicit Machine Learning Application |
| :--- | :--- | :--- |
| **Mean, Variance, & Std Dev** | • Mean ($\mu$): Central tendency.<br>• Variance ($\sigma^2$): Average squared deviation from the mean.<br>• Standard Deviation ($\sigma$): Standard spread in native units. | Crucial for **Feature Scaling** (e.g., $Z = \frac{X - \mu}{\sigma}$ in Standardization). Algorithms relying on distance metrics (KNN, SVM, K-Means) or Gradient Descent fail if feature scales vary drastically. |
| **Probability Distributions** | Continuous or discrete mathematical functions that describe the likelihood of observing systematic values of a random variable. | • **Gaussian (Normal):** Assumed by Linear Regression errors.<br>• **Multinomial/Bernoulli:** Foundation of text-based Naive Bayes.<br>• **Mixture Models (GMM):** Soft clustering via probability density estimation. |
| **Bayes' Theorem** | Updates the conditional probability of an event based on prior knowledge of related conditions:<br>$$P(A \vert B) = \frac{P(B \vert A) P(A)}{P(B)}$$ | • **Naive Bayes Classifiers:** Computes the posterior class probability given input features.<br>• **Bayesian Optimization:** Automates hyperparameter tuning by managing an uncertainty-based surrogate model. |
| **Hypothesis Testing** | A formal statistical framework to determine if there is sufficient evidence in a sample data pool to reject a baseline null hypothesis ($H_0$). | • **Feature Selection:** Using ANOVA or Chi-Square tests to drop non-informative features.<br>• **A/B Testing:** Validating whether a new model variant significantly alters product conversion rates. |
| **Correlation & Covariance** | • Covariance: Direction of the linear relationship between two variables.<br>• Correlation (Pearson's $r$): Strength and direction bounded between $[-1, 1]$. | • **Collinearity Detection:** Identifying highly redundant features to avoid unstable regression coefficients.<br>• **PCA (Principal Component Analysis):** Diagonalizing the Covariance Matrix to extract orthogonal eigenvectors (components). |
| **Maximum Likelihood Estimation (MLE)** | An optimization framework that finds the parameter values ($\theta$) that maximize the likelihood function, making the observed data most probable:<br>$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} L(\theta \vert X)$$ | Used directly to derive optimization loss targets. Minimizing **Cross-Entropy Loss** in Logistic Regression and Neural Networks is mathematically equivalent to maximizing the likelihood under a Bernoulli distribution. |
| **Expected Value** | The long-run average value of repetitions of a random variable experiment:<br>$$\mathbb{E}[X] = \sum x_i p_i \quad \text{or} \quad \int x f(x)dx$$ | • Used to formulate foundational objectives like **Expected Risk Minimization (ERM)**.<br>• Serves as the math behind evaluating loss functions and defining Reinforcement Learning reward equations. |
| **Law of Large Numbers (LLN)** | As a sample size grows ($n \to \infty$), its sample mean ($\bar{X}_n$) converges almost surely to the true expected population mean ($\mu$). | Explains mathematically **why more training data helps**. Larger datasets guarantee that empirical training errors mirror true real-world distribution errors, stabilizing model parameters. |
| **Central Limit Theorem (CLT)** | Given a sufficiently large sample size ($n \ge 30$), the sampling distribution of the sample mean will approximate a normal distribution, regardless of the population's underlying distribution shape. | Validates why **Gaussian distribution assumptions** work so frequently in practice. It allows us to safely model aggregate parameter updates, residual errors, and system noise using standard parametric methods. |

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

Linear Regression models the relationship between input features $X$ and a continuous scalar output $Y$ by fitting a linear equation to observed data. Geometrically, this represents a **straight line** in two dimensions, or a hyperplane when expanding to multiple dimensions.



### The Formula

#### Simple Linear Regression (1 feature):
$$\hat{y} = mx + b$$ 

Where:
* $\hat{y}$ = Predicted target value.
* $m$ = Slope of the line (weight / coefficient).
* $x$ = Input feature.
* $b$ = Intercept (bias).

#### Multiple Linear Regression ($n$ features):
$$\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$$

Where $\beta_0$ is the intercept and $\beta_1 \dots \beta_n$ are the feature weights.

---

### Correlation Coefficient ($r$)

Before calculating the parameters of the regression line, the strength and direction of the linear relationship between the independent variable $X$ and dependent variable $Y$ is measured using the **Pearson Correlation Coefficient ($r$)**. 

#### Mathematical Formula:
$$r = \frac{\text{Cov}(X,Y)}{\sigma_x \times \sigma_y}$$

Where:
* $\text{Cov}(X,Y)$ = Covariance between $X$ and $Y$, indicating how they change together.
* $\sigma_x$ = Standard deviation of $X$.
* $\sigma_y$ = Standard deviation of $Y$.

#### Interpretation Matrix for $r$:
| Value of $r$ | Statistical Interpretation | Visual Trend |
| :--- | :--- | :--- |
| **$+1.0$** | Perfect positive correlation | As $X$ increases, $Y$ increases linearly in lockstep. |
| **$0.0$** | No linear correlation | The data points show no discernable linear pattern. |
| **$-1.0$** | Perfect negative correlation | As $X$ increases, $Y$ decreases linearly in lockstep. |

---

### Regression Line Parameters

The parameters for simple linear regression can be computed directly using the correlation coefficient and the data's descriptive statistics.

#### 1. Slope ($m$)
The slope scale factor is determined by the ratio of the variability in $Y$ to the variability in $X$, scaled by their correlation coefficient $r$:
$$m = r \times \left(\frac{\sigma_y}{\sigma_x}\right)$$

#### 2. Intercept ($b$)
The line is forced to pass directly through the center of mass of the data—the point $(\bar{X}, \bar{Y})$. The intercept anchors this line along the Y-axis:
$$b = \bar{Y} - m\bar{X}$$
*(Where $\bar{Y}$ is the sample mean of $Y$, and $\bar{X}$ is the sample mean of $X$).*

---

### Inference & Prediction

Once the structural parameters $m$ and $b$ are locked in, predictions can be evaluated for any arbitrary unseen input value ($X_{\text{new}}$):

$$Y_{\text{pred}} = m \times X_{\text{new}} + b$$

The calculated value $Y_{\text{pred}}$ represents the conditional expected value $\mathbb{E}[Y \vert X = X_{\text{new}}]$ along the constructed regression line.

---

## 6. Basic Python Libraries for ML

| Library | Purpose | Key Use |
|---------|---------|---------|
| **NumPy** | Numerical computing | Arrays, math operations |
| **Pandas** | Data manipulation & analysis | DataFrames, CSV handling |
| **Matplotlib** | Plotting & visualization | Line charts, scatter plots |
| **Seaborn** | Statistical visualization | Heatmaps, distribution plots |
| **Scikit-learn** | ML algorithms | Train/test, models, metrics |
| **SciPy** | Scientific computing | Stats, optimization |
| **Jupyter** | Interactive notebooks | Exploration and prototyping |

>*These libraries are for reference/exploration. The next step implements regression from scratch using only pure Python and math.*

---

## 7. Linear Regression Using Correlation Coefficient

See: [`linear_regression.py`](./linear_regression.py)

### What it does:
- Accepts `n` rows of input dynamically (user types each row)
- Stores X and Y values in separate lists.
- Calculates:
  - Mean of X and Y
  - Standard deviation of X and Y
  - Correlation coefficient (`r`)
- Computes slope `m` and intercept `b`
- Displays the regression equation.
- Predicts Y for a user-provided X value.
- Uses only loops and variables — no libraries

### Formula Flow Used in the Program

```text
Input Data
     ↓
Calculate Means
     ↓
Calculate Standard Deviations
     ↓
Calculate Correlation Coefficient (r)
     ↓
Calculate Slope (m)
     ↓
Calculate Intercept (b)
     ↓
Generate Regression Equation
     ↓
Predict New Value
```

### How to run:
```bash
python linear_regression.py
```

### Sample Run

```text
Enter number of data entries: 4

Data Entry 1
Enter x value: 1
Enter y value: 2

Data Entry 2
Enter x value: 2
Enter y value: 4

Data Entry 3
Enter x value: 3
Enter y value: 5

Data Entry 4
Enter x value: 4
Enter y value: 4

Correlation coefficient (r) = 0.6324

Regression Line: Y = mX + b
Slope (m): 0.7
Intercept (b): 1.5

Enter x value for prediction: 5

Predicted y value: 5.0
```

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
