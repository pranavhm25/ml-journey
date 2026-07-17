# 📘 Topic 4 — Model Saving, Kernels & Unsupervised Learning

---

## 1. How to Save a Model

After training, you need to **persist** the model so you can reuse it without retraining.

### Option 1: Pickle (Python standard library)
```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```
✅ Works for all sklearn models  
⚠️ Security risk: don't unpickle untrusted files

### Option 2: Joblib (recommended for sklearn)
```python
from joblib import dump, load

dump(model, 'model.joblib')
loaded = load('model.joblib')
```
✅ Faster and more efficient for large numpy arrays  
✅ Standard for sklearn models

### Option 3: ONNX (Open Neural Network Exchange)
- Universal format — export once, run anywhere
- Works across Python, C++, Java, mobile
```python
from skl2onnx import convert_sklearn
model_onnx = convert_sklearn(model, ...)
```

---

## 2. Different Formats for Different Model Types

| Model Type | Best Format | Extension |
|------------|-------------|-----------|
| **sklearn models** | Joblib | `.joblib` |
| **TensorFlow/Keras** | SavedModel or HDF5 | `.pb` / `.h5` |
| **PyTorch** | TorchScript or state dict | `.pt` / `.pth` |
| **XGBoost** | XGBoost native | `.json` / `.ubj` |
| **LightGBM** | LightGBM native | `.txt` |
| **Cross-platform** | ONNX | `.onnx` |

### TensorFlow save/load:
```python
model.save('my_model')              # SavedModel format (folder)
model.save('my_model.h5')           # HDF5 format
loaded = tf.keras.models.load_model('my_model')
```

### PyTorch save/load:
```python
# Save only weights (recommended)
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

## 3. Kernels — The Kernel Trick

### What is a Kernel?

A kernel function `K(x, z)` computes the **dot product in a higher-dimensional space** without explicitly computing the transformation.

```
φ(x) = transformation to higher dimension
K(x, z) = φ(x) · φ(z)    ← computed directly, no explicit φ needed
```

> **Why this matters:** Data that isn't linearly separable in 2D might be separable in 10,000D. The kernel trick lets SVM work in that space without the computational cost.

### Types of Kernels:

#### Linear Kernel
```
K(x, z) = x · z
```
- No transformation — works in original space
- Best for linearly separable data
- Fastest, simplest

#### Polynomial Kernel
```
K(x, z) = (γ·x·z + r)^d
```
- Creates polynomial decision boundaries
- Hyperparameters: degree `d`, coefficient `r`, scale `γ`

#### RBF (Radial Basis Function) / Gaussian Kernel
```
K(x, z) = exp(-γ · ||x - z||²)
```
- Most popular, works for most datasets
- `γ` controls how far influence of a single training point reaches
  - High γ → tightly fit, complex boundary → overfitting
  - Low γ → smooth boundary → underfitting

#### Sigmoid Kernel
```
K(x, z) = tanh(α·x·z + c)
```
- Similar to neural network activation
- Less commonly used

#### Custom Kernels
You can define any valid kernel (positive semi-definite) for domain-specific problems (e.g., string kernels for text, graph kernels for molecules).

### Kernel Choice Guide:
| Data type | Recommended kernel |
|-----------|-------------------|
| Linearly separable | Linear |
| General non-linear | RBF |
| Text/NLP | Linear or RBF |
| Low-feature polynomial patterns | Polynomial |

---

## 4. Unsupervised Learning

No labels. The model finds structure on its own.

### 4.1 Types of Unsupervised Models

| Model | Type | How it works |
|-------|------|--------------|
| **K-Means** | Clustering | Assigns points to K centroids iteratively |
| **DBSCAN** | Clustering | Groups dense regions; marks outliers as noise |
| **Hierarchical Clustering** | Clustering | Builds a dendrogram of merges/splits |
| **GMM (Gaussian Mixture Model)** | Probabilistic clustering | Soft cluster assignment via Gaussian components |
| **PCA** | Dimensionality Reduction | Projects to directions of max variance |
| **t-SNE** | Dimensionality Reduction | Non-linear 2D/3D visualization of high-dim data |
| **UMAP** | Dimensionality Reduction | Faster t-SNE alternative, preserves global structure |
| **Autoencoders** | Representation learning | Neural network that learns compressed encoding |
| **Isolation Forest** | Anomaly detection | Isolates outliers via random splitting |

---

## 5. Clustering in Detail

### K-Means
```
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Recompute centroids as mean of assigned points
4. Repeat until convergence
```

**Choosing K:**
- **Elbow Method:** Plot inertia (within-cluster SSE) vs K → pick elbow point
- **Silhouette Score:** Measures how similar a point is to its own cluster vs others

### DBSCAN
```
For each point p:
  If p has ≥ minPts neighbors within ε:
    Mark as core point, expand cluster
  Else if p is reachable from a core point:
    Mark as border point
  Else:
    Mark as noise/outlier
```

### Hierarchical Clustering
- **Agglomerative (bottom-up):** Start with each point as its own cluster; merge closest
- **Divisive (top-down):** Start with one cluster; split recursively
- Result is a **dendrogram** — cut at any level to get K clusters

---

## 6. Statistical Distributions

### 6.1 Gaussian (Normal) Distribution
```
f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```
- Parameters: mean μ, standard deviation σ
- Bell curve; symmetric
- **Datasets that satisfy this:** Heights, test scores, measurement errors
- **Models to use:** Linear Regression, LDA, GMM

### 6.2 Poisson Distribution
```
P(X=k) = (λ^k * e^(-λ)) / k!
```
- Models count of events in fixed time/space
- Parameter: λ (average rate)
- **Datasets:** Number of emails per hour, arrivals at a queue, typos per page
- **Models:** Poisson Regression

### 6.3 Gamma Distribution
```
f(x) = (x^(α-1) * e^(-x/β)) / (β^α * Γ(α))
```
- Models positive continuous values; time until k events
- Parameters: shape α, scale β
- **Datasets:** Waiting times, insurance claims
- **Models:** Gamma Regression, survival analysis

### 6.4 Bernoulli Distribution
- Binary outcome: success (1) with probability p
- **Datasets:** Single coin flip, email is spam
- **Models:** Logistic Regression

### 6.5 Binomial Distribution
- Count of successes in n Bernoulli trials
- **Datasets:** Number of heads in 10 flips
- **Models:** Logistic Regression (n=1), Binomial Regression

### 6.6 Exponential Distribution
- Time between events in a Poisson process
- **Datasets:** Time between customer arrivals
- **Models:** Survival analysis, Cox Regression

### Distribution → Model Mapping:
| Distribution | ML/Stat Model |
|-------------|--------------|
| Gaussian | Linear Regression, LDA, GMM |
| Bernoulli | Logistic Regression |
| Multinomial | Softmax Regression |
| Poisson | Poisson Regression |
| Gamma | Gamma Regression |
| Any mixture | Gaussian Mixture Model |

---

## Files

| File | Description |
|------|-------------|
| `save_load_demo.py` | Save/load sklearn and keras models |
| `clustering_demo.py` | K-Means, DBSCAN, Hierarchical on toy data |

---

## 🔗 Resources
- [sklearn: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [StatQuest: K-Means](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [Understanding the Kernel Trick](https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f)
