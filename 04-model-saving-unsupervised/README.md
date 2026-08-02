# 📘 Topic 4 — Model Saving, Kernels & Unsupervised Learning

---

## 1. Model Persistence (Serialization & Deserialization)

After executing computationally expensive training pipelines, you must **persist** your model. This involves converting the internal state of the trained object (such as optimized weights, hyperparameter dictionaries, and node splits) into a byte stream that can be stored on disk and reloaded later without retraining.

---

### Comparison Matrix: Persistence Frameworks

| Persistence Approach | Under-the-Hood Mechanism | Computational Efficiency | Ecosystem / Cross-Platform Mobility | Critical Engineering Risks |
| :--- | :--- | :--- | :--- | :--- |
| **Pickle** | Evaluates and reconstructs arbitrary Python object structures dynamically. | **Moderate:** Can struggle and consume massive memory footprints when handling large datasets or vast internal numpy matrices. | Bounded strictly to the **Python ecosystem**. Requires identical library and Python patch versions between environments. | 🚨 **Arbitrary Code Execution:** De-serializing an untrusted file can run malicious shell commands instantly. |
| **Joblib** | Optimized wrapper over Pickle designed specifically to isolate and write large numeric byte arrays quickly. | **High:** Fast disk I/O when dealing with models containing heavy arrays (e.g., Random Forests with millions of nodes or linear coefficients). | Bounded strictly to the **Python ecosystem** and standard Pydata workflows. | 🚨 Shares the same underlying **security vulnerabilities** as native Pickle if loading unvalidated binary payloads. |
| **ONNX** <br>*(Open Neural Network Exchange)* | Compiles the model's forward inference graph into a highly optimized, standardized protocol buffer specification. | **Maximum Inference Speed:** Bypasses the Python runtime entirely during execution, utilizing localized hardware accelerators. | **Cross-Platform Universal:** Can deploy a model built in Python directly into production runtimes written in **C++, C#, Java, Go, JavaScript, or iOS/Android**. | ⚠️ **One-Way Graph Export:** The converted graph is locked exclusively to *inference (prediction only)*. It can never be unpacked or re-trained. |

---

### Code Execution Blocks

#### 1.1 Native Pickle Implementation (Python Standard Library)
```python
import pickle

# Serialization (Writing the byte stream to a binary file)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Deserialization (Reconstructing the model back into active memory)
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

#### 1.2 Joblib Implementation (Highly Recommended for Scikit-Learn)
Joblib writes out huge numpy arrays into distinct matrix memory chunks, dramatically outperforming standard serialization libraries on large models.

```python
import joblib

# Serialization
joblib.dump(model, 'model.joblib', compress=3)  # Optional Zlib compression tier (0-9)

# Deserialization
loaded_model = joblib.load('model.joblib')
```

#### 1.3 ONNX Serialization Implementation
ONNX completely separates inference from the design environment by storing the execution logic in a standard graph representation.

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define the explicit input tensor shape and type for the input layer
# Example: [None, 4] means a variable batch size with exactly 4 input features
initial_type = [('float_input', FloatTensorType([None, 4]))]

# Compile the scikit-learn model structure into an ONNX graph representation
model_onnx = convert_sklearn(model, initial_types=initial_type)

# Write the serialized graph structure explicitly to disk
with open("model.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())
```

### Critical Production Risks: The Dependency Trap
Even if you mitigate security vulnerabilities by verifying your asset files, both Pickle and Joblib are highly fragile. They do not store the underlying code logic—they only store references to class pointers and raw numeric state variables.

If you train a model in an environment running `scikit-learn==1.4` and attempt to deserialize it in an environment running `scikit-learn==1.6`, the execution will often crash with unhandled `AttributeError` or `ModuleNotFoundError` exceptions due to internal class structure refactoring. Production architectures must enforce strict parity between training and inference environments using container tools like Docker or migrate completely to universal graph standards like *ONNX*.

---

## 2. Framework-Specific Model Storage Standards

Every major machine learning framework has built-in production persistence formats optimized for its unique computational graph structure and array handling.

---

### Comparative Storage Architecture

| Framework Engine | Production Standard Format | Native Extension | Architectural Blueprint & Implementation Details |
| :--- | :--- | :--- | :--- |
| **Scikit-Learn** | Joblib Serialization | `.joblib` | Stores numeric data structures directly. Ideal for memory-mapped arrays and multi-threaded CPU architectures. |
| **TensorFlow / Keras** | **SavedModel** (Default) or HDF5 | Directory / `.h5` | SavedModel creates a comprehensive asset directory containing the compiled computational runtime graph (`saved_model.pb`) and explicit parameter weight checkpoints. |
| **PyTorch** | **State Dictionary** or TorchScript | `.pth` / `.pt` | A Python dictionary mapping every tensor layer to its exact optimized weight matrix array. Compiled variants use TorchScript for production execution. |
| **XGBoost** | Native JSON or Universal Binary JSON | `.json` / `.ubj` | Bypasses general serialization protocols entirely. Encodes tree paths, node gains, and boosting targets cleanly into raw language-agnostic tables. |
| **LightGBM** | Native Text File Spec | `.txt` | Formats leaves and split thresholds into an optimized, human-readable text syntax that initializes in milliseconds. |
| **Universal Ecosystem**| ONNX Representation | `.onnx` | Compiles deep networks and classical algorithms alike into serialized mathematical operators for deployment outside of Python. |

---

### Code Implementation Blocks

#### 2.1 TensorFlow / Keras Pipeline Storage
TensorFlow provides two primary persistence formats. The native **SavedModel** directory structure is highly recommended over the older monolithic HDF5 file format for production deployment because it natively encapsulates asset vocabularies and graph optimization signatures.

```python
import tensorflow as tf

# 1. SavedModel Production Directory Deployment (Recommended)
# Generates a folder structure tracking graph operations, variables, and signatures
model.save('production_model_directory')
loaded_dir_model = tf.keras.models.load_model('production_model_directory')

# 2. HDF5 Monolithic Storage Paradigm
# Packs architecture definition and weight matrices into a single container file
model.save('model_checkpoint.h5')
loaded_h5_model = tf.keras.models.load_model('model_checkpoint.h5')

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
