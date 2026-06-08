# 🤖 ML Journey — From Scratch to Deployment

> A structured, self-documented learning path through Machine Learning, Neural Networks, and MLOps.  
> Topics guided by a senior mentor, extended with additional depth and context.

---

## 📚 Table of Contents

| # | Topic | Status |
|---|-------|--------|
| 1 | [Intro to ML](./01-intro-to-ml/README.md) | ✅ Complete |
| 2 | [Basic ML Models](./02-ml-models/README.md) | ✅ Complete |
| 3 | [Implementations with Libraries](./03-implementations/README.md) | ✅ Complete |
| 4 | [Model Saving, Kernels & Unsupervised Learning](./04-model-saving-unsupervised/README.md) | ✅ Complete |
| 5 | [Neural Networks & Git Workflow](./05-neural-networks/README.md) | ✅ Complete |
| 6 | [Testing & Docker Deployment](./06-deployment/README.md) | ✅ Complete |
| — | [Resources & Cheatsheets](./resources/README.md) | ✅ Complete |

---

## 🗂️ Repo Structure

```
ml-journey/
├── README.md
├── 01-intro-to-ml/
│   ├── README.md               # Theory: What is ML, stats, linear regression
│   ├── linear_regression_scratch.py
│   └── sample_data.csv
├── 02-ml-models/
│   ├── README.md               # Classification, SVM, Trees, Bias-Variance, Boosting
│   └── concepts/
│       └── bias_variance.md
├── 03-implementations/
│   ├── README.md               # All 4 models, datasets, graphs, metrics
│   ├── linear_regression_sklearn.py
│   ├── logistic_regression_iris.py
│   ├── svm_iris.py
│   ├── decision_tree_iris.py
│   └── plots/                  # Decision boundary images go here
├── 04-model-saving-unsupervised/
│   ├── README.md               # Saving formats, kernels, unsupervised models, distributions
│   ├── save_load_demo.py
│   └── clustering_demo.py
├── 05-neural-networks/
│   ├── README.md               # NN types, softmax, CNN vs KNN, git workflow
│   ├── softmax_iris.py
│   ├── cnn_mnist.py
│   └── knn_mnist.py
├── 06-deployment/
│   ├── README.md               # Model testing, Docker theory + practice
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── app/
│       └── server.py           # Flask/FastAPI model server
└── resources/
    ├── README.md
    ├── cheatsheet_metrics.md
    ├── cheatsheet_algorithms.md
    └── datasets.md
```

---

## 🚀 How to Use This Repo

1. Follow topics **in order** — each builds on the previous
2. Each folder has its own `README.md` with theory + code explanations
3. Run `.py` files locally after installing requirements
4. Check `resources/` for quick-reference cheatsheets

```bash
# Install all dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torch jupyter
```

---
## Future Development: Transformers, RAG, fine-tuning

## 🧑‍💻 Author

**H M Pranav** — Information Science & Engineering, M S Ramaiah Institute of Technology

---

*"The best way to learn ML is to implement it, break it, and fix it."*
