import os
import numpy as np
import matplotlib
matplotlib.use('Agg')                           # Save plots instead of showing (works headless/in repo)
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler          # Needed for LR and SVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

os.makedirs("plots", exist_ok=True)

# ── 1. PRE-PROCESSING ────────────────────────────────────────────────────────
iris = datasets.load_iris()

# Using petal features (2,3) instead of sepal (0,1)
# Petals are far more discriminative → cleaner, more meaningful decision boundaries
X = iris.data[:, 2:]   # petal length, petal width
y = iris.target
class_names  = iris.target_names
feature_names = ['Petal Length', 'Petal Width']

# Split: 80% Train, 10% Val, 10% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Decision Tree doesn't need it, but we scale all for consistency.
# Scaler is fit ONLY on training data to prevent data leakage.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

# ── 2. MODELS INITIALIZATION ─────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (Linear kernel)": SVC(kernel='linear', probability=True),
    "Decision Tree":       DecisionTreeClassifier(max_depth=3)
}

# ── 3. TRAINING & EVALUATION LOOP ────────────────────────────────────────────
def run_analysis():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Iris Dataset — Decision Boundaries (Petal Features)", fontsize=14, fontweight='bold')

    for i, (name, clf) in enumerate(models.items()):

        # ── Fit ──────────────────────────────────────────────────────────────
        clf.fit(X_train_s, y_train)

        # ── Metrics: Train / Val / Test ──────────────────────────────────────
        train_acc = accuracy_score(y_train, clf.predict(X_train_s)) # Train accuracy
        val_acc   = accuracy_score(y_val,   clf.predict(X_val_s))   # Validation accuracy
        test_acc  = accuracy_score(y_test,  clf.predict(X_test_s))  # Test accuracy

        y_test_pred = clf.predict(X_test_s)

        prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test,  y_test_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test,     y_test_pred, average='weighted', zero_division=0)

        # Full metrics table per model
        print(f"{'─'*45}")
        print(f"  {name}")
        print(f"{'─'*45}")
        print(f"  Train Acc : {train_acc:.4f}")
        print(f"  Val Acc   : {val_acc:.4f}")
        print(f"  Test Acc  : {test_acc:.4f}")
        print(f"  Precision : {prec:.4f}  (weighted)")
        print(f"  Recall    : {rec:.4f}  (weighted)")
        print(f"  F1 Score  : {f1:.4f}  (weighted)")
        print(f"\n{classification_report(y_test, y_test_pred, target_names=class_names)}")

        # ── Decision Boundary ─────────────────────────────────────────────────
        x_min, x_max = X_train_s[:, 0].min() - 0.5, X_train_s[:, 0].max() + 0.5
        y_min, y_max = X_train_s[:, 1].min() - 0.5, X_train_s[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                              np.arange(y_min, y_max, 0.02))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
        scatter = axes[i].scatter(
            X_train_s[:, 0], X_train_s[:, 1],
            c=y_train, edgecolors='k', s=30, cmap='RdYlGn'
        )
        # Mark test points with X so you can see where the model is being evaluated
        axes[i].scatter(
            X_test_s[:, 0], X_test_s[:, 1],
            c=y_test, edgecolors='red', s=60, marker='X', cmap='RdYlGn',
            label='Test points'
        )
        axes[i].set_xlabel(feature_names[0])
        axes[i].set_ylabel(feature_names[1])
        axes[i].set_title(f"{name}\nTrain={train_acc:.2f} | Val={val_acc:.2f} | Test={test_acc:.2f}")
        axes[i].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("plots/iris_all_models.png", dpi=150, bbox_inches='tight')
    print("💾 Plot saved → plots/iris_all_models.png")

run_analysis()
