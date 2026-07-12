import numpy as np
import matplotlib
matplotlib.use('Agg')                           # Save plots instead of showing (works headless/in repo)
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import os

os.makedirs("plots", exist_ok=True)

# ── 1. PRE-PROCESSING ────────────────────────────────────────────────────────
data = fetch_california_housing()
X = data.data[:, :1]   # Median Income only — keeps plots 2D and readable
y = data.target

# Split: 80% Train, 10% Val, 10% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# CRITICAL STEP FOR SVM: Scaling
# SVM math (kernels) is based on distances. If features aren't scaled, it fails.
# Linear Regression doesn't strictly need scaling, but we do it here for fair comparison.
# Fit scaler ONLY on training data — never on val/test (prevents data leakage).
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

X_val_scaled  = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

print(f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

# ── 2. TRAINING ──────────────────────────────────────────────────────────────

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_scaled)

# Linear SVR
svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train_scaled)

# ── 3. EVALUATION (POST-PROCESSING) ─────────────────────────────────────────
def evaluate(model_name, model, X_s, y_orig):
    """Predict, inverse-transform back to original price scale, then compute metrics."""
    preds_scaled = model.predict(X_s)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    mse  = mean_squared_error(y_orig, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_orig, preds)      # Mean Absolute Error
    r2   = r2_score(y_orig, preds)

    print(f"  MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return preds

print("─" * 50)
print("  Linear Regression")
print("─" * 50)
print("  [Training]  ", end=""); lr_train_preds = evaluate("LR", lr_model, X_train_scaled, y_train)
print("  [Validation]", end=""); evaluate("LR", lr_model, X_val_scaled, y_val)
print("  [Test]      ", end=""); lr_test_preds  = evaluate("LR", lr_model, X_test_scaled, y_test)

print("\n─" * 50)
print("  Linear SVR")
print("─" * 50)
print("  [Training]  ", end=""); evaluate("SVR", svr_model, X_train_scaled, y_train)
print("  [Validation]", end=""); evaluate("SVR", svr_model, X_val_scaled, y_val)
print("  [Test]      ", end=""); svr_test_preds = evaluate("SVR", svr_model, X_test_scaled, y_test)

# ── 4. GRAPHICAL VIEW ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("California Housing — Linear Regression vs Linear SVR", fontsize=13, fontweight='bold')

sorted_idx = np.argsort(X_test_scaled.flatten())
X_sorted   = X_test_scaled[sorted_idx]

for ax, (model, preds, title, color) in zip(axes, [
    (lr_model,  lr_test_preds,  "Linear Regression", "steelblue"),
    (svr_model, svr_test_preds, "Linear SVR",        "green"),
]):
    ax.scatter(X_test_scaled, y_test, color='blue', alpha=0.2, s=10, label='Actual Data')
    ax.plot(X_sorted,
            scaler_y.inverse_transform(model.predict(X_sorted).reshape(-1, 1)),
            color=color, linewidth=2.5, label=f'{title} fit')

    # ADDED: Epsilon tube for SVR (shows the margin where no penalty is applied)
    if title == "Linear SVR":
        preds_sorted = scaler_y.inverse_transform(
            model.predict(X_sorted).reshape(-1, 1)
        ).flatten()
        eps_orig = scaler_y.scale_[0] * svr_model.epsilon   # convert ε back to original scale
        ax.fill_between(
            X_sorted.flatten(),
            preds_sorted - eps_orig,
            preds_sorted + eps_orig,
            alpha=0.25, color=color, label=f'ε-tube (±{eps_orig:.2f})'
        )

    ax.set_xlabel('Scaled Median Income')
    ax.set_ylabel('House Value ($100k)')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
# savefig instead of show
plt.savefig("plots/housing_lr_svr.png", dpi=150, bbox_inches='tight')
print("\n Plot saved → plots/housing_lr_svr.png")
