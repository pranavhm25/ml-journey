# linear_regression_scratch.py
# ─────────────────────────────────────────────────────────────────────────────
# Formula used:
#   m = [n*Σ(xi*yi) - Σxi * Σyi] / [n*Σ(xi²) - (Σxi)²]
#   b = (Σyi - m*Σxi) / n
#   ŷ = m*x + b
# ─────────────────────────────────────────────────────────────────────────────

def get_inputs():       # Take dynamic n rows of (x, y) input from user.
    n = int(input("Enter number of data points: "))
    x_vals = []
    y_vals = []
    for i in range(1, n + 1):
        x = float(input(f"  Enter x[{i}]: "))
        y = float(input(f"  Enter y[{i}]: "))
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals

def compute_slope_intercept(x_vals, y_vals):        # Apply the OLS formulas to compute slope (m) and intercept (b).
    n = len(x_vals)

    # Accumulators
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0

    for i in range(n):
        sum_x  += x_vals[i]
        sum_y  += y_vals[i]
        sum_xy += x_vals[i] * y_vals[i]
        sum_x2 += x_vals[i] ** 2

    # OLS formula
    numerator_m   = n * sum_xy - sum_x * sum_y
    denominator_m = n * sum_x2 - sum_x ** 2

    if denominator_m == 0:
        print("ERROR: All x values are the same — cannot compute slope.")
        return None, None

    m = numerator_m / denominator_m
    b = (sum_y - m * sum_x) / n

    return m, b

def predict(m, b, x_new):
    return m * x_new + b        # Apply the linear equation ŷ = m*x + b.

def compute_mse(x_vals, y_vals, m, b):      # Compute Mean Squared Error on training data.
    n = len(x_vals)
    total_error = 0.0
    for i in range(n):
        y_pred = predict(m, b, x_vals[i])
        total_error += (y_vals[i] - y_pred) ** 2
    return total_error / n


def compute_r_squared(x_vals, y_vals, m, b):        # Compute R² (coefficient of determination).
    n = len(x_vals)
    mean_y = sum(y_vals) / n
    ss_tot = sum((y - mean_y) ** 2 for y in y_vals)
    ss_res = sum((y_vals[i] - predict(m, b, x_vals[i])) ** 2 for i in range(n))
    if ss_tot == 0:
        return 1.0
    return 1 - (ss_res / ss_tot)

def main():
    print("=" * 20)
    print("  Linear Regression")
    print("=" * 20)
    x_vals, y_vals = get_inputs()
    m, b = compute_slope_intercept(x_vals, y_vals)
    if m is None:
        return
    print(f"\nModel Parameters:")
    print(f"   Slope     (m) = {m:.4f}")
    print(f"   Intercept (b) = {b:.4f}")
    print(f"   Equation: ŷ = {m:.4f}x + ({b:.4f})")

    # Evaluation on training data
    mse = compute_mse(x_vals, y_vals, m, b)
    r2  = compute_r_squared(x_vals, y_vals, m, b)
    print(f"\nTraining Metrics:")
    print(f"   MSE  = {mse:.4f}")
    print(f"   RMSE = {mse**0.5:.4f}")
    print(f"   R²   = {r2:.4f}")

    # Prediction loop
    print("\nPrediction Mode (type 'q' to quit)")
    while True:
        raw = input("   Enter x to predict: ")
        if raw.strip().lower() == 'q':
            break
        try:
            x_new = float(raw)
            y_hat = predict(m, b, x_new)
            print(f"   Predicted y = {y_hat:.4f}")
        except ValueError:
            print("   Invalid input. Enter a number or 'q' to quit.")

    print("\nExited...")

if __name__ == "__main__":
    main()