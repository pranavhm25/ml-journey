import math

def get_data():
    n = int(input("Enter number of data entries: "))

    X = []  # Stores x-values
    Y = []  # Stores y-values
    
    for i in range(n):
        print(f"\nData Entry {i+1}")
        x = float(input("Enter x value: "))
        y = float(input("Enter y value: "))
        
        X.append(x)
        Y.append(y)
    
    return X, Y, n


# Function to calculate Pearson's correlation coefficient (r)
def calculate_r(X, Y, n):
    
    sum_x = sum(X)
    sum_y = sum(Y)
    
    mean_x = sum_x / n
    mean_y = sum_y / n
    
    # Variables for variance and covariance calculations
    sum_x_dev_sq = 0   # Sum of squared deviations of X
    sum_y_dev_sq = 0   # Sum of squared deviations of Y
    sum_xy_dev = 0     # Sum of product of deviations
    
    # Calculate deviations from the mean
    for i in range(n):
        x_dev = X[i] - mean_x
        y_dev = Y[i] - mean_y
        
        # Accumulate squared deviations and covariance term
        sum_x_dev_sq += x_dev ** 2
        sum_y_dev_sq += y_dev ** 2
        sum_xy_dev += x_dev * y_dev
    
    # Calculate standard deviations of X and Y
    sig_x = math.sqrt(sum_x_dev_sq / n)
    sig_y = math.sqrt(sum_y_dev_sq / n)
    
    # Pearson correlation coefficient formula
    r = sum_xy_dev / (n * sig_x * sig_y)
    
    return r, mean_x, mean_y, sig_x, sig_y


# Function to calculate regression line parameters
# Equation: Y = mX + b
def calculate_regression(r, sig_x, sig_y, mean_x, mean_y):
    
    # Calculate slope using correlation coefficient
    m = r * (sig_y / sig_x)
    
    # Calculate intercept
    b = mean_y - (m * mean_x)
    
    return m, b


# Function to predict Y for a given X value
def predict(m, b):
    
    # Input x value for prediction
    x_value = float(input("\nEnter x value for prediction: "))
    
    # Apply regression equation
    y_pred = m * x_value + b
    
    print("Predicted y value:", y_pred)


# ---------------- Main Program ---------------- #

# Get input data from user
X, Y, n = get_data()

# Calculate correlation coefficient and statistical measures
r, mean_x, mean_y, sig_x, sig_y = calculate_r(X, Y, n)

print("\nCorrelation coefficient (r) =", r)

# Calculate regression line equation
m, b = calculate_regression(r, sig_x, sig_y, mean_x, mean_y)

print("\nRegression Line: Y = mX + b")
print("Slope (m):", m)
print("Intercept (b):", b)

# Predict Y value for a user-specified X
predict(m, b)