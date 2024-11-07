# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic data for demonstration
# Let's use a quadratic relationship with some noise
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)
X = X[:, np.newaxis]  # Reshaping for scikit-learn compatibility

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# 2. Polynomial Regression (Degree 2 for demonstration)
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Train the model on polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)

# Evaluation metrics
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f'Linear Regression Mean Squared Error: {mse_linear:.2f}')
print(f'Linear Regression R-squared: {r2_linear:.2f}')
print(f'Polynomial Regression (Degree 2) Mean Squared Error: {mse_poly:.2f}')
print(f'Polynomial Regression (Degree 2) R-squared: {r2_poly:.2f}')

# Plotting the results
plt.scatter(X, y, color='lightblue', label='Data Points')
plt.plot(X_test, y_pred_linear, color='red', label='Linear Regression')
plt.scatter(X_test, y_pred_poly, color='green', s=10, label='Polynomial Regression (Degree 2)')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear vs Polynomial Regression")
plt.show()
