# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
import pandas as pd

# Loading the Diabetes dataset
data = load_diabetes()
X = data.data  # Features
y = data.target  # Target variable (disease progression metric)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing the Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating the Mean Squared Error (MSE) and R-squared (R^2) score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Displaying the model coefficients
coefficients = pd.DataFrame(model.coef_, index=data.feature_names, columns=['Coefficient'])
print("\nCoefficients:")
print(coefficients)
