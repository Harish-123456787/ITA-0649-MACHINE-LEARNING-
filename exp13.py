import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create a synthetic dataset with similar structure
np.random.seed(42)  # For reproducibility

# Generate random data for each feature
data = {
    'normalized-losses': np.random.randint(65, 250, 100),
    'engine-size': np.random.randint(70, 200, 100),
    'horsepower': np.random.randint(48, 288, 100),
    'curb-weight': np.random.randint(1500, 4000, 100),
    'highway-mpg': np.random.randint(15, 45, 100),
    'price': np.random.randint(5000, 45000, 100)  # Target variable
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Select features and target variable
X = df[['normalized-losses', 'engine-size', 'horsepower', 'curb-weight', 'highway-mpg']]
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
