# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import pandas as pd

# Loading the dataset (using the Iris dataset for demonstration)
data = load_iris()
X = data.data  # Features
y = data.target  # Target variable

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing the Logistic Regression model
model = LogisticRegression(max_iter=200, multi_class='ovr')  # 'ovr' for one-vs-rest

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Displaying the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=data.target_names, columns=data.target_names)
print("\nConfusion Matrix:")
print(conf_matrix_df)
