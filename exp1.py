from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset from sklearn
iris = load_iris()
# Convert it to a DataFrame for easier manipulation
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target to the DataFrame and rename it as 'target'
data['target'] = iris.target

# Convert the target to a binary classification (e.g., "Yes" for class 1, "No" for other classes)
data['target'] = data['target'].apply(lambda x: 'Yes' if x == 1 else 'No')

# Display the prepared data
print("Sample of the dataset:")
print(data.head())

# FIND-S algorithm functions
def most_specific_hypothesis(num_attributes):
    return ['∅'] * num_attributes

def update_hypothesis(hypothesis, example):
    for i, value in enumerate(example):
        if hypothesis[i] == '∅':
            hypothesis[i] = value
        elif hypothesis[i] != value:
            hypothesis[i] = '?'
    return hypothesis

def find_s_algorithm(data):
    hypothesis = most_specific_hypothesis(len(data.columns) - 1)
    for _, row in data.iterrows():
        if row['target'] == 'Yes':
            hypothesis = update_hypothesis(hypothesis, row[:-1])
    return hypothesis

# Run FIND-S on the prepared Iris dataset
hypothesis = find_s_algorithm(data)
print("Most specific hypothesis that fits the positive examples:", hypothesis)
