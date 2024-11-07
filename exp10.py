# Importing necessary libraries
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic data
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=42)

# Plotting the data with default color
plt.scatter(X[:, 0], X[:, 1], s=30, color='b', label='Data Points')
plt.title("Generated Data")
plt.legend()
plt.show()

# Initializing and fitting the Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, max_iter=100, random_state=42)
gmm.fit(X)

# Predicting cluster labels for the data
y_pred = gmm.predict(X)

# Extracting the parameters
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

# Displaying the results
print("Estimated Means:\n", means)
print("\nEstimated Covariances:\n", covariances)
print("\nEstimated Weights:\n", weights)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=30, cmap='viridis', label='Clustered Data')
plt.scatter(means[:, 0], means[:, 1], c='red', s=100, marker='x', label='Centroids')
plt.title("Clusters Identified by EM Algorithm")
plt.legend()
plt.show()
