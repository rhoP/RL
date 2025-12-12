import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Example data: list of integers
data = [2, 8, 0, 4, 1, 9, 9, 0]

# Convert to 2D array (hierarchical clustering expects 2D)
# Each integer becomes a 1D point: [[2], [8], [0], [4], [1], [9], [9], [0]]
X = np.array(data).reshape(-1, 1)

# Perform hierarchical clustering
linked = linkage(X, method="ward", metric="euclidean")

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(
    linked,
    orientation="top",
    labels=data,
    distance_sort="descending",
    show_leaf_counts=True,
)
plt.title("Dendrogram for Integer Clustering")
plt.xlabel("Integer Value")
plt.ylabel("Distance")
plt.show()
