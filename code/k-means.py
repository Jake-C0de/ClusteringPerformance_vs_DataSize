# -------------------------
# How to Run: 
# 1) Install matplotpib scikit-learn (pip install matplotlib scikit-learn)
# 2) python k-means.py
# -------------------------

import matplotlib.pyplot as plt
import time
import tracemalloc
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# load dataset
iris = load_iris()
X = iris.data

k_values = range(2, 11)  # silhouette requires k >= 2

runtimes = []
sil_scores = []
memories = []

# start memory tracking
tracemalloc.start()

for k in k_values:
    start = time.time()

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    end = time.time()

    # runtime
    runtimes.append(end - start)

    # silhouette score
    score = silhouette_score(X, labels)
    sil_scores.append(score)

    # memory usage (in MB)
    current, peak = tracemalloc.get_traced_memory()
    memories.append(peak / (1024 * 1024))

# stop memory tracking
tracemalloc.stop()

# -------------------------
# 1. Clustering Graph (k=3)
# -------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-means Clustering (k=3)")
plt.savefig("../graphs/kmeans_iris_graphs/kmeansclustering.png")

# -------------------------
# 2. Runtime Graph
# -------------------------
plt.figure()
plt.plot(k_values, runtimes, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs K")
plt.savefig("../graphs/kmeans_iris_graphs/kmeansruntime.png")

# -------------------------
# 3. Silhouette Score Graph
# -------------------------
plt.figure()
plt.plot(k_values, sil_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.savefig("../graphs/kmeans_iris_graphs/kmeanssilhouette.png")

# -------------------------
# 4. Memory Usage Graph
# -------------------------
plt.figure()
plt.plot(k_values, memories, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage vs K")
plt.savefig("../graphs/kmeans_iris_graphs/kmeansmemory.png")

plt.show()
