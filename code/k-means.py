import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

#start time
start = time.time()

# load dataset
iris = load_iris()
X = iris.data

# create K-means model
kmeans = KMeans(n_clusters=3, random_state=42)

# fit model
kmeans.fit(X)

# get cluster labels
labels = kmeans.labels_

# plot (using first two features)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-means on Iris Dataset")

# end time
end = time.time()
print("Runtime:", end - start, "seconds")

# show and save plot
plt.savefig("results/K-means.png")
plt.show()
