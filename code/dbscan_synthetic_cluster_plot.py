# -----------------------------------
# Make DBSCAN cluster plot for synthetic data
# -----------------------------------
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# choose dataset size
n = 5000

# same settings from your scaling experiment
eps = 0.5
min_samples = 5

# generate synthetic data
X, y = make_blobs(
    n_samples=n,
    centers=3,
    n_features=2,
    cluster_std=1.0,
    random_state=42
)

# run DBSCAN
model = DBSCAN(eps=eps, min_samples=min_samples)
labels = model.fit_predict(X)

# build graph path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
graphs_dir = os.path.join(project_root, "graphs", "dbscan_scaling_graphs")
os.makedirs(graphs_dir, exist_ok=True)

# plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title(f"DBSCAN Clusters on Synthetic Data\nn={n}, eps={eps}, min_samples={min_samples}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()

output_path = os.path.join(graphs_dir, f"dbscan_synthetic_clusters_n{n}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved: {output_path}")