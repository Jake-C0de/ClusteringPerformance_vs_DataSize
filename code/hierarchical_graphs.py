# Purpose: generates graphs for Hierarchical Iris and scaling experiments separately.
#
# HOW TO RUN:
#   This file is usually not run directly.
#   It is called automatically when you run:
#
#       py code/hierarchical.py
#       py code/hierarchical_scaling.py
#
# REQUIRED PACKAGES:
#   If packages are missing, run:
#
#       pip install numpy pandas matplotlib scikit-learn
#
# NOTES:
#   - Iris memory graph is NOT generated because Iris uses the same dataset size
#     for every trial, so memory by trial is misleading.
#   - Scaling memory graph IS generated because dataset size changes.
#   - Cluster graphs are generated for both Iris and scaling.

"""
What this script does:
1. loads Iris results and creates graphs (trial-based)
2. loads scaling results and creates graphs (n-based)
3. saves them into separate folders so nothing is overwritten
4. creates cluster graphs for Iris and synthetic scaling data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

from utils import load_iris_data


# -----------------------------------
# Helper function to calculate centroids
# Hierarchical clustering does not give centroids automatically,
# so we calculate the average point in each cluster.
# -----------------------------------
def calculate_centroids(X, labels):

    centroids = []

    for cluster_id in np.unique(labels):

        cluster_points = X[labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    return np.array(centroids)


# -----------------------------------
# Helper function to create graphs
# -----------------------------------
def plot_metric(df, x_col, y_col, title, y_label, output_path):

    plt.figure(figsize=(8, 6))
    plt.plot(df[x_col], df[y_col], marker="o")

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------------
# Helper function to create cluster graphs
# -----------------------------------
def plot_clusters(X, labels, centroids, title, output_path):

    plt.figure(figsize=(8, 6))

    # Plot data points using cluster labels as colors
    plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolor="k", alpha=0.75)

    # Plot calculated centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=250,
        edgecolor="k",
        label="Centroids"
    )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------------
# Iris graph generator
# -----------------------------------
def generate_iris_graphs():

    results_dir = "results"
    graphs_dir = "graphs/hierarchical_iris_graphs"

    os.makedirs(graphs_dir, exist_ok=True)

    # load Iris results
    df = pd.read_csv(os.path.join(results_dir, "hierarchical_iris_results.csv"))

    # create trial-based graphs
    plot_metric(df, "trial", "runtime_seconds", "Runtime", "Seconds", f"{graphs_dir}/runtime.png")

    # Memory graph removed for Iris.
    # Iris uses the same dataset size for every trial, so memory by trial
    # can decrease or fluctuate because of Python memory management.
    # That makes it misleading for the report.

    plot_metric(df, "trial", "sse", "SSE", "Value", f"{graphs_dir}/sse.png")
    plot_metric(df, "trial", "silhouette_score", "Silhouette", "Score", f"{graphs_dir}/silhouette.png")

    # create Iris cluster graph
    X, y = load_iris_data()

    model = AgglomerativeClustering(n_clusters=3)
    labels = model.fit_predict(X)

    centroids = calculate_centroids(X, labels)

    plot_clusters(
        X,
        labels,
        centroids,
        "Hierarchical Clustering on Iris Dataset",
        f"{graphs_dir}/cluster_graph.png"
    )


# -----------------------------------
# Scaling graph generator
# -----------------------------------
def generate_scaling_graphs():

    results_dir = "results"
    graphs_dir = "graphs/hierarchical_scaling_graphs"

    os.makedirs(graphs_dir, exist_ok=True)

    # load scaling summary
    df = pd.read_csv(os.path.join(results_dir, "hierarchical_scaling_summary.csv"))

    df["memory_mb"] = df["memory_mb"].clip(lower=0)
    df = df.sort_values(by="n")

    # create scaling graphs
    plot_metric(df, "n", "runtime_seconds", "Runtime vs n", "Seconds", f"{graphs_dir}/runtime.png")
    plot_metric(df, "n", "memory_mb", "Memory vs n", "MB", f"{graphs_dir}/memory.png")
    plot_metric(df, "n", "sse", "SSE vs n", "Value", f"{graphs_dir}/sse.png")
    plot_metric(df, "n", "silhouette_score", "Silhouette vs n", "Score", f"{graphs_dir}/silhouette.png")

    # create representative scaling cluster graph
    # n = 2000 is used because it is large enough to show clear clusters
    # but not so large that the graph becomes too crowded.
    X, y = make_blobs(
        n_samples=2000,
        centers=3,
        random_state=42
    )

    model = AgglomerativeClustering(n_clusters=3)
    labels = model.fit_predict(X)

    centroids = calculate_centroids(X, labels)

    plot_clusters(
        X,
        labels,
        centroids,
        "Hierarchical Clustering on Synthetic Dataset (n = 2000)",
        f"{graphs_dir}/cluster_graph.png"
    )