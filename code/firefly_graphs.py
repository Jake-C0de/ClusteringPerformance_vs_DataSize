# Purpose: generates Firefly graphs for Iris and scaling experiments
#
# HOW TO RUN:
#   This file is usually not run directly.
#   It is called automatically when you run:
#
#       py code/run_firefly_experiment.py
#       py code/firefly_scaling.py
#
# REQUIRED PACKAGES:
#   If packages are missing, run:
#
#       pip install numpy pandas matplotlib scikit-learn
#
# NOTES:
#   - Iris memory graph is NOT generated because Iris uses the same dataset size
#     for every trial, so memory by trial can look misleading.
#   - Scaling memory graph IS generated because dataset size changes.
#   - A cluster graph is generated for Iris.
#   - A cluster graph is generated for scaling using n = 2000.
#   - Firefly centroids can sometimes appear off-center because Firefly is
#     approximate and random.
#   - For the cluster graphs, centroids are recalculated as the mean of the
#     assigned cluster points so the graph shows the actual center of each group.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from utils import load_iris_data
from firefly import FireflyClustering


# -----------------------------------
# Helper function to create line graphs
# -----------------------------------
def plot(df, x, y, title, ylabel, path):

    plt.figure()
    plt.plot(df[x], df[y], marker="o")

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True)

    plt.savefig(path)
    plt.close()


# -----------------------------------
# Helper function to fix centroids
# -----------------------------------
def fix_centroids(X, labels, old_centroids, n_clusters):

    # Firefly gives approximate centroids.
    # This recalculates centroids as the average position of all points
    # assigned to each cluster.
    #
    # This does NOT change the cluster labels.
    # It only makes the centroid markers appear in the correct center
    # for the graph.

    new_centroids = []

    for cluster_id in range(n_clusters):

        # get points assigned to this cluster
        cluster_points = X[labels == cluster_id]

        # if the cluster has points, use the mean as the centroid
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))

        # if the cluster is empty, keep the original Firefly centroid
        else:
            new_centroids.append(old_centroids[cluster_id])

    return np.array(new_centroids)


# -----------------------------------
# Helper function to create cluster graphs
# -----------------------------------
def plot_clusters(X, labels, centroids, title, path):

    plt.figure()

    # Plot each data point.
    # The color is based on the cluster label assigned by Firefly.
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        edgecolor="k",
        alpha=0.75
    )

    # Plot centroid locations using large X markers.
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=220,
        edgecolor="k",
        label="Centroids"
    )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)

    plt.savefig(path)
    plt.close()


# -----------------------------------
# Iris graph generator
# -----------------------------------
def generate_iris_graphs():

    graphs_dir = "graphs/firefly_iris_graphs"

    os.makedirs(graphs_dir, exist_ok=True)

    # Load Iris experiment results
    df = pd.read_csv("results/firefly_iris_results.csv")

    # Runtime graph
    plot(
        df,
        "trial",
        "runtime_seconds",
        "Runtime",
        "Seconds",
        f"{graphs_dir}/runtime.png"
    )

    # Memory graph is intentionally removed for Iris.
    # Iris uses the same dataset size for every trial, so memory by trial
    # can decrease or fluctuate because of Python memory management.
    # This makes the Iris memory graph misleading.

    # SSE graph
    plot(
        df,
        "trial",
        "sse",
        "SSE",
        "Value",
        f"{graphs_dir}/sse.png"
    )

    # Silhouette graph
    plot(
        df,
        "trial",
        "silhouette_score",
        "Silhouette",
        "Score",
        f"{graphs_dir}/silhouette.png"
    )

    # -----------------------------------
    # Iris cluster graph
    # -----------------------------------

    # Load Iris dataset
    X, y = load_iris_data()

    # Run Firefly on Iris
    model = FireflyClustering()
    labels = model.fit(X)

    # Recalculate centroids so they appear in the center of each cluster
    fixed_centroids = fix_centroids(
        X,
        labels,
        model.centroids_,
        model.n_clusters
    )

    # Save Iris cluster graph
    plot_clusters(
        X,
        labels,
        fixed_centroids,
        "Firefly Clustering on Iris Dataset",
        f"{graphs_dir}/cluster_graph.png"
    )


# -----------------------------------
# Scaling graph generator
# -----------------------------------
def generate_scaling_graphs():

    graphs_dir = "graphs/firefly_scaling_graphs"

    os.makedirs(graphs_dir, exist_ok=True)

    # Load scaling summary results
    df = pd.read_csv("results/firefly_scaling_summary.csv")

    # Sort by n so graphs are in correct order
    df = df.sort_values(by="n")

    # Runtime scaling graph
    plot(
        df,
        "n",
        "runtime_seconds",
        "Runtime vs n",
        "Seconds",
        f"{graphs_dir}/runtime.png"
    )

    # Memory scaling graph
    plot(
        df,
        "n",
        "memory_mb",
        "Memory vs n",
        "MB",
        f"{graphs_dir}/memory.png"
    )

    # SSE scaling graph
    plot(
        df,
        "n",
        "sse",
        "SSE vs n",
        "Value",
        f"{graphs_dir}/sse.png"
    )

    # Silhouette scaling graph
    plot(
        df,
        "n",
        "silhouette_score",
        "Silhouette vs n",
        "Score",
        f"{graphs_dir}/silhouette.png"
    )

    # -----------------------------------
    # Scaling cluster graph
    # -----------------------------------

    # Create one representative synthetic dataset for the cluster graph.
    # n = 2000 is large enough to show clear clusters,
    # but not so large that the graph becomes too crowded.
    X, y = make_blobs(
        n_samples=2000,
        centers=3,
        random_state=42
    )

    # Run Firefly on the synthetic dataset
    model = FireflyClustering()
    labels = model.fit(X)

    # Recalculate centroids so they appear in the center of each cluster.
    # This fixes the issue where raw Firefly centroids can appear far away.
    fixed_centroids = fix_centroids(
        X,
        labels,
        model.centroids_,
        model.n_clusters
    )

    # Save scaling cluster graph
    plot_clusters(
        X,
        labels,
        fixed_centroids,
        "Firefly Clustering on Synthetic Dataset (n = 2000)",
        f"{graphs_dir}/cluster_graph.png"
    )