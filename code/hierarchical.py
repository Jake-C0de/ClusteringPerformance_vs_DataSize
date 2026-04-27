# Purpose: runs the Hierarchical Clustering experiment on the Iris dataset (n = 150),
# records results, saves CSV files, and generates Iris-only graphs.
#
# HOW TO RUN:
#   Open terminal in the project folder, then run:
#
#       py code/hierarchical.py
#
# REQUIRED PACKAGES:
#   If packages are missing, run:
#
#       pip install numpy pandas matplotlib scikit-learn
#
# NOTE:
#   Memory is NOT recorded for Iris because the dataset size does not change.
#   Memory by trial can look misleading, so Iris focuses on runtime, SSE,
#   silhouette score, and the cluster graph.

"""
What this script does:
1. loads the Iris dataset
2. runs Hierarchical Clustering multiple times (10 trials)
3. records runtime, SSE, and silhouette score
4. saves raw and summary results
5. prints results
6. generates ONLY Iris graphs
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from utils import load_iris_data


# -----------------------------------
# SSE calculation function
# -----------------------------------
def calculate_sse(X, labels):

    # initialize SSE
    sse = 0

    # loop through each cluster
    for cluster_id in np.unique(labels):

        # get points belonging to this cluster
        cluster_points = X[labels == cluster_id]

        # compute centroid of cluster
        centroid = np.mean(cluster_points, axis=0)

        # add squared distances to SSE
        sse += np.sum((cluster_points - centroid) ** 2)

    return sse


# -----------------------------------
# Main program
# -----------------------------------
def main():

    # -----------------------------------
    # STEP 1: load the data
    # -----------------------------------
    X, y = load_iris_data()

    # -----------------------------------
    # STEP 2: create folders if needed
    # -----------------------------------
    os.makedirs("results", exist_ok=True)

    results = []

    # -----------------------------------
    # STEP 3: run algorithm (10 trials)
    # -----------------------------------
    for trial in range(1, 11):

        print(f"Running Hierarchical trial {trial}")

        # start timing
        start = time.perf_counter()

        # create model
        model = AgglomerativeClustering(n_clusters=3)

        # run clustering
        labels = model.fit_predict(X)

        # end timing
        runtime = time.perf_counter() - start

        # compute metrics
        sse = calculate_sse(X, labels)
        sil = silhouette_score(X, labels)

        # save trial results
        results.append({
            "algorithm": "Hierarchical",
            "dataset": "Iris",
            "n": len(X),
            "trial": trial,
            "runtime_seconds": runtime,
            "sse": sse,
            "silhouette_score": sil
        })

    # -----------------------------------
    # STEP 4: save raw results
    # -----------------------------------
    df = pd.DataFrame(results)
    df.to_csv("results/hierarchical_iris_results.csv", index=False)

    print("\nRAW RESULTS:")
    print(df)

    # -----------------------------------
    # STEP 5: compute summary
    # -----------------------------------
    summary_df = df.groupby(
        ["algorithm", "dataset", "n"],
        as_index=False
    ).agg({
        "runtime_seconds": "mean",
        "sse": "mean",
        "silhouette_score": "mean"
    })

    summary_df = summary_df.round(4)

    summary_df.to_csv("results/hierarchical_iris_summary.csv", index=False)

    print("\nSUMMARY:")
    print(summary_df)

    # -----------------------------------
    # STEP 6: generate Iris graphs ONLY
    # -----------------------------------
    import hierarchical_graphs
    hierarchical_graphs.generate_iris_graphs()


# -----------------------------------
# RUN PROGRAM
# -----------------------------------
if __name__ == "__main__":
    main()