# Purpose: runs the Hierarchical Clustering experiment on the Iris dataset (n = 150),
# records results, saves CSV files, and generates Iris-only graphs.

"""
What this script does:
1. loads the Iris dataset
2. runs Hierarchical Clustering multiple times (10 trials)
3. records runtime, memory, SSE, and silhouette score
4. saves raw and summary results
5. prints results
6. generates ONLY Iris graphs
"""

import os
import time
import psutil
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

    # -----------------------------------
    # STEP 3: setup memory tracking
    # -----------------------------------
    process = psutil.Process(os.getpid())

    results = []

    # -----------------------------------
    # STEP 4: run algorithm (10 trials)
    # -----------------------------------
    for trial in range(1, 11):

        print(f"Running Hierarchical trial {trial}")

        # memory before running algorithm
        mem_before = process.memory_info().rss / (1024 ** 2)

        # start timing
        start = time.perf_counter()

        # create model
        model = AgglomerativeClustering(n_clusters=3)

        # run clustering
        labels = model.fit_predict(X)

        # end timing
        runtime = time.perf_counter() - start

        # memory after running algorithm
        mem_after = process.memory_info().rss / (1024 ** 2)

        # compute memory used
        memory_used = max(mem_after - mem_before, 0)

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
            "memory_mb": memory_used,
            "sse": sse,
            "silhouette_score": sil
        })

    # -----------------------------------
    # STEP 5: save raw results
    # -----------------------------------
    df = pd.DataFrame(results)
    df.to_csv("results/hierarchical_iris_results.csv", index=False)

    print("\nRAW RESULTS:")
    print(df)

    # -----------------------------------
    # STEP 6: compute summary
    # -----------------------------------
    summary_df = df.groupby(
        ["algorithm", "dataset", "n"],
        as_index=False
    ).agg({
        "runtime_seconds": "mean",
        "memory_mb": "mean",
        "sse": "mean",
        "silhouette_score": "mean"
    })

    summary_df = summary_df.round(4)

    # ensure memory is non-negative
    summary_df["memory_mb"] = summary_df["memory_mb"].clip(lower=0)

    summary_df.to_csv("results/hierarchical_iris_summary.csv", index=False)

    print("\nSUMMARY:")
    print(summary_df)

    # -----------------------------------
    # STEP 7: generate Iris graphs ONLY
    # -----------------------------------
    import hierarchical_graphs
    hierarchical_graphs.generate_iris_graphs()


# -----------------------------------
# RUN PROGRAM
# -----------------------------------
if __name__ == "__main__":
    main()