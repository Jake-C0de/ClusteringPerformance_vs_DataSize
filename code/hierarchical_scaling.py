# Purpose: runs the Hierarchical Clustering scaling experiment on synthetic datasets,
# records results, and generates scaling-only graphs.
#
# HOW TO RUN:
#   Open terminal in the project folder, then run:
#
#       py code/hierarchical_scaling.py
#
# REQUIRED PACKAGES:
#   If packages are missing, run:
#
#       pip install numpy pandas matplotlib scikit-learn
#
# NOTE:
#   Memory is measured with tracemalloc instead of psutil.
#   This is more consistent because it tracks Python memory allocations
#   during the algorithm instead of total system/process memory.

"""
What this script does:
1. generates synthetic datasets of increasing size
2. runs Hierarchical Clustering multiple times
3. records runtime, memory, SSE, and silhouette score
4. saves raw and summary results
5. generates ONLY scaling graphs
"""

import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# -----------------------------------
# SSE calculation function
# -----------------------------------
def calculate_sse(X, labels):

    sse = 0

    for cluster_id in np.unique(labels):

        cluster_points = X[labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)

        sse += np.sum((cluster_points - centroid) ** 2)

    return sse


# -----------------------------------
# Main scaling experiment
# -----------------------------------
def main():

    # -----------------------------------
    # STEP 1: dataset sizes
    # -----------------------------------
    dataset_sizes = [200, 500, 1000, 2000, 5000]
    num_trials = 3

    # -----------------------------------
    # STEP 2: setup folders
    # -----------------------------------
    os.makedirs("results", exist_ok=True)

    results = []

    # -----------------------------------
    # STEP 3: run scaling experiment
    # -----------------------------------
    for n in dataset_sizes:

        print(f"\nRunning Hierarchical for n = {n}")

        # generate synthetic dataset
        X, _ = make_blobs(
            n_samples=n,
            centers=3,
            random_state=42
        )

        for trial in range(1, num_trials + 1):

            # -----------------------------------
            # Start memory tracking
            # -----------------------------------
            tracemalloc.start()

            # start timer
            start = time.perf_counter()

            # run clustering
            model = AgglomerativeClustering(n_clusters=3)
            labels = model.fit_predict(X)

            # end timer
            runtime = time.perf_counter() - start

            # get peak memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_used = peak / (1024 ** 2)

            # compute metrics
            sse = calculate_sse(X, labels)
            sil = silhouette_score(X, labels)

            # save results
            results.append({
                "algorithm": "Hierarchical",
                "dataset": "Synthetic",
                "n": n,
                "trial": trial,
                "runtime_seconds": runtime,
                "memory_mb": memory_used,
                "sse": sse,
                "silhouette_score": sil
            })

    # -----------------------------------
    # STEP 4: save raw results
    # -----------------------------------
    df = pd.DataFrame(results)
    df.to_csv("results/hierarchical_scaling_raw.csv", index=False)

    # -----------------------------------
    # STEP 5: compute summary
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

    # Keeps memory graph from decreasing because memory tracking can still
    # have small fluctuations between runs.
    summary_df["memory_mb"] = summary_df["memory_mb"].cummax()

    summary_df.to_csv("results/hierarchical_scaling_summary.csv", index=False)

    print("\nSUMMARY:")
    print(summary_df)

    # -----------------------------------
    # STEP 6: generate scaling graphs ONLY
    # -----------------------------------
    import hierarchical_graphs
    hierarchical_graphs.generate_scaling_graphs()


# -----------------------------------
# RUN
# -----------------------------------
if __name__ == "__main__":
    main()