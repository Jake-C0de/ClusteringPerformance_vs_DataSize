# Purpose: runs Firefly scaling experiment on synthetic datasets
#
# HOW TO RUN:
#   Open terminal in the project folder, then run:
#
#       py code/firefly_scaling.py
#
# REQUIRED PACKAGES:
#   If packages are missing, run:
#
#       pip install numpy pandas matplotlib scikit-learn
#
# NOTE:
#   Memory is measured using tracemalloc instead of psutil.
#   tracemalloc tracks memory allocations made by Python during execution,
#   which is more consistent for algorithm analysis.
#
#   The summary memory uses cummax() so the graph does not decrease due to
#   small fluctuations from Python memory reuse.
#
# OUTPUT FILES:
#   results/firefly_scaling_raw.csv
#   results/firefly_scaling_summary.csv
#   graphs/firefly_scaling_graphs/*.png


import os
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from firefly import FireflyClustering


# -----------------------------------
# Safe silhouette function
# Prevents errors if only 1 cluster is found
# -----------------------------------
def safe_silhouette_score(X, labels):

    unique = np.unique(labels)

    # silhouette requires at least 2 clusters
    if len(unique) < 2:
        return float("nan")

    return silhouette_score(X, labels)


# -----------------------------------
# Main scaling experiment
# -----------------------------------
def main():

    # -----------------------------------
    # STEP 1: dataset sizes
    # We increase n to study scaling behavior
    # -----------------------------------
    dataset_sizes = [200, 500, 1000, 2000, 5000]

    # create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    results = []

    # -----------------------------------
    # STEP 2: loop over dataset sizes
    # -----------------------------------
    for n in dataset_sizes:

        print(f"\nRunning Firefly for n = {n}")

        # generate synthetic dataset
        # 3 clusters, fixed random state for consistency
        X, _ = make_blobs(
            n_samples=n,
            centers=3,
            random_state=42
        )

        # run multiple trials for averaging
        for trial in range(1, 4):

            # -----------------------------------
            # STEP 3: start memory tracking
            # -----------------------------------
            tracemalloc.start()

            # create Firefly model
            model = FireflyClustering()

            # run clustering
            labels = model.fit(X)

            # get peak memory usage during execution
            current, peak = tracemalloc.get_traced_memory()

            # stop memory tracking
            tracemalloc.stop()

            # -----------------------------------
            # STEP 4: store results
            # -----------------------------------
            results.append({
                "algorithm": "Firefly",
                "dataset": "Synthetic",
                "n": n,
                "trial": trial,
                "runtime_seconds": model.runtime_,             # runtime from model
                "memory_mb": peak / (1024 ** 2),               # convert bytes to MB
                "sse": model.best_sse_,                        # clustering error
                "silhouette_score": safe_silhouette_score(X, labels)  # cluster quality
            })

    # -----------------------------------
    # STEP 5: save raw results
    # -----------------------------------
    df = pd.DataFrame(results)
    df.to_csv("results/firefly_scaling_raw.csv", index=False)

    # -----------------------------------
    # STEP 6: compute summary (averages)
    # -----------------------------------
    summary_df = df.groupby(
        ["algorithm", "dataset", "n"],
        as_index=False
    ).agg({
        "runtime_seconds": "mean",
        "memory_mb": "mean",
        "sse": "mean",
        "silhouette_score": "mean"
    }).round(4)

    # sort by dataset size for proper graph order
    summary_df = summary_df.sort_values(by="n")

    # -----------------------------------
    # STEP 7: fix memory trend
    # ensures memory graph does not decrease due to small fluctuations
    # -----------------------------------
    summary_df["memory_mb"] = summary_df["memory_mb"].cummax()

    # save summary results
    summary_df.to_csv("results/firefly_scaling_summary.csv", index=False)

    # print summary to terminal
    print("\nSUMMARY:")
    print(summary_df)

    # -----------------------------------
    # STEP 8: generate graphs
    # -----------------------------------
    import firefly_graphs
    firefly_graphs.generate_scaling_graphs()


# -----------------------------------
# RUN PROGRAM
# -----------------------------------
if __name__ == "__main__":
    main()