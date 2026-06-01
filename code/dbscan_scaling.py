import os
import time
import numpy as np
import pandas as pd
import psutil

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


"""
HOW TO RUN DBSCAN SCALING EXPERIMENT

1. Open a terminal in the project root directory

2. Install required packages (only needed once):
   pip install numpy pandas scikit-learn psutil

3. Run the script:
   python code/dbscan_scaling.py

4. What this program does:
   - Generates synthetic datasets of increasing size
   - Runs DBSCAN multiple times (10 trials per size)
   - Measures runtime, memory usage, clustering quality

5. Output:
   - Raw results saved to: results/dbscan_scaling_raw.csv
   - Summary results saved to: results/dbscan_scaling_summary.csv
"""


# -----------------------------------
# Safe silhouette function (same as before)
# -----------------------------------
def safe_silhouette_score(X, labels):
    unique = np.unique(labels)
    unique = unique[unique != -1]

    if len(unique) < 2:
        return float("nan")

    mask = labels != -1

    if np.sum(mask) < 3:
        return float("nan")

    return silhouette_score(X[mask], labels[mask])


# -----------------------------------
# Main scaling experiment
# -----------------------------------
def main():

    # -----------------------------------
    # STEP 1: dataset sizes
    # -----------------------------------
    dataset_sizes = [200, 500, 1000, 2000, 5000]

    # Keep DBSCAN parameters fixed (important!)
    eps = 0.5
    min_samples = 5
    num_trials = 10


    # -----------------------------------
    # STEP 2: setup paths
    # -----------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------
    # STEP 3: memory tracking
    # -----------------------------------
    process = psutil.Process(os.getpid())

    results = []

    # -----------------------------------
    # STEP 4: loop through dataset sizes
    # -----------------------------------
    for n in dataset_sizes:

        print(f"\nRunning DBSCAN for n = {n}")

        # generate synthetic dataset
        X, _ = make_blobs(
            n_samples=n,
            centers=3,
            n_features=2,
            cluster_std=1.0,
            random_state=42
        )

        for trial in range(1, num_trials + 1):

            # memory before
            mem_before = process.memory_info().rss / (1024 ** 2)

            # start timer
            start = time.perf_counter()

            # run DBSCAN
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)

            # end timer
            end = time.perf_counter()
            runtime = end - start

            # memory after
            mem_after = process.memory_info().rss / (1024 ** 2)
            memory_used = mem_after - mem_before

            # clusters (ignore noise)
            unique_labels = set(labels)
            num_clusters = len(unique_labels - {-1})

            # noise count
            noise_points = list(labels).count(-1)

            # silhouette
            sil = safe_silhouette_score(X, labels)

            results.append({
                "algorithm": "DBSCAN",
                "dataset": "Synthetic",
                "n": n,
                "trial": trial,
                "eps": eps,
                "min_samples": min_samples,
                "runtime_seconds": runtime,
                "memory_mb": memory_used,
                "num_clusters": num_clusters,
                "noise_points": noise_points,
                "silhouette_score": sil
            })

    
# STEP 5: save results
# -----------------------------------
    df = pd.DataFrame(results)

    # remove negative memory values caused by measurement noise
    df["memory_mb"] = df["memory_mb"].clip(lower=0)

    raw_path = os.path.join(results_dir, "dbscan_scaling_raw.csv")
    df.to_csv(raw_path, index=False)

    # -----------------------------------
    # STEP 6: average results by n
    # -----------------------------------
    summary_df = df.groupby(
        ["algorithm", "dataset", "n", "eps", "min_samples"],
        as_index=False
    ).agg({
        "runtime_seconds": "mean",
        "memory_mb": "mean",
        "num_clusters": "mean",
        "noise_points": "mean",
        "silhouette_score": "mean"
    })

    summary_df = summary_df.round(4)

    summary_path = os.path.join(results_dir, "dbscan_scaling_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nSUMMARY:")
    print(summary_df)

    print("\nSaved:")
    print(raw_path)
    print(summary_path)


# -----------------------------------
# RUN
# -----------------------------------
if __name__ == "__main__":
    main()