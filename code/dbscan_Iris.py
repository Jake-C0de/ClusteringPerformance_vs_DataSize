# Purpose: runs the DBSCAN experiment multiple times on the Iris dataset,
# records the results, and saves them as a file for analysis.

"""
What this script does:
1. loads the Iris dataset
2. runs DBSCAN using several parameter settings
3. runs each setting 5 times
4. saves results (runtime, memory, number of clusters, noise points, silhouette score)
5. prints results
"""

import os
import time
from tabulate import tabulate
import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from utils import load_iris_data


# this function makes sure silhouette score does not crash
def safe_silhouette_score(X, labels):

    # get unique clusters
    unique = np.unique(labels)

    # remove noise label if it exists
    unique = unique[unique != -1]

    # if not enough real clusters, return nothing
    if len(unique) < 2:
        return float("nan")

    # only use non-noise points for silhouette score
    mask = labels != -1

    # if too few points remain, return nothing
    if np.sum(mask) < 3:
        return float("nan")

    # otherwise calculate score
    return silhouette_score(X[mask], labels[mask])


def main():

    # -----------------------------------
    # STEP 1: load the data
    # -----------------------------------
    X, y = load_iris_data()

    # -----------------------------------
    # STEP 2: make folders if needed
    # -----------------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    # -----------------------------------
    # STEP 3: choose DBSCAN settings
    # -----------------------------------
    eps_values = [0.3, 0.5, 0.7, 0.9]
    min_samples_values = [3, 5, 7]
    num_trials = 5

    # process for memory measurement
    process = psutil.Process(os.getpid())

    # -----------------------------------
    # STEP 4: run the algorithm
    # -----------------------------------
    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            for trial in range(1, num_trials + 1):

                # memory before
                memory_before = process.memory_info().rss / (1024 ** 2)

                # start timer
                start_time = time.perf_counter()

                # create the model
                model = DBSCAN(eps=eps, min_samples=min_samples)

                # run the algorithm
                labels = model.fit_predict(X)

                # end timer
                end_time = time.perf_counter()
                runtime = end_time - start_time

                # memory after
                memory_after = process.memory_info().rss / (1024 ** 2)
                memory_used_mb = memory_after - memory_before

                # count clusters (ignore noise = -1)
                unique_labels = set(labels)
                num_clusters = len(unique_labels - {-1})

                # count noise points
                noise_points = list(labels).count(-1)

                # calculate silhouette score
                sil = safe_silhouette_score(X, labels)

                # save results for this run
                results.append({
                    "algorithm": "DBSCAN",
                    "dataset": "Iris",
                    "n": len(X),
                    "trial": trial,
                    "eps": eps,
                    "min_samples": min_samples,
                    "runtime_seconds": runtime,
                    "memory_mb": memory_used_mb,
                    "num_clusters": num_clusters,
                    "noise_points": noise_points,
                    "silhouette_score": sil
                })

    # -----------------------------------
    # STEP 5: Create Data Frame
    # -----------------------------------
    df = pd.DataFrame(results)

    # -----------------------------------
    # STEP 6: make averaged summary
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

    # round values for readability
    summary_df = summary_df.round({
        "runtime_seconds": 4,
        "memory_mb": 4,
        "num_clusters": 2,
        "noise_points": 2,
        "silhouette_score": 4
    })

    # -----------------------------------
    # STEP 7: print results
    # -----------------------------------
    

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    raw_csv_path = os.path.join(results_dir, "dbscan_iris_results.csv")
    summary_csv_path = os.path.join(results_dir, "dbscan_iris_summary.csv")

    df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nRAW TRIAL RESULTS:")
    print(df.to_string(index=False))

    print("\nAVERAGED SUMMARY:")
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))

    print(f"\nRaw results saved to: {raw_csv_path}")
    print(f"Summary results saved to: {summary_csv_path}")

   

    # -----------------------------------
    # STEP 8: run graph generation
    # -----------------------------------
    print("Generating graphs...")

    import dbscan_graphs
    dbscan_graphs.main()


# run the program
if __name__ == "__main__":
    main()