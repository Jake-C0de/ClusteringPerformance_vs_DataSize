# Purpose: runs the Firefly experiment on the Iris dataset (n = 150),
# records results, saves CSV files, and generates Iris-only graphs.

"""
What this script does:
1. loads the Iris dataset
2. runs Firefly multiple times (10 trials)
3. records runtime, memory, SSE, and silhouette score
4. saves raw and summary results
5. generates ONLY Iris graphs
"""

import os
import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from utils import load_iris_data
from firefly import FireflyClustering


# -----------------------------------
# Safe silhouette function
# -----------------------------------
def safe_silhouette_score(X, labels):
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    return silhouette_score(X, labels)


# -----------------------------------
# Main program
# -----------------------------------
def main():

    # STEP 1: load data
    X, y = load_iris_data()

    # STEP 2: folders
    os.makedirs("results", exist_ok=True)

    # STEP 3: memory tracking
    process = psutil.Process(os.getpid())

    results = []

    # STEP 4: run 10 trials
    for trial in range(1, 11):

        print(f"Running Firefly trial {trial}")

        mem_before = process.memory_info().rss / (1024 ** 2)

        model = FireflyClustering()
        labels = model.fit(X)

        mem_after = process.memory_info().rss / (1024 ** 2)

        results.append({
            "algorithm": "Firefly",
            "dataset": "Iris",
            "n": len(X),
            "trial": trial,
            "runtime_seconds": model.runtime_,
            "memory_mb": max(mem_after - mem_before, 0),
            "sse": model.best_sse_,
            "silhouette_score": safe_silhouette_score(X, labels)
        })

    # STEP 5: save raw
    df = pd.DataFrame(results)
    df.to_csv("results/firefly_iris_results.csv", index=False)

    print("\nRAW RESULTS:")
    print(df)

    # STEP 6: summary
    summary_df = df.groupby(
        ["algorithm", "dataset", "n"],
        as_index=False
    ).agg({
        "runtime_seconds": "mean",
        "memory_mb": "mean",
        "sse": "mean",
        "silhouette_score": "mean"
    }).round(4)

    summary_df["memory_mb"] = summary_df["memory_mb"].clip(lower=0)
    summary_df.to_csv("results/firefly_iris_summary.csv", index=False)

    print("\nSUMMARY:")
    print(summary_df)

    # STEP 7: graphs
    import firefly_graphs
    firefly_graphs.generate_iris_graphs()


if __name__ == "__main__":
    main()