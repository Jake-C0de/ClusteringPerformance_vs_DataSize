# Purpose: runs Firefly scaling experiment on synthetic datasets

import os
import psutil
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from firefly import FireflyClustering


def safe_silhouette_score(X, labels):
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    return silhouette_score(X, labels)


def main():

    dataset_sizes = [200, 500, 1000, 2000, 5000]
    os.makedirs("results", exist_ok=True)

    process = psutil.Process(os.getpid())
    results = []

    for n in dataset_sizes:

        print(f"\nRunning Firefly for n = {n}")

        X, _ = make_blobs(n_samples=n, centers=3, random_state=42)

        for trial in range(1, 4):

            mem_before = process.memory_info().rss / (1024 ** 2)

            model = FireflyClustering()
            labels = model.fit(X)

            mem_after = process.memory_info().rss / (1024 ** 2)

            results.append({
                "algorithm": "Firefly",
                "dataset": "Synthetic",
                "n": n,
                "trial": trial,
                "runtime_seconds": model.runtime_,
                "memory_mb": max(mem_after - mem_before, 0),
                "sse": model.best_sse_,
                "silhouette_score": safe_silhouette_score(X, labels)
            })

    df = pd.DataFrame(results)
    df.to_csv("results/firefly_scaling_raw.csv", index=False)

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
    summary_df.to_csv("results/firefly_scaling_summary.csv", index=False)

    print("\nSUMMARY:")
    print(summary_df)

    import firefly_graphs
    firefly_graphs.generate_scaling_graphs()


if __name__ == "__main__":
    main()