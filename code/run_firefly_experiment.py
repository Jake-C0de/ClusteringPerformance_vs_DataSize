# Purpose: runs the Firefly experiment on the Iris dataset (n = 150),
# records results, saves CSV files, and generates Iris-only graphs.

"""
HOW TO RUN:
    py code/run_firefly_experiment.py

REQUIRED PACKAGES:
    pip install numpy pandas matplotlib scikit-learn

NOTE:
    Memory is NOT recorded for Iris because dataset size does not change,
    and memory measurements are inconsistent / not meaningful.
"""

import os
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

    results = []

    # STEP 3: run 10 trials
    for trial in range(1, 11):

        print(f"Running Firefly trial {trial}")

        model = FireflyClustering()
        labels = model.fit(X)

        results.append({
            "algorithm": "Firefly",
            "dataset": "Iris",
            "n": len(X),
            "trial": trial,
            "runtime_seconds": model.runtime_,
            "sse": model.best_sse_,
            "silhouette_score": safe_silhouette_score(X, labels)
        })

    # STEP 4: save raw
    df = pd.DataFrame(results)
    df.to_csv("results/firefly_iris_results.csv", index=False)

    print("\nRAW RESULTS:")
    print(df)

    # STEP 5: summary
    summary_df = df.groupby(
        ["algorithm", "dataset", "n"],
        as_index=False
    ).agg({
        "runtime_seconds": "mean",
        "sse": "mean",
        "silhouette_score": "mean"
    }).round(4)

    summary_df.to_csv("results/firefly_iris_summary.csv", index=False)

    print("\nSUMMARY:")
    print(summary_df)

    # STEP 6: graphs
    import firefly_graphs
    firefly_graphs.generate_iris_graphs()


if __name__ == "__main__":
    main()