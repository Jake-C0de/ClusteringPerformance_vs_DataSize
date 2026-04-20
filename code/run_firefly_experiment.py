# Purpose: runs the Firefly experiment multiple times on the Iris dataset,
# records the results, and saves them as a file and graphs for analysis.

"""
What this script does:
1. loads the Iris dataset
2. runs the Firefly algorithm 10 times
3. saves results (runtime, SSE, silhouette score)
4. makes graphs
5. prints results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from utils import load_iris_data
from firefly import FireflyClustering


# this function makes sure silhouette score does not crash
def safe_silhouette_score(X, labels):

    # get unique clusters
    unique = np.unique(labels)

    # if not enough clusters, return nothing
    if len(unique) < 2:
        return float("nan")

    # otherwise calculate score
    return silhouette_score(X, labels)


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
    # STEP 3: run the algorithm multiple times
    # -----------------------------------

    results = []

    for trial in range(10):

        # create the model
        model = FireflyClustering()

        # run the algorithm
        labels = model.fit(X)

        # calculate silhouette score
        sil = safe_silhouette_score(X, labels)

        # save results for this run
        results.append({
            "trial": trial + 1,
            "runtime_seconds": model.runtime_,
            "sse": model.best_sse_,
            "silhouette_score": sil
        })

    # -----------------------------------
    # STEP 4: save results to file
    # -----------------------------------

    df = pd.DataFrame(results)
    df.to_csv("results/firefly_iris_results.csv", index=False)

    # -----------------------------------
    # STEP 5: make runtime graph
    # -----------------------------------

    plt.figure()
    plt.plot(df["trial"], df["runtime_seconds"])
    plt.title("Runtime")
    plt.xlabel("Trial")
    plt.ylabel("Seconds")
    plt.savefig("graphs/firefly_runtime.png")
    plt.close()

    # -----------------------------------
    # STEP 6: make quality graph
    # -----------------------------------

    plt.figure()
    plt.plot(df["trial"], df["silhouette_score"])
    plt.title("Silhouette Score")
    plt.xlabel("Trial")
    plt.ylabel("Score")
    plt.savefig("graphs/firefly_quality.png")
    plt.close()

    # -----------------------------------
    # STEP 7: make cluster graph
    # -----------------------------------

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("Firefly Clustering")
    plt.savefig("graphs/firefly_clusters.png")
    plt.close()

    # -----------------------------------
    # STEP 8: print results to screen
    # -----------------------------------

    print(df)


# run the program
if __name__ == "__main__":
    main()