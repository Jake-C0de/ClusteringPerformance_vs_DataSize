# Purpose: runs the DBSCAN experiment multiple times on the Iris dataset,
# records the results, and saves them as a file for analysis.

"""
What this script does:
1. loads the Iris dataset
2. runs DBSCAN using several parameter settings
3. saves results (runtime, number of clusters, noise points, silhouette score)
4. prints results
"""

import os
import time
import numpy as np
import pandas as pd
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

    # -----------------------------------
    # STEP 4: run the algorithm
    # -----------------------------------

    results = []
    last_labels = None

    for eps in eps_values:
        for min_samples in min_samples_values:

            # start timer
            start_time = time.time()

            # create the model
            model = DBSCAN(eps=eps, min_samples=min_samples)

            # run the algorithm
            labels = model.fit_predict(X)

            # end timer
            end_time = time.time()
            runtime = end_time - start_time

            # save last labels for final cluster graph
            last_labels = labels

            # count clusters (ignore noise = -1)
            unique_labels = set(labels)
            num_clusters = len(unique_labels - {-1})

            # count noise points
            noise_points = list(labels).count(-1)

            # calculate silhouette score
            sil = safe_silhouette_score(X, labels)

            # save results for this run
            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "runtime_seconds": runtime,
                "num_clusters": num_clusters,
                "noise_points": noise_points,
                "silhouette_score": sil
            })

    # -----------------------------------
    # STEP 5: save results to file
    # -----------------------------------

    df = pd.DataFrame(results)
    df.to_csv("results/dbscan_iris_results.csv", index=False)

    # -----------------------------------
    # STEP 6: print results to screen
    # -----------------------------------

    print(df)

    # -----------------------------------
    # STEP 7: run graph generation
    # -----------------------------------

    print("Generating graphs...")

    import dbscan_graphs
    dbscan_graphs.main()

# run the program
if __name__ == "__main__":
    main()