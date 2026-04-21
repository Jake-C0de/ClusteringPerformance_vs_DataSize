# Purpose: loads the DBSCAN results and makes graphs for analysis.

"""
What this script does:
1. loads the Iris dataset
2. loads saved DBSCAN results
3. makes graphs
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from utils import load_iris_data


def main():

    # -----------------------------------
    # STEP 1: load the data
    # -----------------------------------

    X, y = load_iris_data()

    # -----------------------------------
    # STEP 2: make folders if needed
    # -----------------------------------

    os.makedirs("graphs", exist_ok=True)

    # -----------------------------------
    # STEP 3: load results from file
    # -----------------------------------

    df = pd.read_csv("results/dbscan_iris_results.csv")

    # -----------------------------------
    # STEP 4: prepare graph labels
    # -----------------------------------

    x_labels = [f"eps={row['eps']}, min={row['min_samples']}" for _, row in df.iterrows()]
    x_positions = range(len(df))

    # -----------------------------------
    # STEP 5: make runtime graph
    # -----------------------------------

    plt.figure()
    plt.plot(x_positions, df["runtime_seconds"])
    plt.title("DBSCAN Runtime")
    plt.xlabel("Parameter Setting")
    plt.ylabel("Seconds")
    plt.xticks(x_positions, x_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("graphs/dbscan_runtime.png")
    plt.close()

    # -----------------------------------
    # STEP 6: make quality graph
    # -----------------------------------

    plt.figure()
    plt.plot(x_positions, df["silhouette_score"])
    plt.title("DBSCAN Silhouette Score")
    plt.xlabel("Parameter Setting")
    plt.ylabel("Score")
    plt.xticks(x_positions, x_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("graphs/dbscan_quality.png")
    plt.close()

    # -----------------------------------
    # STEP 7: make noise graph
    # -----------------------------------

    plt.figure()
    plt.plot(x_positions, df["noise_points"])
    plt.title("DBSCAN Noise Points")
    plt.xlabel("Parameter Setting")
    plt.ylabel("Number of Noise Points")
    plt.xticks(x_positions, x_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("graphs/dbscan_noise.png")
    plt.close()

    # -----------------------------------
    # STEP 8: make cluster graph
    # -----------------------------------

    # use the last parameter setting from the results file
    last_row = df.iloc[-1]

    model = DBSCAN(eps=last_row["eps"], min_samples=int(last_row["min_samples"]))
    labels = model.fit_predict(X)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("DBSCAN Clustering")
    plt.savefig("graphs/dbscan_clusters.png")
    plt.close()


# run the program
if __name__ == "__main__":
    main()