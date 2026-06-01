import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from utils import load_iris_data


# -----------------------------------
# Helper function for plotting metrics
# -----------------------------------
def plot_metric(df, metric_col, y_label, title, output_path):
    plt.figure(figsize=(8, 6))

    for min_samples in sorted(df["min_samples"].unique()):
        subset = df[df["min_samples"] == min_samples].sort_values("eps")

        plt.plot(
            subset["eps"],
            subset[metric_col],
            marker="o",
            label=f"min_samples={int(min_samples)}"
        )

    plt.title(title)
    plt.xlabel("eps")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


# -----------------------------------
# Main function
# -----------------------------------
def main():

    # STEP 1: build paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    results_dir = os.path.join(project_root, "results")
    graphs_dir = os.path.join(project_root, "graphs")

    os.makedirs(graphs_dir, exist_ok=True)

    summary_csv_path = os.path.join(results_dir, "dbscan_iris_summary.csv")

    print("Looking for summary CSV at:", summary_csv_path)

    if not os.path.exists(summary_csv_path):
        print("ERROR: summary CSV not found.")
        return

    # STEP 2: load summary data
    df = pd.read_csv(summary_csv_path)

    # Clean memory values (fix small negatives)
    df["memory_mb"] = df["memory_mb"].clip(lower=0)

    # Sort for consistent graphs
    df = df.sort_values(by=["min_samples", "eps"]).reset_index(drop=True)

    print("\nLoaded summary data:")
    print(df.to_string(index=False))

    # -----------------------------------
    # STEP 3: create graphs
    # -----------------------------------

    plot_metric(
        df,
        "runtime_seconds",
        "Runtime (seconds)",
        "DBSCAN Runtime on Iris",
        os.path.join(graphs_dir, "dbscan_runtime.png")
    )

    plot_metric(
        df,
        "memory_mb",
        "Memory (MB)",
        "DBSCAN Memory Usage on Iris",
        os.path.join(graphs_dir, "dbscan_memory.png")
    )

    plot_metric(
        df,
        "silhouette_score",
        "Silhouette Score",
        "DBSCAN Silhouette Score on Iris",
        os.path.join(graphs_dir, "dbscan_silhouette.png")
    )

    plot_metric(
        df,
        "noise_points",
        "Noise Points",
        "DBSCAN Noise Points on Iris",
        os.path.join(graphs_dir, "dbscan_noise.png")
    )

    plot_metric(
        df,
        "num_clusters",
        "Number of Clusters",
        "DBSCAN Number of Clusters on Iris",
        os.path.join(graphs_dir, "dbscan_num_clusters.png")
    )

    # -----------------------------------
    # STEP 4: best clustering visualization
    # -----------------------------------

    X, y = load_iris_data()

    best_row = df.loc[df["silhouette_score"].idxmax()]
    best_eps = best_row["eps"]
    best_min_samples = int(best_row["min_samples"])

    model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    labels = model.fit_predict(X)

    cluster_path = os.path.join(graphs_dir, "dbscan_clusters_best.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title(
        f"Best DBSCAN Clustering\n"
        f"eps={best_eps}, min_samples={best_min_samples}"
    )
    plt.tight_layout()
    plt.savefig(cluster_path, dpi=300)
    plt.close()

    print(f"Saved: {cluster_path}")
    print("\nAll DBSCAN graphs saved successfully.")


# -----------------------------------
# Run program
# -----------------------------------
if __name__ == "__main__":
    main()