import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_scaling_metric(df, metric_col, y_label, title, output_path):
    plt.figure(figsize=(8, 6))

    plt.plot(
        df["n"],
        df[metric_col],
        marker="o"
    )

    plt.title(title)
    plt.xlabel("Dataset Size (n)")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    # -----------------------------------
    # STEP 1: build paths
    # -----------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    results_dir = os.path.join(project_root, "results")
    graphs_dir = os.path.join(project_root, "graphs", "dbscan_scaling_graphs")

    os.makedirs(graphs_dir, exist_ok=True)

    summary_csv_path = os.path.join(results_dir, "dbscan_scaling_summary.csv")

    print("Looking for summary CSV at:", summary_csv_path)

    if not os.path.exists(summary_csv_path):
        print("ERROR: dbscan_scaling_summary.csv not found.")
        return

    # -----------------------------------
    # STEP 2: load summary data
    # -----------------------------------
    df = pd.read_csv(summary_csv_path)

    # Clean small negative memory values
    df["memory_mb"] = df["memory_mb"].clip(lower=0)

    # Sort by n so the line graph is correct
    df = df.sort_values(by="n").reset_index(drop=True)

    print("\nLoaded DBSCAN scaling summary:")
    print(df.to_string(index=False))

    # -----------------------------------
    # STEP 3: make scaling graphs
    # -----------------------------------

    plot_scaling_metric(
        df,
        metric_col="runtime_seconds",
        y_label="Runtime (seconds)",
        title="DBSCAN Runtime vs Dataset Size",
        output_path=os.path.join(graphs_dir, "dbscan_scaling_runtime.png")
    )

    plot_scaling_metric(
        df,
        metric_col="memory_mb",
        y_label="Memory Usage (MB)",
        title="DBSCAN Memory Usage vs Dataset Size",
        output_path=os.path.join(graphs_dir, "dbscan_scaling_memory.png")
    )

    plot_scaling_metric(
        df,
        metric_col="silhouette_score",
        y_label="Silhouette Score",
        title="DBSCAN Silhouette Score vs Dataset Size",
        output_path=os.path.join(graphs_dir, "dbscan_scaling_silhouette.png")
    )

    plot_scaling_metric(
        df,
        metric_col="num_clusters",
        y_label="Number of Clusters",
        title="DBSCAN Number of Clusters vs Dataset Size",
        output_path=os.path.join(graphs_dir, "dbscan_scaling_num_clusters.png")
    )

    plot_scaling_metric(
        df,
        metric_col="noise_points",
        y_label="Noise Points",
        title="DBSCAN Noise Points vs Dataset Size",
        output_path=os.path.join(graphs_dir, "dbscan_scaling_noise.png")
    )

    print("\nAll DBSCAN scaling graphs saved successfully.")


if __name__ == "__main__":
    main()