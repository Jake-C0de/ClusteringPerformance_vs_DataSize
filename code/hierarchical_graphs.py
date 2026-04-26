# Purpose: generates graphs for Hierarchical Iris and scaling experiments separately.

"""
What this script does:
1. loads Iris results and creates graphs (trial-based)
2. loads scaling results and creates graphs (n-based)
3. saves them into separate folders so nothing is overwritten
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------
# Helper function to create graphs
# -----------------------------------
def plot_metric(df, x_col, y_col, title, y_label, output_path):

    plt.figure(figsize=(8, 6))
    plt.plot(df[x_col], df[y_col], marker="o")

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------------
# Iris graph generator
# -----------------------------------
def generate_iris_graphs():

    results_dir = "results"
    graphs_dir = "graphs/hierarchical_iris_graphs"

    os.makedirs(graphs_dir, exist_ok=True)

    # load Iris results
    df = pd.read_csv(os.path.join(results_dir, "hierarchical_iris_results.csv"))

    # clean memory values
    df["memory_mb"] = df["memory_mb"].clip(lower=0)

    # create graphs
    plot_metric(df, "trial", "runtime_seconds", "Runtime", "Seconds", f"{graphs_dir}/runtime.png")
    plot_metric(df, "trial", "memory_mb", "Memory", "MB", f"{graphs_dir}/memory.png")
    plot_metric(df, "trial", "sse", "SSE", "Value", f"{graphs_dir}/sse.png")
    plot_metric(df, "trial", "silhouette_score", "Silhouette", "Score", f"{graphs_dir}/silhouette.png")


# -----------------------------------
# Scaling graph generator
# -----------------------------------
def generate_scaling_graphs():

    results_dir = "results"
    graphs_dir = "graphs/hierarchical_scaling_graphs"

    os.makedirs(graphs_dir, exist_ok=True)

    # load scaling summary
    df = pd.read_csv(os.path.join(results_dir, "hierarchical_scaling_summary.csv"))

    df["memory_mb"] = df["memory_mb"].clip(lower=0)
    df = df.sort_values(by="n")

    # create graphs
    plot_metric(df, "n", "runtime_seconds", "Runtime vs n", "Seconds", f"{graphs_dir}/runtime.png")
    plot_metric(df, "n", "memory_mb", "Memory vs n", "MB", f"{graphs_dir}/memory.png")
    plot_metric(df, "n", "sse", "SSE vs n", "Value", f"{graphs_dir}/sse.png")
    plot_metric(df, "n", "silhouette_score", "Silhouette vs n", "Score", f"{graphs_dir}/silhouette.png")