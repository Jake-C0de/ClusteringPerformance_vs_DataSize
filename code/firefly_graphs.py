# Purpose: generates Firefly graphs for Iris and scaling experiments

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot(df, x, y, title, ylabel, path):
    plt.figure()
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def generate_iris_graphs():

    os.makedirs("graphs/firefly_iris_graphs", exist_ok=True)

    df = pd.read_csv("results/firefly_iris_results.csv")

    plot(df, "trial", "runtime_seconds", "Runtime", "Seconds", "graphs/firefly_iris_graphs/runtime.png")
    plot(df, "trial", "memory_mb", "Memory", "MB", "graphs/firefly_iris_graphs/memory.png")
    plot(df, "trial", "sse", "SSE", "Value", "graphs/firefly_iris_graphs/sse.png")
    plot(df, "trial", "silhouette_score", "Silhouette", "Score", "graphs/firefly_iris_graphs/silhouette.png")


def generate_scaling_graphs():

    os.makedirs("graphs/firefly_scaling_graphs", exist_ok=True)

    df = pd.read_csv("results/firefly_scaling_summary.csv")

    plot(df, "n", "runtime_seconds", "Runtime vs n", "Seconds", "graphs/firefly_scaling_graphs/runtime.png")
    plot(df, "n", "memory_mb", "Memory vs n", "MB", "graphs/firefly_scaling_graphs/memory.png")
    plot(df, "n", "sse", "SSE vs n", "Value", "graphs/firefly_scaling_graphs/sse.png")
    plot(df, "n", "silhouette_score", "Silhouette vs n", "Score", "graphs/firefly_scaling_graphs/silhouette.png")