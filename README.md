# ClusteringPerformance_vs_DataSize

# Team Members
Patriya Murray, 

# Project Description
This project compares different clustering algorithms using the Iris dataset. The goal is to evaluate how each algorithm performs based on runtime and clustering quality.

The algorithms used in this project are:
- K-Means
- DBSCAN
- Firefly Algorithm

# Dataset
We use the Iris dataset from scikit-learn. It contains measurements of flowers and is commonly used for clustering problems.

# What the Code Does
- Runs clustering algorithms on the dataset
- Measures runtime
- Measures clustering quality (SSE and silhouette score)
- Saves results to a CSV file
- Generates graphs for analysis

# How to Run the Code
1. Install required libraries: pip install numpy pandas matplotlib scikit-learn
2. Run: k-means.py,  , run_firefly_experiment.py
3. Output:
- Results file will be saved in `/results`
- Graphs will be saved in `/graphs`

# Folder Structure
- `code/` → all Python files  
- `results/` → CSV output  
- `graphs/` → generated graphs  

# GenAI Usage Disclosure
GenAI tools were used to help understand concepts, debug code, generate code, and explain algorithms.