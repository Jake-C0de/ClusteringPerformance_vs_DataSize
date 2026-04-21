# ClusteringPerformance_vs_DataSize

# Team Members
Patriya Murray, Nevaeh Zumbrun, Jake Batton

# Project Description
This project compares different clustering algorithms using the Iris dataset. The goal is to evaluate how each algorithm performs based on runtime and clustering quality.

# Algorithms
The algorithms used in this project are:
- K-Means: centroid-based algorithm
- DBSCAN: density-based algorithm
- Firefly Algorithm: nature-inspired

# Dataset
We use the Iris dataset from scikit-learn. It contains 150 samples with 4 features and 3 natural groups of flowers. It is commonly used for clustering and classification tasks.

# What the Code Does
- Runs clustering algorithms on the dataset
- Tests different parameter settings
- Measures runtime and clustering quality
- Saves results to CSV files
- Generates graphs for analysis

# Running the Code
1. Install required libraries: pip install numpy pandas matplotlib scikit-learn
2. Files to Run: k-means.py, dbscan.py, run_firefly_experiment.py
    Commands to run:
        python code/k-means.py
        python code/dbscan.py
        python code/run_firefly_experiment.py
3. Output:
- Results file will be saved in `/results`
- Graphs will be saved in `/graphs`

# Folder Structure
- code/ -> all Python files
- results/ -> CSV output
- graphs/ -> generated graphs
- report/ -> written report
- slides/ -> presentation
- references/ -> sources  

# GenAI Usage Disclosure
GenAI tools were used to help understand concepts, debug code, generate code, and explain algorithms.