# ClusteringPerformance_vs_DataSize

# Team Members
Patriya Murray, Nevaeh Zumbrun, Jake Batton

# Project Description
This project compares multiple clustering algorithms using both the Iris dataset and synthetic datasets. The goal is to evaluate how each algorithm performs in terms of runtime, memory usage, and clustering quality.

The project also analyzes how algorithm performance scales as dataset size increases.

---

# Algorithms
The algorithms used in this project are:

- K-Means: centroid-based clustering algorithm
- DBSCAN: density-based clustering algorithm
- Firefly Algorithm: nature-inspired optimization algorithm
- Hierarchical Clustering: agglomerative clustering method

---

# Datasets

## Iris Dataset
- 150 samples
- 4 features
- 3 natural clusters
- Used for evaluating clustering accuracy and consistency

## Synthetic Dataset
- Generated using `make_blobs` from scikit-learn
- Sizes: 200, 500, 1000, 2000, 5000
- Used for scaling analysis

---

# What the Code Does

## Iris Experiments
- Runs each algorithm multiple times
- Measures:
  - runtime
  - SSE (sum of squared errors)
  - silhouette score
- Generates graphs based on trials
- Generates cluster visualization

## Scaling Experiments
- Runs algorithms on increasing dataset sizes
- Measures:
  - runtime
  - memory usage
  - SSE
  - silhouette score
- Generates graphs based on dataset size (n)
- Generates a representative cluster graph

---

# Metrics Used

- Runtime (seconds): measures execution time
- Memory (MB): measured using tracemalloc (scaling only)
- SSE (Sum of Squared Errors): measures clustering error
- Silhouette Score: measures cluster quality (-1 to 1)

---

# Important Notes

- Memory for Firefly and Hierarchical is **not included for Iris graphs** because dataset size is constant, making memory comparisons misleading.
- Memory is measured using **tracemalloc**, which tracks Python memory allocations.
- Firefly centroids are recalculated for visualization to ensure they appear at the center of clusters.
- Hierarchical clustering is deterministic, so repeated trials produce identical results.

---

## Install Required Libraries
pip install numpy pandas matplotlib scikit-learn psutil

# Running the Code

## Run Iris Experiments
python code/k-means.py
python code/dbscan_Iris.py
python code/run_firefly_experiment.py
python code/hierarchical.py

## Run Scaling Experiments
python code/k-means.py
python code/dbscan_scaling.py
python code/firefly_scaling.py
python code/hierarchical_scaling.py

python code/dbscan_scaling_graphs.py
python code/kmeans_scaling_graphs.py
python code/firefly_scaling_graphs.py
python code/hierarchical_scaling_graphs.py


---

# Output

## Results
- Saved in `/results`
- Includes raw and summary CSV files

## Graphs
- Saved in `/graphs`

### K-Means
- `/graphs/kmeans_iris_graphs/`
- `/graphs/kmeans_scaling_graphs/`

### DBSCAN
- `/graphs/dbscan_iris_graphs/`
- `/graphs/dbscan_scaling_graphs/`

### Firefly
- `/graphs/firefly_iris_graphs/`
- `/graphs/firefly_scaling_graphs/`

### Hierarchical
- `/graphs/hierarchical_iris_graphs/`
- `/graphs/hierarchical_scaling_graphs/`

---

# Folder Structure

- `code/` → all Python files
- `results/` → CSV output files
- `graphs/` → generated graphs
- `report/` → written report
- `slides/` → presentation
- `references/` → sources

---

# GenAI Usage Disclosure

GenAI tools were used to:
- help understand clustering concepts
- generate code
- debug code
- explain algorithm behavior and results
