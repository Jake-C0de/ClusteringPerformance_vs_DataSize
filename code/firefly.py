# Purpose: Represents the Firefly Algorithm for clustering

"""
Some NOTES about the firefly algorithm:
- The firefies represents a set of cluster centers
- The brighter the fireflies are, the better the clustering is.
- Dim fireflies move towards brighter ones.
- If they are the same brightness then they move randomly.
- The fireflies will continue to move towards the better solutions.
- Overtime, the fireflies will gather around the best solution.

What the algorithm does:
1. Generate random cluster centers (initial solutions)
2. Evaluate each solution using SSE (Lower SSE = Better/Brighter)
    (SSE = Sum of Squared Errors)
3. Move worse solutions toward better ones
4. Add randomness to avoid getting stuck
5. Repeat for many iterations
6. Return the best clustering found
"""

import numpy as np
import time

class FireflyClustering:

    def __init__(self):
        # Number of clusters we want (Iris has 3 types of flowers)
        self.n_clusters = 3

        # Number of fireflies (solutions we try at once)
        # More fireflies = more exploration but slower
        self.n_fireflies = 10

        # Number of times we update/move fireflies
        # More iterations = better results but slower
        self.max_iter = 30

        # Randomness factor
        # Helps avoid getting stuck in bad solutions
        self.alpha = 0.2

        # Attraction strength
        # How strongly fireflies move toward better ones
        self.beta = 1.0

        # Light absorption factor
        # Controls how attraction decreases with distance
        self.gamma = 1.0


    def fit(self, X):
        """
        This is the MAIN function.

        This function:
        1. Creates random solutions
        2. Improves them over time
        3. Returns the best clustering found
        """

        # Start timer (we measure runtime for the project)
        start_time = time.time()


        # ============================================
        # STEP 1: CREATE RANDOM FIREFLIES (SOLUTIONS)
        # ============================================

        # Each firefly = a set of centroids (cluster centers)
        fireflies = []

        for i in range(self.n_fireflies):

            centroids = []

            # Create each cluster center
            for j in range(self.n_clusters):

                point = []

                # For each feature (Iris has 4 features)
                for k in range(len(X[0])):

                    # Find min and max values in that feature
                    min_val = np.min(X[:, k])
                    max_val = np.max(X[:, k])

                    # Pick a random value between min and max
                    value = np.random.uniform(min_val, max_val)

                    point.append(value)

                # This point becomes one centroid
                centroids.append(point)

            # One firefly = set of all centroids
            fireflies.append(np.array(centroids))


        # ============================================
        # STEP 2: DEFINE HOW WE MEASURE QUALITY (SSE)
        # ============================================

        def calculate_sse(centroids):

            labels = []

            # Assign each data point to closest centroid
            for point in X:

                distances = []

                for c in centroids:
                    dist = np.linalg.norm(point - c)
                    distances.append(dist)

                # Pick the closest centroid
                labels.append(np.argmin(distances))

            sse = 0

            # Calculate error for each cluster
            for i in range(self.n_clusters):

                cluster_points = []

                # Collect points in cluster i
                for index in range(len(X)):
                    if labels[index] == i:
                        cluster_points.append(X[index])

                # If cluster has points
                if len(cluster_points) > 0:

                    # Add squared distance to centroid
                    for p in cluster_points:
                        sse += np.sum((p - centroids[i]) ** 2)

                else:
                    # Empty cluster = bad → punish heavily
                    sse += 1000000

            return sse


        # ============================================
        # STEP 3: CALCULATE INITIAL SCORES
        # ============================================

        # Score each firefly (lower is better)
        scores = []

        for f in fireflies:
            scores.append(calculate_sse(f))

        scores = np.array(scores)


        # ============================================
        # STEP 4: MOVE FIREFLIES (MAIN LOGIC)
        # ============================================

        for iteration in range(self.max_iter):

            # Compare every firefly with every other firefly
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):

                    # If firefly j is better (lower SSE)
                    if scores[j] < scores[i]:

                        # Distance between the two solutions
                        distance = np.linalg.norm(fireflies[i] - fireflies[j])

                        # Attraction formula
                        # Closer = stronger attraction
                        attractiveness = self.beta * np.exp(-self.gamma * distance ** 2)

                        # Move toward better solution
                        movement = attractiveness * (fireflies[j] - fireflies[i])

                        # Add randomness (prevents getting stuck)
                        random_part = self.alpha * np.random.randn(*fireflies[i].shape)

                        # Update firefly position
                        fireflies[i] = fireflies[i] + movement + random_part

                        # Recalculate score after moving
                        scores[i] = calculate_sse(fireflies[i])


        # ============================================
        # STEP 5: PICK THE BEST SOLUTION
        # ============================================

        # Find best firefly (lowest SSE)
        best_index = np.argmin(scores)

        best_centroids = fireflies[best_index]


        # ============================================
        # STEP 6: ASSIGN FINAL LABELS
        # ============================================

        final_labels = []

        for point in X:

            distances = []

            for c in best_centroids:
                dist = np.linalg.norm(point - c)
                distances.append(dist)

            # Assign point to closest centroid
            final_labels.append(np.argmin(distances))


        # Save results (used later in experiments)
        self.labels_ = np.array(final_labels)
        self.centroids_ = best_centroids
        self.best_sse_ = scores[best_index]
        self.runtime_ = time.time() - start_time


        # Return cluster labels
        return self.labels_