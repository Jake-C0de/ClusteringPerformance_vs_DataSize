# Purpose: Loading and scaling the Iris dataset

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def load_iris_data():
    # load the Iris dataset
    X, y = load_iris(return_X_y=True)

    # makes the data easier to work with by scaling it
    # this puts the columns on a similar size
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # X_scaled = the flower measurements
    # y = the real flower types
    return X_scaled, y