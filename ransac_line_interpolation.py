import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


ransac = RANSACRegressor(
    estimator=LinearRegression(),
    residual_threshold=0.2,
    min_samples=2,
    max_trials=1000,
)


def fit_ransac_line(xs, ys):
    X = np.array(xs).reshape(-1, 1)
    Y = np.array(ys)
    if len(X) < 2:
        raise ValueError("Need at least 2 points for RANSAC line fitting")

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=0.2,
        min_samples=2,
        max_trials=1000,
    )
    ransac.fit(X, Y)
    line_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    line_y = ransac.predict(line_x)
    return line_x, line_y
