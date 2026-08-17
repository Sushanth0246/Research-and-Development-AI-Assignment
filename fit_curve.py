"""
Fit the parametric curve

    x(t) = t*cos(theta) - exp(M*|t|)*sin(0.3t)*sin(theta) + X
    y(t) = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3t)*cos(theta)

to the (x, y) points in data/xy_data.csv, recovering theta, M, X.

Key observation: the CSV gives only (x, y) pairs, NOT the parameter t,
and the rows are NOT ordered by t (consecutive rows jump around rather
than moving smoothly along the curve). So we cannot use standard
curve_fit(t, x, y), which requires known (t_i, x_i, y_i) correspondence.

Approach: correspondence-free ("ICP-style") fitting.
  1. For a candidate (theta, M, X), generate a dense sample of the curve
     over t in [6, 60] (thousands of points).
  2. Build a KD-tree over that dense curve.
  3. For every data point, find the distance to its nearest neighbor on
     the candidate curve (no assumption about which t it corresponds to).
  4. Minimize the mean squared nearest-neighbor distance over
     (theta, M, X), using the given bounds.

This only requires that data points lie ON the curve, not in what order
they were sampled -- so it's robust to the shuffled rows.

Optimization is done in two stages:
  - global search (differential_evolution) across the full bounded
    parameter space, to avoid getting stuck in a local minimum caused
    by the oscillatory sin(0.3t) term,
  - local polish (Nelder-Mead) to refine the global result.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution, minimize

DATA_PATH = "xy_data.csv"
T_MIN, T_MAX = 6, 60
N_DENSE = 6000  # resolution of the candidate curve template

# Parameter bounds as given in the assignment
BOUNDS = [
    (1e-4, 50 - 1e-4),     # theta, degrees
    (-0.05 + 1e-4, 0.05 - 1e-4),  # M
    (1e-4, 100 - 1e-4),    # X
]


def curve_xy(theta_deg: float, M: float, X: float, t: np.ndarray) -> np.ndarray:
    """Evaluate the parametric curve at parameter values t. Returns Nx2 array."""
    theta_rad = np.deg2rad(theta_deg)
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    envelope = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * ct - envelope * st + X
    y = 42 + t * st + envelope * ct
    return np.column_stack([x, y])


def make_loss(data_pts: np.ndarray, t_dense: np.ndarray):
    def loss(params):
        theta_deg, M, X = params
        curve = curve_xy(theta_deg, M, X, t_dense)
        tree = cKDTree(curve)
        dists, _ = tree.query(data_pts, k=1)
        return np.mean(dists ** 2)
    return loss


def fit(data_pts: np.ndarray):
    t_dense = np.linspace(T_MIN, T_MAX, N_DENSE)
    loss = make_loss(data_pts, t_dense)

    global_result = differential_evolution(
        loss, BOUNDS, maxiter=200, tol=1e-12, seed=42, polish=True
    )
    local_result = minimize(
        loss, global_result.x, method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000},
    )
    return local_result.x, local_result.fun


def evaluate_fit(data_pts: np.ndarray, theta_deg: float, M: float, X: float):
    """Report mean/max/rms nearest-point distance and L1 residuals as a fit-quality check."""
    t_dense = np.linspace(T_MIN, T_MAX, N_DENSE)
    curve = curve_xy(theta_deg, M, X, t_dense)
    tree = cKDTree(curve)
    
    # L2 distances (Euclidean)
    dists_l2, indices = tree.query(data_pts, k=1)
    
    # L1 distances (Manhattan)
    l1_dists = np.sum(np.abs(data_pts - curve[indices]), axis=1)
    
    # Calculate residual statistics
    mean_l2 = dists_l2.mean()
    max_l2 = dists_l2.max()
    rms_l2 = np.sqrt(np.mean(dists_l2 ** 2))
    
    mean_l1 = l1_dists.mean()
    max_l1 = l1_dists.max()
    rms_l1 = np.sqrt(np.mean(l1_dists ** 2))
    
    return mean_l2, max_l2, rms_l2, mean_l1, max_l1, rms_l1


def main():
    df = pd.read_csv(DATA_PATH)
    data_pts = df[["x", "y"]].to_numpy()

    (theta_deg, M, X), final_loss = fit(data_pts)
    mean_l2, max_l2, rms_l2, mean_l1, max_l1, rms_l1 = evaluate_fit(data_pts, theta_deg, M, X)

    print("Fitted parameters:")
    print(f"  theta = {theta_deg:.6f} deg ({np.deg2rad(theta_deg):.6f} rad)")
    print(f"  M     = {M:.6f}")
    print(f"  X     = {X:.6f}")
    
    print("\nFit quality (L2/Euclidean):")
    print(f"  mean nearest-point dist = {mean_l2:.6f}")
    print(f"  max nearest-point dist  = {max_l2:.6f}")
    print(f"  rms nearest-point dist  = {rms_l2:.6f}")
    
    print("\nFit quality (L1/Manhattan):")
    print(f"  mean nearest-point residual = {mean_l1:.6f}")
    print(f"  max nearest-point residual  = {max_l1:.6f}")
    print(f"  rms nearest-point residual  = {rms_l1:.6f}")

    theta_rad = np.deg2rad(theta_deg)
    equation = (
        f"\\left(t*\\cos({theta_rad:.6f})-e^{{{M:.6f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\sin({theta_rad:.6f})+{X:.6f},"
        f"42+t*\\sin({theta_rad:.6f})+e^{{{M:.6f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\cos({theta_rad:.6f})\\right)"
    )
    print("\nFinal parametric equation (Desmos LaTeX format):")
    print(equation)


if __name__ == "__main__":
    main()