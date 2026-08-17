"""Generate the diagnostic plots in assets/: raw data scatter and fitted overlay."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_curve import curve_xy, T_MIN, T_MAX, N_DENSE

THETA_DEG, M, X = 30.000009, 0.030000, 55.000009  # from fit_curve.py


def main():
    df = pd.read_csv("xy_data.csv")

    # Raw scatter (sanity check: is it a single smooth curve?)
    plt.figure(figsize=(7, 7))
    plt.scatter(df.x, df.y, s=3)
    plt.gca().set_aspect("equal")
    plt.title("Raw data: x vs y")
    plt.savefig("assets/raw_scatter.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Fitted curve overlay
    t_dense = np.linspace(T_MIN, T_MAX, N_DENSE)
    curve = curve_xy(THETA_DEG, M, X, t_dense)

    plt.figure(figsize=(7, 7))
    plt.scatter(df.x, df.y, s=8, alpha=0.5, label="data")
    plt.plot(curve[:, 0], curve[:, 1], color="red", linewidth=1,
              label=f"fitted curve (θ={THETA_DEG:.0f}°, M={M:.2f}, X={X:.0f})")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.title("Data vs fitted curve")
    plt.savefig("assets/fit_overlay.png", dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
