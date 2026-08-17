# Flam R&D/AI Assignment — Parametric Curve Fitting

## Problem

Recover the unknown parameters `theta`, `M`, `X` in:

```
x(t) = t*cos(theta) - exp(M*|t|)*sin(0.3t)*sin(theta) + X
y(t) = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3t)*cos(theta)
```

given 1500 `(x, y)` points sampled from the curve for `6 < t < 60`, with
`0° < theta < 50°`, `-0.05 < M < 0.05`, `0 < X < 100`.

## Answer

| Parameter | Value |
|---|---|
| theta | 30° (0.523599 rad) |
| M | 0.03 |
| X | 55 |

Desmos-style expression (theta in radians):

```
\left(t*\cos(0.523599)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.523599)+55,42+t*\sin(0.523599)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.523599)\right)
```

## Process

**1. Inspect the data.** `xy_data.csv` has 1500 rows of `(x, y)` only —
no `t` column. Plotting the raw points (`assets/raw_scatter.png`) shows they
lie on a single smooth curve, but consecutive CSV rows do **not** move
smoothly along it (e.g. successive x-values jump around rather than
increasing/decreasing monotonically). So the rows are shuffled and there's
no way to know which `t` each row corresponds to.

**2. Why plain nonlinear least squares doesn't apply directly.** The usual
approach (`scipy.optimize.curve_fit`) requires known `(t_i, x_i, y_i)`
triples so it can compare `predicted(t_i)` against `(x_i, y_i)` row by row.
Without a `t` per row, that comparison isn't available.

**3. Correspondence-free fitting.** Instead of matching by row, we match by
geometry:
- For a candidate `(theta, M, X)`, densely sample the candidate curve over
  `t in [6, 60]` (6000 points).
- Build a KD-tree over those candidate-curve points.
- For every data point, find the distance to its *nearest neighbor* on the
  candidate curve — this doesn't require knowing which `t` produced that
  data point, only that it lies close to *some* point on the curve.
- Loss = mean squared nearest-neighbor distance, minimized over
  `(theta, M, X)` within the given bounds.

**4. Optimization.** The `sin(0.3t)` term makes the loss landscape
oscillatory/non-convex, so a purely local optimizer risks getting stuck.
We run:
- `scipy.optimize.differential_evolution` (global, bounded) to find the
  right basin,
- then `Nelder-Mead` to polish the result to high precision.

**5. Validation.**
- Mean nearest-point distance ≈ 0.0026, max ≈ 0.0078, on a curve spanning
  roughly 50 units in each axis — essentially numerical noise.
- The overlay plot (`assets/fit_overlay.png`) shows the fitted curve tracks
  the data almost exactly.
- The optimizer converges to clean values (30, 0.03, 55) rather than messy
  decimals, which is a strong indicator these are the true generating
  parameters rather than a noisy local fit.

## Repo structure

```
.
├── data/xy_data.csv       # provided data
├── fit_curve.py           # fitting logic (run this to reproduce the answer)
├── plot_fit.py            # regenerates assets/*.png
├── assets/
│   ├── raw_scatter.png    # raw data, shows a single smooth curve
│   └── fit_overlay.png    # fitted curve overlaid on data
├── requirements.txt
└── README.md
```

## Reproduce

```bash
pip install -r requirements.txt
python fit_curve.py   # prints fitted theta, M, X and fit-quality metrics
python plot_fit.py    # regenerates the plots in assets/
```

## Limitations / things I'd do with more time

- The nearest-neighbor loss assumes the dense template resolution
  (6000 points over t∈[6,60]) is fine enough that no data point is closer
  to a "gap" in the template than to the true curve; this was checked by
  confirming residuals stay near the floating-point noise level.
- A two-sided distance (data→curve AND curve→data, i.e. full Chamfer
  distance) would guard against a curve that "hides" between sparse
  regions of data; here it wasn't necessary since the fit already
  converges to a near-perfect residual with the one-sided version.
