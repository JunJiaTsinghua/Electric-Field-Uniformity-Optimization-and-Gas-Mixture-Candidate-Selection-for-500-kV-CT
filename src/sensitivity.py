"""Sobol indices of the surrogate mean under independent uniform inputs."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .io_utils import output_directory, write_json
from .surrogate import FEATURES
BOUNDS = np.array([[50.,90.],[80.,160.],[.30,.65],[30.,90.]])
def integrate_keep(array: np.ndarray, weights: np.ndarray, keep: tuple[int, ...]=()) -> np.ndarray:
    """Integrate product uniform measure; descending axes preserve indices."""
    out = array
    for axis in reversed(range(array.ndim)):
        if axis not in keep:
            out = np.tensordot(out, weights, axes=([axis], [0]))
    return out

def sobol_from_tensor(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    mean = float(integrate_keep(values, weights))
    centered = values - mean
    variance = float(integrate_keep(centered * centered, weights))
    if variance <= 0:
        raise ValueError('Sobol variance must be positive')
    first, total = ([], [])
    for axis in range(values.ndim):
        conditional_i = integrate_keep(centered, weights, (axis,))
        first.append(float(np.dot(weights, conditional_i ** 2)) / variance)
        others = tuple((j for j in range(values.ndim) if j != axis))
        conditional_others = integrate_keep(centered, weights, others)
        total.append(1.0 - float(integrate_keep(conditional_others ** 2, weights)) / variance)
    return (mean, variance, np.array(first), np.array(total))

def check_sobol_integrator() -> None:
    """Analytic nontrivial ANOVA check: additive plus X1*X2 interaction."""
    nodes, weights = np.polynomial.legendre.leggauss(5)
    weights = weights / 2.0
    x = np.meshgrid(nodes, nodes, nodes, nodes, indexing='ij')
    values = x[0] + 2 * x[1] + x[0] * x[1]
    _, variance, first, total = sobol_from_tensor(values, weights)
    np.testing.assert_allclose(variance, 16 / 9, atol=1e-12)
    np.testing.assert_allclose(first, [3 / 16, 12 / 16, 0, 0], atol=1e-12)
    np.testing.assert_allclose(total, [4 / 16, 13 / 16, 0, 0], atol=1e-12)

def quadrature(artifact: dict, order: int, batch_size: int) -> dict:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    weights = weights / 2.0
    coordinates = [lo + (nodes + 1.0) * (hi - lo) / 2.0 for lo, hi in BOUNDS]
    grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1).reshape(-1, 4)
    predictions = np.empty(len(grid))
    gp = artifact['gaussian_process']
    scaler = artifact['input_scaler']
    for start in range(0, len(grid), batch_size):
        stop = min(start + batch_size, len(grid))
        predictions[start:stop] = gp.predict(scaler.transform(grid[start:stop]))
    mean, variance, first, total = sobol_from_tensor(predictions.reshape((order,) * 4), weights)
    return {'order': order, 'evaluations': int(len(grid)), 'mean': mean, 'variance': variance, 'first': first, 'total': total}
def run_sensitivity(root, out):
    root = Path(root)
    destination = output_directory(root, out, "sensitivity")
    config = json.loads((root / "config/analysis.json").read_text(encoding="utf-8"))["sobol"]
    artifact = joblib.load(root / "models/surrogate/Kriging_Emax_model.joblib")
    if list(artifact["features"]) != FEATURES:
        raise ValueError("Unexpected surrogate input order")
    check_sobol_integrator()
    rows, integrals = [], []
    with threadpool_limits(limits=1):
        for order in config["orders"]:
            result = quadrature(artifact, int(order), int(config["batch_size"]))
            integrals.append({k: v for k,v in result.items() if k not in ("first","total")})
            for i, feature in enumerate(FEATURES):
                rows.append({"order": order, "parameter": feature,
                             "lower_bound": BOUNDS[i,0], "upper_bound": BOUNDS[i,1],
                             "S1": float(result["first"][i]), "ST": float(result["total"][i])})
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "sobol_indices.csv", index=False)
    first = frame[frame.order == config["orders"][0]]
    last = frame[frame.order == config["orders"][-1]]
    summary = {"method": "tensor-product Gauss-Legendre quadrature of surrogate mean",
               "input_measure": config["measure"], "bounds": BOUNDS.tolist(),
               "orders": config["orders"], "integrals": integrals,
               "maximum_index_change": float(np.abs(last[["S1","ST"]].to_numpy() -
                                                     first[["S1","ST"]].to_numpy()).max()),
               "sum_first_order": float(last.S1.sum()),
               "maximum_total_minus_first_order": float((last.ST - last.S1).max()),
               "scope": "The rectangular input domain includes points outside the finite training design convex hull."}
    write_json(destination / "sobol_summary.json", summary)
    return summary
