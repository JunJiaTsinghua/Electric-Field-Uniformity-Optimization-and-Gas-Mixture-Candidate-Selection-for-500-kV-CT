"""Validate or refit the four-input Gaussian-process electric-field surrogate."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .io_utils import output_directory, write_json

FEATURES = ["r_hv_eq_mm", "d_mm", "h_m", "l_m_mm"]
TARGET = "Emax_kV_per_mm"


def metrics(observed, predicted):
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    return {"r2": float(r2_score(observed, predicted)), "rmse_kV_per_mm": rmse,
            "normalized_rmse_percent_of_validation_range": float(100 * rmse / np.ptp(observed)),
            "mae_kV_per_mm": float(mean_absolute_error(observed, predicted)),
            "max_abs_error_kV_per_mm": float(np.max(np.abs(observed - predicted)))}


def fit_surrogate(training):
    x, y = training[FEATURES].to_numpy(float), training[TARGET].to_numpy(float)
    scaler = StandardScaler().fit(x)
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e3), nu=2.5) +
              WhiteKernel(noise_level=1e-7, noise_level_bounds=(1e-12, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True,
                                  n_restarts_optimizer=8, random_state=20260812)
    gp.fit(scaler.transform(x), y)
    return {"input_scaler": scaler, "gaussian_process": gp, "features": FEATURES,
            "target": TARGET, "training_seed": 20260812, "validation_seed": 20260813,
            "comsol_version": "6.2.0.290", "mesh_level": 5,
            "geometry_scope": "simplified 2D axisymmetric verification baseline"}


def run_surrogate(root, out, refit=False):
    root = Path(root)
    destination = output_directory(root, out, "surrogate")
    training = pd.read_csv(root / "data/surrogate/Kriging_training_100_COMSOL.csv")
    validation = pd.read_csv(root / "data/surrogate/Kriging_validation_30_COMSOL.csv")
    if len(training) != 100 or len(validation) != 30:
        raise ValueError("Expected 100 training and 30 independent validation records")
    with threadpool_limits(limits=1):
        artifact = (fit_surrogate(training) if refit else
                    joblib.load(root / "models/surrogate/Kriging_Emax_model.joblib"))
        gp, scaler = artifact["gaussian_process"], artifact["input_scaler"]
        if list(artifact["features"]) != FEATURES:
            raise ValueError("Unexpected surrogate input order")
        predicted, std = gp.predict(scaler.transform(validation[FEATURES].to_numpy(float)), return_std=True)
        observed = validation[TARGET].to_numpy(float)
        inverse_l = solve_triangular(gp.L_, np.eye(len(training)), lower=True)
        inverse_diag = (inverse_l * inverse_l).sum(axis=0)
        y_train = training[TARGET].to_numpy(float)
        mean, scale = float(gp._y_train_mean), float(gp._y_train_std)
        normalized = (y_train - mean) / scale
        loo = (normalized - gp.alpha_ / inverse_diag) * scale + mean
    records = validation.copy()
    records["Emax_predicted_kV_per_mm"] = predicted
    records["prediction_std_kV_per_mm"] = std
    records["residual_kV_per_mm"] = predicted - observed
    records["absolute_error_kV_per_mm"] = np.abs(predicted - observed)
    records["relative_error_percent"] = 100 * (predicted - observed) / observed
    records.to_csv(destination / "validation_predictions.csv", index=False)
    summary = {"training_rows": len(training), "independent_validation_rows": len(validation),
               "leave_one_out_training": metrics(y_train, loo),
               "independent_validation": metrics(observed, predicted),
               "observed_Emax_range_kV_per_mm": [float(observed.min()), float(observed.max())],
               "kernel_log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
               "optimized_kernel": str(gp.kernel_), "refitted_from_csv": bool(refit)}
    write_json(destination / "validation_metrics.json", summary)
    if refit:
        joblib.dump(artifact, destination / "Kriging_Emax_model.joblib")
    return summary
