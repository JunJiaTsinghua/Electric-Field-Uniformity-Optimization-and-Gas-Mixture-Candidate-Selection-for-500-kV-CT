"""Gas-mixture properties, scenario scores and candidate evaluation.

Design order: r_hv_eq_mm, d_mm, h_m, l_m_mm, alpha_N2, alpha_O2, p_MPa.
The electric-field response is the finite-resolution equivalent-geometry model.
"""
from __future__ import annotations
import math
import json
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
_CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config/scoring.json').read_text(encoding='utf-8'))
T0_K = float(_CONFIG['reference_temperature_K'])
P0_MPA = float(_CONFIG['reference_pressure_MPa'])
VALIDATION_RMSE = float(_CONFIG['validation_RMSE_kV_per_mm'])
GAS = _CONFIG['gas']
VARIABLE_BOUNDS = _CONFIG['variable_bounds']
INTERNAL_WEIGHTS = _CONFIG['internal_weights']
NORMALIZATION_BOUNDS = _CONFIG['normalization']['bounds']
R_GAS = 8.314462618

def clip_norm(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError('Normalization upper bound must exceed lower bound')
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

def interaction_phi(property_i: float, property_j: float, mass_i: float, mass_j: float) -> float:
    numerator = (1.0 + math.sqrt(property_i / property_j) * (mass_j / mass_i) ** 0.25) ** 2
    denominator = math.sqrt(8.0 * (1.0 + mass_i / mass_j))
    return numerator / denominator

def mixture_property_wilke(fractions: np.ndarray, values: np.ndarray, masses: np.ndarray) -> float:
    total = 0.0
    for i in range(3):
        denominator = 0.0
        for j in range(3):
            denominator += fractions[j] * interaction_phi(values[i], values[j], masses[i], masses[j])
        total += fractions[i] * values[i] / denominator
    return float(total)

def mixture_properties(alpha_n2: float, alpha_o2: float, alpha_he: float, p_mpa: float) -> dict[str, float]:
    fractions = np.array([alpha_n2, alpha_o2, alpha_he], dtype=float)
    masses_g = np.array([GAS[g]['molar_mass_g_per_mol'] for g in ('N2', 'O2', 'He')], dtype=float)
    conductivities = np.array([GAS[g]['thermal_conductivity_W_per_mK_300K'] for g in ('N2', 'O2', 'He')], dtype=float)
    viscosities = np.array([GAS[g]['dynamic_viscosity_Pa_s_300K'] for g in ('N2', 'O2', 'He')], dtype=float)
    cp_molar = np.array([GAS[g]['molar_cp_J_per_molK_300K'] for g in ('N2', 'O2', 'He')], dtype=float)
    molar_mass_kg = float(np.dot(fractions, masses_g) / 1000.0)
    k_mix = mixture_property_wilke(fractions, conductivities, masses_g)
    mu_mix = mixture_property_wilke(fractions, viscosities, masses_g)
    cp_mass = float(np.dot(fractions, cp_molar) / molar_mass_kg)
    rho = float(p_mpa * 1000000.0 * molar_mass_kg / (R_GAS * T0_K))
    convection_indicator = rho * rho * cp_mass / (mu_mix * k_mix)
    return {'molar_mass_kg_per_mol': molar_mass_kg, 'lambda_mix_W_per_mK': k_mix, 'mu_mix_Pa_s': mu_mix, 'cp_mix_J_per_kgK': cp_mass, 'rho_kg_per_m3': rho, 'natural_convection_indicator': convection_indicator}

def feasible_gas_grid() -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for n2 in np.linspace(0.4, 0.85, 91):
        for o2 in np.linspace(0.1, 0.25, 31):
            he = 1.0 - n2 - o2
            if -1e-12 <= he <= 0.5 + 1e-12:
                for pressure in (0.4, 0.8):
                    rows.append((float(n2), float(o2), float(max(0.0, he)), pressure))
    return rows

def calibrate_bounds() -> dict[str, list[float]]:
    raw: dict[str, list[float]] = {'lambda_mix_W_per_mK': [], 'natural_convection_indicator': [], 'O3_rate_proxy': [], 'NOx_rate_proxy': [], 'He_tracer_proxy': [], 'Kgas': [], 'Kseal': [], 'Kmfg': []}
    for n2, o2, he, pressure in feasible_gas_grid():
        props = mixture_properties(n2, o2, he, pressure)
        raw['lambda_mix_W_per_mK'].append(props['lambda_mix_W_per_mK'])
        raw['natural_convection_indicator'].append(props['natural_convection_indicator'])
        raw['O3_rate_proxy'].append(o2 * pressure)
        raw['NOx_rate_proxy'].append(n2 * o2 * pressure)
        raw['He_tracer_proxy'].append(he * math.sqrt(28.0134 / 4.002602) / pressure)
        raw['Kgas'].append(n2 + 1.2 * o2 + 25.0 * he)
        raw['Kseal'].append(pressure / 0.4 + 5.0 * he)
        raw['Kmfg'].append(1.0 + 2.0 * he + 0.5 * abs(o2 - 0.21) / 0.11)
    bounds = {name: [float(min(values)), float(max(values))] for name, values in raw.items()}
    bounds.update({'ME': [1.0, 5.5], 'MPD': [1.1, 2.675], 'Mimp': [1.1, 2.45], 'GWP100': [0.0, 23500.0], 'oxidizer_handling_index': [0.0, 1.0]})
    return bounds

@dataclass(frozen=True)
class Scenario:
    name: str = 'baseline'
    he_ebd0_kv_per_mm: float = 0.9
    he_cost_multiplier: float = 25.0
    temperature_k: float = T0_K

def evaluate_design(design: np.ndarray, artifact: dict, scenario: Scenario=Scenario('baseline')) -> dict[str, float]:
    rhv, gap, hm, lm, n2, o2, pressure = [float(value) for value in design]
    he = 1.0 - n2 - o2
    x_predict = np.array([[rhv, gap, hm, lm]], dtype=float)
    emax = float(artifact['gaussian_process'].predict(artifact['input_scaler'].transform(x_predict))[0])
    ebd0 = 3.0 * (n2 + o2) + scenario.he_ebd0_kv_per_mm * he
    ebd = ebd0 * (pressure / P0_MPA) * (T0_K / scenario.temperature_k)
    me = ebd / emax
    mpd = 0.75 + 0.35 * me
    mimp = 0.8 + 0.3 * me
    s_ins = (clip_norm(me, *NORMALIZATION_BOUNDS['ME']) + clip_norm(mpd, *NORMALIZATION_BOUNDS['MPD']) + clip_norm(mimp, *NORMALIZATION_BOUNDS['Mimp'])) / 3.0
    props = mixture_properties(n2, o2, he, pressure)
    n_lambda = clip_norm(props['lambda_mix_W_per_mK'], *NORMALIZATION_BOUNDS['lambda_mix_W_per_mK'])
    n_mt = clip_norm(props['natural_convection_indicator'], *NORMALIZATION_BOUNDS['natural_convection_indicator'])
    s_th = n_lambda * n_mt
    o3_rate = o2 * pressure
    nox_rate = n2 * o2 * pressure
    he_tracer = he * math.sqrt(28.0134 / 4.002602) / pressure
    s_det = INTERNAL_WEIGHTS['Eq17_b']['O3_rate'] * clip_norm(o3_rate, *NORMALIZATION_BOUNDS['O3_rate_proxy']) + INTERNAL_WEIGHTS['Eq17_b']['NOx_rate'] * clip_norm(nox_rate, *NORMALIZATION_BOUNDS['NOx_rate_proxy']) + INTERNAL_WEIGHTS['Eq17_b']['He_tracer'] * clip_norm(he_tracer, *NORMALIZATION_BOUNDS['He_tracer_proxy'])
    gwp100 = 0.0
    oxidizer_index = float(np.clip((o2 - 0.21) / 0.04, 0.0, 1.0))
    s_env = 1.0 - clip_norm(gwp100, *NORMALIZATION_BOUNDS['GWP100']) - INTERNAL_WEIGHTS['Eq18_gamma'] * clip_norm(oxidizer_index, *NORMALIZATION_BOUNDS['oxidizer_handling_index'])
    kgas = n2 + 1.2 * o2 + scenario.he_cost_multiplier * he
    kseal = pressure / 0.4 + 5.0 * he
    kmfg = 1.0 + 2.0 * he + 0.5 * abs(o2 - 0.21) / 0.11
    s_cost = INTERNAL_WEIGHTS['Eq19_c']['gas'] * clip_norm(kgas, *NORMALIZATION_BOUNDS['Kgas']) + INTERNAL_WEIGHTS['Eq19_c']['sealing'] * clip_norm(kseal, *NORMALIZATION_BOUNDS['Kseal']) + INTERNAL_WEIGHTS['Eq19_c']['manufacturing'] * clip_norm(kmfg, *NORMALIZATION_BOUNDS['Kmfg'])
    return {'r_hv_eq_mm': rhv, 'd_mm': gap, 'h_m': hm, 'l_m_mm': lm, 'alpha_N2': n2, 'alpha_O2': o2, 'alpha_He': he, 'p_MPa': pressure, 'Emax_pred_kV_per_mm': emax, 'Ebd0_kV_per_mm': ebd0, 'Ebd_kV_per_mm': ebd, 'M_E': me, 'M_PD': mpd, 'M_imp': mimp, 'lambda_mix_W_per_mK': props['lambda_mix_W_per_mK'], 'natural_convection_indicator': props['natural_convection_indicator'], 'O3_rate_proxy': o3_rate, 'NOx_rate_proxy': nox_rate, 'He_tracer_proxy': he_tracer, 'GWP100': gwp100, 'oxidizer_handling_index': oxidizer_index, 'Kgas': kgas, 'Kseal': kseal, 'Kmfg': kmfg, 'S_ins': s_ins, 'S_th': s_th, 'S_det': s_det, 'S_env': s_env, 'S_cost': s_cost}

def farthest_point_subset(objectives: np.ndarray, count: int) -> np.ndarray:
    if len(objectives) <= count:
        return np.arange(len(objectives), dtype=int)
    minimum = objectives.min(axis=0)
    span = np.where(objectives.max(axis=0) - minimum > 0.0, objectives.max(axis=0) - minimum, 1.0)
    normalized = (objectives - minimum) / span
    selected = [int(np.argmin(normalized.sum(axis=1))), int(np.argmax(normalized.sum(axis=1)))]
    selected = list(dict.fromkeys(selected))
    while len(selected) < count:
        distances = np.full(len(normalized), np.inf)
        for chosen in selected:
            distances = np.minimum(distances, np.linalg.norm(normalized - normalized[chosen], axis=1))
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return np.array(selected, dtype=int)

def ahp_weights() -> tuple[np.ndarray, float, float, float]:
    matrix = np.array([[1, 2, 2, 1, 4], [1 / 2, 1, 1, 1, 2], [1 / 2, 1, 1, 1 / 2, 2], [1, 1, 2, 1, 3], [1 / 4, 1 / 2, 1 / 2, 1 / 3, 1]], dtype=float)
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmax(values.real))
    lambda_max = float(values[index].real)
    weights = np.abs(vectors[:, index].real)
    weights = weights / weights.sum()
    ci = (lambda_max - 5.0) / 4.0
    cr = ci / 1.12
    return (weights, lambda_max, ci, cr)

def entropy_weights(benefits: np.ndarray) -> np.ndarray:
    minimum = benefits.min(axis=0, keepdims=True)
    span = benefits.max(axis=0, keepdims=True) - minimum
    span = np.where(span > 1e-12, span, 1.0)
    shifted = np.clip((benefits - minimum) / span, 0.0, 1.0) + 1e-12
    probabilities = shifted / shifted.sum(axis=0, keepdims=True)
    entropy = -(probabilities * np.log(probabilities)).sum(axis=0) / math.log(len(benefits))
    divergence = 1.0 - entropy
    return divergence / divergence.sum()
DESIGN_COLUMNS = ['r_hv_eq_mm', 'd_mm', 'h_m', 'l_m_mm', 'alpha_N2', 'alpha_O2', 'p_MPa']
CRITERIA = ['dielectric', 'thermal', 'detection', 'environment', 'cost_benefit']

class CandidateEvaluator:
    """Load the supplied Gaussian-process artifact once for repeated scoring."""

    def __init__(self, model_path: str | Path):
        self.artifact = joblib.load(Path(model_path))
        expected = ['r_hv_eq_mm', 'd_mm', 'h_m', 'l_m_mm']
        if list(self.artifact['features']) != expected:
            raise ValueError('Unexpected surrogate feature order')

    def evaluate(self, design, scenario: Scenario=Scenario()):
        values = np.asarray(design, dtype=float)
        if values.shape != (7,):
            raise ValueError('A candidate must contain the seven ordered design variables')
        return evaluate_design(values, self.artifact, scenario)
    __call__ = evaluate

def benefit_vector(evaluated: dict) -> np.ndarray:
    return np.asarray([evaluated[k] for k in ('S_ins', 'S_th', 'S_det', 'S_env')] + [1.0 - evaluated['S_cost']], dtype=float)
