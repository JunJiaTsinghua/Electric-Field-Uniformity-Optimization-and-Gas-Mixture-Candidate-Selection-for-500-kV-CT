"""AHP/entropy decisions, weight perturbations and fixed-geometry helium scan."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from threadpoolctl import threadpool_limits
from .io_utils import output_directory, write_json
from .scoring import CandidateEvaluator, Scenario, DESIGN_COLUMNS, CRITERIA, benefit_vector, mixture_properties, evaluate_design
AHP_MATRIX = np.array([[1,2,2,1,4],[.5,1,1,1,2],[.5,1,1,.5,2],[1,1,2,1,3],[.25,.5,.5,1/3,1]],float)
def ahp_and_entropy(benefits: np.ndarray, epsilon: float=1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    eigenvalues, eigenvectors = np.linalg.eig(AHP_MATRIX)
    index = int(np.argmax(eigenvalues.real))
    largest = float(eigenvalues[index].real)
    subjective = np.abs(eigenvectors[:, index].real)
    subjective /= subjective.sum()
    span = np.ptp(benefits, axis=0)
    span = np.where(span > 1e-12, span, 1.0)
    shifted = np.clip((benefits - benefits.min(axis=0)) / span, 0, 1) + epsilon
    proportions = shifted / shifted.sum(axis=0)
    entropy = -(proportions * np.log(proportions)).sum(axis=0) / np.log(len(benefits))
    divergence = 1.0 - entropy
    objective = divergence / divergence.sum()
    combined = subjective * objective
    combined /= combined.sum()
    return (subjective, objective, combined, {'lambda_max': largest, 'CI': (largest - 5.0) / 4.0, 'RI': 1.12, 'CR': (largest - 5.0) / 4.0 / 1.12, 'entropy_epsilon': epsilon})

def analyze_rankings(source: Path, output: Path, seed: int, draws: int, concentration: float) -> dict:
    frame = pd.read_csv(source / 'Pareto_67_complete.csv')
    if len(frame) != 67 or frame.candidate_id.nunique() != 67:
        raise ValueError('Expected the complete 67-candidate matrix')
    benefits = frame[['S_ins', 'S_th', 'S_det', 'S_env']].to_numpy(float)
    benefits = np.column_stack([benefits, 1.0 - frame.S_cost.to_numpy(float)])
    subjective, objective, combined, ahp = ahp_and_entropy(benefits)
    scores = benefits @ combined
    baseline = rankdata(-scores, method='min')
    distances = np.sqrt(((1.0 - benefits) ** 2 * combined).sum(axis=1))
    ids = frame.candidate_id.to_numpy()
    leader = int(np.argmax(scores))
    baseline_top5 = set(np.argsort(-scores, kind='stable')[:5])
    candidates = pd.DataFrame({'candidate_id': ids, 'J_recomputed': scores, 'rank_J_recomputed': baseline.astype(int), 'ideal_distance_recomputed': distances, 'rank_ideal_recomputed': rankdata(distances, method='min').astype(int)})
    candidates.to_csv(output / 'candidate_ranking_recomputed.csv', index=False)
    pd.DataFrame({'criterion': CRITERIA, 'AHP': subjective, 'entropy': objective, 'combined': combined}).to_csv(output / 'weights_recomputed.csv', index=False)
    alternative_scores = {'AHP subjective weights only': benefits @ subjective, 'Entropy objective weights only': benefits @ objective, 'Equal weights': benefits @ np.full(5, 0.2), 'Minimum weighted distance to ideal': -distances}
    rows = []
    for name, other_score in alternative_scores.items():
        ranks = rankdata(-other_score, method='min')
        top5 = set(np.argsort(-other_score, kind='stable')[:5])
        rows.append({'decision_rule': name, 'spearman_rho': float(spearmanr(baseline, ranks).statistic), 'kendall_tau': float(kendalltau(baseline, ranks).statistic), 'rank_of_baseline_leader': int(ranks[leader]), 'selected_candidate_id': str(ids[int(np.argmax(other_score))]), 'top5_overlap': len(baseline_top5.intersection(top5))})
    pd.DataFrame(rows).to_csv(output / 'alternative_decision_rules.csv', index=False)
    rng = np.random.default_rng(seed)
    perturbed = rng.dirichlet(concentration * combined, size=draws)
    perturbed_scores = benefits @ perturbed.T
    perturbed_ranks = rankdata(-perturbed_scores, axis=0, method='min')
    base_centered = baseline - baseline.mean()
    draw_centered = perturbed_ranks - perturbed_ranks.mean(axis=0)
    rho = base_centered @ draw_centered / np.sqrt(np.sum(base_centered ** 2) * np.sum(draw_centered ** 2, axis=0))
    lead_ranks = perturbed_ranks[leader]
    trial_frame = pd.DataFrame(perturbed, columns=[f'weight_{name}' for name in CRITERIA])
    trial_frame.insert(0, 'draw', np.arange(1, draws + 1))
    trial_frame['spearman_rho'] = rho
    trial_frame['rank_of_baseline_leader'] = lead_ranks.astype(int)
    trial_frame['selected_candidate_id'] = ids[np.argmax(perturbed_scores, axis=0)]
    trial_frame.to_csv(output / 'dirichlet_weight_draws.csv', index=False)
    rank_freq = np.bincount(lead_ranks.astype(int), minlength=len(frame) + 1)[1:]
    pd.DataFrame({'rank': np.arange(1, len(frame) + 1), 'count': rank_freq, 'proportion': rank_freq / draws}).to_csv(output / 'baseline_leader_rank_frequency.csv', index=False)
    weight_errors = []
    for epsilon in [1e-12, 1e-06, 0.0001]:
        _, alt_objective, _, _ = ahp_and_entropy(benefits, epsilon)
        weight_errors.append({'epsilon': epsilon, 'maximum_absolute_change_from_1e_12': float(np.max(np.abs(alt_objective - objective)))})
    original_score_error = float(np.max(np.abs(scores - frame.J_combined.to_numpy())))
    if original_score_error > 1e-10:
        raise AssertionError(f'Candidate score reproduction mismatch: {original_score_error}')
    return {'AHP': ahp, 'subjective_weights': subjective.tolist(), 'objective_weights': objective.tolist(), 'combined_weights': combined.tolist(), 'max_score_error_against_archive': original_score_error, 'baseline_leader': str(ids[leader]), 'baseline_leader_score': float(scores[leader]), 'alternative_decision_rules': rows, 'entropy_epsilon_check': weight_errors, 'Dirichlet': {'status': 'Seeded weight-sensitivity experiment', 'rng': 'numpy.random.default_rng PCG64', 'seed': seed, 'draws': draws, 'concentration': concentration, 'center': combined.tolist(), 'spearman_median': float(np.median(rho)), 'spearman_5th_percentile': float(np.quantile(rho, 0.05)), 'spearman_minimum': float(np.min(rho)), 'leader_first_place_fraction': float(np.mean(lead_ranks == 1)), 'leader_top3_fraction': float(np.mean(lead_ranks <= 3)), 'leader_worst_rank': int(np.max(lead_ranks)), 'quantile_method': 'NumPy linear', 'rank_ties': 'minimum rank; no artificial tie breaking'}}
def run_ranking(root, out):
    root = Path(root)
    destination = output_directory(root, out, "ranking")
    config = json.loads((root / "config/analysis.json").read_text(encoding="utf-8"))["ranking"]
    with threadpool_limits(limits=1):
        summary = analyze_rankings(root / "data/reference", destination, int(config["dirichlet_seed"]),
                                   int(config["draws"]), float(config["concentration"]))
    evaluator = CandidateEvaluator(root / "models/surrogate/Kriging_Emax_model.joblib")
    pareto = pd.read_csv(root / "data/reference/Pareto_67_complete.csv")
    combined = np.asarray(summary["combined_weights"])
    scenarios = [Scenario(), Scenario("conservative_He_breakdown", .5), Scenario("optimistic_He_breakdown",1.3),
                 Scenario("lower_He_cost",he_cost_multiplier=10), Scenario("higher_He_cost",he_cost_multiplier=50),
                 Scenario("cold_233K",temperature_k=233),Scenario("hot_333K",temperature_k=333)]
    records = []
    baseline_evaluated = None
    cold_evaluated = None
    def record_scores(name, kind, evaluated, weights):
        benefits = np.asarray([benefit_vector(e) for e in evaluated])
        scores = benefits @ weights
        ranks = rankdata(-scores, method="min").astype(int)
        for i, (e, score) in enumerate(zip(evaluated, scores)):
            records.append({"scenario": name, "scenario_type": kind,
                            "candidate_id": pareto.iloc[i].candidate_id,
                            "J": float(score), "M_E": e["M_E"],
                            "S_ins": e["S_ins"], "rank_J": int(ranks[i])})
    for scenario in scenarios:
        evaluated = [evaluator(row[DESIGN_COLUMNS].to_numpy(float),scenario) for _,row in pareto.iterrows()]
        record_scores(scenario.name, "physical_or_cost_input", evaluated, combined)
        if scenario.name == "baseline": baseline_evaluated = evaluated
        if scenario.name == "cold_233K": cold_evaluated = evaluated
    for name, weights in config["preference_weights"].items():
        weights = np.asarray(weights, float)
        if weights.shape != (5,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("Preference weights must contain five nonnegative values summing to one")
        record_scores(name, "objective_preference", baseline_evaluated, weights)
    for name, upper, clipped in [("cold_233K_no_clipping", 5.5, False),
                                  ("cold_233K_ME_upper_6p0", 6.0, True),
                                  ("cold_233K_ME_upper_6p5", 6.5, True)]:
        diagnostic = []
        for e in cold_evaluated:
            score = (e["M_E"] - 1.0) / (upper - 1.0)
            diagnostic.append({**e, "S_ins": float(np.clip(score, 0, 1) if clipped else score)})
        record_scores(name, "normalization_diagnostic", diagnostic, combined)
    pd.DataFrame(records).to_csv(destination / "scenario_candidate_scores.csv",index=False)
    names = [s.name for s in scenarios] + list(config["preference_weights"])
    summary["scenario_leaders"] = [min((r for r in records if r["scenario"]==name),key=lambda r:r["rank_J"]) for name in names]
    summary["cold_normalization"] = {
        "temperature_K": 233, "pressure_condition": "fixed operating pressure",
        "n_candidates_above_ME_5p5": int(sum(e["M_E"] > 5.5 for e in cold_evaluated)),
        "normalization": "Affine PD/impulse definitions give the same normalized margin; diagnostic upper bounds are changed consistently.",
        "comparison": [r for r in records if r["candidate_id"] in ("P01", "P21") and
                       (r["scenario"] == "cold_233K" or r["scenario_type"] == "normalization_diagnostic")]
    }
    write_json(destination / "ranking_summary.json", summary)
    return summary

class _FixedPrediction:
    def __init__(self, value): self.value = value
    def predict(self, x): return np.full(len(x), self.value)
    def transform(self, x): return x

def run_scan(root, out):
    root = Path(root)
    destination = output_directory(root, out, "scan")
    config = json.loads((root / "config/analysis.json").read_text(encoding="utf-8"))["scan"]
    pareto = pd.read_csv(root / "data/reference/Pareto_67_complete.csv")
    p01 = pareto[pareto.candidate_id=="P01"].iloc[0]
    weights = pd.read_csv(root / "data/reference/AHP_entropy_combined_weights.csv").combined_multiplicative_weight.to_numpy(float)
    fixed = _FixedPrediction(float(p01.Emax_pred_kV_per_mm))
    artifact = {"gaussian_process":fixed,"input_scaler":fixed}
    ratio = p01.alpha_N2/(p01.alpha_N2+p01.alpha_O2)
    def evaluate(he):
        n2, o2 = (1-he)*ratio, (1-he)*(1-ratio)
        x = np.array([p01.r_hv_eq_mm,p01.d_mm,p01.h_m,p01.l_m_mm,n2,o2,p01.p_MPa])
        e = evaluate_design(x,artifact)
        return {"He_pct":he*100,"alpha_N2":n2,"alpha_O2":o2,"lam":e["lambda_mix_W_per_mK"],
                "Ebd0":e["Ebd0_kV_per_mm"],"M_E":e["M_E"],
                **{k:e[k] for k in ("S_ins","S_th","S_det","S_env","S_cost")},
                "J":float(benefit_vector(e)@weights),
                "within_composition_bounds":bool(.4<=n2<=.85 and .1<=o2<=.25 and 0<=he<=.5)}
    tabulated = [evaluate(h/100) for h in config["He_tabulated_percent"]]
    for row in tabulated:
        for key,label in [("lam","lam_pct"),("M_E","ME_pct"),("J","J_pct")]:
            row[label] = 100*(row[key]/tabulated[0][key]-1)
    pd.DataFrame(tabulated).to_csv(destination / "controlled_He_scan.csv",index=False)
    dense = [evaluate(float(h)) for h in np.linspace(0,.5,int(config["dense_points"]))]
    pd.DataFrame(dense).to_csv(destination / "controlled_He_scan_dense.csv",index=False)
    feasible = [r for r in dense if r["within_composition_bounds"]]
    summary = {"fixed_candidate":"P01","fixed_Emax_kV_per_mm":float(p01.Emax_pred_kV_per_mm),
               "fixed_N2_to_N2plusO2_ratio":float(ratio),
               "dense_feasible_maximum":max(feasible,key=lambda r:r["J"]),
               "grid_step_He_percentage_points":50/(config["dense_points"]-1),
               "scope":"Fixed geometry, nominal pressure and N2:O2 ratio; 50% He is diagnostic and outside the N2 lower bound."}
    write_json(destination / "controlled_scan_summary.json",summary)
    return summary
