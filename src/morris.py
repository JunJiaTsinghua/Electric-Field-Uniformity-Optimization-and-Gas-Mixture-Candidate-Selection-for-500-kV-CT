"""Morris sensitivity analysis from saved finite-element samples."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class JavaRandom:
    """The java.util.Random generator used for the stored trajectory designs."""
    def __init__(self, seed):
        self.state = (int(seed) ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def bits(self, count):
        self.state = (self.state * 0x5DEECE66D + 11) & ((1 << 48) - 1)
        return self.state >> (48 - count)

    def integer(self, bound):
        if bound & -bound == bound:
            return (bound * self.bits(31)) >> 31
        while True:
            bits = self.bits(31)
            value = bits % bound
            if bits - value + bound - 1 < (1 << 31):
                return value


def seeded_design(seed, metadata):
    """Reconstruct the original input design without running COMSOL."""
    bounds = np.array(metadata["bounds"])
    variables = metadata["variables"]
    delta = metadata["normalized_step_magnitude"]
    random = JavaRandom(seed)
    rows = []
    for trajectory in range(1, metadata["trajectories"] + 1):
        coordinates, signs = [], []
        for _ in variables:
            signs.append(1 if random.bits(1) else -1)
            lower = random.integer(2) / 3
            coordinates.append(lower if signs[-1] > 0 else lower + delta)
        order = list(range(len(variables)))
        for i in range(len(order) - 1, 0, -1):
            j = random.integer(i + 1)
            order[i], order[j] = order[j], order[i]
        for point in range(1, len(variables) + 2):
            changed = -1 if point == 1 else order[point - 2]
            step = 0.0 if point == 1 else signs[changed] * delta
            if point > 1:
                coordinates[changed] += step
            values = bounds[:, 0] + np.array(coordinates) * np.diff(bounds, axis=1)[:, 0]
            rows.append(dict(trajectory=trajectory, point=point,
                             changed_variable_index=changed, normalized_step=step,
                             **dict(zip(variables, values))))
    return rows


def _effects(samples, seed, metadata):
    variables = metadata["variables"]
    expected = seeded_design(seed, metadata)
    if len(samples) != len(expected):
        raise ValueError(f"Seed {seed}: expected {len(expected)} sample rows")
    effects = np.full((metadata["trajectories"], len(variables)), np.nan)
    for index, (row, exp) in enumerate(zip(samples, expected)):
        for key in exp:
            if abs(float(row[key]) - float(exp[key])) > 1e-10:
                raise ValueError(f"Seed {seed}, row {index + 1}: input design mismatch in {key}")
        response = float(row["Emax_kV_per_mm"])
        if not np.isfinite(response):
            raise ValueError(f"Seed {seed}: nonfinite response")
        if exp["point"] > 1:
            j = exp["changed_variable_index"]
            before = samples[index - 1]
            changed = [k for k, variable in enumerate(variables)
                       if abs(float(row[variable]) - float(before[variable])) > 1e-10]
            if changed != [j]:
                raise ValueError(f"Seed {seed}: a trajectory step changes multiple variables")
            effects[exp["trajectory"] - 1, j] = (
                response - float(before["Emax_kV_per_mm"])) / exp["normalized_step"]
    if not np.isfinite(effects).all():
        raise ValueError(f"Seed {seed}: incomplete elementary effects")
    return effects


def run_morris(root, out, bootstrap=20000):
    """Recompute independent designs and trajectory bootstrap; save scientific outputs."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "data/morris/metadata.json").read_text(encoding="utf-8"))
    variables = metadata["variables"]
    threshold = metadata["mu_star_threshold_kV_per_mm"]
    seeds = metadata["design_seeds"]
    saved_samples = _read_csv(root / metadata["source_file"])
    effects_by_seed, statistics, effect_rows = {}, [], []
    for seed in seeds:
        samples = [r for r in saved_samples if int(r["design_seed"]) == seed]
        effects = _effects(samples, seed, metadata)
        effects_by_seed[seed] = effects
        mu_star = np.abs(effects).mean(axis=0)
        ranks = np.argsort(np.argsort(-mu_star)) + 1
        for j, variable in enumerate(variables):
            statistics.append(dict(seed=seed, variable=variable,
                mu_kV_per_mm=float(effects[:, j].mean()),
                mu_star_kV_per_mm=float(mu_star[j]),
                sigma_kV_per_mm=float(effects[:, j].std(ddof=1)),
                rank=int(ranks[j]), retained_top4=bool(ranks[j] <= 4),
                retained_threshold=bool(mu_star[j] >= threshold)))
        effect_rows.extend(dict(seed=seed, trajectory=t + 1, variable=variables[j],
                               elementary_effect_kV_per_mm=float(effects[t, j]))
                           for t in range(len(effects)) for j in range(len(variables)))
    _write_csv(out / "elementary_effects.csv", effect_rows)
    _write_csv(out / "design_statistics.csv", statistics)
    reference = effects_by_seed[metadata["reference_seed"]]
    original_mu = np.abs(reference).mean(axis=0)
    original_rank = np.argsort(np.argsort(-original_mu)) + 1
    original_top4 = set(np.argsort(-original_mu)[:4])
    rank_rows = []
    for seed in seeds:
        values = np.abs(effects_by_seed[seed]).mean(axis=0)
        order = np.argsort(-values)
        rank = np.argsort(order) + 1
        rank_rows.append(dict(seed=seed,
            is_reference=seed == metadata["reference_seed"],
            ranking=" > ".join(variables[j] for j in order),
            same_top4_as_reference=set(order[:4]) == original_top4,
            same_full_order_as_reference=bool(np.array_equal(rank, original_rank)),
            spearman_vs_reference=float(1 - 6 * np.sum((rank - original_rank)**2) /
                                         (len(variables) * (len(variables)**2 - 1))),
            threshold_retained_count=int(np.sum(values >= threshold)),
            threshold_variables=";".join(variables[j] for j in order if values[j] >= threshold)))
    _write_csv(out / "ranking_stability.csv", rank_rows)
    random = np.random.default_rng(metadata["bootstrap_seed"])
    indices = random.integers(0, len(reference), size=(bootstrap, len(reference)))
    bootstrap_mu = np.abs(reference[indices]).mean(axis=1)
    bootstrap_rank = np.argsort(np.argsort(-bootstrap_mu, axis=1), axis=1) + 1
    summary = []
    d_index = variables.index("d_mm")
    for j, variable in enumerate(variables):
        observed = [r["mu_star_kV_per_mm"] for r in statistics if r["variable"] == variable]
        gap_by_seed={r['seed']:r['mu_star_kV_per_mm'] for r in statistics if r['variable']=='d_mm'}
        same_design_ratios=[r['mu_star_kV_per_mm']/gap_by_seed[r['seed']]
                            for r in statistics if r['variable']==variable]
        low, high = np.quantile(bootstrap_mu[:, j], [.025, .975])
        summary.append(dict(variable=variable, reference_mu_star_kV_per_mm=float(original_mu[j]),
            independent_design_min_mu_star_kV_per_mm=float(min(observed)),
            independent_design_max_mu_star_kV_per_mm=float(max(observed)),
            maximum_same_design_ratio_to_gap=float(max(same_design_ratios)),
            bootstrap_ci025_kV_per_mm=float(low), bootstrap_ci975_kV_per_mm=float(high),
            bootstrap_top4_frequency=float(np.mean(bootstrap_rank[:, j] <= 4)),
            bootstrap_exceeds_0_10_frequency=float(np.mean(bootstrap_mu[:, j] >= threshold))))
    _write_csv(out / "variable_ranges_and_bootstrap.csv", summary)
    jackknife = []
    for t in range(len(reference)):
        mu = np.abs(np.delete(reference, t, axis=0)).mean(axis=0)
        order = np.argsort(-mu)
        jackknife.append(dict(omitted_trajectory=t + 1,
            ranking=" > ".join(variables[j] for j in order),
            same_top4_as_reference=set(order[:4]) == original_top4,
            **dict(zip(variables, map(float, mu)))))
    _write_csv(out / "leave_one_trajectory_out.csv", jackknife)
    same_top4 = np.all((bootstrap_rank <= 4) == (original_rank <= 4), axis=1)
    same_threshold_set=np.all((bootstrap_mu>=threshold)==(original_mu>=threshold),axis=1)
    rho = 1 - 6 * np.sum((bootstrap_rank - original_rank)**2, axis=1) / (7 * (7**2 - 1))
    result = dict(design_count=len(seeds), stored_sample_rows=len(seeds)*len(reference)*8,
        additional_independent_designs=len(seeds) - 1, new_finite_element_solves=0,
        response=metadata["response"], bootstrap_replicates=int(bootstrap),
        bootstrap_seed=metadata["bootstrap_seed"], bootstrap_unit="complete trajectory",
        bootstrap_same_top4_frequency=float(same_top4.mean()),
        bootstrap_same_threshold_set_frequency=float(same_threshold_set.mean()),
        bootstrap_same_full_order_frequency=float(np.mean(np.all(bootstrap_rank == original_rank, axis=1))),
        bootstrap_spearman_median=float(np.median(rho)),
        bootstrap_spearman_ci025=float(np.quantile(rho, .025)),
        bootstrap_spearman_ci975=float(np.quantile(rho, .975)),
        independent_same_top4_count=sum(r["same_top4_as_reference"] for r in rank_rows[1:]),
        independent_same_full_order_count=sum(r["same_full_order_as_reference"] for r in rank_rows[1:]),
        independent_exact_four_variable_threshold_count=sum(
            set(r["threshold_variables"].split(";")) == {variables[j] for j in original_top4}
            for r in rank_rows[1:]),
        leave_one_out_same_top4_count=sum(r["same_top4_as_reference"] for r in jackknife),
        geometry_scope=metadata["geometry_scope"])
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = np.array([r["reference_mu_star_kV_per_mm"] for r in summary])
    lows = np.array([r["independent_design_min_mu_star_kV_per_mm"] for r in summary])
    highs = np.array([r["independent_design_max_mu_star_kV_per_mm"] for r in summary])
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.errorbar(range(7), means, yerr=np.vstack([means-lows, highs-means]), fmt="o", capsize=4,
                  label="Reference estimate and range across 11 designs")
    axis.axhline(threshold, color="grey", linestyle="--", label=r"$\mu^*=0.10$ kV/mm")
    axis.set_xticks(range(7), [r"$r_{hv,eq}$", "$d$", r"$r_{g,eq}$", "$l_m$", "$h_m$", r"$l_{s,eq}$", r"$\theta_s$"])
    axis.set_ylabel(r"$\mu^*$ (kV/mm)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "morris_design_ranges.png", dpi=300)
    plt.close(figure)
    return result
