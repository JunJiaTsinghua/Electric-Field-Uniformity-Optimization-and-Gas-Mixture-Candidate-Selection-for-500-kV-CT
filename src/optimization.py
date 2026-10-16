"""Saved-run analysis and optional NSGA-II search for the gas-CT screening model."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import moocore
import numpy as np
import pandas as pd

from .scoring import CandidateEvaluator, Scenario, VARIABLE_BOUNDS

DESIGN = ['r_hv_eq_mm', 'd_mm', 'h_m', 'l_m_mm', 'alpha_N2', 'alpha_O2', 'p_MPa']
SCORES = ['S_ins', 'S_th', 'S_det', 'S_env', 'S_cost']
SHIFT = np.array([1., 1., 1., 1., 0.])


def _read(path):
    return pd.read_csv(path, float_precision='round_trip')


def _settings(root):
    root = Path(root).resolve()
    config = json.loads((root / 'config/optimization.json').read_text(encoding='utf-8'))
    weights = _read(root / config['fixed_weight_file'])['combined_multiplicative_weight'].to_numpy()
    return root, config, weights


def costs(frame):
    """Fixed minimization coordinates, without run-specific rescaling."""
    return np.column_stack([1. - frame[SCORES[:4]].to_numpy(), frame.S_cost.to_numpy()])


def _hv(points, reference):
    return float(moocore.hypervolume(points, ref=reference)) if len(points) else 0.


def _igd_plus(approximation, reference):
    delta = np.maximum(approximation[None, :, :] - reference[:, None, :], 0.)
    return float(np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1).mean())


def _ranges(frame, columns):
    return {name: {'min': float(frame[name].min()), 'max': float(frame[name].max())} for name in columns}


def _prepare_output(root, output):
    output = Path(output).resolve()
    for name in ('data', 'models', 'config', 'src'):
        protected = root / name
        if output == protected or protected in output.parents:
            raise ValueError('Choose an output directory outside data, models, config and src.')
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_optimization_analysis(root, out):
    """Recalculate saved fronts, archive HV, Table 14 and paired budget changes.

    No optimizer or COMSOL solve is launched. Shared records represent exactly
    equal threshold scenarios and the 200-generation prefixes of longer runs.
    """
    root, config, weights = _settings(root)
    out = _prepare_output(root, out)
    folder = root / 'data/optimization'
    fronts = _read(folder / 'all_final_fronts.csv')
    histories = _read(folder / 'all_convergence.csv')
    runs = _read(folder / 'run_summary.csv')
    reference = np.asarray(config['reference_point'])
    leaders, statistics, logical_baselines = [], [], []
    for run in runs.itertuples(index=False):
        front = fronts.loc[(fronts.scenario == run.data_scenario) & (fronts.seed == run.seed) & (fronts.generations == run.generations)].copy()
        if len(front) != run.front_count:
            raise ValueError(f'Front row count mismatch: {run.scenario}/{run.seed}/{run.generations}')
        c = costs(front)
        hv = _hv(c, reference)
        if abs(hv - run.final_hypervolume) > 1e-12:
            raise ValueError(f'Final hypervolume mismatch: {run.scenario}/{run.seed}/{run.generations}')
        j = (1. - c) @ weights
        leader = front.iloc[int(np.argmax(j))].to_dict()
        leader.update(scenario=run.scenario, seed=int(run.seed), generations=int(run.generations), J_fixed_weights=float(j.max()))
        leaders.append(leader)
        statistics.append({'scenario': run.scenario, 'seed': int(run.seed), 'generations': int(run.generations), 'front_count': len(front), 'final_hypervolume': hv, 'best_J_fixed_weights': float(j.max())})
        if run.scenario == 'baseline' and run.generations == 200:
            logical_baselines.append(c)
    base_costs = np.vstack(logical_baselines)
    pooled_reference = base_costs[moocore.is_nondominated(base_costs)]
    stats = pd.DataFrame(statistics)
    for i, run in stats.loc[(stats.scenario == 'baseline') & (stats.generations == 200)].iterrows():
        f = fronts.loc[(fronts.scenario == 'baseline') & (fronts.seed == run.seed) & (fronts.generations == 200)]
        stats.loc[i, 'IGD_plus_to_pooled_reference'] = _igd_plus(costs(f), pooled_reference)
    leaders = pd.DataFrame(leaders)
    base_leaders = leaders.loc[(leaders.scenario == 'baseline') & (leaders.generations == 200)].copy()
    extended_leaders = leaders.loc[(leaders.scenario == 'baseline') & (leaders.generations == 500)].copy()

    # Feasible evaluated initial/offspring records are sufficient for archive HV;
    # survivor-population traces and repeated derived cost columns are unnecessary.
    archive_rows = []
    for seed in config['baseline_seeds']:
        with np.load(folder / f'evaluation_traces/baseline_seed_{seed}.npz') as arrays:
            generation, f, g = arrays['generation'], arrays['F'], arrays['G']
        if len(f) != int(generation.max()) * config['population'] or f.shape[1] != 5 or g.shape != f.shape:
            raise ValueError(f'Invalid evaluated objective/constraint trace: {seed}')
        if not np.isfinite(f).all() or not np.isfinite(g).all():
            raise ValueError(f'Nonfinite evaluation trace: {seed}')
        feasible = np.all(g <= 0, axis=1)
        checkpoints = [1] + list(range(25, int(generation.max()) + 1, 25))
        archive = np.empty((0, 5))
        previous = 0
        for checkpoint in checkpoints:
            new = f[feasible & (generation > previous) & (generation <= checkpoint)] + SHIFT
            combined = np.vstack([archive, new])
            archive = combined[moocore.is_nondominated(combined)] if len(combined) else combined
            archive_rows.append({'seed': int(seed), 'generation': checkpoint, 'cumulative_archive_HV': _hv(archive, reference), 'n_cumulative_nondominated_archive': len(archive), 'evaluations': checkpoint * config['population']})
            previous = checkpoint
    archive = pd.DataFrame(archive_rows)

    baseline_history = histories.loc[(histories.scenario == 'baseline') & (histories.generation <= 200)].copy()
    baseline_archive = archive.loc[archive.generation <= 200]
    current_aggregate = baseline_history.groupby('generation').hypervolume.agg(['count', 'mean', 'min', 'max']).reset_index()
    archive_aggregate = baseline_archive.groupby('generation').cumulative_archive_HV.agg(['count', 'mean', 'min', 'max']).reset_index()
    if not current_aggregate['count'].eq(10).all() or not archive_aggregate['count'].eq(10).all():
        raise ValueError('Baseline aggregation requires all ten independent seeds.')
    figure8 = current_aggregate.rename(columns={name: f'current_{name}' for name in ['count', 'mean', 'min', 'max']}).merge(
        archive_aggregate.rename(columns={name: f'archive_{name}' for name in ['count', 'mean', 'min', 'max']}), on='generation', how='left')

    scenario_rows, paired_rows = [], []
    reference_candidates=_read(root/'data/reference/Pareto_67_complete.csv')
    reference_front=_read(root/'data/reference/Pareto_final_front_all.csv')
    pooled_baseline=fronts.loc[(fronts.scenario=='baseline') & (fronts.generations==200)]
    for name, thresholds in config['thresholds'].items():
        lead = leaders.loc[(leaders.scenario == name) & (leaders.generations == 200)]
        run_stats = stats.loc[(stats.scenario == name) & (stats.generations == 200)]
        pd_min, imp_min = thresholds
        def retained(frame):
            return frame.loc[(frame.M_E>=1.2) & (frame.M_PD>=pd_min) & (frame.M_imp>=imp_min)]
        retained_candidates=retained(reference_candidates)
        retained_scores=(1.-costs(retained_candidates))@weights
        retained_leader=retained_candidates.iloc[int(np.argmax(retained_scores))]
        row = {'scenario': name, 'n_runs': len(lead), 'pd_min': pd_min, 'imp_min': imp_min,
               'reference_subset_retained':len(retained_candidates),
               'reference_subset_total':len(reference_candidates),
               'reference_subset_leader':str(retained_leader.candidate_id),
               'fixed_baseline_pool_HV':_hv(costs(retained(pooled_baseline)),reference),
               'effective_ME_min': max(1.2, (pd_min - .75) / .35, (imp_min - .80) / .30),
               'HV_mean': float(run_stats.final_hypervolume.mean()), 'HV_sd': float(run_stats.final_hypervolume.std(ddof=1)),
               'best_J_mean': float(lead.J_fixed_weights.mean()), 'N2_mean': float(lead.alpha_N2.mean()),
               'O2_mean': float(lead.alpha_O2.mean()), 'He_mean': float(lead.alpha_He.mean()),
               'He_min': float(lead.alpha_He.min()), 'He_max': float(lead.alpha_He.max()),
               'pressure_min_MPa': float(lead.p_MPa.min()), 'pressure_max_MPa': float(lead.p_MPa.max())}
        matched = lead.merge(base_leaders[['seed', 'alpha_He', 'p_MPa']], on='seed', suffixes=('', '_baseline'))
        delta = 100. * (matched.alpha_He - matched.alpha_He_baseline)
        row['paired_He_mean_difference_pp'] = float(delta.mean())
        row['paired_He_min_difference_pp'] = float(delta.min())
        row['paired_He_max_difference_pp'] = float(delta.max())
        scenario_rows.append(row)
        for (_, match), change in zip(matched.iterrows(), delta):
            paired_rows.append({'scenario': name, 'seed': int(match.seed), 'He_baseline': float(match.alpha_He_baseline), 'He_scenario': float(match.alpha_He), 'He_difference_pp': float(change), 'pressure_MPa': float(match.p_MPa)})
    scenarios = pd.DataFrame(scenario_rows)
    budget = extended_leaders[['seed', 'alpha_N2', 'alpha_O2', 'alpha_He', 'p_MPa', 'J_fixed_weights']].merge(
        base_leaders[['seed', 'alpha_N2', 'alpha_O2', 'alpha_He', 'p_MPa', 'J_fixed_weights']], on='seed', suffixes=('_500', '_200'))
    for name in ['alpha_N2', 'alpha_O2', 'alpha_He']:
        budget[f'{name}_difference_pp'] = 100. * (budget[f'{name}_500'] - budget[f'{name}_200'])
    late_rows = []
    for seed in config['baseline_seeds']:
        curve = archive.loc[archive.seed == seed].set_index('generation').cumulative_archive_HV
        row = {'seed': int(seed), 'archive150_to_200_percent': float(100. * (curve.loc[200] / curve.loc[150] - 1.))}
        if seed in config['extended_seeds']:
            row.update(archive200_to_500_percent=float(100. * (curve.loc[500] / curve.loc[200] - 1.)), archive450_to_500_percent=float(100. * (curve.loc[500] / curve.loc[450] - 1.)))
        late_rows.append(row)
    late = pd.DataFrame(late_rows)
    baseline_stats = stats.loc[(stats.scenario == 'baseline') & (stats.generations == 200)]
    summary = {'baseline_runs': len(base_leaders), 'threshold_runs': 30, 'extended_budget_checks': len(extended_leaders),
               'reference_subset_leader_score':float(((1.-costs(reference_candidates))@weights).max()),
               'reference_full_front_leader_score':float(((1.-costs(reference_front))@weights).max()),
               'baseline_final_population_HV_CV_percent': float(100. * baseline_stats.final_hypervolume.std(ddof=1) / baseline_stats.final_hypervolume.mean()),
               'pooled_empirical_reference_count': len(pooled_reference),
               'baseline_IGD_plus': _ranges(baseline_stats, ['IGD_plus_to_pooled_reference']),
               'baseline_leader_ranges': _ranges(base_leaders, DESIGN + ['alpha_He']),
               'extended_leader_ranges': _ranges(extended_leaders, DESIGN + ['alpha_He']),
               'matched_budget_He_change_pp': _ranges(budget, ['alpha_He_difference_pp']),
               'archive_late_change_percent': _ranges(late, ['archive150_to_200_percent', 'archive200_to_500_percent', 'archive450_to_500_percent']),
               'scope': 'Finite-budget computational screening. Archive coverage is distinct from current-population HV. Small late HV changes do not establish stable composition or global convergence.'}
    for name, frame in [('run_statistics', stats), ('full_front_leaders', leaders), ('threshold_summary', scenarios), ('paired_threshold_changes', pd.DataFrame(paired_rows)), ('paired_budget_changes', budget), ('archive_HV', archive), ('figure8_data', figure8)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    (out / 'optimization_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    _plot_figure8(figure8, out / 'figure8.png')
    return summary


def _plot_figure8(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35), layout='constrained')
    for ax, prefix, title in zip(axes, ['archive', 'current'], ['(a) Cumulative nondominated archive', '(b) Current population']):
        d = data.dropna(subset=[f'{prefix}_mean'])
        ax.fill_between(d.generation, d[f'{prefix}_min'], d[f'{prefix}_max'], color='#C6D8EA', label='Range across ten runs')
        ax.plot(d.generation, d[f'{prefix}_mean'], color='#1F4E79', lw=1.4, label='Mean across ten runs')
        ax.set(xlabel='Generation', ylabel='Hypervolume', title=title, xlim=(1, 200))
        ax.grid(axis='y', color='#D9D9D9', lw=.5)
        ax.legend(frameon=False, fontsize=7, loc='lower right')
    fig.savefig(path, dpi=300)
    plt.close(fig)


def reoptimize(root, out, seeds=None, generations=200, scenario='baseline'):
    """Run optional NSGA-II searches with the published settings.

    Outputs are new computations and are kept separate from saved input data.
    Longer runs replay the same seed from generation 1, rather than resuming it.
    """
    if importlib.metadata.version('pymoo') != '0.6.1.5':
        raise RuntimeError('Use pymoo==0.6.1.5 for the published optimizer settings.')
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize

    root, config, weights = _settings(root)
    out = _prepare_output(root, out)
    if scenario not in config['thresholds'] or generations < 1:
        raise ValueError('Choose a configured scenario and a positive generation count.')
    seeds = config['baseline_seeds'] if seeds is None and scenario == 'baseline' else (config['threshold_seeds'] if seeds is None else list(seeds))
    evaluator = CandidateEvaluator(root / config['surrogate_file'])
    pd_min, imp_min = config['thresholds'][scenario]
    reference = np.asarray(config['reference_point'])

    class GasCTProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=7, n_obj=5, n_ieq_constr=5,
                             xl=np.array([VARIABLE_BOUNDS[k][0] for k in DESIGN]),
                             xu=np.array([VARIABLE_BOUNDS[k][1] for k in DESIGN]))

        def _evaluate(self, x, result, *args, **kwargs):
            d = evaluator.evaluate(x, Scenario())
            result['F'] = np.array([-d[k] for k in SCORES[:4]] + [d['S_cost']])
            result['G'] = np.array([-d['alpha_He'], d['alpha_He'] - .5, 1.2 - d['M_E'], pd_min - d['M_PD'], imp_min - d['M_imp']])

    class Trace(Callback):
        def __init__(self):
            super().__init__()
            self.rows, self.generation, self.F, self.G = [], [], [], []
            self.archive = np.empty((0, 5))
            self.pending = []

        def notify(self, algorithm):
            gen = int(algorithm.n_gen)
            f = np.asarray(algorithm.off.get('F')).copy()
            g = np.asarray(algorithm.off.get('G')).copy()
            self.generation.append(np.full(len(f), gen, dtype=np.int16))
            self.F.append(f)
            self.G.append(g)
            self.pending.append(f[np.all(g <= 0, axis=1)] + SHIFT)
            checkpoint = gen == 1 or gen % 25 == 0 or gen == generations
            if checkpoint:
                combined = np.vstack([self.archive] + self.pending)
                self.archive = combined[moocore.is_nondominated(combined)] if len(combined) else combined
                self.pending.clear()
            opt = algorithm.opt
            good = np.all(opt.get('G') <= 0, axis=1)
            c = opt.get('F')[good] + SHIFT
            self.rows.append({'generation': gen, 'evaluations': int(algorithm.evaluator.n_eval), 'n_feasible_front': len(c), 'hypervolume': _hv(c, reference), 'best_J_fixed_weights': float(np.max((1. - c) @ weights)) if len(c) else np.nan,
                              'cumulative_archive_HV': _hv(self.archive, reference) if checkpoint else np.nan})

    output_records = []
    for seed in seeds:
        directory = out / f'{scenario}_seed{seed}_gen{generations}'
        directory.mkdir(parents=True, exist_ok=False)
        algorithm = NSGA2(pop_size=config['population'], eliminate_duplicates=True,
                          crossover=SBX(prob=.9, prob_var=.5, eta=15), mutation=PM(prob=.9, prob_var=1./7., eta=20))
        result = minimize(GasCTProblem(), algorithm, ('n_gen', generations), seed=int(seed), callback=Trace(), verbose=False)
        trace = result.algorithm.callback
        if result.X is None:
            raise RuntimeError(f'No feasible front for seed {seed}.')
        front = pd.DataFrame([evaluator.evaluate(x, Scenario()) for x in np.atleast_2d(result.X)])
        front.insert(0, 'front_index', np.arange(1, len(front) + 1))
        front['J_fixed_weights'] = (1. - costs(front)) @ weights
        front.to_csv(directory / 'front.csv', index=False)
        pd.DataFrame(trace.rows).to_csv(directory / 'convergence.csv', index=False)
        np.savez_compressed(directory / 'evaluation_trace.npz', generation=np.concatenate(trace.generation), F=np.vstack(trace.F), G=np.vstack(trace.G))
        record = {'scenario': scenario, 'seed': int(seed), 'generations': generations, 'population': config['population'], 'pd_min': pd_min, 'imp_min': imp_min, 'evaluations': int(result.algorithm.evaluator.n_eval), 'front_count': len(front), 'final_hypervolume': float(trace.rows[-1]['hypervolume'])}
        (directory / 'run.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        output_records.append(record)
    return output_records
