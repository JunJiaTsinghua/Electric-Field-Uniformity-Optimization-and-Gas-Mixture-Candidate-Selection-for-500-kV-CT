"""Conditional Monte Carlo statistics and predictive-scale sensitivity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .io_utils import output_directory

FEATURES = ['r_hv_eq_mm', 'd_mm', 'h_m', 'l_m_mm']


def run_uncertainty(root, out):
    """Recompute published statistics from the single saved sample table."""
    from .scoring import CandidateEvaluator
    root = Path(root).resolve()
    out = output_directory(root, out, 'uncertainty')
    config = json.loads((root/'config/uncertainty.json').read_text(encoding='utf-8'))
    raw = pd.read_csv(root/'data/uncertainty/samples.csv.gz', float_precision='round_trip')
    expected = config['candidate_count'] * config['samples_per_candidate']
    if len(raw) != expected or not raw.groupby('candidate_id').size().eq(config['samples_per_candidate']).all():
        raise ValueError('Monte Carlo sample counts differ from the declared configuration.')
    if not np.isfinite(raw[FEATURES + ['M_E', 'J', 'Emax_kV_per_mm', 'Ebd_kV_per_mm']].to_numpy()).all():
        raise ValueError('Monte Carlo data contain nonfinite values.')
    if np.max(np.abs(raw[['alpha_N2','alpha_O2','alpha_He']].sum(axis=1)-1)) > 1e-12:
        raise ValueError('Monte Carlo compositions do not sum to one.')

    gp_artifact = CandidateEvaluator(root/'models/surrogate/Kriging_Emax_model.joblib').artifact
    gp, scaler = gp_artifact['gaussian_process'], gp_artifact['input_scaler']
    means, stds = [], []
    with threadpool_limits(limits=1):
        for start in range(0, len(raw), 4096):
            mean, std = gp.predict(scaler.transform(raw[FEATURES].iloc[start:start+4096].to_numpy()), return_std=True)
            means.append(mean); stds.append(std)
    means, stds = np.concatenate(means), np.concatenate(stds)
    residual = raw.surrogate_error_kV_per_mm.to_numpy()
    base_field = np.maximum(config['minimum_field_kV_per_mm'], means + residual)
    reconstruction_error = float(np.max(np.abs(base_field-raw.Emax_kV_per_mm.to_numpy())))
    if reconstruction_error > 1e-9:
        raise ValueError(f'Archived field reconstruction mismatch: {reconstruction_error}')
    alt_field = np.maximum(config['minimum_field_kV_per_mm'], means + residual/config['validation_RMSE_kV_per_mm']*stds)
    alternative_margin = raw.Ebd_kV_per_mm.to_numpy()/alt_field

    mean_limit=config['feasibility_summary']['mean_margin_min']
    percentile_limit=config['feasibility_summary']['P5_margin_min']
    summaries, convergence, predictive = [], [], []
    for cid, group in raw.groupby('candidate_id', sort=False):
        ix = group.index.to_numpy(); me = group.M_E.to_numpy(); score = group.J.to_numpy(); alt = alternative_margin[ix]
        summaries.append({'candidate_id':cid,'n_samples':len(group),'ME_mean':float(me.mean()),'ME_std':float(me.std(ddof=1)),
            'ME_P5':float(np.quantile(me,.05)),'ME_P50':float(np.quantile(me,.5)),'ME_P95':float(np.quantile(me,.95)),
            'robustness_eta_mean_over_std':float(me.mean()/me.std(ddof=1)),'probability_ME_ge_1p2':float(np.mean(me>=1.2)),
            'J_mean':float(score.mean()),'J_std':float(score.std(ddof=1))})
        for count in config['percentile_checkpoints']:
            convergence.append({'candidate_id':cid,'n_samples':count,'ME_P5':float(np.quantile(me[:count],.05))})
        predictive.append({'candidate_id':cid,'n':len(group),'GP_std_min':float(stds[ix].min()),'GP_std_median':float(np.median(stds[ix])),
            'GP_std_max':float(stds[ix].max()),'original_mean_ME':float(me.mean()),'original_P5_ME':float(np.quantile(me,.05)),
            'diagnostic_mean_ME':float(alt.mean()),'diagnostic_P5_ME':float(np.quantile(alt,.05)),
            'original_pass':bool(me.mean()>=mean_limit and np.quantile(me,.05)>=percentile_limit),
            'diagnostic_pass':bool(alt.mean()>=mean_limit and np.quantile(alt,.05)>=percentile_limit)})
    base = pd.DataFrame(summaries); paired = pd.DataFrame(predictive)
    base.to_csv(out/'monte_carlo_summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(convergence).to_csv(out/'percentile_convergence.csv',index=False,float_format='%.17g')
    paired.to_csv(out/'predictive_scale_summary.csv',index=False,float_format='%.17g')
    report = {'samples':len(raw),'candidates':len(base),'mean_margin_range':[float(base.ME_mean.min()),float(base.ME_mean.max())],
        'P5_margin_range':[float(base.ME_P5.min()),float(base.ME_P5.max())],
        'GP_standard_deviation_quantiles_kV_per_mm':np.quantile(stds,[0,.05,.5,.95,1]).tolist(),
        'field_reconstruction_max_error_kV_per_mm':reconstruction_error,
        'constant_RMSE_pass_count':int(paired.original_pass.sum()),'predictive_scale_pass_count':int(paired.diagnostic_pass.sum()),
        'max_absolute_mean_margin_change':float(np.max(np.abs(paired.diagnostic_mean_ME-paired.original_mean_ME))),
        'max_absolute_P5_margin_change':float(np.max(np.abs(paired.diagnostic_P5_ME-paired.original_P5_ME))),
        'scope':'Conditional sampling under the specified proxy models. GP return_std includes the fitted WhiteKernel; neither error specification includes unresolved sharp-end discretization bias.'}
    (out/'uncertainty_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    _plot_margin_distributions(raw, out)
    return report


def _plot_margin_distributions(raw, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    for cid in ['P01','P02','P09','P21','P30','P37','P54']:
        a = np.sort(raw.loc[raw.candidate_id==cid,'M_E'].to_numpy())
        if len(a):ax.plot(a, np.arange(1,len(a)+1)/len(a), label=cid)
    ax.axvline(1.2,color='black',linestyle='--',linewidth=.8)
    ax.set(xlabel='Electric-field margin',ylabel='Empirical cumulative probability',ylim=(0,1))
    ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(Path(out)/'margin_distributions.png',dpi=300);plt.close(fig)


def regenerate_samples(root, out, samples_per_candidate=2000, seed=20260815):
    """Generate a fresh draw table, preserving the recorded per-sample RNG order."""
    from .scoring import CandidateEvaluator, Scenario, evaluate_design, clip_norm, NORMALIZATION_BOUNDS, VARIABLE_BOUNDS
    root = Path(root).resolve()
    out = output_directory(root, out, 'samples_generated')
    if samples_per_candidate < 1:raise ValueError('samples_per_candidate must be positive.')
    out.mkdir(parents=True,exist_ok=True)
    candidates = pd.read_csv(root/'data/reference/Pareto_67_complete.csv',float_precision='round_trip')
    weights = pd.read_csv(root/'data/reference/AHP_entropy_combined_weights.csv',float_precision='round_trip').combined_multiplicative_weight.to_numpy()
    artifact = CandidateEvaluator(root/'models/surrogate/Kriging_Emax_model.joblib').artifact
    cfg = json.loads((root/'config/uncertainty.json').read_text(encoding='utf-8'))
    deviations=cfg['geometry_gaussian_sd'];factor_cfg=cfg['gas_model_factor']
    if factor_cfg['distribution']!='normal' or factor_cfg['operation']!='clip':
        raise ValueError('The supplied sampler implements a clipped-normal breakdown factor.')
    rng = np.random.default_rng(seed); rows=[]
    for candidate in candidates.itertuples(index=False):
        for sample_id in range(1,samples_per_candidate+1):
            rhv=float(np.clip(rng.normal(candidate.r_hv_eq_mm,deviations[0]),*VARIABLE_BOUNDS['r_hv_eq_mm']));gap=float(np.clip(rng.normal(candidate.d_mm,deviations[1]),*VARIABLE_BOUNDS['d_mm']))
            hm=float(np.clip(rng.normal(candidate.h_m,deviations[2]),*VARIABLE_BOUNDS['h_m']));lm=float(np.clip(rng.normal(candidate.l_m_mm,deviations[3]),*VARIABLE_BOUNDS['l_m_mm']))
            fractions=np.clip(np.array([candidate.alpha_N2,candidate.alpha_O2,candidate.alpha_He])+rng.normal(0,cfg['composition_sd'],3),0,None);fractions/=fractions.sum()
            temperature=float(rng.uniform(*cfg['temperature_uniform_K']));fill=float(np.clip(rng.normal(0,cfg['fill_error_sd']),*cfg['fill_error_clip']))
            pressure=float(candidate.p_MPa*(temperature/293.15)*(1+fill))
            design=np.array([rhv,gap,hm,lm,fractions[0],fractions[1],pressure])
            value=evaluate_design(design,artifact,Scenario('MC',temperature_k=temperature))
            residual=float(rng.normal(0,cfg['validation_RMSE_kV_per_mm']));factor=float(np.clip(rng.normal(factor_cfg['mean'],factor_cfg['sd']),*factor_cfg['bounds']))
            field=max(cfg['minimum_field_kV_per_mm'],value['Emax_pred_kV_per_mm']+residual);breakdown=value['Ebd_kV_per_mm']*factor;me=breakdown/field
            sins=(clip_norm(me,*NORMALIZATION_BOUNDS['ME'])+clip_norm(.75+.35*me,*NORMALIZATION_BOUNDS['MPD'])+clip_norm(.8+.3*me,*NORMALIZATION_BOUNDS['Mimp']))/3
            score=float(np.dot([sins,value['S_th'],value['S_det'],value['S_env'],1-value['S_cost']],weights))
            rows.append(dict(candidate_id=candidate.candidate_id,sample_id=sample_id,seed=seed,r_hv_eq_mm=rhv,d_mm=gap,h_m=hm,l_m_mm=lm,
                alpha_N2=fractions[0],alpha_O2=fractions[1],alpha_He=fractions[2],temperature_K=temperature,p_actual_MPa_closed_volume=pressure,
                surrogate_error_kV_per_mm=residual,gas_model_factor=factor,Emax_kV_per_mm=field,Ebd_kV_per_mm=breakdown,M_E=me,J=score,pass_ME_ge_1p2=int(me>=1.2)))
    path=out/'samples.csv.gz';pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g',compression='gzip')
    return {'samples':len(rows),'seed':seed,'output':str(path)}
