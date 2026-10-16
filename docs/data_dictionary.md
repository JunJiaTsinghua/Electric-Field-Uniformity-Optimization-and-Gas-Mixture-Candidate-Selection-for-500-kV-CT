# Data and result guide

CSV files use a header row. Geometry is in millimetres, field magnitude in kV/mm, potential in kV, pressure in MPa and temperature in K unless the column name states another unit. Gas fractions are dimensionless and sum to one. The saved numerical inputs are read without overwriting them.

## Geometry and surrogate

The four surrogate features are ordered as `r_hv_eq_mm`, `d_mm`, `h_m`, `l_m_mm`. Their ranges are 50–90 mm, 80–160 mm, 0.30–0.65 and 30–90 mm. `h_m` specifies axial position as a fraction of the 500 mm model height. The response is the gas-domain maximum `Emax_kV_per_mm`; training and validation files also include floating-electrode potential.

The seven Morris proxies and their ranges are defined in `data/morris/metadata.json`. `design_seed` identifies a design; `trajectory` and `point` identify its ordered points. There are 11 distinct designs × 80 rows. An identical repeated reference design is represented by the same rows. Ground-electrode, creepage and angle-labelled proxies parameterize the additional shielding geometry; they are not resolved physical features of a full manufactured CT.

## Candidate and optimizer tables

`candidate_id` identifies P01–P67 in the reference reporting subset. `Pareto_final_front_all.csv` is the original 100-point reference front; the 67-point subset uses farthest-point selection in normalized objective space. P01 is the combined-weight leader of that subset, not a claim of a unique global optimum.

Design columns are `r_hv_eq_mm`, `d_mm`, `h_m`, `l_m_mm`, `alpha_N2`, `alpha_O2`, `p_MPa`; helium is `1-alpha_N2-alpha_O2`. The five scores are `S_ins`, `S_th`, `S_det`, `S_env`, `S_cost`. Comprehensive benefit uses the first four scores and `1-S_cost` with the supplied weights.

Optimization tables identify `scenario`, `seed` and generation. `config/optimization.json` lists thresholds, scenario aliases, search parameters and the hypervolume reference point. There are ten baseline searches, six settings using five matched seeds each, and three baseline extensions to 500 generations. Two alternative settings have the same effective constraint, giving 35 distinct seed/constraint combinations among the 40 initial executions; their data are shared through the alias mapping.

The ten NPZ traces preserve float64 evaluated objective/constraint arrays and generation labels. Three traces cover 500 generations and supply both their full extension and the baseline prefix through generation 200; the other seven cover 200. Raw minimization objectives are `(-S_ins,-S_th,-S_det,-S_env,S_cost)`. Hypervolume uses `(1-S_ins,1-S_th,1-S_det,1-S_env,S_cost)` with reference point `(1.1,1.1,1.1,1.1,1.1)`.

The cumulative archive is the nondominated subset of all feasible evaluations, including feasible offspring discarded during population selection. Current-population hypervolume is a different measure and can decrease. The 0.79% baseline coefficient of variation refers to final current-population hypervolume.

## Conditional samples

`data/uncertainty/samples.csv.gz` contains 67 candidates × 2,000 draws. Its keys are `candidate_id` and `sample_id`; `seed` records the sampling realization. It includes perturbed geometry/composition, temperature, closed-volume pressure, surrogate residual, breakdown factor, field, margin and composite score.

The breakdown factor is drawn from N(1, 0.08²) and clipped to [0.76,1.24]. It is not rejection-sampled from a conditionally truncated distribution. `config/uncertainty.json` specifies all perturbations. The paired GP-scale calculation reuses these standardized residual draws; fitted predictive standard deviation includes the WhiteKernel term. It does not represent unresolved end-field discretization bias.

## Mesh tables and models

`data/mesh/local/` tables use `case` and `level` to identify two reference end shapes and three boundary mesh sizes. `sharp_rect` is rectangular; `rounded_R2` has 2 mm semicircular ends. L01/L02/L03 have end-cap maximum boundary size 0.20/0.10/0.05 mm. These local user-controlled meshes are distinct from the original coarse/medium/fine physics-controlled presets, identified by `mesh` in the preset tables.

`peaks` tables identify the boundary/domain peak and its native r,z coordinates. Statistics and connectivity tables describe actual meshes. `hv_profile`, `hv_offset_0p05mm`, `shared_corner_rays` and the two `end_gas_offset` tables describe common physical probes. Values labelled 0.00001 mm and 0.05 mm are different sampling offsets. Seventy-two near-boundary values on the coarsest rounded mesh are NaN and remain missing; they are not assigned zero. Finer rounded near-boundary profiles and all 0.05 mm profiles are complete.

The 11 unique MPH files comprise three original mesh references, two sampling models and six locally refined solved models. Their source settings and Java jobs are defined by `config/finite_element.json` and `src/comsol_java/`. The original source model contains the recorded solution; aliases do not create additional independent solves.

## Recalculate paper results

All paths below are produced by `python main.py analyze`; they are generated outputs rather than additional required inputs.

| Paper content | Task | Key output |
|---|---|---|
| Mesh comparison, peak locations and Supplementary Figs. S2/S3 | `mesh` | Mesh summary, peak shifts and comparison profiles/plots |
| Surrogate validation and Table 6 | `surrogate` | `surrogate/validation_predictions.csv`, `validation_metrics.json` |
| Morris effects and Supplementary Fig. S1 | `morris` | Design statistics, range and trajectory-resampling summaries |
| Sobol indices and normalized Figure 4(b) shares | `sobol` | `sensitivity/sobol_indices.csv`, `sobol_summary.json` |
| Combined weights, candidates and ranking sensitivity, Tables 8/9/11 | `ranking` | `ranking/weights_recomputed.csv`, candidate/decision-rule tables |
| Fixed-P01 He scan, Table 10 and Fig. 6 | `scan` | `scan/controlled_He_scan_dense.csv` |
| Conditional margin statistics and Fig. 7, Table 12 | `uncertainty` | `uncertainty/monte_carlo_summary.csv`, margin distributions |
| Input/preference scenario selection, Table 13 | `ranking` | `ranking/scenario_candidate_scores.csv` |
| Baseline/extended searches and Fig. 8, threshold Table 14 | `optimization` | Archive/population HV, leaders and matched-setting comparisons |

Plot data and statistics can reproduce the numerical content. Schematic artwork in the main manuscript is not needed to execute the calculations. Sobol plot percentages divide each index by the sum of its own first-order or total-effect group; the underlying indices are retained separately.

## Read the comparison outputs

`results/optimization/threshold_summary.csv` reports the effective margin bound, the retained reference-subset count and leader, independently optimized leaders and fixed-pool hypervolume. The fixed pool contains the 1,000 terminal candidates from the ten baseline runs. Tightening constraints cannot increase hypervolume on this same pool.

`results/mesh/successive_mesh_comparisons.csv` reports profile differences relative to the previous and current mesh. Use `relative_to_current_maximum_percent` for comparisons normalized by the finer mesh, as in the paper. Missing probes remain missing and are counted separately.

`results/morris/variable_ranges_and_bootstrap.csv` reports `maximum_same_design_ratio_to_gap`: divide each input's mean absolute effect by the gap effect in the same design, then take the maximum over all eleven designs. The summary distinguishes rank-based top-four frequency from threshold-based selection.
