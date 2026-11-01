# V5 thesis figure reproduction — CSV-driven, no ABM rerun

Both Python scripts read their input CSVs from **the folder containing the script itself** by default (`Path(__file__).resolve().parent`). They do not depend on the VS Code terminal's working directory. Use `--input-dir` only if you deliberately store the CSVs elsewhere.

## Files to place together

Place these files in one folder (all eight names are exact and verified against `V5_output (1).zip`):

```
YOUR_V5_FOLDER/
  generate_thesis_figures_v5.py
  verify_v5_results.py
  requirements_v5_figures.txt
  factorial_timeseries_v5.csv
  factorial_ex_post_summary_v5.csv
  population_sweep_v5.csv
  population_sweep_summary_v5.csv
  sensitivity_sweeps_v5.csv
  sensitivity_sweeps_summary_v5.csv
  sensitivity_factorial_grid_v5.csv
  sensitivity_factorial_grid_summary_v5.csv
```

**The figure script needs only five run-level CSVs** (factorial timeseries, factorial ex-post, population sweep, sensitivity sweeps, sensitivity factorial grid). The statistics script also loads the other three summary CSVs and cross-checks their means against raw observations. No `parameters.csv`, V5 model script, or Mesa installation is needed for this analysis.

## Run on Windows (PowerShell / VS Code)

Open the folder containing these files and run:

```powershell
python -m pip install -r .\requirements_v5_figures.txt
python .\generate_thesis_figures_v5.py
python .\verify_v5_results.py
```

Or run either script from ANY OTHER directory, providing the full path to the script:

```powershell
& "C:\Users\cengo\AppData\Local\Microsoft\WindowsApps\python3.13.exe" "C:\Users\cengo\OneDrive\Masaüstü\TUM\game theory\Code\v5_bundle\generate_thesis_figures_v5.py"
```

The second approach still reads the CSVs from the **script's folder**, not the terminal directory.

Optional overrides:

```powershell
python .\generate_thesis_figures_v5.py --input-dir "C:\path\to\outputs" --dpi 300
python .\generate_thesis_figures_v5.py --formats png
python .\verify_v5_results.py --input-dir "C:\path\to\outputs"
```

## Outputs (automatically generated)

```
YOUR_V5_FOLDER/
  figures_v5/
    png/                 # printable bitmap figures
    pdf/                 # vector figures for thesis insertion
    plot_data/           # exact aggregated numbers used in figures
    figure_manifest_v5.csv
  verified_stats_v5/
    experiment_completeness.csv
    factorial_final_timeseries_consistency.csv
    raw_vs_summary_mean_checks.csv
    factorial_seed_means_and_ci.csv
    factorial_paired_seed_contrasts_and_did.csv
    population_all_25_T3_minus_T2_paired.csv
    six_univariate_sweep_cell_stats.csv
    bivariate_grid_cell_stats.csv
    baseline_mechanism_diagnostics.csv
    economics_reconstructed_by_seed.csv
    economics_treatment_summary.csv
    verification_report.txt
```

Original eight input CSVs are never modified.

## Figure manifest (actual column names and source CSVs)

| Chapter | Generated figure | Source | Data columns / formula |
|---|---|---|---|
| 5.1 | `Figure_5_1_baseline_ribbons` | `factorial_timeseries_v5.csv` | `rho_C`, `q_C`, `mean_mu`, `mean_omega` grouped by `round` and `treatment`; Student-t 95% CI over distinct `seed` |
| 5.2 | `Figure_5_2_paired_factorial_effects` | `factorial_ex_post_summary_v5.csv` | `final_rho_C` pivoted by `seed` × `treatment`; T1−T0, T3−T2, (T3−T2)−(T1−T0) |
| 5.3 | `Figure_5_3_population_heatmap` | `population_sweep_v5.csv` | T3−T2 mean `final_rho_C` grouped by `p_C0`,`p_TypeA` |
| 5.4 | `Figure_5_4_sensitivity_panels` | `sensitivity_sweeps_v5.csv` | `final_rho_C` over `parameter_value` for each actual `swept_parameter` (`alpha`, `lambda_L`, `w`, `p_base`, `gamma`, `p_rewire`); T2/T3 and seed CIs |
| 5.5 | `Figure_5_5_rewiring_incentive_heatmap` | `sensitivity_factorial_grid_v5.csv` | T3−T2 mean `final_rho_C` over `p_rewire_sweep`,`w_sweep` |
| 5.6 | `Figure_5_6_transition_connectivity` | `sensitivity_sweeps_v5.csv` | Baseline rows selected by `swept_parameter='p_rewire'` and `parameter_value==baseline_value`; `total_C_to_D_transitions`, `total_D_to_C_transitions`, `mean_connected_after` |
| 5.7 | `Figure_5_7_accounting_comparison` | `factorial_ex_post_summary_v5.csv` | Per seed: N = `final_n_cooperators_predecision` + `final_n_defectors_predecision`; outlay = N×`w`×(`mean_p_final` + `mean_R_c_final`); recorded synergy numerator = `ROI_platform`×outlay. Means calculated AFTER multiplying per-seed numbers. |

**Five additional appendix figures** are also created: population neighborhood contrast, bivariate neighborhood contrast, bivariate reported ROI contrast, threshold/streak dashboard, and network connectedness versus rewiring probability.

## Statistical and conceptual safeguards

- **Independence is at the seed level.** The time-series confidence ribbons calculate variability over the 100 seeds **at each round**. A repeated round is not a new independent replication.
- **Paired seed inference** is used for all treatment contrasts. The verifier reports Student-t confidence intervals with n−1 degrees of freedom, not normal 1.96×SE intervals.
- **Composition comparisons** report all 25 cells and Holm-adjusted p-values, rather than only the cells where T3 wins.
- **ROI** is the model's synergy-to-token-outlay ratio, not observed real-world investment return. At `w=0`, the ratio is undefined and the ROI heatmap displays N/A.
- **Cumulative transitions** count switches, not unique farmers; the diagnostics do not identify transitions as Type B imitation events.
- **`q_C` versus Type A `q*`** are different statistics and are visualized separately and clearly labeled as such.
- **The figures verify neither the underlying ABM implementation nor field validity.** They reproduce and summarize the available outputs only.
- **No external internet, no V3/V5 simulation, and no editing of original V5 data** are involved.

## Tested against the supplied ZIP

The scripts were executed against all eight CSVs in `V5_output (1).zip`. The figure script completed successfully and produced seven principal figures plus five appendix figures (12 PNG + 12 PDF + 12 plotted-value CSVs). The numerical verifier passed input coverage, duplicate-key, four-primary-outcome aggregation, and final time-series consistency tests, producing 11 tables and its report. These checks can be repeated on your machine with the commands above.
