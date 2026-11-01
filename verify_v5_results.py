#!/usr/bin/env python3
"""
Recompute V5 thesis statistics from run-level CSVs, without rerunning the ABM.

Usage (place the script beside extracted CSV files):
    python verify_v5_results.py

The script uses its OWN DIRECTORY, not the terminal's current directory.
Optional: python verify_v5_results.py --input-dir "C:/path/to/CSVs"
Results: <csv_directory>/verified_stats_v5/*.csv and verification_report.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests

RAW = {
    "factorial": "factorial_ex_post_summary_v5.csv",
    "timeseries": "factorial_timeseries_v5.csv",
    "population": "population_sweep_v5.csv",
    "sensitivity": "sensitivity_sweeps_v5.csv",
    "grid": "sensitivity_factorial_grid_v5.csv",
}
SUMMARY = {
    "population": "population_sweep_summary_v5.csv",
    "sensitivity": "sensitivity_sweeps_summary_v5.csv",
    "grid": "sensitivity_factorial_grid_summary_v5.csv",
}
T = ["T0_Static_NoInc", "T1_Fluid_NoInc", "T2_Static_Active", "T3_Fluid_Active"]
FOUR = ["final_rho_C", "final_q_C", "final_mu", "final_omega"]
EXTRA = ["final_type_A_mean_q_star", "final_mean_mu_applied", "ROI_platform"]


def read(root: Path):
    needed = list(RAW.values()) + list(SUMMARY.values())
    missing = [f for f in needed if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CSVs beside script in {root}: {missing}\n"
                                "Extract all eight V5 CSV files into this directory.")
    datasets = {k: pd.read_csv(root / v, low_memory=False) for k,v in RAW.items()}
    summaries = {k: pd.read_csv(root / v, low_memory=False) for k,v in SUMMARY.items()}
    for key,frame in datasets.items():
        print(f"Loaded {RAW[key]:44s} {len(frame):>6,} rows")
    return datasets,summaries


def estimate(v: pd.Series) -> dict:
    d=v.dropna().astype(float)
    n=len(d)
    if n < 1:
        return dict(n=0,mean=np.nan,sd=np.nan,se=np.nan,ci95_lower=np.nan,
                    ci95_upper=np.nan,t_stat=np.nan,p_two_sided=np.nan)
    m=float(d.mean());sd=float(d.std(ddof=1)) if n>1 else np.nan
    se=sd/np.sqrt(n) if n>1 else np.nan
    h=float(student_t.ppf(.975,n-1)*se) if n>1 else np.nan
    tt=m/se if n>1 and se>0 else np.nan
    pp=2*student_t.sf(abs(tt),n-1) if np.isfinite(tt) else np.nan
    return dict(n=n,mean=m,sd=sd,se=se,ci95_lower=m-h,ci95_upper=m+h,t_stat=tt,p_two_sided=pp)


def export(df: pd.DataFrame, output: Path, file: str):
    df.to_csv(output/file,index=False,float_format="%.12g")
    print(f"Wrote {file} ({len(df)} rows)")


def run_checks(datasets, summaries, output):
    notes=[];failures=[]
    ts=datasets['timeseries'];fac=datasets['factorial'];pop=datasets['population'];sw=datasets['sensitivity'];grid=datasets['grid']
    for key,cols in {
      'factorial':['treatment','seed'],
      'timeseries':['treatment','seed','round'],
      'population':['treatment','seed','p_C0','p_TypeA'],
      'sensitivity':['treatment','seed','swept_parameter','parameter_value'],
      'grid':['treatment','seed','p_rewire_sweep','w_sweep']
    }.items():
        df=datasets[key]
        miss=[c for c in cols if c not in df.columns]
        if miss:raise ValueError(f"{RAW[key]} missing columns {miss}")
        duplicates=int(df.duplicated(cols).sum())
        notes.append(f"{RAW[key]}: rows={len(df)}; duplicated experiment keys={duplicates}")
        if duplicates:failures.append(f"{RAW[key]} has {duplicates} duplicate experiment keys")

    ns=fac.seed.nunique();nr=ts['round'].nunique();nc=pop[['p_C0','p_TypeA']].drop_duplicates().shape[0]
    nsw=sw[['swept_parameter','parameter_value']].drop_duplicates().shape[0]
    ng=grid[['p_rewire_sweep','w_sweep']].drop_duplicates().shape[0]
    sizes={"factorial":(len(fac),4*ns),"timeseries":(len(ts),4*ns*nr),
           "population":(len(pop),4*ns*nc),"sensitivity":(len(sw),4*ns*nsw),"grid":(len(grid),4*ns*ng)}
    report=[]
    for key,(observed,expected) in sizes.items():
        report.append(dict(experiment=key,observed_rows=observed,expected_rows=expected,
                           complete=observed==expected,seed_count=ns))
        if observed!=expected:failures.append(f"{key}: expected {expected} but has {observed} rows")
    export(pd.DataFrame(report),output,'experiment_completeness.csv')

    # Validate original final time-series outcomes against the factorial file.
    last=ts.loc[ts.groupby(['treatment','seed'])['round'].idxmax()].set_index(['treatment','seed'])
    ff=fac.set_index(['treatment','seed'])
    temporal=[]
    for base,fin in [('rho_C','final_rho_C'),('q_C','final_q_C'),('mean_mu','final_mu'),('mean_omega','final_omega')]:
        joined=ff[[fin]].join(last[[base]],how='inner',validate='one_to_one')
        worst=float((joined[fin]-joined[base]).abs().max())
        temporal.append(dict(metric=base,max_abs_difference=worst,n_matched=len(joined),pass_tol_1e_10=worst<1e-10))
        if worst>=1e-10:failures.append(f"Final timeseries {base} mismatch: {worst}")
    export(pd.DataFrame(temporal),output,'factorial_final_timeseries_consistency.csv')

    # Compare all original 'summary' files against raw-cell means.
    # We verify the FOUR primary final outcome means only; undefined thresholds and
    # zero-cost ROI are intentionally not treated as missing-data errors.
    key_sets={"population":['p_C0','p_TypeA','treatment'],
              "sensitivity":['swept_parameter','parameter_value','treatment'],
              "grid":['p_rewire_sweep','w_sweep','treatment']}
    summary_rows=[]
    for typ,keys in key_sets.items():
        agg=datasets[typ].groupby(keys)[FOUR].mean().reset_index()
        sample=summaries[typ]
        expected=[f'{m}_mean' for m in FOUR]
        missing=set(keys+expected)-set(sample.columns)
        if missing:raise ValueError(f"{SUMMARY[typ]} missing columns: {sorted(missing)}")
        merged=agg.merge(sample[keys+expected],on=keys,validate='one_to_one',how='outer',indicator=True)
        if (merged._merge!='both').any():failures.append(f"{typ}: summary/raw group-key discrepancy")
        for met in FOUR:
            diff=(merged[met]-merged[f'{met}_mean']).abs()
            worst=float(diff.max())
            summary_rows.append(dict(experiment=typ,metric=met,n_groups=len(merged),
                                     max_abs_mean_error=worst,pass_tol_1e_10=worst<1e-10))
            if worst>=1e-10:failures.append(f"{typ} summary {met} mismatch: {worst}")
    export(pd.DataFrame(summary_rows),output,'raw_vs_summary_mean_checks.csv')

    # Factorial descriptive means and CIs by seed (never agent-round pooling).
    descr=[]
    for outcome in FOUR+EXTRA:
        for tr,g in fac.groupby('treatment'):
            descr.append(dict(treatment=tr,metric=outcome,**estimate(g[outcome])))
    export(pd.DataFrame(descr),output,'factorial_seed_means_and_ci.csv')

    # Seed-paired simple effects and 2x2 factorial interactions for each outcome.
    pairs=[('T1_minus_T0',T[1],T[0]),('T2_minus_T0',T[2],T[0]),
           ('T3_minus_T2',T[3],T[2]),('T3_minus_T1',T[3],T[1]),
           ('T3_minus_T0',T[3],T[0])]
    paired=[]
    for outcome in FOUR+EXTRA:
        wide=fac.pivot(index='seed',columns='treatment',values=outcome)
        for label,a,b in pairs:
            if wide[a].notna().any() and wide[b].notna().any():
                paired.append(dict(metric=outcome,contrast=label,**estimate(wide[a]-wide[b])))
        if wide.notna().all().all():
            did=(wide[T[3]]-wide[T[2]])-(wide[T[1]]-wide[T[0]])
            paired.append(dict(metric=outcome,contrast='factorial_interaction',**estimate(did)))
    export(pd.DataFrame(paired),output,'factorial_paired_seed_contrasts_and_did.csv')

    # 25 population cell contrasts, with family-wise Holm correction for rho_C.
    comp=[]
    for (c,a),g in pop.groupby(['p_C0','p_TypeA']):
        w=g.pivot(index='seed',columns='treatment',values='final_rho_C')
        r=estimate(w[T[3]]-w[T[2]])
        comp.append(dict(p_C0=c,p_TypeA=a,T2_mean=w[T[2]].mean(),T3_mean=w[T[3]].mean(),**r))
    comp=pd.DataFrame(comp)
    comp['holm_adjusted_p_25cells']=np.nan
    mask=comp.p_two_sided.notna()
    comp.loc[mask,'holm_adjusted_p_25cells']=multipletests(comp.loc[mask,'p_two_sided'],method='holm')[1]
    export(comp,output,'population_all_25_T3_minus_T2_paired.csv')

    # Parameter means by cell; includes all six original sweeps and bivariate grid.
    col=[*FOUR,'final_type_A_mean_q_star','mean_connected_after','ROI_platform']
    u=sw.groupby(['swept_parameter','parameter_value','treatment'])[col].agg(['mean','std','count']).reset_index()
    u.columns=['_'.join(v).rstrip('_') for v in u.columns]
    export(u,output,'six_univariate_sweep_cell_stats.csv')
    b=grid.groupby(['p_rewire_sweep','w_sweep','treatment'])[col].agg(['mean','std','count']).reset_index()
    b.columns=['_'.join(v).rstrip('_') for v in b.columns]
    export(b,output,'bivariate_grid_cell_stats.csv')

    # Baseline diagnostics from original p_rewire univariate sweep.
    d=sw[(sw.swept_parameter=='p_rewire') & np.isclose(sw.parameter_value,sw.baseline_value)]
    diag_cols=[c for c in ['total_C_to_D_transitions','total_D_to_C_transitions',
                          'total_strategy_transition_count','total_existing_streak_reset_events',
                          'total_successful_edge_swaps','total_rewire_pair_attempts',
                          'mean_connected_after','mean_type_A_threshold_met_rate',
                          'final_type_A_mean_q_star','final_mean_mu_applied',
                          'final_mean_total_marginal_incentive_cooperators',
                          'monitoring_mechanism_implemented',
                          'graduated_sanction_mechanism_implemented'] if c in d.columns]
    db=d.groupby('treatment')[diag_cols].mean().reset_index()
    export(db,output,'baseline_mechanism_diagnostics.csv')

    # Economic reconstruction: multiply per-seed ratio and per-seed outlay BEFORE means.
    e=fac[['treatment','seed','w','mean_p_final','mean_R_c_final','ROI_platform',
           'final_n_cooperators_predecision','final_n_defectors_predecision']].copy()
    e['population_size']=e.final_n_cooperators_predecision+e.final_n_defectors_predecision
    e['implied_token_outlay']=e.population_size*e.w*(e.mean_p_final+e.mean_R_c_final)
    e['ROI_implied_recorded_synergy']=e.ROI_platform*e.implied_token_outlay
    e['recorded_synergy_less_token_outlay']=e.ROI_implied_recorded_synergy-e.implied_token_outlay
    export(e,output,'economics_reconstructed_by_seed.csv')
    eb=e.groupby('treatment')[['implied_token_outlay','ROI_implied_recorded_synergy',
                               'recorded_synergy_less_token_outlay','ROI_platform']].agg(['mean','std']).reset_index()
    eb.columns=['_'.join(v).rstrip('_') for v in eb.columns]
    export(eb,output,'economics_treatment_summary.csv')

    # General caveats and deterministic record of program logic.
    notes += [
        f"Experiment coverage based on baseline {ns} seed labels and {nr} time steps.",
        "CIs: 95% Student-t intervals across seeds; difference tests are paired by seed.",
        "Population Holms: family-wise correction over 25 2-sided cell comparisons.",
        "ROI_platform is a model synergy/token-outlay ratio; not empirically validated financial ROI.",
        "At w=0, no-incentive ROI is undefined; no-incentive wallet/reputation points may still be recorded.",
        "Lambda_L=1 can produce undefined analytic q*; other valid outcomes remain retained.",
        "This script validates CSV aggregation and numerical inference; it does not validate model source code.",
        "None of these operations reruns or changes the ABM or source CSVs.",
    ]
    with (output/'verification_report.txt').open('w',encoding='utf-8') as out:
        out.write('V5 REPRODUCTION / VERIFICATION REPORT\n')
        out.write('====================================\n\n')
        out.write('\n'.join(notes))
        out.write('\n\nERRORS: '+('; '.join(failures) if failures else 'None in the tested aggregation and completeness checks.')+'\n')
    if failures:
        raise ValueError("Verification found problems: "+'; '.join(failures))
    print('SUCCESS: aggregate comparisons, experiment completeness and time-series equality checks passed.')


def main():
    parser=argparse.ArgumentParser(description="Verify V5 stats, reading CSVs from the script directory")
    parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path,default=None)
    a=parser.parse_args()
    root=a.input_dir.expanduser().resolve()
    output=a.output_dir.expanduser().resolve() if a.output_dir else root/'verified_stats_v5'
    try:
        datasets,summaries=read(root)
        output.mkdir(parents=True,exist_ok=True)
        run_checks(datasets,summaries,output)
        print('Output directory:',output)
        return 0
    except (FileNotFoundError,ValueError,KeyError) as e:
        print('ERROR:',e,file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
