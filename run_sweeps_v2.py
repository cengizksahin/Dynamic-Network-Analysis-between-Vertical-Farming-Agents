"""Version 2.0: thesis-aligned one-factor-at-a-time parameter sweeps.

Sweeps the six thesis parameters and reports the four pre-specified outcomes:
 rho_C = system-wide cooperator fraction
 q_C = mean fraction of cooperative neighbours among current cooperators
 mean_mu = average streak-tier multiplier
 mean_omega = mean observable network wealth

Uses common random seeds across values within each treatment where possible. This is
an exploratory sensitivity analysis, not a global optimum search.
"""
from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd
from vfarm_abm_v2 import load_parameters, VerticalFarmingModel

VERSION = "2.0"
METRICS = ["final_rho_C", "final_q_C", "final_mean_mu", "final_mean_omega"]
# Ranges include the baseline and stay within interpretable/non-negative ranges.
DEFAULT_GRIDS = {
    "alpha": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],                 # agricultural synergy
    "lambda_L": [1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00],       # loss aversion
    "w": [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50],              # digital exchange rate
    "p_base": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],                # digital base reward
    "gamma": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],          # sensitivity rate
    "p_rewire": [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],       # rewiring rate
}
TREATMENTS = {
    "T0_Static_NoInc": (0.0, 0.0),
    "T1_Fluid_NoInc": (None, 0.0),
    "T2_Static_Active": (0.0, None),
    "T3_Fluid_Active": (None, None),
}


def run_one(params, treatment, seed, rounds):
    p_rewire, w = TREATMENTS[treatment]
    model = VerticalFarmingModel(params,
        p_rewire=params["p_rewire"] if p_rewire is None else p_rewire,
        w=params["w"] if w is None else w, seed=seed)
    for _ in range(rounds):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    ex_post = model.compute_ex_post_metrics()
    return {
        "treatment": treatment, "seed": seed,
        "final_rho_C": float(df["rho_C"].iloc[-1]),
        "final_q_C": float(df["q_C"].iloc[-1]),
        "final_mean_mu": float(df["mean_mu"].iloc[-1]),
        "final_mean_omega": float(df["mean_omega"].iloc[-1]),
        "ROI_platform": ex_post["ROI_platform"],
        "mean_p_final": ex_post["mean_p_final"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=Path(__file__).resolve().with_name("parameters.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--rounds", type=int, default=None, help="Override T in parameters.csv")
    parser.add_argument("--seeds", type=int, default=None, help="Override S_max in parameters.csv")
    parser.add_argument("--quick", action="store_true", help="Use 10 seeds for a quick pipeline test")
    args = parser.parse_args()
    params = load_parameters(args.parameters)
    rounds = args.rounds or int(params["T"])
    seeds = 10 if args.quick else (args.seeds or int(params["S_max"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Explicit baseline factorial estimates allow parameter effects to be compared
    # against all four treatment configurations at each value.
    records = []
    cells = sum(len(v) for v in DEFAULT_GRIDS.values()) * len(TREATMENTS) * seeds
    print(f"run_sweeps.py v{VERSION}; rounds={rounds}, seeds={seeds}; planned runs={cells}")
    count = 0
    for parameter, values in DEFAULT_GRIDS.items():
        for value in values:
            p_run = params.copy()
            p_run[parameter] = float(value)
            for treatment in TREATMENTS:
                for seed in range(1, seeds + 1):
                    result = run_one(p_run, treatment, seed, rounds)
                    result.update({"swept_parameter": parameter, "swept_value": float(value),
                                   "baseline_value": float(params[parameter]), "version": VERSION})
                    records.append(result)
                    count += 1
                    if count == 1 or count % max(1, cells // 20) == 0 or count == cells:
                        print(f"[{count}/{cells}] {parameter}={value}, {treatment}, seed={seed}")
    raw = pd.DataFrame(records)
    raw_path = args.output_dir / "sensitivity_sweeps_v2.csv"
    raw.to_csv(raw_path, index=False)

    # Means and standard errors over seeds, preserving each treatment/parameter cell.
    group_cols = ["swept_parameter", "swept_value", "baseline_value", "treatment"]
    agg = raw.groupby(group_cols)[METRICS + ["ROI_platform", "mean_p_final"]].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    summary_path = args.output_dir / "sensitivity_sweeps_summary_v2.csv"
    agg.to_csv(summary_path, index=False)

    # Wide treatment comparison per parameter/value; T3 only counts as a joint win
    # if both pre-specified cooperation metrics exceed each comparator's means.
    means = raw.groupby(["swept_parameter", "swept_value", "treatment"])[["final_rho_C", "final_q_C"]].mean().reset_index()
    wins = []
    for (parameter, value), frame in means.groupby(["swept_parameter", "swept_value"]):
        by = frame.set_index("treatment")
        t3 = by.loc["T3_Fluid_Active"]
        comparators = ["T0_Static_NoInc", "T1_Fluid_NoInc", "T2_Static_Active"]
        wins.append({"swept_parameter": parameter, "swept_value": value,
                     "T3_rho_C_mean": t3["final_rho_C"], "T3_q_C_mean": t3["final_q_C"],
                     "T3_beats_all_on_both": all(
                         t3["final_rho_C"] > by.loc[t, "final_rho_C"] and
                         t3["final_q_C"] > by.loc[t, "final_q_C"] for t in comparators)})
    wins_df = pd.DataFrame(wins)
    wins_path = args.output_dir / "sensitivity_sweeps_T3_comparison_v2.csv"
    wins_df.to_csv(wins_path, index=False)

    print("\nSweep summary saved:")
    print(f"  Raw results: {raw_path}")
    print(f"  Metric summary: {summary_path}")
    print(f"  T3 comparison: {wins_path}")
    print("\nT3 joint wins across all three comparators:", int(wins_df["T3_beats_all_on_both"].sum()), "/", len(wins_df))
    print("Metrics: rho_C=system-wide cooperator fraction; q_C=mean cooperative-neighbour ratio among cooperators;")
    print("mean_mu=average streak tier multiplier; mean_omega=mean observable network wealth.")
    print("Caution: default sweep is one-factor-at-a-time; it does not identify interactions between parameters.")

if __name__ == "__main__":
    main()
