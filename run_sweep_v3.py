"""
run_sweep_v3.py
Two-dimensional population-composition experiment for vfarm_abm_v3.py.

Sweeps:
- Initial cooperator fraction p_C0 (C/D composition)
- Initial Type A fraction p_TypeA (Type A/Type B composition)

Each composition cell is crossed with all four existing treatments and the
same seed set. Outputs:
  population_sweep_v3.csv          one row per treatment x composition x seed
  population_sweep_summary_v3.csv  mean/std/SEM by treatment x composition
  population_sweep_timeseries_v3.csv (optional; disabled by default)

Run from the folder containing vfarm_abm_v3.py and parameters.csv:
  python run_sweep_v3.py
"""
from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd

from vfarm_abm_v3 import load_parameters, VerticalFarmingModel


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_C_RATIOS = [0.10, 0.25, 0.50, 0.75, 0.90]
DEFAULT_TYPE_A_RATIOS = [0.10, 0.25, 0.50, 0.75, 0.90]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default=str(SCRIPT_DIR / "parameters.csv"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output_v3"))
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override S_max for quick tests; defaults to parameters.csv S_max.")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override T for quick tests; defaults to parameters.csv T.")
    parser.add_argument("--c-ratios", default=",".join(map(str, DEFAULT_C_RATIOS)))
    parser.add_argument("--type-a-ratios", default=",".join(map(str, DEFAULT_TYPE_A_RATIOS)))
    parser.add_argument("--save-timeseries", action="store_true")
    args = parser.parse_args()

    params = load_parameters(args.parameters)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = int(args.seeds if args.seeds is not None else params["S_max"])
    rounds = int(args.rounds if args.rounds is not None else params["T"])
    c_ratios = [float(x.strip()) for x in args.c_ratios.split(",") if x.strip()]
    type_a_ratios = [float(x.strip()) for x in args.type_a_ratios.split(",") if x.strip()]
    for label, values in (("C/D", c_ratios), ("Type A/B", type_a_ratios)):
        if not values or any(v < 0 or v > 1 for v in values):
            raise ValueError(f"All {label} ratios must be in [0, 1].")
    treatments = {
        "T0_Static_NoInc": {"p_rewire": 0.0, "w": 0.0},
        "T1_Fluid_NoInc": {"p_rewire": params["p_rewire"], "w": 0.0},
        "T2_Static_Active": {"p_rewire": 0.0, "w": params["w"]},
        "T3_Fluid_Active": {"p_rewire": params["p_rewire"], "w": params["w"]},
    }
    total = len(c_ratios) * len(type_a_ratios) * len(treatments) * seeds
    print(f"v3 population sweep: {len(c_ratios)} C/D ratios x "
          f"{len(type_a_ratios)} Type A/B ratios x {len(treatments)} treatments "
          f"x {seeds} seeds = {total:,} runs; {rounds} rounds/run.")
    print("Metrics: rho_C, q_C, mean_mu, mean_omega, ROI_platform, mean_p_final")

    records, ts_records = [], []
    run_number = 0
    for p_c, p_a, (treatment, cfg), seed in itertools.product(
            c_ratios, type_a_ratios, treatments.items(), range(1, seeds + 1)):
        run_number += 1
        model = VerticalFarmingModel(
            params, p_rewire=cfg["p_rewire"], w=cfg["w"],
            seed=seed, p_C0=p_c, p_TypeA=p_a
        )
        for _ in range(rounds):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        ex_post = model.compute_ex_post_metrics()
        records.append({
            "p_C0": p_c,
            "initial_defector_fraction": 1.0 - p_c,
            "p_TypeA": p_a,
            "initial_type_B_fraction": 1.0 - p_a,
            "treatment": treatment,
            "p_rewire": cfg["p_rewire"],
            "w": cfg["w"],
            "seed": seed,
            "rounds": rounds,
            "final_rho_C": float(df["rho_C"].iloc[-1]),
            "final_q_C": float(df["q_C"].iloc[-1]),
            "final_mu": float(df["mean_mu"].iloc[-1]),
            "final_omega": float(df["mean_omega"].iloc[-1]),
            **ex_post,
        })
        if args.save_timeseries:
            df = df.copy()
            df["round"] = np.arange(1, len(df) + 1)
            df["p_C0"], df["p_TypeA"] = p_c, p_a
            df["treatment"], df["seed"] = treatment, seed
            ts_records.append(df)
        if run_number % max(1, total // 20) == 0 or run_number == total:
            print(f"[{run_number:,}/{total:,}] p_C0={p_c:.2f}, "
                  f"p_TypeA={p_a:.2f}, {treatment}, seed={seed}")

    raw_path = output_dir / "population_sweep_v3.csv"
    df_raw = pd.DataFrame(records)
    df_raw.to_csv(raw_path, index=False)

    metric_cols = ["final_rho_C", "final_q_C", "final_mu", "final_omega",
                   "ROI_platform", "mean_p_final"]
    grouped = df_raw.groupby(["p_C0", "p_TypeA", "treatment"], dropna=False)[metric_cols]
    summary = grouped.agg(["mean", "std", "count"])
    # Flatten columns and add standard error for each metric.
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    for metric in metric_cols:
        count_col = f"{metric}_count"
        std_col = f"{metric}_std"
        summary[f"{metric}_sem"] = summary[std_col] / np.sqrt(summary[count_col])
    summary = summary.reset_index()
    summary.to_csv(output_dir / "population_sweep_summary_v3.csv", index=False)

    if args.save_timeseries:
        pd.concat(ts_records, ignore_index=True).to_csv(
            output_dir / "population_sweep_timeseries_v3.csv", index=False
        )

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {output_dir / 'population_sweep_summary_v3.csv'}")
    if args.save_timeseries:
        print(f"  {output_dir / 'population_sweep_timeseries_v3.csv'}")
    print("\nSuggested first comparison: filter to T2_Static_Active and "
          "T3_Fluid_Active, then compare mean final_rho_C and final_q_C "
          "across both composition dimensions.")


if __name__ == "__main__":
    main()
