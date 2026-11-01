"""Parameter sweep for the vertical-farming ABM.

Run from the folder containing this file, vfarm_abmV3.py, and parameters.csv:
    python run_sweeps.py

The script compares T0-T3 at every (p_rewire, w) setting using the same seeds.
It writes sensitivity_sweeps.csv plus a compact best-settings summary.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-module", default="vfarm_abmV3",
                        help="Python module containing VerticalFarmingModel and load_parameters")
    parser.add_argument("--parameters", type=Path, default=None,
                        help="Path to parameters.csv (default: script folder, then ../Code/parameters.csv)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: ./output beside this script)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Monte Carlo seeds per treatment/configuration (default: min(S_max, 30))")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Rounds per run (default: T from parameters.csv)")
    parser.add_argument("--p-values", default="0,0.05,0.1,0.2,0.3,0.4",
                        help="Comma-separated rewiring probabilities in [0,1]")
    parser.add_argument("--w-multipliers", default="0,0.5,0.75,1,1.25,1.5,2",
                        help="Comma-separated multipliers of baseline parameter w")
    return parser.parse_args()


def csv_floats(text, name, low=None, high=None):
    try:
        values = [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid {name}: {text!r}") from exc
    if not values:
        raise SystemExit(f"{name} must contain at least one value")
    if low is not None and any(x < low for x in values):
        raise SystemExit(f"All {name} values must be >= {low}")
    if high is not None and any(x > high for x in values):
        raise SystemExit(f"All {name} values must be <= {high}")
    return sorted(set(values))


def locate_parameters(script_dir: Path, explicit: Path | None) -> Path:
    if explicit:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Parameters file not found: {path}")
        return path
    candidates = [script_dir / "parameters.csv", script_dir.parent / "Code" / "parameters.csv"]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise SystemExit(
        "Could not find parameters.csv. Put it beside run_sweeps.py or pass --parameters '...\\parameters.csv'."
    )


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    parameters_path = locate_parameters(script_dir, args.parameters)
    output_dir = (args.output_dir or (script_dir / "output")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_module = importlib.import_module(args.model_module)
        ModelClass = getattr(model_module, "VerticalFarmingModel")
        load_parameters = getattr(model_module, "load_parameters")
    except (ImportError, AttributeError) as exc:
        raise SystemExit(
            f"Could not import model from {args.model_module!r}. Ensure {args.model_module}.py is "
            "in the same folder and defines VerticalFarmingModel and load_parameters. "
            "Use --model-module vfarm_abm if that is your filename.\n"
            f"Details: {exc}"
        ) from exc

    params = load_parameters(str(parameters_path))
    base_p = float(params["p_rewire"])
    base_w = float(params["w"])
    rounds = int(args.rounds if args.rounds is not None else params["T"])
    n_seeds = int(args.seeds if args.seeds is not None else min(int(params["S_max"]), 30))
    if rounds < 1 or n_seeds < 1:
        raise SystemExit("--rounds and --seeds must be positive integers")

    p_values = csv_floats(args.p_values, "p-values", 0.0, 1.0)
    w_multipliers = csv_floats(args.w_multipliers, "w-multipliers", 0.0)
    w_values = sorted(set(round(base_w * m, 10) for m in w_multipliers))

    treatments = ("T0_Static_NoInc", "T1_Fluid_NoInc", "T2_Static_Active", "T3_Fluid_Active")
    records = []
    total = len(p_values) * len(w_values) * len(treatments) * n_seeds
    run_number = 0
    print(f"Parameters: {parameters_path}")
    print(f"Baseline p_rewire={base_p:g}, w={base_w:g}; rounds={rounds}, seeds={n_seeds}")
    print(f"Grid: {len(p_values)} rewiring values x {len(w_values)} incentive values")
    print(f"Runs planned: {total}")

    # Each p/w cell compares all four factorial treatments. Shared seed IDs make
    # the comparisons less sensitive to random initialization differences.
    for p in p_values:
        for w in w_values:
            cfgs = {
                "T0_Static_NoInc": (0.0, 0.0),
                "T1_Fluid_NoInc": (p, 0.0),
                "T2_Static_Active": (0.0, w),
                "T3_Fluid_Active": (p, w),
            }
            for treatment in treatments:
                p_run, w_run = cfgs[treatment]
                for seed in range(1, n_seeds + 1):
                    run_number += 1
                    if run_number == 1 or run_number % max(1, total // 20) == 0 or run_number == total:
                        print(f"[{run_number}/{total}] p={p:g}, w={w_run:g}, {treatment}, seed={seed}")
                    model = ModelClass(params, p_rewire=p_run, w=w_run, seed=seed)
                    for _ in range(rounds):
                        model.step()
                    ts = model.datacollector.get_model_vars_dataframe()
                    last = ts.iloc[-1]
                    ex_post = model.compute_ex_post_metrics() if hasattr(model, "compute_ex_post_metrics") else {}
                    records.append({
                        "p_rewire_sweep": p,
                        "w_sweep": w,
                        "w_multiplier": (w_run / base_w if base_w else np.nan),
                        "treatment": treatment,
                        "p_rewire_used": p_run,
                        "w_used": w_run,
                        "seed": seed,
                        "rounds": rounds,
                        "final_rho_C": float(last["rho_C"]),
                        "final_q_C": float(last["q_C"]),
                        "final_mu": float(last["mean_mu"]),
                        "final_omega": float(last["mean_omega"]),
                        **ex_post,
                    })

    raw = pd.DataFrame(records)
    raw_path = output_dir / "sensitivity_sweeps.csv"
    raw.to_csv(raw_path, index=False)

    # Summarize average outcomes per parameter cell and treatment.
    metrics = ["final_rho_C", "final_q_C", "final_mu", "ROI_platform"]
    available_metrics = [m for m in metrics if m in raw.columns]
    summary = (raw.groupby(["p_rewire_sweep", "w_sweep", "w_multiplier", "treatment"], as_index=False)
                   [available_metrics].mean())
    summary_path = output_dir / "sensitivity_sweeps_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Directly test the stated hypothesis: T3 exceeds each alternative on BOTH
    # metrics at the same parameter cell. Differences are paired by seed.
    wide = raw.pivot_table(
        index=["p_rewire_sweep", "w_sweep", "seed"],
        columns="treatment", values=["final_rho_C", "final_q_C"], aggfunc="mean"
    )
    comparison_rows = []
    for p in p_values:
        for w in w_values:
            cell = wide.xs((p, w), level=("p_rewire_sweep", "w_sweep"), drop_level=False)
            row = {"p_rewire_sweep": p, "w_sweep": w,
                   "w_multiplier": (w / base_w if base_w else np.nan)}
            all_beaten = True
            for metric in ("final_rho_C", "final_q_C"):
                t3 = cell[(metric, "T3_Fluid_Active")]
                for competitor in ("T0_Static_NoInc", "T1_Fluid_NoInc", "T2_Static_Active"):
                    diff = t3.to_numpy() - cell[(metric, competitor)].to_numpy()
                    row[f"delta_{metric}_vs_{competitor}"] = float(np.mean(diff))
                    row[f"T3_mean_higher_{metric}_vs_{competitor}"] = bool(np.mean(diff) > 0)
                    if np.mean(diff) <= 0:
                        all_beaten = False
            row["T3_beats_all_on_both_mean_metrics"] = all_beaten
            comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "sensitivity_sweeps_T3_test.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n--- Best parameter cells by mean population-wide cooperation (rho_C) ---")
    print(summary.sort_values("final_rho_C", ascending=False).head(10).to_string(index=False))
    winners = comparison[comparison["T3_beats_all_on_both_mean_metrics"]]
    print(f"\nParameter cells where T3 beats T0, T1 and T2 on BOTH mean metrics: {len(winners)} / {len(comparison)}")
    if not winners.empty:
        print(winners[["p_rewire_sweep", "w_sweep", "w_multiplier"]].to_string(index=False))
    print("\nSaved files:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {comparison_path}")
    print("Note: local clustering is operationalized as q_C, the model's mean fraction of cooperative neighbors among cooperators.")


if __name__ == "__main__":
    main()
