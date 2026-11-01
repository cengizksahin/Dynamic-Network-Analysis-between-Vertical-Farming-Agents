import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vfarm_abm import VerticalFarmingModel, load_parameters

def run_univariate_sweeps(csv_path="parameters.csv", output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    base_params = load_parameters(csv_path)
    T_rounds = int(base_params["T"])
    S_max = int(base_params["S_max"])

    # Section 4.2.3 & Table 8 Parameter Sweep Grids
    sweep_grids = {
        "p_rewire": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
        "lambda_L": [1.00, 1.50, 2.00, 2.25, 2.50, 3.00, 3.50],
        "alpha":    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "w":        [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
        "p_base":   [2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00],
        "gamma":    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    }

    sweep_rows = []
    for param_name, grid_values in sweep_grids.items():
        for val in grid_values:
            print(f"Sweeping {param_name} = {val:.2f} across {S_max} seeds...")
            p_copy = copy.deepcopy(base_params)
            p_copy[param_name] = val

            for seed in range(1, S_max + 1):
                model = VerticalFarmingModel(p_copy, seed=seed)
                for _ in range(T_rounds):
                    model.step()
                df_m = model.datacollector.get_model_vars_dataframe()
                ex_post = model.compute_ex_post_metrics()
                sweep_rows.append({
                    "swept_parameter": param_name,
                    "parameter_value": val,
                    "seed": seed,
                    "final_rho_C": df_m["rho_C"].iloc[-1],
                    "final_q_C": df_m["q_C"].iloc[-1],
                    "final_mu": df_m["mean_mu"].iloc[-1],
                    "final_omega": df_m["mean_omega"].iloc[-1],
                    "ROI_platform": ex_post["ROI_platform"],
                    "mean_p_final": ex_post["mean_p_final"],
                })

    df_sweeps = pd.DataFrame(sweep_rows)
    out_csv = os.path.join(output_dir, "sensitivity_sweeps.csv")
    df_sweeps.to_csv(out_csv, index=False)
    print(f"Saved all 6 sensitivity sweeps to {out_csv}")
    return df_sweeps


def plot_factorial_trajectories(output_dir="./output"):
    """Plots the 4 primary dependent metrics across T0, T1, T2, T3 with 95% CI bands."""
    df_ts = pd.read_csv(os.path.join(output_dir, "factorial_timeseries.csv"))
    metrics = [
        ("rho_C", r"System-Wide Cooperator Fraction ($\rho_{C,t}$)"),
        ("q_C", r"Mean Cooperator Neighborhood Ratio ($q_{C,t}$)"),
        ("mean_mu", r"Average Streak Tier Multiplier ($\bar{\mu}_t$)"),
        ("mean_omega", r"Mean Observable Network Wealth ($\bar{\Omega}_t$, EUR)"),
    ]
    colors = {
        "T0_Static_NoInc": "tab:red",
        "T1_Fluid_NoInc": "tab:orange",
        "T2_Static_Active": "tab:blue",
        "T3_Fluid_Active": "tab:green",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (col, title) in zip(axes.flat, metrics):
        grouped = df_ts.groupby(["treatment", "round"])[col].agg(["mean", "std", "count"]).reset_index()
        grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"])

        for t_name, df_sub in grouped.groupby("treatment"):
            ax.plot(df_sub["round"], df_sub["mean"], label=t_name, color=colors[t_name], linewidth=2)
            ax.fill_between(
                df_sub["round"],
                df_sub["mean"] - df_sub["ci95"],
                df_sub["mean"] + df_sub["ci95"],
                color=colors[t_name], alpha=0.18
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Simulation Round (t)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "factorial_treatments_plot.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_factorial_trajectories()
    run_univariate_sweeps()