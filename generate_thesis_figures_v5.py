#!/usr/bin/env python3
"""
Reproduce the V5 thesis figures directly from the ORIGINAL run-level V5 CSV files.

Run with the .py file in the SAME directory as the CSVs:
    python generate_thesis_figures_v5.py

The working directory does not matter (useful when launched from VS Code).
Optional:
    python generate_thesis_figures_v5.py --input-dir "C:/path/to/CSVs" --dpi 300

No simulation is executed and no source CSV is edited. Outputs are saved under
<csv_directory>/figures_v5/{png,pdf,plot_data}/.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

REQUIRED = {
    "timeseries": "factorial_timeseries_v5.csv",
    "factorial": "factorial_ex_post_summary_v5.csv",
    "population": "population_sweep_v5.csv",
    "sensitivity": "sensitivity_sweeps_v5.csv",
    "grid": "sensitivity_factorial_grid_v5.csv",
}
TREATMENTS = ["T0_Static_NoInc", "T1_Fluid_NoInc", "T2_Static_Active", "T3_Fluid_Active"]
SHORT = {tr: f"T{i}" for i, tr in enumerate(TREATMENTS)}
COLORS = dict(zip(TREATMENTS, ["#64748b", "#2563eb", "#ca8a04", "#15803d"]))
SENSITIVITY_LABELS = {
    "alpha": "Agricultural synergy α",
    "lambda_L": "Loss aversion λL",
    "w": "Digital exchange rate w",
    "p_base": "Digital base reward",
    "gamma": "Streak sensitivity γ",
    "p_rewire": "Rewiring probability",
}


def require(df: pd.DataFrame, cols: list[str], filename: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{filename}: missing required column(s): {', '.join(missing)}")


def read_inputs(root: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED.values() if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required CSV file(s) missing from {root}: {', '.join(missing)}\n"
            "Put this Python script BESIDE the extracted V5 CSV files; "
            "or use --input-dir to specify their location."
        )
    frames = {key: pd.read_csv(root / filename, low_memory=False) for key, filename in REQUIRED.items()}
    column_map = {
        "timeseries": ["round", "treatment", "seed", "rho_C", "q_C", "mean_mu", "mean_omega"],
        "factorial": ["treatment", "seed", "final_rho_C", "final_q_C", "final_mu", "final_omega", "w", "mean_p_final", "mean_R_c_final", "ROI_platform",
                      "final_n_cooperators_predecision", "final_n_defectors_predecision"],
        "population": ["treatment", "seed", "p_C0", "p_TypeA", "final_rho_C", "final_q_C"],
        "sensitivity": ["treatment", "seed", "swept_parameter", "parameter_value", "baseline_value", "final_rho_C", "final_q_C", "total_C_to_D_transitions", "total_D_to_C_transitions", "mean_connected_after"],
        "grid": ["treatment", "seed", "p_rewire_sweep", "w_sweep", "final_rho_C", "final_q_C", "ROI_platform"],
    }
    for key, d in frames.items():
        require(d, column_map[key], REQUIRED[key])
        unknown = set(d.treatment.dropna().unique()) - set(TREATMENTS)
        if unknown:
            raise ValueError(f"Unknown treatments in {REQUIRED[key]}: {unknown}")
        print(f"Loaded {REQUIRED[key]} ({len(d):,} rows × {len(d.columns)} columns)")
    # Detect accidental duplicated experiment keys rather than averaging them silently.
    keys = {
        "timeseries": ["seed", "treatment", "round"],
        "factorial": ["seed", "treatment"],
        "population": ["seed", "treatment", "p_C0", "p_TypeA"],
        "sensitivity": ["seed", "treatment", "swept_parameter", "parameter_value"],
        "grid": ["seed", "treatment", "p_rewire_sweep", "w_sweep"],
    }
    for key, cols in keys.items():
        n_dup = frames[key].duplicated(cols).sum()
        if n_dup:
            raise ValueError(f"{REQUIRED[key]} contains {n_dup} duplicate experiment keys ({cols}).")
    return frames


def mean_ci(df: pd.DataFrame, groups: list[str], val: str) -> pd.DataFrame:
    result = (df.groupby(groups, dropna=False)[val]
              .agg(mean="mean", sd="std", n="count").reset_index())
    result["sem"] = result["sd"] / np.sqrt(result["n"])
    result["half_ci"] = student_t.ppf(0.975, result["n"] - 1) * result["sem"]
    result["ci_low"] = result["mean"] - result["half_ci"]
    result["ci_high"] = result["mean"] + result["half_ci"]
    return result


def paired_estimate(wide: pd.DataFrame, first: str, second: str) -> tuple[float, float, float, int]:
    d = (wide[first] - wide[second]).dropna()
    n = len(d)
    if n < 2:
        raise ValueError(f"Not enough matched seeds for {first} minus {second}: n={n}")
    m = float(d.mean());se = float(d.std(ddof=1) / np.sqrt(n))
    h = float(student_t.ppf(.975, n - 1) * se)
    return m, m - h, m + h, n


def save_plot(fig: plt.Figure, output: Path, name: str, dpi: int, manifest: list[dict],
              *, section: str, sources: str, description: str, formats: list[str]) -> None:
    for ext in formats:
        d = output / ext
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.{ext}", dpi=dpi if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append(dict(section=section, figure=name, source_csv_files=sources,
                         description=description, png_or_pdf=", ".join(formats)))


def save_data(frame: pd.DataFrame, output: Path, name: str) -> None:
    d = output / "plot_data";d.mkdir(parents=True, exist_ok=True)
    frame.to_csv(d / f"{name}_plotted_values.csv", index=False, float_format="%.12g")


def figure_51(frames, output, dpi, manifest, formats):
    """Figure 5.1 — four-panel trajectories, t(0.975,n-1) ribbons BY SEED."""
    ts = frames["timeseries"]
    panels = [("rho_C", "System-wide cooperator fraction", "Proportion"),
              ("q_C", "Cooperator-neighborhood ratio", "Proportion"),
              ("mean_mu", "Mean streak tier μ", "Tier"),
              ("mean_omega", "Mean observable wealth Ω", "Model €-equivalent")]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), sharex=True)
    all_data = []
    for ax, (var, title, ylabel) in zip(axes.flat, panels):
        out = mean_ci(ts, ["round", "treatment"], var)
        out["metric"] = var
        all_data.append(out)
        for tr in TREATMENTS:
            sub = out[out.treatment == tr].sort_values("round")
            ax.plot(sub["round"].to_numpy(), sub["mean"].to_numpy(),
                    label=SHORT[tr], color=COLORS[tr], linewidth=1.9)
            ax.fill_between(sub["round"].to_numpy(), sub["ci_low"].to_numpy(),
                            sub["ci_high"].to_numpy(), color=COLORS[tr], alpha=0.16)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, int(ts["round"].max()))
        ax.grid(alpha=.2)
    axes[1, 0].set_xlabel("Recorded round")
    axes[1, 1].set_xlabel("Recorded round")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, -.007), frameon=False)
    fig.suptitle("V5 baseline: mean ± 95% confidence intervals across independent seeds", fontsize=13)
    fig.tight_layout(rect=[0, .055, 1, .96])
    name = "Figure_5_1_baseline_ribbons"
    save_plot(fig, output, name, dpi, manifest, section="5.1",
              sources=REQUIRED["timeseries"],
              description="Four original outcome trajectories with per-round Student-t seed-level ribbons", formats=formats)
    save_data(pd.concat(all_data, ignore_index=True), output, name)


def figure_52(frames, output, dpi, manifest, formats):
    """Figure 5.2 — forest plot for the two rewiring contrasts plus 2×2 interaction."""
    f = frames["factorial"]
    wide = f.pivot(index="seed", columns="treatment", values="final_rho_C")
    labels = ["Rewiring, no incentives: T1 − T0", "Rewiring, active incentives: T3 − T2", "Factorial interaction (difference-in-differences)"]
    series = [wide[TREATMENTS[1]] - wide[TREATMENTS[0]],
              wide[TREATMENTS[3]] - wide[TREATMENTS[2]],
              (wide[TREATMENTS[3]] - wide[TREATMENTS[2]]) - (wide[TREATMENTS[1]] - wide[TREATMENTS[0]])]
    rows = []
    for label, d in zip(labels, series):
        d = d.dropna();m = d.mean();se=d.std(ddof=1)/np.sqrt(len(d));h=student_t.ppf(.975,len(d)-1)*se
        rows.append({"contrast":label, "estimate":m,"ci_low":m-h,"ci_high":m+h,"n":len(d)})
    plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    yy = np.arange(len(plot))
    est = 100 * plot.estimate.to_numpy();low = 100*plot.ci_low.to_numpy();high=100*plot.ci_high.to_numpy()
    ax.errorbar(est, yy, xerr=np.vstack([est-low, high-est]), fmt="o",capsize=4,
                linewidth=2, markersize=6, color="#334155")
    ax.axvline(0, ls="--", lw=1, color="gray")
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Final cooperation difference (percentage points), paired 95% Student-t CI")
    ax.set_title("Simple rewiring effects and their factorial interaction")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    name="Figure_5_2_paired_factorial_effects"
    save_plot(fig,output,name,dpi,manifest,section="5.2",sources=REQUIRED["factorial"],
              description="Two paired rewiring contrasts and interaction; 100 matched seeds",formats=formats)
    save_data(plot,output,name)


def draw_heatmap(data: pd.DataFrame, *, title: str, xlab: str, ylab: str,
                 val_label: str, percent: bool = True, vlim: float | None = None,
                 annotate: bool = True) -> plt.Figure:
    """Diverging heatmap; incoming data are FRACTIONS unless otherwise noted."""
    mat = data.to_numpy(dtype=float)
    z = mat * 100 if percent else mat
    valid = z[np.isfinite(z)]
    if not valid.size:
        raise ValueError(f"No finite observations for heatmap {title}")
    vmax = float(vlim) if vlim is not None else float(np.max(np.abs(valid)))
    if vmax <= 0: vmax = 1
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    masked=np.ma.masked_invalid(z)
    im=ax.imshow(masked, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(data.shape[1]), [f"{x:g}" for x in data.columns])
    ax.set_yticks(np.arange(data.shape[0]), [f"{x:g}" for x in data.index])
    ax.set_xlabel(xlab);ax.set_ylabel(ylab)
    ax.set_title(title)
    if annotate:
        for (i,j),value in np.ndenumerate(z):
            if np.isfinite(value):
                ax.text(j,i,f"{value:+.1f}" if percent else f"{value:+.2f}",
                        ha="center",va="center",fontsize=8.7,
                        color="white" if abs(value)>vmax*.68 else "black")
            else:ax.text(j,i,"N/A",ha="center",va="center",fontsize=8)
    fig.colorbar(im, ax=ax, label=val_label, fraction=.045, pad=.04)
    fig.tight_layout()
    return fig


def figure_53(frames, output, dpi, manifest, formats):
    """Figure 5.3 — cell means from 25 population-composition cells, NOT seed pooling."""
    pop = frames["population"]
    grouped = pop.groupby(["p_C0", "p_TypeA", "treatment"])[["final_rho_C","final_q_C"]].mean()
    for metric, code, desc in [("final_rho_C", "Figure_5_3_population_heatmap", "Final cooperation"),
                               ("final_q_C", "Appendix_5_3_population_neighborhood_heatmap", "Cooperator-neighborhood fraction")]:
        tmp=grouped[metric].unstack("treatment")
        grid=(tmp[TREATMENTS[3]]-tmp[TREATMENTS[2]]).unstack("p_TypeA")
        fig=draw_heatmap(grid,title=f"Population composition: T3 − T2 {desc.lower()}",
                         xlab="Initial Type A fraction",ylab="Initial cooperator fraction",
                         val_label="T3 − T2 (percentage points)")
        save_plot(fig,output,code,dpi,manifest,section="5.3",sources=REQUIRED["population"],
                  description=f"Paired-composition cell mean difference in {metric}",formats=formats)
        save_data(grid.stack().rename("T3_minus_T2").reset_index(),output,code)


def figure_54(frames, output, dpi, manifest, formats):
    """Figure 5.4 — six original univariate parameter sweeps with seed uncertainty."""
    sw=frames["sensitivity"]
    expected=set(SENSITIVITY_LABELS)
    actual=set(sw.swept_parameter.unique())
    if actual!=expected:
        raise ValueError(f"Sensitivity parameter set differs from expected: missing={expected-actual}; extra={actual-expected}")
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.4))
    all_data=[]
    for ax,param in zip(axes.flat,SENSITIVITY_LABELS):
        df=sw[sw.swept_parameter==param]
        result=mean_ci(df,["parameter_value","treatment"],"final_rho_C")
        result.insert(0,"swept_parameter",param)
        all_data.append(result)
        for tr in TREATMENTS[2:]:
            ss=result[result.treatment==tr].sort_values("parameter_value")
            x=ss.parameter_value.to_numpy();m=ss["mean"].to_numpy()
            ax.plot(x,m,"o-",color=COLORS[tr],markersize=4,label=SHORT[tr],linewidth=1.8)
            ax.fill_between(x,ss.ci_low.to_numpy(),ss.ci_high.to_numpy(),alpha=.13,color=COLORS[tr])
        ax.set_title(SENSITIVITY_LABELS[param],fontsize=11)
        ax.set_ylim(-.04,1.04)
        ax.grid(alpha=.2)
        ax.set_xlabel("Parameter value")
    axes[0,0].set_ylabel("Final cooperator fraction")
    axes[1,0].set_ylabel("Final cooperator fraction")
    fig.suptitle("Six V5 univariate sensitivity sweeps — active-incentive treatments",fontsize=13)
    handles, labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.012),ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.055,1,.95])
    name="Figure_5_4_sensitivity_panels"
    save_plot(fig,output,name,dpi,manifest,section="5.4",sources=REQUIRED["sensitivity"],
              description="Six univariate cooperation response curves with 95% seed-level CI",formats=formats)
    save_data(pd.concat(all_data,ignore_index=True),output,name)


def figure_55(frames, output, dpi, manifest, formats):
    """Figure 5.5 — actual bivariate p_rewire_sweep × w_sweep response surfaces."""
    grid=frames["grid"]
    grp=grid.groupby(["p_rewire_sweep","w_sweep","treatment"])[["final_rho_C","final_q_C","ROI_platform"]].mean()
    for metric,name,description,is_percent in [
        ("final_rho_C","Figure_5_5_rewiring_incentive_heatmap","final population cooperation",True),
        ("final_q_C","Appendix_5_5_rewiring_incentive_neighborhood_heatmap","local cooperator-neighborhood ratio",True),
        ("ROI_platform","Appendix_5_5_rewiring_incentive_ROI_heatmap","reported model ratio (not financial ROI)",False)]:
        w=grp[metric].unstack("treatment")
        res=(w[TREATMENTS[3]]-w[TREATMENTS[2]]).unstack("w_sweep")
        fig=draw_heatmap(res,title=f"Rewiring × incentives: T3 − T2 {description}",
                         xlab="Digital exchange rate w",ylab="Rewiring probability p_rewire",
                         val_label="T3 − T2 (percentage points)" if is_percent else "T3 − T2 (ratio units)",
                         percent=is_percent)
        save_plot(fig,output,name,dpi,manifest,section="5.5",sources=REQUIRED["grid"],
                  description=f"T3−T2 parameter-grid cell mean differences for {metric}; w=0 ROI undefined",formats=formats)
        save_data(res.rename_axis(index="p_rewire_sweep",columns="w_sweep").reset_index()
                  .melt(id_vars="p_rewire_sweep",var_name="w_sweep",value_name="T3_minus_T2"),output,name)


def baseline_diagnostics(sw: pd.DataFrame) -> pd.DataFrame:
    # Select exact baseline cell from the original univariate REWIRING sweep.
    # Its baseline p_rewire is read from 'baseline_value' in the CSV (not guessed).
    rows=sw[(sw.swept_parameter=="p_rewire") &
            np.isclose(sw.parameter_value,sw.baseline_value,atol=1e-10)]
    if rows.empty:
        raise ValueError("No baseline p_rewire cell in sensitivity_sweeps_v5.csv")
    if rows.groupby("treatment").seed.nunique().min()<2:
        raise ValueError("Baseline diagnostics require at least 2 seeds per treatment")
    return rows


def figure_56(frames, output, dpi, manifest, formats):
    """Figure 5.6 — descriptive strategy transitions and graph connectedness."""
    d=baseline_diagnostics(frames["sensitivity"])
    cols=["total_C_to_D_transitions","total_D_to_C_transitions","mean_connected_after",
          "mean_type_A_threshold_met_rate","final_type_A_mean_q_star","final_q_C",
          "final_mean_mu_applied","final_mean_total_marginal_incentive_cooperators",
          "total_existing_streak_reset_events","total_successful_edge_swaps"]
    require(d,cols,REQUIRED["sensitivity"])
    vals=d.groupby("treatment")[cols].mean().reindex(TREATMENTS)
    x=np.arange(4)
    fig,ax=plt.subplots(1,2,figsize=(11.3,4.1))
    ax[0].bar(x-.2,vals.total_C_to_D_transitions.to_numpy(),width=.38,label="C → D")
    ax[0].bar(x+.2,vals.total_D_to_C_transitions.to_numpy(),width=.38,label="D → C")
    ax[0].set_xticks(x,["T0","T1","T2","T3"]);ax[0].set_ylabel("Transitions per 100-round run")
    ax[0].set_title("Cumulative strategy switches (all agent types)");ax[0].legend()
    ax[1].bar(x,vals.mean_connected_after.to_numpy()*100,width=.6)
    ax[1].set_xticks(x,["T0","T1","T2","T3"]);ax[1].set_ylim(0,110)
    ax[1].set_title("Rounds with a connected network")
    ax[1].set_ylabel("Connected rounds (%)")
    for a in ax:a.grid(axis="y",alpha=.2)
    fig.tight_layout()
    name="Figure_5_6_transition_connectivity"
    save_plot(fig,output,name,dpi,manifest,section="5.6",sources=REQUIRED["sensitivity"],
              description="Baseline all-agent strategy switches and connected-round fraction",formats=formats)
    save_data(vals.reset_index(),output,name)

    # Appendix diagnostic observations: separate averages with explicit denominator caveat.
    fig, ax = plt.subplots(2,2,figsize=(11,7.2))
    ax[0,0].bar(x-.2,vals.final_type_A_mean_q_star.to_numpy(),width=.4,label="Type A mean q* (final)")
    ax[0,0].bar(x+.2,vals.final_q_C.to_numpy(),width=.4,label="q_C among cooperators (final)")
    ax[0,0].set_title("Threshold and neighborhood: distinct populations")
    ax[0,0].legend(fontsize=8)
    ax[0,1].bar(x,100*vals.mean_type_A_threshold_met_rate.to_numpy())
    ax[0,1].set_title("Run-average Type A threshold-met rate")
    ax[0,1].set_ylabel("Threshold met (% of relevant observations)")
    left=ax[1,0]
    right=left.twinx()
    left.bar(x-.2,vals.final_mean_mu_applied.to_numpy(),width=.38,color="#2563eb",label="Applied tier")
    right.bar(x+.2,vals.final_mean_total_marginal_incentive_cooperators.to_numpy(),width=.38,color="#ea580c",label="M among cooperators")
    left.set_ylabel("Mean tier")
    right.set_ylabel("Conditional marginal reward (points)")
    left.set_title("Applied tiers and conditional reward (two scales)")
    lh,ll=left.get_legend_handles_labels();rh,rl=right.get_legend_handles_labels()
    left.legend(lh+rh,ll+rl,fontsize=8,loc="upper left")
    ax[1,1].bar(x,vals.total_existing_streak_reset_events.to_numpy())
    ax[1,1].set_title("Existing streak resets (not graduated sanctions)")
    ax[1,1].set_ylabel("Resets per 100-round run")
    for a in ax.flat:
        a.set_xticks(x,["T0","T1","T2","T3"])
        a.grid(axis="y",alpha=.17)
    fig.suptitle("Observed model mechanisms — panels use different denominators",fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95])
    name="Appendix_5_6_threshold_streak_dashboard"
    save_plot(fig,output,name,dpi,manifest,section="5.6 appendix",sources=REQUIRED["sensitivity"],
              description="Type A q*, cooperator q_C, threshold pass rate, applied tiers and resets",formats=formats)
    save_data(vals.reset_index(),output,name)

    rew=frames["sensitivity"][frames["sensitivity"].swept_parameter=="p_rewire"]
    summarize=mean_ci(rew,["parameter_value","treatment"],"mean_connected_after")
    fig,ax=plt.subplots(figsize=(8.4,4.4))
    for tr in TREATMENTS[1::2]:
        sub=summarize[summarize.treatment==tr].sort_values("parameter_value")
        x0=sub.parameter_value.to_numpy()
        ax.plot(x0,sub["mean"].to_numpy()*100,"o-",label=SHORT[tr],color=COLORS[tr])
        ax.fill_between(x0,sub.ci_low.to_numpy()*100,sub.ci_high.to_numpy()*100,alpha=.15,color=COLORS[tr])
    ax.set(xlabel="Rewiring probability",ylabel="Mean percentage of rounds connected",title="Network connectedness across rewiring intensities")
    ax.set_ylim(0,105);ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    name="Appendix_5_6_connectivity_by_rewiring"
    save_plot(fig,output,name,dpi,manifest,section="5.6 appendix",sources=REQUIRED["sensitivity"],
              description="Connected-round fractions across rewiring sensitivity sweep",formats=formats)
    save_data(summarize,output,name)


def figure_57(frames, output, dpi, manifest, formats):
    """Figure 5.7 — reconstruct *recorded* ROI numerator from run-level ratios.

    All figures reconstructed PER SEED, then averaged. This is not real-world
    financial ROI; nor does the numerator necessarily include every spillover.
    """
    f=frames["factorial"].copy()
    f["population_size"] = f["final_n_cooperators_predecision"] + f["final_n_defectors_predecision"]
    f["token_outlay"] = f["population_size"] * f["w"] * (f["mean_p_final"] + f["mean_R_c_final"])
    f["recorded_synergy_implied"] = f["token_outlay"] * f["ROI_platform"]
    f["recorded_margin"] = f["recorded_synergy_implied"] - f["token_outlay"]
    act=f[f.treatment.isin(TREATMENTS[2:])]
    grouped=act.groupby("treatment")[["token_outlay","recorded_synergy_implied","recorded_margin","ROI_platform"]].agg(["mean","std"])
    grouped.columns=["_".join(i) for i in grouped.columns]
    grouped=grouped.reindex(TREATMENTS[2:])
    x=np.arange(2)
    fig,ax=plt.subplots(figsize=(8.3,4.6))
    bw=.36
    ax.bar(x-bw/2,grouped.recorded_synergy_implied_mean.to_numpy()/1000,width=bw,
           label="ROI-implied recorded synergy")
    ax.bar(x+bw/2,grouped.token_outlay_mean.to_numpy()/1000,width=bw,
           label="Tokenomic outlay")
    ax.set_xticks(x,["T2 Static, active","T3 Fluid, active"])
    ax.set_ylabel("Model €-equivalent, thousands (100 agents × 100 rounds)")
    ax.set_title("Recorded synergy and platform-token outlay: model accounting only")
    ax.legend();ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    name="Figure_5_7_accounting_comparison"
    save_plot(fig,output,name,dpi,manifest,section="5.7",sources=REQUIRED["factorial"],
              description="Per-seed reconstructed recorded synergy and token outlay, then averaged",formats=formats)
    save_data(grouped.reset_index(),output,name)


def main() -> int:
    parser=argparse.ArgumentParser(description="Generate V5 thesis figures directly from SAME-DIRECTORY CSVs")
    parser.add_argument("--input-dir",type=Path,default=Path(__file__).resolve().parent,
                        help="Directory containing CSV files (default: location of this script)")
    parser.add_argument("--output-dir",type=Path,default=None,
                        help="Figure output directory (default: <input-dir>/figures_v5)")
    parser.add_argument("--dpi",type=int,default=240,help="PNG resolution (default: 240)")
    parser.add_argument("--formats",nargs="+",choices=["png","pdf"],default=["png","pdf"],
                        help="Figure formats to create (default: png pdf)")
    args=parser.parse_args()
    root=args.input_dir.expanduser().resolve()
    output=(args.output_dir.expanduser().resolve() if args.output_dir else root/"figures_v5")
    try:
        frames=read_inputs(root)
        plt.rcParams.update({"font.size":9.5,"axes.spines.top":False,"axes.spines.right":False,
                              "pdf.fonttype":42,"figure.facecolor":"white","savefig.facecolor":"white"})
        manifest=[]
        for fn in [figure_51,figure_52,figure_53,figure_54,figure_55,figure_56,figure_57]:
            fn(frames,output,args.dpi,manifest,args.formats)
            print(f"Created figures for {fn.__name__}")
        mf=output/"figure_manifest_v5.csv";mf.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(manifest).to_csv(mf,index=False)
        print(f"\nSUCCESS: {len(manifest)} figures in {output}")
        print(f"Manifest: {mf}")
        print("Original CSV files were not modified; ABM was not rerun.")
        return 0
    except (ValueError,FileNotFoundError,KeyError) as e:
        print(f"ERROR: {e}",file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
