"""V5 experiment runner: original 5x5 composition sweep AND all historical sensitivity designs.

Place alongside vfarm_abm_v5.py and parameters.csv; run `python run_sweeps_v5.py`.
The model is NOT modified by this file. It runs the four T0–T3 treatments at
matched seed IDs for every composition and parameter cell.

Results are saved separately, never confused with V3 or baseline factorial:
- population_sweep_v5.csv
- population_sweep_summary_v5.csv
- sensitivity_sweeps_v5.csv
- sensitivity_sweeps_summary_v5.csv
- sensitivity_factorial_grid_v5.csv
- sensitivity_factorial_grid_summary_v5.csv

The factorial files are generated independently by `vfarm_abm_v5.py`.
"""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from vfarm_abm_v5 import VerticalFarmingModel, load_parameters

HERE = Path(__file__).resolve().parent

# Exact values from the original user-supplied run_sweeps.py, preserved in V5.
ORIGINAL_SENSITIVITY_GRIDS = {
    'p_rewire': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    'lambda_L': [1.00, 1.50, 2.00, 2.25, 2.50, 3.00, 3.50],
    'alpha':    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    'w':        [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
    'p_base':   [2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00],
    'gamma':    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
}
# Earlier V2 rewiring x incentive experiment, verified against the supplied
# sensitivity_sweeps.csv (6 x 7 x 4 treatments x 30 seeds = 5,040 rows).
INTERACTION_REWIRE_GRID = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]
INTERACTION_INCENTIVE_GRID = [0.00, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
C_GRID = [0.10, 0.25, 0.50, 0.75, 0.90]
A_GRID = [0.10, 0.25, 0.50, 0.75, 0.90]

# Four original primary outcomes PLUS the four observational families.
# q* is the Type A theoretical decision threshold, not observed q_C.
TIMESERIES_METRICS = {
    'rho_C': 'final_rho_C',
    'q_C': 'final_q_C',
    'mean_mu': 'final_mu',
    'mean_omega': 'final_omega',
    'type_A_mean_q_star': 'final_type_A_mean_q_star',
    'type_A_median_q_star': 'final_type_A_median_q_star',
    'type_A_q_star_n': 'final_type_A_q_star_n',
    'mean_mu_applied': 'final_mean_mu_applied',
    'mean_point_reward_applied': 'final_mean_point_reward_applied',
    'mean_reputation_reward_applied': 'final_mean_reputation_reward_applied',
    'mean_total_marginal_incentive_cooperators': 'final_mean_total_marginal_incentive_cooperators',
    'mean_public_reputation_R_c': 'final_mean_public_reputation_R_c',
    'mean_observable_wealth_predecision': 'final_mean_observable_wealth_predecision',
}
REQUIRED = ('rho_C', 'q_C', 'mean_mu', 'mean_omega', 'type_A_mean_q_star',
            'mean_mu_applied', 'mean_point_reward_applied', 'mean_reputation_reward_applied',
            'mean_public_reputation_R_c')


def treatments(params):
    return {
        'T0_Static_NoInc': (0.0, 0.0),
        'T1_Fluid_NoInc': (float(params['p_rewire']), 0.0),
        'T2_Static_Active': (0.0, float(params['w'])),
        'T3_Fluid_Active': (float(params['p_rewire']), float(params['w'])),
    }


def simulate(params, *, treatment, p_rewire, w, seed, rounds, p_C0=None, p_TypeA=None):
    model = VerticalFarmingModel(params, p_rewire=p_rewire, w=w, seed=seed,
                                 p_C0=p_C0, p_TypeA=p_TypeA)
    for _ in range(rounds):
        model.step()
    time_series = model.datacollector.get_model_vars_dataframe()
    if time_series.empty:
        raise RuntimeError('No time series collected by V5 ABM.')
    missing = [c for c in REQUIRED if c not in time_series.columns]
    if missing:
        raise RuntimeError(f'vfarm_abm_v5.py does not expose required V5 metrics: {missing}')
    last = time_series.iloc[-1]
    result = {
        'treatment': treatment,
        'seed': seed,
        'rounds': rounds,
        'p_rewire_used': float(p_rewire),
        'w_used': float(w),
        'p_C0': float(params['p_C0'] if p_C0 is None else p_C0),
        'p_TypeA': float(params['p_TypeA'] if p_TypeA is None else p_TypeA),
    }
    for src, dest in TIMESERIES_METRICS.items():
        result[dest] = float(last[src]) if src in last else np.nan
    ex_post = model.compute_ex_post_metrics()
    result.update(ex_post)

    # Ostrom: indicators, not a claim that explicit sanction/monitoring rules exist.
    # Each metric is calculated from existing observations; no behavioral changes.
    diag = pd.DataFrame(model.diagnostics_history)
    monitoring = pd.DataFrame(model.monitoring_history)
    for key in ('C_to_D_transitions', 'D_to_C_transitions', 'strategy_transition_count',
                'successful_edge_swaps', 'rewire_pair_attempts', 'candidate_conflict_edges',
                'selected_conflict_edges', 'existing_streak_reset_events'):
        source = diag if key in diag else monitoring
        result['total_' + key] = float(source[key].sum()) if key in source else np.nan
    for key in ('type_A_threshold_met_rate', 'connected_after', 'degree_sequence_preserved',
                'edge_count_preserved', 'self_loop_count', 'duplicate_edge_count'):
        result['mean_' + key] = float(diag[key].mean()) if key in diag else np.nan
    for key in ('mean_observable_wealth_omega', 'mean_public_reputation_R_c',
                'strategy_observations_available'):
        result['monitoring_mean_' + key] = float(monitoring[key].mean()) if key in monitoring else np.nan
    result['monitoring_mechanism_implemented'] = 0
    result['graduated_sanction_mechanism_implemented'] = 0
    return result


def aggregate(df, keys):
    # Seed is the independent Monte Carlo unit. Never treat rounds as independent.
    excluded = set(keys) | {'seed', 'rounds', 'p_rewire_used', 'w_used',
                            'initial_defector_fraction', 'initial_type_B_fraction',
                            'baseline_value', 'parameter_value'}
    numeric = [c for c in df.select_dtypes(include='number').columns if c not in excluded]
    result = df.groupby(keys, dropna=False)[numeric].agg(['mean', 'std', 'count'])
    result.columns = [f'{k}_{stat}' for k, stat in result.columns]
    for col in numeric:
        n = result[f'{col}_count']
        sem = result[f'{col}_std'] / np.sqrt(n)
        result[f'{col}_sem'] = sem
        result[f'{col}_ci95_low'] = result[f'{col}_mean'] - 1.96 * sem
        result[f'{col}_ci95_high'] = result[f'{col}_mean'] + 1.96 * sem
    return result.reset_index()


def population(params, seeds, rounds, out):
    rows = []
    specs = list(itertools.product(C_GRID, A_GRID, treatments(params).items(), range(1, seeds+1)))
    expected = len(specs)
    print(f'Population composition: {expected:,} planned runs (5x5x4x{seeds}).', flush=True)
    for i, (pc, pa, (name, (pr, w)), seed) in enumerate(specs, 1):
        row = simulate(params, treatment=name, p_rewire=pr, w=w, seed=seed,
                       rounds=rounds, p_C0=pc, p_TypeA=pa)
        row['initial_defector_fraction'] = 1.0 - pc
        row['initial_type_B_fraction'] = 1.0 - pa
        rows.append(row)
        if i % max(1, expected // 20) == 0 or i == expected:
            print(f'  population: {i:,}/{expected:,}', flush=True)
    raw = pd.DataFrame(rows)
    assert len(raw) == expected
    assert not raw.duplicated(['p_C0', 'p_TypeA', 'treatment', 'seed']).any()
    raw.to_csv(out/'population_sweep_v5.csv', index=False)
    aggregate(raw, ['p_C0', 'p_TypeA', 'treatment']).to_csv(out/'population_sweep_summary_v5.csv', index=False)


def sensitivity(params, seeds, rounds, out):
    rows = []
    total = sum(len(x) for x in ORIGINAL_SENSITIVITY_GRIDS.values()) * 4 * seeds
    print(f'Six original univariate sensitivity sweeps: {total:,} planned runs.', flush=True)
    i = 0
    for parameter, grid in ORIGINAL_SENSITIVITY_GRIDS.items():
        for value in grid:
            altered = copy.deepcopy(params)
            altered[parameter] = float(value)
            # Cross all 4 treatments using the altered parameter value.
            # For p_rewire, static remains p=0; for w, inactive remains w=0.
            for name, (pr, w) in treatments(altered).items():
                for seed in range(1, seeds+1):
                    row = simulate(altered, treatment=name, p_rewire=pr, w=w,
                                   seed=seed, rounds=rounds)
                    row.update({'swept_parameter': parameter, 'parameter_value': float(value),
                                'baseline_value': float(params[parameter])})
                    rows.append(row)
                    i += 1
                    if i % max(1,total//25) == 0 or i == total:
                        print(f'  sensitivity: {i:,}/{total:,} ({parameter}={value:g})', flush=True)
    raw = pd.DataFrame(rows)
    assert len(raw) == total
    assert not raw.duplicated(['swept_parameter', 'parameter_value', 'treatment', 'seed']).any()
    raw.to_csv(out/'sensitivity_sweeps_v5.csv', index=False)
    aggregate(raw, ['swept_parameter', 'parameter_value', 'treatment']).to_csv(
        out/'sensitivity_sweeps_summary_v5.csv', index=False)


def factorial_grid(params, seeds, rounds, out):
    """Restore earlier 6x7 (rewiring x incentive) cross-treatment experiment."""
    rows = []
    total = len(INTERACTION_REWIRE_GRID) * len(INTERACTION_INCENTIVE_GRID) * 4 * seeds
    print(f'Rewiring x incentive factorial grid: {total:,} planned runs.', flush=True)
    i = 0
    for p, w in itertools.product(INTERACTION_REWIRE_GRID, INTERACTION_INCENTIVE_GRID):
        for name, (pr, inc) in {
            'T0_Static_NoInc': (0.0, 0.0),
            'T1_Fluid_NoInc': (p, 0.0),
            'T2_Static_Active': (0.0, w),
            'T3_Fluid_Active': (p, w),
        }.items():
            for seed in range(1, seeds+1):
                result = simulate(params, treatment=name, p_rewire=pr, w=inc,
                                  seed=seed, rounds=rounds)
                result.update({'p_rewire_sweep': p, 'w_sweep': w})
                rows.append(result)
                i += 1
                if i % max(1, total//20) == 0 or i == total:
                    print(f'  interaction grid: {i:,}/{total:,}', flush=True)
    raw = pd.DataFrame(rows)
    assert len(raw) == total
    assert not raw.duplicated(['p_rewire_sweep', 'w_sweep', 'treatment', 'seed']).any()
    raw.to_csv(out/'sensitivity_factorial_grid_v5.csv', index=False)
    aggregate(raw, ['p_rewire_sweep', 'w_sweep', 'treatment']).to_csv(
        out/'sensitivity_factorial_grid_summary_v5.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', default=str(HERE/'parameters.csv'))
    parser.add_argument('--output-dir', default=str(HERE))
    parser.add_argument('--seeds', type=int, default=None)
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--only', choices=['all', 'population', 'sensitivity', 'factorial-grid'], default='all')
    args = parser.parse_args()
    params = load_parameters(args.parameters)
    seeds = int(params['S_max']) if args.seeds is None else args.seeds
    rounds = int(params['T']) if args.rounds is None else args.rounds
    if seeds < 1 or rounds < 1:
        raise ValueError('seeds and rounds must be >= 1')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f'V5 parameters: {Path(args.parameters).resolve()} | {seeds} seeds | {rounds} rounds', flush=True)
    if args.only in ('all', 'population'):
        population(params, seeds, rounds, output)
    if args.only in ('all', 'sensitivity'):
        sensitivity(params, seeds, rounds, output)
    if args.only in ('all', 'factorial-grid'):
        factorial_grid(params, seeds, rounds, output)
    print('V5 sweep outputs complete.', flush=True)


if __name__ == '__main__':
    main()
