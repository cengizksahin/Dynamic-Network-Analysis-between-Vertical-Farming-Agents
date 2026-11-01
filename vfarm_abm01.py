"""Vertical Farming ABM, thesis V3 (Sections 3.5, 4.1, 4.3, Appendix).

Usage: python vfarm_abm_V3.py
Keep parameters.csv alongside this file. Outputs go to ../output.
Requires Mesa >= 3, networkx, numpy and pandas.

One cycle = one 30-day harvest. Four treatments use paired Monte Carlo seeds.
Separate Rand/Szolnoki validation experiments are NOT part of this Phase-2 model.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

ROOT = Path(__file__).resolve().parent
REQUIRED = set('L T S_max p_C0 p_TypeA d l eta_min eta_max v unit_water_cost unit_elec_cost unit_nutr_cost lambda_L W_h p_rewire t_coop alpha K p_base b gamma w'.split())
TREATMENTS = ('T0_Static_NoInc', 'T1_Fluid_NoInc', 'T2_Static_Active', 'T3_Fluid_Active')


def load_parameters(path: str | Path = ROOT / 'parameters.csv') -> dict[str, float]:
    """Parse ordinary CSV or supplied whole-line-quoted CSV; preserve symbol case."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Could not find {path}; put parameters.csv beside the script.')
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if not lines:
        raise ValueError('Parameter CSV is empty.')
    if lines[0].strip().startswith('"symbol,'):
        lines = [line[1:-1].replace('""', '"') if line.startswith('"') and line.endswith('"') else line for line in lines]
    df = pd.read_csv(StringIO('\n'.join(lines)), sep=None, engine='python')
    df.columns = df.columns.astype(str).str.strip().str.lower()
    if not {'symbol', 'value'} <= set(df.columns):
        raise ValueError(f'Expected symbol and value CSV columns; received {df.columns.tolist()}')
    symbols = df.symbol.astype(str).str.strip()
    if symbols.duplicated().any():
        raise ValueError(f'Duplicate symbols: {symbols[symbols.duplicated()].tolist()}')
    values = pd.to_numeric(df.value, errors='raise').astype(float)
    if not np.isfinite(values).all():
        raise ValueError('Parameter values must be finite.')
    p = dict(zip(symbols, values))
    if missing := REQUIRED - p.keys():
        raise ValueError(f'Missing parameters: {sorted(missing)}')
    for key in ('L', 'T', 'S_max'):
        if p[key] < 1 or not p[key].is_integer():
            raise ValueError(f'{key} must be a positive integer')
    if p['L'] < 3:
        raise ValueError('L must be at least 3 to provide four distinct torus neighbors.')
    for key in ('p_C0', 'p_TypeA', 'p_rewire'):
        if not 0 <= p[key] <= 1:
            raise ValueError(f'{key} must lie between 0 and 1')
    if not 0 < p['eta_min'] <= p['eta_max'] or p['K'] <= 0 or p['lambda_L'] < 1:
        raise ValueError('Require 0 < eta_min <= eta_max, K > 0, lambda_L >= 1.')
    if min(p[k] for k in ('d', 'l', 'v', 'alpha', 'W_h', 't_coop', 'p_base', 'b', 'gamma', 'w')) < 0:
        raise ValueError('Production, effort and tokenomics parameters must be nonnegative.')
    if p['W_h'] * p['t_coop'] <= 0:
        raise ValueError('Coordination cost k must be strictly positive.')
    return p


class VerticalFarmerAgent(Agent):
    """Single grower: strategy, agronomic efficiency, private wallet, public reputation."""

    def __init__(self, model, strategy: int, phenotype: str, efficiency: float):
        super().__init__(model)
        self.strategy = self.next_strategy = strategy
        self.phenotype = phenotype
        self.eta = efficiency
        self.H = model.d * model.layers * efficiency * model.v
        self.cost = model.harvest_cost
        self.B = self.H * model.alpha
        self.streak = 0
        self.mu = 1
        self.wallet = 0.0  # PRIVATE: never considered in Fermi decisions
        self.reputation = 0.0
        self.omega = self.H - self.cost
        self.payoff = self.H - self.cost
        self.n_cooperators = 0
        self.neighbours = []

    def point_reward(self, tier: int | None = None) -> float:
        m = self.model
        tier = self.mu if tier is None else tier
        return m.p_base + m.p_base ** (m.gamma * tier)

    def reputation_reward(self, tier: int | None = None) -> float:
        m = self.model
        tier = self.mu if tier is None else tier
        return m.p_base + m.b / (1 + m.gamma * tier)

    def digital_reward(self) -> float:
        return self.point_reward() + self.reputation_reward()

    def threshold(self) -> float:
        """Analytical q*: infinity at lambda=1 unless rewards cover effort.

        Strict best-response inequality EU(C) > EU(D); no tie cooperation.
        """
        m = self.model
        numerator = m.lambda_L * m.k - (m.w * self.digital_reward() if m.incentives else 0.0)
        denominator = (m.lambda_L - 1) * m.k
        return numerator / denominator if denominator > 0 else (-np.inf if numerator < 0 else np.inf)

    def choose_next(self) -> None:
        m = self.model
        if self.phenotype == 'Type_A':
            # Equivalent to n_C/z > q* for lambda>1; works for lambda=1 sweeps.
            q = self.n_cooperators / m.degree
            digital = m.w * self.digital_reward() if m.incentives else 0.0
            self.next_strategy = int(q * (m.lambda_L - 1) * m.k > m.lambda_L * m.k - digital)
        else:
            neighbor = self.random.choice(self.neighbours)
            delta = (self.omega - neighbor.omega) / m.K
            adoption_probability = 1 / (1 + np.exp(np.clip(delta, -700, 700)))
            self.next_strategy = neighbor.strategy if self.random.random() < adoption_probability else self.strategy


def cooperation_rate(m) -> float:
    return sum(a.strategy for a in m.agents) / m.N


def neighbour_ratio(m) -> float:
    sample = [a.n_cooperators / m.degree for a in m.agents if a.strategy]
    return float(np.mean(sample)) if sample else 0.0


def average(m, field: str) -> float:
    return float(np.mean([getattr(a, field) for a in m.agents]))


class VerticalFarmingModel(Model):
    """Phase-2 2D torus, degree-conserving conflict-edge swaps, synchronous choice."""

    def __init__(self, params: dict[str, float], p_rewire: float | None = None,
                 w: float | None = None, seed: int = 1):
        super().__init__(seed=seed)
        self.params = dict(params)
        self.L = int(params['L'])
        self.N, self.degree = self.L ** 2, 4
        self.d, self.layers, self.v = (params[x] for x in ('d', 'l', 'v'))
        self.alpha, self.lambda_L, self.K = (params[x] for x in ('alpha', 'lambda_L', 'K'))
        self.p_base, self.b, self.gamma = (params[x] for x in ('p_base', 'b', 'gamma'))
        self.k = params['t_coop'] * params['W_h']
        self.harvest_cost = self.d * self.layers * sum(params[x] for x in
            ('unit_water_cost', 'unit_elec_cost', 'unit_nutr_cost'))
        self.p_rewire = params['p_rewire'] if p_rewire is None else p_rewire
        self.w = params['w'] if w is None else w
        self.incentives = self.w > 0  # w=0: no points/reputation in T0/T1
        self.total_synergy = 0.0
        self.platform_outlay = 0.0
        self.rounds = 0
        self.edge_swaps = 0

        graph = nx.grid_2d_graph(self.L, self.L, periodic=True)
        self.grid = NetworkGrid(graph)
        for node in graph:
            strategy = int(self.random.random() < params['p_C0'])
            phenotype = 'Type_A' if self.random.random() < params['p_TypeA'] else 'Type_B'
            eta = self.random.uniform(params['eta_min'], params['eta_max'])
            self.grid.place_agent(VerticalFarmerAgent(self, strategy, phenotype, eta), node)
        self.agents_by_node = {a.pos: a for a in self.agents}
        self._refresh_neighbours()
        self.datacollector = DataCollector(model_reporters={
            'rho_C': cooperation_rate, 'q_C': neighbour_ratio,
            'mean_mu': lambda m: average(m, 'mu'),
            'mean_omega': lambda m: average(m, 'omega'),
            'discordant_edges': lambda m: sum(
                m.agents_by_node[u].strategy != m.agents_by_node[v].strategy
                for u, v in m.grid.G.edges),
        })

    def _refresh_neighbours(self) -> None:
        for agent in self.agents:
            agent.neighbours = [self.agents_by_node[n] for n in self.grid.G.neighbors(agent.pos)]
        if any(len(a.neighbours) != self.degree for a in self.agents):
            raise RuntimeError('Network degree invariance violated')

    def _rewire(self) -> None:
        if self.p_rewire == 0:
            return
        graph = self.grid.G
        # Orient every discordant edge from cooperator to defector.
        conflicts = [(u, v) if self.agents_by_node[u].strategy else (v, u)
                     for u, v in graph.edges
                     if self.agents_by_node[u].strategy != self.agents_by_node[v].strategy
                     and self.random.random() < self.p_rewire]
        self.random.shuffle(conflicts)
        altered = False
        while len(conflicts) >= 2:
            (c1, d1), (c2, d2) = conflicts.pop(), conflicts.pop()
            if not (graph.has_edge(c1, d1) and graph.has_edge(c2, d2)):
                continue
            # As stipulated in ODD 3.5.7: randomly cross-pair endpoints.
            # 50% possibility of C-C and D-D; otherwise exchange conflict ties.
            new_edges = ((c1, c2), (d1, d2)) if self.random.random() < 0.5 else ((c1, d2), (d1, c2))
            if any(u == v or graph.has_edge(u, v) for u, v in new_edges):
                continue
            graph.remove_edges_from(((c1, d1), (c2, d2)))
            graph.add_edges_from(new_edges)
            altered = True
            self.edge_swaps += 1
        if altered:
            self._refresh_neighbours()

    def step(self) -> None:
        """Game -> platform state -> synchronous decisions -> rewiring -> commit.

        Reports are collected after commit, describing the actual final network state.
        """
        for a in self.agents:
            a.n_cooperators = sum(n.strategy for n in a.neighbours)
        for a in self.agents:
            synergy = a.n_cooperators * a.B
            digital = self.w * a.digital_reward() if a.strategy and self.incentives else 0.0
            a.payoff = a.H - a.cost + synergy - (self.k if a.strategy else 0.0) + digital
            self.total_synergy += synergy
            self.platform_outlay += digital
        for a in self.agents:
            if a.strategy:
                if self.incentives:
                    a.wallet += a.point_reward()
                    a.reputation += a.reputation_reward()
                a.streak += 1
                a.mu = min(5, 1 + a.streak // 10)
            else:
                a.streak, a.mu = 0, 1
            a.omega = a.H - a.cost + (self.w * a.reputation if self.incentives else 0.0)
        for a in self.agents:
            a.choose_next()
        self._rewire()
        for a in self.agents:
            a.strategy = a.next_strategy
        for a in self.agents:
            a.n_cooperators = sum(n.strategy for n in a.neighbours)
        self.rounds += 1
        self.datacollector.collect(self)

    def compute_ex_post_metrics(self) -> dict[str, float]:
        return {
            'ROI_platform': (self.total_synergy / self.platform_outlay
                             if self.platform_outlay > 0 else np.nan),
            'mean_p_final': average(self, 'wallet'),
            'mean_wallet_eur': self.w * average(self, 'wallet'),
            'total_synergy': self.total_synergy,
            'platform_outlay': self.platform_outlay,
            'edge_swaps': self.edge_swaps,
        }

    # Backward-compatible alias, useful for analysis scripts.
    ex_post_metrics = compute_ex_post_metrics


def run_experiments(params: dict[str, float], seeds: int | None = None,
                    rounds: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four conditions with matching initialization seeds (not forced equal RNG draws)."""
    seeds = int(params['S_max']) if seeds is None else seeds
    rounds = int(params['T']) if rounds is None else rounds
    conditions = dict(zip(TREATMENTS, ((0, 0), (params['p_rewire'], 0),
                                       (0, params['w']), (params['p_rewire'], params['w']))))
    histories, summaries = [], []
    for label, (rewiring, exchange) in conditions.items():
        print(f'{label}: {seeds} seeds x {rounds} rounds', flush=True)
        for seed in range(1, seeds + 1):
            model = VerticalFarmingModel(params, p_rewire=rewiring, w=exchange, seed=seed)
            for _ in range(rounds):
                model.step()
            history = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
            history.insert(0, 'round', np.arange(1, rounds + 1))
            history.insert(0, 'seed', seed)
            history.insert(0, 'treatment', label)
            histories.append(history)
            final = history.iloc[-1]
            summaries.append({
                'treatment': label, 'seed': seed,
                'final_rho_C': final.rho_C, 'final_q_C': final.q_C,
                'final_mu': final.mean_mu, 'final_omega': final.mean_omega,
                **model.compute_ex_post_metrics(),
            })
    return pd.concat(histories, ignore_index=True), pd.DataFrame(summaries)


def main() -> None:
    parameters = load_parameters()
    output = ROOT.parent / 'output'
    output.mkdir(parents=True, exist_ok=True)
    trajectories, results = run_experiments(parameters)
    trajectories.to_csv(output / 'factorial_timeseries.csv', index=False)
    results.to_csv(output / 'factorial_ex_post_summary.csv', index=False)
    pd.DataFrame(parameters.items(), columns=['symbol', 'value']).to_csv(
        output / 'parameters_used.csv', index=False)
    print('\nMean final outcomes by treatment:')
    print(results.groupby('treatment', sort=False)[
        ['final_rho_C', 'final_q_C', 'final_mu', 'ROI_platform']].mean().to_string())
    print(f'\nOutput files saved in {output}')


if __name__ == '__main__':
    main()