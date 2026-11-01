"""V5 measurement/validation revision of the V3 vertical-farming ABM.

Behavioral rules are intentionally held constant except for rejecting invalid
rewiring swaps that could create self-loops/duplicate edges. The V3 model's
measurement timing is preserved for legacy metrics: they are collected after
streak/reward updates and before strategy revision and rewiring. Decision and
rewiring diagnostics are stored separately and keyed to the same round.
"""
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

SCRIPT_DIR = Path(__file__).resolve().parent


def load_parameters(csv_path="parameters.csv"):
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = SCRIPT_DIR / csv_path
    df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if not {"symbol", "value"}.issubset(df.columns):
        raise ValueError(f"Expected columns symbol,value in {csv_path}; got {df.columns.tolist()}")
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="raise")
    return dict(zip(df["symbol"], df["value"]))


class VerticalFarmerAgent(Agent):
    def __init__(self, model, initial_strategy, phenotype, efficiency):
        super().__init__(model)
        self.strategy = int(initial_strategy)  # 1 = cooperator; 0 = defector
        self.next_strategy = self.strategy
        self.phenotype = phenotype
        self.eta = efficiency
        self.H_i = self.model.d * self.model.l * self.eta * self.model.v
        self.c_i = self.model.c_i_const
        self.B_i = self.H_i * self.model.alpha
        self.streak = 0
        self.mu = 1
        self.p_current = 0.0  # private point wallet; not visible to neighbors
        self.R_c = 0.0        # public cumulative reputation
        self.omega = self.H_i - self.c_i
        self.payoff = 0.0
        self.N_C = 0
        self.my_neighbor_agents = []
        # Instrumentation only; none of these fields affect decisions.
        self.last_mu_applied = 1
        self.last_point_reward = 0.0
        self.last_reputation_reward = 0.0
        self.last_q_star = np.nan
        self.last_threshold_met = np.nan
        self.last_strategy_before_decision = self.strategy
        self.last_streak_reset = 0

    def marginal_point_reward(self, mu_val):
        return self.model.p_base + self.model.p_base ** (self.model.gamma * mu_val)

    def marginal_reputation_reward(self, mu_val):
        return self.model.p_base + self.model.b / (1.0 + self.model.gamma * mu_val)

    def total_marginal_incentive(self, mu_val):
        return self.marginal_point_reward(mu_val) + self.marginal_reputation_reward(mu_val)

    def get_tipping_threshold_q_star(self):
        # Exact V3 implemented equation; not the observed neighborhood ratio q_C.
        w_M = self.model.w * self.total_marginal_incentive(self.mu)
        numerator = (self.model.lambda_L * self.model.k) - w_M
        denominator = (self.model.lambda_L - 1.0) * self.model.k
        if abs(denominator) < 1e-12:
            return np.inf if numerator >= 0 else -np.inf
        return numerator / denominator

    def stage_1_agronomic(self):
        self.N_C = sum(nb.strategy for nb in self.my_neighbor_agents)

    def stage_2_donation_game(self):
        synergy_gain = self.N_C * self.B_i
        if self.strategy == 1:
            M_i = self.total_marginal_incentive(self.mu)
            self.payoff = synergy_gain - self.model.k + (self.model.w * M_i)
            self.model.cumulative_synergy += synergy_gain
            self.model.cumulative_platform_outlay += self.model.w * M_i
        else:
            self.payoff = synergy_gain

    def stage_3_platform_update(self):
        # Capture the multiplier actually used for this round's reward before
        # changing mu for the next round's streak tier.
        if self.strategy == 1:
            mu_applied = self.mu
            point_reward = self.marginal_point_reward(mu_applied)
            reputation_reward = self.marginal_reputation_reward(mu_applied)
            self.last_mu_applied = mu_applied
            self.last_point_reward = point_reward
            self.last_reputation_reward = reputation_reward
            self.last_streak_reset = 0
            self.p_current += point_reward
            self.R_c += reputation_reward
            self.streak += 1
            self.mu = min(5, 1 + (self.streak // 10))
        else:
            self.last_mu_applied = self.mu
            self.last_point_reward = 0.0
            self.last_reputation_reward = 0.0
            self.last_streak_reset = int(self.streak > 0 or self.mu != 1)
            self.streak = 0
            self.mu = 1
        self.omega = (self.H_i - self.c_i) + (self.model.w * self.R_c)

    def stage_4_revise_strategy(self):
        self.last_strategy_before_decision = self.strategy
        if self.phenotype == "Type_A":
            q_star = self.get_tipping_threshold_q_star()
            threshold_met = self.N_C >= (self.model.z * q_star)
            self.last_q_star = q_star
            self.last_threshold_met = int(threshold_met)
            self.next_strategy = 1 if threshold_met else 0
        else:
            self.last_q_star = np.nan
            self.last_threshold_met = np.nan
            if not self.my_neighbor_agents:
                self.next_strategy = self.strategy
                return
            neighbor = self.random.choice(self.my_neighbor_agents)
            if self.strategy != neighbor.strategy:
                exponent = float(np.clip((self.omega - neighbor.omega) / self.model.K, -700, 700))
                prob_imitate = 1.0 / (1.0 + np.exp(exponent))
                self.next_strategy = neighbor.strategy if self.random.random() < prob_imitate else self.strategy
            else:
                self.next_strategy = self.strategy


# Original V3 metrics, retaining their definitions.
def compute_cooperator_fraction(model):
    return sum(a.strategy for a in model.agents) / model.num_nodes

def compute_cooperator_neighbor_ratio(model):
    coops = [a.N_C / model.z for a in model.agents if a.strategy == 1]
    return float(np.mean(coops)) if coops else 0.0

def compute_mean_tier_multiplier(model):
    return float(np.mean([a.mu for a in model.agents]))

def compute_mean_observable_wealth(model):
    return float(np.mean([a.omega for a in model.agents]))

def compute_type_a_fraction(model):
    return sum(a.phenotype == "Type_A" for a in model.agents) / model.num_nodes


class VerticalFarmingModel(Model):
    def __init__(self, params, p_rewire=None, w=None, seed=42, p_C0=None, p_TypeA=None):
        super().__init__(seed=seed)
        self.L = int(params["L"])
        self.num_nodes = self.L * self.L
        self.z = 4
        self.d, self.l, self.v = params["d"], params["l"], params["v"]
        self.c_i_const = self.d * self.l * (params["unit_water_cost"] + params["unit_elec_cost"] + params["unit_nutr_cost"])
        self.alpha = params["alpha"]
        self.k = params["t_coop"] * params["W_h"]
        self.lambda_L = params["lambda_L"]
        self.K = params["K"]
        if self.K <= 0:
            raise ValueError("Parameter K must be positive for the Fermi imitation rule.")
        self.p_base, self.b, self.gamma = params["p_base"], params["b"], params["gamma"]
        self.p_rewire = float(params["p_rewire"] if p_rewire is None else p_rewire)
        self.w = float(params["w"] if w is None else w)
        self.p_C0 = float(params["p_C0"] if p_C0 is None else p_C0)
        self.p_TypeA = float(params["p_TypeA"] if p_TypeA is None else p_TypeA)
        for name, value in (("p_C0", self.p_C0), ("p_TypeA", self.p_TypeA), ("p_rewire", self.p_rewire)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]; got {value}")
        self.cumulative_synergy = 0.0
        self.cumulative_platform_outlay = 0.0
        G_nx = nx.grid_2d_graph(self.L, self.L, periodic=True)
        self.grid = NetworkGrid(G_nx)
        n_coop = int(round(self.p_C0 * self.num_nodes))
        n_type_a = int(round(self.p_TypeA * self.num_nodes))
        strategies = [1] * n_coop + [0] * (self.num_nodes - n_coop)
        phenotypes = ["Type_A"] * n_type_a + ["Type_B"] * (self.num_nodes - n_type_a)
        self.random.shuffle(strategies)
        self.random.shuffle(phenotypes)
        for i, node in enumerate(G_nx.nodes()):
            eff = self.random.uniform(params["eta_min"], params["eta_max"])
            self.grid.place_agent(VerticalFarmerAgent(self, strategies[i], phenotypes[i], eff), node)
        self.node_to_agent = {a.pos: a for a in self.agents}
        self._refresh_neighbor_cache()
        self.round_index = 0
        self.diagnostics_history = []
        self.monitoring_history = []
        self._round_edge_count_before = self.grid.G.number_of_edges()
        self.datacollector = DataCollector(model_reporters={
            # Legacy V3 fields: preserve names and collection timing.
            "rho_C": compute_cooperator_fraction,
            "q_C": compute_cooperator_neighbor_ratio,
            "mean_mu": compute_mean_tier_multiplier,
            "mean_omega": compute_mean_observable_wealth,
            "type_A_fraction": compute_type_a_fraction,
            # New measurement family 1: theoretical Type A q*.
            "type_A_mean_q_star": lambda m: _mean_finite([a.last_q_star for a in m.agents if a.phenotype == "Type_A"]),
            "type_A_median_q_star": lambda m: _median_finite([a.last_q_star for a in m.agents if a.phenotype == "Type_A"]),
            "type_A_q_star_n": lambda m: int(sum(a.phenotype == "Type_A" and np.isfinite(a.last_q_star) for a in m.agents)),
            # New measurement family 2: actual streak multiplier/reward amounts.
            "mean_mu_applied": lambda m: float(np.mean([a.last_mu_applied for a in m.agents])),
            "mean_point_reward_applied": lambda m: float(np.mean([a.last_point_reward for a in m.agents])),
            "mean_reputation_reward_applied": lambda m: float(np.mean([a.last_reputation_reward for a in m.agents])),
            "mean_total_marginal_incentive_cooperators": lambda m: _mean_finite([a.total_marginal_incentive(a.last_mu_applied) for a in m.agents if a.strategy == 1]),
            # New monitoring dashboard; these are separate descriptive metrics, not an index.
            "mean_public_reputation_R_c": lambda m: float(np.mean([a.R_c for a in m.agents])),
            "mean_observable_wealth_predecision": compute_mean_observable_wealth,
            "n_cooperators_predecision": lambda m: int(sum(a.strategy == 1 for a in m.agents)),
            "n_defectors_predecision": lambda m: int(sum(a.strategy == 0 for a in m.agents)),
        })

    def _refresh_neighbor_cache(self):
        for a in self.agents:
            a.my_neighbor_agents = [self.node_to_agent[n] for n in self.grid.G.neighbors(a.pos)]

    def stage_5_rewire_network(self):
        """V3 degree-preserving edge swaps, rejecting invalid/self-loop swaps."""
        G = self.grid.G
        before_edges = G.number_of_edges()
        before_degrees = dict(G.degree())
        before_components = nx.number_connected_components(G)
        candidate_conflicts = 0
        selected_edges = 0
        attempted_pairs = 0
        successful_swaps = 0
        if self.p_rewire > 0.0:
            conflict_edges = [
                (u, v) if self.node_to_agent[u].strategy == 1 else (v, u)
                for u, v in G.edges()
                if self.node_to_agent[u].strategy != self.node_to_agent[v].strategy
            ]
            candidate_conflicts = len(conflict_edges)
            queue = [edge for edge in conflict_edges if self.random.random() < self.p_rewire]
            selected_edges = len(queue)
            self.random.shuffle(queue)
            while len(queue) >= 2:
                u1, v1 = queue.pop()
                u2, v2 = queue.pop()
                attempted_pairs += 1
                if not (G.has_edge(u1, v1) and G.has_edge(u2, v2)):
                    continue
                # Shared-endpoint edge pairs are rejected to prevent loops and
                # ambiguous swaps. This preserves the intended edge-swap rule.
                if len({u1, v1, u2, v2}) < 4:
                    continue
                x2, y2 = (u2, v2) if self.random.random() < 0.5 else (v2, u2)
                new_edges = [(u1, x2), (v1, y2)]
                if any(a == b for a, b in new_edges):
                    continue
                if any(G.has_edge(a, b) for a, b in new_edges):
                    continue
                G.remove_edge(u1, v1)
                G.remove_edge(u2, v2)
                G.add_edge(*new_edges[0])
                G.add_edge(*new_edges[1])
                successful_swaps += 1
        if successful_swaps:
            self._refresh_neighbor_cache()
        after_degrees = dict(G.degree())
        after_edges = G.number_of_edges()
        self._last_rewire = {
            "candidate_conflict_edges": candidate_conflicts,
            "selected_conflict_edges": selected_edges,
            "rewire_pair_attempts": attempted_pairs,
            "successful_edge_swaps": successful_swaps,
            "edges_before": before_edges,
            "edges_after": after_edges,
            "edge_count_preserved": int(before_edges == after_edges),
            "degree_sequence_preserved": int(before_degrees == after_degrees),
            "self_loop_count": nx.number_of_selfloops(G),
            "duplicate_edge_count": int(G.number_of_edges() - len({frozenset(e) for e in G.edges()})),
            "components_before": before_components,
            "components_after": nx.number_connected_components(G),
            "connected_after": int(nx.is_connected(G)),
        }

    def step(self):
        self.round_index += 1
        strategies_before = {a.unique_id: int(a.strategy) for a in self.agents}
        for a in self.agents:
            a.stage_1_agronomic()
        for a in self.agents:
            a.stage_2_donation_game()
        for a in self.agents:
            a.stage_3_platform_update()
        # Compute threshold at the exact decision stage, after platform update.
        # This is instrumentation only and does not alter the decision rule.
        type_a = [a for a in self.agents if a.phenotype == "Type_A"]
        for a in type_a:
            a.last_q_star = a.get_tipping_threshold_q_star()
            a.last_threshold_met = int(a.N_C >= self.z * a.last_q_star)
        # Preserve V3 collection point: before strategy revision and rewiring.
        self.datacollector.collect(self)
        for a in self.agents:
            a.stage_4_revise_strategy()
        transitions = []
        for a in self.agents:
            old = strategies_before[a.unique_id]
            new = int(a.next_strategy)
            transitions.append((old, new))
        n_C_to_D = sum(old == 1 and new == 0 for old, new in transitions)
        n_D_to_C = sum(old == 0 and new == 1 for old, new in transitions)
        n_stay_C = sum(old == 1 and new == 1 for old, new in transitions)
        n_stay_D = sum(old == 0 and new == 0 for old, new in transitions)
        threshold_agents = [a for a in type_a if np.isfinite(a.last_q_star)]
        threshold_crossings = sum(a.last_threshold_met == 1 for a in threshold_agents)
        self.stage_5_rewire_network()
        self._last_rewire = getattr(self, "_last_rewire", {})
        for a in self.agents:
            a.strategy = a.next_strategy
        self.diagnostics_history.append({
            "round": self.round_index,
            "type_A_decision_count": len(type_a),
            "type_A_finite_threshold_count": len(threshold_agents),
            "type_A_threshold_met_count": int(threshold_crossings),
            "type_A_threshold_met_rate": threshold_crossings / len(threshold_agents) if threshold_agents else np.nan,
            "C_to_D_transitions": n_C_to_D,
            "D_to_C_transitions": n_D_to_C,
            "C_to_C_transitions": n_stay_C,
            "D_to_D_transitions": n_stay_D,
            "strategy_transition_count": n_C_to_D + n_D_to_C,
            "conflict_edges_after_strategy_decision_before_rewire": int(sum(
                self.node_to_agent[u].next_strategy != self.node_to_agent[v].next_strategy
                for u, v in self.grid.G.edges()
            )),
            **self._last_rewire,
        })
        # Ostrom monitoring dashboard: raw measures, deliberately no composite.
        self.monitoring_history.append({
            "round": self.round_index,
            "mean_public_reputation_R_c": float(np.mean([a.R_c for a in self.agents])),
            "mean_observable_wealth_omega": float(np.mean([a.omega for a in self.agents])),
            "observable_wealth_min": float(np.min([a.omega for a in self.agents])),
            "observable_wealth_max": float(np.max([a.omega for a in self.agents])),
            "public_reputation_range": float(max(a.R_c for a in self.agents) - min(a.R_c for a in self.agents)),
            "strategy_transition_count": n_C_to_D + n_D_to_C,
            "C_to_D_transitions": n_C_to_D,
            "D_to_C_transitions": n_D_to_C,
            "strategy_observations_available": int(sum(len(a.my_neighbor_agents) for a in self.agents)),
            "neighbor_strategy_cooperator_observations": int(sum(a.N_C for a in self.agents)),
            "monitoring_mechanism_implemented": 0,
            "graduated_sanction_mechanism_implemented": 0,
            "existing_streak_reset_events": int(sum(a.last_streak_reset for a in self.agents)),
        })

    def compute_ex_post_metrics(self):
        roi = (self.cumulative_synergy / self.cumulative_platform_outlay
               if self.cumulative_platform_outlay > 0.0 else np.nan)
        return {
            "ROI_platform": roi,
            "mean_p_final": float(np.mean([a.p_current for a in self.agents])),
            "mean_R_c_final": float(np.mean([a.R_c for a in self.agents])),
            "mean_streak_final": float(np.mean([a.streak for a in self.agents])),
            "mean_mu_applied_final_round": float(np.mean([a.last_mu_applied for a in self.agents])),
            "mean_point_reward_applied_final_round": float(np.mean([a.last_point_reward for a in self.agents])),
            "mean_reputation_reward_applied_final_round": float(np.mean([a.last_reputation_reward for a in self.agents])),
            "mean_type_A_q_star_final_decision": _mean_finite([a.last_q_star for a in self.agents if a.phenotype == "Type_A"]),
            "mean_existing_defection_reset_count_per_run": float(sum(a.streak == 0 for a in self.agents)),
            "graduated_sanction_mechanism_implemented": 0,
        }


def _mean_finite(values):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else np.nan

def _median_finite(values):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else np.nan


if __name__ == "__main__":
    # Preserve the V3 architecture: running the model file executes the
    # four-treatment factorial baseline. Population-composition sweeps are
    # handled separately by run_sweep_v5.py.
    import argparse
    parser = argparse.ArgumentParser(description="Run the V5 baseline factorial experiment.")
    parser.add_argument("--parameters", default=str(SCRIPT_DIR / "parameters.csv"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--seeds", type=int, default=None, help="Override S_max for smoke tests.")
    parser.add_argument("--rounds", type=int, default=None, help="Override T for smoke tests.")
    args = parser.parse_args()

    params = load_parameters(args.parameters)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = int(args.rounds if args.rounds is not None else params["T"])
    seeds = int(args.seeds if args.seeds is not None else params["S_max"])
    treatments = {
        "T0_Static_NoInc": {"p_rewire": 0.0, "w": 0.0},
        "T1_Fluid_NoInc": {"p_rewire": params["p_rewire"], "w": 0.0},
        "T2_Static_Active": {"p_rewire": 0.0, "w": params["w"]},
        "T3_Fluid_Active": {"p_rewire": params["p_rewire"], "w": params["w"]},
    }
    ts_records, summary_records = [], []
    total = len(treatments) * seeds
    run_number = 0
    for treatment, cfg in treatments.items():
        for seed in range(1, seeds + 1):
            run_number += 1
            model = VerticalFarmingModel(params, p_rewire=cfg["p_rewire"], w=cfg["w"], seed=seed)
            for _ in range(rounds):
                model.step()
            df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
            df["round"] = np.arange(1, len(df) + 1)
            df["treatment"], df["seed"] = treatment, seed
            df["p_rewire"], df["w"] = cfg["p_rewire"], cfg["w"]
            df["p_C0"], df["p_TypeA"] = model.p_C0, model.p_TypeA
            ts_records.append(df)
            final = df.iloc[-1]
            summary_records.append({
                "treatment": treatment, "seed": seed, "p_rewire": cfg["p_rewire"], "w": cfg["w"],
                "rounds": rounds, "p_C0": model.p_C0, "p_TypeA": model.p_TypeA,
                "final_rho_C": final["rho_C"], "final_q_C": final["q_C"],
                "final_mu": final["mean_mu"], "final_omega": final["mean_omega"],
                **{f"final_{c}": final[c] for c in df.columns if c.startswith(("type_A_", "mean_mu_applied", "mean_point_reward", "mean_reputation_reward", "mean_total_marginal", "mean_public_reputation", "mean_observable_wealth", "n_cooperators", "n_defectors"))},
                **model.compute_ex_post_metrics(),
            })
            if run_number % max(1, total // 20) == 0 or run_number == total:
                print(f"[factorial {run_number:,}/{total:,}] {treatment}, seed={seed}")
    ts_path = output_dir / "factorial_timeseries_v5.csv"
    summary_path = output_dir / "factorial_ex_post_summary_v5.csv"
    pd.concat(ts_records, ignore_index=True).to_csv(ts_path, index=False)
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
    print("\nSaved factorial outputs:")
    print(f"  {ts_path}")
    print(f"  {summary_path}")
