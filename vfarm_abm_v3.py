"""
vfarm_abm_v3.py
Version 3 of the vertical-farming agent-based model.

Changes from v2:
- Exposes p_C0 (initial cooperator fraction) and p_TypeA (Type A fraction)
  as explicit experiment-level overrides.
- Initializes exact population counts for both dimensions, then shuffles labels:
  this makes ratio comparisons interpretable and avoids binomial composition noise.
- Reports initial composition and final composition metrics.
- Uses portable paths; no machine-specific paths are hard-coded.
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
    from pathlib import Path
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_path

    df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = {"symbol", "value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)}, "
            f"but found {df.columns.tolist()}. "
            f"Check the delimiter and header in {csv_path}"
        )

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="raise")

    return dict(zip(df["symbol"], df["value"]))




class VerticalFarmerAgent(Agent):
    def __init__(self, model, initial_strategy, phenotype, efficiency):
        super().__init__(model)
        self.strategy = int(initial_strategy)  # 1 = cooperator, 0 = defector
        self.next_strategy = self.strategy
        self.phenotype = phenotype             # Type_A or Type_B
        self.eta = efficiency

        self.H_i = self.model.d * self.model.l * self.eta * self.model.v
        self.c_i = self.model.c_i_const
        self.B_i = self.H_i * self.model.alpha

        self.streak = 0
        self.mu = 1
        self.p_current = 0.0
        self.R_c = 0.0
        self.omega = self.H_i - self.c_i
        self.payoff = 0.0
        self.N_C = 0
        self.my_neighbor_agents = []

    def marginal_point_reward(self, mu_val):
        return self.model.p_base + self.model.p_base ** (self.model.gamma * mu_val)

    def marginal_reputation_reward(self, mu_val):
        return self.model.p_base + self.model.b / (1.0 + self.model.gamma * mu_val)

    def total_marginal_incentive(self, mu_val):
        return self.marginal_point_reward(mu_val) + self.marginal_reputation_reward(mu_val)

    def get_tipping_threshold_q_star(self):
        w_M = self.model.w * self.total_marginal_incentive(self.mu)
        numerator = (self.model.lambda_L * self.model.k) - w_M
        denominator = (self.model.lambda_L - 1.0) * self.model.k
        # Keep original model equation, but avoid division-by-zero crashing a sweep.
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
        if self.strategy == 1:
            self.p_current += self.marginal_point_reward(self.mu)
            self.R_c += self.marginal_reputation_reward(self.mu)
            self.streak += 1
            self.mu = min(5, 1 + (self.streak // 10))
        else:
            self.streak = 0
            self.mu = 1
        self.omega = (self.H_i - self.c_i) + (self.model.w * self.R_c)

    def stage_4_revise_strategy(self):
        if self.phenotype == "Type_A":
            q_star = self.get_tipping_threshold_q_star()
            self.next_strategy = 1 if self.N_C >= (self.model.z * q_star) else 0
        else:
            if not self.my_neighbor_agents:
                self.next_strategy = self.strategy
                return
            neighbor = self.random.choice(self.my_neighbor_agents)
            if self.strategy != neighbor.strategy:
                # Clip exponent to avoid overflow while preserving logistic behavior.
                exponent = float(np.clip((self.omega - neighbor.omega) / self.model.K, -700, 700))
                prob_imitate = 1.0 / (1.0 + np.exp(exponent))
                self.next_strategy = neighbor.strategy if self.random.random() < prob_imitate else self.strategy
            else:
                self.next_strategy = self.strategy


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
    def __init__(self, params, p_rewire=None, w=None, seed=42,
                 p_C0=None, p_TypeA=None):
        # Mesa 3.x accepts seed; keep consistent with the v2 implementation.
        super().__init__(seed=seed)
        self.L = int(params["L"])
        self.num_nodes = self.L * self.L
        self.z = 4

        self.d, self.l, self.v = params["d"], params["l"], params["v"]
        self.c_i_const = self.d * self.l * (
            params["unit_water_cost"] + params["unit_elec_cost"] + params["unit_nutr_cost"]
        )
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
        for name, value in (("p_C0", self.p_C0), ("p_TypeA", self.p_TypeA),
                            ("p_rewire", self.p_rewire)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1; got {value}")

        self.cumulative_synergy = 0.0
        self.cumulative_platform_outlay = 0.0
        G_nx = nx.grid_2d_graph(self.L, self.L, periodic=True)
        self.grid = NetworkGrid(G_nx)

        # Exact composition counts, randomly assigned to nodes.
        n_coop = int(round(self.p_C0 * self.num_nodes))
        n_type_a = int(round(self.p_TypeA * self.num_nodes))
        strategies = [1] * n_coop + [0] * (self.num_nodes - n_coop)
        phenotypes = ["Type_A"] * n_type_a + ["Type_B"] * (self.num_nodes - n_type_a)
        self.random.shuffle(strategies)
        self.random.shuffle(phenotypes)

        for i, node in enumerate(G_nx.nodes()):
            eff = self.random.uniform(params["eta_min"], params["eta_max"])
            agent = VerticalFarmerAgent(self, strategies[i], phenotypes[i], eff)
            self.grid.place_agent(agent, node)

        self.node_to_agent = {a.pos: a for a in self.agents}
        self._refresh_neighbor_cache()
        self.datacollector = DataCollector(model_reporters={
            "rho_C": compute_cooperator_fraction,
            "q_C": compute_cooperator_neighbor_ratio,
            "mean_mu": compute_mean_tier_multiplier,
            "mean_omega": compute_mean_observable_wealth,
            "type_A_fraction": compute_type_a_fraction,
        })

    def _refresh_neighbor_cache(self):
        for a in self.agents:
            a.my_neighbor_agents = [self.node_to_agent[n] for n in self.grid.G.neighbors(a.pos)]

    def stage_5_rewire_network(self):
        if self.p_rewire <= 0.0:
            return
        G = self.grid.G
        conflict_edges = [
            (u, v) if self.node_to_agent[u].strategy == 1 else (v, u)
            for u, v in G.edges()
            if self.node_to_agent[u].strategy != self.node_to_agent[v].strategy
        ]
        queue = [edge for edge in conflict_edges if self.random.random() < self.p_rewire]
        self.random.shuffle(queue)
        swapped = False
        while len(queue) >= 2:
            u1, v1 = queue.pop()
            u2, v2 = queue.pop()
            if not (G.has_edge(u1, v1) and G.has_edge(u2, v2)):
                continue
            x2, y2 = (u2, v2) if self.random.random() < 0.5 else (v2, u2)
            if (u1 != x2 and v1 != y2 and
                not G.has_edge(u1, x2) and not G.has_edge(v1, y2)):
                G.remove_edge(u1, v1)
                G.remove_edge(u2, v2)
                G.add_edge(u1, x2)
                G.add_edge(v1, y2)
                swapped = True
        if swapped:
            self._refresh_neighbor_cache()

    def step(self):
        for a in self.agents:
            a.stage_1_agronomic()
        for a in self.agents:
            a.stage_2_donation_game()
        for a in self.agents:
            a.stage_3_platform_update()
        self.datacollector.collect(self)
        for a in self.agents:
            a.stage_4_revise_strategy()
        self.stage_5_rewire_network()
        for a in self.agents:
            a.strategy = a.next_strategy

    def compute_ex_post_metrics(self):
        roi = (self.cumulative_synergy / self.cumulative_platform_outlay
               if self.cumulative_platform_outlay > 0.0 else np.nan)
        return {
            "ROI_platform": roi,
            "mean_p_final": float(np.mean([a.p_current for a in self.agents])),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the vfarm ABM v3 baseline factorial.")
    parser.add_argument("--parameters", default=str(SCRIPT_DIR / "parameters.csv"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output_v3"))
    args = parser.parse_args()
    params = load_parameters(args.parameters)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds, seeds = int(params["T"]), int(params["S_max"])
    treatments = {
        "T0_Static_NoInc": {"p_rewire": 0.0, "w": 0.0},
        "T1_Fluid_NoInc": {"p_rewire": params["p_rewire"], "w": 0.0},
        "T2_Static_Active": {"p_rewire": 0.0, "w": params["w"]},
        "T3_Fluid_Active": {"p_rewire": params["p_rewire"], "w": params["w"]},
    }
    ts_records, summary_records = [], []
    for treatment, cfg in treatments.items():
        for seed in range(1, seeds + 1):
            model = VerticalFarmingModel(params, **cfg, seed=seed)
            for _ in range(rounds):
                model.step()
            df = model.datacollector.get_model_vars_dataframe()
            df["round"] = np.arange(1, len(df) + 1)
            df["treatment"], df["seed"] = treatment, seed
            ts_records.append(df)
            summary_records.append({
                "treatment": treatment, "seed": seed,
                "p_C0": model.p_C0, "p_TypeA": model.p_TypeA,
                "final_rho_C": df["rho_C"].iloc[-1],
                "final_q_C": df["q_C"].iloc[-1],
                "final_mu": df["mean_mu"].iloc[-1],
                "final_omega": df["mean_omega"].iloc[-1],
                **model.compute_ex_post_metrics(),
            })
    pd.concat(ts_records, ignore_index=True).to_csv(output_dir / "factorial_timeseries_v3.csv", index=False)
    pd.DataFrame(summary_records).to_csv(output_dir / "factorial_ex_post_summary_v3.csv", index=False)
    print(pd.DataFrame(summary_records).groupby("treatment")[[
        "final_rho_C", "final_q_C", "final_mu", "final_omega", "ROI_platform"
    ]].mean())
