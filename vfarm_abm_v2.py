import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

# ==========================================
# 1. EXTERNAL PARAMETER LOADER
# ==========================================
def load_parameters(csv_path=None):
    """Load parameter table robustly, including legacy files with each row quoted as one field."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.csv")
    df = pd.read_csv(csv_path)
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        # Repair legacy CSVs where each entire line was quoted.
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        rows = [row[0].split(",", 5) if len(row) == 1 else row for row in rows]
        df = pd.DataFrame(rows[1:], columns=rows[0])
    if not {"symbol", "value"}.issubset(df.columns):
        raise ValueError(f"Parameter file must contain 'symbol' and 'value' columns: {csv_path}")
    return {str(row["symbol"]).strip(): float(row["value"]) for _, row in df.iterrows()}


# ==========================================
# 2. THE AGENT CLASS (Mesa 3.0+ Compatible)
# ==========================================
class VerticalFarmerAgent(Agent):
    def __init__(self, model, initial_strategy, phenotype, efficiency):
        super().__init__(model)
        self.strategy = initial_strategy       # s_i: 1 = Cooperate, 0 = Defect
        self.next_strategy = initial_strategy  # Double-buffer for Stage 4 -> Stage 5
        self.phenotype = phenotype             # "Type_A" (Prospect) or "Type_B" (Fermi)
        self.eta = efficiency                  # System efficiency ratio in [0.50, 1.00]

        # Static Offline Agronomic Attributes (Table 2 & Section 4.2.2)
        self.H_i = self.model.d * self.model.l * self.eta * self.model.v
        self.c_i = self.model.c_i_const
        self.B_i = self.H_i * self.model.alpha

        # Dynamic Platform State Variables (Table 1 & Section 3.5.2)
        self.streak = 0
        self.mu = 1                            # Streak tier multiplier in {1, 2, 3, 4, 5}
        self.p_current = 0.0                   # Hidden private point wallet
        self.R_c = 0.0                         # Public reputation score on Leadership Table
        self.omega = self.H_i - self.c_i       # Observable Network Wealth (Omega_i,t)
        self.payoff = 0.0                      # Single-round marginal game payoff (Pi_i,t)
        self.N_C = 0                           # Active cooperating neighbor count (N_C,i,t)
        self.my_neighbor_agents = []           # Cached topological neighbor pointers

    def marginal_point_reward(self, mu_val):
        """Private wallet increment: p_base + p_base^(gamma * mu_i)."""
        return self.model.p_base + (self.model.p_base ** (self.model.gamma * mu_val))

    def marginal_reputation_reward(self, mu_val):
        """Public reputation increment: p_base + b / (1 + gamma * mu_i)."""
        return self.model.p_base + (self.model.b / (1.0 + self.model.gamma * mu_val))

    def total_marginal_incentive(self, mu_val):
        """Total step-function digital incentive M_i(mu_i) (Table 3)."""
        return self.marginal_point_reward(mu_val) + self.marginal_reputation_reward(mu_val)

    def get_tipping_threshold_q_star(self):
        """Analytical tipping threshold q* for Type A Prospect Calculators (Table 4)."""
        w_M = self.model.w * self.total_marginal_incentive(self.mu)
        numerator = (self.model.lambda_L * self.model.k) - w_M
        denominator = (self.model.lambda_L - 1.0) * self.model.k
        return numerator / denominator

    def stage_1_agronomic(self):
        """Stage 1: Count active cooperating neighbors across z=4 topological ties."""
        self.N_C = sum(nb.strategy for nb in self.my_neighbor_agents)

    def stage_2_donation_game(self):
        """Stage 2: Simultaneous 2x2 Network Donation Game payoff aggregation (Table 5)."""
        synergy_gain = self.N_C * self.B_i
        if self.strategy == 1:
            M_i = self.total_marginal_incentive(self.mu)
            self.payoff = synergy_gain - self.model.k + (self.model.w * M_i)
            self.model.cumulative_synergy += synergy_gain
            self.model.cumulative_platform_outlay += (self.model.w * M_i)
        else:
            self.payoff = synergy_gain

    def stage_3_platform_update(self):
        """Stage 3: Update points, reputation, streaks, and Leadership Table Omega_i,t."""
        if self.strategy == 1:
            self.p_current += self.marginal_point_reward(self.mu)
            self.R_c += self.marginal_reputation_reward(self.mu)
            self.streak += 1
            self.mu = min(5, 1 + (self.streak // 10))
        else:
            # First-Order Graduated Sanction: Algorithmic Streak-Slashing
            self.streak = 0
            self.mu = 1
        self.omega = (self.H_i - self.c_i) + (self.model.w * self.R_c)

    def stage_4_revise_strategy(self):
        """Stage 4: Bifurcated behavioral strategy revision (stored in next_strategy)."""
        if self.phenotype == "Type_A":
            # Cumulative Prospect Theory Best Response: N_C,i,t >= 4 * q*
            q_star = self.get_tipping_threshold_q_star()
            self.next_strategy = 1 if self.N_C >= (self.model.z * q_star) else 0
        else:
            # Inward Fermi Heuristic Imitation on Observable Network Wealth (Omega)
            neighbor = self.random.choice(self.my_neighbor_agents)
            if self.strategy != neighbor.strategy:
                omega_diff = self.omega - neighbor.omega
                prob_imitate = 1.0 / (1.0 + np.exp(omega_diff / self.model.K))
                self.next_strategy = neighbor.strategy if self.random.random() < prob_imitate else self.strategy
            else:
                self.next_strategy = self.strategy


# ==========================================
# 3. DATA COLLECTION REPORTERS (Section 4.1)
# ==========================================
def compute_cooperator_fraction(model):
    return sum(a.strategy for a in model.agents) / model.num_nodes

def compute_cooperator_neighbor_ratio(model):
    coops = [a.N_C / model.z for a in model.agents if a.strategy == 1]
    return float(np.mean(coops)) if coops else 0.0

def compute_mean_tier_multiplier(model):
    return float(np.mean([a.mu for a in model.agents]))

def compute_mean_observable_wealth(model):
    return float(np.mean([a.omega for a in model.agents]))


# ==========================================
# 4. THE MODEL CLASS (Mesa 3.0+ Compatible)
# ==========================================
class VerticalFarmingModel(Model):
    def __init__(self, params, p_rewire=None, w=None, seed=42):
        super().__init__(seed=seed)
        self.L = int(params["L"])
        self.num_nodes = self.L * self.L
        self.z = 4

        # Unpack Agronomic, Behavioral, and Tokenomic Parameters
        self.d, self.l, self.v = params["d"], params["l"], params["v"]
        self.c_i_const = self.d * self.l * (
            params["unit_water_cost"] + params["unit_elec_cost"] + params["unit_nutr_cost"]
        )
        self.alpha = params["alpha"]
        self.k = params["t_coop"] * params["W_h"]
        self.lambda_L = params["lambda_L"]
        self.K = params["K"]
        self.p_base, self.b, self.gamma = params["p_base"], params["b"], params["gamma"]
        self.p_rewire = params["p_rewire"] if p_rewire is None else float(p_rewire)
        self.w = params["w"] if w is None else float(w)

        # Macro Accounting Accumulators for Ex-Post ROI_platform (Section 4.3)
        self.cumulative_synergy = 0.0
        self.cumulative_platform_outlay = 0.0

        # Generalized Graph: 10x10 Periodic Torus (N=100, z=4)
        G_nx = nx.grid_2d_graph(self.L, self.L, periodic=True)
        self.grid = NetworkGrid(G_nx)

        # Initialize and Place Agents (50% Cooperators, 50% Type A / Type B at t=0)
        for node in G_nx.nodes():
            init_strat = 1 if self.random.random() < params["p_C0"] else 0
            phenotype = "Type_A" if self.random.random() < params["p_TypeA"] else "Type_B"
            eff = self.random.uniform(params["eta_min"], params["eta_max"])
            agent = VerticalFarmerAgent(self, init_strat, phenotype, eff)
            self.grid.place_agent(agent, node)

        # O(1) Node-to-Agent Map and Initial Neighbor Cache
        self.node_to_agent = {a.pos: a for a in self.agents}
        self._refresh_neighbor_cache()

        # Register Mesa DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "rho_C": compute_cooperator_fraction,
                "q_C": compute_cooperator_neighbor_ratio,
                "mean_mu": compute_mean_tier_multiplier,
                "mean_omega": compute_mean_observable_wealth,
            }
        )

    def _refresh_neighbor_cache(self):
        for a in self.agents:
            a.my_neighbor_agents = [self.node_to_agent[n] for n in self.grid.G.neighbors(a.pos)]

    def stage_5_rewire_network(self):
        """Stage 5: Non-omniscient degree-preserving double-edge swap (Section 3.5.7)."""
        if self.p_rewire <= 0.0:
            return
        G = self.grid.G
        # 1. Identify all active conflict edges E_CD = {(u, v) in E | s_u != s_v}
        conflict_edges = [
            (u, v) if self.node_to_agent[u].strategy == 1 else (v, u)
            for u, v in G.edges()
            if self.node_to_agent[u].strategy != self.node_to_agent[v].strategy
        ]
        # 2. Filter candidate edges with probability p_rewire and shuffle into queue Q
        Q = [edge for edge in conflict_edges if self.random.random() < self.p_rewire]
        self.random.shuffle(Q)

        # 3. Pop pairs and execute random stub recombination preserving z=4
        swapped = False
        while len(Q) >= 2:
            u1, v1 = Q.pop()
            u2, v2 = Q.pop()
            if not (G.has_edge(u1, v1) and G.has_edge(u2, v2)):
                continue
            # Randomly orient endpoints of second edge with equal probability 0.50
            x2, y2 = (u2, v2) if self.random.random() < 0.50 else (v2, u2)
            # Parity & Liquidity Invariant: avoid self-loops and duplicate multi-edges
            if u1 != x2 and v1 != y2 and not G.has_edge(u1, x2) and not G.has_edge(v1, y2):
                G.remove_edge(u1, v1)
                G.remove_edge(u2, v2)
                G.add_edge(u1, x2)
                G.add_edge(v1, y2)
                swapped = True
        if swapped:
            self._refresh_neighbor_cache()

    def step(self):
        """Advances the simulation by one discrete 5-stage synchronous round t."""
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
        """Computes Ex-Post Wallet Monetization and ROI_platform (Section 4.3)."""
        roi = (
            self.cumulative_synergy / self.cumulative_platform_outlay
            if self.cumulative_platform_outlay > 0.0 else np.nan
        )
        mean_wallet = float(np.mean([a.p_current for a in self.agents]))
        return {"ROI_platform": roi, "mean_p_final": mean_wallet}


# ==========================================
# 5. 2x2 FACTORIAL EXECUTION & CSV EXPORT
# ==========================================
if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).resolve().parent
    params = load_parameters(here / "parameters.csv")
    output_dir = here / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    T_rounds = int(params["T"])
    S_max = int(params["S_max"])
    treatments = {
        "T0_Static_NoInc":  {"p_rewire": 0.00, "w": 0.00},
        "T1_Fluid_NoInc":   {"p_rewire": params["p_rewire"], "w": 0.00},
        "T2_Static_Active": {"p_rewire": 0.00, "w": params["w"]},
        "T3_Fluid_Active":  {"p_rewire": params["p_rewire"], "w": params["w"]},
    }
    ts_records, summary_records = [], []
    for t_name, cfg in treatments.items():
        print(f"Running {t_name} ({S_max} seeds x {T_rounds} rounds)...")
        for seed in range(1, S_max + 1):
            model = VerticalFarmingModel(params, p_rewire=cfg["p_rewire"], w=cfg["w"], seed=seed)
            for _ in range(T_rounds):
                model.step()
            df_run = model.datacollector.get_model_vars_dataframe().copy()
            df_run["round"] = np.arange(1, len(df_run) + 1)
            df_run["treatment"] = t_name
            df_run["seed"] = seed
            ts_records.append(df_run)
            ex_post = model.compute_ex_post_metrics()
            summary_records.append({
                "treatment": t_name, "seed": seed,
                "final_rho_C": df_run["rho_C"].iloc[-1],
                "final_q_C": df_run["q_C"].iloc[-1],
                "final_mu": df_run["mean_mu"].iloc[-1],
                "final_omega": df_run["mean_omega"].iloc[-1],
                **ex_post,
            })
    pd.concat(ts_records, ignore_index=True).to_csv(output_dir / "factorial_timeseries.csv", index=False)
    summary = pd.DataFrame(summary_records)
    summary.to_csv(output_dir / "factorial_ex_post_summary.csv", index=False)
    print("\nMean final outcomes by treatment:")
    print(summary.groupby("treatment")[["final_rho_C", "final_q_C", "final_mu", "final_omega", "ROI_platform"]].mean())
    print(f"\nOutput files saved in {output_dir}")
