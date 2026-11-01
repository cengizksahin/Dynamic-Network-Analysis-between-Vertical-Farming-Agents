import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

# ==========================================
# 1. THE AGENT CLASS (Mesa 3.0+ Compatible)
# ==========================================
class SpatialPGGAgent(Agent):
    def __init__(self, model, initial_strategy):
        super().__init__(model)
        self.strategy = initial_strategy  # 1 = Cooperator, 0 = Defector
        self.payoff = 0.0
        
        # Pre-cached performance arrays to avoid real-time graph lookups
        self.my_center_nodes = []
        self.my_neighbor_agents = []

    def calculate_total_payoff(self):
        """Highly optimized total payoff calculation using pre-cached groups."""
        total_coops = 0
        for center_node in self.my_center_nodes:
            for member in self.model.group_agents[center_node]:
                total_coops += member.strategy
                
        # Total PGG Equation: (r/G * Total Coops) - (G * personal cost)
        self.payoff = (self.model.r / self.model.G) * total_coops - (self.model.G * self.strategy)

    def step(self):
        """Elementary Asynchronous Step: Fermi Inward Imitation."""
        self.calculate_total_payoff()
        
        # Randomly select a neighbor and calculate their current payoff
        neighbor_agent = self.random.choice(self.my_neighbor_agents)
        neighbor_agent.calculate_total_payoff()
        
        # Fermi Imitation Probability: W(x <- y)
        if self.strategy != neighbor_agent.strategy:
            payoff_diff = self.payoff - neighbor_agent.payoff
            prob_imitate = 1.0 / (1.0 + np.exp(payoff_diff / self.model.K))
            
            if self.random.random() < prob_imitate:
                self.strategy = neighbor_agent.strategy


# ==========================================
# 2. DATA COLLECTION
# ==========================================
def compute_cooperation_density(model):
    """Calculates the ratio of cooperators currently on the grid."""
    cooperators = sum(1 for a in model.agents if a.strategy == 1)
    return cooperators / model.num_nodes


# ==========================================
# 3. THE MODEL CLASS
# ==========================================
class SpatialPGGModel(Model):
    def __init__(self, L, r_norm, K_norm):
        super().__init__()
        self.L = L
        self.num_nodes = L * L
        
        self.z = 4
        self.G = self.z + 1
        
        self.r = r_norm * self.G
        self.K = K_norm * self.G
        
        # Generalized Graph: Toroidal Square Lattice
        G_nx = nx.grid_2d_graph(L, L, periodic=True)
        self.grid = NetworkGrid(G_nx)
        
        # Initialize and Place Agents (50% Cooperators at t=0)
        for node in G_nx.nodes():
            initial_strategy = 1 if self.random.random() < 0.50 else 0
            a = SpatialPGGAgent(self, initial_strategy)
            self.grid.place_agent(a, node)
            
        # --- PERFORMANCE CACHING SYSTEM ---
        self.node_to_agent = {a.pos: a for a in self.agents}
        self.group_agents = {}
        for node in G_nx.nodes():
            members = [node] + list(self.grid.G.neighbors(node))
            self.group_agents[node] = [self.node_to_agent[m] for m in members]
            
        for a in self.agents:
            a.my_center_nodes = [a.pos] + list(self.grid.G.neighbors(a.pos))
            a.my_neighbor_agents = [self.node_to_agent[n] for n in self.grid.G.neighbors(a.pos)]

        # Register data collection
        self.datacollector = DataCollector(
            model_reporters={"Cooperation_Density": compute_cooperation_density}
        )

    def step(self):
        """Advances the model by one Monte Carlo Step (MCS)."""
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")


# ==========================================
# 4. SCALED EXECUTION & CSV EXPORT LOOP
# ==========================================
if __name__ == "__main__":
    # Ensure output directory exists (as requested)
    output_dir = r"C:\Users\cengo\OneDrive\Masaüstü\TUM\game theory\output"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "spatial_pgg_scaled_replication.csv")

    # Scaled Parameters based on the revised Methodological Guideline
    L_size = 50                 # Grid size (50 x 50 = 2,500 agents)
    K_norm = 0.1                # Normalized noise K/G
    thermalize_mcs = 2000       # Scaled steps to reach equilibrium
    sample_mcs = 500            # Scaled steps to average out steady-state
    total_mcs = thermalize_mcs + sample_mcs
    num_runs = 3                # Independent Monte Carlo seeds for stochastic smoothing
    
    # Sweep r/G from 0.70 to 1.10 with a step size of 0.05
    r_norm_values = np.arange(0.70, 1.15, 0.05)
    
    mean_densities = []
    std_densities = []
    csv_rows = []
    
    print(f"Executing Scaled Spatial PGG Validation (Saving to CSV)")
    print(f"Parameters: G=5, K/G={K_norm}, L={L_size} (N={L_size**2})")
    print(f"MCS: {thermalize_mcs} (Thermal) + {sample_mcs} (Sample)")
    print(f"Independent Runs per Data Point: {num_runs}")
    print(f"Destination: {csv_path}")
    print("-" * 65)
    
    for r_n in r_norm_values:
        run_results = []
        for run in range(num_runs):
            print(f"Simulating r/G = {r_n:.2f} | Run {run+1}/{num_runs}...", end="", flush=True)
            
            model = SpatialPGGModel(L=L_size, r_norm=r_n, K_norm=K_norm)
            
            # Execute MCS Loop
            for _ in range(total_mcs):
                model.step()
                
            # Extract data and average the steady state over the sample period
            data = model.datacollector.get_model_vars_dataframe()
            steady_state_avg = data["Cooperation_Density"].iloc[-sample_mcs:].mean()
            run_results.append(steady_state_avg)
            
            # Record individual run row for raw logging
            csv_rows.append({"r_norm": r_n, "run": run + 1, "cooperation_density": steady_state_avg})
            print(f" Density: {steady_state_avg:.3f}")
            
        # Calculate statistical mean and standard deviation across the 3 independent runs
        r_mean = np.mean(run_results)
        r_std = np.std(run_results)
        mean_densities.append(r_mean)
        std_densities.append(r_std)
        
        # Save progress to CSV after every completed r_norm step
        df_temp = pd.DataFrame(csv_rows)
        df_temp.to_csv(csv_path, index=False)
        
        print(f">>> STATS for r/G = {r_n:.2f} | Mean: {r_mean:.3f} | StdDev: {r_std:.4f} [Saved to CSV]\n")

    # Plotting Scaled Validation Figure with Error Bars
    plt.figure(figsize=(9, 6))
    plt.errorbar(r_norm_values, mean_densities, yerr=std_densities, marker='s', 
                 color='black', linestyle='-', linewidth=1.5, capsize=4, label='G=5 (Square Lattice, L=50)')
    
    plt.title(r"Scaled Baseline Validation: Topology-independent Spatial PGG", fontsize=12)
    plt.xlabel(r"Normalized Enhancement Factor ($r/G$)", fontsize=11)
    plt.ylabel(r"Cooperator Density ($\rho_C$)", fontsize=11)
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.xticks(np.arange(0.70, 1.15, 0.05))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()