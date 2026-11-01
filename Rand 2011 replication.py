import itertools
import random
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

# ==========================================
# 1. THE AGENT CLASS (Empirical Human Psychology)
# ==========================================
class RandAgent(Agent):
    def __init__(self, model, initial_strategy):
        super().__init__(model)
        self.strategy = initial_strategy
        self.next_strategy = initial_strategy
        self.got_dumped = False  # Tracks ostracism per round

    def update_strategy(self):
        """Strategy update based strictly on Rand et al. (2011) empirical findings."""
        neighbor_nodes = list(self.model.grid.G.neighbors(self.pos))
        
        # Calculate fraction of cooperating neighbors (f_c)
        if len(neighbor_nodes) > 0:
            coop_count = sum(1 for n_id in neighbor_nodes if self.model.grid.get_cell_list_contents([n_id])[0].strategy == 1)
            f_c = coop_count / len(neighbor_nodes)
        else:
            f_c = 0.0  # Isolated agents have no cooperative reinforcement
            
        # Apply empirical behavioral rules derived from the text
        if self.strategy == 1:
            # COOPERATORS: "The greater the percentage of a subject's partners that defected... 
            # the less likely the subject is to cooperate" (p. 19195)
            # Max baseline cooperation (from Fig 3B) is 0.79. 
            prob_cooperate = f_c * 0.79 
        else:
            # DEFECTORS: "probability of switching to cooperation increases with the number 
            # of players that broke links with her" (p. 19195)
            if self.got_dumped:
                prob_cooperate = 0.49 # Fig 3B (At least one person breaks a link)
            else:
                prob_cooperate = 0.23 # Fig 3B (No one breaks a link)
                
        # Commit to next strategy based on calculated probability
        if self.random.random() < prob_cooperate:
            self.next_strategy = 1
        else:
            self.next_strategy = 0

    def advance_strategy(self):
        self.strategy = self.next_strategy


# ==========================================
# 2. DATA COLLECTION
# ==========================================
def compute_cooperation_rate(model):
    if len(model.agents) == 0: return 0
    cooperators = sum([1 for a in model.agents if a.strategy == 1])
    return cooperators / len(model.agents)


# ==========================================
# 3. THE MODEL CLASS
# ==========================================
class RandModel(Model):
    def __init__(self, k_fraction):
        super().__init__()
        
        # EXACT PARAMETERS FROM RAND ET AL. (2011)
        self.num_agents = 20          # "average network size = 19.6" 
        self.initial_link_prob = 0.20 # "initialized with 20% of possible links" 
        self.k_fraction = k_fraction  # 0.0 (Fixed), 0.10 (Viscous), 0.30 (Fluid) 
        
        # Empirical Rewiring Probabilities (From Figure 3A)
        self.p_make_if_C = 0.81
        self.p_make_if_D = 0.39
        self.p_break_if_D = 0.71
        self.p_break_if_C = 0.08
        
        # Initialize Graph
        G = nx.erdos_renyi_graph(n=self.num_agents, p=self.initial_link_prob)
        self.grid = NetworkGrid(G)
        
        # Initialize Agents
        for i in range(self.num_agents):
            # Starts at roughly 60% cooperation per Figure 1A visual baseline
            initial_strategy = 1 if self.random.random() < 0.60 else 0 
            a = RandAgent(self, initial_strategy)
            self.grid.place_agent(a, i)
            
        self.datacollector = DataCollector(
            model_reporters={"Cooperation_Rate": compute_cooperation_rate}
        )
        
    def rewire_network(self):
        """Executes rewiring and flags agents who suffer ostracism."""
        if self.k_fraction == 0:
            return # Static networks do not update
            
        # "a percentage k of subject pairs are picked at random"
        all_possible_pairs = list(itertools.combinations(range(self.num_agents), 2))
        num_pairs_to_sample = int(len(all_possible_pairs) * self.k_fraction)
        sampled_pairs = self.random.sample(all_possible_pairs, num_pairs_to_sample)
        
        for u, v in sampled_pairs:
            agent_u = self.grid.get_cell_list_contents([u])[0]
            agent_v = self.grid.get_cell_list_contents([v])[0]
            edge_exists = self.grid.G.has_edge(u, v)
            
            # "one of the two (picked at random) is offered the chance"
            decider, target = (agent_u, agent_v) if self.random.random() < 0.5 else (agent_v, agent_u)
            
            if edge_exists:
                prob_break = self.p_break_if_C if target.strategy == 1 else self.p_break_if_D
                if self.random.random() < prob_break:
                    self.grid.G.remove_edge(decider.pos, target.pos)
                    target.got_dumped = True  # Ostracism penalty triggered
            else:
                prob_make = self.p_make_if_C if target.strategy == 1 else self.p_make_if_D
                if self.random.random() < prob_make:
                    self.grid.G.add_edge(decider.pos, target.pos)

    def step(self):
        """Chronological order of a single experimental round."""
        self.datacollector.collect(self)
        
        # 1. Reset psychological state
        for agent in self.agents:
            agent.got_dumped = False
            
        # 2. Network Rewiring
        self.rewire_network()
        
        # 3. Strategy Update
        self.agents.shuffle_do("update_strategy")
        self.agents.do("advance_strategy")


# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # "40 realizations of our network experiments" 
    num_sessions = 40
    rounds = 11 # Plotting exactly 11 rounds as shown in Figure 1A
    
    # Store results for all 3 treatments
    static_results = np.zeros((num_sessions, rounds))
    viscous_results = np.zeros((num_sessions, rounds))
    fluid_results = np.zeros((num_sessions, rounds))
    
    print(f"Running {num_sessions} sessions of Fixed Network (n=0.0)...")
    for s in range(num_sessions):
        model = RandModel(k_fraction=0.0)
        for r in range(rounds):
            model.step()
        static_results[s] = model.datacollector.get_model_vars_dataframe()["Cooperation_Rate"].values

    print(f"Running {num_sessions} sessions of Viscous Network (n=0.10)...")
    for s in range(num_sessions):
        model = RandModel(k_fraction=0.10)
        for r in range(rounds):
            model.step()
        viscous_results[s] = model.datacollector.get_model_vars_dataframe()["Cooperation_Rate"].values

    print(f"Running {num_sessions} sessions of Fluid Network (n=0.30)...")
    for s in range(num_sessions):
        model = RandModel(k_fraction=0.30)
        for r in range(rounds):
            model.step()
        fluid_results[s] = model.datacollector.get_model_vars_dataframe()["Cooperation_Rate"].values

    # Average the 40 sessions
    static_avg = np.mean(static_results, axis=0)
    viscous_avg = np.mean(viscous_results, axis=0)
    fluid_avg = np.mean(fluid_results, axis=0)

    # Plotting to mirror Figure 1A
    plt.figure(figsize=(9, 6))
    plt.plot(range(1, rounds + 1), static_avg, label="Fixed (n=0)", color='green', marker='s', linewidth=2)
    plt.plot(range(1, rounds + 1), viscous_avg, label="Viscous Dynamic (n=0.10)", color='orange', marker='^', linewidth=2)
    plt.plot(range(1, rounds + 1), fluid_avg, label="Fluid Dynamic (n=0.30)", color='dodgerblue', marker='o', linewidth=2)
    
    plt.title("Rand et al. (2011) Empirical Replication (Average of 40 Sessions)")
    plt.xlabel("Round")
    plt.ylabel("Cooperative players")
    plt.ylim(0, 0.8)
    plt.xticks(range(1, rounds + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    print("Done! Generating plot...")
    plt.show()