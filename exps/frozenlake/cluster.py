import numpy as np
import gymnasium as gym
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.cm as cm
from matplotlib.colors import Normalize


class StochasticPolicyLearner:
    def __init__(
        self,
        env_name: str = "FrozenLake-v1",
        map_name: str = "4x4",
        is_slippery: bool = True,
        learning_rate: float = 0.8,
        discount_factor: float = 0.95,
        epsilon: float = 0.7,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        episodes: int = 2000,
        temperature: float = 1.0,
    ):
        """
        Initialize learner for stochastic policy on FrozenLake
        """
        self.env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)
        self.env_name = env_name
        self.map_name = map_name
        self.is_slippery = is_slippery

        # Q-learning parameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.temperature = temperature

        # Initialize Q-table
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # Tracking metrics
        self.rewards_history = []
        self.success_rate_history = []

        # Grid dimensions
        self.grid_size = 4
        self.state_coords = {
            s: (s // self.grid_size, s % self.grid_size) for s in range(self.n_states)
        }

        # Action names for better readability
        self.action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    def softmax_policy(self, state: int) -> np.ndarray:
        """
        Compute softmax policy probabilities for a state
        """
        q_values = self.q_table[state]
        # Subtract max for numerical stability
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / self.temperature)
        return exp_q / np.sum(exp_q)

    def choose_action_softmax(self, state: int) -> int:
        """
        Choose action according to softmax policy
        """
        probs = self.softmax_policy(state)
        return np.random.choice(self.n_actions, p=probs)

    def choose_action_epsilon_greedy(self, state: int) -> int:
        """
        Choose action using epsilon-greedy for training
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def train(self, verbose: bool = True) -> Dict:
        """
        Train using Q-learning with epsilon-greedy exploration
        """
        print(f"Training stochastic policy on {self.map_name} FrozenLake...")
        print(f"Slippery: {self.is_slippery}")
        print("-" * 50)

        success_window = []

        for episode in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action_epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Q-learning update
                best_next = np.max(self.q_table[next_state])
                self.q_table[state, action] += self.lr * (
                    reward + self.gamma * best_next - self.q_table[state, action]
                )

                state = next_state
                total_reward += reward

            # Track success
            success = 1 if total_reward > 0 else 0
            success_window.append(success)
            if len(success_window) > 100:
                success_window.pop(0)

            success_rate = np.mean(success_window)
            self.success_rate_history.append(float(success_rate))
            self.rewards_history.append(float(total_reward))

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (episode + 1) % 2000 == 0:
                print(
                    f"Episode {episode + 1}/{self.episodes} | "
                    f"Success Rate: {success_rate:.3f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        print("-" * 50)
        print(f"Training completed! Final success rate: {success_rate:.3f}")

        return {
            "q_table": self.q_table,
            "success_rate_history": self.success_rate_history,
            "final_success_rate": float(success_rate),
        }

    def get_stochastic_policy(self) -> Dict[int, np.ndarray]:
        """
        Get the learned stochastic policy (softmax probabilities)
        """
        policy = {}
        for state in range(self.n_states):
            policy[state] = self.softmax_policy(state)
        return policy

    def get_deterministic_policy(self) -> Dict[int, int]:
        """
        Get deterministic policy (max probability action)
        """
        policy = {}
        for state in range(self.n_states):
            probs = self.softmax_policy(state)
            policy[state] = int(np.argmax(probs))
        return policy

    def get_state_values(self) -> np.ndarray:
        """
        Get state values V(s) = max_a Q(s,a)
        """
        return np.max(self.q_table, axis=1)

    def cluster_states_by_action(self) -> Dict[int, List[int]]:
        """
        Cluster states by their most probable action
        Returns: {action: [list of states that prefer this action]}
        """
        deterministic_policy = self.get_deterministic_policy()
        clusters = {action: [] for action in range(self.n_actions)}

        for state, action in deterministic_policy.items():
            clusters[action].append(int(state))

        return clusters

    def collect_successful_trajectories(self, n_trajectories: int = 1000) -> List[Dict]:
        """
        Collect successful trajectories using the stochastic policy
        """
        trajectories = []
        attempts = 0
        max_attempts = n_trajectories * 10

        while len(trajectories) < n_trajectories and attempts < max_attempts:
            attempts += 1
            state, _ = self.env.reset()
            done = False
            trajectory = {
                "states": [],
                "actions": [],
                "action_probs": [],
                "rewards": [],
                "next_states": [],
                "success": False,
            }

            while not done:
                probs = self.softmax_policy(state)
                action = np.random.choice(self.n_actions, p=probs)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                trajectory["states"].append(int(state))
                trajectory["actions"].append(int(action))
                trajectory["action_probs"].append([float(p) for p in probs])
                trajectory["rewards"].append(float(reward))
                trajectory["next_states"].append(int(next_state))

                state = next_state

            trajectory["success"] = bool(trajectory["rewards"][-1] > 0)

            if trajectory["success"]:
                trajectories.append(trajectory)

        print(
            f"Collected {len(trajectories)} successful trajectories out of {attempts} attempts"
        )
        return trajectories

    def build_transition_graph(self, trajectories: List[Dict]) -> nx.DiGraph:
        """
        Build transition graph from successful trajectories
        """
        G = nx.DiGraph()

        # Add nodes for all states
        state_values = self.get_state_values()
        for state in range(self.n_states):
            G.add_node(
                int(state),
                pos=self.state_coords[state],
                action=int(self.get_deterministic_policy()[state]),
                value=float(state_values[state]),
            )

        # Count transitions
        transitions = defaultdict(lambda: defaultdict(int))

        for traj in trajectories:
            for i in range(len(traj["states"]) - 1):
                s = traj["states"][i]
                a = traj["actions"][i]
                s_next = traj["next_states"][i]
                transitions[(s, a)][s_next] += 1

        # Add edges with probabilities
        for (s, a), next_states in transitions.items():
            total = sum(next_states.values())
            for s_next, count in next_states.items():
                prob = count / total
                G.add_edge(
                    int(s),
                    int(s_next),
                    action=int(a),
                    probability=float(prob),
                    weight=float(prob),
                )

        return G

    def create_clustered_graph(
        self, trajectories: List[Dict]
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Create a directed graph where nodes are clusters of states that choose the same action
        with transition probabilities on edges and node colors based on sum of state values
        """
        # Get action clusters
        clusters = self.cluster_states_by_action()
        state_values = self.get_state_values()

        # Calculate sum of values for each cluster
        cluster_values = {}
        for action, states in clusters.items():
            if states:
                cluster_values[action] = float(
                    np.sum([state_values[s] for s in states])
                )
            else:
                cluster_values[action] = 0.0

        # Create mapping from state to cluster (action)
        state_to_cluster = {}
        for action, states in clusters.items():
            for state in states:
                state_to_cluster[int(state)] = int(action)

        # Create clustered directed graph
        G_clustered = nx.DiGraph()

        # Add cluster nodes with value sums
        for action, states in clusters.items():
            if states:  # Only add non-empty clusters
                G_clustered.add_node(
                    int(action),
                    name=f"Cluster {self.action_names[action]}",
                    states=[int(s) for s in states],
                    size=int(len(states)),
                    value_sum=cluster_values[action],
                    pos=(action, 0),
                )

        # Count transitions between clusters from trajectories
        cluster_transitions = defaultdict(
            lambda: defaultdict(lambda: {"count": 0, "actions": []})
        )

        for traj in trajectories:
            for i in range(len(traj["states"]) - 1):
                s = traj["states"][i]
                a = traj["actions"][i]
                s_next = traj["next_states"][i]

                cluster_from = state_to_cluster[s]
                cluster_to = state_to_cluster[s_next]

                # Record transition with action information
                cluster_transitions[cluster_from][cluster_to]["count"] += 1
                cluster_transitions[cluster_from][cluster_to]["actions"].append(a)

        # Add directed edges with probabilities and action distributions
        for cluster_from, targets in cluster_transitions.items():
            total_from = sum(targets[to]["count"] for to in targets)

            for cluster_to, data in targets.items():
                prob = data["count"] / total_from

                # Get action distribution for this transition
                action_counts = defaultdict(int)
                for a in data["actions"]:
                    action_counts[a] += 1

                action_dist = {
                    int(a): float(cnt / data["count"])
                    for a, cnt in action_counts.items()
                }

                # Add directed edge
                G_clustered.add_edge(
                    int(cluster_from),
                    int(cluster_to),
                    probability=float(prob),
                    weight=float(prob * 3),  # Scale for visualization
                    count=int(data["count"]),
                    action_distribution=action_dist,
                    dominant_action=(
                        max(action_dist, key=action_dist.get) if action_dist else None
                    ),
                )

        return G_clustered, state_to_cluster, cluster_values

    def visualize_clustered_graph(
        self,
        G_clustered: nx.DiGraph,
        cluster_values: Dict[int, float],
        save_path: Optional[str] = None,
    ):
        """
        Visualize the clustered directed graph with transition probabilities and value-based colormap
        """
        plt.figure(figsize=(14, 10))

        # Colors for different action clusters
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

        # Create colormap for node values
        values = list(cluster_values.values())
        if max(values) > min(values):
            norm = Normalize(vmin=min(values), vmax=max(values))
        else:
            norm = Normalize(vmin=0, vmax=1)

        cmap = cm.viridis

        # Calculate node positions in a circle for better visualization
        pos = nx.circular_layout(G_clustered)

        # Draw nodes with size based on number of states and color based on sum of values
        for action in G_clustered.nodes():
            node_data = G_clustered.nodes[action]
            size = node_data["size"] * 1500  # Scale node size

            # Color based on sum of values
            node_color = cmap(norm(node_data["value_sum"]))

            nx.draw_networkx_nodes(
                G_clustered,
                pos,
                nodelist=[action],
                node_size=size,
                node_color=[node_color],
                edgecolors="black",
                linewidths=2,
                alpha=0.9,
            )

            # Add cluster label
            plt.text(
                pos[action][0],
                pos[action][1] + 0.15,
                f"{self.action_names[action]}\n({node_data['size']} states)",
                horizontalalignment="center",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            # Add value sum text
            plt.text(
                pos[action][0],
                pos[action][1] - 0.15,
                f"ΣV = {node_data['value_sum']:.2f}",
                horizontalalignment="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
            )

        # Draw directed edges with probabilities
        edges = G_clustered.edges(data=True)
        if edges:
            # Separate edges by source for better organization
            for u, v, d in edges:
                # Edge width based on probability
                width = d["weight"]

                # Edge color based on dominant action
                if d["dominant_action"] is not None:
                    edge_color = colors[d["dominant_action"]]
                else:
                    edge_color = "gray"

                # Draw edge
                nx.draw_networkx_edges(
                    G_clustered,
                    pos,
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=edge_color,
                    alpha=0.7,
                    arrows=True,
                    arrowsize=25,
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0.1",
                )

                # Add edge label with probability and dominant action
                mid_point = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
                label = (
                    f"{d['probability']:.2f}\n({self.action_symbols[d['dominant_action']]})"
                    if d["dominant_action"] is not None
                    else f"{d['probability']:.2f}"
                )

                plt.text(
                    mid_point[0],
                    mid_point[1],
                    label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
                )

        # Add colorbar for node values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
        cbar.set_label("Sum of State Values in Cluster", fontsize=12)

        plt.title(
            "Clustered State Transition Graph\n(Nodes colored by sum of state values, edges show transition probabilities)",
            fontsize=14,
            pad=20,
        )
        plt.axis("off")

        # Add legend for edge colors
        legend_elements = []
        for action, color in enumerate(colors):
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=2,
                    label=f"{self.action_names[action]} action",
                )
            )

        plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Clustered graph saved to {save_path}")

        plt.show()

    def visualize_policy_heatmap(self, save_path: Optional[str] = None):
        """
        Visualize the stochastic policy as a heatmap
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        policy = self.get_stochastic_policy()

        for action in range(self.n_actions):
            ax = axes[action // 2, action % 2]
            heatmap = np.zeros((self.grid_size, self.grid_size))

            for state in range(self.n_states):
                row, col = self.state_coords[state]
                heatmap[row, col] = policy[state][action]

            im = ax.imshow(heatmap, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_title(f"Probability of {self.action_names[action]}")
            ax.set_xticks(range(self.grid_size))
            ax.set_yticks(range(self.grid_size))

            # Add text annotations
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    text = ax.text(
                        j,
                        i,
                        f"{heatmap[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

            plt.colorbar(im, ax=ax)

        plt.suptitle("Stochastic Policy Heatmap", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Heatmap saved to {save_path}")

        plt.show()

    def save_all_data(self, base_filename: str = "frozen_lake_stochastic"):
        """
        Save all learned data and analysis
        """
        os.makedirs("stochastic_data", exist_ok=True)

        # Get all data
        stochastic_policy = self.get_stochastic_policy()
        deterministic_policy = self.get_deterministic_policy()
        clusters = self.cluster_states_by_action()
        state_values = self.get_state_values()

        # Collect successful trajectories
        trajectories = self.collect_successful_trajectories(n_trajectories=1000)

        # Build graphs
        transition_graph = self.build_transition_graph(trajectories)
        clustered_graph, state_to_cluster, cluster_values = self.create_clustered_graph(
            trajectories
        )

        # Convert all data to JSON-serializable format
        serializable_data = {
            "stochastic_policy": {
                str(k): [float(p) for p in v] for k, v in stochastic_policy.items()
            },
            "deterministic_policy": {
                str(k): int(v) for k, v in deterministic_policy.items()
            },
            "state_values": {str(k): float(v) for k, v in enumerate(state_values)},
            "clusters": {str(k): [int(s) for s in v] for k, v in clusters.items()},
            "cluster_values": {str(k): float(v) for k, v in cluster_values.items()},
            "state_to_cluster": {str(k): int(v) for k, v in state_to_cluster.items()},
            "training_history": {
                "success_rate": [float(x) for x in self.success_rate_history],
                "rewards": [float(x) for x in self.rewards_history],
            },
            "parameters": {
                "map_name": self.map_name,
                "is_slippery": bool(self.is_slippery),
                "learning_rate": float(self.lr),
                "discount_factor": float(self.gamma),
                "temperature": float(self.temperature),
                "episodes": int(self.episodes),
            },
        }

        # Save as JSON
        with open(f"stochastic_data/{base_filename}_policy.json", "w") as f:
            json.dump(serializable_data, f, indent=2)

        # Save trajectories
        serializable_trajectories = []
        for traj in trajectories:
            serializable_traj = {
                "states": [int(s) for s in traj["states"]],
                "actions": [int(a) for a in traj["actions"]],
                "action_probs": [
                    [float(p) for p in probs] for probs in traj["action_probs"]
                ],
                "rewards": [float(r) for r in traj["rewards"]],
                "next_states": [int(s) for s in traj["next_states"]],
                "success": bool(traj["success"]),
            }
            serializable_trajectories.append(serializable_traj)

        with open(f"stochastic_data/{base_filename}_trajectories.json", "w") as f:
            json.dump(serializable_trajectories, f, indent=2)

        # Save graphs
        with open(f"stochastic_data/{base_filename}_transition_graph.pkl", "wb") as f:
            pickle.dump(transition_graph, f)

        with open(f"stochastic_data/{base_filename}_clustered_graph.pkl", "wb") as f:
            pickle.dump(clustered_graph, f)

        # Create visualizations
        self.visualize_policy_heatmap(
            save_path=f"stochastic_data/{base_filename}_heatmap.png"
        )
        self.visualize_clustered_graph(
            clustered_graph,
            cluster_values,
            save_path=f"stochastic_data/{base_filename}_clustered_graph.png",
        )

        # Save summary
        summary = {
            "final_success_rate": (
                float(self.success_rate_history[-1])
                if self.success_rate_history
                else 0.0
            ),
            "cluster_sizes": {str(k): int(len(v)) for k, v in clusters.items()},
            "cluster_values": {str(k): float(v) for k, v in cluster_values.items()},
            "num_trajectories": int(len(trajectories)),
            "avg_trajectory_length": (
                float(np.mean([len(t["states"]) for t in trajectories]))
                if trajectories
                else 0.0
            ),
        }

        with open(f"stochastic_data/{base_filename}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save transition matrix as CSV for easy viewing
        self.save_transition_matrix(
            clustered_graph, f"stochastic_data/{base_filename}_transition_matrix.csv"
        )

        print(f"\nAll data saved in 'stochastic_data/' directory")
        print(f"Files saved with base name: {base_filename}")

        return (
            serializable_data,
            trajectories,
            transition_graph,
            clustered_graph,
            cluster_values,
        )

    def save_transition_matrix(self, G_clustered: nx.DiGraph, filename: str):
        """
        Save transition matrix as CSV
        """
        import csv

        # Create transition matrix
        matrix = np.zeros((self.n_actions, self.n_actions))

        for u, v, d in G_clustered.edges(data=True):
            matrix[u, v] = d["probability"]

        # Save to CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            header = ["From/To"] + [self.action_names[i] for i in range(self.n_actions)]
            writer.writerow(header)

            # Write rows
            for i in range(self.n_actions):
                row = [self.action_names[i]] + [
                    f"{matrix[i, j]:.3f}" for j in range(self.n_actions)
                ]
                writer.writerow(row)

        print(f"Transition matrix saved to {filename}")


def analyze_clustered_transitions(
    clustered_graph: nx.DiGraph, cluster_values: Dict[int, float]
):
    """
    Analyze and print transition probabilities between clusters
    """
    print("\n" + "=" * 60)
    print("CLUSTER TRANSITION ANALYSIS")
    print("=" * 60)

    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    print("\n" + "=" * 60)
    print("TRANSITION PROBABILITIES BETWEEN CLUSTERS")
    print("=" * 60)

    # Create transition matrix
    transition_matrix = np.zeros((4, 4))

    print("\nDirected Edges with Probabilities:")
    print("-" * 60)
    for edge in clustered_graph.edges(data=True):
        from_cluster = edge[0]
        to_cluster = edge[1]
        data = edge[2]
        prob = data["probability"]
        dominant_action = data["dominant_action"]

        transition_matrix[from_cluster, to_cluster] = prob

        action_info = (
            f" (dominant: {action_symbols[dominant_action]})"
            if dominant_action is not None
            else ""
        )
        print(
            f"{action_names[from_cluster]:6} → {action_names[to_cluster]:6}: {prob:.3f}{action_info}"
        )

        # Print action distribution for this transition
        if "action_distribution" in data:
            print(
                f"      Action dist: {', '.join([f'{action_symbols[a]}:{p:.2f}' for a, p in data['action_distribution'].items()])}"
            )

    print("\n" + "=" * 60)
    print("CLUSTER VALUE ANALYSIS")
    print("=" * 60)

    print("\nCluster Values (sum of state values):")
    for action, value in cluster_values.items():
        print(f"{action_names[action]:6}: {value:.3f}")

    print("\n" + "=" * 60)
    print("TRANSITION MATRIX")
    print("=" * 60)

    print("\n      To:")
    print("      " + " ".join([f"{action_names[i]:6}" for i in range(4)]))
    print("From: " + "-" * 40)
    for i in range(4):
        row_str = f"{action_names[i]:6}: "
        for j in range(4):
            row_str += f"{transition_matrix[i, j]:.3f}   "
        print(row_str)


def main():
    """
    Main experiment function
    """
    print("=" * 60)
    print("STOCHASTIC POLICY LEARNING ON FROZENLAKE")
    print("=" * 60)

    # Initialize learner
    learner = StochasticPolicyLearner(
        map_name="4x4", is_slippery=True, episodes=20000, temperature=1.0
    )

    # Train
    results = learner.train(verbose=True)

    # Get policies
    stochastic_policy = learner.get_stochastic_policy()
    deterministic_policy = learner.get_deterministic_policy()
    state_values = learner.get_state_values()

    print("\n" + "=" * 60)
    print("DETERMINISTIC POLICY (most probable action per state)")
    print("=" * 60)

    # Print policy grid with values
    print("\nPolicy Grid (Action | State Value):")
    print("-" * 45)
    for row in range(4):
        row_str = ""
        for col in range(4):
            state = row * 4 + col
            action = deterministic_policy[state]
            value = state_values[state]
            row_str += f"| {learner.action_symbols[action]} {value:.2f} "
        print(row_str + "|")
        print("-" * 45)

    # Cluster states by action
    clusters = learner.cluster_states_by_action()

    print("\n" + "=" * 60)
    print("STATE CLUSTERS BY PREFERRED ACTION")
    print("=" * 60)

    for action, states in clusters.items():
        print(f"\n{learner.action_names[action]} (Action {action}):")
        print(f"  States: {states}")
        print(f"  Values: {[f'{state_values[s]:.2f}' for s in states]}")
        print(f"  Sum of values: {np.sum([state_values[s] for s in states]):.3f}")

        # Show grid positions for this cluster
        grid = np.full((4, 4), ".", dtype=str)
        for state in states:
            row, col = divmod(state, 4)
            grid[row, col] = learner.action_symbols[action]

        print("  Grid positions:")
        for row in range(4):
            print("   ", " ".join(grid[row]))

    # Collect successful trajectories
    trajectories = learner.collect_successful_trajectories(n_trajectories=1000)

    # Create clustered graph
    clustered_graph, state_to_cluster, cluster_values = learner.create_clustered_graph(
        trajectories
    )

    # Analyze transitions
    analyze_clustered_transitions(clustered_graph, cluster_values)

    # Save all data
    data, traj, trans_graph, clust_graph, clust_values = learner.save_all_data()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Load the saved data from 'stochastic_data/' directory")
    print("2. View the clustered directed graph with transition probabilities")
    print("3. Analyze state-action probabilities and state values")
    print("4. Examine the transition matrix for cluster-to-cluster dynamics")

    return learner, trajectories, clustered_graph, cluster_values


if __name__ == "__main__":
    # Run main experiment
    learner, trajectories, clustered_graph, cluster_values = main()
