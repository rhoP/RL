import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import pickle
import os


class MetaLevelLearner:
    """
    Learn meta-level transitions and values using exploration policy
    """

    def __init__(
        self,
        env,
        refinement_analyzer,
        superior_policy,
        exploration_prob: float = 0.1,
        gamma: float = 0.99,
    ):
        self.env = env
        self.refinement = refinement_analyzer.refinement
        self.state_to_cluster = refinement_analyzer.state_to_cluster
        self.superior_policy = superior_policy
        self.exploration_prob = exploration_prob
        self.gamma = gamma

        self.n_clusters = len(self.refinement["clusters"])
        self.n_actions = env.action_space.n

        # Meta-level models
        self.cluster_values = np.zeros(self.n_clusters)  # V(C)
        self.cluster_q_values = np.zeros((self.n_clusters, self.n_actions))  # Q(C, a)
        self.cluster_transitions = defaultdict(
            lambda: defaultdict(float)
        )  # P(C' | C, a)
        self.cluster_visits = np.zeros(self.n_clusters)
        self.cluster_action_counts = np.zeros((self.n_clusters, self.n_actions))

        # For tracking
        self.meta_trajectories = []
        self.learning_rate = 0.1

    def get_action(self, state: int, use_exploration: bool = True) -> int:
        """
        Get action: superior policy with occasional exploration
        """
        if use_exploration and np.random.random() < self.exploration_prob:
            return np.random.randint(self.n_actions)
        else:
            return self.superior_policy[state]

    def collect_meta_trajectories(self, n_episodes: int = 1000) -> List[Dict]:
        """
        Collect trajectories with exploration to learn meta-level model
        """
        print(
            f"\nCollecting {n_episodes} meta-trajectories with ε={self.exploration_prob}..."
        )

        trajectories = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            meta_trajectory = {
                "states": [],
                "clusters": [],
                "actions": [],
                "rewards": [],
                "next_clusters": [],
                "exploration_used": [],
            }

            while not done:
                # Get action with exploration
                action = self.get_action(state)
                is_exploration = np.random.random() < self.exploration_prob

                # Take step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Get clusters
                current_cluster = self.state_to_cluster[state]
                next_cluster = self.state_to_cluster[next_state]

                # Store meta-transition
                meta_trajectory["states"].append(state)
                meta_trajectory["clusters"].append(current_cluster)
                meta_trajectory["actions"].append(action)
                meta_trajectory["rewards"].append(reward)
                meta_trajectory["next_clusters"].append(next_cluster)
                meta_trajectory["exploration_used"].append(is_exploration)

                # Update counts
                self.cluster_visits[current_cluster] += 1
                self.cluster_action_counts[current_cluster, action] += 1

                # Update transition counts
                self.cluster_transitions[(current_cluster, action)][next_cluster] += 1

                state = next_state
                total_reward += reward

            meta_trajectory["total_reward"] = total_reward
            meta_trajectory["success"] = total_reward > 0
            trajectories.append(meta_trajectory)

            if (episode + 1) % 200 == 0:
                print(f"  Collected {episode + 1} episodes")

        self.meta_trajectories = trajectories

        # Normalize transitions
        for c, a in list(self.cluster_transitions.keys()):
            total = sum(self.cluster_transitions[(c, a)].values())
            if total > 0:
                for next_c in self.cluster_transitions[(c, a)]:
                    self.cluster_transitions[(c, a)][next_c] /= total

        print(f"\nCollected {len(trajectories)} meta-trajectories")
        print(f"Total cluster visits: {np.sum(self.cluster_visits)}")

        return trajectories

    def learn_cluster_values_mc(self):
        """
        Learn cluster values using Monte Carlo returns
        """
        print("\nLearning cluster values via Monte Carlo...")

        # Initialize returns
        cluster_returns = defaultdict(list)

        for traj in self.meta_trajectories:
            G = 0
            # Work backwards
            for t in range(len(traj["rewards"]) - 1, -1, -1):
                G = self.gamma * G + traj["rewards"][t]
                cluster = traj["clusters"][t]
                cluster_returns[cluster].append(G)

        # Update cluster values
        for cluster in range(self.n_clusters):
            if cluster_returns[cluster]:
                self.cluster_values[cluster] = np.mean(cluster_returns[cluster])

        print(
            f"Cluster value range: [{np.min(self.cluster_values):.3f}, {np.max(self.cluster_values):.3f}]"
        )

    def learn_cluster_q_values_td(self, n_iterations: int = 5):
        """
        Learn cluster Q-values using TD learning on meta-transitions
        """
        print("\nLearning cluster Q-values via TD learning...")

        for iteration in range(n_iterations):
            td_errors = []

            for traj in self.meta_trajectories:
                for t in range(len(traj["rewards"])):
                    c = traj["clusters"][t]
                    a = traj["actions"][t]
                    r = traj["rewards"][t]
                    c_next = traj["next_clusters"][t]

                    # Get best next action value (if not terminal)
                    if t < len(traj["rewards"]) - 1:
                        next_max_q = np.max(self.cluster_q_values[c_next])
                    else:
                        next_max_q = 0

                    # TD target
                    target = r + self.gamma * next_max_q

                    # TD error
                    td_error = target - self.cluster_q_values[c, a]
                    td_errors.append(abs(td_error))

                    # Update
                    self.cluster_q_values[c, a] += self.learning_rate * td_error

            if iteration % 1 == 0:
                print(
                    f"  Iteration {iteration}: Mean TD error = {np.mean(td_errors):.4f}"
                )

    def compute_meta_transition_matrix(self) -> np.ndarray:
        """
        Compute meta-level transition matrix P(C' | C)
        """
        trans_matrix = np.zeros((self.n_clusters, self.n_clusters))

        for c in range(self.n_clusters):
            total = 0
            for a in range(self.n_actions):
                if (c, a) in self.cluster_transitions:
                    for c_next, prob in self.cluster_transitions[(c, a)].items():
                        # Weight by action probability
                        action_prob = self.cluster_action_counts[c, a] / max(
                            1, self.cluster_visits[c]
                        )
                        trans_matrix[c, c_next] += prob * action_prob
                        total += prob * action_prob

            if total > 0:
                trans_matrix[c] /= total

        return trans_matrix

    def compute_cluster_entropy(self) -> np.ndarray:
        """
        Compute entropy of transitions from each cluster
        High entropy = stochastic region
        Low entropy = deterministic region
        """
        trans_matrix = self.compute_meta_transition_matrix()
        entropies = []

        for c in range(self.n_clusters):
            probs = trans_matrix[c]
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs))
            else:
                entropy = 0
            entropies.append(entropy)

        return np.array(entropies)

    def identify_bottleneck_clusters(self, threshold: float = 0.7) -> List[int]:
        """
        Identify bottleneck clusters (high betweenness in meta-graph)
        """
        # Build meta-graph
        G = nx.DiGraph()

        for c in range(self.n_clusters):
            G.add_node(c, visits=self.cluster_visits[c])

        trans_matrix = self.compute_meta_transition_matrix()
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if trans_matrix[i, j] > 0.01:
                    G.add_edge(i, j, weight=trans_matrix[i, j])

        # Compute betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G, weight="weight")
            # Normalize by visits
            bottleneck_scores = []
            for c in range(self.n_clusters):
                if self.cluster_visits[c] > 0:
                    score = betweenness[c] * np.log1p(self.cluster_visits[c])
                    bottleneck_scores.append((c, score))

            # Sort by score
            bottleneck_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top clusters
            n_bottlenecks = max(1, int(self.n_clusters * 0.2))
            bottlenecks = [c for c, _ in bottleneck_scores[:n_bottlenecks]]

            return bottlenecks
        except:
            return []

    def find_optimal_meta_path(self, start_state: int, goal_state: int) -> List[int]:
        """
        Find optimal path through clusters using meta-level model
        """
        start_cluster = self.state_to_cluster[start_state]
        goal_cluster = self.state_to_cluster[goal_state]

        # Build graph with negative log probabilities as costs
        G = nx.DiGraph()
        trans_matrix = self.compute_meta_transition_matrix()

        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if trans_matrix[i, j] > 0:
                    # Use negative log probability as cost
                    cost = -np.log(trans_matrix[i, j])
                    G.add_edge(i, j, weight=cost)

        try:
            path = nx.shortest_path(G, start_cluster, goal_cluster, weight="weight")
            return path
        except:
            return []

    def visualize_meta_graph(self, save_path: Optional[str] = None):
        """
        Visualize the meta-level transition graph
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Meta transition matrix
        ax = axes[0]
        trans_matrix = self.compute_meta_transition_matrix()
        im = ax.imshow(trans_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xlabel("To Cluster")
        ax.set_ylabel("From Cluster")
        ax.set_title("Meta Transition Probabilities")
        plt.colorbar(im, ax=ax)

        # Add text for significant transitions
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if trans_matrix[i, j] > 0.1:
                    ax.text(
                        j,
                        i,
                        f"{trans_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if trans_matrix[i, j] > 0.5 else "black",
                    )

        # Plot 2: Cluster values
        ax = axes[1]
        clusters = range(self.n_clusters)
        colors = ["green" if v > 0 else "red" for v in self.cluster_values]
        ax.bar(clusters, self.cluster_values, color=colors, alpha=0.7)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Value")
        ax.set_title("Cluster Values (V(C))")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 3: Meta graph
        ax = axes[2]

        # Build graph
        G = nx.DiGraph()
        for c in range(self.n_clusters):
            G.add_node(c, size=self.cluster_visits[c])

        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if trans_matrix[i, j] > 0.1:
                    G.add_edge(i, j, weight=trans_matrix[i, j])

        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes with size proportional to visits
        node_sizes = [G.nodes[c]["size"] * 10 for c in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7, ax=ax
        )

        # Draw edges with width proportional to probability
        edges = G.edges(data=True)
        if edges:
            edge_weights = [d["weight"] * 3 for _, _, d in edges]
            nx.draw_networkx_edges(
                G,
                pos,
                width=edge_weights,
                alpha=0.5,
                edge_color="gray",
                arrows=True,
                arrowsize=15,
                arrowstyle="->",
                ax=ax,
            )

        # Add labels
        labels = {c: f"C{c}\n{V:.2f}" for c, V in zip(clusters, self.cluster_values)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        ax.set_title(
            "Meta-Level Transition Graph\n(Node size = visits, Edge width = probability)"
        )
        ax.axis("off")

        plt.suptitle("Meta-Level Learning Results", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Meta graph saved to {save_path}")

        plt.show()

    def print_meta_analysis(self):
        """
        Print comprehensive meta-level analysis
        """
        print("\n" + "=" * 60)
        print("META-LEVEL ANALYSIS")
        print("=" * 60)

        print(f"\nMeta-level statistics:")
        print(f"  Number of clusters: {self.n_clusters}")
        print(f"  Total meta-trajectories: {len(self.meta_trajectories)}")
        print(f"  Total cluster visits: {int(np.sum(self.cluster_visits))}")

        # Visit distribution
        visit_probs = self.cluster_visits / np.sum(self.cluster_visits)
        print(f"\nCluster visit distribution:")
        for c in range(self.n_clusters):
            if visit_probs[c] > 0.05:
                print(
                    f"  Cluster {c}: {visit_probs[c]:.2%} of visits, value={self.cluster_values[c]:.3f}"
                )

        # Entropy analysis
        entropies = self.compute_cluster_entropy()
        print(f"\nTransition entropy by cluster:")
        high_entropy = np.where(entropies > np.mean(entropies) + np.std(entropies))[0]
        low_entropy = np.where(entropies < np.mean(entropies) - np.std(entropies))[0]

        if len(high_entropy) > 0:
            print(f"  High entropy (stochastic): {high_entropy}")
        if len(low_entropy) > 0:
            print(f"  Low entropy (deterministic): {low_entropy}")

        # Bottleneck clusters
        bottlenecks = self.identify_bottleneck_clusters()
        if bottlenecks:
            print(f"\nBottleneck clusters (key decision points): {bottlenecks}")

        # Meta transition matrix summary
        trans_matrix = self.compute_meta_transition_matrix()
        print(
            f"\nMeta transition matrix sparsity: {np.sum(trans_matrix > 0)}/{self.n_clusters**2} transitions"
        )

        # Find optimal path
        optimal_path = self.find_optimal_meta_path(0, 15)
        if optimal_path:
            path_str = " → ".join([f"C{c}" for c in optimal_path])
            path_prob = 1.0
            for i in range(len(optimal_path) - 1):
                path_prob *= trans_matrix[optimal_path[i], optimal_path[i + 1]]
            print(f"\nOptimal meta-path: {path_str}")
            print(f"Path probability: {path_prob:.3f}")


def run_meta_learning_experiment(
    refinement_analyzer,
    superior_policy,
    exploration_prob: float = 0.1,
    n_episodes: int = 1000,
):
    """
    Run complete meta-learning experiment
    """
    print("\n" + "=" * 70)
    print("META-LEVEL LEARNING EXPERIMENT")
    print("=" * 70)

    # Create meta-learner
    meta_learner = MetaLevelLearner(
        env=refinement_analyzer.env,
        refinement_analyzer=refinement_analyzer,
        superior_policy=superior_policy,
        exploration_prob=exploration_prob,
    )

    # Collect meta-trajectories with exploration
    trajectories = meta_learner.collect_meta_trajectories(n_episodes=n_episodes)

    # Learn cluster values via Monte Carlo
    meta_learner.learn_cluster_values_mc()

    # Learn cluster Q-values via TD
    meta_learner.learn_cluster_q_values_td(n_iterations=5)

    # Compute meta transition matrix
    trans_matrix = meta_learner.compute_meta_transition_matrix()
    print("\nMeta transition matrix (C → C'):")
    print(np.round(trans_matrix, 3))

    # Identify bottlenecks
    bottlenecks = meta_learner.identify_bottleneck_clusters()
    print(f"\nBottleneck clusters: {bottlenecks}")

    # Find optimal path
    optimal_path = meta_learner.find_optimal_meta_path(0, 15)
    if optimal_path:
        print(f"\nOptimal meta-path from start to goal:")
        path_str = " → ".join([f"C{c}" for c in optimal_path])
        print(f"  {path_str}")

    # Visualize
    meta_learner.visualize_meta_graph(save_path="meta_analysis/meta_graph.png")

    # Print analysis
    meta_learner.print_meta_analysis()

    # Save results
    os.makedirs("meta_analysis", exist_ok=True)

    results = {
        "cluster_values": meta_learner.cluster_values.tolist(),
        "cluster_q_values": meta_learner.cluster_q_values.tolist(),
        "transition_matrix": trans_matrix.tolist(),
        "bottlenecks": bottlenecks,
        "optimal_path": optimal_path if optimal_path else [],
        "entropies": meta_learner.compute_cluster_entropy().tolist(),
    }

    import json

    with open("meta_analysis/meta_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return meta_learner


# Example usage with your refinement analyzer
if __name__ == "__main__":
    # Assuming you have a refinement_analyzer from previous steps
    # and a superior_policy function

    # meta_learner = run_meta_learning_experiment(
    #     refinement_analyzer=refinement_analyzer,
    #     superior_policy=superior_policy,
    #     exploration_prob=0.1,
    #     n_episodes=1000
    # )

    print("\nMeta-learning module ready. Use with your refinement analyzer.")
