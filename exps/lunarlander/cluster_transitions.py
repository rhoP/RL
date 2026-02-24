import numpy as np
import gymnasium as gym
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import ks_2samp, entropy
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import torch as th
import warnings

warnings.filterwarnings("ignore")


class CombinatorialClusterTransitionAnalyzer:
    """
    Analyze transitions between clusters using combinatorial aggregation
    Treats cluster transitions as a Markov chain
    """

    def __init__(self, env, agent, cluster_analyzer):
        self.env = env
        self.agent = agent
        self.cluster_analyzer = cluster_analyzer
        self.clusters = cluster_analyzer.clusters
        self.state_to_cluster = cluster_analyzer.state_to_cluster
        self.n_clusters = len(self.clusters)

        # Transition matrices
        self.theoretical_transitions = None  # P_{theo}(i -> j)
        self.empirical_transitions = None  # P_{emp}(i -> j)

        # Markov chain properties
        self.stationary_distribution = None
        self.mean_first_passage = None
        self.absorption_times = None

    def get_action_from_policy(
        self, state: np.ndarray, deterministic: bool = True
    ) -> int:
        """Get action from the policy network"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        state_tensor = th.as_tensor(state).float().to(self.agent.device)

        with th.no_grad():
            if hasattr(self.agent.policy, "get_distribution"):
                dist = self.agent.policy.get_distribution(state_tensor)
                if deterministic:
                    action = dist.mode()
                else:
                    action = dist.sample()
            else:
                action, _ = self.agent.predict(state, deterministic=deterministic)

        if isinstance(action, th.Tensor):
            action = action.cpu().numpy()

        return int(action[0] if hasattr(action, "__len__") else action)

    def compute_theoretical_transitions_combinatorial(
        self, n_samples_per_state: int = 30
    ) -> np.ndarray:
        """
        Compute theoretical transition probabilities by aggregating over all states
        This creates a Markov chain where states are clusters
        """
        print("\n" + "=" * 60)
        print("COMPUTING THEORETICAL CLUSTER TRANSITIONS (COMBINATORIAL)")
        print("=" * 60)

        # Initialize transition count matrix
        trans_counts = np.zeros((self.n_clusters, self.n_clusters))
        state_counts = np.zeros(self.n_clusters)

        total_states = sum(c.size for c in self.clusters)
        processed = 0

        for i, cluster in enumerate(self.clusters):
            for state in cluster.states:
                # Get action from policy
                action = self.get_action_from_policy(state, deterministic=True)

                # Sample next states
                next_states = []
                for _ in range(n_samples_per_state):
                    self.env.reset()
                    self.env.unwrapped.state = state.copy()
                    next_state, _, terminated, truncated, _ = self.env.step(action)
                    if not (terminated or truncated):
                        next_states.append(next_state)

                if not next_states:
                    continue

                # Find which clusters these next states belong to
                for next_state in next_states:
                    # Find closest cluster by centroid
                    min_dist = float("inf")
                    closest = None
                    for j, target_cluster in enumerate(self.clusters):
                        dist = np.linalg.norm(next_state - target_cluster.centroid)
                        if dist < min_dist:
                            min_dist = dist
                            closest = j

                    if closest is not None:
                        trans_counts[i, closest] += 1

                state_counts[i] += 1
                processed += 1

                if processed % 50 == 0:
                    print(f"  Processed {processed}/{total_states} states")

        # Convert to probabilities
        self.theoretical_transitions = np.zeros((self.n_clusters, self.n_clusters))
        for i in range(self.n_clusters):
            if state_counts[i] > 0:
                self.theoretical_transitions[i] = trans_counts[i] / (
                    state_counts[i] * n_samples_per_state
                )

        # Verify it's a valid Markov chain (rows sum to 1)
        row_sums = self.theoretical_transitions.sum(axis=1)
        for i in range(self.n_clusters):
            if row_sums[i] > 0 and abs(row_sums[i] - 1.0) > 1e-6:
                self.theoretical_transitions[i] /= row_sums[i]

        print(
            f"\nTheoretical transition matrix shape: {self.theoretical_transitions.shape}"
        )
        print(
            f"Average outgoing transitions per cluster: {np.sum(self.theoretical_transitions > 0, axis=1).mean():.1f}"
        )

        return self.theoretical_transitions

    def compute_empirical_transitions_combinatorial(
        self, trajectories: List[Dict]
    ) -> np.ndarray:
        """
        Compute empirical transition probabilities by counting observed cluster transitions
        """
        print("\n" + "=" * 60)
        print("COMPUTING EMPIRICAL CLUSTER TRANSITIONS")
        print("=" * 60)

        trans_counts = np.zeros((self.n_clusters, self.n_clusters))
        start_counts = np.zeros(self.n_clusters)

        total_transitions = 0

        for traj_idx, traj in enumerate(trajectories):
            prev_cluster = None

            for state in traj["states"]:
                # Find current cluster
                current_cluster = self._find_cluster(state)

                if prev_cluster is not None and current_cluster is not None:
                    if prev_cluster != current_cluster:
                        trans_counts[prev_cluster, current_cluster] += 1
                        start_counts[prev_cluster] += 1
                        total_transitions += 1

                prev_cluster = current_cluster

            if (traj_idx + 1) % 50 == 0:
                print(f"  Processed {traj_idx + 1}/{len(trajectories)} trajectories")

        # Convert to probabilities
        self.empirical_transitions = np.zeros((self.n_clusters, self.n_clusters))
        for i in range(self.n_clusters):
            if start_counts[i] > 0:
                self.empirical_transitions[i] = trans_counts[i] / start_counts[i]

        print(f"\nTotal transitions observed: {total_transitions}")
        print(
            f"Average transitions per cluster: {total_transitions / self.n_clusters:.1f}"
        )

        return self.empirical_transitions

    def _find_cluster(self, state: np.ndarray) -> Optional[int]:
        """Find the closest cluster to a given state"""
        min_dist = float("inf")
        closest = None

        for cluster in self.clusters:
            dist = np.linalg.norm(state - cluster.centroid)
            if dist < min_dist:
                min_dist = dist
                closest = cluster.id

        return closest

    def compute_markov_properties(self) -> Dict:
        """
        Compute Markov chain properties from the transition matrices
        """
        if self.theoretical_transitions is None:
            return {}

        properties = {}

        # Stationary distribution (if ergodic)
        try:
            eigenvals, eigenvecs = np.linalg.eig(self.theoretical_transitions.T)
            stationary = eigenvecs[:, np.isclose(eigenvals, 1.0)]
            if stationary.size > 0:
                stationary = stationary[:, 0].real
                stationary = stationary / stationary.sum()
                self.stationary_distribution = stationary
                properties["stationary_distribution"] = stationary.tolist()
        except:
            print("  Could not compute stationary distribution")

        # Mean first passage times (simplified)
        n = self.n_clusters
        mfpt = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Simple Monte Carlo estimation
                    total = 0
                    for _ in range(100):  # Sample 100 paths
                        current = i
                        steps = 0
                        while current != j and steps < 100:
                            probs = self.theoretical_transitions[current]
                            if probs.sum() > 0:
                                current = np.random.choice(n, p=probs)
                            steps += 1
                        if current == j:
                            total += steps
                    mfpt[i, j] = total / 100 if total > 0 else np.inf

        properties["mean_first_passage"] = mfpt.tolist()

        return properties

    def compare_matrices(self) -> Dict:
        """
        Compare theoretical and empirical transition matrices
        """
        if self.theoretical_transitions is None or self.empirical_transitions is None:
            return {}

        comparison = {
            "mse": float(
                np.mean(
                    (self.theoretical_transitions - self.empirical_transitions) ** 2
                )
            ),
            "mae": float(
                np.mean(
                    np.abs(self.theoretical_transitions - self.empirical_transitions)
                )
            ),
            "max_error": float(
                np.max(
                    np.abs(self.theoretical_transitions - self.empirical_transitions)
                )
            ),
            "l1_norm": float(
                np.sum(
                    np.abs(self.theoretical_transitions - self.empirical_transitions)
                )
            ),
            "row_correlations": [],
            "row_kl": [],
        }

        # Per-row analysis
        for i in range(self.n_clusters):
            p = self.theoretical_transitions[i] + 1e-10
            q = self.empirical_transitions[i] + 1e-10
            p /= p.sum()
            q /= q.sum()

            # Correlation
            corr = np.corrcoef(p, q)[0, 1] if len(p) > 1 else 0
            comparison["row_correlations"].append(float(corr))

            # KL divergence
            from scipy.stats import entropy

            kl = entropy(p, q)
            comparison["row_kl"].append(float(kl))

        comparison["avg_correlation"] = float(np.mean(comparison["row_correlations"]))
        comparison["avg_kl"] = float(np.mean(comparison["row_kl"]))

        return comparison

    def find_absorbing_sets(self, threshold: float = 0.9) -> List[List[int]]:
        """
        Find absorbing sets of clusters (where probability of leaving is low)
        """
        if self.theoretical_transitions is None:
            return []

        absorbing_sets = []
        visited = set()

        for i in range(self.n_clusters):
            if i in visited:
                continue

            # Check if this cluster has high self-loop probability
            if self.theoretical_transitions[i, i] > threshold:
                # Find all clusters reachable with high probability
                component = [i]
                stack = [i]
                visited.add(i)

                while stack:
                    current = stack.pop()
                    for j in range(self.n_clusters):
                        if j not in visited and self.theoretical_transitions[
                            current, j
                        ] > (1 - threshold):
                            visited.add(j)
                            stack.append(j)
                            component.append(j)

                if len(component) > 1 or self.theoretical_transitions[i, i] > threshold:
                    absorbing_sets.append(component)

        return absorbing_sets

    def compute_transition_entropy(self) -> np.ndarray:
        """
        Compute entropy of outgoing transitions for each cluster
        High entropy = many possible next clusters (stochastic region)
        Low entropy = deterministic behavior
        """
        if self.theoretical_transitions is None:
            return np.zeros(self.n_clusters)

        entropies = []
        for i in range(self.n_clusters):
            probs = self.theoretical_transitions[i]
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs))
            else:
                entropy = 0
            entropies.append(entropy)

        return np.array(entropies)

    def visualize_transition_graph(self, save_path: Optional[str] = None):
        """
        Visualize the cluster transition graph
        """
        if self.theoretical_transitions is None:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Theoretical transition matrix heatmap
        ax = axes[0]
        im = ax.imshow(
            self.theoretical_transitions, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1
        )
        ax.set_title("Theoretical Transition Matrix")
        ax.set_xlabel("To Cluster")
        ax.set_ylabel("From Cluster")
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if self.theoretical_transitions[i, j] > 0.05:
                    ax.text(
                        j,
                        i,
                        f"{self.theoretical_transitions[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white"
                        if self.theoretical_transitions[i, j] > 0.5
                        else "black",
                    )

        # Plot 2: NetworkX graph
        ax = axes[1]
        G = nx.DiGraph()

        # Add nodes
        for i, cluster in enumerate(self.clusters):
            G.add_node(i, size=cluster.size, action=cluster.dominant_action)

        # Add edges (only significant ones)
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if i != j and self.theoretical_transitions[i, j] > 0.1:
                    G.add_edge(i, j, weight=self.theoretical_transitions[i, j])

        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes with size proportional to cluster size
        node_sizes = [G.nodes[i]["size"] * 50 for i in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7, ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, {i: f"C{i}" for i in G.nodes()}, font_size=8, ax=ax
        )

        # Draw edges with width proportional to probability
        edges = G.edges(data=True)
        if edges:
            edge_weights = [d["weight"] * 3 for (_, _, d) in edges]
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

        ax.set_title("Cluster Transition Graph\n(edges > 0.1)")
        ax.axis("off")

        # Plot 3: Transition entropy
        ax = axes[2]
        entropies = self.compute_transition_entropy()
        bars = ax.bar(range(self.n_clusters), entropies)

        # Color bars by entropy
        for bar, ent in zip(bars, entropies):
            if ent > np.mean(entropies):
                bar.set_color("red")
            else:
                bar.set_color("blue")

        ax.axhline(
            y=np.mean(entropies),
            color="black",
            linestyle="--",
            label=f"Mean: {np.mean(entropies):.2f}",
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title(
            "Transition Entropy per Cluster\n(High = stochastic, Low = deterministic)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Cluster Transition Analysis - {self.cluster_analyzer.name}", fontsize=14
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Graph saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print a comprehensive summary"""
        print("\n" + "=" * 70)
        print(f"CLUSTER TRANSITION SUMMARY - {self.cluster_analyzer.name}")
        print("=" * 70)

        if self.theoretical_transitions is not None:
            print("\nTHEORETICAL TRANSITIONS:")
            print(f"  Number of clusters: {self.n_clusters}")
            print(
                f"  Average outgoing transitions per cluster: {np.sum(self.theoretical_transitions > 0.01, axis=1).mean():.1f}"
            )
            print(
                f"  Max self-loop probability: {np.max(np.diag(self.theoretical_transitions)):.3f}"
            )
            print(
                f"  Average off-diagonal probability: {np.mean(self.theoretical_transitions[~np.eye(self.n_clusters, dtype=bool)]):.3f}"
            )

        if self.empirical_transitions is not None:
            print("\nEMPIRICAL TRANSITIONS:")
            print(
                f"  Average outgoing transitions per cluster: {np.sum(self.empirical_transitions > 0.01, axis=1).mean():.1f}"
            )

        comparison = self.compare_matrices()
        if comparison:
            print("\nTHEORETICAL vs EMPIRICAL:")
            print(f"  MSE: {comparison['mse']:.4f}")
            print(f"  MAE: {comparison['mae']:.4f}")
            print(f"  Average correlation: {comparison['avg_correlation']:.4f}")
            print(f"  Average KL divergence: {comparison['avg_kl']:.4f}")

        absorbing = self.find_absorbing_sets()
        if absorbing:
            print("\nABSORBING SETS:")
            for i, aset in enumerate(absorbing):
                print(f"  Set {i + 1}: Clusters {aset}")

        entropies = self.compute_transition_entropy()
        print("\nTRANSITION ENTROPY:")
        print(f"  Average entropy: {np.mean(entropies):.3f} bits")
        print(
            f"  Most deterministic cluster: C{np.argmin(entropies)} ({entropies[np.argmin(entropies)]:.3f} bits)"
        )
        print(
            f"  Most stochastic cluster: C{np.argmax(entropies)} ({entropies[np.argmax(entropies)]:.3f} bits)"
        )


def analyze_cluster_transitions_combinatorial(agent, analyzer, trajectories, env):
    """
    Comprehensive combinatorial analysis of cluster transitions
    """
    print("\n" + "=" * 80)
    print(f"COMBINATORIAL CLUSTER TRANSITION ANALYSIS - {analyzer.name}")
    print("=" * 80)

    # Initialize analyzer
    trans_analyzer = CombinatorialClusterTransitionAnalyzer(env, agent, analyzer)

    # Compute theoretical transitions
    theoretical = trans_analyzer.compute_theoretical_transitions_combinatorial(
        n_samples_per_state=30
    )

    # Print theoretical matrix
    print("\nTheoretical transition matrix (rows sum to 1):")
    print(np.round(theoretical, 3))

    # Compute empirical transitions
    empirical = trans_analyzer.compute_empirical_transitions_combinatorial(trajectories)

    # Print empirical matrix
    print("\nEmpirical transition matrix (rows sum to 1):")
    print(np.round(empirical, 3))

    # Compare matrices
    comparison = trans_analyzer.compare_matrices()
    print(f"\nComparison metrics:")
    print(f"  MSE: {comparison['mse']:.4f}")
    print(f"  MAE: {comparison['mae']:.4f}")
    print(f"  Avg correlation: {comparison['avg_correlation']:.4f}")

    # Compute Markov properties
    properties = trans_analyzer.compute_markov_properties()
    if "stationary_distribution" in properties:
        print(f"\nStationary distribution (long-term visit probability):")
        for i, prob in enumerate(properties["stationary_distribution"]):
            print(f"  Cluster {i}: {prob:.3f}")

    # Visualize
    trans_analyzer.visualize_transition_graph(
        save_path=f"lunar_analysis/{analyzer.name.lower()}_transition_graph.png"
    )

    # Print summary
    trans_analyzer.print_summary()

    return trans_analyzer


class ClusterTransitionAnalyzer:
    """
    Analyze transitions between clusters using both theoretical and empirical approaches
    """

    def __init__(self, env, agent, cluster_analyzer):
        self.env = env
        self.agent = agent
        self.cluster_analyzer = cluster_analyzer
        self.clusters = cluster_analyzer.clusters
        self.state_to_cluster = cluster_analyzer.state_to_cluster

        # Storage for transition probabilities
        self.theoretical_transitions = None
        self.empirical_transitions = None
        self.hausdorff_distances = None

    def get_action_from_policy(
        self, state: np.ndarray, deterministic: bool = True
    ) -> int:
        """
        Get action from the policy network
        Works for both PPO and A2C
        """
        # Ensure state is in correct format
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # Convert to tensor
        state_tensor = th.as_tensor(state).float().to(self.agent.device)

        # Get action distribution
        with th.no_grad():
            if hasattr(self.agent.policy, "get_distribution"):
                # For policies with distribution
                dist = self.agent.policy.get_distribution(state_tensor)
                if deterministic:
                    action = dist.mode()
                else:
                    action = dist.sample()
            else:
                # For simpler policies
                action, _ = self.agent.predict(state, deterministic=deterministic)

        if isinstance(action, th.Tensor):
            action = action.cpu().numpy()

        return int(action[0] if hasattr(action, "__len__") else action)

    def get_deterministic_policy(self, state: np.ndarray) -> int:
        """Wrapper to get deterministic action"""
        return self.get_action_from_policy(state, deterministic=True)

    def compute_hausdorff_distances(self) -> np.ndarray:
        """
        Compute Hausdorff distances between all pairs of clusters
        """
        n_clusters = len(self.clusters)
        hausdorff_dist = np.zeros((n_clusters, n_clusters))

        # Get state arrays for each cluster
        cluster_states = {}
        for cluster in self.clusters:
            cluster_states[cluster.id] = cluster.states

        print("\nComputing Hausdorff distances...")
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Compute symmetric Hausdorff distance
                dist_ij = directed_hausdorff(cluster_states[i], cluster_states[j])[0]
                dist_ji = directed_hausdorff(cluster_states[j], cluster_states[i])[0]
                hausdorff_dist[i, j] = hausdorff_dist[j, i] = max(dist_ij, dist_ji)

                if (i * n_clusters + j) % 10 == 0:
                    print(f"  Progress: {i}/{n_clusters}, {j}/{n_clusters}")

        self.hausdorff_distances = hausdorff_dist
        return hausdorff_dist

    def compute_theoretical_transitions(
        self, n_samples_per_state: int = 50
    ) -> np.ndarray:
        """
        Compute theoretical transition probabilities between clusters using the policy
        P(C_i -> C_j) = sum_{s in C_i} [π(s) * P(s' in C_j | s, π(s))]
        """
        n_clusters = len(self.clusters)
        transitions = np.zeros((n_clusters, n_clusters))

        print("\nComputing theoretical transition probabilities...")
        print(f"Using {n_samples_per_state} samples per state")

        total_states = sum(c.size for c in self.clusters)
        states_processed = 0

        for i, cluster_i in enumerate(self.clusters):
            for state in cluster_i.states:
                # Get action from policy
                action = self.get_deterministic_policy(state)

                # Sample next states using environment dynamics
                next_states = []
                for _ in range(n_samples_per_state):
                    # Reset environment and set state
                    self.env.reset()
                    self.env.unwrapped.state = state.copy()
                    next_state, _, terminated, truncated, _ = self.env.step(action)
                    if not (terminated or truncated):
                        next_states.append(next_state)

                if not next_states:
                    continue

                next_states = np.array(next_states)

                # Count transitions to each cluster
                for next_state in next_states:
                    # Find closest cluster
                    min_dist = float("inf")
                    closest_cluster = None

                    for j, cluster_j in enumerate(self.clusters):
                        dist = np.linalg.norm(next_state - cluster_j.centroid)
                        if dist < min_dist:
                            min_dist = dist
                            closest_cluster = j

                    if closest_cluster is not None:
                        transitions[i, closest_cluster] += 1

                states_processed += 1
                if states_processed % 50 == 0:
                    print(f"  Processed {states_processed}/{total_states} states")

        # Normalize rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = np.divide(transitions, row_sums, where=row_sums > 0)

        self.theoretical_transitions = transitions
        return transitions

    def compute_empirical_transitions(self, trajectories: List[Dict]) -> np.ndarray:
        """
        Compute empirical transition probabilities from collected trajectories
        """
        n_clusters = len(self.clusters)
        transitions = np.zeros((n_clusters, n_clusters))
        counts = np.zeros(n_clusters)

        print("\nComputing empirical transition probabilities...")

        for traj_idx, traj in enumerate(trajectories):
            prev_cluster = None

            for i, state in enumerate(traj["states"]):
                # Find current cluster by nearest centroid
                current_cluster = self._find_cluster(state)

                if prev_cluster is not None and prev_cluster != current_cluster:
                    transitions[prev_cluster, current_cluster] += 1
                    counts[prev_cluster] += 1

                prev_cluster = current_cluster

            if (traj_idx + 1) % 50 == 0:
                print(f"  Processed {traj_idx + 1}/{len(trajectories)} trajectories")

        # Normalize
        for i in range(n_clusters):
            if counts[i] > 0:
                transitions[i, :] /= counts[i]

        self.empirical_transitions = transitions
        return transitions

    def _find_cluster(self, state: np.ndarray) -> int:
        """Find the closest cluster to a given state"""
        min_dist = float("inf")
        closest_cluster = None

        for cluster in self.clusters:
            dist = np.linalg.norm(state - cluster.centroid)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster.id

        return closest_cluster

    def compute_flow_probability(
        self, start_cluster: int, end_cluster: int, n_steps: int = 10
    ) -> Dict:
        """
        Compute probability of flowing from start to end cluster within n_steps
        """
        result = {"theoretical": 0.0, "empirical": 0.0, "step_by_step": []}

        # Theoretical flow
        if self.theoretical_transitions is not None:
            trans_matrix = self.theoretical_transitions
            current_probs = np.zeros(len(self.clusters))
            current_probs[start_cluster] = 1.0

            for step in range(n_steps):
                current_probs = current_probs @ trans_matrix
                result["step_by_step"].append(
                    {"step": step + 1, "prob": float(current_probs[end_cluster])}
                )

            result["theoretical"] = float(current_probs[end_cluster])

        # Empirical flow via Monte Carlo
        if self.empirical_transitions is not None:
            n_simulations = 1000
            reached_count = 0

            for _ in range(n_simulations):
                current = start_cluster
                for _ in range(n_steps):
                    probs = self.empirical_transitions[current]
                    if probs.sum() > 0:
                        current = np.random.choice(len(probs), p=probs)
                    if current == end_cluster:
                        reached_count += 1
                        break

            result["empirical"] = reached_count / n_simulations

        return result

    def compare_transition_matrices(self) -> Dict:
        """
        Compare theoretical and empirical transition matrices
        """
        if self.theoretical_transitions is None or self.empirical_transitions is None:
            return {}

        n_clusters = len(self.clusters)

        comparison = {
            "mse": float(
                np.mean(
                    (self.theoretical_transitions - self.empirical_transitions) ** 2
                )
            ),
            "mae": float(
                np.mean(
                    np.abs(self.theoretical_transitions - self.empirical_transitions)
                )
            ),
            "max_error": float(
                np.max(
                    np.abs(self.theoretical_transitions - self.empirical_transitions)
                )
            ),
            "kl_divergence": [],
            "correlation": [],
            "row_wise": [],
        }

        # Per-row analysis
        for i in range(n_clusters):
            p = self.theoretical_transitions[i].copy()
            q = self.empirical_transitions[i].copy()

            # Add small constant to avoid zeros
            p = p + 1e-10
            q = q + 1e-10
            p /= p.sum()
            q /= q.sum()

            # KL divergence
            kl = entropy(p, q)
            comparison["kl_divergence"].append(float(kl))

            # Correlation
            corr = np.corrcoef(p, q)[0, 1] if len(p) > 1 else 0
            comparison["correlation"].append(float(corr))

            # Row-wise metrics
            comparison["row_wise"].append(
                {
                    "cluster": i,
                    "kl": float(kl),
                    "correlation": float(corr),
                    "mse": float(np.mean((p - q) ** 2)),
                }
            )

        comparison["avg_kl"] = float(np.mean(comparison["kl_divergence"]))
        comparison["avg_correlation"] = float(np.mean(comparison["correlation"]))

        return comparison

    def analyze_distance_relationship(self) -> Dict:
        """
        Analyze relationship between Hausdorff distance and transition probability
        """
        if self.hausdorff_distances is None:
            self.compute_hausdorff_distances()

        if self.theoretical_transitions is None:
            return {}

        n_clusters = len(self.clusters)
        distances = []
        probs = []

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j and self.theoretical_transitions[i, j] > 0:
                    distances.append(self.hausdorff_distances[i, j])
                    probs.append(self.theoretical_transitions[i, j])

        if not distances:
            return {}

        distances = np.array(distances)
        probs = np.array(probs)

        # Compute correlation
        correlation = np.corrcoef(distances, probs)[0, 1]

        # Fit exponential decay model: P ~ exp(-alpha * d)
        try:
            from scipy.optimize import curve_fit

            def exp_decay(d, alpha, beta):
                return beta * np.exp(-alpha * d)

            popt, _ = curve_fit(exp_decay, distances, probs, p0=[1.0, 1.0], maxfev=5000)
            alpha, beta = popt

            # Calculate R-squared
            residuals = probs - exp_decay(distances, alpha, beta)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((probs - np.mean(probs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        except Exception as e:
            print(f"  Curve fitting failed: {e}")
            alpha, beta, r_squared = None, None, None

        return {
            "correlation": float(correlation),
            "exp_decay_alpha": float(alpha) if alpha is not None else None,
            "exp_decay_beta": float(beta) if beta is not None else None,
            "r_squared": float(r_squared) if r_squared is not None else None,
            "distances": distances.tolist(),
            "probabilities": probs.tolist(),
        }

    def visualize_comparison(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization comparing theoretical and empirical transitions
        """
        if self.theoretical_transitions is None or self.empirical_transitions is None:
            print("Need both theoretical and empirical transitions computed first")
            return

        n_clusters = len(self.clusters)

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))

        # Plot 1: Theoretical transition matrix
        ax = axes[0, 0]
        im1 = ax.imshow(
            self.theoretical_transitions, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1
        )
        ax.set_title(
            "Theoretical Transition Probabilities\n(Policy + Dynamics)", fontsize=12
        )
        ax.set_xlabel("To Cluster")
        ax.set_ylabel("From Cluster")
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations for significant probabilities
        for i in range(n_clusters):
            for j in range(n_clusters):
                if self.theoretical_transitions[i, j] > 0.05:
                    ax.text(
                        j,
                        i,
                        f"{self.theoretical_transitions[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if self.theoretical_transitions[i, j] > 0.5
                            else "black"
                        ),
                        fontsize=8,
                    )

        # Plot 2: Empirical transition matrix
        ax = axes[0, 1]
        im2 = ax.imshow(
            self.empirical_transitions, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1
        )
        ax.set_title(
            "Empirical Transition Probabilities\n(From Trajectories)", fontsize=12
        )
        ax.set_xlabel("To Cluster")
        ax.set_ylabel("From Cluster")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

        for i in range(n_clusters):
            for j in range(n_clusters):
                if self.empirical_transitions[i, j] > 0.05:
                    ax.text(
                        j,
                        i,
                        f"{self.empirical_transitions[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if self.empirical_transitions[i, j] > 0.5
                            else "black"
                        ),
                        fontsize=8,
                    )

        # Plot 3: Difference matrix
        ax = axes[0, 2]
        diff = self.theoretical_transitions - self.empirical_transitions
        im3 = ax.imshow(diff, cmap="RdBu", aspect="auto", vmin=-0.5, vmax=0.5)
        ax.set_title("Theoretical - Empirical Difference", fontsize=12)
        ax.set_xlabel("To Cluster")
        ax.set_ylabel("From Cluster")
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

        for i in range(n_clusters):
            for j in range(n_clusters):
                if abs(diff[i, j]) > 0.05:
                    ax.text(
                        j,
                        i,
                        f"{diff[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if abs(diff[i, j]) > 0.25 else "black",
                        fontsize=8,
                    )

        # Plot 4: Hausdorff distance matrix
        ax = axes[1, 0]
        if self.hausdorff_distances is not None:
            im4 = ax.imshow(self.hausdorff_distances, cmap="viridis", aspect="auto")
            ax.set_title("Hausdorff Distances Between Clusters", fontsize=12)
            ax.set_xlabel("To Cluster")
            ax.set_ylabel("From Cluster")
            plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

            for i in range(n_clusters):
                for j in range(n_clusters):
                    if i != j and self.hausdorff_distances[i, j] > 0:
                        ax.text(
                            j,
                            i,
                            f"{self.hausdorff_distances[i, j]:.1f}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=7,
                        )

        # Plot 5: Transition probability vs Hausdorff distance
        ax = axes[1, 1]
        distance_data = self.analyze_distance_relationship()

        if "distances" in distance_data and distance_data["distances"]:
            ax.scatter(
                distance_data["distances"],
                distance_data["probabilities"],
                alpha=0.6,
                s=50,
                label="Cluster pairs",
            )

            # Add trend line if exponential fit worked
            if distance_data["exp_decay_alpha"] is not None:
                d_range = np.linspace(
                    min(distance_data["distances"]),
                    max(distance_data["distances"]),
                    100,
                )
                p_pred = distance_data["exp_decay_beta"] * np.exp(
                    -distance_data["exp_decay_alpha"] * d_range
                )
                ax.plot(
                    d_range,
                    p_pred,
                    "r--",
                    linewidth=2,
                    label=f"Exp fit: α={distance_data['exp_decay_alpha']:.2f}\nR²={distance_data['r_squared']:.3f}",
                )

            ax.set_xlabel("Hausdorff Distance")
            ax.set_ylabel("Transition Probability")
            ax.set_title(
                f"Distance-Probability Relationship\nCorrelation: {distance_data['correlation']:.3f}"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 6: KL divergence per cluster
        ax = axes[1, 2]
        comparison = self.compare_transition_matrices()

        if "kl_divergence" in comparison and comparison["kl_divergence"]:
            clusters = range(len(comparison["kl_divergence"]))
            bars = ax.bar(clusters, comparison["kl_divergence"], alpha=0.7)

            # Color bars by divergence magnitude
            for bar, kl in zip(bars, comparison["kl_divergence"]):
                if kl > comparison["avg_kl"]:
                    bar.set_color("red")
                else:
                    bar.set_color("blue")

            ax.axhline(
                y=comparison["avg_kl"],
                color="black",
                linestyle="--",
                label=f"Avg: {comparison['avg_kl']:.3f}",
            )
            ax.set_xlabel("Cluster")
            ax.set_ylabel("KL Divergence")
            ax.set_title("KL Divergence: Theoretical vs Empirical")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Cluster Transition Analysis - {self.cluster_analyzer.name}",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Comparison visualization saved to {save_path}")

        plt.show()

    def compute_path_probabilities(self, path: List[int]) -> Dict:
        """
        Compute probability of following a specific path through clusters
        """
        result = {"theoretical": 1.0, "empirical": 1.0, "segment_probs": []}

        for i in range(len(path) - 1):
            from_c, to_c = path[i], path[i + 1]

            segment = {
                "from": from_c,
                "to": to_c,
                "theoretical": (
                    self.theoretical_transitions[from_c, to_c]
                    if self.theoretical_transitions is not None
                    else None
                ),
                "empirical": (
                    self.empirical_transitions[from_c, to_c]
                    if self.empirical_transitions is not None
                    else None
                ),
            }

            if segment["theoretical"] is not None:
                result["theoretical"] *= segment["theoretical"]
            if segment["empirical"] is not None:
                result["empirical"] *= segment["empirical"]

            result["segment_probs"].append(segment)

        return result

    def print_summary(self):
        """Print a summary of the transition analysis"""
        print("\n" + "=" * 60)
        print(f"TRANSITION ANALYSIS SUMMARY - {self.cluster_analyzer.name}")
        print("=" * 60)

        if self.theoretical_transitions is not None:
            print(f"\nTheoretical transitions:")
            print(
                f"  Average non-zero probability: {np.mean(self.theoretical_transitions[self.theoretical_transitions > 0]):.3f}"
            )
            print(f"  Max probability: {np.max(self.theoretical_transitions):.3f}")
            print(
                f"  Number of possible transitions: {np.sum(self.theoretical_transitions > 0)}"
            )

        if self.empirical_transitions is not None:
            print(f"\nEmpirical transitions:")
            print(
                f"  Average non-zero probability: {np.mean(self.empirical_transitions[self.empirical_transitions > 0]):.3f}"
            )
            print(f"  Max probability: {np.max(self.empirical_transitions):.3f}")
            print(
                f"  Number of observed transitions: {np.sum(self.empirical_transitions > 0)}"
            )

        comparison = self.compare_transition_matrices()
        if comparison:
            print(f"\nComparison metrics:")
            print(f"  MSE: {comparison['mse']:.4f}")
            print(f"  MAE: {comparison['mae']:.4f}")
            print(f"  Avg KL Divergence: {comparison['avg_kl']:.4f}")
            print(f"  Avg Correlation: {comparison['avg_correlation']:.4f}")

            print(f"\nPer-cluster KL divergence:")
            for i, kl in enumerate(comparison["kl_divergence"]):
                print(f"  Cluster {i}: {kl:.4f}")

        dist_rel = self.analyze_distance_relationship()
        if dist_rel:
            print(f"\nDistance-probability relationship:")
            print(f"  Correlation: {dist_rel['correlation']:.4f}")
            if dist_rel["exp_decay_alpha"]:
                print(
                    f"  Exponential decay: P = {dist_rel['exp_decay_beta']:.2f} * exp(-{dist_rel['exp_decay_alpha']:.2f} * d)"
                )
                print(f"  R²: {dist_rel['r_squared']:.4f}")


def analyze_cluster_transitions(agent, analyzer, trajectories, env):
    """
    Comprehensive analysis of cluster transitions
    """
    print("\n" + "=" * 80)
    print(f"CLUSTER TRANSITION ANALYSIS - {analyzer.name}")
    print("=" * 80)

    # Initialize transition analyzer
    trans_analyzer = ClusterTransitionAnalyzer(env, agent, analyzer)

    # Compute Hausdorff distances
    print("\n1. Computing Hausdorff distances between clusters...")
    hausdorff = trans_analyzer.compute_hausdorff_distances()
    print(f"   Average Hausdorff distance: {np.mean(hausdorff[hausdorff > 0]):.3f}")

    # Compute theoretical transitions
    print("\n2. Computing theoretical transition probabilities...")
    theoretical = trans_analyzer.compute_theoretical_transitions(n_samples_per_state=30)

    # Compute empirical transitions
    print("\n3. Computing empirical transition probabilities...")
    empirical = trans_analyzer.compute_empirical_transitions(trajectories)

    # Print summary
    trans_analyzer.print_summary()

    # Visualize
    trans_analyzer.visualize_comparison(
        save_path=f"lunar_analysis/{analyzer.name.lower()}_transition_comparison.png"
    )

    return trans_analyzer


# Add this to your LunarLanderClusterAnalyzer class to ensure it has the needed attributes
def enhance_lunar_analyzer(analyzer):
    """
    Add any missing attributes to the analyzer
    """
    if not hasattr(analyzer, "name"):
        analyzer.name = "Policy"
    return analyzer


# Update your main comparison function
def run_enhanced_lunar_comparison():
    """
    Run comparison with transition analysis
    """
    from stable_baselines3 import PPO, A2C

    print("\n" + "=" * 80)
    print("ENHANCED LUNAR LANDER COMPARISON WITH TRANSITION ANALYSIS")
    print("=" * 80)

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("lunar_analysis", exist_ok=True)

    # Load or train your models
    # For this example, we'll assume models are already trained
    # You can modify this to train new models

    try:
        # Try to load existing models
        ppo_model = PPO.load("models/ppo_lunar_lander_v3")
        a2c_model = A2C.load("models/a2c_lunar_lander_v3")
        print("Loaded existing models")
    except:
        print("Please train models first or provide correct paths")
        return

    # Create evaluation environments
    eval_env_ppo = create_lunar_lander_env()
    eval_env_a2c = create_lunar_lander_env()

    # Create analyzers (assuming you have these from previous steps)
    # You'll need to run your clustering analysis first
    # This is just a template - adapt to your actual analyzer creation

    # Analyze transitions for both algorithms
    results = {}

    for name, model, env in [
        ("PPO", ppo_model, eval_env_ppo),
        ("A2C", a2c_model, eval_env_a2c),
    ]:
        # Create analyzer (you'll need to adapt this to your actual analyzer creation)
        analyzer = LunarLanderClusterAnalyzer(model, env, name)

        # Collect trajectories if not already done
        if (
            not hasattr(analyzer, "collected_trajectories")
            or not analyzer.collected_trajectories
        ):
            trajectories = analyzer.collect_trajectories(n_episodes=200)
        else:
            trajectories = analyzer.collected_trajectories

        # Ensure clusters are computed
        if not analyzer.clusters:
            analyzer.cluster_by_action_preference(
                n_clusters=12, method="kmeans", use_pca=True
            )

        # Analyze transitions
        trans_analyzer = analyze_cluster_transitions(model, analyzer, trajectories, env)
        results[name.lower()] = trans_analyzer

    return results
