import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, TD3, PPO
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import wasserstein_distance
import networkx as nx
from collections import defaultdict
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class ValueFunctionReebGraph:
    """
    Construct and visualize the Reeb graph of a value function over the state space.
    """

    def __init__(self, env, policy, n_samples=300):
        self.env = env
        self.policy = policy
        self.n_samples = n_samples
        self.state_dim = env.observation_space.shape[0]

        # Get state bounds
        self.state_low = env.observation_space.low
        self.state_high = env.observation_space.high

        # Replace infinities with reasonable bounds
        self.state_low = np.where(np.isfinite(self.state_low), self.state_low, -10)
        self.state_high = np.where(np.isfinite(self.state_high), self.state_high, 10)

    def _estimate_value_function(self, states):
        """Estimate value of states using the policy."""
        values = []
        states = np.array(states).reshape(-1, self.state_dim)

        for state in states:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                # Get action from policy
                if hasattr(self.policy, "predict"):
                    action, _ = self.policy.predict(state, deterministic=True)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0)
                else:
                    action_tensor = self.policy(state_tensor)
                    action = action_tensor.numpy()[0]

                value = None

                # Try different methods to get value
                if hasattr(self.policy, "critic"):
                    try:
                        if isinstance(self.policy.critic, tuple) or isinstance(
                            self.policy.critic, list
                        ):
                            values_list = []
                            for critic in self.policy.critic:
                                v = critic(state_tensor, action_tensor).item()
                                values_list.append(v)
                            value = np.min(values_list)
                        else:
                            value = self.policy.critic(
                                state_tensor, action_tensor
                            ).item()
                    except:
                        pass

                if value is None and hasattr(self.policy, "value_net"):
                    try:
                        value = self.policy.value_net(state_tensor).item()
                    except:
                        pass

                if value is None:
                    value = -np.linalg.norm(action)

                values.append(float(value))

        return np.array(values)

    def sample_state_space(self, n_samples=None):
        """Uniformly sample states from the environment's state space."""
        if n_samples is None:
            n_samples = self.n_samples

        samples = []
        for _ in range(n_samples):
            state = np.random.uniform(self.state_low, self.state_high)
            samples.append(state)
        return np.array(samples)

    def compute_reeb_graph(self, n_levels=10, connectivity_radius=None):
        """Compute the Reeb graph of the value function."""
        # Sample state space
        states = self.sample_state_space()

        # Compute values
        print("Estimating values for sampled states...")
        values = self._estimate_value_function(states)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

        # Normalize values
        v_min, v_max = values.min(), values.max()
        if v_max - v_min < 1e-6:
            v_max = v_min + 1.0
        values_norm = (values - v_min) / (v_max - v_min)
        values_norm = np.clip(values_norm, 0, 1)

        # Create level sets
        level_boundaries = np.linspace(0, 1, n_levels + 1)
        level_sets = []

        for i in range(n_levels):
            mask = (values_norm >= level_boundaries[i]) & (
                values_norm < level_boundaries[i + 1]
            )
            level_states = states[mask]
            level_values = values[mask]

            if len(level_states) > 0:
                level_sets.append(
                    {
                        "states": level_states,
                        "values": level_values,
                        "level": i,
                        "value_range": (
                            level_boundaries[i] * (v_max - v_min) + v_min,
                            level_boundaries[i + 1] * (v_max - v_min) + v_min,
                        ),
                    }
                )

        # Estimate connectivity radius
        if connectivity_radius is None:
            if len(states) > 1:
                state_span = np.max(states, axis=0) - np.min(states, axis=0)
                state_span = np.where(state_span > 0, state_span, 1.0)
                connectivity_radius = 0.1 * np.mean(state_span)
            else:
                connectivity_radius = 0.1

        # Build graph nodes
        nodes = []
        node_id = 0

        for level_idx, level_data in enumerate(level_sets):
            if len(level_data["states"]) < 2:
                if len(level_data["states"]) == 1:
                    nodes.append(
                        {
                            "id": node_id,
                            "level": level_idx,
                            "centroid": level_data["states"][0],
                            "mean_value": level_data["values"][0],
                            "size": 1,
                            "states": level_data["states"],
                        }
                    )
                    node_id += 1
                continue

            # Cluster states
            try:
                dist_matrix = pdist(level_data["states"])
                linkage_matrix = linkage(dist_matrix, method="single")
                max_dist = connectivity_radius * np.sqrt(self.state_dim)
                clusters = fcluster(linkage_matrix, max_dist, criterion="distance")

                unique_clusters = np.unique(clusters)
                for cluster_id in unique_clusters:
                    cluster_mask = clusters == cluster_id
                    comp_states = level_data["states"][cluster_mask]
                    comp_vals = level_data["values"][cluster_mask]

                    if len(comp_states) > 0:
                        centroid = np.mean(comp_states, axis=0)
                        mean_value = np.mean(comp_vals)

                        nodes.append(
                            {
                                "id": node_id,
                                "level": level_idx,
                                "centroid": centroid,
                                "mean_value": mean_value,
                                "size": len(comp_states),
                                "states": comp_states,
                            }
                        )
                        node_id += 1
            except:
                # Fallback: treat all as one component
                centroid = np.mean(level_data["states"], axis=0)
                mean_value = np.mean(level_data["values"])
                nodes.append(
                    {
                        "id": node_id,
                        "level": level_idx,
                        "centroid": centroid,
                        "mean_value": mean_value,
                        "size": len(level_data["states"]),
                        "states": level_data["states"],
                    }
                )
                node_id += 1

        # Build graph
        G = nx.Graph()
        for node in nodes:
            G.add_node(
                node["id"],
                level=node["level"],
                centroid=node["centroid"],
                value=node["mean_value"],
                size=node["size"],
            )

        # Add edges between adjacent levels
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if node2["level"] == node1["level"] + 1:
                    dist = np.linalg.norm(node1["centroid"] - node2["centroid"])
                    if dist < 2 * connectivity_radius:
                        G.add_edge(node1["id"], node2["id"])

        return G, {"nodes": nodes, "values": values, "states": states}


class TransitionBasedPartitionComparison:
    """
    Compare Reeb graph partitions using metrics based on transition probabilities.
    """

    def __init__(self, env, policy, reeb_graph, node_data):
        self.env = env
        self.policy = policy
        self.reeb_graph = reeb_graph
        self.node_data = node_data

        # Get list of node IDs from the graph
        self.node_list = list(reeb_graph.nodes())
        self.n_partitions = len(self.node_list)

        print(f"Initialized with {self.n_partitions} partitions")

        # Store node centroids and values
        self.node_centroids = []
        self.node_values = []

        for node_id in self.node_list:
            self.node_centroids.append(reeb_graph.nodes[node_id]["centroid"])
            self.node_values.append(reeb_graph.nodes[node_id]["value"])

        self.node_centroids = np.array(self.node_centroids)
        self.node_values = np.array(self.node_values)

        # Extract partition labels for sampled states
        self.partition_labels = self._extract_partition_labels()

    def _extract_partition_labels(self):
        """Extract partition labels from Reeb graph nodes."""
        labels = -np.ones(len(self.node_data["states"]))

        if len(self.node_list) == 0:
            return labels

        for i, state in enumerate(self.node_data["states"]):
            # Find closest node centroid
            distances = np.linalg.norm(self.node_centroids - state, axis=1)
            closest_idx = np.argmin(distances)
            labels[i] = self.node_list[closest_idx]

        return labels

    def _set_env_state(self, env, state):
        """Safely set the environment state based on environment type."""
        env_id = env.spec.id

        if "Pendulum" in env_id:
            # Pendulum state is [cos(theta), sin(theta), thetadot]
            # But the internal state is (theta, thetadot)
            # Convert from observation to internal state
            cos_th, sin_th, thdot = state
            theta = np.arctan2(sin_th, cos_th)
            env.unwrapped.state = np.array([theta, thdot])
        elif "MountainCar" in env_id:
            # MountainCar state is [position, velocity]
            if hasattr(env.unwrapped, "state"):
                env.unwrapped.state = state.copy()
        else:
            # Generic attempt
            try:
                if hasattr(env.unwrapped, "state"):
                    env.unwrapped.state = state.copy()
            except:
                pass

        # Return the observation after setting state
        return env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else state

    def estimate_transition_kernel(self, n_samples_per_partition=20, horizon=1):
        """
        Estimate transition probabilities between partitions.
        """
        # Group states by partition
        partition_states = defaultdict(list)
        for i, state in enumerate(self.node_data["states"]):
            partition = self.partition_labels[i]
            if partition >= 0:
                partition_states[partition].append(state)

        # Initialize transition count matrix
        transition_counts = np.zeros((self.n_partitions, self.n_partitions))

        # Create a mapping from node_id to matrix index
        node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_list)}

        # For each partition, sample states and simulate transitions
        for p1 in self.node_list:
            if len(partition_states[p1]) == 0:
                continue

            # Sample states from this partition
            states = np.array(partition_states[p1])
            n_samples = min(n_samples_per_partition, len(states))
            if n_samples > 0:
                idx = np.random.choice(len(states), n_samples, replace=False)
                states = states[idx]

            for state in states:
                # Create environment copy
                env_copy = gym.make(self.env.spec.id)
                env_copy.reset()

                # Set the state appropriately for this environment
                obs = self._set_env_state(env_copy, state)

                # Run trajectory
                for step in range(horizon):
                    with torch.no_grad():
                        action, _ = self.policy.predict(obs, deterministic=True)

                    # Ensure action has correct shape
                    if not isinstance(action, np.ndarray):
                        action = np.array([action])

                    obs, reward, terminated, truncated, info = env_copy.step(action)

                    if terminated or truncated:
                        break

                # Find which partition the final state belongs to
                distances = np.linalg.norm(self.node_centroids - obs.flatten(), axis=1)
                p2 = self.node_list[np.argmin(distances)]

                # Update transition count
                i = node_to_idx[p1]
                j = node_to_idx[p2]
                transition_counts[i, j] += 1

                env_copy.close()

        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_counts / row_sums

        return transition_matrix, transition_counts

    def compute_partition_metrics(self):
        """
        Compute various metrics to compare partitions based on transition probabilities.
        """
        metrics = {}

        # Estimate transition kernel
        print("  Estimating transition kernel...")
        trans_matrix, trans_counts = self.estimate_transition_kernel(horizon=1)
        metrics["transition_matrix"] = trans_matrix
        metrics["transition_counts"] = trans_counts

        # 1. Transition entropy for each partition
        with np.errstate(divide="ignore", invalid="ignore"):
            trans_entropy = -np.sum(
                trans_matrix * np.log2(trans_matrix + 1e-10), axis=1
            )
            trans_entropy = np.nan_to_num(trans_entropy)
        metrics["partition_entropy"] = trans_entropy
        metrics["mean_transition_entropy"] = float(np.mean(trans_entropy))

        # 2. Wasserstein distance between transition distributions
        print("  Computing Wasserstein distances...")
        wasserstein_dists = np.zeros((self.n_partitions, self.n_partitions))
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                wasserstein_dists[i, j] = wasserstein_distance(
                    trans_matrix[i], trans_matrix[j]
                )
        metrics["wasserstein_distances"] = wasserstein_dists
        metrics["mean_wasserstein"] = float(np.mean(wasserstein_dists))

        # 3. Transition matrix spectral properties
        print("  Computing spectral properties...")
        try:
            eigenvals = np.linalg.eigvals(trans_matrix)
            # Sort eigenvalues by magnitude
            eigenvals = sorted(eigenvals, key=lambda x: abs(x), reverse=True)
            if len(eigenvals) > 1:
                metrics["spectral_gap"] = float(1 - abs(eigenvals[1]))
                if abs(eigenvals[1]) < 1 and abs(eigenvals[1]) > 0:
                    metrics["mixing_time"] = float(-1 / np.log(abs(eigenvals[1])))
                else:
                    metrics["mixing_time"] = np.inf
            else:
                metrics["spectral_gap"] = 0.0
                metrics["mixing_time"] = np.inf
        except Exception as e:
            print(f"  Spectral computation error: {e}")
            metrics["spectral_gap"] = 0.0
            metrics["mixing_time"] = np.inf

        # 4. Stationary distribution
        print("  Computing stationary distribution...")
        try:
            eigenvals, eigenvecs = np.linalg.eig(trans_matrix.T)
            # Find eigenvector corresponding to eigenvalue closest to 1
            idx = np.argmin(np.abs(eigenvals - 1))
            stationary = np.real(eigenvecs[:, idx])
            stationary = stationary / np.sum(stationary)
            metrics["stationary_distribution"] = stationary
        except:
            metrics["stationary_distribution"] = (
                np.ones(self.n_partitions) / self.n_partitions
            )

        # 5. KL divergence
        print("  Computing KL divergences...")
        kl_divergences = np.zeros((self.n_partitions, self.n_partitions))
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                p = trans_matrix[i] + 1e-10
                q = trans_matrix[j] + 1e-10
                kl_divergences[i, j] = float(np.sum(p * np.log2(p / q)))
        metrics["kl_divergences"] = kl_divergences
        metrics["mean_kl"] = float(np.mean(kl_divergences))

        return metrics

    def visualize_partition_comparison(self, metrics):
        """
        Create comprehensive visualization of partition comparisons.
        """
        fig = plt.figure(figsize=(20, 12))

        trans_matrix = metrics["transition_matrix"]

        # 1. Transition matrix heatmap
        ax1 = fig.add_subplot(231)
        sns.heatmap(
            trans_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax1,
            cbar_kws={"label": "Probability"},
        )
        ax1.set_xlabel("To Partition")
        ax1.set_ylabel("From Partition")
        ax1.set_title("Transition Probability Matrix")

        # 2. Partition entropy
        ax2 = fig.add_subplot(232)
        entropy = metrics["partition_entropy"]
        bars = ax2.bar(range(self.n_partitions), entropy)
        ax2.set_xlabel("Partition")
        ax2.set_ylabel("Transition Entropy (bits)")
        ax2.set_title("Partition Transition Entropy")
        ax2.grid(True, alpha=0.3)

        # Color bars by value
        if len(entropy) > 0 and max(entropy) > 0:
            for bar, e in zip(bars, entropy):
                bar.set_color(plt.cm.viridis(e / max(entropy)))

        # 3. Wasserstein distance matrix
        ax3 = fig.add_subplot(233)
        sns.heatmap(
            metrics["wasserstein_distances"],
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax3,
            cbar_kws={"label": "Wasserstein Distance"},
        )
        ax3.set_xlabel("Partition")
        ax3.set_ylabel("Partition")
        ax3.set_title("Wasserstein Distance Between\nTransition Distributions")

        # 4. KL divergence matrix
        ax4 = fig.add_subplot(234)
        sns.heatmap(
            metrics["kl_divergences"],
            annot=True,
            fmt=".1f",
            cmap="viridis",
            ax=ax4,
            cbar_kws={"label": "KL Divergence (bits)"},
        )
        ax4.set_xlabel("Partition")
        ax4.set_ylabel("Partition")
        ax4.set_title("KL Divergence Between\nTransition Distributions")

        # 5. Stationary distribution
        ax5 = fig.add_subplot(235)
        stationary = metrics["stationary_distribution"]
        ax5.bar(range(self.n_partitions), stationary)
        ax5.set_xlabel("Partition")
        ax5.set_ylabel("Probability")
        ax5.set_title("Stationary Distribution")
        ax5.grid(True, alpha=0.3)

        # 6. Summary statistics
        ax6 = fig.add_subplot(236)
        ax6.axis("off")

        summary = f"""Transition-Based Metrics Summary:

Number of Partitions: {self.n_partitions}

Mean Transition Entropy: {metrics["mean_transition_entropy"]:.3f} bits
Mean Wasserstein Distance: {metrics["mean_wasserstein"]:.3f}
Mean KL Divergence: {metrics["mean_kl"]:.3f} bits

Spectral Gap: {metrics["spectral_gap"]:.3f}
Mixing Time: {metrics["mixing_time"]:.1f} steps

Top Transitions:
"""
        # Find top 3 transitions
        flat_indices = np.argsort(trans_matrix.flatten())[-3:][::-1]
        for idx in flat_indices:
            i, j = np.unravel_index(idx, trans_matrix.shape)
            if trans_matrix[i, j] > 0.1:
                summary += f"  {i} → {j}: {trans_matrix[i, j]:.2f}\n"

        ax6.text(
            0.1,
            0.5,
            summary,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        plt.suptitle(
            "Partition Comparison Based on Transition Probabilities", fontsize=16
        )
        plt.tight_layout()
        plt.savefig("partition_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

        return trans_matrix


def analyze_pendulum_reeb_partitions():
    """Analyze the Pendulum-v1 environment with SAC using Reeb graph partitions."""

    print("=" * 60)
    print("Analyzing Pendulum-v1 with SAC - Reeb Graph Partition Comparison")
    print("=" * 60)

    # Train SAC on Pendulum
    env = gym.make("Pendulum-v1")
    env.reset()
    print("Training SAC on Pendulum-v1...")
    model = SAC("MlpPolicy", env, verbose=0, learning_starts=1000)
    model.load("sac_pendulum")
    print("loaded pre-trained sac on pendulum")
    # model.learn(total_timesteps=30000)

    # Create Reeb graph
    print("\nCreating Reeb graph...")
    reeb = ValueFunctionReebGraph(env, model.policy, n_samples=200)
    graph, node_data = reeb.compute_reeb_graph(n_levels=6)

    print(f"Reeb graph has {graph.number_of_nodes()} nodes")

    # Create partition comparator
    comparator = TransitionBasedPartitionComparison(env, model.policy, graph, node_data)

    # Compute metrics
    print("\nComputing transition-based metrics...")
    metrics = comparator.compute_partition_metrics()

    # Visualize results
    print("\nGenerating visualizations...")
    comparator.visualize_partition_comparison(metrics)

    # Print metrics
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Mean Transition Entropy: {metrics['mean_transition_entropy']:.3f} bits")
    print(f"Mean Wasserstein Distance: {metrics['mean_wasserstein']:.3f}")
    print(f"Mean KL Divergence: {metrics['mean_kl']:.3f} bits")
    print(f"Spectral Gap: {metrics['spectral_gap']:.3f}")
    print(f"Mixing Time: {metrics['mixing_time']:.1f} steps")

    # Analyze relationship with value
    print("\n" + "=" * 60)
    print("Analyzing relationship between value and transition dynamics")
    print("=" * 60)

    # Get partition values
    partition_values = []
    for node_id in comparator.node_list:
        partition_values.append(graph.nodes[node_id]["value"])

    # Correlate transition entropy with value
    entropy = metrics["partition_entropy"]

    if len(partition_values) > 1 and len(entropy) > 1:
        correlation = np.corrcoef(partition_values, entropy)[0, 1]
        print(
            f"\nCorrelation between partition value and transition entropy: {correlation:.3f}"
        )

        if correlation > 0.5:
            print("→ High-value regions tend to have more unpredictable transitions")
        elif correlation < -0.5:
            print("→ High-value regions tend to have more predictable transitions")
        else:
            print(
                "→ No strong relationship between value and transition predictability"
            )

    # Analyze transition patterns
    print("\nTransition Pattern Analysis:")
    print("-" * 40)

    trans_matrix = metrics["transition_matrix"]

    # Find self-loop probabilities
    self_loops = np.diag(trans_matrix)
    for i, (p, val) in enumerate(zip(self_loops, partition_values)):
        print(f"Partition {i}: Self-loop = {p:.3f}, Value = {val:.2f}")

    # Find absorbing partitions
    absorbing = np.where(self_loops > 0.8)[0]
    if len(absorbing) > 0:
        print(f"\nPotential absorbing partitions: {absorbing}")

    env.close()
    return comparator, metrics, graph, node_data


# Run the analysis
if __name__ == "__main__":
    comparator, metrics, graph, node_data = analyze_pendulum_reeb_partitions()

    # Additional analysis: Compare with random baseline
    print("\n" + "=" * 60)
    print("Comparison with Random Policy Baseline")
    print("=" * 60)

    env = gym.make("Pendulum-v1")

    # Create a random policy
    class RandomPolicy:
        def predict(self, obs, deterministic=True):
            return np.array([np.random.uniform(-2, 2)]), None

    random_policy = RandomPolicy()

    # Create random comparator using same graph structure
    random_comparator = TransitionBasedPartitionComparison(
        env, random_policy, graph, node_data
    )
    random_metrics = random_comparator.compute_partition_metrics()

    print(
        f"\nTrained Policy - Mean Transition Entropy: {metrics['mean_transition_entropy']:.3f}"
    )
    print(
        f"Random Policy - Mean Transition Entropy: {random_metrics['mean_transition_entropy']:.3f}"
    )

    if metrics["mean_transition_entropy"] < random_metrics["mean_transition_entropy"]:
        print("→ Trained policy makes transitions MORE predictable")
    else:
        print("→ Trained policy makes transitions LESS predictable")

    env.close()
