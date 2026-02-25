import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import pickle
import os
import torch as th
from stable_baselines3 import PPO, A2C
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class LunarLanderClusterRefinement:
    """
    Find coarsest common refinement of two clusterings from PPO and A2C
    and use Bellman optimality to form a superior policy
    """

    def __init__(
        self,
        ppo_model_path: str,
        a2c_model_path: str,
        seed: int = 42,
        n_clusters: int = 12,
    ):
        print("\n" + "=" * 80)
        print("LUNAR LANDER CLUSTER REFINEMENT ANALYSIS")
        print("=" * 80)

        self.seed = seed
        np.random.seed(seed)

        # Create environment
        self.env = gym.make("LunarLander-v3")
        self.env.reset(seed=seed)

        # Load models
        print(f"\nLoading models...")
        self.ppo_model = PPO.load(ppo_model_path)
        self.a2c_model = A2C.load(a2c_model_path)

        # Environment specs
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.n_clusters_target = n_clusters

        # Action names
        self.action_names = {
            0: "Do nothing",
            1: "Fire left engine",
            2: "Fire main engine",
            3: "Fire right engine",
        }

        # Storage
        self.ppo_clustering = None
        self.a2c_clustering = None
        self.refinement = None
        self.q_refinement = None
        self.superior_policy_fn = None

        # For visualization
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def collect_trajectories(
        self, model, n_episodes: int = 200, deterministic: bool = True
    ) -> Dict:
        """
        Collect trajectories and extract states, actions, and values
        """
        print(f"  Collecting {n_episodes} trajectories...")

        states = []
        actions = []
        values = []
        trajectories = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            traj_states = []
            traj_actions = []

            while not done:
                action, _ = model.predict(state, deterministic=deterministic)

                # Get state value
                state_tensor = th.as_tensor(state).float().to(model.device)
                with th.no_grad():
                    if hasattr(model.policy, "predict_values"):
                        value = (
                            model.policy.predict_values(state_tensor.unsqueeze(0))
                            .cpu()
                            .numpy()[0, 0]
                        )
                    else:
                        value = 0

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state.copy())
                actions.append(int(action))
                values.append(float(value))
                traj_states.append(state.copy())
                traj_actions.append(int(action))

                state = next_state

            trajectories.append(
                {
                    "states": traj_states,
                    "actions": traj_actions,
                    "success": reward > 200,  # Successful landing
                }
            )

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "values": np.array(values),
            "trajectories": trajectories,
        }

    def cluster_states_by_action(self, data: Dict, name: str) -> Dict:
        """
        Cluster states based on their features and action preferences
        Returns clustering similar to FrozenLake format
        """
        print(f"\nClustering {name} states...")

        states = data["states"]
        actions = data["actions"]

        # Standardize features
        states_scaled = self.scaler.fit_transform(states)

        # PCA for clustering (reduce dimensionality)
        states_pca = PCA(n_components=min(4, self.state_dim)).fit_transform(
            states_scaled
        )

        # Cluster using KMeans
        kmeans = KMeans(
            n_clusters=self.n_clusters_target, random_state=self.seed, n_init=10
        )
        cluster_labels = kmeans.fit_predict(states_pca)

        # For each cluster, determine dominant action
        clusters = []
        state_to_cluster = {}

        for cid in range(self.n_clusters_target):
            mask = cluster_labels == cid
            if not np.any(mask):
                continue

            cluster_states = states[mask]
            cluster_actions = actions[mask]

            # Find dominant action
            unique_actions, counts = np.unique(cluster_actions, return_counts=True)
            dominant_action = int(unique_actions[np.argmax(counts)])

            # Find centroid in original space
            centroid = np.mean(cluster_states, axis=0)

            clusters.append(
                {
                    "id": cid,
                    "action": dominant_action,
                    "states": [
                        i for i, label in enumerate(cluster_labels) if label == cid
                    ],
                    "state_vectors": cluster_states,
                    "centroid": centroid,
                    "size": len(cluster_states),
                    "action_distribution": {
                        int(a): float(c / len(cluster_states))
                        for a, c in zip(unique_actions, counts)
                    },
                }
            )

            # Map original data indices to cluster
            indices = np.where(mask)[0]
            for idx in indices:
                state_to_cluster[idx] = cid

        print(f"  Created {len(clusters)} clusters")
        print(f"  Avg cluster size: {np.mean([c['size'] for c in clusters]):.1f}")

        return {
            "name": name,
            "clusters": clusters,
            "state_to_cluster": state_to_cluster,
            "n_clusters": len(clusters),
            "data": data,
            "cluster_labels": cluster_labels,
            "kmeans": kmeans,
            "pca": PCA(n_components=2),  # Store for visualization
        }

    def compute_coarsest_common_refinement(self):
        """
        Find the coarsest clustering that refines both PPO and A2C clusterings
        Two data points are in the same refined cluster iff they are in the same cluster
        in both original clusterings
        """
        if self.ppo_clustering is None or self.a2c_clustering is None:
            raise ValueError("Both clusterings must be computed first")

        print("\n" + "=" * 60)
        print("COMPUTING COARSEST COMMON REFINEMENT")
        print("=" * 60)

        # We need to align the data points - they may be different sizes
        # Use a subset of points that exist in both datasets
        # For simplicity, we'll use the cluster assignments from each model's own data
        # and create a mapping based on state similarity

        print("\nAligning clusterings...")

        # Get all states from both datasets
        ppo_states = self.ppo_clustering["data"]["states"]
        a2c_states = self.a2c_clustering["data"]["states"]

        # For each PPO state, find closest A2C state
        from scipy.spatial import cKDTree

        a2c_tree = cKDTree(a2c_states)

        # Create mapping from PPO data indices to A2C cluster assignments
        ppo_to_a2c_cluster = {}

        for i, ppo_state in enumerate(ppo_states):
            # Find closest A2C state
            dist, idx = a2c_tree.query(ppo_state)
            if dist < 1.0:  # Threshold for considering them the "same" state
                a2c_cluster = self.a2c_clustering["cluster_labels"][idx]
                ppo_to_a2c_cluster[i] = a2c_cluster

        print(
            f"  Matched {len(ppo_to_a2c_cluster)}/{len(ppo_states)} PPO states to A2C clusters"
        )

        # Create equivalence classes based on (ppo_cluster, a2c_cluster) pairs
        pair_to_indices = defaultdict(list)

        for ppo_idx, a2c_cluster in ppo_to_a2c_cluster.items():
            ppo_cluster = self.ppo_clustering["cluster_labels"][ppo_idx]
            pair_to_indices[(ppo_cluster, a2c_cluster)].append(ppo_idx)

        print(f"  Found {len(pair_to_indices)} equivalence classes")

        # Create refined clusters
        refined_clusters = []
        state_to_refined = {}
        refined_id = 0

        for (c1, c2), indices in pair_to_indices.items():
            # Get actions from both clusterings
            ppo_cluster_info = next(
                c for c in self.ppo_clustering["clusters"] if c["id"] == c1
            )
            a2c_cluster_info = next(
                c for c in self.a2c_clustering["clusters"] if c["id"] == c2
            )

            # Get the actual state vectors for these indices
            state_vectors = ppo_states[indices]

            refined_clusters.append(
                {
                    "id": refined_id,
                    "c1_id": c1,
                    "c2_id": c2,
                    "action1": ppo_cluster_info["action"],
                    "action2": a2c_cluster_info["action"],
                    "indices": indices,
                    "state_vectors": state_vectors,
                    "centroid": np.mean(state_vectors, axis=0)
                    if len(state_vectors) > 0
                    else None,
                    "size": len(indices),
                    "agreement": ppo_cluster_info["action"]
                    == a2c_cluster_info["action"],
                }
            )

            for idx in indices:
                state_to_refined[idx] = refined_id

            refined_id += 1

        self.refinement = {
            "clusters": refined_clusters,
            "state_to_cluster": state_to_refined,
            "n_clusters": len(refined_clusters),
            "agreement_rate": np.mean([c["agreement"] for c in refined_clusters]),
        }

        print(f"\nCreated {self.refinement['n_clusters']} refined clusters")
        print(f"Action agreement rate: {self.refinement['agreement_rate']:.3f}")

        # Print disagreement clusters
        print("\nRefined clusters with action disagreement:")
        for cluster in refined_clusters:
            if (
                not cluster["agreement"] and cluster["size"] > 5
            ):  # Show significant ones
                action1_name = self.action_names[cluster["action1"]]
                action2_name = self.action_names[cluster["action2"]]
                print(
                    f"  Cluster {cluster['id']}: size={cluster['size']} "
                    f"(PPO:{action1_name} vs A2C:{action2_name})"
                )

        return self.refinement

    def compute_refinement_q_values(self, gamma: float = 0.99):
        """
        Compute Q-values for refined clusters using Bellman optimality
        Q(C, a) = max over both policies' Q-values for states in C
        """
        if self.refinement is None:
            raise ValueError("Must compute refinement first")

        print("\n" + "=" * 60)
        print("COMPUTING REFINEMENT Q-VALUES")
        print("=" * 60)

        n_refined = self.refinement["n_clusters"]
        n_actions = self.n_actions

        self.q_refinement = np.zeros((n_refined, n_actions))

        # For each refined cluster, we need Q-values from both policies
        # We'll estimate Q-values using the value function and rewards

        for cluster in self.refinement["clusters"]:
            cid = cluster["id"]

            for action in range(n_actions):
                max_q = -np.inf

                # For each state in this cluster, estimate Q(s,a) from both policies
                # We'll use a combination of value and advantage
                for state_vec in cluster["state_vectors"]:
                    # PPO Q-value estimate
                    q_ppo = self._estimate_q_value(self.ppo_model, state_vec, action)

                    # A2C Q-value estimate
                    q_a2c = self._estimate_q_value(self.a2c_model, state_vec, action)

                    max_q = max(max_q, q_ppo, q_a2c)

                self.q_refinement[cid, action] = max_q if max_q > -np.inf else 0

        print(f"Computed Q-values for {n_refined} refined clusters")
        print(
            f"Q-value range: [{np.min(self.q_refinement):.3f}, {np.max(self.q_refinement):.3f}]"
        )

        return self.q_refinement

    def _estimate_q_value(self, model, state: np.ndarray, action: int) -> float:
        """
        Estimate Q(s,a) using the model's value function and policy
        This is an approximation since we don't have explicit Q-values
        """
        state_tensor = th.as_tensor(state).float().to(model.device).unsqueeze(0)

        with th.no_grad():
            # Get value V(s)
            if hasattr(model.policy, "predict_values"):
                value = model.policy.predict_values(state_tensor).cpu().numpy()[0, 0]
            else:
                value = 0

            # Get action probability
            if hasattr(model.policy, "get_distribution"):
                dist = model.policy.get_distribution(state_tensor)
                log_prob = (
                    dist.log_prob(th.as_tensor([action]).to(model.device))
                    .cpu()
                    .numpy()[0]
                )
                prob = np.exp(log_prob)
            else:
                prob = 0.25  # Uniform if can't get distribution

            # Rough Q estimate: Q ≈ V + advantage
            # Advantage is approximated by how much better this action is than average
            advantage = (prob - 0.25) * 2  # Scale factor

        return float(value + advantage)

    def compute_superior_policy(self):
        """
        Compute policy on refined clusters using Bellman optimality
        π*(C) = argmax_a Q_refinement(C, a)
        Then create a function that maps any state to an action
        """
        if self.q_refinement is None:
            raise ValueError("Must compute refinement Q-values first")

        print("\n" + "=" * 60)
        print("COMPUTING SUPERIOR POLICY")
        print("=" * 60)

        # Policy on refined clusters
        cluster_policy = {}
        for cid in range(self.refinement["n_clusters"]):
            cluster_policy[cid] = int(np.argmax(self.q_refinement[cid]))

        # Create a function that maps any state to an action
        # by finding the nearest refined cluster centroid
        centroids = np.array(
            [
                c["centroid"]
                for c in self.refinement["clusters"]
                if c["centroid"] is not None
            ]
        )
        cluster_ids = [
            c["id"] for c in self.refinement["clusters"] if c["centroid"] is not None
        ]

        from scipy.spatial import cKDTree

        self.cluster_tree = cKDTree(centroids)
        self.cluster_ids = np.array(cluster_ids)
        self.cluster_policy = cluster_policy

        def superior_policy(state, deterministic=True):
            # Find nearest cluster
            dist, idx = self.cluster_tree.query(state)
            cluster_id = self.cluster_ids[idx]
            return cluster_policy[cluster_id]

        self.superior_policy_fn = superior_policy

        # Evaluate agreement with original policies on a test set
        test_states = self.ppo_clustering["data"]["states"][:100]  # First 100 states

        agreements = {"ppo": 0, "a2c": 0, "both": 0}

        for state in test_states:
            sup_action = superior_policy(state)

            # Get PPO action
            ppo_action, _ = self.ppo_model.predict(state, deterministic=True)

            # Get A2C action
            a2c_action, _ = self.a2c_model.predict(state, deterministic=True)

            if sup_action == ppo_action:
                agreements["ppo"] += 1
            if sup_action == a2c_action:
                agreements["a2c"] += 1
            if sup_action == ppo_action == a2c_action:
                agreements["both"] += 1

        n = len(test_states)
        print(f"\nPolicy agreement on test states:")
        print(f"  With PPO: {agreements['ppo'] / n:.3f}")
        print(f"  With A2C: {agreements['a2c'] / n:.3f}")
        print(f"  With both: {agreements['both'] / n:.3f}")

        return superior_policy

    def evaluate_policies(self, n_episodes: int = 100):
        """
        Evaluate all policies (PPO, A2C, Superior)
        """
        print("\n" + "=" * 60)
        print("POLICY EVALUATION")
        print("=" * 60)

        results = {}

        for name, policy_fn in [
            ("PPO", lambda s: self.ppo_model.predict(s, deterministic=True)[0]),
            ("A2C", lambda s: self.a2c_model.predict(s, deterministic=True)[0]),
            ("Superior", self.superior_policy_fn),
        ]:
            print(f"\nEvaluating {name} policy...")

            rewards = []
            successes = 0

            for episode in range(n_episodes):
                state, _ = self.env.reset(seed=self.seed + episode)
                done = False
                total_reward = 0

                while not done:
                    action = policy_fn(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    state = next_state

                rewards.append(total_reward)
                if total_reward > 200:
                    successes += 1

            results[name] = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "success_rate": successes / n_episodes,
                "min_reward": np.min(rewards),
                "max_reward": np.max(rewards),
            }

            print(
                f"  Mean reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}"
            )
            print(f"  Success rate: {results[name]['success_rate']:.3f}")

        # Compare improvements
        print("\n" + "-" * 40)
        print("IMPROVEMENT ANALYSIS")
        print("-" * 40)

        sup_vs_ppo = (
            (results["Superior"]["mean_reward"] - results["PPO"]["mean_reward"])
            / abs(results["PPO"]["mean_reward"])
            * 100
        )
        sup_vs_a2c = (
            (results["Superior"]["mean_reward"] - results["A2C"]["mean_reward"])
            / abs(results["A2C"]["mean_reward"])
            * 100
        )

        print(f"Superior vs PPO: {sup_vs_ppo:+.1f}%")
        print(f"Superior vs A2C: {sup_vs_a2c:+.1f}%")

        return results

    def visualize_refinement_2d(self, save_path: Optional[str] = None):
        """
        Visualize the refinement using PCA
        Fixed key error for 'indices'
        """
        if self.refinement is None:
            return

        print("\nCreating 2D visualization...")

        # Prepare data
        all_states = self.ppo_clustering["data"]["states"]
        all_states_scaled = self.scaler.fit_transform(all_states)

        # PCA to 2D for visualization
        pca_viz = PCA(n_components=2)
        states_2d = pca_viz.fit_transform(all_states_scaled)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: PPO clusters
        ax = axes[0, 0]
        colors = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        # Get PPO cluster labels
        ppo_labels = self.ppo_clustering["cluster_labels"]
        for i, cluster in enumerate(self.ppo_clustering["clusters"]):
            # Find indices where cluster label equals this cluster's id
            mask = ppo_labels == cluster["id"]
            if np.any(mask):
                ax.scatter(
                    states_2d[mask, 0],
                    states_2d[mask, 1],
                    c=[colors[i % len(colors)]],
                    alpha=0.3,
                    s=2,
                    label=f"C{i}" if i < 5 else "",
                )
        ax.set_title(f"PPO Clusters ({self.ppo_clustering['n_clusters']} clusters)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Plot 2: A2C clusters
        ax = axes[0, 1]
        # For A2C, we need to project its states into the same PCA space
        a2c_states = self.a2c_clustering["data"]["states"]
        a2c_states_scaled = self.scaler.transform(a2c_states)  # Use same scaler
        a2c_2d = pca_viz.transform(a2c_states_scaled)

        a2c_labels = self.a2c_clustering["cluster_labels"]
        for i, cluster in enumerate(self.a2c_clustering["clusters"]):
            mask = a2c_labels == cluster["id"]
            if np.any(mask):
                ax.scatter(
                    a2c_2d[mask, 0],
                    a2c_2d[mask, 1],
                    c=[colors[i % len(colors)]],
                    alpha=0.3,
                    s=2,
                    label=f"C{i}" if i < 5 else "",
                )
        ax.set_title(f"A2C Clusters ({self.a2c_clustering['n_clusters']} clusters)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Plot 3: Refinement - color by agreement
        ax = axes[0, 2]

        # Create a color map for each state based on its refined cluster
        for cluster in self.refinement["clusters"]:
            indices = cluster["indices"]  # These are indices into PPO data
            if len(indices) > 0:
                color = "green" if cluster["agreement"] else "red"
                ax.scatter(
                    states_2d[indices, 0],
                    states_2d[indices, 1],
                    c=color,
                    alpha=0.3,
                    s=2,
                )

        ax.set_title(
            f"Refinement: Green=Agree, Red=Disagree\n"
            f"Agreement rate: {self.refinement['agreement_rate']:.3f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Plot 4: Action distribution pie charts for disagreement clusters
        # Create a new axes for pie charts - we'll use the bottom left
        ax = axes[1, 0]
        disagreement_clusters = [
            c
            for c in self.refinement["clusters"]
            if not c["agreement"] and c["size"] > 10
        ]

        if disagreement_clusters:
            # Clear the axis
            ax.clear()
            ax.axis("off")

            # Create a grid of pie charts within this axis
            n_plots = min(4, len(disagreement_clusters))
            n_cols = 2
            n_rows = (n_plots + 1) // 2

            for i in range(n_plots):
                cluster = disagreement_clusters[i]
                sub_ax = plt.subplot(2, 2, i + 1)

                # Get action names
                action1_name = self.action_names[cluster["action1"]]
                action2_name = self.action_names[cluster["action2"]]

                # Create pie chart showing the split
                sub_ax.pie(
                    [cluster["size"] / 2, cluster["size"] / 2],
                    labels=[f"PPO:{action1_name[:5]}", f"A2C:{action2_name[:5]}"],
                    colors=["red", "blue"],
                    autopct="%1.0f%%",
                    textprops={"fontsize": 8},
                )
                sub_ax.set_title(f"C{cluster['id']} (n={cluster['size']})", fontsize=9)

            # Set the main title
            axes[1, 0].set_title("Action Disagreements", fontsize=10)
        else:
            ax.text(
                0.5,
                0.5,
                "No significant disagreement clusters",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Action Disagreements")

        # Plot 5: Refinement Q-values
        ax = axes[1, 1]
        if self.q_refinement is not None:
            # Get valid cluster ids that actually exist in refinement
            valid_cluster_ids = [c["id"] for c in self.refinement["clusters"]]
            n_clusters = len(valid_cluster_ids)
            x = np.arange(n_clusters)
            width = 0.2

            for action in range(self.n_actions):
                offset = (action - 1.5) * width
                q_values = [self.q_refinement[cid, action] for cid in valid_cluster_ids]
                ax.bar(
                    x + offset,
                    q_values,
                    width,
                    label=self.action_names[action],
                    alpha=0.7,
                )

            ax.set_xlabel("Refined Cluster ID")
            ax.set_ylabel("Q-value")
            ax.set_title("Refinement Q-values")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot 6: Policy comparison (quick evaluation)
        ax = axes[1, 2]
        # Do a quick evaluation (fewer episodes for speed)
        results = {}
        for name, policy_fn in [
            ("PPO", lambda s: self.ppo_model.predict(s, deterministic=True)[0]),
            ("A2C", lambda s: self.a2c_model.predict(s, deterministic=True)[0]),
            ("Superior", self.superior_policy_fn),
        ]:
            if name == "Superior" and self.superior_policy_fn is None:
                continue

            rewards = []
            for episode in range(20):  # Quick evaluation
                state, _ = self.env.reset(seed=self.seed + episode)
                done = False
                total_reward = 0
                while not done:
                    action = policy_fn(state)
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                rewards.append(total_reward)

            results[name] = {
                "mean_reward": np.mean(rewards),
                "success_rate": np.mean([r > 200 for r in rewards]),
            }

        if results:
            names = list(results.keys())
            x = np.arange(len(names))
            width = 0.35

            rewards = [results[n]["mean_reward"] for n in names]
            successes = [results[n]["success_rate"] for n in names]

            ax.bar(
                x - width / 2,
                rewards,
                width,
                label="Mean Reward",
                alpha=0.7,
                color="blue",
            )
            ax.bar(
                x + width / 2,
                successes,
                width,
                label="Success Rate",
                alpha=0.7,
                color="green",
            )

            ax.set_xlabel("Policy")
            ax.set_ylabel("Value")
            ax.set_title("Policy Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Lunar Lander Cluster Refinement Analysis", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()

    def save_analysis(self, save_dir: str = "lunar_refinement_results"):
        """
        Save all analysis results
        """
        os.makedirs(save_dir, exist_ok=True)

        results = {
            "ppo_clusters": self.ppo_clustering["n_clusters"],
            "a2c_clusters": self.a2c_clustering["n_clusters"],
            "refinement_clusters": self.refinement["n_clusters"],
            "agreement_rate": self.refinement["agreement_rate"],
            "q_refinement": self.q_refinement.tolist()
            if self.q_refinement is not None
            else None,
        }

        # Add policy evaluation
        eval_results = self.evaluate_policies(n_episodes=100)
        results["policy_evaluation"] = {
            name: {
                k: float(v) if isinstance(v, np.floating) else v
                for k, v in data.items()
            }
            for name, data in eval_results.items()
        }

        with open(f"{save_dir}/refinement_results.json", "w") as f:
            import json

            json.dump(results, f, indent=2)

        print(f"\nResults saved to {save_dir}/")

        return results


def run_lunar_refinement_experiment(ppo_path: str, a2c_path: str, seed: int = 42):
    """
    Run complete refinement experiment for Lunar Lander
    """
    # Initialize analyzer
    analyzer = LunarLanderClusterRefinement(
        ppo_model_path=ppo_path, a2c_model_path=a2c_path, seed=seed, n_clusters=12
    )

    # Collect data from both policies
    print("\n" + "=" * 60)
    print("COLLECTING TRAJECTORY DATA")
    print("=" * 60)

    print("\nPPO policy:")
    ppo_data = analyzer.collect_trajectories(analyzer.ppo_model, n_episodes=200)

    print("\nA2C policy:")
    a2c_data = analyzer.collect_trajectories(analyzer.a2c_model, n_episodes=200)

    # Cluster states
    analyzer.ppo_clustering = analyzer.cluster_states_by_action(ppo_data, "PPO")
    analyzer.a2c_clustering = analyzer.cluster_states_by_action(a2c_data, "A2C")

    # Compute refinement
    refinement = analyzer.compute_coarsest_common_refinement()

    # Compute Q-values
    q_values = analyzer.compute_refinement_q_values()

    # Compute superior policy
    superior_policy = analyzer.compute_superior_policy()

    # Evaluate all policies
    results = analyzer.evaluate_policies(n_episodes=100)

    # Visualize
    analyzer.visualize_refinement_2d(save_path="lunar_refinement_visualization.png")

    # Save results
    analyzer.save_analysis()

    return analyzer


if __name__ == "__main__":
    # Example usage - update paths to your saved models
    ppo_model_path = "models/ppo_lunar_lander_v3.zip"
    a2c_model_path = "models/a2c_lunar_lander_v3.zip"

    # Check if files exist
    if not os.path.exists(ppo_model_path):
        print(f"PPO model not found at {ppo_model_path}")
        print("Please update the paths to your saved models")
    elif not os.path.exists(a2c_model_path):
        print(f"A2C model not found at {a2c_model_path}")
    else:
        analyzer = run_lunar_refinement_experiment(
            ppo_path=ppo_model_path, a2c_path=a2c_model_path, seed=42
        )
