import numpy as np
import gymnasium as gym
import pickle
import os
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
import json
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import warnings
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import ks_2samp, entropy

# Stable Baselines3 imports
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch as th
import matplotlib.gridspec as gridspec
from cluster_transitions import (
    CombinatorialClusterTransitionAnalyzer as ClusterTransitionAnalyzer,
    analyze_cluster_transitions,
)

warnings.filterwarnings("ignore")


class ContinuousStateCluster:
    """
    Represents a cluster in continuous state space
    """

    def __init__(
        self,
        cluster_id: int,
        states: np.ndarray,
        actions: List[int],
        values: np.ndarray,
        centroids: np.ndarray,
        state_dim: int,
    ):
        self.id = cluster_id
        self.states = states  # N x dim array of states in this cluster
        self.actions = actions  # Actions taken in these states
        self.values = values  # State values
        self.centroid = centroids  # Cluster centroid in original space
        self.size = len(states)
        self.state_dim = state_dim

        # Action distribution
        unique_actions, counts = np.unique(actions, return_counts=True)
        self.action_distribution = {
            int(a): float(c / self.size) for a, c in zip(unique_actions, counts)
        }
        self.dominant_action = (
            int(unique_actions[np.argmax(counts)]) if len(unique_actions) > 0 else -1
        )

        # Value statistics
        self.value_mean = float(np.mean(values)) if len(values) > 0 else 0
        self.value_std = float(np.std(values)) if len(values) > 0 else 0
        self.value_min = float(np.min(values)) if len(values) > 0 else 0
        self.value_max = float(np.max(values)) if len(values) > 0 else 0

        # Cluster spread (average distance from centroid)
        if len(states) > 0:
            distances = np.linalg.norm(states - self.centroid.reshape(1, -1), axis=1)
            self.spread = float(np.mean(distances))
            self.max_distance = float(np.max(distances))
        else:
            self.spread = 0
            self.max_distance = 0

    def to_dict(self):
        return {
            "id": self.id,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "size": self.size,
            "action_distribution": self.action_distribution,
            "dominant_action": self.dominant_action,
            "value_mean": self.value_mean,
            "value_std": self.value_std,
            "value_range": [self.value_min, self.value_max],
            "spread": self.spread,
            "max_distance": self.max_distance,
        }


class LunarLanderClusterAnalyzer:
    """
    Analyze clusters in Lunar Lander's continuous state space
    Updated for LunarLander-v3
    """

    def __init__(self, model, env, name: str = "policy", device: str = "auto"):
        self.model = model
        self.env = env
        self.name = name
        self.device = device

        # Environment specs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        # Action names for Lunar Lander (same for v2 and v3)
        self.action_names = {
            0: "Do nothing",
            1: "Fire left engine",
            2: "Fire main engine",
            3: "Fire right engine",
        }

        # State component names (same for v2 and v3)
        self.state_names = [
            "x_pos",
            "y_pos",
            "x_vel",
            "y_vel",
            "angle",
            "angular_vel",
            "left_contact",
            "right_contact",
        ]

        # Storage for collected data
        self.collected_states = None
        self.collected_actions = None
        self.collected_values = None
        self.collected_rewards = None
        self.collected_trajectories = []

        # Clustering results
        self.clusters = []
        self.state_to_cluster = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.tsne_result = None
        self.use_pca = False
        self.n_pca_components = min(4, self.state_dim)

    def collect_trajectories(
        self, n_episodes: int = 1000, deterministic: bool = False
    ) -> List[Dict]:
        """
        Collect trajectories by running the policy
        Updated for v3 - success threshold might be different
        """
        trajectories = []
        all_states = []
        all_actions = []
        all_values = []

        print(f"\nCollecting {n_episodes} trajectories from {self.name}...")

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "values": [],
                "next_states": [],
                "success": False,
            }

            while not done:
                # Get action and value
                action, _ = self.model.predict(state, deterministic=deterministic)

                # Get state value (if using SB3's value function)
                if hasattr(self.model, "policy") and hasattr(
                    self.model.policy, "predict_values"
                ):
                    state_tensor = th.as_tensor(state).float().to(self.model.device)
                    with th.no_grad():
                        value = (
                            self.model.policy.predict_values(state_tensor.unsqueeze(0))
                            .cpu()
                            .numpy()[0, 0]
                        )
                else:
                    value = 0

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store data
                trajectory["states"].append(state.copy())
                trajectory["actions"].append(int(action))
                trajectory["rewards"].append(float(reward))
                trajectory["values"].append(float(value))
                trajectory["next_states"].append(next_state.copy())

                all_states.append(state.copy())
                all_actions.append(int(action))
                all_values.append(float(value))

                state = next_state
                total_reward += reward

            # v3 has same reward structure as v2, but you might want to adjust threshold
            # Success if landed softly (reward > 200 is typical for good landing)
            trajectory["success"] = bool(total_reward > 200)
            trajectory["total_reward"] = total_reward
            trajectories.append(trajectory)

            if (episode + 1) % 100 == 0:
                print(f"  Collected {episode + 1} episodes")

        self.collected_states = np.array(all_states)
        self.collected_actions = np.array(all_actions)
        self.collected_values = np.array(all_values)
        self.collected_trajectories = trajectories

        print(f"Total states collected: {len(self.collected_states)}")

        return trajectories

    def preprocess_states(self, use_pca: bool = True, n_components: int = 4):
        """
        Preprocess states with scaling and optional PCA
        Returns both scaled and optionally reduced data, and keeps track of transformation
        """
        if self.collected_states is None:
            raise ValueError("No states collected yet. Run collect_trajectories first.")

        # Standardize features
        states_scaled = self.scaler.fit_transform(self.collected_states)

        self.use_pca = use_pca
        self.n_pca_components = min(n_components, self.state_dim)

        if use_pca and self.state_dim > n_components:
            self.pca = PCA(n_components=self.n_pca_components)
            states_reduced = self.pca.fit_transform(states_scaled)
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            print(
                f"PCA explained variance with {self.n_pca_components} components: {explained_var:.3f}"
            )
            return states_reduced, True
        else:
            return states_scaled, False

    def cluster_by_action_preference(
        self,
        n_clusters: Optional[int] = None,
        method: str = "kmeans",
        use_pca: bool = True,
    ) -> List[ContinuousStateCluster]:
        """
        Cluster states based on their features and action preferences
        Now handles dimensionality consistently
        """
        # Preprocess states - get both reduced and original scaled versions
        states_processed, reduced = self.preprocess_states(use_pca=use_pca)

        # Determine number of clusters if not specified
        if n_clusters is None:
            # Use heuristic based on data size
            n_clusters = min(15, max(5, len(self.collected_states) // 500))

        print(
            f"\nClustering {len(self.collected_states)} states into {n_clusters} clusters..."
        )
        print(
            f"Using {method} on {'PCA-reduced' if reduced else 'original scaled'} data"
        )

        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(states_processed)

            # Get centroids in the processed space
            centroids_processed = clusterer.cluster_centers_

            # Transform centroids back to original space
            if reduced and self.pca is not None:
                # First inverse transform PCA, then inverse scale
                centroids_scaled = self.pca.inverse_transform(centroids_processed)
                centroids_original = self.scaler.inverse_transform(centroids_scaled)
            else:
                # Just inverse scale
                centroids_original = self.scaler.inverse_transform(centroids_processed)

        elif method == "gmm":
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = clusterer.fit_predict(states_processed)

            # Get centroids (means) in processed space
            centroids_processed = clusterer.means_

            # Transform back to original space
            if reduced and self.pca is not None:
                centroids_scaled = self.pca.inverse_transform(centroids_processed)
                centroids_original = self.scaler.inverse_transform(centroids_scaled)
            else:
                centroids_original = self.scaler.inverse_transform(centroids_processed)

        elif method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(states_processed)

            # Compute centroids manually in processed space
            centroids_processed = np.zeros((n_clusters, states_processed.shape[1]))
            for i in range(n_clusters):
                mask = labels == i
                if np.any(mask):
                    centroids_processed[i] = np.mean(states_processed[mask], axis=0)

            # Transform back to original space
            if reduced and self.pca is not None:
                centroids_scaled = self.pca.inverse_transform(centroids_processed)
                centroids_original = self.scaler.inverse_transform(centroids_scaled)
            else:
                centroids_original = self.scaler.inverse_transform(centroids_processed)

        elif method == "dbscan":
            # DBSCAN determines number of clusters automatically
            clusterer = DBSCAN(eps=0.5, min_samples=50)
            labels = clusterer.fit_predict(states_processed)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            print(f"DBSCAN found {n_clusters} clusters + noise")

            # Compute centroids for non-noise points
            centroids_processed = []
            valid_cluster_ids = []
            for i in range(n_clusters):
                mask = labels == i
                if np.any(mask):
                    centroids_processed.append(np.mean(states_processed[mask], axis=0))
                    valid_cluster_ids.append(i)

            centroids_processed = np.array(centroids_processed)

            # Transform back to original space
            if reduced and self.pca is not None:
                centroids_scaled = self.pca.inverse_transform(centroids_processed)
                centroids_original = self.scaler.inverse_transform(centroids_scaled)
            else:
                centroids_original = self.scaler.inverse_transform(centroids_processed)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Create cluster objects
        self.clusters = []
        self.state_to_cluster = {}

        # Handle DBSCAN case where we might have fewer clusters than n_clusters
        if method == "dbscan":
            cluster_ids = valid_cluster_ids
        else:
            cluster_ids = range(n_clusters)

        for i, cluster_id in enumerate(cluster_ids):
            mask = labels == cluster_id
            if not np.any(mask):
                continue

            cluster_states = self.collected_states[mask]
            cluster_actions = self.collected_actions[mask]
            cluster_values = self.collected_values[mask]

            cluster = ContinuousStateCluster(
                cluster_id=i,
                states=cluster_states,
                actions=cluster_actions.tolist(),
                values=cluster_values,
                centroids=centroids_original[i],
                state_dim=self.state_dim,
            )

            self.clusters.append(cluster)

            # Map original indices to cluster
            indices = np.where(mask)[0]
            for idx in indices:
                self.state_to_cluster[idx] = i

        print(f"Created {len(self.clusters)} clusters")

        # Print cluster statistics
        for cluster in self.clusters:
            print(
                f"  Cluster {cluster.id}: size={cluster.size}, "
                f"dominant action={self.action_names[cluster.dominant_action] if cluster.dominant_action >= 0 else 'unknown'}, "
                f"value_mean={cluster.value_mean:.3f}, spread={cluster.spread:.3f}"
            )

        return self.clusters

    def build_transition_graph(self) -> nx.DiGraph:
        """
        Build transition graph between clusters using trajectory data
        """
        if not self.clusters:
            raise ValueError("Must run clustering first")

        G = nx.DiGraph()

        # Add nodes
        for cluster in self.clusters:
            G.add_node(
                cluster.id,
                size=cluster.size,
                dominant_action=cluster.dominant_action,
                action_dist=cluster.action_distribution,
                value_mean=cluster.value_mean,
                centroid=cluster.centroid.tolist()
                if cluster.centroid is not None
                else None,
                spread=cluster.spread,
            )

        # Count transitions between clusters
        transitions = defaultdict(lambda: defaultdict(int))

        for traj in self.collected_trajectories:
            prev_cluster = None

            for i, state in enumerate(traj["states"]):
                # Find which cluster this state belongs to by nearest centroid
                state_array = np.array(state)

                # Compute distances to all cluster centroids
                min_dist = float("inf")
                current_cluster = None

                for cluster in self.clusters:
                    if cluster.centroid is not None:
                        dist = np.linalg.norm(state_array - cluster.centroid)
                        if dist < min_dist:
                            min_dist = dist
                            current_cluster = cluster.id

                if (
                    current_cluster is not None
                    and prev_cluster is not None
                    and prev_cluster != current_cluster
                ):
                    transitions[prev_cluster][current_cluster] += 1

                prev_cluster = current_cluster

        # Add edges with probabilities
        for c_from, targets in transitions.items():
            total = sum(targets.values())
            for c_to, count in targets.items():
                prob = count / total if total > 0 else 0
                G.add_edge(
                    c_from,
                    c_to,
                    probability=float(prob),
                    weight=float(prob * 5),  # Scale for visualization
                    count=int(count),
                )

        print(
            f"Built transition graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )

        return G

    def visualize_clusters_2d(self, save_path: Optional[str] = None):
        """
        Visualize clusters in 2D using t-SNE
        Fixed subplot indexing for pie charts
        """
        if self.collected_states is None:
            raise ValueError("No data collected. Run collect_trajectories first.")

        fig = plt.figure(figsize=(18, 14))

        # Create a 2x2 grid for the main plots
        gs_main = fig.add_gridspec(
            2, 2, left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3
        )

        # Prepare data - use scaled data for t-SNE
        states_scaled = self.scaler.fit_transform(self.collected_states)

        # Compute t-SNE if not already done
        if self.tsne_result is None:
            print("Computing t-SNE (this may take a moment)...")
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(states_scaled) - 1),
            )
            self.tsne_result = tsne.fit_transform(states_scaled)

        states_2d = self.tsne_result

        # Plot 1: States colored by action
        ax1 = fig.add_subplot(gs_main[0, 0])
        colors = ["red", "blue", "green", "purple"]
        for action in range(self.n_actions):
            mask = self.collected_actions == action
            if np.any(mask):
                ax1.scatter(
                    states_2d[mask, 0],
                    states_2d[mask, 1],
                    c=colors[action],
                    label=self.action_names[action],
                    alpha=0.3,
                    s=2,
                )
        ax1.set_title(f"{self.name}: States Colored by Action")
        ax1.legend(markerscale=2)
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")

        # Plot 2: States colored by value
        ax2 = fig.add_subplot(gs_main[0, 1])
        scatter = ax2.scatter(
            states_2d[:, 0],
            states_2d[:, 1],
            c=self.collected_values,
            cmap="viridis",
            alpha=0.3,
            s=2,
            vmin=-1,
            vmax=1,
        )
        ax2.set_title(f"{self.name}: States Colored by Value")
        plt.colorbar(scatter, ax=ax2, label="State Value")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")

        # Plot 3: Clusters
        ax3 = fig.add_subplot(gs_main[1, 0])
        if self.clusters:
            # Assign colors to clusters
            cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(self.clusters)))

            # Plot each cluster's points
            for i, cluster in enumerate(self.clusters):
                # Get indices for this cluster
                cluster_indices = [
                    idx
                    for idx, cid in self.state_to_cluster.items()
                    if cid == cluster.id
                ]
                if cluster_indices:
                    ax3.scatter(
                        states_2d[cluster_indices, 0],
                        states_2d[cluster_indices, 1],
                        c=[cluster_colors[i]],
                        alpha=0.2,
                        s=2,
                        label=f"C{cluster.id}",
                    )

                # Find approximate centroid in t-SNE space
                if cluster_indices:
                    centroid_2d = np.mean(states_2d[cluster_indices], axis=0)
                    ax3.plot(
                        centroid_2d[0],
                        centroid_2d[1],
                        "x",
                        color="black",
                        markersize=10,
                        markeredgewidth=2,
                    )
                    ax3.annotate(
                        f"C{cluster.id}",
                        centroid_2d,
                        fontsize=9,
                        weight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="white", alpha=0.7
                        ),
                    )

        ax3.set_title(f"{self.name}: Cluster Assignments")
        ax3.set_xlabel("t-SNE 1")
        ax3.set_ylabel("t-SNE 2")

        # Plot 4: Action distribution pie charts for each cluster
        # Create a separate grid for pie charts in the bottom right
        if self.clusters:
            n_clusters = len(self.clusters)
            # Calculate grid size for pie charts
            n_cols = int(np.ceil(np.sqrt(n_clusters)))
            n_rows = int(np.ceil(n_clusters / n_cols))

            # Create a new gridspec for pie charts within the bottom right cell
            gs_pie = gridspec.GridSpecFromSubplotSpec(
                n_rows, n_cols, subplot_spec=gs_main[1, 1]
            )

            colors_pie = ["red", "blue", "green", "purple"]
            for i, cluster in enumerate(self.clusters):
                if i < n_rows * n_cols:  # Ensure we don't exceed the grid
                    sub_ax = fig.add_subplot(gs_pie[i // n_cols, i % n_cols])

                    actions = list(cluster.action_distribution.keys())
                    sizes = list(cluster.action_distribution.values())
                    colors_sub = [colors_pie[a] for a in actions]

                    wedges, texts, autotexts = sub_ax.pie(
                        sizes,
                        colors=colors_sub,
                        autopct="%1.0f%%",
                        startangle=90,
                        textprops={"fontsize": 6},
                    )
                    sub_ax.set_title(
                        f"C{cluster.id}\nμ={cluster.value_mean:.2f}", fontsize=8
                    )

                    for autotext in autotexts:
                        autotext.set_fontsize(5)
        else:
            ax4 = fig.add_subplot(gs_main[1, 1])
            ax4.text(
                0.5,
                0.5,
                "No clusters to display",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Action Distribution")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()

    def visualize_state_space_3d(self, save_path: Optional[str] = None):
        """
        Create 3D visualization of important state dimensions using PCA
        """
        if self.collected_states is None:
            raise ValueError("No data collected. Run collect_trajectories first.")

        # Prepare data
        states_scaled = self.scaler.fit_transform(self.collected_states)

        # PCA to 3D
        pca_3d = PCA(n_components=3)
        states_3d = pca_3d.fit_transform(states_scaled)

        print(
            f"3D PCA explained variance: {np.sum(pca_3d.explained_variance_ratio_):.3f}"
        )

        fig = plt.figure(figsize=(18, 6))

        # Plot 1: 3D scatter colored by action
        ax1 = fig.add_subplot(131, projection="3d")
        colors = ["red", "blue", "green", "purple"]
        for action in range(self.n_actions):
            mask = self.collected_actions == action
            if np.any(mask):
                ax1.scatter(
                    states_3d[mask, 0],
                    states_3d[mask, 1],
                    states_3d[mask, 2],
                    c=colors[action],
                    label=self.action_names[action],
                    alpha=0.3,
                    s=2,
                )
        ax1.set_title(f"{self.name}: Actions in 3D PCA Space")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_zlabel("PC3")
        ax1.legend()

        # Plot 2: 3D scatter colored by value
        ax2 = fig.add_subplot(132, projection="3d")
        scatter = ax2.scatter(
            states_3d[:, 0],
            states_3d[:, 1],
            states_3d[:, 2],
            c=self.collected_values,
            cmap="viridis",
            alpha=0.3,
            s=2,
            vmin=-1,
            vmax=1,
        )
        ax2.set_title(f"{self.name}: Values in 3D PCA Space")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("PC3")
        plt.colorbar(scatter, ax=ax2, label="State Value", shrink=0.5)

        # Plot 3: 3D scatter with cluster centers
        ax3 = fig.add_subplot(133, projection="3d")
        if self.clusters:
            # Plot all points lightly
            ax3.scatter(
                states_3d[:, 0],
                states_3d[:, 1],
                states_3d[:, 2],
                c="gray",
                alpha=0.1,
                s=1,
            )

            # Plot cluster centers
            for cluster in self.clusters:
                if cluster.centroid is not None:
                    # Transform centroid to PCA space
                    centroid_scaled = self.scaler.transform(
                        cluster.centroid.reshape(1, -1)
                    )
                    centroid_3d = pca_3d.transform(centroid_scaled)[0]

                    ax3.scatter(
                        centroid_3d[0],
                        centroid_3d[1],
                        centroid_3d[2],
                        c=colors[cluster.dominant_action]
                        if cluster.dominant_action >= 0
                        else "black",
                        s=200,
                        marker="o",
                        alpha=0.8,
                    )
                    ax3.text(
                        centroid_3d[0],
                        centroid_3d[1],
                        centroid_3d[2],
                        f"C{cluster.id}",
                        fontsize=8,
                        weight="bold",
                    )

        ax3.set_title(f"{self.name}: Cluster Centers in PCA Space")
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.set_zlabel("PC3")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"3D visualization saved to {save_path}")

        plt.show()

    def analyze_cluster_transitions(self, G: nx.DiGraph) -> Dict:
        """
        Analyze transition patterns between clusters
        """
        if G.number_of_nodes() == 0:
            return {}

        analysis = {
            "n_clusters": len(self.clusters),
            "n_edges": G.number_of_edges(),
            "transition_matrix": [],
            "cluster_importance": {},
            "bottleneck_clusters": [],
            "absorption_clusters": [],
        }

        # Build transition matrix
        n = len(self.clusters)
        trans_matrix = np.zeros((n, n))

        for u, v, d in G.edges(data=True):
            if u < n and v < n:
                trans_matrix[u, v] = d["probability"]

        analysis["transition_matrix"] = trans_matrix.tolist()

        # Compute PageRank (cluster importance)
        try:
            pagerank = nx.pagerank(G, weight="probability")
            # Filter to only include existing nodes
            analysis["cluster_importance"] = {
                int(k): float(v) for k, v in pagerank.items() if k in range(n)
            }
        except Exception as e:
            print(f"PageRank computation failed: {e}")

        # Find bottleneck clusters (high betweenness centrality)
        try:
            betweenness = nx.betweenness_centrality(G, weight="probability")
            if betweenness:
                betweenness_values = list(betweenness.values())
                bottleneck_threshold = np.mean(betweenness_values) + np.std(
                    betweenness_values
                )
                analysis["bottleneck_clusters"] = [
                    int(k)
                    for k, v in betweenness.items()
                    if v > bottleneck_threshold and k in range(n)
                ]
        except Exception as e:
            print(f"Betweenness computation failed: {e}")

        # Find absorption clusters (high in-degree)
        in_degrees = dict(G.in_degree(weight="probability"))
        if in_degrees:
            in_degree_values = list(in_degrees.values())
            mean_in_degree = np.mean(in_degree_values)
            analysis["absorption_clusters"] = [
                int(k)
                for k, v in in_degrees.items()
                if v > mean_in_degree and k in range(n)
            ]

        return analysis

    def save_analysis(self, base_filename: str):
        """
        Save all analysis results
        """
        os.makedirs("lunar_analysis", exist_ok=True)

        # Prepare data for saving
        cluster_data = {
            "name": self.name,
            "n_clusters": len(self.clusters),
            "clusters": [c.to_dict() for c in self.clusters],
            "action_names": self.action_names,
            "state_names": self.state_names,
        }

        # Save cluster data
        with open(f"lunar_analysis/{base_filename}_clusters.json", "w") as f:
            json.dump(cluster_data, f, indent=2)

        # Save collected trajectories (sample)
        trajectory_sample = []
        for i, traj in enumerate(self.collected_trajectories[:10]):  # Save first 10
            traj_sample = {
                "states": [s.tolist() for s in traj["states"][:100]],  # First 100 steps
                "actions": traj["actions"][:100],
                "rewards": traj["rewards"][:100],
                "total_reward": float(traj["total_reward"]),
                "success": bool(traj["success"]),
            }
            trajectory_sample.append(traj_sample)

        with open(f"lunar_analysis/{base_filename}_trajectories_sample.json", "w") as f:
            json.dump(trajectory_sample, f, indent=2)

        print(f"\nAnalysis saved to lunar_analysis/{base_filename}_*")


def create_lunar_lander_env(render_mode: Optional[str] = None):
    """
    Create Lunar Lander environment with proper wrappers
    Updated to use LunarLander-v3
    """
    # Changed from "LunarLander-v2" to "LunarLander-v3"
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    env = Monitor(env)
    return env


def train_ppo_lunar_lander(total_timesteps: int = 300000, render: bool = False):
    """
    Train PPO agent on Lunar Lander v3
    """
    print("\n" + "=" * 60)
    print("Training PPO on LunarLander-v3")
    print("=" * 60)

    # Create environment
    env = create_lunar_lander_env(render_mode="human" if render else None)
    env = DummyVecEnv([lambda: env])

    # Normalize observations (optional but recommended)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log="./logs/ppo_lunar/",
    )

    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    training_time = time.time() - start_time

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"\nPPO Training completed in {training_time:.2f} seconds")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model, env, training_time, mean_reward


def train_a2c_lunar_lander(total_timesteps: int = 300000, render: bool = False):
    """
    Train A2C agent on Lunar Lander v3
    """
    print("\n" + "=" * 60)
    print("Training A2C on LunarLander-v3")
    print("=" * 60)

    # Create environment
    env = create_lunar_lander_env(render_mode="human" if render else None)
    env = DummyVecEnv([lambda: env])

    # Normalize observations
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Initialize A2C
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        verbose=0,
        tensorboard_log="./logs/a2c_lunar/",
    )

    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    training_time = time.time() - start_time

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"\nA2C Training completed in {training_time:.2f} seconds")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model, env, training_time, mean_reward


def run_lunar_lander_comparison():
    """
    Run comprehensive comparison between PPO and A2C on Lunar Lander v3
    """
    print("\n" + "=" * 80)
    print("PPO vs A2C COMPARISON ON LUNAR LANDER v3")
    print("=" * 80)

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("lunar_analysis", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    results = {}

    # 1. Train PPO
    ppo_model, ppo_env, ppo_time, ppo_reward = train_ppo_lunar_lander(
        total_timesteps=300000
    )
    ppo_model.save("models/ppo_lunar_lander_v3")

    # 2. Train A2C
    a2c_model, a2c_env, a2c_time, a2c_reward = train_a2c_lunar_lander(
        total_timesteps=300000
    )
    a2c_model.save("models/a2c_lunar_lander_v3")

    # 3. Create evaluation environments (without normalization for analysis)
    eval_env_ppo = create_lunar_lander_env()
    eval_env_a2c = create_lunar_lander_env()

    # 4. Analyze PPO
    print("\n" + "-" * 80)
    print("ANALYZING PPO CLUSTERS")
    print("-" * 80)

    ppo_analyzer = LunarLanderClusterAnalyzer(ppo_model, eval_env_ppo, "PPO")
    ppo_trajectories = ppo_analyzer.collect_trajectories(n_episodes=200)
    ppo_analyzer.cluster_by_action_preference(
        n_clusters=12, method="kmeans", use_pca=True
    )
    ppo_graph = ppo_analyzer.build_transition_graph()
    ppo_analyzer.visualize_clusters_2d(save_path="lunar_analysis/ppo_clusters_2d.png")
    ppo_analyzer.visualize_state_space_3d(
        save_path="lunar_analysis/ppo_clusters_3d.png"
    )
    ppo_analyzer.save_analysis("ppo")
    ppo_transition_analysis = ppo_analyzer.analyze_cluster_transitions(ppo_graph)

    # 5. Analyze A2C
    print("\n" + "-" * 80)
    print("ANALYZING A2C CLUSTERS")
    print("-" * 80)

    a2c_analyzer = LunarLanderClusterAnalyzer(a2c_model, eval_env_a2c, "A2C")
    a2c_trajectories = a2c_analyzer.collect_trajectories(n_episodes=200)
    a2c_analyzer.cluster_by_action_preference(
        n_clusters=12, method="kmeans", use_pca=True
    )
    a2c_graph = a2c_analyzer.build_transition_graph()
    a2c_analyzer.visualize_clusters_2d(save_path="lunar_analysis/a2c_clusters_2d.png")
    a2c_analyzer.visualize_state_space_3d(
        save_path="lunar_analysis/a2c_clusters_3d.png"
    )
    a2c_analyzer.save_analysis("a2c")
    a2c_transition_analysis = a2c_analyzer.analyze_cluster_transitions(a2c_graph)

    # 6. Compare cluster characteristics
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'PPO':>20} {'A2C':>20}")
    print("-" * 72)

    # Training metrics
    print(f"{'Training Time (s)':<30} {ppo_time:>20.2f} {a2c_time:>20.2f}")
    print(f"{'Mean Reward':<30} {ppo_reward:>20.2f} {a2c_reward:>20.2f}")

    # Cluster statistics
    print(
        f"{'Number of Clusters':<30} {len(ppo_analyzer.clusters):>20} {len(a2c_analyzer.clusters):>20}"
    )

    # Average cluster properties
    ppo_avg_size = (
        np.mean([c.size for c in ppo_analyzer.clusters]) if ppo_analyzer.clusters else 0
    )
    a2c_avg_size = (
        np.mean([c.size for c in a2c_analyzer.clusters]) if a2c_analyzer.clusters else 0
    )
    print(f"{'Avg Cluster Size':<30} {ppo_avg_size:>20.1f} {a2c_avg_size:>20.1f}")

    ppo_avg_value = (
        np.mean([c.value_mean for c in ppo_analyzer.clusters])
        if ppo_analyzer.clusters
        else 0
    )
    a2c_avg_value = (
        np.mean([c.value_mean for c in a2c_analyzer.clusters])
        if a2c_analyzer.clusters
        else 0
    )
    print(f"{'Avg Cluster Value':<30} {ppo_avg_value:>20.3f} {a2c_avg_value:>20.3f}")

    ppo_avg_spread = (
        np.mean([c.spread for c in ppo_analyzer.clusters])
        if ppo_analyzer.clusters
        else 0
    )
    a2c_avg_spread = (
        np.mean([c.spread for c in a2c_analyzer.clusters])
        if a2c_analyzer.clusters
        else 0
    )
    print(f"{'Avg Cluster Spread':<30} {ppo_avg_spread:>20.3f} {a2c_avg_spread:>20.3f}")

    # Action distribution across clusters
    print("\nAction Distribution by Algorithm:")
    for action in range(4):
        ppo_action_clusters = sum(
            1 for c in ppo_analyzer.clusters if c.dominant_action == action
        )
        a2c_action_clusters = sum(
            1 for c in a2c_analyzer.clusters if c.dominant_action == action
        )
        ppo_pct = (
            (ppo_action_clusters / len(ppo_analyzer.clusters) * 100)
            if ppo_analyzer.clusters
            else 0
        )
        a2c_pct = (
            (a2c_action_clusters / len(a2c_analyzer.clusters) * 100)
            if a2c_analyzer.clusters
            else 0
        )
        print(f"  {ppo_analyzer.action_names[action]}:")
        print(f"    PPO: {ppo_action_clusters} clusters ({ppo_pct:.1f}%)")
        print(f"    A2C: {a2c_action_clusters} clusters ({a2c_pct:.1f}%)")

    # 7. Generate report
    report = f"""
LUNAR LANDER v3 CLUSTER ANALYSIS REPORT
{"=" * 80}

Environment: LunarLander-v3
State Dimensions: 8 (position, velocity, angle, contacts)
Actions: 4 discrete engines

ALGORITHM COMPARISON
{"-" * 80}

PPO (Proximal Policy Optimization):
  - Training Time: {ppo_time:.2f}s
  - Mean Reward: {ppo_reward:.2f}
  - Clusters: {len(ppo_analyzer.clusters)}
  - Avg Cluster Size: {ppo_avg_size:.1f}
  - Avg Cluster Value: {ppo_avg_value:.3f}
  - Avg Cluster Spread: {ppo_avg_spread:.3f}
  - Bottleneck Clusters: {ppo_transition_analysis.get("bottleneck_clusters", [])}
  - Absorption Clusters: {ppo_transition_analysis.get("absorption_clusters", [])}

A2C (Advantage Actor Critic):
  - Training Time: {a2c_time:.2f}s
  - Mean Reward: {a2c_reward:.2f}
  - Clusters: {len(a2c_analyzer.clusters)}
  - Avg Cluster Size: {a2c_avg_size:.1f}
  - Avg Cluster Value: {a2c_avg_value:.3f}
  - Avg Cluster Spread: {a2c_avg_spread:.3f}
  - Bottleneck Clusters: {a2c_transition_analysis.get("bottleneck_clusters", [])}
  - Absorption Clusters: {a2c_transition_analysis.get("absorption_clusters", [])}

KEY INSIGHTS
{"-" * 80}

1. State Space Coverage:
   - PPO clusters are {"more" if ppo_avg_spread < a2c_avg_spread else "less"} compact (spread {ppo_avg_spread:.3f} vs {a2c_avg_spread:.3f})
   - This suggests PPO {"focuses on specific regions" if ppo_avg_spread < a2c_avg_spread else "explores more broadly"}

2. Action Preferences:
   - PPO dominant actions: {", ".join([f"{ppo_analyzer.action_names[c.dominant_action]}" for c in ppo_analyzer.clusters[:5] if c.dominant_action >= 0])}...
   - A2C dominant actions: {", ".join([f"{a2c_analyzer.action_names[c.dominant_action]}" for c in a2c_analyzer.clusters[:5] if c.dominant_action >= 0])}...

3. Value Distribution:
   - PPO value range: [{min(c.value_mean for c in ppo_analyzer.clusters):.2f}, {max(c.value_mean for c in ppo_analyzer.clusters):.2f}]
   - A2C value range: [{min(c.value_mean for c in a2c_analyzer.clusters):.2f}, {max(c.value_mean for c in a2c_analyzer.clusters):.2f}]

4. Transition Dynamics:
   - PPO has {len(ppo_transition_analysis.get("bottleneck_clusters", []))} bottleneck clusters (critical decision points)
   - A2C has {len(a2c_transition_analysis.get("bottleneck_clusters", []))} bottleneck clusters

Files Generated:
  - models/ppo_lunar_lander_v3.zip
  - models/a2c_lunar_lander_v3.zip
  - lunar_analysis/ppo_clusters.json
  - lunar_analysis/a2c_clusters.json
  - lunar_analysis/*_2d.png (t-SNE visualizations)
  - lunar_analysis/*_3d.png (3D PCA visualizations)
"""

    with open("lunar_analysis/comparison_report_v3.txt", "w") as f:
        f.write(report)

    print(report)

    return {
        "ppo": (ppo_model, ppo_analyzer),
        "a2c": (a2c_model, a2c_analyzer),
        "ppo_transitions": ppo_transition_analysis,
        "a2c_transitions": a2c_transition_analysis,
    }


def load_and_visualize_saved_model(model_path: str, algorithm: str = "PPO"):
    """
    Load a saved model and visualize its clustering
    """
    print(f"\nLoading {algorithm} model from {model_path}")

    # Create environment
    env = create_lunar_lander_env()

    # Load model
    if algorithm.upper() == "PPO":
        model = PPO.load(model_path)
    else:
        model = A2C.load(model_path)

    # Analyze
    analyzer = LunarLanderClusterAnalyzer(model, env, algorithm)
    trajectories = analyzer.collect_trajectories(n_episodes=100)
    analyzer.cluster_by_action_preference(n_clusters=10, method="kmeans", use_pca=True)
    analyzer.visualize_clusters_2d(
        save_path=f"lunar_analysis/{algorithm.lower()}_loaded_visualization.png"
    )

    # Analyze transitions for PPO
    print("\n" + "=" * 80)
    print("PPO TRANSITION ANALYSIS")
    print("=" * 80)
    ppo_trans_analyzer = analyze_cluster_transitions(model, analyzer, trajectories, env)

    # Compare flow probabilities for specific paths
    print("\n" + "=" * 80)
    print("PATH PROBABILITY COMPARISON")
    print("=" * 80)

    # Find interesting paths (e.g., start to landing)
    start_cluster = analyzer.state_to_cluster.get(0)  # Assuming state 0 is start
    goal_cluster = analyzer.state_to_cluster.get(max(analyzer.state_to_cluster.keys()))

    if start_cluster is not None and goal_cluster is not None:
        path = [start_cluster, goal_cluster]  # Simplified, would need actual path

        ppo_path_probs = ppo_trans_analyzer.compute_path_probabilities(path)

        print(f"\nPath C{start_cluster} → C{goal_cluster}:")
        print(f"  PPO theoretical: {ppo_path_probs['theoretical']:.4f}")
        print(f"  PPO empirical: {ppo_path_probs['empirical']:.4f}")

    return analyzer


if __name__ == "__main__":
    # Install required packages if not already installed
    # pip install stable-baselines3[extra] gymnasium[box2d] scikit-learn

    # Run comparison on v3
    # results = run_lunar_lander_comparison()

    # Example: Load and visualize a saved model
    analyzer = load_and_visualize_saved_model("models/ppo_lunar_lander_v3.zip", "PPO")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(
        "\nCheck the 'lunar_analysis/' directory for detailed results and visualizations."
    )
