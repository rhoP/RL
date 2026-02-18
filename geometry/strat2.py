"""
Generalized Stratification Analysis for Learned Policies
Works with any gym environment and any SB3 model type
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")

# Core dependencies
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

# Optional TDA libraries
try:
    import gudhi as gd
    import persim

    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

try:
    from ripser import ripser
    from persim import plot_diagrams

    RIPSER_AVAILABLE = False  # Set to True if you have ripser
except ImportError:
    RIPSER_AVAILABLE = False


@dataclass
class StratificationConfig:
    """Configuration for stratification analysis"""

    # Data collection
    n_episodes: int = 100
    max_steps_per_episode: Optional[int] = None

    # Analysis methods
    use_geometric_clustering: bool = True
    use_q_value_analysis: bool = True
    use_topological_analysis: bool = True
    use_dynamics_analysis: bool = True

    # Clustering parameters
    clustering_method: str = "hdbscan"  # 'dbscan', 'hdbscan', 'gmm'
    min_cluster_size: int = 20
    eps: float = 0.3

    # Local dimension estimation
    dim_estimation_method: str = "mle"  # 'mle', 'pca'
    n_neighbors_dim: int = 15

    # Topological analysis
    max_points_tda: int = 500  # Sample for computational efficiency

    # Visualization
    save_plots: bool = True
    plot_dir: str = "./stratification_analysis"


class FeatureExtractor(ABC):
    """Abstract base for extracting features from a learned policy"""

    @abstractmethod
    def get_hidden_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract hidden layer representations"""
        pass

    @abstractmethod
    def get_action_values(self, obs: np.ndarray) -> np.ndarray:
        """Get action values/Q-values for observations"""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Return model type (DQN, PPO, etc.)"""
        pass


class SB3FeatureExtractor(FeatureExtractor):
    """Feature extractor for Stable-Baselines3 models"""

    def __init__(self, model):
        self.model = model
        self.model_type = type(model).__name__

    def get_hidden_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract features before the final layer"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            # Handle different model architectures
            if hasattr(self.model, "policy"):
                policy = self.model.policy
            else:
                policy = self.model

            # Extract features based on model type
            if self.model_type == "DQN":
                if hasattr(policy, "q_net") and hasattr(
                    policy.q_net, "features_extractor"
                ):
                    features = policy.q_net.features_extractor(obs_tensor)
                else:
                    features = obs_tensor  # Fallback

            elif self.model_type in ["PPO", "A2C", "SAC"]:
                if hasattr(policy, "mlp_extractor"):
                    # Get features from shared network
                    if hasattr(policy.mlp_extractor, "shared_net"):
                        features = policy.mlp_extractor.shared_net(obs_tensor)
                    else:
                        # For policies with separate value/action networks
                        latent = policy.mlp_extractor(obs_tensor)
                        if isinstance(latent, tuple):
                            features = latent[0]  # Use policy features
                        else:
                            features = latent
                else:
                    features = obs_tensor
            else:
                features = obs_tensor

            return features.cpu().numpy()

    def get_action_values(self, obs: np.ndarray) -> np.ndarray:
        """Get action values/Q-values"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            if self.model_type == "DQN":
                # Get Q-values
                if hasattr(self.model, "q_net"):
                    features = self.model.q_net.features_extractor(obs_tensor)
                    values = self.model.q_net.q_net(features)
                else:
                    values = torch.zeros((obs_tensor.shape[0], 2))  # Fallback

            elif self.model_type in ["PPO", "A2C"]:
                # Get action probabilities or values
                if hasattr(self.model, "policy"):
                    dist = self.model.policy.get_distribution(obs_tensor)
                    if hasattr(dist, "probs"):
                        values = dist.probs
                    else:
                        values = (
                            dist.distribution.probs
                            if hasattr(dist, "distribution")
                            else None
                        )
                else:
                    values = None
            else:
                values = None

            if values is not None:
                return values.cpu().numpy()
            return None

    def get_model_type(self) -> str:
        return self.model_type


class StratificationAnalyzer:
    """
    Main class for analyzing stratification in learned policies
    """

    def __init__(
        self,
        env: gym.Env,
        feature_extractor: FeatureExtractor,
        config: StratificationConfig,
    ):
        """
        Args:
            env: Gym environment
            feature_extractor: Extracts features from model
            config: Analysis configuration
        """
        self.env = env
        self.feature_extractor = feature_extractor
        self.config = config

        # Storage for collected data
        self.data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "hidden_features": [],
            "action_values": [],
            "episode_ids": [],
            "timesteps": [],
        }

        # Create output directory
        if config.save_plots:
            import os

            os.makedirs(config.plot_dir, exist_ok=True)

    def collect_data(self):
        """Collect trajectories from the policy"""
        print(f"\n=== Collecting Data ===")
        print(f"Environment: {self.env.spec.id if self.env.spec else 'Unknown'}")
        print(f"Model Type: {self.feature_extractor.get_model_type()}")
        print(f"Episodes: {self.config.n_episodes}")

        episode_rewards = []

        for episode in range(self.config.n_episodes):
            obs, _ = self.env.reset()
            if isinstance(obs, tuple):  # Handle newer gym API
                obs = obs[0]

            done = False
            truncated = False
            episode_reward = 0
            step = 0

            while not (done or truncated):
                # Get model outputs
                hidden_features = self.feature_extractor.get_hidden_features(obs)
                action_values = self.feature_extractor.get_action_values(obs)

                # Get action
                if hasattr(self.feature_extractor.model, "predict"):
                    action, _ = self.feature_extractor.model.predict(
                        obs, deterministic=True
                    )
                else:
                    # Fallback: sample from action values
                    if action_values is not None:
                        action = np.argmax(action_values[0])
                    else:
                        action = self.env.action_space.sample()

                # Store data
                self.data["observations"].append(
                    obs.copy() if hasattr(obs, "copy") else obs
                )
                self.data["actions"].append(action)
                self.data["hidden_features"].append(
                    hidden_features[0]
                    if len(hidden_features.shape) > 1
                    else hidden_features
                )
                if action_values is not None:
                    self.data["action_values"].append(
                        action_values[0]
                        if len(action_values.shape) > 1
                        else action_values
                    )
                self.data["episode_ids"].append(episode)
                self.data["timesteps"].append(step)

                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)

                self.data["rewards"].append(reward)
                self.data["dones"].append(done or truncated)

                episode_reward += reward
                step += 1

                if (
                    self.config.max_steps_per_episode
                    and step >= self.config.max_steps_per_episode
                ):
                    break

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}: Reward = {episode_reward:.1f}")

        # Convert to numpy arrays
        for key in self.data:
            if self.data[key] and isinstance(
                self.data[key][0], (int, float, np.number)
            ):
                self.data[key] = np.array(self.data[key])
            elif self.data[key] and hasattr(self.data[key][0], "__array__"):
                self.data[key] = np.array([x.flatten() for x in self.data[key]])

        print(f"\nCollected {len(self.data['observations'])} transitions")
        print(
            f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
        )

        return self

    def estimate_local_dimension(
        self, data: np.ndarray, method: str = "mle"
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate local intrinsic dimension

        Args:
            data: Point cloud
            method: 'mle' or 'pca'

        Returns:
            local_dim: Per-point dimension estimates
            global_dim: Average dimension
        """
        nbrs = NearestNeighbors(n_neighbors=self.config.n_neighbors_dim + 1).fit(data)
        distances, indices = nbrs.kneighbors(data)

        # Use distances to k-th neighbor (excluding self)
        r_k = distances[:, self.config.n_neighbors_dim]

        if method == "mle":
            # MLE dimension estimator
            m = self.config.n_neighbors_dim - 1
            local_dim = []
            for i in range(len(data)):
                # Use distances to m nearest neighbors
                r_m = distances[i, 1 : m + 1]
                # MLE estimate
                dim = -m / np.sum(np.log(r_m / r_k[i] + 1e-10))
                local_dim.append(dim)

            local_dim = np.array(local_dim)
            local_dim = np.clip(local_dim, 0, data.shape[1])

        elif method == "pca":
            # PCA-based local dimension
            local_dim = []
            for i in range(len(data)):
                neighbor_idx = indices[i, 1 : self.config.n_neighbors_dim + 1]
                neighbor_data = data[neighbor_idx]

                pca = PCA()
                pca.fit(neighbor_data - np.mean(neighbor_data, axis=0))

                # Find number of components explaining 90% variance
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                dim = np.sum(cumsum < 0.9) + 1
                local_dim.append(dim)

            local_dim = np.array(local_dim)

        return local_dim, np.mean(local_dim)

    def geometric_stratification(self) -> Dict[str, Any]:
        """
        Detect strata based on local geometric properties
        """
        print("\n=== Geometric Stratification ===")

        obs = self.data["observations"]

        # Estimate local properties
        local_dim, avg_dim = self.estimate_local_dimension(
            obs, method=self.config.dim_estimation_method
        )
        print(f"Average local dimension: {avg_dim:.2f}")

        # Estimate local curvature proxy
        nbrs = NearestNeighbors(n_neighbors=min(20, len(obs))).fit(obs)
        distances, _ = nbrs.kneighbors(obs)
        curvatures = np.std(distances[:, 1:], axis=1)

        # Create feature vector for clustering
        geometric_features = np.column_stack(
            [
                local_dim,
                curvatures / (np.max(curvatures) + 1e-10),
            ]
        )

        # Add observation dimensions if not too many
        if obs.shape[1] <= 10:
            geometric_features = np.column_stack(
                [geometric_features, StandardScaler().fit_transform(obs)]
            )

        # Normalize
        geometric_features = StandardScaler().fit_transform(geometric_features)

        # Cluster
        labels = self._cluster_data(geometric_features)

        # Analyze clusters
        cluster_stats = self._analyze_clusters(labels, obs)

        return {
            "labels": labels,
            "features": geometric_features,
            "local_dim": local_dim,
            "curvatures": curvatures,
            "stats": cluster_stats,
        }

    def value_function_stratification(self) -> Dict[str, Any]:
        """
        Detect strata based on value function patterns
        """
        print("\n=== Value Function Stratification ===")

        if not self.data["action_values"] or len(self.data["action_values"]) == 0:
            print("No action values available")
            return None

        action_values = np.array(self.data["action_values"])

        # Create features from value function
        value_features = []

        # 1. Raw values
        value_features.append(action_values)

        # 2. Value differences
        if action_values.shape[1] > 1:
            value_diff = np.abs(action_values[:, 0] - action_values[:, 1])
            value_features.append(value_diff.reshape(-1, 1))

        # 3. Value entropy (uncertainty)
        if action_values.shape[1] > 1:
            probs = np.exp(action_values) / np.sum(
                np.exp(action_values), axis=1, keepdims=True
            )
            value_entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            value_features.append(value_entropy.reshape(-1, 1))

        value_features = np.hstack(value_features)
        value_features = StandardScaler().fit_transform(value_features)

        # Cluster
        labels = self._cluster_data(value_features)

        # Analyze clusters
        cluster_stats = {}
        for label in np.unique(labels):
            if label == -1:
                continue
            mask = labels == label
            cluster_stats[f"stratum_{label}"] = {
                "size": np.sum(mask),
                "mean_values": np.mean(action_values[mask], axis=0),
                "value_range": np.ptp(action_values[mask], axis=0),
            }
            print(f"\nStratum {label}:")
            print(f"  Size: {np.sum(mask)} points")
            print(f"  Mean values: {np.mean(action_values[mask], axis=0)}")

        return {"labels": labels, "features": value_features, "stats": cluster_stats}

    def dynamics_stratification(self) -> Dict[str, Any]:
        """
        Detect strata based on transition dynamics
        """
        print("\n=== Dynamics Stratification ===")

        obs = self.data["observations"]
        next_obs = np.roll(obs, -1, axis=0)
        actions = self.data["actions"]

        # Remove last point (no next state)
        obs = obs[:-1]
        next_obs = next_obs[:-1]
        actions = actions[:-1]

        # Compute dynamics features
        # 1. State change magnitude
        state_change = np.linalg.norm(next_obs - obs, axis=1)

        # 2. Directional change
        if obs.shape[1] > 1:
            direction_change = np.arccos(
                np.clip(
                    np.sum(obs * next_obs, axis=1)
                    / (
                        np.linalg.norm(obs, axis=1) * np.linalg.norm(next_obs, axis=1)
                        + 1e-10
                    ),
                    -1,
                    1,
                )
            )
        else:
            direction_change = np.zeros_like(state_change)

        # 3. Action-conditioned change
        dynamics_features = np.column_stack(
            [
                state_change,
                direction_change,
                actions[:-1],  # Align actions
            ]
        )

        # Normalize
        dynamics_features = StandardScaler().fit_transform(dynamics_features)

        # Cluster
        labels = self._cluster_data(dynamics_features)

        return {
            "labels": labels,
            "features": dynamics_features,
            "state_change": state_change,
            "direction_change": direction_change,
        }

    def topological_analysis(self) -> Optional[Dict]:
        """
        Use persistent homology to detect topological structure
        """
        print("\n=== Topological Data Analysis ===")

        if not RIPSER_AVAILABLE:
            print("Ripser not available. Skipping TDA.")
            return None

        # Sample subset for efficiency
        obs = self.data["observations"]
        if len(obs) > self.config.max_points_tda:
            idx = np.random.choice(len(obs), self.config.max_points_tda, replace=False)
            obs_subset = obs[idx]
        else:
            obs_subset = obs

        print(f"Computing persistence on {len(obs_subset)} points...")

        # Compute persistence diagrams
        diagrams = ripser(obs_subset, maxdim=min(2, obs.shape[1] - 1))["dgms"]

        # Analyze persistence
        results = {"diagrams": diagrams}

        for dim, diagram in enumerate(diagrams):
            if len(diagram) > 0:
                persistence = diagram[:, 1] - diagram[:, 0]
                long_lived = persistence > np.percentile(persistence, 75)
                results[f"H{dim}_features"] = len(diagram)
                results[f"H{dim}_long_lived"] = np.sum(long_lived)
                print(
                    f"  H{dim}: {len(diagram)} features, {np.sum(long_lived)} long-lived"
                )

        return results

    def _cluster_data(self, features: np.ndarray) -> np.ndarray:
        """Cluster data using specified method"""
        if self.config.clustering_method == "dbscan":
            clusterer = DBSCAN(
                eps=self.config.eps, min_samples=self.config.min_cluster_size
            )
        elif self.config.clustering_method == "hdbscan":
            try:
                clusterer = HDBSCAN(min_cluster_size=self.config.min_cluster_size)
            except NameError:
                print("HDBSCAN not installed, falling back to DBSCAN")
                clusterer = DBSCAN(
                    eps=self.config.eps, min_samples=self.config.min_cluster_size
                )
        elif self.config.clustering_method == "gmm":
            # Determine optimal number of components
            best_bic = np.inf
            best_gmm = None
            for n in range(2, min(6, len(features) // 10)):
                gmm = GaussianMixture(n_components=n, random_state=42)
                gmm.fit(features)
                bic = gmm.bic(features)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm

            if best_gmm is not None:
                return best_gmm.predict(features)
            else:
                return np.zeros(len(features)) - 1

        labels = clusterer.fit_predict(features)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"Found {n_clusters} clusters")
        print(f"Noise points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

        return labels

    def _analyze_clusters(self, labels: np.ndarray, observations: np.ndarray) -> Dict:
        """Analyze characteristics of each cluster"""
        stats = {}

        for label in np.unique(labels):
            if label == -1:
                continue

            mask = labels == label
            cluster_obs = observations[mask]

            # Basic statistics
            stats[f"stratum_{label}"] = {
                "size": np.sum(mask),
                "mean": np.mean(cluster_obs, axis=0),
                "std": np.std(cluster_obs, axis=0),
                "range": np.ptp(cluster_obs, axis=0),
            }

            # Print summary
            print(f"\nStratum {label}:")
            print(f"  Size: {np.sum(mask)} ({np.sum(mask) / len(mask) * 100:.1f}%)")

            # Show first few dimensions of mean
            mean_str = ", ".join(
                [f"{x:.3f}" for x in stats[f"stratum_{label}"]["mean"][:3]]
            )
            print(f"  Mean (first 3 dims): [{mean_str}]")

        return stats

    def visualize_all(self, results: Dict):
        """Create comprehensive visualization"""
        print("\n=== Creating Visualizations ===")

        n_plots = 6
        fig = plt.figure(figsize=(20, 4 * ((n_plots + 1) // 2)))

        obs = self.data["observations"]

        # 1. PCA projection of observations
        ax1 = fig.add_subplot(2, 3, 1)
        pca = PCA(n_components=2)
        obs_2d = pca.fit_transform(obs)

        # Color by reward if available
        scatter1 = ax1.scatter(
            obs_2d[:, 0],
            obs_2d[:, 1],
            c=self.data["rewards"],
            cmap="viridis",
            s=5,
            alpha=0.6,
        )
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_title("State Space (PCA, colored by reward)")
        plt.colorbar(scatter1, ax=ax1)

        # 2. Actions over trajectory
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(self.data["actions"][:500], "b-", alpha=0.7, linewidth=0.5)
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Action")
        ax2.set_title("Action Sequence (first 500 steps)")
        ax2.set_ylim([-0.5, self.env.action_space.n - 0.5])

        # 3. Geometric strata if available
        ax3 = fig.add_subplot(2, 3, 3)
        if "geometric" in results and results["geometric"] is not None:
            labels = results["geometric"]["labels"]
            scatter3 = ax3.scatter(
                obs_2d[:, 0], obs_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.6
            )
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC2")
            ax3.set_title("Geometric Strata")

        # 4. Hidden features space
        ax4 = fig.add_subplot(2, 3, 4)
        if len(self.data["hidden_features"]) > 0:
            hidden = np.array(self.data["hidden_features"])
            if hidden.shape[1] > 2:
                pca_hidden = PCA(n_components=2)
                hidden_2d = pca_hidden.fit_transform(hidden)
            else:
                hidden_2d = hidden

            scatter4 = ax4.scatter(
                hidden_2d[:, 0],
                hidden_2d[:, 1],
                c=self.data["rewards"],
                cmap="viridis",
                s=5,
                alpha=0.6,
            )
            ax4.set_xlabel("Component 1")
            ax4.set_ylabel("Component 2")
            ax4.set_title("Hidden Features Space")
            plt.colorbar(scatter4, ax=ax4)

        # 5. Value function landscape
        ax5 = fig.add_subplot(2, 3, 5)
        if len(self.data["action_values"]) > 0:
            values = np.array(self.data["action_values"])
            if values.shape[1] >= 2:
                value_diff = values[:, 0] - values[:, 1]
                scatter5 = ax5.scatter(
                    obs_2d[:, 0],
                    obs_2d[:, 1],
                    c=value_diff,
                    cmap="RdBu",
                    s=5,
                    alpha=0.6,
                )
                ax5.set_xlabel("PC1")
                ax5.set_ylabel("PC2")
                ax5.set_title("Value Difference (Action 0 - Action 1)")
                plt.colorbar(scatter5, ax=ax5)

        # 6. Local dimension distribution
        ax6 = fig.add_subplot(2, 3, 6)
        if "geometric" in results and results["geometric"] is not None:
            local_dim = results["geometric"]["local_dim"]
            ax6.hist(local_dim, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
            ax6.set_xlabel("Local Dimension")
            ax6.set_ylabel("Frequency")
            ax6.set_title("Distribution of Local Dimension")
            ax6.axvline(
                np.mean(local_dim),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(local_dim):.2f}",
            )
            ax6.legend()

        plt.tight_layout()

        if self.config.save_plots:
            plt.savefig(
                f"{self.config.plot_dir}/stratification_analysis.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

        # Additional TDA plot if available
        if "topological" in results and results["topological"] is not None:
            self._plot_persistence(results["topological"])

    def _plot_persistence(self, tda_results: Dict):
        """Plot persistence diagrams"""
        if "diagrams" not in tda_results:
            return

        diagrams = tda_results["diagrams"]
        n_diags = len(diagrams)

        fig, axes = plt.subplots(1, n_diags, figsize=(5 * n_diags, 4))
        if n_diags == 1:
            axes = [axes]

        titles = ["H₀ (Components)", "H₁ (Loops)", "H₂ (Voids)"]

        for i, (diagram, ax) in enumerate(zip(diagrams, axes)):
            if i < len(titles):
                if len(diagram) > 0:
                    plot_diagrams(diagram, ax=ax)
                    ax.set_title(titles[i])
                else:
                    ax.text(0.5, 0.5, "No features", ha="center", va="center")
                    ax.set_title(titles[i])

        plt.tight_layout()

        if self.config.save_plots:
            plt.savefig(
                f"{self.config.plot_dir}/persistence_diagrams.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

    def run_analysis(self) -> Dict:
        """Run complete stratification analysis pipeline"""
        print("=" * 70)
        print("GENERALIZED STRATIFICATION ANALYSIS FOR LEARNED POLICIES")
        print("=" * 70)

        # Collect data
        self.collect_data()

        results = {}

        # Run selected analyses
        if self.config.use_geometric_clustering:
            results["geometric"] = self.geometric_stratification()

        if self.config.use_q_value_analysis:
            results["value_function"] = self.value_function_stratification()

        if self.config.use_dynamics_analysis:
            results["dynamics"] = self.dynamics_stratification()

        if self.config.use_topological_analysis:
            results["topological"] = self.topological_analysis()

        # Visualize
        self.visualize_all(results)

        # Summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """Print summary of findings"""
        print("\n" + "=" * 70)
        print("SUMMARY: STRATIFICATION DETECTION")
        print("=" * 70)

        evidence = []

        # Geometric evidence
        if "geometric" in results and results["geometric"] is not None:
            labels = results["geometric"]["labels"]
            n_strata = len(set(labels)) - (1 if -1 in labels else 0)
            if n_strata >= 2:
                evidence.append(f"✓ Geometric analysis: {n_strata} distinct strata")
            else:
                evidence.append("✗ No clear geometric stratification")

        # Value function evidence
        if "value_function" in results and results["value_function"] is not None:
            labels = results["value_function"]["labels"]
            if labels is not None:
                n_val_strata = len(set(labels)) - (1 if -1 in labels else 0)
                if n_val_strata >= 2:
                    evidence.append(
                        f"✓ Value function: {n_val_strata} distinct regions"
                    )

        # Dynamics evidence
        if "dynamics" in results and results["dynamics"] is not None:
            labels = results["dynamics"]["labels"]
            if labels is not None:
                n_dyn_strata = len(set(labels)) - (1 if -1 in labels else 0)
                if n_dyn_strata >= 2:
                    evidence.append(
                        f"✓ Dynamics: {n_dyn_strata} distinct transition patterns"
                    )

        # Topological evidence
        if "topological" in results and results["topological"] is not None:
            for dim in range(3):
                key = f"H{dim}_long_lived"
                if key in results["topological"] and results["topological"][key] > 0:
                    evidence.append(f"✓ Topology: persistent H{dim} features detected")

        # Print evidence
        for e in evidence:
            print(e)

        # Conclusion
        print("\n" + "=" * 70)
        print("CONCLUSION:")

        evidence_score = len([e for e in evidence if e.startswith("✓")])
        total_checks = len(
            [
                k
                for k in ["geometric", "value_function", "dynamics", "topological"]
                if k in results and results[k] is not None
            ]
        )

        if total_checks == 0:
            print("No analyses completed successfully.")
        elif evidence_score >= total_checks * 0.5:
            print("The environment exhibits CLEAR STRATIFIED/HIERARCHICAL STRUCTURE!")
            print("\nThe learned policy has discovered distinct regions in state space")
            print("with different geometric properties, value patterns, or dynamics.")
        elif evidence_score > 0:
            print("The environment shows SOME EVIDENCE of stratification.")
            print("\nThere are hints of distinct regions, but the boundaries may be")
            print("smooth rather than sharp.")
        else:
            print("The environment appears CONTINUOUS with no clear stratification.")
            print(
                "\nThe state space is likely smoothly varying without sharp boundaries."
            )

        print("=" * 70)


# Example usage function
def analyze_policy(
    model_path: str,
    env_name: str,
    model_type: str = "auto",
    n_episodes: int = 100,
    **kwargs,
):
    """
    Convenience function to analyze a trained policy

    Args:
        model_path: Path to saved model
        env_name: Gym environment name
        model_type: 'DQN', 'PPO', 'A2C', 'SAC', or 'auto'
        n_episodes: Number of episodes to collect
        **kwargs: Additional config parameters
    """
    # Load environment
    env = gym.make(env_name)

    # Load model based on type
    if model_type == "auto":
        # Try to infer from file name or content
        if "dqn" in model_path.lower():
            model_type = "DQN"
        elif "ppo" in model_path.lower():
            model_type = "PPO"
        elif "a2c" in model_path.lower():
            model_type = "A2C"
        elif "sac" in model_path.lower():
            model_type = "SAC"
        else:
            model_type = "DQN"  # Default

    # Import appropriate SB3 model
    if model_type == "DQN":
        from stable_baselines3 import DQN

        model = DQN.load(model_path)
    elif model_type == "PPO":
        from stable_baselines3 import PPO

        model = PPO.load(model_path)
    elif model_type == "A2C":
        from stable_baselines3 import A2C

        model = A2C.load(model_path)
    elif model_type == "SAC":
        from stable_baselines3 import SAC

        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create feature extractor
    extractor = SB3FeatureExtractor(model)

    # Create config
    config = StratificationConfig(n_episodes=n_episodes, **kwargs)

    # Run analysis
    analyzer = StratificationAnalyzer(env, extractor, config)
    results = analyzer.run_analysis()

    return results, analyzer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze policy for stratification")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "DQN", "PPO", "A2C", "SAC"],
        help="Type of model",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "geometric", "value", "dynamics", "topological"],
        help="Analysis method to run",
    )

    args = parser.parse_args()

    # Map method to config flags
    method_flags = {
        "geometric": {
            "use_geometric_clustering": True,
            "use_q_value_analysis": False,
            "use_dynamics_analysis": False,
            "use_topological_analysis": False,
        },
        "value": {
            "use_geometric_clustering": False,
            "use_q_value_analysis": True,
            "use_dynamics_analysis": False,
            "use_topological_analysis": False,
        },
        "dynamics": {
            "use_geometric_clustering": False,
            "use_q_value_analysis": False,
            "use_dynamics_analysis": True,
            "use_topological_analysis": False,
        },
        "topological": {
            "use_geometric_clustering": False,
            "use_q_value_analysis": False,
            "use_dynamics_analysis": False,
            "use_topological_analysis": True,
        },
        "all": {},
    }

    flags = method_flags[args.method]

    # Run analysis
    results, analyzer = analyze_policy(
        model_path=args.model_path,
        env_name=args.env,
        model_type=args.model_type,
        n_episodes=args.episodes,
        **flags,
    )
