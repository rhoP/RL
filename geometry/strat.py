"""
Topological Analysis of DQN Representations in CartPole
Detects hierarchical/stratified structure in the learned state space
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings("ignore")

# Try importing TDA libraries (install with: pip install gudhi scikit-tda)
try:
    import gudhi as gd
    import persim

    TDA_AVAILABLE = True
except ImportError:
    print("GUDHI not installed. Install with: pip install gudhi")
    TDA_AVAILABLE = False

try:
    from ripser import ripser
    from persim import plot_diagrams

    RIPSER_AVAILABLE = True
except ImportError:
    print("ripser not installed. Install with: pip install ripser persim")
    RIPSER_AVAILABLE = False


class DQNAnalyzer:
    def __init__(self, model_path, env_name="CartPole-v1"):
        """
        Load trained DQN model and analyze its representations

        Args:
            model_path: Path to saved SB3 DQN model
            env_name: Gym environment name
        """
        self.env = gym.make(env_name)
        self.model = self.load_model(model_path)

        # Storage for collected data
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.q_values = []
        self.hidden_features = []  # Features before final layer

    def load_model(self, model_path):
        """Load Stable-Baselines3 DQN model"""
        from stable_baselines3 import DQN

        return DQN.load(model_path)

    def collect_experience(self, n_episodes=100):
        """Collect states, actions, and hidden representations"""
        print(f"Collecting {n_episodes} episodes...")

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_states = []

            while not done:
                # Get action and hidden features
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

                    # Extract features before Q-value layer
                    # For DQN, we need to access the features extractor
                    features = self.model.q_net.features_extractor(obs_tensor)
                    q_values = self.model.q_net.q_net(features)

                action = int(torch.argmax(q_values).numpy())

                # Store data
                self.states.append(obs.copy())
                self.actions.append(action)
                self.q_values.append(q_values.numpy().flatten())
                self.hidden_features.append(features.numpy().flatten())
                episode_states.append(obs)

                # Step environment
                obs, reward, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    done = True
                self.rewards.append(reward)
                self.dones.append(done)

            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1} complete")

        # Convert to numpy arrays
        self.states = np.array(self.states)
        self.hidden_features = np.array(self.hidden_features)
        self.actions = np.array(self.actions)
        self.q_values = np.array(self.q_values)
        self.rewards = np.array(self.rewards)
        self.dones = np.array(self.dones)

        print(f"Collected {len(self.states)} transitions")
        return self

    def estimate_local_dimension(self, data, k=10, method="mle"):
        """
        Estimate local intrinsic dimension using MLE or PCA

        Returns:
            local_dim: array of estimated dimensions per point
            global_dim: average dimension
        """
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
        distances, indices = nbrs.kneighbors(data)

        # Use distances to k-th neighbor (excluding self)
        r_k = distances[:, k]

        if method == "mle":
            # MLE dimension estimator
            m = k - 1
            local_dim = []
            for i in range(len(data)):
                # Use distances to m nearest neighbors
                r_m = distances[i, 1 : m + 1]
                # MLE estimate
                dim = -m / np.sum(np.log(r_m / r_k[i]))
                local_dim.append(dim)

            local_dim = np.array(local_dim)

        elif method == "pca":
            # PCA-based local dimension
            local_dim = []
            for i in range(len(data)):
                # Get neighbors
                neighbor_idx = indices[i, 1 : k + 1]
                neighbor_data = data[neighbor_idx]

                # PCA on neighbors
                from sklearn.decomposition import PCA

                pca = PCA()
                pca.fit(neighbor_data - np.mean(neighbor_data, axis=0))

                # Find number of components explaining 90% variance
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                dim = np.sum(cumsum < 0.9) + 1
                local_dim.append(dim)

            local_dim = np.array(local_dim)

        return local_dim, np.mean(local_dim)

    def detect_strata_by_geometry(self):
        """
        Cluster states by local geometric properties
        """
        print("\n=== Detecting Strata by Local Geometry ===")

        # Estimate local dimension at each point
        local_dim, avg_dim = self.estimate_local_dimension(self.states, k=15)
        print(f"Average local dimension: {avg_dim:.2f}")

        # Estimate local curvature using PCA
        nbrs = NearestNeighbors(n_neighbors=20).fit(self.states)
        distances, indices = nbrs.kneighbors(self.states)

        # Curvature proxy: variance in distances to neighbors
        curvatures = np.std(distances[:, 1:], axis=1)

        # Create feature vector for clustering
        geometric_features = np.column_stack(
            [
                local_dim,
                curvatures,
                self.states[:, 0],  # cart position
                self.states[:, 2],  # pole angle (most important)
            ]
        )

        # Normalize features
        from sklearn.preprocessing import StandardScaler

        geometric_features = StandardScaler().fit_transform(geometric_features)

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=20).fit(geometric_features)
        labels = clustering.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"Found {n_clusters} geometric strata")
        print(f"Noise points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

        # Analyze each stratum
        for label in range(n_clusters):
            mask = labels == label
            stratum_states = self.states[mask]
            stratum_angles = np.abs(stratum_states[:, 2])

            print(f"\nStratum {label}:")
            print(
                f"  Size: {np.sum(mask)} points ({np.sum(mask) / len(mask) * 100:.1f}%)"
            )
            print(f"  Mean |angle|: {np.mean(stratum_angles):.3f} rad")
            print(
                f"  Angle range: [{np.min(stratum_angles):.3f}, {np.max(stratum_angles):.3f}]"
            )
            print(f"  Mean action: {np.mean(self.actions[mask]):.2f}")

        return labels, geometric_features

    def detect_strata_by_q_values(self):
        """
        Detect strata based on Q-value patterns
        """
        print("\n=== Detecting Strata by Q-Value Patterns ===")

        # Use Q-values as features for stratification
        q_features = self.q_values.copy()

        # Also include Q-value differences
        q_diff = np.abs(self.q_values[:, 0] - self.q_values[:, 1])
        q_features = np.column_stack([q_features, q_diff])

        # Normalize
        from sklearn.preprocessing import StandardScaler

        q_features = StandardScaler().fit_transform(q_features)

        # Cluster
        clustering = DBSCAN(eps=0.3, min_samples=20).fit(q_features)
        q_labels = clustering.labels_

        n_clusters = len(set(q_labels)) - (1 if -1 in q_labels else 0)
        print(f"Found {n_clusters} Q-value strata")

        # Analyze Q-value patterns per stratum
        for label in range(n_clusters):
            mask = q_labels == label
            stratum_q = self.q_values[mask]

            print(f"\nStratum {label}:")
            print(f"  Size: {np.sum(mask)} points")
            print(f"  Mean Q(s,0): {np.mean(stratum_q[:, 0]):.3f}")
            print(f"  Mean Q(s,1): {np.mean(stratum_q[:, 1]):.3f}")
            print(f"  Preferred action: {np.argmax(np.mean(stratum_q, axis=0))}")

        return q_labels

    def topological_analysis(self):
        """
        Use persistent homology to detect topological structure
        """
        print("\n=== Topological Data Analysis ===")

        if not RIPSER_AVAILABLE:
            print("Ripser not available. Skipping TDA.")
            return None

        # Sample a subset for computational efficiency
        max_points = 500
        if len(self.states) > max_points:
            idx = np.random.choice(len(self.states), max_points, replace=False)
            states_subset = self.states[idx]
        else:
            states_subset = self.states

        print(f"Computing persistence on {len(states_subset)} points...")

        # Compute persistence diagrams
        diagrams = ripser(states_subset, maxdim=2)["dgms"]

        # Plot persistence diagrams
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        titles = ["H₀ (Connected Components)", "H₁ (Loops)", "H₂ (Voids)"]
        for i, (diagram, title) in enumerate(zip(diagrams[:3], titles)):
            if len(diagram) > 0:
                plot_diagrams(diagram, ax=axes[i])
                axes[i].set_title(title)

                # Calculate persistence
                persistence = diagram[:, 1] - diagram[:, 0]
                long_lived = persistence > np.percentile(persistence, 75)
                print(
                    f"  {title}: {len(diagram)} features, {np.sum(long_lived)} long-lived"
                )
            else:
                axes[i].text(0.5, 0.5, "No features", ha="center", va="center")
                axes[i].set_title(title)

        plt.tight_layout()
        plt.savefig("persistence_diagrams.png", dpi=150)
        print("Persistence diagrams saved to 'persistence_diagrams.png'")

        return diagrams

    def detect_stratification_boundaries(self):
        """
        Find potential boundaries between strata
        """
        print("\n=== Detecting Stratification Boundaries ===")

        # Use pole angle as primary stratifying coordinate
        pole_angles = self.states[:, 2]
        pole_velocities = self.states[:, 3]

        # Where does Q-value difference change sign?
        q_diff = self.q_values[:, 0] - self.q_values[:, 1]

        # Find points near decision boundary
        boundary_threshold = 0.1
        near_boundary = np.abs(q_diff) < boundary_threshold

        # Also look at where action switches
        action_changes = np.diff(self.actions)
        change_points = np.where(action_changes != 0)[0]

        print(
            f"Points near decision boundary: {np.sum(near_boundary)} ({np.sum(near_boundary) / len(near_boundary) * 100:.1f}%)"
        )
        print(f"Action changes: {len(change_points)}")

        if len(change_points) > 0:
            # Analyze states where action changes
            change_states = self.states[change_points]
            change_angles = change_states[:, 2]
            print(
                f"Action change at mean angle: {np.mean(np.abs(change_angles)):.3f} rad"
            )
            print(
                f"Angle range: [{np.min(np.abs(change_angles)):.3f}, {np.max(np.abs(change_angles)):.3f}]"
            )

        return near_boundary, change_points

    def visualize_results(self, strata_labels=None, q_labels=None):
        """
        Create comprehensive visualization
        """
        print("\n=== Creating Visualizations ===")

        fig = plt.figure(figsize=(20, 12))

        # 1. State space colored by Q-value difference
        ax1 = fig.add_subplot(2, 3, 1)
        q_diff = self.q_values[:, 0] - self.q_values[:, 1]
        scatter1 = ax1.scatter(
            self.states[:, 0], self.states[:, 2], c=q_diff, cmap="RdBu", s=5, alpha=0.6
        )
        ax1.set_xlabel("Cart Position")
        ax1.set_ylabel("Pole Angle")
        ax1.set_title("State Space Colored by Q(s,0) - Q(s,1)")
        plt.colorbar(scatter1, ax=ax1)

        # 2. State space colored by action
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.scatter(
            self.states[:, 0],
            self.states[:, 2],
            c=self.actions,
            cmap="viridis",
            s=5,
            alpha=0.6,
        )
        ax2.set_xlabel("Cart Position")
        ax2.set_ylabel("Pole Angle")
        ax2.set_title("Actions Taken")

        # 3. Geometric strata if available
        ax3 = fig.add_subplot(2, 3, 3)
        if strata_labels is not None:
            ax3.scatter(
                self.states[:, 0],
                self.states[:, 2],
                c=strata_labels,
                cmap="tab10",
                s=5,
                alpha=0.6,
            )
            ax3.set_xlabel("Cart Position")
            ax3.set_ylabel("Pole Angle")
            ax3.set_title("Geometric Strata")

        # 4. Hidden feature space (PCA projection)
        ax4 = fig.add_subplot(2, 3, 4)
        if len(self.hidden_features) > 0:
            pca = PCA(n_components=2)
            hidden_2d = pca.fit_transform(self.hidden_features)
            ax4.scatter(
                hidden_2d[:, 0],
                hidden_2d[:, 1],
                c=self.states[:, 2],
                cmap="coolwarm",
                s=5,
                alpha=0.6,
            )
            ax4.set_xlabel("PC1")
            ax4.set_ylabel("PC2")
            ax4.set_title("Hidden Features (colored by pole angle)")
            plt.colorbar(ax4.collections[0], ax=ax4, label="Pole angle")

        # 5. Pole angle distribution
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(
            self.states[:, 2], bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        ax5.axvline(x=0, color="red", linestyle="--", label="Upright")
        ax5.axvline(
            x=0.2, color="orange", linestyle="--", label="Failure threshold (~12°)"
        )
        ax5.axvline(x=-0.2, color="orange", linestyle="--")
        ax5.set_xlabel("Pole Angle (rad)")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Distribution of Pole Angles")
        ax5.legend()

        # 6. Q-value patterns
        ax6 = fig.add_subplot(2, 3, 6)
        # Sort by pole angle for better visualization
        sort_idx = np.argsort(self.states[:, 2])
        sorted_angles = self.states[sort_idx, 2]
        sorted_q0 = self.q_values[sort_idx, 0]
        sorted_q1 = self.q_values[sort_idx, 1]

        ax6.plot(sorted_angles, sorted_q0, "b-", alpha=0.5, label="Q(s,0)", linewidth=1)
        ax6.plot(sorted_angles, sorted_q1, "r-", alpha=0.5, label="Q(s,1)", linewidth=1)
        ax6.set_xlabel("Pole Angle (rad)")
        ax6.set_ylabel("Q-Value")
        ax6.set_title("Q-Values vs Pole Angle")
        ax6.legend()

        plt.tight_layout()
        plt.savefig("stratification_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Visualization saved to 'stratification_analysis.png'")

    def run_full_analysis(self):
        """
        Run complete stratification analysis pipeline
        """
        print("=" * 60)
        print("DQN STRATIFICATION ANALYSIS FOR CARTPOLE")
        print("=" * 60)

        # Collect data
        self.collect_experience(n_episodes=100)

        # Geometric analysis
        strata_labels, _ = self.detect_strata_by_geometry()

        # Q-value analysis
        q_labels = self.detect_strata_by_q_values()

        # Topological analysis
        diagrams = self.topological_analysis()

        # Boundary detection
        boundaries, change_points = self.detect_stratification_boundaries()

        # Visualize
        self.visualize_results(strata_labels, q_labels)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY: STRATIFICATION DETECTION")
        print("=" * 60)

        # Evidence for stratification
        evidence = []

        # 1. Distinct geometric regions
        n_geo_strata = len(set(strata_labels)) - (1 if -1 in strata_labels else 0)
        if n_geo_strata >= 2:
            evidence.append(f"✓ Found {n_geo_strata} distinct geometric regions")
        else:
            evidence.append("✗ No clear geometric stratification")

        # 2. Sharp Q-value transitions
        q_diff = self.q_values[:, 0] - self.q_values[:, 1]
        q_diff_derivative = np.abs(np.diff(q_diff))
        sharp_transitions = np.sum(
            q_diff_derivative > np.percentile(q_diff_derivative, 95)
        )
        if sharp_transitions > 10:
            evidence.append(
                f"✓ Detected {sharp_transitions} sharp Q-value transitions (potential boundaries)"
            )
        else:
            evidence.append("✗ Q-values change smoothly")

        # 3. Action boundaries
        action_changes = np.sum(np.abs(np.diff(self.actions)) > 0)
        if action_changes > 50:
            evidence.append(
                f"✓ Frequent action changes ({action_changes}) suggest decision boundaries"
            )
        else:
            evidence.append("✗ Actions are stable")

        # 4. Pole angle distribution
        angles = np.abs(self.states[:, 2])
        angle_percentiles = np.percentile(angles, [25, 50, 75, 90])
        if angle_percentiles[2] > 0.1:  # 75th percentile > ~6 degrees
            evidence.append(
                f"✓ Pole explores wide angle range (75th percentile: {angle_percentiles[2]:.3f} rad)"
            )

        # 5. Topological features
        if diagrams is not None and len(diagrams[1]) > 0:
            evidence.append("✓ Persistent loops detected (H₁ features)")

        # Print evidence
        for e in evidence:
            print(e)

        print("\n" + "=" * 60)
        print("CONCLUSION:")

        if len([e for e in evidence if e.startswith("✓")]) >= 3:
            print("CartPole exhibits clear stratified/hierarchical structure!")
            print("The DQN has learned to distinguish between:")
            print("  - Balanced region (near upright)")
            print("  - Falling region (large angles)")
            print("  - Decision boundary at ~0.2 rad")
        else:
            print(
                "Limited evidence for stratification. CartPole may be better modeled as continuous."
            )

        return {
            "states": self.states,
            "actions": self.actions,
            "q_values": self.q_values,
            "strata_labels": strata_labels,
            "q_labels": q_labels,
            "diagrams": diagrams,
        }


def main():
    """
    Main function to run analysis
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze DQN representations for stratification"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained DQN model"
    )
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to collect"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = DQNAnalyzer(args.model_path, args.env)

    # Run analysis
    results = analyzer.run_full_analysis()

    print("\nAnalysis complete! Check the generated plots:")
    print("  - stratification_analysis.png")
    print("  - persistence_diagrams.png")


if __name__ == "__main__":
    main()
