import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import gudhi as gd
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional
from ripser import Rips
import persim
from persim import plot_diagrams


class RLTrajectoryStorage:
    def __init__(self, max_trajectories=1000):
        self.max_trajectories = max_trajectories
        self.trajectories = []
        self.state_dim = None
        self.action_dim = None

    def add_trajectory(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
        infos: Optional[List[Dict]] = None,
    ):
        """Store a complete trajectory"""
        trajectory = {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "infos": infos if infos is not None else [],
            "length": len(states) - 1,  # states include initial state
            "total_reward": np.sum(rewards),
        }

        if self.state_dim is None and len(states) > 0:
            self.state_dim = (
                states[0].shape[0] if hasattr(states[0], "shape") else len(states[0])
            )
        if self.action_dim is None and len(actions) > 0:
            self.action_dim = (
                actions[0].shape[0] if hasattr(actions[0], "shape") else len(actions[0])
            )

        self.trajectories.append(trajectory)

        if len(self.trajectories) > self.max_trajectories:
            self.trajectories = self.trajectories[-self.max_trajectories :]

    def get_state_action_pairs(self, max_samples=10000) -> np.ndarray:
        """Extract state-action pairs from all trajectories"""
        state_action_pairs = []

        for traj in self.trajectories:
            states = traj["states"][
                :-1
            ]  # Exclude final state (no action taken from it)
            actions = traj["actions"]

            min_len = min(len(states), len(actions))
            for i in range(min_len):
                state = states[i].flatten()
                action = actions[i].flatten()
                state_action = np.concatenate([state, action])
                state_action_pairs.append(state_action)

                if len(state_action_pairs) >= max_samples:
                    return np.array(state_action_pairs)

        return np.array(state_action_pairs)

    def get_all_states(self) -> np.ndarray:
        """Get all states from all trajectories"""
        states = []
        for traj in self.trajectories:
            states.extend(traj["states"])
        return np.array(states)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "trajectories": self.trajectories,
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                },
                f,
            )

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.trajectories = data["trajectories"]
            self.state_dim = data["state_dim"]
            self.action_dim = data["action_dim"]


class PersistentHomologyAnalyzer:
    def __init__(self, max_edge_length=2.0, max_dim=2):
        """
        Initialize persistent homology analyzer.

        Args:
            max_edge_length: Maximum edge length for filtration
            max_dim: Maximum homology dimension to compute (0=components, 1=loops, 2=voids)
        """
        self.max_edge_length = max_edge_length
        self.max_dim = max_dim
        self.rips_complex = None
        self.persistence = None
        self.diagrams = None

    def compute_persistence_from_points(
        self, points: np.ndarray, subsample_size: int = None, use_ripser: bool = True
    ) -> Dict[str, Any]:
        """
        Compute persistent homology from point cloud.

        Args:
            points: Point cloud data [n_points, n_dimensions]
            subsample_size: Optional subsampling for large datasets
            use_ripser: Use ripser library (faster) or GUDHI
        """
        print(f"Computing persistent homology for {len(points)} points...")

        # Subsample if needed
        if subsample_size and len(points) > subsample_size:
            print(f"Subsampling to {subsample_size} points...")
            rng = np.random.RandomState(42)
            indices = rng.choice(len(points), subsample_size, replace=False)
            points = points[indices]

        # Normalize data for better distance scaling
        scaler = StandardScaler()
        points_normalized = scaler.fit_transform(points)

        start_time = time.time()

        if use_ripser:
            # Use ripser (generally faster for lower dimensions)
            self._compute_with_ripser(points_normalized)
        else:
            # Use GUDHI (more features, but can be slower)
            self._compute_with_gudhi(points_normalized)

        end_time = time.time()
        print(f"Persistent homology computed in {end_time - start_time:.2f} seconds")

        # Extract topological features
        features = self._extract_topological_features()

        return features

    def _compute_with_ripser(self, points: np.ndarray):
        """Compute persistence using ripser library"""
        rips = Rips(maxdim=self.max_dim, thresh=self.max_edge_length)

        # Compute persistence diagrams
        self.diagrams = rips.fit_transform(points)

        # Convert to GUDHI-like format for consistency
        self.persistence = []
        for dim, diagram in enumerate(self.diagrams):
            for point in diagram:
                if point[1] != np.inf:  # Skip infinite points
                    self.persistence.append((dim, (point[0], point[1])))

    def _compute_with_gudhi(self, points: np.ndarray):
        """Compute persistence using GUDHI library"""
        # Create Rips complex
        self.rips_complex = gd.RipsComplex(
            points=points, max_edge_length=self.max_edge_length
        )

        # Create simplex tree
        simplex_tree = self.rips_complex.create_simplex_tree(
            max_dimension=self.max_dim + 1
        )

        # Compute persistent homology
        self.persistence = simplex_tree.persistence()

        # Extract diagrams
        self.diagrams = simplex_tree.persistence_intervals_in_dimension
        if not callable(self.diagrams):
            self.diagrams = [
                simplex_tree.persistence_intervals_in_dimension(dim)
                for dim in range(self.max_dim + 1)
            ]

    def _extract_topological_features(self) -> Dict[str, Any]:
        """Extract meaningful topological features from persistence diagrams"""
        features = {
            "num_features_by_dim": {},
            "persistence_by_dim": {},
            "lifespans_by_dim": {},
            "betti_numbers": {},
            "topological_entropy": 0.0,
        }

        if self.diagrams is None:
            return features

        # Analyze each dimension
        for dim in range(self.max_dim + 1):
            if isinstance(self.diagrams, list) and dim < len(self.diagrams):
                diagram = self.diagrams[dim]
            elif hasattr(self.diagrams, "__call__"):
                diagram = self.diagrams(dim)
            else:
                diagram = []

            if len(diagram) == 0:
                continue

            # Filter out infinite persistence points
            finite_points = []
            for point in diagram:
                if len(point) == 2 and point[1] != np.inf:
                    finite_points.append(point)

            if len(finite_points) == 0:
                continue

            finite_points = np.array(finite_points)

            # Calculate features
            lifespans = finite_points[:, 1] - finite_points[:, 0]

            features["num_features_by_dim"][dim] = len(finite_points)
            features["persistence_by_dim"][dim] = finite_points
            features["lifespans_by_dim"][dim] = lifespans

            # Betti numbers (number of features with persistence > threshold)
            persistence_threshold = (
                np.percentile(lifespans, 50) if len(lifespans) > 0 else 0
            )
            betti = np.sum(lifespans > persistence_threshold)
            features["betti_numbers"][dim] = betti

            # Topological entropy (simplified)
            if len(lifespans) > 0:
                normalized_lifespans = lifespans / np.max(lifespans)
                entropy = -np.sum(
                    normalized_lifespans * np.log(normalized_lifespans + 1e-10)
                )
                features["topological_entropy"] += entropy

        return features

    def plot_persistence_diagrams(self, title: str = "Persistence Diagrams"):
        """Plot persistence diagrams for all dimensions"""
        if self.diagrams is None:
            print("No persistence diagrams to plot")
            return

        if isinstance(self.diagrams, list):
            # For ripser format
            plot_diagrams(self.diagrams, show=False)
            plt.title(title)
            plt.show()
        else:
            # For GUDHI format
            fig, axes = plt.subplots(
                1, self.max_dim + 1, figsize=(5 * (self.max_dim + 1), 5)
            )

            for dim in range(self.max_dim + 1):
                if hasattr(self.diagrams, "__call__"):
                    diagram = self.diagrams(dim)
                else:
                    diagram = self.diagrams.persistence_intervals_in_dimension(dim)

                if len(diagram) == 0:
                    axes[dim].set_title(f"Dimension {dim} - No features")
                    continue

                # Convert to numpy array
                diagram_array = np.array(diagram)

                # Plot birth vs death
                axes[dim].scatter(diagram_array[:, 0], diagram_array[:, 1], alpha=0.6)
                axes[dim].plot(
                    [0, self.max_edge_length],
                    [0, self.max_edge_length],
                    "k--",
                    alpha=0.3,
                )
                axes[dim].set_xlabel("Birth")
                axes[dim].set_ylabel("Death")
                axes[dim].set_title(f"Dimension {dim} - {len(diagram_array)} features")
                axes[dim].set_aspect("equal")
                axes[dim].grid(True, alpha=0.3)

            plt.suptitle(title)
            plt.tight_layout()
            plt.show()

    def plot_persistence_barcode(self, title: str = "Persistence Barcode"):
        """Plot persistence barcode"""
        if self.persistence is None:
            print("No persistence data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ["blue", "red", "green", "orange", "purple"]

        # Sort persistence by dimension and birth time
        sorted_persistence = sorted(self.persistence, key=lambda x: (x[0], x[1][0]))

        # Plot each persistence interval
        y_pos = 0
        y_ticks = []
        y_labels = []

        for dim, (birth, death) in sorted_persistence:
            if death == np.inf:  # Skip infinite persistence
                continue

            # Plot the bar
            ax.hlines(
                y=y_pos,
                xmin=birth,
                xmax=death,
                color=colors[dim % len(colors)],
                linewidth=2,
            )

            # Add dimension label
            if y_pos not in y_ticks:
                y_ticks.append(y_pos)
                y_labels.append(f"Dim {dim}")

            y_pos += 1

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Filtration Parameter")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()

    def compute_wasserstein_distance(self, other_diagrams, dimension=1, p=2):
        """Compute Wasserstein distance between two persistence diagrams"""
        if self.diagrams is None or other_diagrams is None:
            return None

        if isinstance(self.diagrams, list):
            diagram1 = (
                self.diagrams[dimension] if dimension < len(self.diagrams) else []
            )
            diagram2 = (
                other_diagrams[dimension] if dimension < len(other_diagrams) else []
            )
        else:
            diagram1 = self.diagrams.persistence_intervals_in_dimension(dimension)
            diagram2 = other_diagrams.persistence_intervals_in_dimension(dimension)

        if len(diagram1) == 0 or len(diagram2) == 0:
            return None

        # Convert to numpy arrays
        diagram1 = np.array(diagram1)
        diagram2 = np.array(diagram2)

        # Compute Wasserstein distance using persim
        distance = persim.wasserstein(diagram1, diagram2, matching=False)
        return distance

    def analyze_trajectory_homology(
        self, storage, use_state_action: bool = True, max_points: int = 5000
    ) -> Dict[str, Any]:
        """
        Analyze persistent homology of RL trajectories.

        Args:
            storage: RLTrajectoryStorage instance
            use_state_action: If True, use state-action pairs; if False, use states only
            max_points: Maximum number of points for analysis
        """
        print("Analyzing trajectory topology...")

        # Extract data based on preference
        if use_state_action:
            data = storage.get_state_action_pairs(max_samples=max_points)
            data_type = "state-action"
        else:
            # Extract just states
            all_states = []
            for traj in storage.trajectories:
                if "states" in traj:
                    all_states.extend(traj["states"])
            data = np.array(all_states)
            if len(data) > max_points:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(data), max_points, replace=False)
                data = data[indices]
            data_type = "state"

        print(f"Using {data_type} data with {len(data)} points")

        # Compute persistent homology
        features = self.compute_persistence_from_points(
            data,
            subsample_size=min(3000, len(data)),  # Subsample for speed
            use_ripser=True,
        )

        # Add trajectory-specific analysis with proper handling
        trajectory_features = self._analyze_trajectory_structure(storage, features)
        features.update(trajectory_features)

        return features

    def _analyze_trajectory_structure(
        self, storage, ph_features: Dict
    ) -> Dict[str, Any]:
        """Analyze trajectory-specific topological structure"""

        # Calculate trajectory lengths
        trajectory_lengths = []
        for traj in storage.trajectories:
            # Length is number of actions (states has one more element than actions)
            length = len(traj["actions"]) if "actions" in traj else 0
            trajectory_lengths.append(length)

        trajectory_features = {
            "num_trajectories": len(storage.trajectories),
            "avg_trajectory_length": np.mean(trajectory_lengths)
            if trajectory_lengths
            else 0,
            "reward_vs_topology": {},
            "trajectory_clusters": None,
        }

        # Analyze correlation between rewards and topological features
        if len(storage.trajectories) > 1:
            rewards = []
            for traj in storage.trajectories:
                if "rewards" in traj:
                    total_reward = (
                        np.sum(traj["rewards"]) if len(traj["rewards"]) > 0 else 0
                    )
                    rewards.append(total_reward)
                else:
                    rewards.append(0)

            # If we have enough trajectories, try to find clusters
            if len(rewards) >= 10 and len(trajectory_lengths) >= 10:
                from sklearn.cluster import KMeans

                # Use rewards and lengths for clustering
                traj_data = np.column_stack([rewards, trajectory_lengths])

                # Scale data
                traj_data_scaled = StandardScaler().fit_transform(traj_data)

                # Find optimal number of clusters (simplified)
                n_clusters = min(5, len(traj_data) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(traj_data_scaled)

                trajectory_features["trajectory_clusters"] = {
                    "labels": clusters,
                    "centers": kmeans.cluster_centers_,
                    "sizes": [np.sum(clusters == i) for i in range(n_clusters)],
                }

        return trajectory_features


class TopologicalTrajectoryAnalyzer:
    """High-level analyzer for topological properties of RL trajectories"""

    def __init__(self):
        self.ph_analyzer = PersistentHomologyAnalyzer(max_dim=2)
        self.comparison_results = {}

    def analyze_multiple_policies(
        self,
        storage_dict: Dict[str, RLTrajectoryStorage],
        policy_names: List[str],
        max_points_per_policy: int = 3000,
    ):
        """Compare topological properties of trajectories from different policies"""

        print("Comparing topological properties of multiple policies...")

        results = {}
        diagrams = {}

        for name in policy_names:
            if name not in storage_dict:
                print(f"Warning: No storage found for policy {name}")
                continue

            print(f"\nAnalyzing policy: {name}")
            storage = storage_dict[name]

            # Compute persistent homology
            features = self.ph_analyzer.analyze_trajectory_homology(
                storage, use_state_action=True, max_points=max_points_per_policy
            )

            results[name] = features
            diagrams[name] = self.ph_analyzer.diagrams

            # Plot diagrams
            self.ph_analyzer.plot_persistence_diagrams(
                title=f"Persistence Diagrams - {name}"
            )

        # Compare policies using topological distances
        self._compare_policies_topologically(results, diagrams, policy_names)

        return results

    def _compare_policies_topologically(self, results, diagrams, policy_names):
        """Compute topological distances between policies"""

        print("\n" + "=" * 50)
        print("Topological Comparison of Policies")
        print("=" * 50)

        # Create distance matrix
        n_policies = len(policy_names)
        distance_matrix = np.zeros((n_policies, n_policies))

        for i, name1 in enumerate(policy_names):
            for j, name2 in enumerate(policy_names):
                if i >= j:
                    continue

                if name1 in diagrams and name2 in diagrams:
                    # Compute Wasserstein distance for dimension 1 (loops)
                    distance = self.ph_analyzer.compute_wasserstein_distance(
                        diagrams[name2], dimension=1
                    )

                    if distance is not None:
                        distance_matrix[i, j] = distance
                        distance_matrix[j, i] = distance

                        print(f"Distance between {name1} and {name2}: {distance:.4f}")

        # Plot distance matrix
        if n_policies > 1:
            self._plot_distance_matrix(distance_matrix, policy_names)

    def _plot_distance_matrix(self, distance_matrix, policy_names):
        """Plot topological distance matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(distance_matrix, cmap="viridis")

        # Add labels
        ax.set_xticks(range(len(policy_names)))
        ax.set_yticks(range(len(policy_names)))
        ax.set_xticklabels(policy_names, rotation=45, ha="right")
        ax.set_yticklabels(policy_names)

        # Add text annotations
        for i in range(len(policy_names)):
            for j in range(len(policy_names)):
                if i != j:
                    text = ax.text(
                        j,
                        i,
                        f"{distance_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="w",
                    )

        ax.set_title("Topological Distance Matrix (Wasserstein Distances)")
        plt.colorbar(im, ax=ax, label="Wasserstein Distance")
        plt.tight_layout()
        plt.show()


def demonstrate_trajectory_homology():
    """Demonstrate persistent homology analysis on Lunar Lander trajectories"""

    # Load trajectories (assuming you have saved them)
    storage = RLTrajectoryStorage()
    try:
        storage.load("ll_ppo.pkl")
        print(f"Loaded {len(storage.trajectories)} trajectories")
    except:
        print("No saved trajectories found. Please run trajectory collection first.")
        return

    # Create analyzer
    analyzer = PersistentHomologyAnalyzer(max_dim=2)

    # Analyze state-action space
    print("\n" + "=" * 50)
    print("Analyzing State-Action Space Topology")
    print("=" * 50)

    features_sa = analyzer.analyze_trajectory_homology(
        storage, use_state_action=True, max_points=5000
    )

    # Plot results
    analyzer.plot_persistence_diagrams("State-Action Space Persistence Diagrams")
    analyzer.plot_persistence_barcode("State-Action Space Persistence Barcode")

    # Print key findings
    print("\n" + "=" * 50)
    print("Topological Features Summary")
    print("=" * 50)

    for dim in range(analyzer.max_dim + 1):
        if dim in features_sa["num_features_by_dim"]:
            num_features = features_sa["num_features_by_dim"][dim]
            betti = features_sa["betti_numbers"].get(dim, 0)

            if dim == 0:
                print(f"Dimension 0 (Connected Components):")
                print(f"  - Total features: {num_features}")
                print(f"  - Betti number (significant components): {betti}")
            elif dim == 1:
                print(f"\nDimension 1 (Loops/Holes):")
                print(f"  - Total features: {num_features}")
                print(f"  - Betti number (significant loops): {betti}")

                # Analyze loops
                if (
                    "lifespans_by_dim" in features_sa
                    and dim in features_sa["lifespans_by_dim"]
                ):
                    lifespans = features_sa["lifespans_by_dim"][dim]
                    if len(lifespans) > 0:
                        print(f"  - Average loop lifespan: {np.mean(lifespans):.4f}")
                        print(f"  - Max loop lifespan: {np.max(lifespans):.4f}")
            elif dim == 2:
                print(f"\nDimension 2 (Voids/Cavities):")
                print(f"  - Total features: {num_features}")
                print(f"  - Betti number (significant voids): {betti}")

    # Compare with state-only analysis
    print("\n" + "=" * 50)
    print("Comparing State-Only vs State-Action")
    print("=" * 50)

    # Create new analyzer for comparison
    analyzer_state = PersistentHomologyAnalyzer(max_dim=1)
    features_state = analyzer_state.analyze_trajectory_homology(
        storage, use_state_action=False, max_points=5000
    )

    # Compute topological distance
    distance = analyzer.compute_wasserstein_distance(
        analyzer_state.diagrams, dimension=1
    )

    if distance is not None:
        print(
            f"\nTopological distance (Wasserstein) between state and state-action: {distance:.4f}"
        )
        print(
            "Higher distance indicates that actions significantly change the topological structure."
        )

    return analyzer, features_sa


# Example of comparing multiple policies
def compare_policies_example():
    """Example of comparing trajectories from different policies"""

    # Assume you have trajectories from different policies
    storages = {}
    policy_names = []

    # Load or create different trajectory sets
    for policy_name in ["random", "ppo_early", "ppo_trained"]:
        try:
            storage = RLTrajectoryStorage()
            storage.load(f"trajectories_{policy_name}.pkl")
            storages[policy_name] = storage
            policy_names.append(policy_name)
            print(f"Loaded {len(storage.trajectories)} trajectories for {policy_name}")
        except:
            print(f"Could not load trajectories for {policy_name}")

    if len(storages) < 2:
        print("Need at least 2 policies for comparison")
        return

    # Compare topologically
    comparator = TopologicalTrajectoryAnalyzer()
    results = comparator.analyze_multiple_policies(storages, policy_names)

    return results


if __name__ == "__main__":
    analyzer, features = demonstrate_trajectory_homology()

    # Save results
    with open("persistent_homology_results.pkl", "wb") as f:
        pickle.dump({"analyzer": analyzer, "features": features}, f)

    print("\nAnalysis complete! Results saved to 'persistent_homology_results.pkl'")
