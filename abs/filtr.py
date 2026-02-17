"""
Topological Filtration for Deterministic Abstract MDPs
Fixed version - removes problematic persim import
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import SpectralClustering, KMeans
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

# Try importing persistent homology libraries with fallbacks
try:
    from ripser import ripser
    from persim import plot_diagrams

    PERSISTENCE_AVAILABLE = True
except ImportError:
    print("Ripser or persim not installed. Install with: pip install ripser persim")
    print("Proceeding without persistence diagrams...")
    PERSISTENCE_AVAILABLE = False

    # Define dummy functions if needed
    def ripser(*args, **kwargs):
        return {"dgms": [np.array([]), np.array([])]}

    def plot_diagrams(*args, **kwargs):
        plt.gca().set_title("Persistence Diagram (unavailable)")

# ============ 1. BEHAVIORAL DISTANCE METRIC ============


class BehavioralDistance:
    """
    Compute d_B((s,a), (s',a')) = W1(P(·|s,a), P(·|s',a')) + |r(s,a) - r(s',a')|
    """

    def __init__(self, env, n_samples=100, gamma=0.95):
        self.env = env
        self.n_samples = n_samples
        self.gamma = gamma

        # Environment properties
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.state_dim = env.observation_space.n
            self.is_discrete = True
        else:
            self.state_dim = env.observation_space.shape[0]
            self.is_discrete = False

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
            self.continuous_actions = False
        else:
            self.action_dim = 5  # Discretize continuous actions
            self.continuous_actions = True
            self.action_bins = np.linspace(
                env.action_space.low[0], env.action_space.high[0], self.action_dim
            )

        # Storage for transition data
        self.transition_data = {}  # (s_idx, a_idx) -> list of next states
        self.reward_data = {}  # (s_idx, a_idx) -> list of rewards

    def _discretize_state(self, state):
        """Convert continuous state to discrete index for storage"""
        if self.is_discrete:
            return state
        # Simple grid discretization for continuous states
        if hasattr(self.env, "observation_space"):
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            # Avoid division by zero
            range_ = np.where(high - low > 0, high - low, 1.0)
            grid_point = np.floor(((state - low) / range_ * 10)).astype(int)
            # Hash to single index
            return hash(tuple(grid_point)) % 10000
        return 0

    def collect_transitions(self, n_episodes=100):
        """Collect transition data by running random policies"""
        print(f"Collecting {n_episodes} episodes of transition data...")

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False

            while not done:
                # Sample random action
                if self.continuous_actions:
                    action = np.random.choice(self.action_dim)
                    action_val = self.action_bins[action]
                else:
                    action = self.env.action_space.sample()
                    action_val = action

                # Take step
                next_state, reward, terminated, truncated, _ = self.env.step(action_val)
                done = terminated or truncated

                # Store transition
                s_idx = self._discretize_state(state)
                a_idx = action if not self.continuous_actions else action

                key = (s_idx, a_idx)
                if key not in self.transition_data:
                    self.transition_data[key] = []
                    self.reward_data[key] = []

                self.transition_data[key].append(next_state)
                self.reward_data[key].append(reward)

                state = next_state

            if (episode + 1) % 20 == 0:
                print(f"  Episode {episode + 1}/{n_episodes} complete")

        print(f"Collected {len(self.transition_data)} unique (state,action) pairs")

    def _wasserstein_distance(self, next_states1, next_states2):
        """Approximate Wasserstein-1 distance between next-state distributions"""
        if len(next_states1) == 0 or len(next_states2) == 0:
            return float("inf")

        # Convert to arrays
        if self.is_discrete:
            # For discrete states, use empirical distributions
            unique1, counts1 = np.unique(next_states1, return_counts=True)
            unique2, counts2 = np.unique(next_states2, return_counts=True)

            # Create probability vectors over all possible states
            all_states = np.unique(np.concatenate([unique1, unique2]))
            p1 = np.zeros(len(all_states))
            p2 = np.zeros(len(all_states))

            for i, s in enumerate(all_states):
                if s in unique1:
                    p1[i] = counts1[unique1 == s] / len(next_states1)
                if s in unique2:
                    p2[i] = counts2[unique2 == s] / len(next_states2)

            # L1 distance as Wasserstein-1 for discrete with unit cost
            return np.sum(np.abs(p1 - p2))
        else:
            # For continuous, use heuristic: mean of nearest neighbor distances
            states1 = np.array(next_states1)[: min(len(next_states1), 50)]
            states2 = np.array(next_states2)[: min(len(next_states2), 50)]

            if len(states1) == 0 or len(states2) == 0:
                return float("inf")

            # Compute pairwise distances
            dists = []
            for s1 in states1:
                min_dist = np.min([np.linalg.norm(s1 - s2) for s2 in states2])
                dists.append(min_dist)

            return np.mean(dists)

    def compute_distance_matrix(self, sample_size=200):
        """Compute behavioral distance matrix for sampled (s,a) pairs"""
        # Sample keys if too many
        keys = list(self.transition_data.keys())
        if len(keys) > sample_size:
            indices = np.random.choice(len(keys), sample_size, replace=False)
            sampled_keys = [keys[i] for i in indices]
        else:
            sampled_keys = keys

        n = len(sampled_keys)
        dist_matrix = np.zeros((n, n))

        print(f"Computing {n}x{n} behavioral distance matrix...")

        for i, key_i in enumerate(sampled_keys):
            for j, key_j in enumerate(sampled_keys[i:], i):
                if i == j:
                    dist_matrix[i, j] = 0
                else:
                    # Reward difference
                    r_i = (
                        np.mean(self.reward_data[key_i])
                        if self.reward_data[key_i]
                        else 0
                    )
                    r_j = (
                        np.mean(self.reward_data[key_j])
                        if self.reward_data[key_j]
                        else 0
                    )
                    r_diff = abs(r_i - r_j)

                    # Wasserstein distance on transitions
                    w_dist = self._wasserstein_distance(
                        self.transition_data[key_i], self.transition_data[key_j]
                    )

                    dist = r_diff + self.gamma * w_dist
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{n} rows")

        self.sampled_keys = sampled_keys
        self.distance_matrix = dist_matrix
        return dist_matrix, sampled_keys


# ============ 2. FILTRATION CONSTRUCTION ============


class BehavioralFiltration:
    """
    Construct filtration F_r of behavioral prototypes using Vietoris-Rips
    """

    def __init__(self, distance_matrix, keys):
        self.dist_matrix = distance_matrix
        self.keys = keys
        self.n_points = len(keys)

        # Store filtration data
        self.filtration = {}  # r -> list of simplices (as sets of indices)
        self.persistence_diagrams = None

    def build_vietoris_rips(self, r_values=None):
        """Build Vietoris-Rips filtration at given r values"""
        if r_values is None:
            # Automatically choose r values based on distance distribution
            distances = self.dist_matrix[self.dist_matrix > 0]
            if len(distances) > 0:
                r_values = np.percentile(distances, [10, 25, 40, 55, 70, 85])
            else:
                r_values = [0.1, 0.5, 1.0, 2.0, 5.0]

        print(f"Building filtration at r = {[f'{r:.3f}' for r in r_values]}")

        for r in r_values:
            # Find all pairs within distance r
            edges = []
            for i in range(self.n_points):
                for j in range(i + 1, self.n_points):
                    if self.dist_matrix[i, j] <= r:
                        edges.append((i, j))

            # Build simplicial complex (simplices up to dimension 2 for visualization)
            simplices = []

            # Add vertices
            for i in range(self.n_points):
                simplices.append((i,))

            # Add edges
            for i, j in edges:
                simplices.append((i, j))

            # Add triangles (3-cliques)
            edge_set = set(edges)
            for i in range(self.n_points):
                for j in range(i + 1, self.n_points):
                    for k in range(j + 1, self.n_points):
                        if (
                            ((i, j) in edge_set or (j, i) in edge_set)
                            and ((i, k) in edge_set or (k, i) in edge_set)
                            and ((j, k) in edge_set or (k, j) in edge_set)
                        ):
                            simplices.append((i, j, k))

            self.filtration[r] = simplices

        return self.filtration

    def compute_persistence(self):
        """Compute persistent homology using Ripser"""
        if not PERSISTENCE_AVAILABLE:
            print("Ripser not available, skipping persistence computation")
            return None

        print("Computing persistent homology...")
        try:
            self.persistence_diagrams = ripser(self.dist_matrix, distance_matrix=True)[
                "dgms"
            ]
            return self.persistence_diagrams
        except Exception as e:
            print(f"Persistence computation failed: {e}")
            return None

    def get_persistent_clusters(self, persistence_threshold=0.5):
        """Extract clusters that persist across scales using H0 persistence"""
        if self.persistence_diagrams is None:
            self.compute_persistence()

        if self.persistence_diagrams is None or len(self.persistence_diagrams[0]) == 0:
            print("No persistence data, using spectral method instead")
            return max(2, self.n_points // 10)

        # Use H0 persistence to find connected components at different scales
        h0_diagram = self.persistence_diagrams[0]

        if len(h0_diagram) == 0:
            return max(2, self.n_points // 10)

        # Filter by persistence (birth-death)
        persistent_features = h0_diagram[
            h0_diagram[:, 1] - h0_diagram[:, 0] > persistence_threshold
        ]

        # The number of persistent clusters is the number of features that persist
        n_persistent = len(persistent_features)

        print(
            f"Found {n_persistent} persistent features with threshold {persistence_threshold}"
        )

        return max(2, n_persistent)


# ============ 3. SPECTRAL CLUSTERING ON FILTRATION ============


class FiltrationSpectralClustering:
    """
    Perform spectral clustering on the filtration-induced graph
    """

    def __init__(self, distance_matrix, keys):
        self.dist_matrix = distance_matrix
        self.keys = keys
        self.n_points = len(keys)

        # Store results
        self.cluster_assignments = None
        self.persistent_clusters = None

    def build_behavioral_graph(self, r, sigma=1.0):
        """Build graph G_r with exponential similarity kernel"""
        # Similarity matrix
        S = np.exp(-(self.dist_matrix**2) / (2 * sigma**2 * r**2))
        np.fill_diagonal(S, 0)

        # Threshold at r
        S[self.dist_matrix > r] = 0

        return S

    def spectral_clustering_at_scale(self, r, n_clusters=None, sigma=1.0):
        """Perform spectral clustering at specific scale r"""
        S = self.build_behavioral_graph(r, sigma)

        # Compute degree matrix
        D = np.diag(S.sum(axis=1) + 1e-10)

        # Compute normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(S.sum(axis=1), 1e-10)))
        L_norm = np.eye(self.n_points) - D_inv_sqrt @ S @ D_inv_sqrt

        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = eigh(
                L_norm, subset_by_index=[0, min(10, self.n_points - 1)]
            )

            # Determine number of clusters from eigengap
            if n_clusters is None:
                if len(eigenvalues) > 1:
                    eigengaps = np.diff(eigenvalues[: min(10, len(eigenvalues))])
                    if len(eigengaps) > 0:
                        n_clusters = (
                            np.argmax(eigengaps[1:]) + 2 if len(eigengaps) > 1 else 2
                        )
                    else:
                        n_clusters = 2
                else:
                    n_clusters = 2

            # Use first n_clusters eigenvectors (excluding the first which is constant)
            features = eigenvectors[:, 1 : min(n_clusters, eigenvectors.shape[1])]

            # K-means on eigenfeatures
            if features.shape[1] > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
            else:
                labels = np.zeros(self.n_points, dtype=int)

            return labels, n_clusters

        except Exception as e:
            print(f"Spectral clustering failed at r={r:.3f}: {e}")
            # Fallback to distance-based clustering
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(eps=r, min_samples=2, metric="precomputed")
            labels = clustering.fit_predict(self.dist_matrix)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            return labels, n_clusters

    def persistent_spectral_clustering(self, r_range=None, persistence_threshold=0.3):
        """Track clusters across scales to find persistent ones"""
        if r_range is None:
            # Use percentiles of distances
            distances = self.dist_matrix[self.dist_matrix > 0]
            if len(distances) > 0:
                r_range = np.percentile(distances, [10, 25, 40, 55, 70, 85])
            else:
                r_range = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]

        # Store clusterings at each scale
        clusterings = []

        for r in r_range:
            labels, n = self.spectral_clustering_at_scale(r)
            clusterings.append({"r": r, "labels": labels.copy(), "n_clusters": n})
            print(f"  r={r:.3f}: {n} clusters")

        # Track cluster persistence using Jaccard similarity
        n_scales = len(clusterings)
        cluster_lifetimes = defaultdict(float)

        # For each cluster in first scale, track its presence
        if n_scales > 0:
            first_labels = clusterings[0]["labels"]
            unique_first = np.unique(first_labels)

            for c1 in unique_first:
                if c1 == -1:
                    continue
                mask1 = first_labels == c1
                if not np.any(mask1):
                    continue

                # Track through scales
                for i in range(n_scales - 1):
                    r_current = clusterings[i]["r"]
                    r_next = clusterings[i + 1]["r"]
                    labels_current = clusterings[i]["labels"]
                    labels_next = clusterings[i + 1]["labels"]

                    # Find best matching cluster
                    best_overlap = 0
                    unique_next = np.unique(labels_next)

                    for c2 in unique_next:
                        if c2 == -1:
                            continue
                        mask2 = labels_next == c2
                        overlap = np.sum(mask1 & mask2) / np.maximum(
                            np.sum(mask1 | mask2), 1
                        )
                        best_overlap = max(best_overlap, overlap)

                    if best_overlap > 0.5:
                        cluster_lifetimes[c1] += r_next - r_current

        # Select persistent clusters
        total_range = r_range[-1] - r_range[0]
        persistent_clusters = []
        for c, lifetime in cluster_lifetimes.items():
            if lifetime > persistence_threshold * total_range:
                persistent_clusters.append(c)

        print(f"  Found {len(persistent_clusters)} persistent clusters")

        # Use the clustering at the median scale
        median_idx = len(r_range) // 2
        final_labels = clusterings[median_idx]["labels"].copy()

        # Remap to persistent clusters only
        if len(persistent_clusters) > 0:
            # Create mapping from original labels to persistent IDs
            persistent_mapping = {}
            for i, c in enumerate(persistent_clusters):
                persistent_mapping[c] = i

            new_labels = np.full_like(final_labels, -1)
            for i, label in enumerate(final_labels):
                if label in persistent_mapping:
                    new_labels[i] = persistent_mapping[label]

            final_labels = new_labels

        self.cluster_assignments = final_labels
        self.persistent_clusters = persistent_clusters
        self.clusterings = clusterings

        return final_labels, persistent_clusters, clusterings


# ============ 4. DETERMINISTIC ABSTRACT MDP CONSTRUCTION ============


class DeterministicAbstractMDP:
    """
    Construct deterministic abstract MDP from persistent clusters
    """

    def __init__(self, cluster_assignments, keys, transition_data, reward_data, env):
        self.cluster_assignments = cluster_assignments
        self.keys = keys
        self.transition_data = transition_data
        self.reward_data = reward_data
        self.env = env

        # Map from key index to cluster
        self.key_to_cluster = {}
        for idx, key in enumerate(keys):
            self.key_to_cluster[key] = cluster_assignments[idx]

        # Build abstract MDP components
        self.H = None  # Abstract state space
        self.A = None  # Action space
        self.T = None  # Deterministic transition function
        self.R = None  # Reward function
        self.cluster_to_id = {}  # Map from cluster label to abstract state ID

        self._build_abstract_mdp()

    def _discretize_state(self, state):
        """Helper to discretize state for key lookup"""
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            return state

        low = self.env.observation_space.low
        high = self.env.observation_space.high
        range_ = np.where(high - low > 0, high - low, 1.0)
        grid_point = np.floor(((state - low) / range_ * 10)).astype(int)
        return hash(tuple(grid_point)) % 10000

    def _build_abstract_mdp(self):
        """Build deterministic abstract MDP from clusters"""
        # Get unique clusters (excluding noise -1)
        unique_clusters = np.unique(self.cluster_assignments)
        unique_clusters = unique_clusters[unique_clusters >= 0]

        if len(unique_clusters) == 0:
            print("Warning: No valid clusters found. Creating default clusters.")
            unique_clusters = np.array([0])
            self.cluster_assignments = np.zeros(len(self.keys), dtype=int)

        # Map cluster labels to abstract state IDs
        self.cluster_to_id = {c: i for i, c in enumerate(unique_clusters)}
        self.H = list(range(len(unique_clusters)))

        # Action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.A = list(range(self.env.action_space.n))
        else:
            self.A = list(range(5))  # Discretized actions

        # Initialize transition counts and reward sums
        trans_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        reward_sums = defaultdict(lambda: defaultdict(float))
        count_za = defaultdict(lambda: defaultdict(int))

        # Aggregate transitions
        for idx, key in enumerate(self.keys):
            s_idx, a = key
            c = self.cluster_assignments[idx]

            if c < 0:  # Skip noise
                continue

            h = self.cluster_to_id[c]

            # Get next states for this (s,a)
            next_states = self.transition_data.get(key, [])
            rewards = self.reward_data.get(key, [])

            for next_state, reward in zip(next_states, rewards):
                # Find cluster of next state
                next_key = None
                next_s_idx = self._discretize_state(next_state)

                # Try exact match first
                for k in self.keys:
                    if k[0] == next_s_idx and k[1] == a:
                        next_key = k
                        break

                # If no exact match, try any action
                if next_key is None:
                    for k in self.keys:
                        if k[0] == next_s_idx:
                            next_key = k
                            break

                if next_key and next_key in self.key_to_cluster:
                    next_c = self.key_to_cluster[next_key]
                    if next_c >= 0 and next_c in self.cluster_to_id:
                        next_h = self.cluster_to_id[next_c]
                        trans_counts[h][a][next_h] += 1
                        reward_sums[h][a] += reward
                        count_za[h][a] += 1

        # Build deterministic transitions (mode) and average rewards
        self.T = {}
        self.R = {}

        for h in self.H:
            self.T[h] = {}
            self.R[h] = {}
            for a in self.A:
                if count_za[h][a] > 0:
                    # Most common next abstract state
                    if trans_counts[h][a]:
                        next_h = max(trans_counts[h][a].items(), key=lambda x: x[1])[0]
                    else:
                        next_h = h  # Stay in same state if no data

                    self.T[h][a] = next_h
                    self.R[h][a] = reward_sums[h][a] / count_za[h][a]
                else:
                    # No data, default to self-loop with zero reward
                    self.T[h][a] = h
                    self.R[h][a] = 0.0

        print(f"\nBuilt deterministic abstract MDP:")
        print(f"  • Abstract states: {len(self.H)}")
        print(f"  • Actions: {len(self.A)}")
        print(f"  • Transitions: deterministic")

    def get_abstract_transition_matrix(self):
        """Return transition matrix as numpy array"""
        n_states = len(self.H)
        n_actions = len(self.A)

        trans_matrix = np.zeros((n_states, n_actions), dtype=int)
        reward_matrix = np.zeros((n_states, n_actions))

        for h in self.H:
            for a in self.A:
                trans_matrix[h, a] = self.T[h][a]
                reward_matrix[h, a] = self.R[h][a]

        return trans_matrix, reward_matrix

    def visualize_abstract_mdp(self):
        """Visualize the deterministic abstract MDP as a graph"""
        G = nx.MultiDiGraph()

        # Add nodes
        for h in self.H:
            G.add_node(h)

        # Add edges
        for h in self.H:
            for a in self.A:
                next_h = self.T[h][a]
                reward = self.R[h][a]
                G.add_edge(h, next_h, action=a, reward=reward)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        # Draw edges
        edge_colors = plt.cm.tab10(np.linspace(0, 1, len(self.A)))
        drawn_edges = set()

        for u, v, data in G.edges(data=True):
            if (u, v) in drawn_edges:
                # Offset parallel edges
                offset = 0.1
                rad = offset
            else:
                rad = 0.0
                drawn_edges.add((u, v))

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                arrowstyle="->",
                arrowsize=20,
                width=2,
                alpha=0.6,
                edge_color=[edge_colors[data["action"]]],
                connectionstyle=f"arc3,rad={rad}",
            )

            # Label with action and reward
            mid = (np.array(pos[u]) + np.array(pos[v])) / 2
            if rad != 0:
                # Add offset for parallel edges
                perp = np.array([pos[v][1] - pos[u][1], pos[u][0] - pos[v][0]])
                perp = perp / np.linalg.norm(perp) * 0.2
                mid = mid + perp

            plt.annotate(
                f"a={data['action']}\nr={data['reward']:.2f}",
                xy=mid,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
            )

        plt.title("Deterministic Abstract MDP")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# ============ 5. VISUALIZATION UTILITIES ============


class FiltrationVisualizer:
    """Visualize filtration and persistence results"""

    def __init__(self, behavioral_dist, filtration, spectral_clustering, abstract_mdp):
        self.behavioral = behavioral_dist
        self.filtration = filtration
        self.spectral = spectral_clustering
        self.abstract_mdp = abstract_mdp

    def plot_distance_matrix(self):
        """Visualize behavioral distance matrix"""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.behavioral.distance_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(label="Behavioral Distance")
        plt.title("Behavioral Distance Matrix")
        plt.xlabel("State-Action Index")
        plt.ylabel("State-Action Index")
        plt.tight_layout()
        plt.show()

    def plot_persistence_diagram(self):
        """Plot persistence diagram if available"""
        if self.filtration.persistence_diagrams is not None:
            plt.figure(figsize=(8, 6))
            plot_diagrams(self.filtration.persistence_diagrams)
            plt.title("Persistence Diagram")
            plt.tight_layout()
            plt.show()

    def plot_filtration_at_scales(self, r_values=None):
        """Plot the filtration at different scales using MDS"""
        if r_values is None:
            all_r = sorted(self.filtration.filtration.keys())
            if len(all_r) > 4:
                # Take evenly spaced values
                indices = np.linspace(0, len(all_r) - 1, 4, dtype=int)
                r_values = [all_r[i] for i in indices]
            else:
                r_values = all_r

        # MDS for 2D embedding
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress=False,
            max_iter=300,
        )
        coords = mds.fit_transform(self.behavioral.distance_matrix)

        n_plots = len(r_values)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for idx, r in enumerate(r_values):
            ax = axes[idx]

            # Plot points
            ax.scatter(coords[:, 0], coords[:, 1], c="blue", s=30, alpha=0.6)

            # Plot edges for this r
            simplices = self.filtration.filtration.get(r, [])
            for simplex in simplices:
                if len(simplex) == 2:  # Edge
                    i, j = simplex
                    ax.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        "r-",
                        alpha=0.3,
                        linewidth=1,
                    )

            ax.set_title(f"Filtration at r={r:.3f}")
            ax.set_xlabel("MDS1")
            ax.set_ylabel("MDS2")
            ax.set_aspect("equal")

        plt.suptitle("Behavioral Filtration at Different Scales")
        plt.tight_layout()
        plt.show()

    def plot_persistent_clusters(self):
        """Visualize persistent clusters in MDS space"""
        if self.spectral.cluster_assignments is None:
            print("No cluster assignments available")
            return

        # MDS embedding
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress=False,
            max_iter=300,
        )
        coords = mds.fit_transform(self.behavioral.distance_matrix)

        plt.figure(figsize=(10, 8))

        # Plot all points with cluster colors
        labels = self.spectral.cluster_assignments
        unique_labels = np.unique(labels)

        # Use a colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                # Noise points
                plt.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c="gray",
                    marker="x",
                    s=50,
                    alpha=0.5,
                    label="Noise",
                )
            else:
                plt.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    color=colors[i],
                    alpha=0.7,
                    s=100,
                    label=f"Cluster {label}",
                )

        plt.title("Persistent Behavioral Clusters")
        plt.xlabel("MDS1")
        plt.ylabel("MDS2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_cluster_persistence_heatmap(self):
        """Visualize how clusters evolve across scales"""
        if (
            not hasattr(self.spectral, "clusterings")
            or len(self.spectral.clusterings) == 0
        ):
            return

        n_scales = len(self.spectral.clusterings)
        n_points = len(self.behavioral.sampled_keys)

        # Create matrix of cluster assignments
        cluster_matrix = np.zeros((n_scales, n_points))
        for i, c in enumerate(self.spectral.clusterings):
            cluster_matrix[i, :] = c["labels"]

        plt.figure(figsize=(12, 6))
        plt.imshow(cluster_matrix, aspect="auto", cmap="tab20", interpolation="nearest")
        plt.colorbar(label="Cluster ID")
        plt.xlabel("State-Action Point Index")
        plt.ylabel("Filtration Scale Index")
        r_labels = [f"{c['r']:.3f}" for c in self.spectral.clusterings]
        plt.yticks(range(n_scales), r_labels)
        plt.title("Cluster Evolution Across Scales")
        plt.tight_layout()
        plt.show()


# ============ 6. MAIN PIPELINE ============


def run_topological_abstraction_pipeline(
    env_name="FrozenLake-v1", n_episodes=200, persistence_threshold=0.3, n_clusters=None
):
    """Run complete topological filtration pipeline"""

    print("\n" + "=" * 70)
    print("TOPOLOGICAL FILTRATION FOR DETERMINISTIC ABSTRACT MDPs")
    print("=" * 70)

    # Create environment
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Error creating environment {env_name}: {e}")
        print("Falling back to FrozenLake-v1")
        env = gym.make("FrozenLake-v1")

    print(f"\nEnvironment: {env_name}")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Step 1: Collect transition data and compute behavioral distances
    print("\n" + "-" * 50)
    print("STEP 1: Collecting behavioral data")
    print("-" * 50)

    behavioral = BehavioralDistance(env)
    behavioral.collect_transitions(n_episodes=n_episodes)

    if len(behavioral.transition_data) < 10:
        print("Warning: Very few transitions collected. Try increasing n_episodes.")

    dist_matrix, keys = behavioral.compute_distance_matrix(
        sample_size=min(150, len(behavioral.transition_data))
    )

    # Step 2: Build filtration
    print("\n" + "-" * 50)
    print("STEP 2: Building behavioral filtration")
    print("-" * 50)

    filtration = BehavioralFiltration(dist_matrix, keys)
    distances = dist_matrix[dist_matrix > 0]
    if len(distances) > 0:
        r_values = np.percentile(distances, [10, 25, 40, 55, 70, 85])
    else:
        r_values = [0.1, 0.5, 1.0, 1.5, 2.0]

    filtration.build_vietoris_rips(r_values)

    # Step 3: Compute persistence
    print("\n" + "-" * 50)
    print("STEP 3: Computing persistent homology")
    print("-" * 50)

    persistence = filtration.compute_persistence()
    n_persistent = filtration.get_persistent_clusters(
        persistence_threshold=persistence_threshold
    )

    # Step 4: Spectral clustering on filtration
    print("\n" + "-" * 50)
    print("STEP 4: Persistent spectral clustering")
    print("-" * 50)

    spectral = FiltrationSpectralClustering(dist_matrix, keys)

    if n_clusters is None:
        n_clusters = max(2, min(n_persistent, 10))

    labels, persistent_clusters, clusterings = spectral.persistent_spectral_clustering(
        r_range=r_values, persistence_threshold=persistence_threshold
    )

    # Step 5: Build deterministic abstract MDP
    print("\n" + "-" * 50)
    print("STEP 5: Constructing deterministic abstract MDP")
    print("-" * 50)

    abstract_mdp = DeterministicAbstractMDP(
        labels, keys, behavioral.transition_data, behavioral.reward_data, env
    )

    # Step 6: Visualizations
    print("\n" + "-" * 50)
    print("STEP 6: Generating visualizations")
    print("-" * 50)

    visualizer = FiltrationVisualizer(behavioral, filtration, spectral, abstract_mdp)

    visualizer.plot_distance_matrix()

    if persistence is not None and PERSISTENCE_AVAILABLE:
        visualizer.plot_persistence_diagram()

    visualizer.plot_filtration_at_scales()
    visualizer.plot_persistent_clusters()
    visualizer.plot_cluster_persistence_heatmap()

    abstract_mdp.visualize_abstract_mdp()

    # Step 7: Return results
    print("\n" + "=" * 70)
    print("ABSTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  • Abstract states: {len(abstract_mdp.H)}")
    print(f"  • Persistent clusters: {len(persistent_clusters)}")
    print(f"  • Deterministic transitions: Yes")
    trans_matrix, _ = abstract_mdp.get_abstract_transition_matrix()
    print(f"  • Transition matrix shape: {trans_matrix.shape}")
    print("=" * 70)

    env.close()
    return {
        "behavioral": behavioral,
        "filtration": filtration,
        "spectral": spectral,
        "abstract_mdp": abstract_mdp,
        "visualizer": visualizer,
    }


# ============ 7. TEST ON MULTIPLE ENVIRONMENTS ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="FrozenLake-v1",
        choices=[
            "FrozenLake-v1",
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "Taxi-v3",
            "ALE/Breakout-v5",
        ],
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Persistence threshold for clusters",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Number of abstract states (optional)",
    )

    args = parser.parse_args()

    # Run pipeline
    try:
        results = run_topological_abstraction_pipeline(
            env_name=args.env,
            n_episodes=args.episodes,
            persistence_threshold=args.threshold,
            n_clusters=args.clusters,
        )

        print("\n Pipeline finished successfully!")

    except Exception as e:
        print(f"\n Error during pipeline execution: {e}")
        import traceback

        traceback.print_exc()
