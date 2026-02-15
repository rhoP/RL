"""
Ultrametric Analysis for Topological Abstraction
Focuses on Wasserstein distance and identifies ultrametric structure in abstract states
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
import networkx as nx
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Try importing persistent homology
try:
    from ripser import ripser
    from persim import plot_diagrams

    PERSISTENCE_AVAILABLE = True
except ImportError:
    print("Ripser/persim not available - persistence diagrams disabled")
    PERSISTENCE_AVAILABLE = False

# ============ 1. WASSERSTEIN-ONLY BEHAVIORAL DISTANCE ============


class WassersteinBehavioralDistance:
    """
    Compute d_W((s,a), (s',a')) = W1(P(·|s,a), P(·|s',a'))
    No reward component - focusing purely on transition dynamics
    """

    def __init__(self, env, n_episodes=200, sample_size=100):
        self.env = env
        self.n_episodes = n_episodes
        self.sample_size = sample_size

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
            self.action_dim = 5
            self.continuous_actions = True
            self.action_bins = np.linspace(
                env.action_space.low[0], env.action_space.high[0], self.action_dim
            )

        # Storage
        self.transition_data = {}  # (s_idx, a_idx) -> list of next states
        self.sampled_keys = []
        self.distance_matrix = None

    def _discretize_state(self, state):
        """Convert continuous state to discrete index"""
        if self.is_discrete:
            return state

        low = self.env.observation_space.low
        high = self.env.observation_space.high
        range_ = np.where(high - low > 0, high - low, 1.0)
        grid_point = np.floor(((state - low) / range_ * 10)).astype(int)
        return hash(tuple(grid_point)) % 10000

    def collect_transitions(self):
        """Collect transition data using random policy"""
        print(f"Collecting {self.n_episodes} episodes of transition data...")

        for episode in range(self.n_episodes):
            state, _ = self.env.reset()
            done = False

            while not done:
                if self.continuous_actions:
                    action = np.random.choice(self.action_dim)
                    action_val = self.action_bins[action]
                else:
                    action = self.env.action_space.sample()
                    action_val = action

                next_state, reward, terminated, truncated, _ = self.env.step(action_val)
                done = terminated or truncated

                s_idx = self._discretize_state(state)
                a_idx = action if not self.continuous_actions else action
                key = (s_idx, a_idx)

                if key not in self.transition_data:
                    self.transition_data[key] = []

                self.transition_data[key].append(next_state)
                state = next_state

            if (episode + 1) % 50 == 0:
                print(f"  Episode {episode + 1}/{self.n_episodes}")

        print(f"Collected {len(self.transition_data)} unique (state,action) pairs")

    def _wasserstein_distance(self, next_states1, next_states2):
        """Approximate Wasserstein-1 distance"""
        if len(next_states1) == 0 or len(next_states2) == 0:
            return float("inf")

        if self.is_discrete:
            # For discrete states, use L1 on empirical distributions
            unique1, counts1 = np.unique(next_states1, return_counts=True)
            unique2, counts2 = np.unique(next_states2, return_counts=True)

            all_states = np.unique(np.concatenate([unique1, unique2]))
            p1 = np.zeros(len(all_states))
            p2 = np.zeros(len(all_states))

            for i, s in enumerate(all_states):
                if s in unique1:
                    p1[i] = counts1[unique1 == s] / len(next_states1)
                if s in unique2:
                    p2[i] = counts2[unique2 == s] / len(next_states2)

            return np.sum(np.abs(p1 - p2))
        else:
            # For continuous, use mean of nearest neighbor distances
            states1 = np.array(next_states1)[: min(len(next_states1), 50)]
            states2 = np.array(next_states2)[: min(len(next_states2), 50)]

            if len(states1) == 0 or len(states2) == 0:
                return float("inf")

            dists = []
            for s1 in states1:
                min_dist = np.min([np.linalg.norm(s1 - s2) for s2 in states2])
                dists.append(min_dist)

            return np.mean(dists)

    def compute_distance_matrix(self):
        """Compute Wasserstein distance matrix"""
        keys = list(self.transition_data.keys())
        if len(keys) > self.sample_size:
            indices = np.random.choice(len(keys), self.sample_size, replace=False)
            self.sampled_keys = [keys[i] for i in indices]
        else:
            self.sampled_keys = keys

        n = len(self.sampled_keys)
        dist_matrix = np.zeros((n, n))

        print(f"Computing {n}x{n} Wasserstein distance matrix...")

        for i, key_i in enumerate(self.sampled_keys):
            for j, key_j in enumerate(self.sampled_keys[i:], i):
                if i == j:
                    continue

                w_dist = self._wasserstein_distance(
                    self.transition_data[key_i], self.transition_data[key_j]
                )

                # Handle infinite distances
                if np.isinf(w_dist):
                    w_dist = 1.0  # Default max distance

                dist_matrix[i, j] = w_dist
                dist_matrix[j, i] = w_dist

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{n}")

        self.distance_matrix = dist_matrix
        return dist_matrix, self.sampled_keys


# ============ 2. ULTRAMETRIC ANALYSIS ============


class UltrametricAnalyzer:
    """
    Analyze ultrametric structure in the Wasserstein distance space
    """

    def __init__(self, distance_matrix, keys):
        self.dist_matrix = distance_matrix
        self.keys = keys
        self.n_points = len(keys)

        # Storage for ultrametric analysis
        self.linkage_matrix = None
        self.ultrametric_matrix = None
        self.cophenetic_correlation = None
        self.hierarchy_levels = None

    def compute_hierarchical_clustering(self, method="single"):
        """
        Perform hierarchical clustering and extract ultrametric structure
        Single linkage is the natural choice for ultrametric properties
        """
        print(f"\nComputing hierarchical clustering with {method} linkage...")

        # Convert distance matrix to condensed form
        condensed_dist = squareform(self.dist_matrix)

        # Perform hierarchical clustering
        self.linkage_matrix = linkage(condensed_dist, method=method)

        # Compute cophenetic correlation (how well tree preserves distances)
        self.cophenetic_correlation, self.ultrametric_matrix = cophenet(
            self.linkage_matrix, condensed_dist
        )

        print(f"  Cophenetic correlation: {self.cophenetic_correlation:.4f}")
        print(f"  (Closer to 1.0 indicates better ultrametric structure)")

        return self.linkage_matrix

    def extract_ultrametric_levels(self, n_levels=10):
        """Extract clusterings at different levels of the hierarchy"""
        if self.linkage_matrix is None:
            self.compute_hierarchical_clustering()

        # Get distance range
        max_dist = np.max(self.linkage_matrix[:, 2])
        thresholds = np.linspace(0.1 * max_dist, 0.9 * max_dist, n_levels)

        self.hierarchy_levels = []

        for i, t in enumerate(thresholds):
            # Cut the dendrogram at threshold t
            clusters = fcluster(self.linkage_matrix, t, criterion="distance")
            n_clusters = len(np.unique(clusters))

            self.hierarchy_levels.append(
                {
                    "threshold": t,
                    "clusters": clusters,
                    "n_clusters": n_clusters,
                    "level": i,
                }
            )

            print(f"  Level {i + 1}: threshold={t:.3f}, clusters={n_clusters}")

        return self.hierarchy_levels

    def compute_ultrametric_deviation(self):
        """
        Measure how far the space is from being ultrametric
        For an ultrametric, all triangles are isosceles with two equal larger sides
        """
        n = self.n_points
        violations = 0
        total = 0

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    total += 1
                    d_ij = self.dist_matrix[i, j]
                    d_ik = self.dist_matrix[i, k]
                    d_jk = self.dist_matrix[j, k]

                    # In an ultrametric, the two largest distances are equal
                    sorted_d = sorted([d_ij, d_ik, d_jk])
                    if abs(sorted_d[1] - sorted_d[2]) > 1e-6:
                        violations += 1

        ultrametric_ratio = 1.0 - (violations / total) if total > 0 else 0
        print(f"\nUltrametric Analysis:")
        print(f"  Triangle violations: {violations}/{total}")
        print(f"  Ultrametric ratio: {ultrametric_ratio:.4f}")

        return ultrametric_ratio

    def get_ultrametric_clusters(self, method="consistent"):
        """
        Get the most "ultrametric" clustering using different criteria
        """
        if self.hierarchy_levels is None:
            self.extract_ultrametric_levels()

        results = []

        for level in self.hierarchy_levels:
            clusters = level["clusters"]
            n_clusters = level["n_clusters"]

            # Silhouette score
            if n_clusters > 1 and n_clusters < self.n_points:
                try:
                    sil = silhouette_score(
                        self.dist_matrix, clusters, metric="precomputed"
                    )
                except:
                    sil = -1
            else:
                sil = -1

            # Cluster size distribution
            unique, counts = np.unique(clusters, return_counts=True)
            size_entropy = -np.sum(
                (counts / len(clusters)) * np.log(counts / len(clusters) + 1e-10)
            )

            results.append(
                {
                    "threshold": level["threshold"],
                    "n_clusters": n_clusters,
                    "silhouette": sil,
                    "size_entropy": size_entropy,
                    "clusters": clusters,
                }
            )

        # Find best clustering by combined metric
        df = pd.DataFrame(results)
        df["score"] = df["silhouette"] - 0.1 * df["size_entropy"]
        best_idx = df["score"].idxmax()

        print(f"\nBest ultrametric clustering:")
        print(f"  Threshold: {df.loc[best_idx, 'threshold']:.3f}")
        print(f"  Clusters: {df.loc[best_idx, 'n_clusters']}")
        print(f"  Silhouette: {df.loc[best_idx, 'silhouette']:.3f}")

        return df.loc[best_idx]


# ============ 3. DETERMINISTIC ABSTRACT MDP FROM ULTRAMETRIC ============


class UltrametricAbstractMDP:
    """
    Construct abstract MDP using ultrametric structure
    """

    def __init__(self, clusters, keys, transition_data, env):
        self.clusters = clusters
        self.keys = keys
        self.transition_data = transition_data
        self.env = env

        # Map keys to clusters
        self.key_to_cluster = {keys[i]: clusters[i] for i in range(len(keys))}

        # Build abstract MDP
        self._build_abstract_mdp()

    def _discretize_state(self, state):
        """Helper for state discretization"""
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            return state

        low = self.env.observation_space.low
        high = self.env.observation_space.high
        range_ = np.where(high - low > 0, high - low, 1.0)
        grid_point = np.floor(((state - low) / range_ * 10)).astype(int)
        return hash(tuple(grid_point)) % 10000

    def _build_abstract_mdp(self):
        """Build abstract MDP with deterministic transitions"""
        unique_clusters = np.unique(self.clusters)
        unique_clusters = unique_clusters[unique_clusters >= 0]

        self.cluster_to_id = {c: i for i, c in enumerate(unique_clusters)}
        self.H = list(range(len(unique_clusters)))

        # Action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.A = list(range(self.env.action_space.n))
        else:
            self.A = list(range(5))

        # Initialize transition counts
        trans_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        count_za = defaultdict(lambda: defaultdict(int))

        # Aggregate transitions
        for idx, key in enumerate(self.keys):
            s_idx, a = key
            c = self.clusters[idx]

            if c < 0:
                continue

            h = self.cluster_to_id[c]

            # Get next states
            next_states = self.transition_data.get(key, [])

            for next_state in next_states:
                # Find cluster of next state
                next_key = None
                next_s_idx = self._discretize_state(next_state)

                # Try to find matching key
                for k in self.keys:
                    if k[0] == next_s_idx and k[1] == a:
                        next_key = k
                        break

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
                        count_za[h][a] += 1

        # Build deterministic transitions (mode)
        self.T = {}

        for h in self.H:
            self.T[h] = {}
            for a in self.A:
                if count_za[h][a] > 0 and trans_counts[h][a]:
                    next_h = max(trans_counts[h][a].items(), key=lambda x: x[1])[0]
                else:
                    next_h = h
                self.T[h][a] = next_h

        print(f"\nBuilt ultrametric abstract MDP:")
        print(f"  Abstract states: {len(self.H)}")
        print(f"  Actions: {len(self.A)}")

    def get_transition_matrix(self):
        """Return deterministic transition matrix"""
        n_states = len(self.H)
        n_actions = len(self.A)
        trans_matrix = np.zeros((n_states, n_actions), dtype=int)

        for h in self.H:
            for a in self.A:
                trans_matrix[h, a] = self.T[h][a]

        return trans_matrix


# ============ 4. VISUALIZATION ============


class UltrametricVisualizer:
    """
    Visualize ultrametric structure and abstractions
    """

    def __init__(self, analyzer, abstract_mdp=None):
        self.analyzer = analyzer
        self.abstract_mdp = abstract_mdp

    def plot_dendrogram(self, max_d=None):
        """Plot hierarchical clustering dendrogram"""
        if self.analyzer.linkage_matrix is None:
            self.analyzer.compute_hierarchical_clustering()

        plt.figure(figsize=(12, 6))

        # Plot dendrogram
        dendrogram(
            self.analyzer.linkage_matrix,
            truncate_mode="level",
            p=5,
            color_threshold=0.7 * max(self.analyzer.linkage_matrix[:, 2]),
            above_threshold_color="gray",
        )

        plt.title(
            f"Ultrametric Structure (Cophenetic Correlation: {self.analyzer.cophenetic_correlation:.3f})"
        )
        plt.xlabel("State-Action Pair Index")
        plt.ylabel("Wasserstein Distance")

        if max_d:
            plt.axhline(
                y=max_d, color="r", linestyle="--", label=f"Threshold={max_d:.3f}"
            )
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_ultrametric_heatmap(self):
        """Compare original vs ultrametric distance matrices"""
        if self.analyzer.ultrametric_matrix is None:
            self.analyzer.compute_hierarchical_clustering()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original distance matrix
        im1 = axes[0].imshow(self.analyzer.dist_matrix, cmap="viridis", aspect="auto")
        axes[0].set_title("Original Wasserstein Distances")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Index")
        plt.colorbar(im1, ax=axes[0])

        # Ultrametric distance matrix (from cophenetic distances)
        if self.analyzer.ultrametric_matrix is not None:
            ultramatrix = squareform(self.analyzer.ultrametric_matrix)
            im2 = axes[1].imshow(ultramatrix, cmap="plasma", aspect="auto")
            axes[1].set_title("Ultrametric Distances")
            axes[1].set_xlabel("Index")
            axes[1].set_ylabel("Index")
            plt.colorbar(im2, ax=axes[1])

            # Difference
            diff = np.abs(self.analyzer.dist_matrix - ultramatrix)
            im3 = axes[2].imshow(diff, cmap="RdBu_r", aspect="auto", vmin=0, vmax=0.5)
            axes[2].set_title("Absolute Difference")
            axes[2].set_xlabel("Index")
            axes[2].set_ylabel("Index")
            plt.colorbar(im3, ax=axes[2])

        plt.suptitle("Ultrametric Analysis", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_cluster_hierarchy(self):
        """Visualize cluster hierarchy at different levels"""
        if self.analyzer.hierarchy_levels is None:
            self.analyzer.extract_ultrametric_levels()

        # MDS for 2D embedding
        mds = MDS(
            n_components=2, dissimilarity="precomputed", random_state=42, max_iter=300
        )
        coords = mds.fit_transform(self.analyzer.dist_matrix)

        n_levels = min(6, len(self.analyzer.hierarchy_levels))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx in range(n_levels):
            level = self.analyzer.hierarchy_levels[idx]

            ax = axes[idx]
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=level["clusters"],
                cmap="tab20",
                s=50,
                alpha=0.7,
            )
            ax.set_title(
                f"Threshold={level['threshold']:.3f}\n{level['n_clusters']} clusters"
            )
            ax.set_xlabel("MDS1")
            ax.set_ylabel("MDS2")
            ax.grid(True, alpha=0.3)

            # Add colorbar for first plot
            if idx == 0:
                plt.colorbar(scatter, ax=ax, label="Cluster ID")

        # Hide empty subplots
        for idx in range(n_levels, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Hierarchical Clustering Levels", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_ultrametric_tree(self):
        """Plot the ultrametric tree as a network"""
        if self.analyzer.linkage_matrix is None:
            self.analyzer.compute_hierarchical_clustering()

        from scipy.cluster.hierarchy import to_tree

        # Convert to tree structure
        root = to_tree(self.analyzer.linkage_matrix)

        # Create networkx graph
        G = nx.Graph()

        def add_nodes(node, parent=None):
            if node.is_leaf():
                node_id = f"leaf_{node.id}"
                G.add_node(node_id, leaf=True, size=10)
            else:
                node_id = f"node_{node.id}"
                G.add_node(node_id, leaf=False, size=5)

            if parent:
                G.add_edge(parent, node_id, weight=node.dist)

            if not node.is_leaf():
                add_nodes(node.left, node_id)
                add_nodes(node.right, node_id)

        add_nodes(root)

        # Plot tree
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Draw nodes
        leaf_nodes = [n for n in G.nodes if G.nodes[n].get("leaf", False)]
        internal_nodes = [n for n in G.nodes if not G.nodes[n].get("leaf", False)]

        nx.draw_networkx_nodes(
            G, pos, nodelist=internal_nodes, node_color="lightblue", node_size=200
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=leaf_nodes, node_color="lightgreen", node_size=100
        )

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)

        plt.title("Ultrametric Tree Structure")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_abstract_transition_graph(self):
        """Visualize the deterministic abstract MDP"""
        if self.abstract_mdp is None:
            return

        trans_matrix = self.abstract_mdp.get_transition_matrix()

        G = nx.MultiDiGraph()

        # Add nodes
        for h in self.abstract_mdp.H:
            G.add_node(h)

        # Add edges
        for h in self.abstract_mdp.H:
            for a in self.abstract_mdp.A:
                next_h = trans_matrix[h, a]
                G.add_edge(h, next_h, action=a)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        # Draw edges
        edge_colors = plt.cm.tab10(np.linspace(0, 1, len(self.abstract_mdp.A)))
        drawn_edges = set()

        for u, v, data in G.edges(data=True):
            if (u, v) in drawn_edges:
                rad = 0.1
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

            mid = (np.array(pos[u]) + np.array(pos[v])) / 2
            if rad != 0:
                perp = np.array([pos[v][1] - pos[u][1], pos[u][0] - pos[v][0]])
                perp = perp / np.linalg.norm(perp) * 0.2
                mid = mid + perp

            plt.annotate(
                f"a={data['action']}",
                xy=mid,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
            )

        plt.title("Deterministic Abstract MDP (Ultrametric Structure)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# ============ 5. MAIN PIPELINE ============


def run_ultrametric_pipeline(
    env_name="FrozenLake-v1", n_episodes=200, sample_size=100, method="single"
):
    """Run the complete ultrametric analysis pipeline"""

    print("\n" + "=" * 70)
    print("ULTRAMETRIC ANALYSIS FOR TOPOLOGICAL ABSTRACTION")
    print("=" * 70)

    # Create environment
    env = gym.make(env_name)
    print(f"\nEnvironment: {env_name}")

    # Step 1: Collect Wasserstein distances
    print("\n" + "-" * 50)
    print("STEP 1: Computing Wasserstein distances")
    print("-" * 50)

    wasserstein = WassersteinBehavioralDistance(env, n_episodes, sample_size)
    wasserstein.collect_transitions()
    dist_matrix, keys = wasserstein.compute_distance_matrix()

    # Step 2: Ultrametric analysis
    print("\n" + "-" * 50)
    print("STEP 2: Analyzing ultrametric structure")
    print("-" * 50)

    analyzer = UltrametricAnalyzer(dist_matrix, keys)

    # Compute hierarchical clustering
    analyzer.compute_hierarchical_clustering(method=method)

    # Measure ultrametric deviation
    ultrametric_ratio = analyzer.compute_ultrametric_deviation()

    # Extract hierarchy levels
    analyzer.extract_ultrametric_levels(n_levels=8)

    # Get best ultrametric clustering
    best_clustering = analyzer.get_ultrametric_clusters()

    # Step 3: Build abstract MDP
    print("\n" + "-" * 50)
    print("STEP 3: Building abstract MDP from ultrametric structure")
    print("-" * 50)

    abstract_mdp = UltrametricAbstractMDP(
        best_clustering["clusters"], keys, wasserstein.transition_data, env
    )

    # Step 4: Visualizations
    print("\n" + "-" * 50)
    print("STEP 4: Generating visualizations")
    print("-" * 50)

    visualizer = UltrametricVisualizer(analyzer, abstract_mdp)

    visualizer.plot_dendrogram(max_d=best_clustering["threshold"])
    visualizer.plot_ultrametric_heatmap()
    visualizer.plot_cluster_hierarchy()
    visualizer.plot_ultrametric_tree()
    visualizer.plot_abstract_transition_graph()

    # Summary
    print("\n" + "=" * 70)
    print("ULTRAMETRIC ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nUltrametric Properties:")
    print(f"  • Cophenetic correlation: {analyzer.cophenetic_correlation:.4f}")
    print(f"  • Ultrametric ratio: {ultrametric_ratio:.4f}")
    print(f"  • Best threshold: {best_clustering['threshold']:.3f}")
    print(f"  • Number of abstract states: {best_clustering['n_clusters']}")
    print(f"  • Silhouette score: {best_clustering['silhouette']:.3f}")
    print(f"\nAbstract MDP:")
    print(f"  • States: {len(abstract_mdp.H)}")
    print(f"  • Actions: {len(abstract_mdp.A)}")
    print(f"  • Transitions: Deterministic")
    print("=" * 70)

    env.close()

    return {
        "wasserstein": wasserstein,
        "analyzer": analyzer,
        "abstract_mdp": abstract_mdp,
        "visualizer": visualizer,
        "best_clustering": best_clustering,
        "ultrametric_ratio": ultrametric_ratio,
    }


# ============ 6. DIAGNOSTIC SUITE ============


class UltrametricDiagnostics:
    """
    Diagnostic tools for ultrametric analysis
    """

    def __init__(self, env_name="FrozenLake-v1"):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.results = {}

    def analyze_linkage_methods(self, n_episodes=150, sample_size=80):
        """Compare different hierarchical linkage methods"""
        methods = ["single", "complete", "average", "ward"]
        results = []

        print("\nComparing linkage methods:")
        print("-" * 50)

        # Collect data once
        wasserstein = WassersteinBehavioralDistance(self.env, n_episodes, sample_size)
        wasserstein.collect_transitions()
        dist_matrix, keys = wasserstein.compute_distance_matrix()

        for method in methods:
            print(f"\nTesting {method} linkage...")

            analyzer = UltrametricAnalyzer(dist_matrix, keys)
            analyzer.compute_hierarchical_clustering(method=method)

            # Compute ultrametric properties
            ultrametric_ratio = analyzer.compute_ultrametric_deviation()

            # Get best clustering
            analyzer.extract_ultrametric_levels(n_levels=5)
            best = analyzer.get_ultrametric_clusters()

            results.append(
                {
                    "method": method,
                    "cophenetic": analyzer.cophenetic_correlation,
                    "ultrametric_ratio": ultrametric_ratio,
                    "n_clusters": best["n_clusters"],
                    "silhouette": best["silhouette"],
                }
            )

        # Display results
        df = pd.DataFrame(results)
        print("\n" + "=" * 50)
        print("Linkage Method Comparison")
        print("=" * 50)
        print(df.to_string(index=False))

        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics = ["cophenetic", "ultrametric_ratio", "silhouette", "n_clusters"]
        titles = [
            "Cophenetic Correlation",
            "Ultrametric Ratio",
            "Silhouette Score",
            "Number of Clusters",
        ]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            ax.bar(df["method"], df[metric], color="skyblue", edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel("Linkage Method")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for i, v in enumerate(df[metric]):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")

        plt.suptitle("Comparison of Hierarchical Linkage Methods", fontsize=14)
        plt.tight_layout()
        plt.show()

        return df

    def analyze_threshold_sensitivity(self, n_episodes=150, sample_size=80):
        """Analyze sensitivity to clustering threshold"""

        wasserstein = WassersteinBehavioralDistance(self.env, n_episodes, sample_size)
        wasserstein.collect_transitions()
        dist_matrix, keys = wasserstein.compute_distance_matrix()

        analyzer = UltrametricAnalyzer(dist_matrix, keys)
        analyzer.compute_hierarchical_clustering(method="single")
        analyzer.extract_ultrametric_levels(n_levels=20)

        # Extract metrics
        thresholds = []
        n_clusters = []
        silhouettes = []

        for level in analyzer.hierarchy_levels:
            thresholds.append(level["threshold"])
            n_clusters.append(level["n_clusters"])

            if level["n_clusters"] > 1:
                try:
                    sil = silhouette_score(
                        dist_matrix, level["clusters"], metric="precomputed"
                    )
                except:
                    sil = -1
            else:
                sil = -1
            silhouettes.append(sil)

        # Plot sensitivity
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(thresholds, n_clusters, "o-", color="blue")
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("Number of Clusters")
        axes[0].set_title("Clusters vs Threshold")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(thresholds, silhouettes, "o-", color="green")
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette vs Threshold")
        axes[1].grid(True, alpha=0.3)

        # Find optimal threshold
        best_idx = np.argmax(silhouettes)
        axes[1].axvline(
            thresholds[best_idx],
            color="red",
            linestyle="--",
            label=f"Best: {thresholds[best_idx]:.3f}",
        )
        axes[1].legend()

        # Stability
        stability = np.gradient(n_clusters) / np.gradient(thresholds)
        axes[2].plot(thresholds[1:], stability, "o-", color="purple")
        axes[2].set_xlabel("Threshold")
        axes[2].set_ylabel("Rate of Change")
        axes[2].set_title("Cluster Stability")
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("Threshold Sensitivity Analysis", fontsize=14)
        plt.tight_layout()
        plt.show()

        return pd.DataFrame(
            {
                "threshold": thresholds,
                "n_clusters": n_clusters,
                "silhouette": silhouettes,
            }
        )


# ============ 7. MAIN EXECUTION ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ultrametric Analysis for Topological Abstraction"
    )
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
        ],
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument(
        "--method",
        type=str,
        default="single",
        choices=["single", "complete", "average", "ward"],
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run diagnostic suite instead of main pipeline",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare linkage methods"
    )

    args = parser.parse_args()

    if args.diagnose:
        print("\n" + "=" * 70)
        print("ULTRAMETRIC DIAGNOSTIC SUITE")
        print("=" * 70)

        diagnostics = UltrametricDiagnostics(args.env)

        if args.compare:
            diagnostics.analyze_linkage_methods(args.episodes, args.sample)
        else:
            diagnostics.analyze_threshold_sensitivity(args.episodes, args.sample)

    else:
        # Run main pipeline
        results = run_ultrametric_pipeline(
            env_name=args.env,
            n_episodes=args.episodes,
            sample_size=args.sample,
            method=args.method,
        )

    plt.show()
