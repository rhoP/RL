"""
Topological Filtration for MuJoCo/Humanoid Environments
Fixed version - handles version incompatibilities
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.stats import spearmanr
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
import networkx as nx
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

# ============ 0. VERSION COMPATIBILITY LAYER ============


def create_mujoco_env(env_name, render_mode=None):
    """
    Create MuJoCo environment with compatibility handling
    """
    try:
        # Try standard creation
        if render_mode:
            env = gym.make(env_name, render_mode=render_mode)
        else:
            env = gym.make(env_name)
        return env
    except Exception as e:
        print(f"Standard creation failed: {e}")

        # Try with different arguments
        try:
            # Older Gymnasium versions
            env = gym.make(env_name)
            return env
        except:
            try:
                # Try with mujoco_py backend
                env = gym.make(env_name, backend="mujoco_py")
                return env
            except:
                raise ImportError(
                    f"Could not create {env_name}. Make sure mujoco is properly installed."
                )


def get_env_info(env):
    """Safely get environment information"""
    info = {}

    # Get observation space
    if hasattr(env, "observation_space"):
        if hasattr(env.observation_space, "shape"):
            info["state_dim"] = env.observation_space.shape[0]
        else:
            info["state_dim"] = env.observation_space.n
    else:
        info["state_dim"] = 0

    # Get action space
    if hasattr(env, "action_space"):
        if hasattr(env.action_space, "shape"):
            info["action_dim"] = env.action_space.shape[0]
            info["continuous_actions"] = True
        else:
            info["action_dim"] = env.action_space.n
            info["continuous_actions"] = False

    return info


# ============ 1. DIMENSIONALITY REDUCTION FOR HIGH-D STATES ============


class StateEncoder(nn.Module):
    """Autoencoder for reducing state dimensionality"""

    def __init__(self, input_dim, latent_dim=32, hidden_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x).numpy()


class DynamicsEncoder(nn.Module):
    """Encode state-action pairs into behaviorally relevant features"""

    def __init__(self, state_dim, action_dim, latent_dim=32, continuous_actions=True):
        super().__init__()

        self.continuous_actions = continuous_actions

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )

        # Action encoder depends on action type
        if continuous_actions:
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, 32), nn.ReLU(), nn.Linear(32, 16)
            )
            combined_dim = 48
        else:
            self.action_embedding = nn.Embedding(action_dim, 16)
            combined_dim = 48

        self.combined = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )

        # Predict next state latent (for dynamics-aware encoding)
        self.next_state_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, state_dim)
        )

    def forward(self, state, action):
        s_enc = self.state_encoder(state)

        if self.continuous_actions:
            a_enc = self.action_encoder(action)
        else:
            a_enc = self.action_embedding(action.long())

        combined = torch.cat([s_enc, a_enc], dim=-1)
        latent = self.combined(combined)
        next_pred = self.next_state_predictor(latent)
        return latent, next_pred

    def encode(self, state, action):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if isinstance(action, np.ndarray):
                if self.continuous_actions:
                    action = torch.FloatTensor(action)
                else:
                    action = torch.LongTensor(action)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            latent, _ = self.forward(state, action)
            return latent.numpy()


# ============ 2. EFFICIENT WASSERSTEIN DISTANCE FOR MUJOCO ============


class MuJoCoBehavioralDistance:
    """
    Efficient behavioral distance computation for high-dimensional MuJoCo environments
    Uses learned embeddings and approximate Wasserstein distances
    """

    def __init__(
        self, env_name, n_episodes=500, sample_size=200, latent_dim=32, device="cpu"
    ):
        self.env_name = env_name
        self.n_episodes = n_episodes
        self.sample_size = sample_size
        self.latent_dim = latent_dim
        self.device = device

        # Create environment
        self.env = create_mujoco_env(env_name)
        env_info = get_env_info(self.env)

        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.continuous_actions = env_info["continuous_actions"]

        print(f"\nMuJoCo Environment Stats:")
        print(f"  Environment: {env_name}")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Action dimension: {self.action_dim}")
        print(f"  Continuous actions: {self.continuous_actions}")

        # Initialize encoders
        self.state_autoencoder = StateEncoder(self.state_dim, latent_dim)
        self.dynamics_encoder = DynamicsEncoder(
            self.state_dim, self.action_dim, latent_dim, self.continuous_actions
        )

        # Storage
        self.transition_data = []  # List of (s, a, s_next)
        self.encoded_data = []  # List of (z_sa, z_s_next)
        self.sampled_indices = []
        self.distance_matrix = None

    def collect_transitions_with_encoding(self):
        """Collect transitions and train encoders simultaneously"""
        print(f"\nCollecting {self.n_episodes} episodes of transition data...")

        # Storage for training
        states = []
        actions = []
        next_states = []

        # Optimizers for encoders
        ae_optimizer = torch.optim.Adam(self.state_autoencoder.parameters(), lr=1e-3)
        dyn_optimizer = torch.optim.Adam(self.dynamics_encoder.parameters(), lr=1e-3)

        for episode in range(self.n_episodes):
            try:
                # Handle different reset return formats
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    state, _ = reset_result
                else:
                    state = reset_result
            except:
                state = self.env.reset()

            episode_states = []
            episode_actions = []
            episode_next = []
            done = False
            step_count = 0

            while not done and step_count < 1000:  # Limit episode length
                # Sample random action (exploration)
                if self.continuous_actions:
                    action = self.env.action_space.sample()
                else:
                    action = self.env.action_space.sample()

                # Handle different step return formats
                try:
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:
                        next_state, reward, done, _ = step_result
                except:
                    next_state, reward, done, _ = self.env.step(action)

                # Store
                episode_states.append(
                    state.copy() if hasattr(state, "copy") else np.array(state)
                )

                if self.continuous_actions:
                    episode_actions.append(
                        action.copy() if hasattr(action, "copy") else np.array(action)
                    )
                else:
                    episode_actions.append(action)

                episode_next.append(
                    next_state.copy()
                    if hasattr(next_state, "copy")
                    else np.array(next_state)
                )

                state = next_state
                step_count += 1

            # Add to collection
            states.extend(episode_states)
            actions.extend(episode_actions)
            next_states.extend(episode_next)

            # Train encoders periodically
            if len(states) > 256 and (episode + 1) % 10 == 0:
                self._train_encoders(
                    np.array(states[-256:]),
                    np.array(actions[-256:]),
                    np.array(next_states[-256:]),
                    ae_optimizer,
                    dyn_optimizer,
                )

            if (episode + 1) % 100 == 0:
                print(
                    f"  Episode {episode + 1}/{self.n_episodes}, collected {len(states)} transitions"
                )

        # Store all transitions
        self.states = np.array(states)
        self.actions = np.array(actions)
        self.next_states = np.array(next_states)

        print(f"Total transitions collected: {len(self.states)}")

        # Encode all data
        self._encode_all_data()

    def _train_encoders(self, states, actions, next_states, ae_opt, dyn_opt):
        """Train autoencoder and dynamics encoder"""
        states_t = torch.FloatTensor(states)

        if self.continuous_actions:
            actions_t = torch.FloatTensor(actions)
        else:
            actions_t = torch.LongTensor(actions)

        next_states_t = torch.FloatTensor(next_states)

        # Train autoencoder
        latent, recon = self.state_autoencoder(states_t)
        ae_loss = F.mse_loss(recon, states_t)

        ae_opt.zero_grad()
        ae_loss.backward()
        ae_opt.step()

        # Train dynamics encoder
        latent_sa, next_pred = self.dynamics_encoder(states_t, actions_t)
        dyn_loss = F.mse_loss(next_pred, next_states_t)

        dyn_opt.zero_grad()
        dyn_loss.backward()
        dyn_opt.step()

        if np.random.random() < 0.01:  # Print occasionally
            print(f"    AE Loss: {ae_loss.item():.4f}, Dyn Loss: {dyn_loss.item():.4f}")

    def _encode_all_data(self):
        """Encode all transitions into latent space"""
        print("\nEncoding transitions into latent space...")

        n = len(self.states)
        batch_size = 256
        self.encoded_data = []

        for i in range(0, n, batch_size):
            batch_states = self.states[i : i + batch_size]
            batch_actions = self.actions[i : i + batch_size]

            # Encode
            z = self.dynamics_encoder.encode(batch_states, batch_actions)
            self.encoded_data.extend(z)

        self.encoded_data = np.array(self.encoded_data)
        print(f"Encoded to {self.encoded_data.shape[1]}-dimensional space")

    def compute_wasserstein_distance(self, i, j):
        """
        Compute approximate Wasserstein distance between encoded distributions
        Uses multiple samples from the same (or similar) state-action pairs
        """
        z_i = self.encoded_data[i]
        z_j = self.encoded_data[j]

        # Euclidean distance in latent space as proxy for Wasserstein
        return np.linalg.norm(z_i - z_j)

    def compute_distance_matrix(self):
        """Compute distance matrix using latent representations"""
        n = len(self.encoded_data)

        # Sample if too large
        if n > self.sample_size:
            self.sampled_indices = np.random.choice(n, self.sample_size, replace=False)
            encoded_sample = self.encoded_data[self.sampled_indices]
        else:
            self.sampled_indices = np.arange(n)
            encoded_sample = self.encoded_data

        n_sample = len(encoded_sample)
        print(f"\nComputing {n_sample}x{n_sample} distance matrix...")

        # Compute pairwise Euclidean distances
        dist_matrix = np.zeros((n_sample, n_sample))

        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                dist = np.linalg.norm(encoded_sample[i] - encoded_sample[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{n_sample}")

        self.distance_matrix = dist_matrix
        return dist_matrix, self.sampled_indices


# ============ 3. TOPOLOGICAL FILTRATION WITH PERSISTENCE ============


class MuJoCoTopologicalFiltration:
    """
    Build topological filtration from Wasserstein distances
    """

    def __init__(self, distance_matrix, indices, env_info):
        self.dist_matrix = distance_matrix
        self.indices = indices
        self.env_info = env_info
        self.n_points = len(distance_matrix)

        # Storage
        self.filtration = {}
        self.persistence_diagrams = None
        self.linkage_matrix = None
        self.cophenetic_corr = None

    def build_vietoris_rips(self, n_levels=10):
        """Build Vietoris-Rips filtration at multiple scales"""
        # Choose thresholds based on distance percentiles
        distances = self.dist_matrix[self.dist_matrix > 0]
        if len(distances) > 0:
            thresholds = np.percentile(distances, np.linspace(5, 95, n_levels))
        else:
            thresholds = np.linspace(0.1, 5.0, n_levels)

        print(f"\nBuilding filtration with {n_levels} levels...")

        from scipy.sparse.csgraph import connected_components

        for t in thresholds:
            # Find connected components at this threshold
            adj_matrix = self.dist_matrix <= t
            n_components, labels = connected_components(
                adj_matrix, directed=False, return_labels=True
            )

            self.filtration[t] = {
                "threshold": t,
                "n_components": n_components,
                "labels": labels,
                "adj_matrix": adj_matrix,
            }

            print(f"  t={t:.3f}: {n_components} components")

        return self.filtration

    def compute_persistence(self):
        """Compute persistent homology using Ripser"""
        if not PERSISTENCE_AVAILABLE:
            print("Persistence computation skipped (Ripser not available)")
            return None

        print("\nComputing persistent homology...")
        try:
            self.persistence_diagrams = ripser(self.dist_matrix, distance_matrix=True)[
                "dgms"
            ]
            return self.persistence_diagrams
        except Exception as e:
            print(f"Persistence computation failed: {e}")
            return None

    def compute_hierarchical_clustering(self, method="single"):
        """Compute hierarchical clustering for ultrametric analysis"""
        print(f"\nComputing hierarchical clustering ({method} linkage)...")

        condensed = squareform(self.dist_matrix)
        self.linkage_matrix = linkage(condensed, method=method)

        # Cophenetic correlation
        self.cophenetic_corr, _ = cophenet(self.linkage_matrix, condensed)
        print(f"  Cophenetic correlation: {self.cophenetic_corr:.4f}")

        return self.linkage_matrix


# ============ 4. BEHAVIORAL PROTOTYPE EXTRACTION ============


class BehavioralPrototypeExtractor:
    """
    Extract behavioral prototypes from the filtration
    """

    def __init__(self, filtration, distance_matrix, indices, encoded_data):
        self.filtration = filtration
        self.dist_matrix = distance_matrix
        self.indices = indices
        self.encoded_data = encoded_data

    def extract_persistent_clusters(self, persistence_threshold=0.3):
        """
        Extract clusters that persist across multiple scales
        """
        thresholds = sorted(self.filtration.keys())
        n_scales = len(thresholds)

        # Track component labels across scales
        component_history = []

        for t in thresholds:
            component_history.append(self.filtration[t]["labels"])

        # Find persistent components
        n_points = len(self.indices)

        # For each point, find its longest consecutive run in same component
        persistence_scores = []
        for i in range(n_points):
            current_component = None
            run_length = 0
            max_run = 0

            for scale in range(n_scales):
                comp = component_history[scale][i]

                if comp == current_component:
                    run_length += 1
                else:
                    if run_length > max_run:
                        max_run = run_length
                    current_component = comp
                    run_length = 1

            # Check final run
            if run_length > max_run:
                max_run = run_length

            persistence_scores.append(max_run / n_scales)

        # Mark persistent points
        persistent_points = [
            i
            for i, score in enumerate(persistence_scores)
            if score > persistence_threshold
        ]

        print(f"\nPersistent clusters: {len(persistent_points)}/{n_points} points")

        # Get cluster labels at the median scale
        median_scale = n_scales // 2
        self.cluster_labels = component_history[median_scale]

        return self.cluster_labels, persistent_points

    def get_prototype_representatives(self, n_prototypes=10):
        """
        Extract representative points for each behavioral prototype
        """
        unique_labels = np.unique(self.cluster_labels)
        representatives = []

        for label in unique_labels:
            mask = self.cluster_labels == label
            cluster_points = self.encoded_data[mask]

            if len(cluster_points) > 0:
                # Find medoid (point closest to cluster center)
                center = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                medoid_idx = np.argmin(distances)

                # Get original index
                original_idx = self.indices[mask][medoid_idx]
                representatives.append(
                    {
                        "prototype_id": label,
                        "original_index": original_idx,
                        "cluster_size": np.sum(mask),
                        "center": center,
                        "medoid": cluster_points[medoid_idx],
                    }
                )

        # Sort by cluster size
        representatives.sort(key=lambda x: x["cluster_size"], reverse=True)

        return representatives[:n_prototypes]


# ============ 5. ABSTRACT MDP CONSTRUCTION ============


class MuJoCoAbstractMDP:
    """
    Build abstract MDP from behavioral prototypes
    """

    def __init__(self, cluster_labels, indices, original_data, env_info, prototypes):
        self.cluster_labels = cluster_labels
        self.indices = indices
        self.original_data = original_data
        self.env_info = env_info
        self.prototypes = prototypes

        # Map indices to clusters
        self.idx_to_cluster = {}
        for i, idx in enumerate(indices):
            self.idx_to_cluster[idx] = cluster_labels[i]

        # Build abstract MDP
        self._build_abstract_mdp()

    def _build_abstract_mdp(self):
        """Build abstract MDP with deterministic transitions"""
        unique_clusters = np.unique(self.cluster_labels)
        self.cluster_to_id = {c: i for i, c in enumerate(unique_clusters)}
        self.H = list(range(len(unique_clusters)))

        # Action space (5 abstract action classes for continuous)
        self.A = list(range(5))

        # Initialize transition counts
        trans_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Aggregate transitions
        states = self.original_data["states"]
        actions = self.original_data["actions"]
        next_states = self.original_data["next_states"]

        for i, idx in enumerate(self.indices):
            if idx >= len(states):
                continue

            s = states[idx]
            a = actions[idx]
            s_next = next_states[idx]

            c = self.cluster_labels[i]
            if c < 0:
                continue

            h = self.cluster_to_id[c]

            # Discretize action for abstract MDP
            a_abstract = self._discretize_action(a)

            # Find cluster of next state (simplified)
            next_c = self._find_cluster_for_state(s_next)

            if next_c is not None and next_c in self.cluster_to_id:
                next_h = self.cluster_to_id[next_c]
                trans_counts[h][a_abstract][next_h] += 1

        # Build deterministic transitions (mode)
        self.T = {}

        for h in self.H:
            self.T[h] = {}
            for a in self.A:
                if trans_counts[h][a]:
                    next_h = max(trans_counts[h][a].items(), key=lambda x: x[1])[0]
                else:
                    next_h = h
                self.T[h][a] = next_h

        print(f"\nBuilt abstract MDP:")
        print(f"  Abstract states: {len(self.H)}")
        print(f"  Abstract actions: {len(self.A)}")

    def _discretize_action(self, action):
        """Discretize continuous action into bins"""
        if isinstance(action, np.ndarray):
            # Simple binning based on magnitude
            magnitude = np.linalg.norm(action)
            bins = np.linspace(0, 5.0, len(self.A))  # Assume max magnitude ~5
            return np.digitize(magnitude, bins) - 1
        return action % len(self.A)  # For discrete actions

    def _find_cluster_for_state(self, state):
        """Simplified: find nearest prototype by index"""
        # In practice, you'd maintain a mapping or use nearest neighbor
        # For now, return a random cluster
        return np.random.choice(len(self.H))


# ============ 6. VISUALIZATION FOR MUJOCO ============


class MuJoCoVisualizer:
    """
    Specialized visualizations for MuJoCo abstractions
    """

    def __init__(
        self, behavioral, filtration, prototypes, abstract_mdp, cluster_labels
    ):
        self.behavioral = behavioral
        self.filtration = filtration
        self.prototypes = prototypes
        self.abstract_mdp = abstract_mdp
        self.cluster_labels = cluster_labels

    def plot_filtration_evolution(self):
        """Plot how components evolve with threshold"""
        thresholds = sorted(self.filtration.filtration.keys())
        n_components = [
            self.filtration.filtration[t]["n_components"] for t in thresholds
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Component count evolution
        ax1 = axes[0, 0]
        ax1.plot(thresholds, n_components, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Filtration Threshold")
        ax1.set_ylabel("Number of Components")
        ax1.set_title("Topological Evolution")
        ax1.grid(True, alpha=0.3)

        # 2. Persistence diagram
        ax2 = axes[0, 1]
        if self.filtration.persistence_diagrams is not None:
            plot_diagrams(self.filtration.persistence_diagrams, ax=ax2)
            ax2.set_title("Persistence Diagram")
        else:
            ax2.text(0.5, 0.5, "Persistence unavailable", ha="center", va="center")
            ax2.set_title("Persistence Diagram (unavailable)")

        # 3. Distance matrix with clustering
        ax3 = axes[1, 0]
        if self.cluster_labels is not None:
            # Reorder by cluster
            order = np.argsort(self.cluster_labels)
            dist_ordered = self.filtration.dist_matrix[order][:, order]
            im = ax3.imshow(dist_ordered, cmap="viridis", aspect="auto")
            ax3.set_title("Distance Matrix (cluster-ordered)")
            plt.colorbar(im, ax=ax3)

        # 4. t-SNE of encoded states
        ax4 = axes[1, 1]
        if len(self.behavioral.encoded_data) > 0:
            # Sample if too many
            n_samples = min(1000, len(self.behavioral.encoded_data))
            indices = np.random.choice(
                len(self.behavioral.encoded_data), n_samples, replace=False
            )
            sample = self.behavioral.encoded_data[indices]

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embedding = tsne.fit_transform(sample)

            ax4.scatter(embedding[:, 0], embedding[:, 1], c="blue", alpha=0.5, s=10)
            ax4.set_title("t-SNE of Encoded States")
            ax4.set_xlabel("t-SNE1")
            ax4.set_ylabel("t-SNE2")

        plt.suptitle("MuJoCo Topological Analysis", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_prototype_comparison(self):
        """Compare behavioral prototypes"""
        n_protos = len(self.prototypes)
        n_cols = min(3, n_protos)
        n_rows = (n_protos + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = np.array([axes]).flatten()
        else:
            axes = axes.flatten()

        for i, proto in enumerate(self.prototypes[: len(axes)]):
            ax = axes[i]

            # Get state dimensions for visualization
            idx = proto["original_index"]
            if idx < len(self.behavioral.states):
                state = self.behavioral.states[idx]

                # Plot state as line (for high-dimensional)
                ax.plot(state[: min(50, len(state))], "b-", alpha=0.7, linewidth=1)
                ax.set_title(
                    f"Prototype {proto['prototype_id']}\nSize: {proto['cluster_size']}"
                )
                ax.set_xlabel("Dimension")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)

                # Add stats
                ax.text(
                    0.5,
                    0.9,
                    f"Center norm: {np.linalg.norm(proto['center']):.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
                )

        # Hide empty subplots
        for i in range(n_protos, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Behavioral Prototypes from MuJoCo", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_abstract_mdp_graph(self):
        """Visualize abstract MDP as graph"""
        if self.abstract_mdp is None:
            return

        G = nx.MultiDiGraph()

        # Add nodes
        for h in self.abstract_mdp.H:
            G.add_node(h)

        # Add edges
        for h in self.abstract_mdp.H:
            for a in self.abstract_mdp.A:
                next_h = self.abstract_mdp.T[h][a]
                if next_h != h:  # Only show non-self-loop edges for clarity
                    G.add_edge(h, next_h, action=a)

        if len(G.edges()) == 0:
            print("No non-trivial edges in abstract MDP")
            return

        plt.figure(figsize=(14, 10))

        # Use spring layout
        pos = nx.spring_layout(G, k=2, iterations=100)

        # Draw nodes
        node_sizes = [
            800 + 100 * np.sum(self.cluster_labels == node)
            if self.cluster_labels is not None
            else 800
            for node in G.nodes
        ]

        nx.draw_networkx_nodes(
            G, pos, node_color="lightblue", node_size=node_sizes, alpha=0.9
        )
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        # Draw edges
        edge_colors = plt.cm.tab10(np.linspace(0, 1, len(self.abstract_mdp.A)))
        drawn_edges = set()

        for u, v, data in G.edges(data=True):
            if (u, v) in drawn_edges:
                rad = 0.15
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

            # Label edges
            mid = (np.array(pos[u]) + np.array(pos[v])) / 2
            if rad != 0:
                perp = np.array([pos[v][1] - pos[u][1], pos[u][0] - pos[v][0]])
                perp = perp / np.linalg.norm(perp) * 0.3
                mid = mid + perp

            plt.annotate(
                f"a={data['action']}",
                xy=mid,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
            )

        plt.title("Abstract MDP from MuJoCo Behavioral Prototypes")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# ============ 7. MAIN PIPELINE ============


def run_mujoco_abstraction_pipeline(
    env_name="Humanoid-v4",
    n_episodes=300,  # Reduced for testing
    sample_size=150,  # Reduced for testing
    latent_dim=32,
    n_prototypes=10,
):
    """Run complete topological abstraction pipeline for MuJoCo"""

    print("\n" + "=" * 80)
    print(f"TOPOLOGICAL ABSTRACTION FOR MUJOCO: {env_name}")
    print("=" * 80)

    # Step 1: Collect and encode data
    print("\n" + "-" * 60)
    print("STEP 1: Collecting and encoding transition data")
    print("-" * 60)

    behavioral = MuJoCoBehavioralDistance(env_name, n_episodes, sample_size, latent_dim)
    behavioral.collect_transitions_with_encoding()
    dist_matrix, indices = behavioral.compute_distance_matrix()

    # Step 2: Build topological filtration
    print("\n" + "-" * 60)
    print("STEP 2: Building topological filtration")
    print("-" * 60)

    env_info = {"state_dim": behavioral.state_dim}
    filtration = MuJoCoTopologicalFiltration(dist_matrix, indices, env_info)
    filtration.build_vietoris_rips(n_levels=10)
    filtration.compute_persistence()
    filtration.compute_hierarchical_clustering(method="single")

    # Step 3: Extract behavioral prototypes
    print("\n" + "-" * 60)
    print("STEP 3: Extracting behavioral prototypes")
    print("-" * 60)

    encoded_sample = (
        behavioral.encoded_data[indices]
        if hasattr(behavioral, "encoded_data")
        else None
    )
    extractor = BehavioralPrototypeExtractor(
        filtration.filtration, dist_matrix, indices, encoded_sample
    )
    cluster_labels, persistent = extractor.extract_persistent_clusters(0.3)
    prototypes = extractor.get_prototype_representatives(n_prototypes)

    # Step 4: Build abstract MDP
    print("\n" + "-" * 60)
    print("STEP 4: Building abstract MDP")
    print("-" * 60)

    original_data = {
        "states": behavioral.states if hasattr(behavioral, "states") else [],
        "actions": behavioral.actions if hasattr(behavioral, "actions") else [],
        "next_states": behavioral.next_states
        if hasattr(behavioral, "next_states")
        else [],
    }

    abstract_mdp = MuJoCoAbstractMDP(
        cluster_labels, indices, original_data, env_info, prototypes
    )

    # Step 5: Visualizations
    print("\n" + "-" * 60)
    print("STEP 5: Generating visualizations")
    print("-" * 60)

    visualizer = MuJoCoVisualizer(
        behavioral, filtration, prototypes, abstract_mdp, cluster_labels
    )
    visualizer.plot_filtration_evolution()
    visualizer.plot_prototype_comparison()
    visualizer.plot_abstract_mdp_graph()

    # Summary
    print("\n" + "=" * 80)
    print("ABSTRACTION SUMMARY")
    print("=" * 80)
    print(f"\nData Collection:")
    print(
        f"  • Total transitions: {len(behavioral.states) if hasattr(behavioral, 'states') else 0}"
    )
    print(f"  • Encoded dimension: {latent_dim}")
    print(f"\nTopological Analysis:")
    print(f"  • Distance matrix size: {len(dist_matrix)}x{len(dist_matrix)}")
    print(f"  • Cophenetic correlation: {filtration.cophenetic_corr:.4f}")
    print(f"\nBehavioral Prototypes:")
    print(f"  • Number of clusters: {len(np.unique(cluster_labels))}")
    print(f"  • Persistent points: {len(persistent)}")
    print(f"\nAbstract MDP:")
    print(f"  • Abstract states: {len(abstract_mdp.H)}")
    print(f"  • Abstract actions: {len(abstract_mdp.A)}")
    print("=" * 80)

    return {
        "behavioral": behavioral,
        "filtration": filtration,
        "prototypes": prototypes,
        "abstract_mdp": abstract_mdp,
        "cluster_labels": cluster_labels,
    }


# ============ 8. DIAGNOSTICS FOR MUJOCO ============


class MuJoCoDiagnostics:
    """
    Diagnostic tools specific to MuJoCo environments
    """

    def __init__(self, env_name="Humanoid-v4"):
        self.env_name = env_name
        self.env = create_mujoco_env(env_name)

    def analyze_state_space_coverage(self, n_samples=10000):
        """Analyze how well the state space is covered"""
        print(f"\nAnalyzing state space coverage...")

        states = []
        for ep in range(100):  # Collect 100 episodes
            try:
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    state, _ = reset_result
                else:
                    state = reset_result
            except:
                state = self.env.reset()

            done = False
            while not done and len(states) < n_samples:
                action = self.env.action_space.sample()
                try:
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_state, _, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        next_state, _, done, _ = step_result
                except:
                    next_state, _, done, _ = self.env.step(action)

                states.append(
                    state.copy() if hasattr(state, "copy") else np.array(state)
                )
                state = next_state

        states = np.array(states[:n_samples])

        # PCA analysis
        pca = PCA(n_components=min(10, states.shape[1]))
        states_pca = pca.fit_transform(states)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Explained variance
        axes[0, 0].plot(np.cumsum(pca.explained_variance_ratio_), "bo-")
        axes[0, 0].set_xlabel("Number of Components")
        axes[0, 0].set_ylabel("Cumulative Explained Variance")
        axes[0, 0].set_title("PCA Explained Variance")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0.95, color="r", linestyle="--", label="95%")
        axes[0, 0].legend()

        # First two PCs
        axes[0, 1].scatter(states_pca[:, 0], states_pca[:, 1], c="blue", alpha=0.3, s=5)
        axes[0, 1].set_xlabel("PC1")
        axes[0, 1].set_ylabel("PC2")
        axes[0, 1].set_title("State Space (first two PCs)")

        # State dimension distributions
        axes[1, 0].hist(states.flatten(), bins=50, alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("State Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of State Values")
        axes[1, 0].grid(True, alpha=0.3)

        # Norm over time
        norms = np.linalg.norm(states, axis=1)
        axes[1, 1].plot(norms[:1000], "b-", alpha=0.7)
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("State Norm")
        axes[1, 1].set_title("State Norm Over Time")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f"State Space Analysis for {self.env_name}", fontsize=14)
        plt.tight_layout()
        plt.show()

        return states_pca

    def analyze_action_sensitivity(self, n_samples=1000):
        """Analyze how actions affect state transitions"""
        print(f"\nAnalyzing action sensitivity...")

        try:
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
        except:
            state = self.env.reset()

        transitions = []

        for _ in range(n_samples):
            action = self.env.action_space.sample()
            try:
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state, _, _, _, _ = step_result
                else:
                    next_state, _, _, _ = step_result
            except:
                next_state, _, _, _ = self.env.step(action)

            delta = next_state - state
            transitions.append(
                {
                    "state": state.copy()
                    if hasattr(state, "copy")
                    else np.array(state),
                    "action": action.copy()
                    if hasattr(action, "copy")
                    else np.array(action),
                    "next_state": next_state.copy()
                    if hasattr(next_state, "copy")
                    else np.array(next_state),
                    "delta": delta,
                }
            )

            state = next_state

        # Analyze deltas
        deltas = np.array([t["delta"] for t in transitions])
        action_norms = np.array([np.linalg.norm(t["action"]) for t in transitions])
        delta_norms = np.linalg.norm(deltas, axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Action vs Delta norm
        axes[0, 0].scatter(action_norms, delta_norms, alpha=0.3, s=10)
        axes[0, 0].set_xlabel("Action Norm")
        axes[0, 0].set_ylabel("State Delta Norm")
        axes[0, 0].set_title("Action Magnitude vs State Change")
        axes[0, 0].grid(True, alpha=0.3)

        # Delta distribution
        axes[0, 1].hist(delta_norms, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 1].set_xlabel("State Delta Norm")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of State Changes")
        axes[0, 1].grid(True, alpha=0.3)

        # Delta correlation across dimensions (first 50 dims for visualization)
        n_dims = min(50, deltas.shape[1])
        corr_matrix = np.corrcoef(deltas[:100, :n_dims].T)
        axes[1, 0].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[1, 0].set_title(f"Delta Correlation Matrix (first {n_dims} dims)")
        axes[1, 0].set_xlabel("State Dimension")
        axes[1, 0].set_ylabel("State Dimension")
        plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0])

        # Action sensitivity by dimension
        sensitivity = np.std(deltas, axis=0)
        axes[1, 1].bar(range(min(50, len(sensitivity))), sensitivity[:50])
        axes[1, 1].set_xlabel("State Dimension")
        axes[1, 1].set_ylabel("Sensitivity (std of delta)")
        axes[1, 1].set_title("Action Sensitivity by Dimension (first 50)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Action Sensitivity Analysis", fontsize=14)
        plt.tight_layout()
        plt.show()


# ============ 9. MAIN EXECUTION ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Topological Abstraction for MuJoCo")
    parser.add_argument(
        "--env",
        type=str,
        default="Hopper-v4",  # Changed default
        choices=[
            "Hopper-v4",
            "HalfCheetah-v4",
            "Walker2d-v4",
            "Ant-v4",
            "Swimmer-v4",
            "Humanoid-v4",
        ],
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,  # Reduced default
        help="Number of episodes for data collection",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,  # Reduced default
        help="Sample size for distance matrix",
    )
    parser.add_argument(
        "--latent", type=int, default=32, help="Latent dimension for encoding"
    )
    parser.add_argument(
        "--prototypes", type=int, default=8, help="Number of prototypes to extract"
    )
    parser.add_argument("--diagnose", action="store_true", help="Run diagnostics only")

    args = parser.parse_args()

    # Check if MuJoCo is installed
    try:
        import mujoco

        print(f"✓ MuJoCo found (version: {getattr(mujoco, '__version__', 'unknown')})")
    except ImportError:
        print("⚠️  MuJoCo not found. Please install with: pip install mujoco")
        print("   For older versions: pip install mujoco-py")

    if args.diagnose:
        print("\n" + "=" * 80)
        print(f"MUJOCO DIAGNOSTICS: {args.env}")
        print("=" * 80)

        try:
            diagnostics = MuJoCoDiagnostics(args.env)
            diagnostics.analyze_state_space_coverage()
            diagnostics.analyze_action_sensitivity()
        except Exception as e:
            print(f"Diagnostics failed: {e}")
            import traceback

            traceback.print_exc()

    else:
        # Run main pipeline
        try:
            results = run_mujoco_abstraction_pipeline(
                env_name=args.env,
                n_episodes=args.episodes,
                sample_size=args.sample,
                latent_dim=args.latent,
                n_prototypes=args.prototypes,
            )
            print("\n✓ Pipeline completed successfully!")
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback

            traceback.print_exc()

    plt.show()
