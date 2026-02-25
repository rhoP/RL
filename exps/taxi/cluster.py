import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import Dict, List, Tuple, Optional
import os
from collections import defaultdict
import pickle
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.gridspec as gridspec


warnings.filterwarnings("ignore")

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class TaxiClusterAnalyzer:
    """
    Cluster analysis for Taxi environment
    """

    def __init__(self, model_path: str = None, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

        # Create environment - important: use render_mode=None for training/collection
        self.env = gym.make("Taxi-v3", render_mode=None)
        self.env.reset(seed=seed)

        # Environment specs
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        # Taxi specific constants
        self.grid_size = 5  # 5x5 grid
        self.n_locations = 4  # R, G, Y, B

        # Action names
        self.action_names = {
            0: "South",
            1: "North",
            2: "East",
            3: "West",
            4: "Pickup",
            5: "Dropoff",
        }

        # Action symbols for visualization
        self.action_symbols = {
            0: "↓",  # South
            1: "↑",  # North
            2: "→",  # East
            3: "←",  # West
            4: "P",  # Pickup
            5: "D",  # Dropoff
        }

        # Locations in grid coordinates (row, col) - Taxi uses 0-indexed rows from top
        self.locations = {
            0: (0, 0),  # R
            1: (0, 4),  # G
            2: (4, 0),  # Y
            3: (4, 3),  # B
        }

        # Color mapping for locations
        self.location_colors = {0: "red", 1: "green", 2: "yellow", 3: "blue"}

        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = DQN.load(model_path)
            print(f"Loaded model from {model_path}")

        # Storage
        self.collected_states = None
        self.collected_actions = None
        self.collected_values = None
        self.collected_taxi_states = (
            None  # Decoded (taxi_row, taxi_col, passenger, destination)
        )
        self.collected_trajectories = []

        # Clustering results
        self.clusters = []
        self.state_to_cluster = {}
        self.cluster_labels = None

    def decode_state(self, state_idx: int) -> Tuple[int, int, int, int]:
        """
        Decode Taxi state index into components
        State = ((taxi_row * 5 + taxi_col) * 5 + passenger) * 4 + destination
        """
        if isinstance(state_idx, np.ndarray):
            state_idx = int(state_idx)

        destination = state_idx % 4
        state_idx = state_idx // 4
        passenger = state_idx % 5
        state_idx = state_idx // 5
        taxi_col = state_idx % 5
        taxi_row = state_idx // 5

        return (taxi_row, taxi_col, passenger, destination)

    def encode_state(
        self, taxi_row: int, taxi_col: int, passenger: int, destination: int
    ) -> int:
        """Encode Taxi state components into index"""
        return int((((taxi_row * 5 + taxi_col) * 5 + passenger) * 4 + destination))

    def train_dqn(
        self, total_timesteps: int = 50000, save_path: str = "models/taxi_dqn"
    ):
        """
        Train a DQN agent on Taxi-v3
        """
        print("\n" + "=" * 60)
        print("Training DQN on Taxi-v3")
        print("=" * 60)

        # Create environment
        env = gym.make("Taxi-v3")
        env = Monitor(env)

        # Initialize DQN
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=1,
            seed=self.seed,
        )

        # Train
        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        training_time = time.time() - start_time

        # Evaluate
        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=100)

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save model
        os.makedirs("models", exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

        return self.model

    def collect_trajectories(
        self, n_episodes: int = 500, deterministic: bool = True
    ) -> List[Dict]:
        """
        Collect trajectories by running the policy
        Fixed to avoid the unhashable type error
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        print(f"\nCollecting {n_episodes} trajectories...")

        trajectories = []
        all_states = []
        all_actions = []
        all_values = []
        all_taxi_states = []

        for episode in range(n_episodes):
            # Reset environment - get initial state as integer
            state, _ = self.env.reset()
            # Ensure state is integer
            if isinstance(state, np.ndarray):
                state = int(state.item()) if state.size == 1 else int(state[0])

            done = False
            total_reward = 0
            trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "taxi_states": [],
                "success": False,
            }

            while not done:
                # Get action - ensure state is integer
                action, _ = self.model.predict(int(state), deterministic=deterministic)

                # Take step
                next_state, reward, terminated, truncated, _ = self.env.step(
                    int(action)
                )
                done = terminated or truncated

                # Ensure next_state is integer
                if isinstance(next_state, np.ndarray):
                    next_state = (
                        int(next_state.item())
                        if next_state.size == 1
                        else int(next_state[0])
                    )

                # Decode state
                taxi_state = self.decode_state(int(state))

                # Store data
                trajectory["states"].append(int(state))
                trajectory["actions"].append(int(action))
                trajectory["rewards"].append(float(reward))
                trajectory["next_states"].append(int(next_state))
                trajectory["taxi_states"].append(taxi_state)

                all_states.append(int(state))
                all_actions.append(int(action))
                all_taxi_states.append(taxi_state)

                # Approximate value (using Q-max as proxy)
                if hasattr(self.model, "q_net"):
                    import torch as th

                    state_tensor = th.as_tensor([int(state)]).float()
                    with th.no_grad():
                        q_values = self.model.q_net(state_tensor).cpu().numpy()[0]
                        value = float(np.max(q_values))
                        all_values.append(value)
                else:
                    all_values.append(0.0)

                state = next_state
                total_reward += reward

            trajectory["success"] = bool(
                total_reward > -50
            )  # Successful if not too negative
            trajectory["total_reward"] = float(total_reward)
            trajectories.append(trajectory)

            if (episode + 1) % 100 == 0:
                print(f"  Collected {episode + 1} episodes")

        self.collected_states = np.array(all_states, dtype=np.int64)
        self.collected_actions = np.array(all_actions, dtype=np.int64)
        self.collected_values = np.array(all_values, dtype=np.float32)
        self.collected_taxi_states = np.array(all_taxi_states, dtype=np.int64)
        self.collected_trajectories = trajectories

        print(f"Total states collected: {len(self.collected_states)}")
        print(
            f"State range: [{np.min(self.collected_states)}, {np.max(self.collected_states)}]"
        )

        return trajectories

    def cluster_by_action(self, n_clusters: int = 10) -> List:
        """
        Cluster states based on their features (taxi position and passenger/destination)
        """
        print(
            f"\nClustering {len(self.collected_states)} states into {n_clusters} clusters..."
        )

        # Create feature matrix: [taxi_row, taxi_col, passenger, destination]
        features = np.array(
            [[t[0], t[1], t[2], t[3]] for t in self.collected_taxi_states],
            dtype=np.float32,
        )

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        self.cluster_labels = kmeans.fit_predict(features_scaled)

        # Create cluster objects
        self.clusters = []
        self.state_to_cluster = {}

        from dataclasses import dataclass

        @dataclass
        class TaxiCluster:
            id: int
            states: List[int]
            indices: List[int]
            size: int
            centroid: np.ndarray
            dominant_action: int
            action_distribution: Dict
            avg_passenger: float
            avg_destination: float
            avg_taxi_row: float
            avg_taxi_col: float

        for cid in range(n_clusters):
            mask = self.cluster_labels == cid
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            cluster_states = self.collected_states[indices]
            cluster_actions = self.collected_actions[indices]
            cluster_features = features[indices]

            # Calculate action distribution
            unique_actions, counts = np.unique(cluster_actions, return_counts=True)
            action_dist = {
                int(a): float(c / len(cluster_actions))
                for a, c in zip(unique_actions, counts)
            }
            dominant_action = int(unique_actions[np.argmax(counts)])

            # Calculate average features
            avg_features = np.mean(cluster_features, axis=0)

            cluster = TaxiCluster(
                id=cid,
                states=cluster_states.tolist(),
                indices=indices.tolist(),
                size=len(indices),
                centroid=avg_features,
                dominant_action=dominant_action,
                action_distribution=action_dist,
                avg_passenger=float(avg_features[2]),
                avg_destination=float(avg_features[3]),
                avg_taxi_row=float(avg_features[0]),
                avg_taxi_col=float(avg_features[1]),
            )

            self.clusters.append(cluster)

            for idx in indices:
                self.state_to_cluster[int(idx)] = cid

        print(f"Created {len(self.clusters)} clusters")

        # Print cluster statistics
        for cluster in self.clusters:
            print(
                f"  Cluster {cluster.id}: size={cluster.size}, "
                f"dominant action={self.action_names[cluster.dominant_action]}"
            )

        return self.clusters

    def visualize_tsne(self, save_path: Optional[str] = None):
        """
        Visualize clusters using t-SNE
        """
        if self.collected_taxi_states is None:
            print("No data collected")
            return

        print("\nComputing t-SNE...")

        # Create features
        features = np.array(
            [[t[0], t[1], t[2], t[3]] for t in self.collected_taxi_states],
            dtype=np.float32,
        )

        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            random_state=self.seed,
            perplexity=min(30, len(features) - 1),
        )
        features_2d = tsne.fit_transform(features)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Colored by cluster
        ax = axes[0, 0]
        n_clusters = len(self.clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

        for i, cluster in enumerate(self.clusters):
            mask = self.cluster_labels == cluster.id
            if np.any(mask):
                ax.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=[colors[i % len(colors)]],
                    s=10,
                    alpha=0.6,
                    label=f"C{cluster.id}" if i < 5 else "",
                )

        ax.set_title(f"t-SNE: States Colored by Cluster ({n_clusters} clusters)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if n_clusters <= 5:
            ax.legend()

        # Plot 2: Colored by action
        ax = axes[0, 1]
        action_colors = ["red", "blue", "green", "purple", "orange", "brown"]

        for action in range(self.n_actions):
            mask = self.collected_actions == action
            if np.any(mask):
                ax.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=action_colors[action],
                    s=10,
                    alpha=0.6,
                    label=self.action_names[action],
                )

        ax.set_title("t-SNE: States Colored by Action")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper right", fontsize=8)

        # Plot 3: Colored by passenger location
        ax = axes[1, 0]
        passenger_colors = ["red", "green", "yellow", "blue", "gray"]
        passenger_names = ["R (0,0)", "G (0,4)", "Y (4,0)", "B (4,3)", "In Taxi"]

        for p in range(5):
            mask = features[:, 2] == p
            if np.any(mask):
                ax.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=passenger_colors[p],
                    s=10,
                    alpha=0.6,
                    label=passenger_names[p],
                )

        ax.set_title("t-SNE: Colored by Passenger Location")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper right", fontsize=8)

        # Plot 4: Colored by destination
        ax = axes[1, 1]
        dest_colors = ["red", "green", "yellow", "blue"]
        dest_names = ["R", "G", "Y", "B"]

        for d in range(4):
            mask = features[:, 3] == d
            if np.any(mask):
                ax.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=dest_colors[d],
                    s=10,
                    alpha=0.6,
                    label=f"Destination {dest_names[d]}",
                )

        ax.set_title("t-SNE: Colored by Destination")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper right", fontsize=8)

        plt.suptitle("Taxi State Space t-SNE Visualization", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"t-SNE visualization saved to {save_path}")

        plt.show()

    def visualize_taxi_grid(self, save_path: Optional[str] = None):
        """
        Visualize clusters on the Taxi grid
        """
        if self.clusters is None:
            print("No clusters to visualize")
            return

        fig = plt.figure(figsize=(18, 12))

        # Create a 2x3 grid of subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Cluster presence map (most common cluster per cell)
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_taxi_grid_background(ax1)

        cluster_grid = np.zeros((5, 5), dtype=int)
        for r in range(5):
            for c in range(5):
                # Find states with taxi at (r,c)
                mask = (self.collected_taxi_states[:, 0] == r) & (
                    self.collected_taxi_states[:, 1] == c
                )
                if np.any(mask):
                    # Get most common cluster for this position
                    clusters_here = self.cluster_labels[mask]
                    unique, counts = np.unique(clusters_here, return_counts=True)
                    cluster_grid[r, c] = unique[np.argmax(counts)]

        # Color by cluster
        n_clusters = len(self.clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

        for r in range(5):
            for c in range(5):
                if cluster_grid[r, c] > 0:
                    rect = Rectangle(
                        (c, 4 - r),
                        1,
                        1,
                        facecolor=colors[cluster_grid[r, c] % len(colors)],
                        alpha=0.6,
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax1.add_patch(rect)
                    ax1.text(
                        c + 0.5,
                        4 - r + 0.5,
                        f"C{cluster_grid[r, c]}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        ax1.set_title("Most Common Cluster by Taxi Position")

        # Plot 2: Dominant action by position
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_taxi_grid_background(ax2)

        action_grid = np.zeros((5, 5), dtype=int)
        for r in range(5):
            for c in range(5):
                mask = (self.collected_taxi_states[:, 0] == r) & (
                    self.collected_taxi_states[:, 1] == c
                )
                if np.any(mask):
                    actions_here = self.collected_actions[mask]
                    unique, counts = np.unique(actions_here, return_counts=True)
                    action_grid[r, c] = unique[np.argmax(counts)]

        action_colors = ["red", "blue", "green", "purple", "orange", "brown"]

        for r in range(5):
            for c in range(5):
                if action_grid[r, c] > 0:
                    rect = Rectangle(
                        (c, 4 - r),
                        1,
                        1,
                        facecolor=action_colors[action_grid[r, c]],
                        alpha=0.6,
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax2.add_patch(rect)
                    symbol = self.action_symbols[action_grid[r, c]]
                    ax2.text(
                        c + 0.5,
                        4 - r + 0.5,
                        symbol,
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

        ax2.set_title("Dominant Action by Taxi Position")

        # Plot 3: Sample trajectory
        ax3 = fig.add_subplot(gs[0, 2])
        self._draw_taxi_grid_background(ax3)

        if self.collected_trajectories:
            # Find a successful trajectory
            successful_trajs = [t for t in self.collected_trajectories if t["success"]]
            if successful_trajs:
                traj = successful_trajs[0]
                taxi_states = traj["taxi_states"]

                # Draw path
                for i, (r, c, p, d) in enumerate(taxi_states):
                    # Draw circle at position
                    circle = Circle(
                        (c + 0.5, 4 - r + 0.5), 0.2, facecolor="blue", alpha=0.5
                    )
                    ax3.add_patch(circle)

                    # Add step number
                    ax3.text(
                        c + 0.5,
                        4 - r + 0.5,
                        str(i),
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )

                    # Draw line to next position
                    if i < len(taxi_states) - 1:
                        nr, nc, _, _ = taxi_states[i + 1]
                        ax3.plot(
                            [c + 0.5, nc + 0.5],
                            [4 - r + 0.5, 4 - nr + 0.5],
                            "gray",
                            linestyle="--",
                            alpha=0.5,
                        )

                ax3.set_title("Sample Successful Trajectory")

        # Plot 4: Passenger distribution by cluster
        ax4 = fig.add_subplot(gs[1, 0])

        passenger_data = []
        cluster_ids = []
        for cluster in self.clusters:
            # Get passenger distribution for this cluster
            mask = self.cluster_labels == cluster.id
            if np.any(mask):
                passengers = self.collected_taxi_states[mask, 2]
                unique, counts = np.unique(passengers, return_counts=True)
                dist = np.zeros(5)
                for p, c in zip(unique, counts):
                    dist[p] = c / len(passengers)
                passenger_data.append(dist)
                cluster_ids.append(cluster.id)

        if passenger_data:
            passenger_data = np.array(passenger_data)
            bottom = np.zeros(len(cluster_ids))
            passenger_colors = ["red", "green", "yellow", "blue", "gray"]
            passenger_labels = ["R", "G", "Y", "B", "In Taxi"]

            for p in range(5):
                values = passenger_data[:, p]
                if np.any(values > 0):
                    ax4.bar(
                        cluster_ids,
                        values,
                        bottom=bottom,
                        color=passenger_colors[p],
                        label=passenger_labels[p],
                        alpha=0.7,
                    )
                    bottom += values

            ax4.set_xlabel("Cluster ID")
            ax4.set_ylabel("Proportion")
            ax4.set_title("Passenger Location by Cluster")
            ax4.legend(fontsize=8)

        # Plot 5: Destination distribution by cluster
        ax5 = fig.add_subplot(gs[1, 1])

        dest_data = []
        for cluster in self.clusters:
            mask = self.cluster_labels == cluster.id
            if np.any(mask):
                dests = self.collected_taxi_states[mask, 3]
                unique, counts = np.unique(dests, return_counts=True)
                dist = np.zeros(4)
                for d, c in zip(unique, counts):
                    dist[d] = c / len(dests)
                dest_data.append(dist)

        if dest_data:
            dest_data = np.array(dest_data)
            bottom = np.zeros(len(cluster_ids))
            dest_colors = ["red", "green", "yellow", "blue"]
            dest_labels = ["R", "G", "Y", "B"]

            for d in range(4):
                values = dest_data[:, d]
                if np.any(values > 0):
                    ax5.bar(
                        cluster_ids,
                        values,
                        bottom=bottom,
                        color=dest_colors[d],
                        label=dest_labels[d],
                        alpha=0.7,
                    )
                    bottom += values

            ax5.set_xlabel("Cluster ID")
            ax5.set_ylabel("Proportion")
            ax5.set_title("Destination by Cluster")
            ax5.legend(fontsize=8)

        # Plot 6: Action distribution pie charts for first 4 clusters
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")

        # Create a 2x2 grid for pie charts within this subplot
        pie_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[1, 2], hspace=0.4, wspace=0.4
        )

        n_clusters_to_show = min(4, len(self.clusters))
        for i, cluster in enumerate(self.clusters[:n_clusters_to_show]):
            pie_ax = fig.add_subplot(pie_gs[i // 2, i % 2])
            actions = list(cluster.action_distribution.keys())
            sizes = list(cluster.action_distribution.values())
            colors_sub = [action_colors[a] for a in actions]

            if sizes and sum(sizes) > 0:
                wedges, texts, autotexts = pie_ax.pie(
                    sizes,
                    colors=colors_sub,
                    autopct="%1.0f%%",
                    startangle=90,
                    textprops={"fontsize": 8},
                )
                pie_ax.set_title(
                    f"Cluster {cluster.id}\n(size={cluster.size})", fontsize=9
                )

        plt.suptitle("Taxi Cluster Analysis", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Grid visualization saved to {save_path}")

        plt.show()

    def visualize_cluster_gallery(self, save_path: Optional[str] = None):
        """
        Create a gallery showing representative states from each cluster
        """
        if not self.clusters:
            print("No clusters to visualize")
            return

        n_clusters_to_show = min(8, len(self.clusters))
        clusters_to_show = sorted(self.clusters, key=lambda c: c.size, reverse=True)[
            :n_clusters_to_show
        ]

        # Create figure with grid
        fig = plt.figure(figsize=(16, 4 * n_clusters_to_show))

        # Create grid spec
        gs = fig.add_gridspec(n_clusters_to_show, 4, hspace=0.3, wspace=0.3)

        for row, cluster in enumerate(clusters_to_show):
            # Column 0: Cluster info
            ax = fig.add_subplot(gs[row, 0])
            ax.axis("off")

            info_text = f"Cluster {cluster.id}\n"
            info_text += f"Size: {cluster.size}\n"
            info_text += f"Dominant: {self.action_names[cluster.dominant_action]}\n"
            info_text += (
                f"Avg taxi: ({cluster.avg_taxi_row:.1f}, {cluster.avg_taxi_col:.1f})\n"
            )
            info_text += f"Avg passenger: {cluster.avg_passenger:.1f}\n"
            info_text += f"Avg dest: {cluster.avg_destination:.1f}\n\n"
            info_text += "Action distribution:\n"

            for a in range(self.n_actions):
                pct = cluster.action_distribution.get(a, 0) * 100
                if pct > 0:
                    info_text += f"  {self.action_names[a]}: {pct:.0f}%\n"

            ax.text(
                0.1,
                0.5,
                info_text,
                transform=ax.transAxes,
                verticalalignment="center",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

            # Columns 1-3: Sample states
            n_samples = min(3, len(cluster.indices))
            if n_samples > 0:
                sample_indices = np.random.choice(
                    cluster.indices, n_samples, replace=False
                )

                for col, idx in enumerate(sample_indices):
                    ax = fig.add_subplot(gs[row, col + 1])
                    state_idx = self.collected_states[idx]
                    taxi_state = self.collected_taxi_states[idx]
                    action = self.collected_actions[idx]

                    self._draw_taxi_state(
                        ax,
                        int(state_idx),
                        taxi_state,
                        title=f"Action: {self.action_symbols[action]}",
                    )

        plt.suptitle("Taxi Cluster Gallery", fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Gallery saved to {save_path}")

        plt.show()

    def _draw_taxi_grid_background(self, ax):
        """Draw the Taxi grid background"""
        # Draw grid
        for i in range(6):
            ax.axhline(y=i, color="black", linewidth=1)
            ax.axvline(x=i, color="black", linewidth=1)

        # Mark special locations
        for loc_idx, (r, c) in self.locations.items():
            # Draw colored square
            rect = Rectangle(
                (c, 4 - r), 1, 1, facecolor=self.location_colors[loc_idx], alpha=0.3
            )
            ax.add_patch(rect)

            # Add letter
            loc_names = ["R", "G", "Y", "B"]
            ax.text(
                c + 0.5,
                4 - r + 0.5,
                loc_names[loc_idx],
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.grid(True, alpha=0.3)

    def _draw_taxi_state(self, ax, state_idx, taxi_state, title=""):
        """Draw a specific Taxi state"""
        ax.clear()
        self._draw_taxi_grid_background(ax)

        taxi_row, taxi_col, passenger, destination = taxi_state

        # Draw destination highlight
        if destination < 4:
            dr, dc = self.locations[destination]
            rect = Rectangle((dc, 4 - dr), 1, 1, facecolor="gold", alpha=0.5)
            ax.add_patch(rect)
            ax.text(
                dc + 0.5,
                4 - dr + 0.5,
                "DEST",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        # Draw passenger
        if passenger < 4:
            pr, pc = self.locations[passenger]
            # Draw passenger circle
            circle = Circle(
                (pc + 0.5, 4 - pr + 0.5), 0.2, facecolor="purple", alpha=0.7
            )
            ax.add_patch(circle)
            ax.text(
                pc + 0.5,
                4 - pr + 0.5,
                "P",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

        # Draw taxi
        taxi_rect = Rectangle(
            (taxi_col + 0.2, 4 - taxi_row + 0.2), 0.6, 0.6, facecolor="black", alpha=0.8
        )
        ax.add_patch(taxi_rect)

        # Draw taxi symbol
        if passenger == 4:  # Passenger inside
            ax.text(
                taxi_col + 0.5,
                4 - taxi_row + 0.5,
                "🚕+",
                ha="center",
                va="center",
                color="yellow",
                fontsize=12,
            )
        else:
            ax.text(
                taxi_col + 0.5,
                4 - taxi_row + 0.5,
                "🚕",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
            )

        ax.set_title(title, fontsize=8)

    def save_analysis(self, base_filename: str = "taxi_analysis"):
        """Save all analysis results"""
        os.makedirs("taxi_analysis", exist_ok=True)

        # Prepare cluster data
        cluster_data = []
        for cluster in self.clusters:
            cluster_data.append(
                {
                    "id": int(cluster.id),
                    "size": int(cluster.size),
                    "dominant_action": int(cluster.dominant_action),
                    "action_distribution": {
                        str(k): float(v) for k, v in cluster.action_distribution.items()
                    },
                    "avg_taxi_row": float(cluster.avg_taxi_row),
                    "avg_taxi_col": float(cluster.avg_taxi_col),
                    "avg_passenger": float(cluster.avg_passenger),
                    "avg_destination": float(cluster.avg_destination),
                }
            )

        results = {
            "n_clusters": len(self.clusters),
            "clusters": cluster_data,
            "total_states": len(self.collected_states)
            if self.collected_states is not None
            else 0,
            "action_names": self.action_names,
        }

        import json

        with open(f"taxi_analysis/{base_filename}.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nAnalysis saved to taxi_analysis/{base_filename}.json")


def run_taxi_experiment(train: bool = True, model_path: str = "models/taxi_dqn"):
    """
    Run complete Taxi experiment
    """
    print("\n" + "=" * 70)
    print("TAXI-V3 CLUSTER ANALYSIS")
    print("=" * 70)

    # Create analyzer
    analyzer = TaxiClusterAnalyzer(seed=42)

    # Train or load model
    if train:
        analyzer.train_dqn(total_timesteps=50000, save_path=model_path)
    else:
        if os.path.exists(model_path):
            analyzer.model = DQN.load(model_path)
        else:
            print(f"Model not found at {model_path}, training new model...")
            analyzer.train_dqn(total_timesteps=50000, save_path=model_path)

    # Collect trajectories
    trajectories = analyzer.collect_trajectories(n_episodes=500)

    # Perform clustering
    clusters = analyzer.cluster_by_action(n_clusters=8)

    # Visualizations
    analyzer.visualize_tsne(save_path="taxi_analysis/tsne_visualization.png")
    analyzer.visualize_taxi_grid(save_path="taxi_analysis/taxi_grid.png")
    analyzer.visualize_cluster_gallery(save_path="taxi_analysis/cluster_gallery.png")

    # Save results
    analyzer.save_analysis()

    return analyzer


if __name__ == "__main__":
    # Run experiment
    analyzer = run_taxi_experiment(train=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nGenerated files in taxi_analysis/ directory:")
    print("  - tsne_visualization.png: t-SNE plot of state space")
    print("  - taxi_grid.png: Cluster visualization on Taxi grid")
    print("  - cluster_gallery.png: Sample states from each cluster")
    print("  - taxi_analysis.json: Cluster statistics")
