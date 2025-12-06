"""
For Lunar Lander (continuous actions) with PPO from Stable-Baselines3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import pickle
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import time
import warnings

warnings.filterwarnings("ignore")

# Gymnasium and SB3
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# Persistent Homology (optional)
try:
    from ripser import Rips
    from persim import plot_diagrams

    PERSISTENT_HOMOLOGY_AVAILABLE = True
except ImportError:
    print("Optional: Install ripser and persim for persistent homology analysis")
    print("pip install ripser persim")
    PERSISTENT_HOMOLOGY_AVAILABLE = False

# ============================================================================
# 1. TRAJECTORY STORAGE
# ============================================================================


class RLTrajectoryStorage:
    """Storage container for complete RL trajectories"""

    def __init__(self, max_trajectories=1000):
        """
        Initialize trajectory storage.

        Args:
            max_trajectories: Maximum number of trajectories to store
        """
        self.max_trajectories = max_trajectories
        self.trajectories = deque(maxlen=max_trajectories)
        self.state_dim = None
        self.action_dim = None
        self.is_continuous = None

    def add_trajectory(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
        infos: Optional[List[Dict]] = None,
    ):
        """
        Add a complete trajectory.

        Args:
            states: List of states [initial_state, s1, s2, ..., sT]
            actions: List of actions [a0, a1, ..., a_{T-1}]
            rewards: List of rewards [r0, r1, ..., r_{T-1}]
            dones: List of done flags
            infos: Optional list of info dictionaries
        """
        # Convert to arrays
        states_array = np.array(states)
        actions_array = np.array(actions)
        rewards_array = np.array(rewards)
        dones_array = np.array(dones)

        trajectory = {
            "states": states_array,
            "actions": actions_array,
            "rewards": rewards_array,
            "dones": dones_array,
            "infos": infos if infos is not None else [],
            "total_reward": np.sum(rewards_array),
            "length": len(actions_array),
        }

        # Update dimensions
        if self.state_dim is None and len(states_array) > 0:
            self.state_dim = (
                states_array[0].shape[0]
                if hasattr(states_array[0], "shape")
                else len(states_array[0])
            )
        if self.action_dim is None and len(actions_array) > 0:
            self.action_dim = (
                actions_array[0].shape[0]
                if hasattr(actions_array[0], "shape")
                else len(actions_array[0])
            )
            # Determine if continuous based on action dimension
            self.is_continuous = (
                self.action_dim > 1 or self.action_dim == 2
            )  # Lunar Lander continuous has 2 actions

        self.trajectories.append(trajectory)

    def get_state_action_pairs(self, max_samples: Optional[int] = None) -> np.ndarray:
        """Extract state-action pairs from all trajectories"""
        state_action_pairs = []

        for traj in self.trajectories:
            states = traj["states"][:-1]  # Exclude final state
            actions = traj["actions"]

            min_len = min(len(states), len(actions))
            for i in range(min_len):
                state = states[i].flatten()
                action = actions[i].flatten()
                state_action = np.concatenate([state, action])
                state_action_pairs.append(state_action)

                if max_samples and len(state_action_pairs) >= max_samples:
                    return np.array(state_action_pairs)

        return np.array(state_action_pairs)

    def get_all_states(self) -> np.ndarray:
        """Get all states from all trajectories"""
        states = []
        for traj in self.trajectories:
            states.extend(traj["states"])
        return np.array(states)

    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """Compute statistics over all trajectories"""
        if not self.trajectories:
            return {}

        total_rewards = [t["total_reward"] for t in self.trajectories]
        lengths = [t["length"] for t in self.trajectories]

        return {
            "num_trajectories": len(self.trajectories),
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "min_reward": np.min(total_rewards),
            "max_reward": np.max(total_rewards),
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
        }

    def save(self, filename: str):
        """Save trajectories to file"""
        data = {
            "trajectories": list(self.trajectories),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "is_continuous": self.is_continuous,
            "max_trajectories": self.max_trajectories,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename: str):
        """Load trajectories from file"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.trajectories = deque(
                data["trajectories"],
                maxlen=data.get("max_trajectories", self.max_trajectories),
            )
            self.state_dim = data["state_dim"]
            self.action_dim = data["action_dim"]
            self.is_continuous = data.get("is_continuous", None)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return list(self.trajectories)[idx]


# ============================================================================
# 2. TRAJECTORY COLLECTION DURING TRAINING
# ============================================================================


class TrajectoryCollectorCallback(BaseCallback):
    """Callback to collect trajectories during PPO training"""

    def __init__(self, storage: RLTrajectoryStorage, verbose=0):
        super(TrajectoryCollectorCallback, self).__init__(verbose)
        self.storage = storage
        self.current_trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
        }
        self.current_state = None
        self.current_info = None

    def _on_training_start(self):
        self.current_trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
        }

    def _on_step(self) -> bool:
        # This is a simplified collection - in practice, you'd need to access
        # the rollout buffer or collect episodes separately
        return True


def collect_trajectories_from_policy(
    model,
    env_name: str,
    num_trajectories: int = 50,
    max_steps: int = 1000,
    continuous: bool = True,
) -> RLTrajectoryStorage:
    """
    Collect trajectories using a trained policy.

    Args:
        model: Trained PPO model
        env_name: Gymnasium environment name
        num_trajectories: Number of trajectories to collect
        max_steps: Maximum steps per trajectory
        continuous: Whether to use continuous actions
    """
    storage = RLTrajectoryStorage(max_trajectories=num_trajectories)
    env = gym.make(env_name, continuous=continuous)

    print(f"Collecting {num_trajectories} trajectories...")

    for episode in range(num_trajectories):
        state, info = env.reset()
        states = [state]
        actions = []
        rewards = []
        dones = []
        infos = [info]

        total_reward = 0
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < max_steps:
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)

            state = next_state
            total_reward += reward
            step += 1

        # Add trajectory to storage
        storage.add_trajectory(states, actions, rewards, dones, infos)

        if (episode + 1) % 10 == 0:
            print(f"  Collected {episode + 1}/{num_trajectories} trajectories")

    env.close()

    stats = storage.get_trajectory_statistics()
    print(f"\nCollection complete:")
    print(f"  Average reward: {stats['avg_reward']:.2f}")
    print(f"  Best reward: {stats['max_reward']:.2f}")
    print(f"  Average length: {stats['avg_length']:.1f}")

    return storage


# ============================================================================
# 3. PPO TRAINING
# ============================================================================


def train_ppo_lunar_lander(
    total_timesteps: int = 100000,
    continuous: bool = True,
    save_path: str = "ppo_lunar_lander",
) -> PPO:
    """
    Train a PPO agent on Lunar Lander.

    Args:
        total_timesteps: Number of training timesteps
        continuous: Whether to use continuous actions
        save_path: Path to save the trained model
    """
    print("=" * 50)
    print(
        f"TRAINING PPO ON LUNAR LANDER ({'CONTINUOUS' if continuous else 'DISCRETE'})"
    )
    print("=" * 50)

    # Create environment
    env = gym.make("LunarLander-v3", continuous=continuous)
    env = DummyVecEnv([lambda: env])

    # Create PPO model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=1000000,  # PPO
        learning_rate=3e-4,
        # n_steps=2048, # PPO
        # batch_size=64, # PPO
        batch_size=256,
        # n_epochs=10, # PPO
        tau=0.01,  # SAC
        gamma=0.99,
        # gae_lambda=0.95, # PPO
        # clip_range=0.2, # PPO
        # ent_coef=0.01, # PPO
        ent_coef="auto",  # SAC
        gradient_steps=1,  # SAC
        learning_starts=1000,  # SAC
        # policy_kwargs=dict(net_arch=[64, 64]), # PPO
        policy_kwargs=dict(net_arch=[400, 300]),  # SAC
    )

    # Train the model
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the trained policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"\nTraining complete!")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    env.close()
    return model


# ============================================================================
# 4. TRAJECTORY PCA AND MANIFOLD VISUALIZATION
# ============================================================================


class TrajectoryPCA:
    """PCA compression for entire trajectories"""

    def __init__(self, state_components=2, action_components=1):
        self.state_components = min(state_components, 8)  # Max 8 for Lunar Lander
        self.action_components = action_components
        self.state_pca = None
        self.action_pca = None
        self.state_dim = None
        self.action_dim = None

    def fit(self, trajectory_storage: RLTrajectoryStorage):
        """Fit PCA models to trajectories"""
        print("\n" + "=" * 50)
        print("FITTING PCA TO TRAJECTORIES")
        print("=" * 50)

        trajectories = list(trajectory_storage.trajectories)
        if not trajectories:
            raise ValueError("No trajectories available")

        # Extract all states and actions
        all_states = []
        all_actions = []

        for traj in trajectories:
            states = traj["states"]
            actions = traj["actions"]

            # Ensure same length
            min_len = min(len(states), len(actions))
            all_states.extend(states[:min_len])

            # Reshape actions if needed
            if len(actions.shape) == 1:
                actions = actions.reshape(-1, 1)
            all_actions.extend(actions[:min_len])

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)

        self.state_dim = all_states.shape[1]
        self.action_dim = all_actions.shape[1]

        print(f"Original state dimension: {self.state_dim}")
        print(f"Original action dimension: {self.action_dim}")
        print(f"Total samples: {len(all_states)}")

        # Fit state PCA
        print(f"\nFitting state PCA ({self.state_components} components)...")
        start_time = time.time()

        if len(all_states) > 10000:
            self.state_pca = IncrementalPCA(n_components=self.state_components)
            batch_size = 1000
            for i in range(0, len(all_states), batch_size):
                batch = all_states[i : i + batch_size]
                self.state_pca.partial_fit(batch)
        else:
            self.state_pca = PCA(n_components=self.state_components)
            self.state_pca.fit(all_states)

        state_var_ratio = np.sum(self.state_pca.explained_variance_ratio_)
        print(f"State PCA variance explained: {state_var_ratio:.3f}")
        print(f"Components: {self.state_pca.explained_variance_ratio_}")

        # Fit action PCA if needed
        if self.action_dim > 1 and self.action_components < self.action_dim:
            print(f"\nFitting action PCA ({self.action_components} components)...")
            self.action_pca = PCA(n_components=self.action_components)
            self.action_pca.fit(all_actions)
            action_var_ratio = np.sum(self.action_pca.explained_variance_ratio_)
            print(f"Action PCA variance explained: {action_var_ratio:.3f}")
        else:
            self.action_pca = None
            print("\nAction dimension is 1 or less, no PCA needed")

        end_time = time.time()
        print(f"\nPCA fitting completed in {end_time - start_time:.2f} seconds")

        return self

    def compress_trajectory(
        self, states: np.ndarray, actions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compress a single trajectory"""
        # Compress states
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        compressed_states = self.state_pca.transform(states)

        # Compress actions
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)

        if self.action_pca is not None:
            compressed_actions = self.action_pca.transform(actions)
        else:
            compressed_actions = actions

        # Ensure same length
        min_len = min(len(compressed_states), len(compressed_actions))

        return {
            "compressed_states": compressed_states[:min_len],
            "compressed_actions": compressed_actions[:min_len],
            "original_states": states[:min_len],
            "original_actions": actions[:min_len],
        }

    def compress_all_trajectories(
        self, trajectory_storage: RLTrajectoryStorage
    ) -> List[Dict[str, Any]]:
        """Compress all trajectories"""
        compressed_trajectories = []

        for traj in trajectory_storage.trajectories:
            compressed = self.compress_trajectory(traj["states"], traj["actions"])
            compressed["total_reward"] = traj["total_reward"]
            compressed["length"] = traj["length"]
            compressed["original_trajectory"] = traj
            compressed_trajectories.append(compressed)

        return compressed_trajectories


class TrajectoryManifoldVisualizer:
    """Visualize trajectories as manifolds in compressed space"""

    def __init__(self):
        self.fig = None
        self.ax = None

    def plot_2d_manifold(
        self,
        compressed_trajectories: List[Dict[str, Any]],
        color_by: str = "reward",
        alpha: float = 0.7,
        linewidth: float = 1.5,
        show_start_end: bool = True,
    ):
        """
        Plot 2D manifold with compressed state on one axis and action on the other
        """
        self.fig, self.ax = plt.subplots(figsize=(14, 10))

        # Prepare colors
        if color_by == "reward":
            rewards = [t["total_reward"] for t in compressed_trajectories]
            norm_rewards = (rewards - np.min(rewards)) / (
                np.max(rewards) - np.min(rewards) + 1e-10
            )
            colors = plt.cm.viridis(norm_rewards)
            color_label = "Total Reward"
        elif color_by == "length":
            lengths = [t["length"] for t in compressed_trajectories]
            norm_lengths = (lengths - np.min(lengths)) / (
                np.max(lengths) - np.min(lengths) + 1e-10
            )
            colors = plt.cm.plasma(norm_lengths)
            color_label = "Trajectory Length"
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, len(compressed_trajectories)))
            color_label = "Trajectory Index"

        # Plot each trajectory
        for i, traj in enumerate(compressed_trajectories):
            states = traj["compressed_states"]
            actions = traj["compressed_actions"]

            # Extract coordinates
            if states.shape[1] == 1:
                # 1D state compression
                x = states[:, 0]
                y = actions[:, 0] if actions.shape[1] == 1 else np.zeros(len(states))
            else:
                # 2D state compression - use first component vs action
                x = states[:, 0]
                y = actions[:, 0] if actions.shape[1] >= 1 else np.zeros(len(states))

            # Plot trajectory
            self.ax.plot(
                x,
                y,
                alpha=alpha,
                linewidth=linewidth,
                color=colors[i] if color_by != "index" else colors[i],
            )

            # Add start and end markers
            if show_start_end:
                if i == 0:  # Only label once
                    self.ax.scatter(
                        x[0],
                        y[0],
                        color="green",
                        s=80,
                        zorder=5,
                        marker="o",
                        label="Start",
                    )
                    self.ax.scatter(
                        x[-1],
                        y[-1],
                        color="red",
                        s=80,
                        zorder=5,
                        marker="s",
                        label="End",
                    )
                else:
                    self.ax.scatter(
                        x[0], y[0], color="green", s=30, zorder=5, marker="o"
                    )
                    self.ax.scatter(
                        x[-1], y[-1], color="red", s=30, zorder=5, marker="s"
                    )

        self.ax.set_xlabel("Compressed State (PC1)", fontsize=12)
        self.ax.set_ylabel("Compressed Action", fontsize=12)
        self.ax.set_title(
            "2D Trajectory Manifold: State vs Action", fontsize=14, fontweight="bold"
        )
        self.ax.legend(fontsize=10)
        self.ax.grid(True, alpha=0.3)

        # Add colorbar
        if color_by in ["reward", "length"]:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.viridis if color_by == "reward" else plt.cm.plasma
            )
            sm.set_array(rewards if color_by == "reward" else lengths)
            cbar = plt.colorbar(sm, ax=self.ax)
            cbar.set_label(color_label, fontsize=12)

        plt.tight_layout()
        return self.fig, self.ax

    def plot_3d_manifold(
        self,
        compressed_trajectories: List[Dict[str, Any]],
        color_by: str = "reward",
        alpha: float = 0.6,
        linewidth: float = 1.0,
        elev: float = 20,
        azim: float = 45,
    ):
        """
        Plot 3D manifold with 2D compressed state and 1D compressed action
        """
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Check if we have 2D state compression
        if compressed_trajectories[0]["compressed_states"].shape[1] < 2:
            print("Warning: Need 2D state compression for 3D plot")
            return self.fig, self.ax

        # Prepare colors
        if color_by == "reward":
            rewards = [t["total_reward"] for t in compressed_trajectories]
            norm_rewards = (rewards - np.min(rewards)) / (
                np.max(rewards) - np.min(rewards) + 1e-10
            )
            colors = plt.cm.viridis(norm_rewards)
            color_label = "Total Reward"
        elif color_by == "velocity":
            velocities = []
            for traj in compressed_trajectories:
                states = traj["original_states"]
                if len(states) > 1 and states.shape[1] >= 4:
                    # Lunar Lander: indices 2,3 are velocities
                    vel_mag = np.mean(np.sqrt(states[:, 2] ** 2 + states[:, 3] ** 2))
                else:
                    vel_mag = 0
                velocities.append(vel_mag)
            norm_velocities = (velocities - np.min(velocities)) / (
                np.max(velocities) - np.min(velocities) + 1e-10
            )
            colors = plt.cm.plasma(norm_velocities)
            color_label = "Average Velocity"
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, len(compressed_trajectories)))
            color_label = "Trajectory Index"

        # Plot each trajectory
        for i, traj in enumerate(compressed_trajectories[:50]):  # Limit for clarity
            states = traj["compressed_states"]
            actions = traj["compressed_actions"]

            if states.shape[1] < 2:
                continue

            # 3D coordinates
            x = states[:, 0]  # State PC1
            y = states[:, 1]  # State PC2
            z = actions[:, 0] if actions.shape[1] >= 1 else np.zeros(len(states))

            # Plot 3D trajectory
            self.ax.plot(
                x,
                y,
                z,
                alpha=alpha,
                linewidth=linewidth,
                color=colors[i] if color_by != "index" else colors[i],
            )

            # Add markers for first and last few trajectories
            if i < 5:
                self.ax.scatter(
                    x[0],
                    y[0],
                    z[0],
                    color="green",
                    s=100,
                    marker="o",
                    depthshade=False,
                    label="Start" if i == 0 else "",
                )
                self.ax.scatter(
                    x[-1],
                    y[-1],
                    z[-1],
                    color="red",
                    s=100,
                    marker="s",
                    depthshade=False,
                    label="End" if i == 0 else "",
                )

            # Add direction arrow
            if len(x) > 10:
                mid_idx = len(x) // 2
                dx = x[mid_idx + 1] - x[mid_idx]
                dy = y[mid_idx + 1] - y[mid_idx]
                dz = z[mid_idx + 1] - z[mid_idx]
                norm = np.sqrt(dx**2 + dy**2 + dz**2)
                if norm > 0.01:
                    self.ax.quiver(
                        x[mid_idx],
                        y[mid_idx],
                        z[mid_idx],
                        dx / norm,
                        dy / norm,
                        dz / norm,
                        length=0.1,
                        color="black",
                        alpha=0.5,
                        arrow_length_ratio=0.3,
                    )

        self.ax.set_xlabel("State PC1", fontsize=12, labelpad=10)
        self.ax.set_ylabel("State PC2", fontsize=12, labelpad=10)
        self.ax.set_zlabel("Action", fontsize=12, labelpad=10)
        self.ax.set_title("3D Trajectory Manifold", fontsize=14, fontweight="bold")
        self.ax.legend(fontsize=10)

        # Add colorbar
        if color_by in ["reward", "velocity"]:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.viridis if color_by == "reward" else plt.cm.plasma
            )
            sm.set_array(rewards if color_by == "reward" else velocities)
            cbar = plt.colorbar(sm, ax=self.ax, pad=0.1)
            cbar.set_label(color_label, fontsize=12)

        # Set view
        self.ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()

        return self.fig, self.ax


# ============================================================================
# 5. PERSISTENT HOMOLOGY ANALYSIS (OPTIONAL)
# ============================================================================

if PERSISTENT_HOMOLOGY_AVAILABLE:

    class PersistentHomologyAnalyzer:
        """Analyze topological properties of trajectories"""

        def __init__(self, max_dim=1, max_edge_length=1.0):
            self.max_dim = max_dim
            self.max_edge_length = max_edge_length
            self.rips = Rips(maxdim=max_dim, thresh=max_edge_length)
            self.diagrams = None

        def analyze(self, points: np.ndarray, subsample: Optional[int] = None):
            """Compute persistent homology for point cloud"""
            if subsample and len(points) > subsample:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(points), subsample, replace=False)
                points = points[indices]

            print(f"Computing persistent homology for {len(points)} points...")
            start_time = time.time()

            self.diagrams = self.rips.fit_transform(points)

            end_time = time.time()
            print(f"Completed in {end_time - start_time:.2f} seconds")

            return self.diagrams

        def plot_diagrams(self, title="Persistence Diagrams"):
            """Plot persistence diagrams"""
            if self.diagrams is None:
                print("No diagrams to plot")
                return

            plot_diagrams(self.diagrams, show=False)
            plt.title(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()

# ============================================================================
# 6. MAIN ANALYSIS PIPELINE
# ============================================================================


class TrajectoryManifoldAnalysis:
    """Complete analysis pipeline for RL trajectory manifolds"""

    def __init__(self, state_components=2, action_components=1):
        self.state_components = state_components
        self.action_components = action_components
        self.pca = TrajectoryPCA(state_components, action_components)
        self.visualizer = TrajectoryManifoldVisualizer()
        self.storage = None
        self.compressed_trajectories = None

        if PERSISTENT_HOMOLOGY_AVAILABLE:
            self.ph_analyzer = PersistentHomologyAnalyzer(max_dim=1)
        else:
            self.ph_analyzer = None

    def train_and_collect(
        self,
        total_timesteps=50000,
        num_trajectories=100,
        continuous=True,
        model_path="ppo_lunar_lander",
    ):
        """Train PPO and collect trajectories"""
        print("=" * 60)
        print("STEP 1: TRAINING PPO AND COLLECTING TRAJECTORIES")
        print("=" * 60)

        # Train or load model
        try:
            model = PPO.load(model_path)
            print(f"Loaded pre-trained model from {model_path}.zip")
        except:
            print("Training new model...")
            model = train_ppo_lunar_lander(
                total_timesteps=total_timesteps,
                continuous=continuous,
                save_path=model_path,
            )

        # Collect trajectories
        self.storage = collect_trajectories_from_policy(
            model=model,
            env_name="LunarLander-v3",
            num_trajectories=num_trajectories,
            continuous=continuous,
        )

        # Save trajectories
        self.storage.save("lunar_lander_trajectories.pkl")
        print(f"Trajectories saved to lunar_lander_trajectories.pkl")

        return model, self.storage

    def analyze_manifold(self, plot_2d=True, plot_3d=True):
        """Analyze trajectory manifold"""
        if self.storage is None:
            print("No trajectories available. Run train_and_collect() first.")
            return

        print("\n" + "=" * 60)
        print("STEP 2: TRAJECTORY MANIFOLD ANALYSIS")
        print("=" * 60)

        # Fit PCA
        self.pca.fit(self.storage)

        # Compress trajectories
        self.compressed_trajectories = self.pca.compress_all_trajectories(self.storage)

        print(f"\nCompressed {len(self.compressed_trajectories)} trajectories")
        print(
            f"State dimension reduced from {self.pca.state_dim} to {self.state_components}"
        )
        print(
            f"Action dimension reduced from {self.pca.action_dim} to {self.action_components}"
        )

        # Plot manifolds
        if plot_2d:
            print("\nGenerating 2D manifold plot...")
            self.visualizer.plot_2d_manifold(
                self.compressed_trajectories, color_by="reward"
            )
            plt.savefig("trajectory_manifold_2d.png", dpi=150, bbox_inches="tight")
            plt.show()

        if plot_3d and self.state_components >= 2:
            print("\nGenerating 3D manifold plot...")
            self.visualizer.plot_3d_manifold(
                self.compressed_trajectories, color_by="reward"
            )
            plt.savefig("trajectory_manifold_3d.png", dpi=150, bbox_inches="tight")
            plt.show()

        # Analyze manifold properties
        self._analyze_manifold_properties()

        return self.compressed_trajectories

    def _analyze_manifold_properties(self):
        """Analyze properties of the trajectory manifold"""
        if self.compressed_trajectories is None:
            return

        print("\n" + "=" * 60)
        print("MANIFOLD PROPERTIES ANALYSIS")
        print("=" * 60)

        # Collect all compressed points
        all_states = np.vstack(
            [t["compressed_states"] for t in self.compressed_trajectories]
        )
        all_actions = np.vstack(
            [t["compressed_actions"] for t in self.compressed_trajectories]
        )

        print(f"\nManifold Statistics:")
        print(f"  Total points: {len(all_states):,}")
        print(f"  State range: {np.ptp(all_states, axis=0)}")
        print(f"  Action range: {np.ptp(all_actions, axis=0)}")

        # Trajectory statistics
        rewards = [t["total_reward"] for t in self.compressed_trajectories]
        lengths = [t["length"] for t in self.compressed_trajectories]

        print(f"\nTrajectory Statistics:")
        print(f"  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Reward range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
        print(f"  Average length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} steps")

        # Correlation between reward and manifold coverage
        coverages = []
        for traj in self.compressed_trajectories:
            states = traj["compressed_states"]
            if states.shape[1] >= 2:
                coverage = np.ptp(states[:, 0]) * np.ptp(states[:, 1])
            else:
                coverage = np.ptp(states[:, 0])
            coverages.append(coverage)

        correlation = np.corrcoef(rewards, coverages)[0, 1]
        print(f"\nCorrelation Analysis:")
        print(f"  Reward vs Manifold Coverage: r = {correlation:.3f}")

        if correlation > 0.3:
            print("  → Higher rewards correlate with broader manifold exploration")
        elif correlation < -0.3:
            print("  → Higher rewards correlate with focused manifold coverage")
        else:
            print("  → No strong correlation between reward and manifold coverage")

    def analyze_topology(self, use_state_action=True, max_points=5000):
        """Analyze topological properties (if available)"""
        if not PERSISTENT_HOMOLOGY_AVAILABLE or self.storage is None:
            print("Persistent homology not available or no trajectories")
            return

        print("\n" + "=" * 60)
        print("STEP 3: TOPOLOGICAL ANALYSIS (PERSISTENT HOMOLOGY)")
        print("=" * 60)

        # Extract points for analysis
        if use_state_action:
            points = self.storage.get_state_action_pairs(max_samples=max_points)
            data_type = "state-action pairs"
        else:
            points = self.storage.get_all_states()
            if len(points) > max_points:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(points), max_points, replace=False)
                points = points[indices]
            data_type = "states only"

        print(f"Analyzing {len(points)} {data_type}...")

        # Compute persistent homology
        diagrams = self.ph_analyzer.analyze(points, subsample=min(3000, len(points)))

        # Plot diagrams
        self.ph_analyzer.plot_diagrams(f"Persistence Diagrams - {data_type}")

        # Analyze features
        self._analyze_topological_features(diagrams)

    def _analyze_topological_features(self, diagrams):
        """Analyze topological features from persistence diagrams"""
        print("\nTopological Features:")

        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                continue

            # Filter infinite points
            finite_points = diagram[diagram[:, 1] != np.inf]
            if len(finite_points) == 0:
                continue

            lifespans = finite_points[:, 1] - finite_points[:, 0]

            print(f"\n  Dimension {dim}:")
            print(f"    Features: {len(finite_points)}")
            print(f"    Avg lifespan: {np.mean(lifespans):.4f}")
            print(f"    Max lifespan: {np.max(lifespans):.4f}")

            if dim == 0:
                print("    Interpretation: Connected components = behavioral modes")
            elif dim == 1:
                print("    Interpretation: Loops = cyclic behaviors")

    def run_complete_analysis(self, total_timesteps=50000, num_trajectories=100):
        """Run complete analysis pipeline"""
        print("=" * 70)
        print("COMPREHENSIVE RL TRAJECTORY MANIFOLD ANALYSIS")
        print("=" * 70)

        # Step 1: Train and collect
        model, storage = self.train_and_collect(
            total_timesteps=total_timesteps, num_trajectories=num_trajectories
        )

        # Step 2: Manifold analysis
        compressed = self.analyze_manifold(plot_2d=True, plot_3d=True)

        # Step 3: Topological analysis (optional)
        if PERSISTENT_HOMOLOGY_AVAILABLE:
            self.analyze_topology(use_state_action=True, max_points=5000)

        # Save results
        self._save_results(model, compressed)

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - ppo_lunar_lander.zip (trained model)")
        print("  - lunar_lander_trajectories.pkl (trajectories)")
        print("  - trajectory_manifold_2d.png (2D manifold plot)")
        print("  - trajectory_manifold_3d.png (3D manifold plot)")
        print("  - analysis_results.pkl (complete analysis results)")

        return model, storage, compressed

    def _save_results(self, model, compressed_trajectories):
        """Save analysis results"""
        results = {
            "storage": self.storage,
            "compressed_trajectories": compressed_trajectories,
            "pca_state": self.pca.state_pca,
            "pca_action": self.pca.action_pca,
            "state_components": self.state_components,
            "action_components": self.action_components,
            "storage_stats": self.storage.get_trajectory_statistics()
            if self.storage
            else None,
        }

        with open("analysis_results.pkl", "wb") as f:
            pickle.dump(results, f)


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function"""
    # Create analyzer
    analyzer = TrajectoryManifoldAnalysis(
        state_components=2,  # Compress state to 2D
        action_components=1,  # Keep action as 1D
    )

    # Run complete analysis
    analyzer.run_complete_analysis(
        total_timesteps=50000,  # Training steps
        num_trajectories=100,  # Trajectories to collect
    )


if __name__ == "__main__":
    # Install required packages:
    # pip install gymnasium stable-baselines3 scikit-learn matplotlib scipy
    # Optional: pip install ripser persim

    main()
