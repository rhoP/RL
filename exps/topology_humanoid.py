"""
Topological RL Analysis with Gymnasium HumanoidStandup-v4
Fixed SciPy compatibility issues
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import KDTree, distance_matrix
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import gymnasium as gym
from typing import List, Tuple, Dict, Optional
import pickle
import time
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# ============================================
# 1. STATE REDUCTION AND PREPROCESSING
# ============================================


class StateReducer:
    """Reduce high-dimensional humanoid states for topological analysis"""

    def __init__(self, method="pca", n_components=3):
        self.method = method
        self.n_components = n_components
        self.reducer = None
        self.is_fitted = False

    def fit(self, states: List[np.ndarray]):
        """Fit the dimensionality reducer"""
        if len(states) < 10:
            return

        states_array = np.vstack(states)

        if self.method == "pca":
            self.reducer = PCA(n_components=self.n_components)
            self.reducer.fit(states_array)
        elif self.method == "tsne":
            # Note: t-SNE is slow, use for visualization only
            self.reducer = TSNE(
                n_components=self.n_components,
                perplexity=min(30, len(states_array) - 1),
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.reducer.fit(states_array)
        self.is_fitted = True

    def transform(self, state: np.ndarray) -> np.ndarray:
        """Transform a state to reduced dimensions"""
        if not self.is_fitted or self.reducer is None:
            # Return first few dimensions as fallback
            return state[: self.n_components].copy()

        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        return self.reducer.transform(state)[0]

    def fit_transform(self, states: List[np.ndarray]) -> List[np.ndarray]:
        """Fit and transform"""
        self.fit(states)
        return [self.transform(s) for s in states]


# ============================================
# 2. ADAPTED LOOP DETECTOR WITH FIXED KD-TREE
# ============================================


@dataclass
class Loop:
    """A detected loop in trajectory data"""

    states: np.ndarray  # Sequence of states (reduced dimensions)
    original_states: np.ndarray  # Original high-dimensional states
    length: int  # Number of states
    stitched_from: int  # How many trajectories were stitched
    reward_sum: float  # Total reward accumulated in loop


class HighDimLoopDetector:
    """Detect loops in high-dimensional state spaces - FIXED KD-TREE"""

    def __init__(self, state_reducer, epsilon=0.3, min_loop_length=10):
        self.state_reducer = state_reducer
        self.epsilon = epsilon  # Reduced for high-dim spaces
        self.min_loop_length = min_loop_length
        self.trajectories = []  # List of (states_reduced, states_original, rewards)
        self.loops_history = []

    def add_trajectory(self, states: List[np.ndarray], rewards: List[float]):
        """Add a new trajectory"""
        # Reduce dimensionality
        states_reduced = [self.state_reducer.transform(s) for s in states]

        self.trajectories.append(
            {
                "states_reduced": np.array(states_reduced),
                "states_original": np.array(states),
                "rewards": np.array(rewards),
                "total_reward": sum(rewards),
            }
        )

        # Keep only recent trajectories
        if len(self.trajectories) > 30:
            self.trajectories = self.trajectories[-20:]

    def _find_neighbors_within_radius(self, kd_tree, points, radius):
        """Find all points within radius using KDTree - compatibility wrapper"""
        neighbors = []

        # Method 1: Use query_ball_point if available
        if hasattr(kd_tree, "query_ball_point"):
            for i, point in enumerate(points):
                indices = kd_tree.query_ball_point(point, radius)
                neighbors.append(indices)
        else:
            # Method 2: Manual distance calculation for older SciPy
            n_points = len(points)
            for i in range(n_points):
                distances = kd_tree.query(points[i], k=n_points)[0]
                indices = np.where(distances <= radius)[0]
                neighbors.append(indices.tolist())

        return neighbors

    def detect_loops_projection(self):
        """Detect loops in the reduced state space"""
        loops = []

        if len(self.trajectories) < 2:
            return loops

        # Get recent trajectories
        recent_trajs = (
            self.trajectories[-10:]
            if len(self.trajectories) > 10
            else self.trajectories
        )

        # Method 1: Temporal proximity in reduced space
        all_states_reduced = []
        trajectory_info = []  # (traj_idx, pos_in_traj, original_state, reward)

        for traj_idx, traj in enumerate(recent_trajs):
            states_red = traj["states_reduced"]
            states_orig = traj["states_original"]
            rewards = traj["rewards"]

            for pos in range(len(states_red)):
                all_states_reduced.append(states_red[pos])
                trajectory_info.append(
                    (
                        traj_idx,
                        pos,
                        states_orig[pos],
                        rewards[pos] if pos < len(rewards) else 0,
                    )
                )

        if len(all_states_reduced) < self.min_loop_length:
            return loops

        all_states_reduced = np.array(all_states_reduced)

        # Build KD-tree for efficient nearest neighbor search
        kd_tree = KDTree(all_states_reduced)

        # Find epsilon-close pairs that could indicate loop closures
        # Using query_ball_point for compatibility
        indices = self._find_neighbors_within_radius(
            kd_tree, all_states_reduced, self.epsilon
        )

        visited_pairs = set()
        potential_closures = []

        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i < j and (i, j) not in visited_pairs:
                    traj_i, pos_i, state_i, reward_i = trajectory_info[i]
                    traj_j, pos_j, state_j, reward_j = trajectory_info[j]

                    # Skip if too close in same trajectory (likely consecutive states)
                    if traj_i == traj_j and abs(pos_i - pos_j) < self.min_loop_length:
                        continue

                    # Check if this could be a loop closure
                    dist = np.linalg.norm(all_states_reduced[i] - all_states_reduced[j])
                    potential_closures.append(
                        (i, j, dist, traj_i, pos_i, traj_j, pos_j)
                    )
                    visited_pairs.add((i, j))

        # Sort by distance (closest first)
        potential_closures.sort(key=lambda x: x[2])

        # Try to construct loops from closures
        for closure in potential_closures[:20]:  # Limit number checked
            i, j, dist, traj_i, pos_i, traj_j, pos_j = closure

            # Ensure we have a meaningful sequence
            if traj_i == traj_j:
                # Loop within same trajectory
                start_pos = min(pos_i, pos_j)
                end_pos = max(pos_i, pos_j)

                if end_pos - start_pos >= self.min_loop_length:
                    traj = recent_trajs[traj_i]
                    loop_states_red = traj["states_reduced"][start_pos : end_pos + 1]
                    loop_states_orig = traj["states_original"][start_pos : end_pos + 1]
                    loop_rewards = traj["rewards"][start_pos:end_pos]

                    # Ensure loop is closed (add start to end)
                    loop_states_red = np.vstack([loop_states_red, loop_states_red[0:1]])
                    loop_states_orig = np.vstack(
                        [loop_states_orig, loop_states_orig[0:1]]
                    )

                    loop = Loop(
                        states=loop_states_red,
                        original_states=loop_states_orig,
                        length=len(loop_states_red),
                        stitched_from=1,
                        reward_sum=np.sum(loop_rewards),
                    )
                    loops.append(loop)
            else:
                # Loop stitching two trajectories
                traj1 = recent_trajs[traj_i]
                traj2 = recent_trajs[traj_j]

                # Determine which comes first (by position or timestamp)
                # Simple heuristic: assume pos_i is end of traj1, pos_j is start of traj2
                if (
                    pos_i > len(traj1["states_reduced"]) // 2
                    and pos_j < len(traj2["states_reduced"]) // 2
                ):
                    # traj1 then traj2
                    states1_red = traj1["states_reduced"][: pos_i + 1]
                    states2_red = traj2["states_reduced"][pos_j:]

                    states1_orig = traj1["states_original"][: pos_i + 1]
                    states2_orig = traj2["states_original"][pos_j:]

                    loop_states_red = np.vstack([states1_red, states2_red])
                    loop_states_orig = np.vstack([states1_orig, states2_orig])

                    if len(loop_states_red) >= self.min_loop_length:
                        loop = Loop(
                            states=loop_states_red,
                            original_states=loop_states_orig,
                            length=len(loop_states_red),
                            stitched_from=2,
                            reward_sum=traj1["total_reward"] + traj2["total_reward"],
                        )
                        loops.append(loop)

        self.loops_history.append(loops)
        return loops

    def detect_loops_simple(self):
        """Simplified loop detection without KDTree dependency"""
        loops = []

        if len(self.trajectories) < 2:
            return loops

        # Get recent trajectories
        recent_trajs = (
            self.trajectories[-5:] if len(self.trajectories) > 5 else self.trajectories
        )

        # Method: Simple endpoint matching
        for i in range(len(recent_trajs)):
            traj_i = recent_trajs[i]
            states_i = traj_i["states_reduced"]

            if len(states_i) < self.min_loop_length:
                continue

            # Check for self-loops (start and end close)
            start_i = states_i[0]
            end_i = states_i[-1]

            if np.linalg.norm(start_i - end_i) < self.epsilon * 2:
                # Self-contained loop
                loop_states_red = np.vstack([states_i, states_i[0:1]])  # Close the loop
                loop_states_orig = np.vstack(
                    [traj_i["states_original"], traj_i["states_original"][0:1]]
                )

                loop = Loop(
                    states=loop_states_red,
                    original_states=loop_states_orig,
                    length=len(loop_states_red),
                    stitched_from=1,
                    reward_sum=traj_i["total_reward"],
                )
                loops.append(loop)

            # Check for loops with other trajectories
            for j in range(i + 1, len(recent_trajs)):
                traj_j = recent_trajs[j]
                states_j = traj_j["states_reduced"]

                if len(states_j) < self.min_loop_length:
                    continue

                # Check if end of i is close to start of j
                end_i = states_i[-1]
                start_j = states_j[0]

                if np.linalg.norm(end_i - start_j) < self.epsilon:
                    # Potential loop: i -> j
                    loop_states_red = np.vstack([states_i, states_j])
                    loop_states_orig = np.vstack(
                        [traj_i["states_original"], traj_j["states_original"]]
                    )

                    # Close the loop if end of j is close to start of i
                    end_j = states_j[-1]
                    start_i = states_i[0]

                    if np.linalg.norm(end_j - start_i) < self.epsilon:
                        loop_states_red = np.vstack(
                            [loop_states_red, loop_states_red[0:1]]
                        )
                        loop_states_orig = np.vstack(
                            [loop_states_orig, loop_states_orig[0:1]]
                        )

                    if len(loop_states_red) >= self.min_loop_length:
                        loop = Loop(
                            states=loop_states_red,
                            original_states=loop_states_orig,
                            length=len(loop_states_red),
                            stitched_from=2,
                            reward_sum=traj_i["total_reward"] + traj_j["total_reward"],
                        )
                        loops.append(loop)

        self.loops_history.append(loops)
        return loops

    def get_loop_statistics(self):
        """Get statistics about detected loops"""
        if not self.loops_history:
            return {"n_loops": 0, "avg_length": 0, "max_length": 0, "avg_reward": 0}

        recent_loops = self.loops_history[-1]

        if not recent_loops:
            return {"n_loops": 0, "avg_length": 0, "max_length": 0, "avg_reward": 0}

        stats = {
            "n_loops": len(recent_loops),
            "avg_length": np.mean([l.length for l in recent_loops]),
            "max_length": max([l.length for l in recent_loops]),
            "avg_reward": np.mean([l.reward_sum for l in recent_loops]),
            "total_stitched": sum([l.stitched_from for l in recent_loops]),
        }

        return stats


# ============================================
# 3. SIMPLIFIED MORSE THEORY ANALYZER
# ============================================


class HighDimMorseAnalyzer:
    """Simplified Morse theory analysis for high-dimensional state spaces"""

    def __init__(self, state_reducer, n_clusters=10):
        self.state_reducer = state_reducer
        self.n_clusters = n_clusters
        self.visited_states = []  # Reduced states
        self.visited_states_original = []  # Original states
        self.reward_history = []

        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_counts = None
        self.cluster_rewards = None

    def update(self, states: List[np.ndarray], rewards: List[float]):
        """Update with new visited states"""
        # Reduce dimensionality
        states_reduced = [self.state_reducer.transform(s) for s in states]

        self.visited_states.extend(states_reduced)
        self.visited_states_original.extend(states)
        self.reward_history.extend(rewards)

        # Keep only recent data
        if len(self.visited_states) > 2000:
            keep_idx = -1000
            self.visited_states = self.visited_states[keep_idx:]
            self.visited_states_original = self.visited_states_original[keep_idx:]
            self.reward_history = self.reward_history[keep_idx:]

        # Update clustering periodically
        if len(self.visited_states) > self.n_clusters * 3:
            self._update_clustering()

    def _update_clustering(self):
        """Update K-means clustering on reduced states"""
        states_array = np.array(self.visited_states)

        # Fit clustering
        self.clusterer.fit(states_array)

        # Count visits per cluster
        labels = self.clusterer.predict(states_array)
        self.cluster_counts = np.zeros(self.n_clusters)
        for label in labels:
            self.cluster_counts[label] += 1

        # Compute average reward per cluster
        self.cluster_rewards = np.zeros(self.n_clusters)
        if len(self.reward_history) >= len(labels):
            recent_rewards = self.reward_history[-len(labels) :]
            for label, reward in zip(labels, recent_rewards):
                self.cluster_rewards[label] += reward

            # Normalize by count
            for i in range(self.n_clusters):
                if self.cluster_counts[i] > 0:
                    self.cluster_rewards[i] /= self.cluster_counts[i]

    def find_critical_clusters(self):
        """Find critical clusters using visitation density and rewards"""
        if self.cluster_counts is None or len(self.visited_states) < 20:
            return []

        critical_clusters = []

        if self.cluster_counts.sum() > 0:
            normalized_counts = self.cluster_counts / self.cluster_counts.sum()

            # Find median reward for non-zero clusters
            non_zero_mask = self.cluster_counts > 0
            if np.any(non_zero_mask):
                median_reward = np.median(self.cluster_rewards[non_zero_mask])

                # Find clusters with extreme properties
                for i in range(self.n_clusters):
                    if self.cluster_counts[i] > 0:
                        # High count, high reward = attractor
                        if (
                            normalized_counts[i] > 0.15
                            and self.cluster_rewards[i] > median_reward
                        ):
                            cluster_center = self.clusterer.cluster_centers_[i]
                            critical_clusters.append(
                                (
                                    cluster_center,
                                    "attractor",
                                    self.cluster_counts[i],
                                    self.cluster_rewards[i],
                                )
                            )

                        # Low count, low reward = repellor
                        elif (
                            normalized_counts[i] < 0.05
                            and self.cluster_rewards[i] < median_reward
                        ):
                            cluster_center = self.clusterer.cluster_centers_[i]
                            critical_clusters.append(
                                (
                                    cluster_center,
                                    "repellor",
                                    self.cluster_counts[i],
                                    self.cluster_rewards[i],
                                )
                            )

        return critical_clusters

    def compute_density_landscape(self, grid_resolution=30):
        """Compute density landscape in 2D projection for visualization"""
        if len(self.visited_states) < 10:
            return None, None, None

        states_array = np.array(self.visited_states)

        # If states are already 2D, use them directly
        if states_array.shape[1] == 2:
            # Create grid
            x_min, y_min = states_array.min(axis=0) - 0.5
            x_max, y_max = states_array.max(axis=0) + 0.5

            x = np.linspace(x_min, x_max, grid_resolution)
            y = np.linspace(y_min, y_max, grid_resolution)
            X, Y = np.meshgrid(x, y)
            grid_points = np.vstack([X.ravel(), Y.ravel()]).T

            # Simple density estimation
            if len(states_array) > 50:
                kde = KernelDensity(bandwidth=0.5)
                kde.fit(states_array)

                # Compute density
                log_density = kde.score_samples(grid_points)
                density = np.exp(log_density).reshape(X.shape)

                # Morse function = -log(density)
                morse_values = -log_density.reshape(X.shape)

                return X, Y, morse_values
            else:
                # Fallback: histogram
                H, xedges, yedges = np.histogram2d(
                    states_array[:, 0], states_array[:, 1], bins=grid_resolution
                )
                X, Y = np.meshgrid(
                    (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
                )
                morse_values = -np.log(H.T + 1e-6)
                return X, Y, morse_values
        else:
            # Project to 2D for visualization
            pca_2d = PCA(n_components=2)
            states_2d = pca_2d.fit_transform(states_array)

            # Create grid in 2D space
            x_min, y_min = states_2d.min(axis=0) - 0.5
            x_max, y_max = states_2d.max(axis=0) + 0.5

            x = np.linspace(x_min, x_max, grid_resolution)
            y = np.linspace(y_min, y_max, grid_resolution)
            X, Y = np.meshgrid(x, y)
            grid_points = np.vstack([X.ravel(), Y.ravel()]).T

            # Density estimation
            if len(states_2d) > 50:
                kde = KernelDensity(bandwidth=0.5)
                kde.fit(states_2d)

                log_density = kde.score_samples(grid_points)
                morse_values = -log_density.reshape(X.shape)
                return X, Y, morse_values
            else:
                H, xedges, yedges = np.histogram2d(
                    states_2d[:, 0], states_2d[:, 1], bins=grid_resolution
                )
                X, Y = np.meshgrid(
                    (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
                )
                morse_values = -np.log(H.T + 1e-6)
                return X, Y, morse_values

    def get_topology_statistics(self):
        """Get statistics about topological features"""
        critical_clusters = self.find_critical_clusters()

        stats = {
            "n_critical_clusters": len(critical_clusters),
            "n_attractors": sum(
                1 for _, t, _, _ in critical_clusters if t == "attractor"
            ),
            "n_repellors": sum(
                1 for _, t, _, _ in critical_clusters if t == "repellor"
            ),
            "total_states": len(self.visited_states),
            "unique_clusters": np.sum(self.cluster_counts > 0)
            if self.cluster_counts is not None
            else 0,
        }

        # Add cluster quality metrics
        if self.cluster_counts is not None and len(self.cluster_counts) > 0:
            non_zero_counts = self.cluster_counts[self.cluster_counts > 0]
            if len(non_zero_counts) > 0:
                stats["avg_cluster_size"] = np.mean(non_zero_counts)
                stats["max_cluster_size"] = np.max(non_zero_counts)
                stats["cluster_entropy"] = self._compute_cluster_entropy()

        return stats

    def _compute_cluster_entropy(self):
        """Compute entropy of cluster distribution"""
        if self.cluster_counts is None or self.cluster_counts.sum() == 0:
            return 0

        probs = self.cluster_counts / self.cluster_counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log(probs + 1e-10))


# ============================================
# 4. SIMPLE HUMANOID AGENT AND ENVIRONMENT
# ============================================


class SimpleHumanoidAgent:
    """Simple agent for HumanoidStandup"""

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Simple linear policy
        self.policy_weights = np.random.randn(state_dim, action_dim) * 0.01
        self.policy_bias = np.zeros(action_dim)

        # Learning parameters
        self.learning_rate = 1e-4
        self.gamma = 0.99

        # Memory
        self.memory = deque(maxlen=1000)

    def get_action(self, state):
        """Get action from policy"""
        # Linear policy with noise for exploration
        mean_action = state @ self.policy_weights + self.policy_bias

        # Add exploration noise
        noise = np.random.normal(0, 0.1, size=self.action_dim)
        action = mean_action + noise

        # Clip to environment bounds
        action = np.clip(action, -1.0, 1.0)

        return action

    def update(self, state, action, reward, next_state, done):
        """Simple policy update"""
        # Store experience
        self.memory.append((state, action, reward, next_state, done))

        # Simple REINFORCE-like update
        if len(self.memory) >= 32:
            batch = list(self.memory)[-32:]

            # Compute advantages (simplified)
            advantages = []
            for _, _, r, _, d in batch:
                if d:
                    advantages.append(r)
                else:
                    advantages.append(r + self.gamma * 10)  # Rough estimate

            advantages = np.array(advantages)
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / advantages.std()

            # Update policy
            states = np.array([s for s, _, _, _, _ in batch])
            actions = np.array([a for _, a, _, _, _ in batch])

            # Policy gradient
            policy_grad = states.T @ (advantages.reshape(-1, 1) * actions)
            self.policy_weights += self.learning_rate * policy_grad / len(batch)


class HumanoidTopologyAnalyzer:
    """Main class for humanoid topology analysis"""

    def __init__(self, env_name="HumanoidStandup-v4", max_steps=200, n_episodes=30):
        self.env_name = env_name
        self.max_steps = max_steps
        self.n_episodes = n_episodes

        # Initialize environment
        try:
            self.env = gym.make(env_name, render_mode=None)
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            print(
                f"✓ Loaded {env_name}: state_dim={self.state_dim}, action_dim={self.action_dim}"
            )
        except Exception as e:
            print(f"✗ Could not load {env_name}: {e}")
            print("Falling back to synthetic data mode...")
            self.env = None
            self.state_dim = 376  # Default humanoid dimensions
            self.action_dim = 17

        # Initialize state reducer
        self.state_reducer = StateReducer(method="pca", n_components=3)

        # Initialize analyzers
        self.loop_detector = HighDimLoopDetector(
            state_reducer=self.state_reducer, epsilon=0.4, min_loop_length=15
        )

        self.morse_analyzer = HighDimMorseAnalyzer(
            state_reducer=self.state_reducer, n_clusters=12
        )

        # Initialize agent
        self.agent = SimpleHumanoidAgent(self.state_dim, self.action_dim)

        # Results storage
        self.results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "loop_stats": [],
            "topology_stats": [],
            "states_history": [],
            "rewards_history": [],
        }

    def run_synthetic_episode(self):
        """Generate synthetic episode data for testing"""
        n_steps = np.random.randint(50, 150)

        # Generate synthetic states (humanoid-like)
        episode_states = []
        episode_rewards = []

        # Start from "standing" configuration
        state = np.zeros(self.state_dim)
        state[2] = 1.0  # z-position (height)

        for step in range(n_steps):
            # Simulate motion
            state = state + np.random.normal(0, 0.02, self.state_dim)

            # Keep z-position reasonable
            state[2] = np.clip(state[2] + np.random.normal(0, 0.05), 0.5, 2.0)

            # Reward for standing (higher is better)
            reward = state[2] * 10  # Reward proportional to height
            reward += np.random.normal(0, 0.5)  # Add noise

            # Penalty for extreme angles
            angle_penalty = np.sum(state[3:10] ** 2) * 0.1
            reward -= angle_penalty

            episode_states.append(state.copy())
            episode_rewards.append(reward)

            # Random termination
            if np.random.random() < 0.02:
                break

        total_reward = sum(episode_rewards)
        return episode_states, episode_rewards, total_reward, len(episode_states)

    def run_real_episode(self):
        """Run a real episode with the environment"""
        if self.env is None:
            return self.run_synthetic_episode()

        state, _ = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        episode_states = [state.copy()]
        episode_rewards = []

        while not done and not truncated and steps < self.max_steps:
            # Get action
            action = self.agent.get_action(state)

            # Take step
            next_state, reward, done, truncated, _ = self.env.step(action)

            # Update agent
            self.agent.update(state, action, reward, next_state, done)

            # Store data
            episode_states.append(next_state.copy())
            episode_rewards.append(reward)

            # Update tracking
            state = next_state
            total_reward += reward
            steps += 1

        return episode_states, episode_rewards, total_reward, steps

    def analyze_episode(self, episode_states, episode_rewards):
        """Analyze topological features of an episode"""
        # Update state reducer with new states
        if len(episode_states) > 10:
            # Use every 5th state to avoid overfitting
            sample_indices = range(
                0, len(episode_states), max(1, len(episode_states) // 20)
            )
            sample_states = [
                episode_states[i] for i in sample_indices if i < len(episode_states)
            ]
            self.state_reducer.fit(sample_states)

        # Add to analyzers
        self.loop_detector.add_trajectory(episode_states, episode_rewards)
        self.morse_analyzer.update(episode_states, episode_rewards)

        # Detect loops
        loops = self.loop_detector.detect_loops_simple()

        # Get statistics
        loop_stats = self.loop_detector.get_loop_statistics()
        topology_stats = self.morse_analyzer.get_topology_statistics()

        return loops, {**loop_stats, **topology_stats}

    def run_experiment(self):
        """Run the main experiment"""
        print(f"\n{'=' * 60}")
        print(f"Running Topological Analysis on {self.env_name}")
        print(f"{'=' * 60}")

        for episode in range(self.n_episodes):
            print(f"\nEpisode {episode + 1}/{self.n_episodes}")

            # Run episode
            if self.env is None:
                print("  Using synthetic data")
                episode_states, episode_rewards, total_reward, steps = (
                    self.run_synthetic_episode()
                )
            else:
                episode_states, episode_rewards, total_reward, steps = (
                    self.run_real_episode()
                )

            # Analyze topology every 3 episodes
            if episode % 3 == 0 and len(episode_states) > 10:
                print("  Analyzing topology...")
                loops, stats = self.analyze_episode(episode_states, episode_rewards)

                # Store results
                self.results["loop_stats"].append(stats)
                self.results["topology_stats"].append(stats)

                print(
                    f"    Loops: {stats['n_loops']}, "
                    f"Clusters: {stats['n_critical_clusters']}, "
                    f"Entropy: {stats.get('cluster_entropy', 0):.3f}"
                )

            # Store episode results
            self.results["episode_rewards"].append(total_reward)
            self.results["episode_lengths"].append(steps)
            self.results["states_history"].append(episode_states)
            self.results["rewards_history"].append(episode_rewards)

            print(f"  Reward: {total_reward:.1f}, Steps: {steps}")

            # Early stopping for synthetic mode
            if (
                self.env is None
                and episode >= 10
                and np.mean(self.results["episode_rewards"][-5:]) > 500
            ):
                print("  Good performance achieved, stopping early.")
                break

        # Final analysis
        self._final_analysis()

        return self.results

    def _final_analysis(self):
        """Perform final analysis and create visualizations"""
        print(f"\n{'=' * 60}")
        print("Experiment Complete - Final Analysis")
        print(f"{'=' * 60}")

        # Print summary
        if self.results["episode_rewards"]:
            final_reward = self.results["episode_rewards"][-1]
            avg_reward = np.mean(self.results["episode_rewards"][-5:])

            print(f"\nPerformance Summary:")
            print(f"  Final episode reward: {final_reward:.1f}")
            print(f"  Average last 5 episodes: {avg_reward:.1f}")
            print(f"  Max reward: {max(self.results['episode_rewards']):.1f}")

        # Topology summary
        if self.results["loop_stats"]:
            final_stats = self.results["loop_stats"][-1]

            print(f"\nTopology Summary:")
            print(
                f"  Total loops detected: {sum(s.get('n_loops', 0) for s in self.results['loop_stats'])}"
            )
            print(f"  Critical clusters: {final_stats.get('n_critical_clusters', 0)}")
            print(f"    - Attractors: {final_stats.get('n_attractors', 0)}")
            print(f"    - Repellors: {final_stats.get('n_repellors', 0)}")
            print(
                f"  State space coverage: {final_stats.get('total_states', 0)} states"
            )
            print(f"  Cluster entropy: {final_stats.get('cluster_entropy', 0):.3f}")

        # Generate visualizations
        self._generate_visualizations()

    def _generate_visualizations(self):
        """Generate visualizations"""
        print("\nGenerating visualizations...")

        # 1. Training progress
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))

        # Episode rewards
        episodes = range(1, len(self.results["episode_rewards"]) + 1)
        axes1[0, 0].plot(episodes, self.results["episode_rewards"], "b-", linewidth=2)
        axes1[0, 0].set_xlabel("Episode")
        axes1[0, 0].set_ylabel("Reward")
        axes1[0, 0].set_title("Training Progress")
        axes1[0, 0].grid(True, alpha=0.3)

        # Add moving average
        if len(self.results["episode_rewards"]) > 5:
            window = min(5, len(self.results["episode_rewards"]))
            moving_avg = np.convolve(
                self.results["episode_rewards"], np.ones(window) / window, mode="valid"
            )
            axes1[0, 0].plot(
                episodes[window - 1 :],
                moving_avg,
                "r-",
                linewidth=3,
                label=f"{window}-episode avg",
            )
            axes1[0, 0].legend()

        # Loop discovery
        if self.results["loop_stats"]:
            loop_episodes = [i * 3 for i in range(len(self.results["loop_stats"]))]
            n_loops = [s.get("n_loops", 0) for s in self.results["loop_stats"]]

            axes1[0, 1].plot(loop_episodes, n_loops, "g-o", linewidth=2)
            axes1[0, 1].set_xlabel("Episode")
            axes1[0, 1].set_ylabel("Loops Detected")
            axes1[0, 1].set_title("Loop Discovery Over Time")
            axes1[0, 1].grid(True, alpha=0.3)

        # Critical clusters
        if self.results["loop_stats"]:
            n_attractors = [
                s.get("n_attractors", 0) for s in self.results["loop_stats"]
            ]
            n_repellors = [s.get("n_repellors", 0) for s in self.results["loop_stats"]]

            axes1[0, 2].plot(
                loop_episodes, n_attractors, "b-", label="Attractors", linewidth=2
            )
            axes1[0, 2].plot(
                loop_episodes, n_repellors, "r-", label="Repellors", linewidth=2
            )
            axes1[0, 2].set_xlabel("Episode")
            axes1[0, 2].set_ylabel("Count")
            axes1[0, 2].set_title("Critical Clusters Evolution")
            axes1[0, 2].legend()
            axes1[0, 2].grid(True, alpha=0.3)

        # State space coverage
        if self.results["loop_stats"]:
            total_states = [
                s.get("total_states", 0) for s in self.results["loop_stats"]
            ]
            axes1[1, 0].plot(loop_episodes, total_states, "purple", linewidth=2)
            axes1[1, 0].set_xlabel("Episode")
            axes1[1, 0].set_ylabel("States Visited")
            axes1[1, 0].set_title("State Space Coverage")
            axes1[1, 0].grid(True, alpha=0.3)

        # Cluster entropy
        if self.results["loop_stats"]:
            entropy = [s.get("cluster_entropy", 0) for s in self.results["loop_stats"]]
            axes1[1, 1].plot(loop_episodes, entropy, "orange", linewidth=2)
            axes1[1, 1].set_xlabel("Episode")
            axes1[1, 1].set_ylabel("Entropy")
            axes1[1, 1].set_title("Exploration Diversity (Entropy)")
            axes1[1, 1].grid(True, alpha=0.3)

        # Average loop length
        if self.results["loop_stats"]:
            avg_length = [s.get("avg_length", 0) for s in self.results["loop_stats"]]
            axes1[1, 2].plot(loop_episodes, avg_length, "brown", linewidth=2)
            axes1[1, 2].set_xlabel("Episode")
            axes1[1, 2].set_ylabel("Average Loop Length")
            axes1[1, 2].set_title("Loop Complexity")
            axes1[1, 2].grid(True, alpha=0.3)

        plt.suptitle(
            f"Topological Analysis: {self.env_name}", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig("topology_analysis_results.png", dpi=150, bbox_inches="tight")

        # 2. State space visualization
        if len(self.results["states_history"]) > 0:
            # Combine all states
            all_states = []
            for episode_states in self.results["states_history"]:
                all_states.extend(episode_states)

            if len(all_states) > 10:
                all_states_array = np.array(all_states)

                # Reduce to 2D for visualization
                pca_2d = PCA(n_components=2)
                states_2d = pca_2d.fit_transform(
                    all_states_array[:1000]
                )  # Limit for speed

                fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

                # Color by episode
                colors = []
                current_idx = 0
                for i, episode_states in enumerate(self.results["states_history"]):
                    n_states = min(len(episode_states), 50)  # Sample each episode
                    colors.extend([i] * n_states)
                    current_idx += n_states

                if len(colors) > len(states_2d):
                    colors = colors[: len(states_2d)]

                scatter1 = axes2[0].scatter(
                    states_2d[: len(colors), 0],
                    states_2d[: len(colors), 1],
                    c=colors[: len(states_2d)],
                    cmap="tab20",
                    alpha=0.6,
                    s=20,
                )
                axes2[0].set_xlabel("PCA Component 1")
                axes2[0].set_ylabel("PCA Component 2")
                axes2[0].set_title("State Space Colored by Episode")
                axes2[0].grid(True, alpha=0.3)

                # Color by reward (use first 1000 states)
                all_rewards = []
                for rewards in self.results["rewards_history"]:
                    all_rewards.extend(rewards)

                if len(all_rewards) >= len(states_2d):
                    scatter2 = axes2[1].scatter(
                        states_2d[:, 0],
                        states_2d[:, 1],
                        c=all_rewards[: len(states_2d)],
                        cmap="viridis",
                        alpha=0.6,
                        s=20,
                    )
                    axes2[1].set_xlabel("PCA Component 1")
                    axes2[1].set_ylabel("PCA Component 2")
                    axes2[1].set_title("State Space Colored by Reward")
                    plt.colorbar(scatter2, ax=axes2[1], label="Reward")
                    axes2[1].grid(True, alpha=0.3)

                plt.suptitle(
                    "State Space Projection (2D PCA)", fontsize=14, fontweight="bold"
                )
                plt.tight_layout()
                plt.savefig("state_space_projection.png", dpi=150, bbox_inches="tight")

        # 3. Morse function landscape
        X, Y, morse_values = self.morse_analyzer.compute_density_landscape()

        if morse_values is not None:
            fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))

            contour = ax3.contourf(
                X, Y, morse_values, levels=20, cmap="viridis", alpha=0.8
            )

            # Plot critical clusters
            critical_clusters = self.morse_analyzer.find_critical_clusters()
            for center, ctype, count, reward in critical_clusters:
                if len(center) >= 2:
                    color = "blue" if ctype == "attractor" else "red"
                    marker = "o" if ctype == "attractor" else "^"
                    size = 50 + min(count, 100) * 0.5  # Scale by count

                    ax3.scatter(
                        center[0],
                        center[1],
                        s=size,
                        c=color,
                        marker=marker,
                        edgecolor="white",
                        linewidth=2,
                        label=f"{ctype.capitalize()} (n={count})",
                    )

            ax3.set_xlabel("Projected Dimension 1")
            ax3.set_ylabel("Projected Dimension 2")
            ax3.set_title("Morse Function Landscape with Critical Clusters")
            ax3.grid(True, alpha=0.3)

            # Remove duplicate labels
            handles, labels = ax3.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax3.legend(by_label.values(), by_label.keys(), fontsize=9)

            plt.colorbar(contour, ax=ax3, label="Morse Function Value (-log density)")
            plt.tight_layout()
            plt.savefig("morse_landscape.png", dpi=150, bbox_inches="tight")

        print("\nVisualizations saved:")
        print("  - topology_analysis_results.png")
        print("  - state_space_projection.png")
        print("  - morse_landscape.png")

        plt.show()

    def save_results(self, filename="topology_results.pkl"):
        """Save results to file"""
        with open(filename, "wb") as f:
            pickle.dump(self.results, f)
        print(f"\nResults saved to {filename}")

    def close(self):
        """Close environment"""
        if self.env is not None:
            self.env.close()


# ============================================
# 5. MAIN EXECUTION
# ============================================


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("TOPOLOGICAL ANALYSIS OF HUMANOID/RL ENVIRONMENTS")
    print("=" * 70)
    print("\nThis script analyzes topological features in reinforcement learning.")
    print("Features analyzed:")
    print("  1. Loops in agent trajectories")
    print("  2. Morse theory critical points (attractors/repellors)")
    print("  3. State space coverage and exploration diversity")

    print("\nOptions:")
    print("  1. Run with HumanoidStandup-v4 (requires MuJoCo)")
    print("  2. Run with synthetic data (no MuJoCo required)")

    try:
        choice = input("\nEnter choice (1 or 2, default=2): ").strip()

        if choice == "1":
            # Try to import gymnasium and check for MuJoCo
            try:
                import gymnasium

                # Try to create the environment
                env = gym.make("HumanoidStandup-v4", render_mode=None)
                env.close()
                print("\n✓ MuJoCo environment available")

                # Run with real environment
                analyzer = HumanoidTopologyAnalyzer(
                    env_name="HumanoidStandup-v4",
                    max_steps=200,
                    n_episodes=20,  # Reduced for speed
                )

            except Exception as e:
                print(f"\n✗ Could not load HumanoidStandup-v4: {e}")
                print("Falling back to synthetic data...")
                analyzer = HumanoidTopologyAnalyzer(
                    env_name="SyntheticHumanoid", max_steps=200, n_episodes=15
                )
        else:
            # Run with synthetic data
            analyzer = HumanoidTopologyAnalyzer(
                env_name="SyntheticHumanoid", max_steps=200, n_episodes=15
            )

        # Run experiment
        results = analyzer.run_experiment()

        # Save results
        analyzer.save_results()

        # Close
        analyzer.close()

        print(f"\n{'=' * 70}")
        print("Experiment completed successfully!")
        print("=" * 70)

        return analyzer, results

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


def quick_test():
    """Quick test without environment dependencies"""
    print("\nRunning quick topological analysis test...")

    # Create synthetic data
    np.random.seed(42)

    # Create a state reducer
    state_reducer = StateReducer(method="pca", n_components=3)

    # Generate synthetic trajectories
    n_trajectories = 5
    all_states = []

    for i in range(n_trajectories):
        # Create a looping trajectory
        n_points = 50
        theta = np.linspace(0, 4 * np.pi, n_points)

        # 3D spiral
        x = np.cos(theta) + np.random.normal(0, 0.1, n_points)
        y = np.sin(theta) + np.random.normal(0, 0.1, n_points)
        z = theta / (4 * np.pi) + np.random.normal(0, 0.05, n_points)

        # Convert to high-dimensional (376D like humanoid)
        states = []
        for j in range(n_points):
            # Create a high-dim state with structure
            state = np.zeros(376)
            state[0] = x[j]  # x position
            state[1] = y[j]  # y position
            state[2] = z[j]  # z position (height)

            # Add some structured noise
            state[3:10] = np.sin(theta[j] + np.arange(7) * 0.5) * 0.1
            state[10:20] = np.cos(theta[j] + np.arange(10) * 0.3) * 0.05

            states.append(state)

        all_states.append(np.array(states))

    # Fit reducer
    sample_states = []
    for traj in all_states:
        sample_states.extend(traj[::5])  # Sample every 5th state

    state_reducer.fit(sample_states)

    # Test loop detector
    print("\nTesting loop detection...")
    loop_detector = HighDimLoopDetector(state_reducer, epsilon=0.3, min_loop_length=10)

    for traj in all_states:
        # Generate synthetic rewards
        rewards = np.random.uniform(0, 1, len(traj))
        loop_detector.add_trajectory(
            [traj[i] for i in range(len(traj))], rewards.tolist()
        )

    loops = loop_detector.detect_loops_simple()
    print(f"Detected {len(loops)} loops")

    # Test Morse analyzer
    print("\nTesting Morse theory analysis...")
    morse_analyzer = HighDimMorseAnalyzer(state_reducer, n_clusters=8)

    for traj in all_states:
        rewards = np.random.uniform(0, 2, len(traj))
        morse_analyzer.update([traj[i] for i in range(len(traj))], rewards.tolist())

    critical_clusters = morse_analyzer.find_critical_clusters()
    print(f"Found {len(critical_clusters)} critical clusters")

    stats = morse_analyzer.get_topology_statistics()
    print(f"State space coverage: {stats['total_states']} states")
    print(f"Cluster entropy: {stats.get('cluster_entropy', 0):.3f}")

    # Simple visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Reduce all states to 2D for visualization
    pca_2d = PCA(n_components=2)
    all_states_flat = np.vstack(all_states[:3])  # First 3 trajectories
    states_2d = pca_2d.fit_transform(all_states_flat)

    # Plot trajectories
    start_idx = 0
    colors = ["blue", "green", "red"]
    for i, traj in enumerate(all_states[:3]):
        n_points = len(traj)
        traj_2d = states_2d[start_idx : start_idx + n_points]
        ax.plot(
            traj_2d[:, 0],
            traj_2d[:, 1],
            "-",
            color=colors[i],
            alpha=0.6,
            linewidth=2,
            label=f"Trajectory {i + 1}",
        )
        start_idx += n_points

    # Plot critical clusters if any
    if critical_clusters:
        cluster_centers = [center for center, _, _, _ in critical_clusters]
        cluster_types = [ctype for _, ctype, _, _ in critical_clusters]

        for center, ctype in zip(cluster_centers, cluster_types):
            if len(center) >= 2:
                color = "red" if ctype == "repellor" else "green"
                marker = "^" if ctype == "repellor" else "o"
                ax.scatter(
                    center[0],
                    center[1],
                    s=100,
                    c=color,
                    marker=marker,
                    edgecolor="white",
                    linewidth=2,
                    label=f"{ctype.capitalize()}",
                )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Topological Analysis Test: Trajectories and Critical Points")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("quick_topology_test.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nQuick test completed!")
    print("Results saved to quick_topology_test.png")

    return loop_detector, morse_analyzer


# ============================================
# 6. ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TOPOLOGICAL RL ANALYSIS")
    print("=" * 70)

    # Check for basic dependencies
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.spatial import KDTree

        print("✓ Basic dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install numpy matplotlib scipy scikit-learn")
        exit(1)

    print("\nChoose mode:")
    print("  1. Full experiment (with or without MuJoCo)")
    print("  2. Quick test (no environment needed)")

    try:
        choice = input("\nEnter choice (1 or 2, default=2): ").strip()

        if choice == "1":
            main()
        else:
            quick_test()

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Running quick test instead...")
        quick_test()

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
