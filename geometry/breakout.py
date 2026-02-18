"""
Stratification Analysis for Atari Breakout
Analyzes hierarchical structure in learned representations
Handles visual inputs and discrete action space
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Core dependencies
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import center_of_mass
import scipy.signal

# Deep learning
import torch.nn as nn
import torch.nn.functional as F

# Optional TDA
try:
    from ripser import ripser
    from persim import plot_diagrams

    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


@dataclass
class BreakoutConfig:
    """Configuration for Breakout analysis"""

    # Data collection
    n_episodes: int = 50
    max_steps_per_episode: int = 10000
    frame_stack: int = 4  # Number of stacked frames
    frame_skip: int = 4  # Action repeat

    # Analysis methods
    analyze_spatial: bool = True  # Ball/paddle positions
    analyze_game_phase: bool = True  # Game progression
    analyze_cnn_features: bool = True  # CNN layer activations
    analyze_temporal: bool = True  # Temporal patterns

    # Clustering parameters
    min_cluster_size: int = 50
    eps: float = 0.3

    # Visualization
    save_plots: bool = True
    plot_dir: str = "./breakout_analysis"
    figsize: Tuple[int, int] = (20, 12)


class BreakoutFeatureExtractor:
    """
    Extracts meaningful features from Breakout observations
    Handles both raw pixels and learned representations
    """

    def __init__(self, model, config: BreakoutConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # For tracking game objects
        self.ball_positions = []
        self.paddle_positions = []
        self.brick_counts = []
        self.scores = []

        # Color thresholds for object detection (Atari specific)
        self.ball_color = (236, 236, 236)  # White
        self.paddle_color = (200, 72, 72)  # Red
        self.brick_colors = [
            (66, 72, 200),  # Blue
            (72, 160, 72),  # Green
            (200, 72, 72),  # Orange/Red
            (236, 236, 236),  # White (top rows)
        ]

    def detect_ball(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Detect ball position in frame"""
        # Convert to RGB if needed
        if frame.shape[-1] == 3:
            frame_rgb = frame
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Look for white ball
        ball_mask = cv2.inRange(frame_rgb, (230, 230, 230), (255, 255, 255))

        # Find contours
        contours, _ = cv2.findContours(
            ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Get largest white object (should be ball)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5:  # Minimum size
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    return cx / frame.shape[1], cy / frame.shape[0]  # Normalize

        return None, None

    def detect_paddle(self, frame: np.ndarray) -> Optional[float]:
        """Detect paddle position"""
        if frame.shape[-1] == 3:
            frame_rgb = frame
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Look for red paddle
        paddle_mask = cv2.inRange(frame_rgb, (180, 50, 50), (220, 100, 100))

        # Find horizontal line (paddle)
        contours, _ = cv2.findContours(
            paddle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x + w / 2) / frame.shape[1]  # Normalized center

        return None

    def count_bricks(self, frame: np.ndarray) -> int:
        """Count remaining bricks"""
        if frame.shape[-1] == 3:
            frame_rgb = frame
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Combine all brick colors
        brick_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for color in self.brick_colors:
            lower = np.array([c - 20 for c in color])
            upper = np.array([c + 20 for c in color])
            brick_mask = cv2.bitwise_or(
                brick_mask, cv2.inRange(frame_rgb, lower, upper)
            )

        # Count connected components
        num_labels, _ = cv2.connectedComponents(brick_mask)
        return max(0, num_labels - 1)  # Subtract background

    def extract_cnn_features(self, obs: torch.Tensor) -> np.ndarray:
        """Extract features from CNN layers"""
        with torch.no_grad():
            # Move to device
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs).to(self.device)
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)

            # Get CNN features
            if hasattr(self.model, "policy") and hasattr(
                self.model.policy, "cnn_extractor"
            ):
                features = self.model.policy.cnn_extractor(obs)
                return features.cpu().numpy().flatten()

            # Try to get features from different model architectures
            elif hasattr(self.model, "cnn_extractor"):
                features = self.model.cnn_extractor(obs)
                return features.cpu().numpy().flatten()

            else:
                return obs.cpu().numpy().flatten()

    def get_game_phase(self, brick_count: int, ball_y: Optional[float]) -> str:
        """Determine game phase"""
        if ball_y is None:
            return "lost_ball"
        elif brick_count > 30:
            return "early_game"
        elif brick_count > 15:
            return "mid_game"
        elif brick_count > 0:
            return "late_game"
        else:
            return "game_over"

    def compute_ball_trajectory_features(self, positions: List[Tuple]) -> Dict:
        """Extract features from ball trajectory"""
        if len(positions) < 10:
            return {}

        positions = np.array(positions)
        x_vals, y_vals = positions[:, 0], positions[:, 1]

        # Remove None values
        valid = ~np.isnan(x_vals)
        if np.sum(valid) < 5:
            return {}

        x_vals = x_vals[valid]
        y_vals = y_vals[valid]

        # Compute features
        features = {
            "x_range": np.ptp(x_vals),
            "y_range": np.ptp(y_vals),
            "x_velocity": np.mean(np.diff(x_vals)) if len(x_vals) > 1 else 0,
            "y_velocity": np.mean(np.diff(y_vals)) if len(y_vals) > 1 else 0,
            "x_variance": np.var(x_vals),
            "y_variance": np.var(y_vals),
            "ball_active_ratio": np.sum(valid) / len(positions),
        }

        return features


class BreakoutStratificationAnalyzer:
    """
    Main analyzer for Breakout environment
    """

    def __init__(self, env: gym.Env, model, config: BreakoutConfig):
        self.env = env
        self.model = model
        self.config = config
        self.extractor = BreakoutFeatureExtractor(model, config)

        # Storage for collected data
        self.data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "cnn_features": [],
            "ball_positions": [],
            "paddle_positions": [],
            "brick_counts": [],
            "game_phases": [],
            "episode_ids": [],
            "timesteps": [],
        }

        # Create output directory
        if config.save_plots:
            import os

            os.makedirs(config.plot_dir, exist_ok=True)

    def collect_data(self):
        """Collect trajectories with game object tracking"""
        print(f"\n{'=' * 60}")
        print("COLLECTING BREAKOUT DATA")
        print(f"{'=' * 60}")
        print(f"Episodes: {self.config.n_episodes}")

        episode_rewards = []

        for episode in range(self.config.n_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            done = False
            truncated = False
            episode_reward = 0
            step = 0

            # Episode-specific tracking
            episode_ball_positions = []

            while not (done or truncated):
                # Get action from model
                if hasattr(self.model, "predict"):
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    # Handle different model types
                    with torch.no_grad():
                        obs_tensor = (
                            torch.FloatTensor(obs)
                            .unsqueeze(0)
                            .to(self.extractor.device)
                        )
                        if hasattr(self.model, "policy"):
                            action_probs = self.model.policy(obs_tensor)
                            action = torch.argmax(action_probs).item()
                        else:
                            action = self.env.action_space.sample()

                # Extract features
                cnn_features = self.extractor.extract_cnn_features(obs)

                # Detect game objects
                if len(obs.shape) == 3 and obs.shape[-1] == 3:
                    frame = obs
                else:
                    # Reshape if needed
                    frame = obs.transpose(1, 2, 0) if obs.shape[0] == 3 else obs
                    if frame.shape[-1] != 3:
                        frame = np.stack([frame] * 3, axis=-1)

                ball_x, ball_y = self.extractor.detect_ball(frame)
                paddle_x = self.extractor.detect_paddle(frame)
                brick_count = self.extractor.count_bricks(frame)
                game_phase = self.extractor.get_game_phase(brick_count, ball_y)

                # Store data
                self.data["observations"].append(
                    obs.flatten() if hasattr(obs, "flatten") else obs
                )
                self.data["actions"].append(action)
                self.data["cnn_features"].append(cnn_features)
                self.data["ball_positions"].append((ball_x, ball_y))
                self.data["paddle_positions"].append(paddle_x)
                self.data["brick_counts"].append(brick_count)
                self.data["game_phases"].append(game_phase)
                self.data["episode_ids"].append(episode)
                self.data["timesteps"].append(step)

                if ball_y is not None:
                    episode_ball_positions.append((ball_x, ball_y))

                # Step environment
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, done, truncated, info = step_result

                self.data["rewards"].append(reward)
                self.data["dones"].append(done or truncated)

                episode_reward += reward
                step += 1

                if step >= self.config.max_steps_per_episode:
                    break

            episode_rewards.append(episode_reward)

            # Store trajectory features
            traj_features = self.extractor.compute_ball_trajectory_features(
                episode_ball_positions
            )

            if (episode + 1) % 5 == 0:
                print(
                    f"Episode {episode + 1}: Reward={episode_reward:.0f}, "
                    f"Bricks={brick_count}, Phase={game_phase}"
                )

        # Convert to numpy arrays where possible
        for key in self.data:
            if key not in ["game_phases", "ball_positions"]:
                try:
                    if self.data[key] and isinstance(
                        self.data[key][0], (int, float, np.number)
                    ):
                        self.data[key] = np.array(self.data[key])
                    elif self.data[key] and hasattr(self.data[key][0], "__array__"):
                        self.data[key] = np.array([x.flatten() for x in self.data[key]])
                except:
                    pass  # Keep as list

        # Summary statistics
        print(f"\n{'=' * 60}")
        print("COLLECTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total transitions: {len(self.data['observations'])}")
        print(
            f"Average reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}"
        )

        # Game phase distribution
        phases, counts = np.unique(self.data["game_phases"], return_counts=True)
        print("\nGame phase distribution:")
        for phase, count in zip(phases, counts):
            print(
                f"  {phase}: {count} ({count / len(self.data['game_phases']) * 100:.1f}%)"
            )

        return self

    def analyze_spatial_structure(self) -> Dict:
        """
        Analyze spatial distribution of ball and paddle
        """
        print(f"\n{'=' * 60}")
        print("SPATIAL STRUCTURE ANALYSIS")
        print(f"{'=' * 60}")

        # Extract valid ball positions
        ball_positions = np.array(
            [
                (x, y)
                for x, y in self.data["ball_positions"]
                if x is not None and y is not None
            ]
        )

        paddle_positions = np.array(
            [p for p in self.data["paddle_positions"] if p is not None]
        )

        if len(ball_positions) == 0:
            print("No valid ball positions detected")
            return {}

        print(f"Valid ball positions: {len(ball_positions)}")
        print(f"Valid paddle positions: {len(paddle_positions)}")

        # 1. Ball position clustering
        ball_clusters = self._cluster_spatial_positions(ball_positions)

        # 2. Paddle-ball relationship
        if len(paddle_positions) > 0 and len(ball_positions) > 0:
            # Align by timestep (approximate)
            min_len = min(len(paddle_positions), len(ball_positions))
            paddle_ball_diff = np.abs(
                paddle_positions[:min_len] - ball_positions[:min_len, 0]
            )

            print(f"\nPaddle-ball alignment:")
            print(f"  Mean horizontal diff: {np.mean(paddle_ball_diff):.3f}")
            print(f"  Std diff: {np.std(paddle_ball_diff):.3f}")

        # 3. Identify spatial strata
        strata = self._identify_spatial_strata(ball_positions)

        return {
            "ball_positions": ball_positions,
            "paddle_positions": paddle_positions,
            "ball_clusters": ball_clusters,
            "spatial_strata": strata,
        }

    def _cluster_spatial_positions(self, positions: np.ndarray) -> np.ndarray:
        """Cluster spatial positions"""
        if len(positions) < self.config.min_cluster_size:
            return np.zeros(len(positions)) - 1

        # Use DBSCAN for spatial clustering
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(positions)
        labels = clustering.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"\nSpatial clustering:")
        print(f"  Found {n_clusters} spatial clusters")

        for label in range(n_clusters):
            mask = labels == label
            cluster_pos = positions[mask]
            print(
                f"  Cluster {label}: center=({np.mean(cluster_pos[:, 0]):.3f}, "
                f"{np.mean(cluster_pos[:, 1]):.3f}), size={np.sum(mask)}"
            )

        return labels

    def _identify_spatial_strata(self, positions: np.ndarray) -> Dict:
        """Identify distinct spatial strata (e.g., left/right, top/bottom)"""
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        # Find natural divisions
        strata = {}

        # Vertical strata (top, middle, bottom of screen)
        y_thresholds = np.percentile(y_coords, [33, 66])
        strata["vertical"] = {
            "top": np.mean(y_coords <= y_thresholds[0]),
            "middle": np.mean(
                (y_coords > y_thresholds[0]) & (y_coords <= y_thresholds[1])
            ),
            "bottom": np.mean(y_coords > y_thresholds[1]),
        }

        # Horizontal strata (left, center, right)
        x_thresholds = np.percentile(x_coords, [25, 75])
        strata["horizontal"] = {
            "left": np.mean(x_coords <= x_thresholds[0]),
            "center": np.mean(
                (x_coords > x_thresholds[0]) & (x_coords <= x_thresholds[1])
            ),
            "right": np.mean(x_coords > x_thresholds[1]),
        }

        print(f"\nSpatial strata proportions:")
        print(f"  Vertical: {strata['vertical']}")
        print(f"  Horizontal: {strata['horizontal']}")

        return strata

    def analyze_game_phase_structure(self) -> Dict:
        """
        Analyze structure across game phases
        """
        print(f"\n{'=' * 60}")
        print("GAME PHASE ANALYSIS")
        print(f"{'=' * 60}")

        phases = np.array(self.data["game_phases"])
        unique_phases = np.unique(phases)

        phase_stats = {}

        for phase in unique_phases:
            mask = phases == phase

            # Get actions in this phase
            phase_actions = np.array(self.data["actions"])[mask]

            # Get rewards
            phase_rewards = np.array(self.data["rewards"])[mask]

            # Get brick counts
            phase_bricks = np.array(self.data["brick_counts"])[mask]

            # Get CNN features
            if len(self.data["cnn_features"]) > 0:
                phase_features = np.array(self.data["cnn_features"])[mask]
                feature_mean = np.mean(phase_features, axis=0)
            else:
                phase_features = None
                feature_mean = None

            phase_stats[phase] = {
                "count": np.sum(mask),
                "action_distribution": np.bincount(phase_actions.astype(int))
                / len(phase_actions),
                "mean_reward": np.mean(phase_rewards),
                "mean_bricks": np.mean(phase_bricks),
                "feature_mean": feature_mean,
            }

            print(f"\nPhase: {phase}")
            print(f"  Samples: {phase_stats[phase]['count']}")
            print(f"  Mean reward: {phase_stats[phase]['mean_reward']:.3f}")
            print(f"  Mean bricks: {phase_stats[phase]['mean_bricks']:.1f}")
            print(f"  Action dist: {phase_stats[phase]['action_distribution']}")

        # Phase transition analysis
        phase_changes = []
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                phase_changes.append((phases[i - 1], phases[i], i))

        print(f"\nPhase transitions: {len(phase_changes)}")
        if len(phase_changes) > 0:
            # Most common transitions
            from collections import Counter

            transitions = Counter([(p[0], p[1]) for p in phase_changes])
            print("  Most common:")
            for (from_p, to_p), count in transitions.most_common(3):
                print(f"    {from_p} -> {to_p}: {count}")

        return phase_stats

    def analyze_cnn_feature_space(self) -> Dict:
        """
        Analyze structure in CNN feature space
        """
        print(f"\n{'=' * 60}")
        print("CNN FEATURE SPACE ANALYSIS")
        print(f"{'=' * 60}")

        if len(self.data["cnn_features"]) == 0:
            print("No CNN features available")
            return {}

        features = np.array(self.data["cnn_features"])
        print(f"Feature dimension: {features.shape[1]}")

        # 1. PCA analysis
        pca = PCA(n_components=min(10, features.shape[1]))
        features_pca = pca.fit_transform(features)

        print(f"\nPCA explained variance:")
        for i, ratio in enumerate(pca.explained_variance_ratio_[:5]):
            print(f"  PC{i + 1}: {ratio:.3f}")
        print(
            f"  Total (5 components): {np.sum(pca.explained_variance_ratio_[:5]):.3f}"
        )

        # 2. Cluster by game phase
        phases = np.array(self.data["game_phases"])
        unique_phases = np.unique(phases)

        # Compute phase separability
        phase_centers = {}
        for phase in unique_phases:
            mask = phases == phase
            phase_features = features_pca[mask, :2]  # Use first 2 PCs
            if len(phase_features) > 0:
                phase_centers[phase] = np.mean(phase_features, axis=0)

        # Compute distances between phase centers
        print(f"\nPhase separation in PC space:")
        phase_list = list(phase_centers.keys())
        for i in range(len(phase_list)):
            for j in range(i + 1, len(phase_list)):
                p1, p2 = phase_list[i], phase_list[j]
                dist = np.linalg.norm(phase_centers[p1] - phase_centers[p2])
                print(f"  {p1} <-> {p2}: {dist:.3f}")

        # 3. Cluster features
        clustering = HDBSCAN(min_cluster_size=self.config.min_cluster_size)
        feature_labels = clustering.fit_predict(features_pca[:, :5])

        n_clusters = len(set(feature_labels)) - (1 if -1 in feature_labels else 0)
        n_noise = list(feature_labels).count(-1)

        print(f"\nFeature clustering:")
        print(f"  Found {n_clusters} clusters")
        print(f"  Noise: {n_noise} ({n_noise / len(feature_labels) * 100:.1f}%)")

        # Analyze cluster composition by phase
        print(f"\nCluster-phase composition:")
        for cluster in range(n_clusters):
            cluster_mask = feature_labels == cluster
            cluster_phases = phases[cluster_mask]
            phase_dist = {
                p: np.sum(cluster_phases == p) / len(cluster_phases)
                for p in unique_phases
            }
            print(f"  Cluster {cluster}: {phase_dist}")

        return {
            "features_pca": features_pca,
            "pca": pca,
            "feature_labels": feature_labels,
            "phase_centers": phase_centers,
        }

    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze temporal dynamics and patterns
        """
        print(f"\n{'=' * 60}")
        print("TEMPORAL PATTERN ANALYSIS")
        print(f"{'=' * 60}")

        actions = np.array(self.data["actions"])
        rewards = np.array(self.data["rewards"])
        brick_counts = np.array(self.data["brick_counts"])

        # 1. Action patterns
        action_changes = np.diff(actions)
        action_switch_rate = np.sum(action_changes != 0) / len(action_changes)

        print(f"Action switch rate: {action_switch_rate:.3f}")

        # 2. Periodicity detection
        if len(actions) > 100:
            # Compute autocorrelation
            autocorr = np.correlate(
                actions - np.mean(actions), actions - np.mean(actions), mode="same"
            )
            autocorr = autocorr[len(autocorr) // 2 :]  # Take second half

            # Find peaks
            peaks = scipy.signal.find_peaks(autocorr, height=0.1 * np.max(autocorr))[0]

            if len(peaks) > 0:
                print(f"Detected periodicity: ~{peaks[0]} steps")

        # 3. Reward patterns
        # Find reward events (brick hits)
        reward_events = rewards > 0
        reward_intervals = np.diff(np.where(reward_events)[0])

        if len(reward_intervals) > 0:
            print(f"\nReward events:")
            print(f"  Total: {np.sum(reward_events)}")
            print(f"  Mean interval: {np.mean(reward_intervals):.1f} steps")
            print(f"  Interval std: {np.std(reward_intervals):.1f}")

        # 4. Brick destruction patterns
        brick_changes = np.diff(brick_counts)
        brick_loss_events = brick_changes < 0

        if np.sum(brick_loss_events) > 0:
            print(f"\nBrick destruction:")
            print(f"  Events: {np.sum(brick_loss_events)}")
            print(
                f"  Mean bricks per event: {-np.mean(brick_changes[brick_changes < 0]):.2f}"
            )

        return {
            "action_switch_rate": action_switch_rate,
            "reward_intervals": reward_intervals if len(reward_intervals) > 0 else None,
            "brick_loss_events": brick_loss_events,
        }

    def topological_analysis(self) -> Optional[Dict]:
        """
        Persistent homology analysis of state space
        """
        print(f"\n{'=' * 60}")
        print("TOPOLOGICAL DATA ANALYSIS")
        print(f"{'=' * 60}")

        if not RIPSER_AVAILABLE:
            print("Ripser not available. Skipping TDA.")
            return None

        # Use CNN features for TDA
        if len(self.data["cnn_features"]) > 0:
            features = np.array(self.data["cnn_features"])
        else:
            # Use ball positions if available
            ball_pos = np.array(
                [
                    (x, y)
                    for x, y in self.data["ball_positions"]
                    if x is not None and y is not None
                ]
            )
            if len(ball_pos) > 0:
                features = ball_pos
            else:
                print("No suitable features for TDA")
                return None

        # Sample for efficiency
        max_points = min(500, len(features))
        idx = np.random.choice(len(features), max_points, replace=False)
        features_sample = features[idx]

        print(f"Computing persistence on {max_points} points...")

        # Compute persistence
        diagrams = ripser(features_sample, maxdim=2)["dgms"]

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

    def visualize_all(self, results: Dict):
        """
        Create comprehensive visualization suite
        """
        print(f"\n{'=' * 60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'=' * 60}")

        fig = plt.figure(figsize=self.config.figsize)

        # 1. Ball position heatmap
        ax1 = fig.add_subplot(2, 3, 1)
        ball_positions = np.array(
            [
                (x, y)
                for x, y in self.data["ball_positions"]
                if x is not None and y is not None
            ]
        )
        if len(ball_positions) > 0:
            heatmap, xedges, yedges = np.histogram2d(
                ball_positions[:, 0],
                ball_positions[:, 1],
                bins=20,
                range=[[0, 1], [0, 1]],
            )
            im1 = ax1.imshow(
                heatmap.T,
                origin="lower",
                aspect="auto",
                extent=[0, 1, 0, 1],
                cmap="hot",
            )
            ax1.set_xlabel("X Position")
            ax1.set_ylabel("Y Position")
            ax1.set_title("Ball Position Heatmap")
            plt.colorbar(im1, ax=ax1)

        # 2. Action distribution by game phase
        ax2 = fig.add_subplot(2, 3, 2)
        phases = np.array(self.data["game_phases"])
        actions = np.array(self.data["actions"])
        unique_phases = np.unique(phases)

        for i, phase in enumerate(unique_phases):
            mask = phases == phase
            phase_actions = actions[mask]
            if len(phase_actions) > 0:
                action_dist = np.bincount(phase_actions.astype(int)) / len(
                    phase_actions
                )
                ax2.bar(
                    np.arange(len(action_dist)) + i * 0.2,
                    action_dist,
                    width=0.2,
                    label=phase,
                    alpha=0.7,
                )

        ax2.set_xlabel("Action")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Actions by Game Phase")
        ax2.legend()
        ax2.set_xticks(np.arange(4))

        # 3. Brick count over time (sample episode)
        ax3 = fig.add_subplot(2, 3, 3)
        episode_0 = np.array(self.data["episode_ids"]) == 0
        if np.sum(episode_0) > 0:
            bricks_ep0 = np.array(self.data["brick_counts"])[episode_0]
            ax3.plot(bricks_ep0, "b-", linewidth=1)
            ax3.set_xlabel("Timestep")
            ax3.set_ylabel("Brick Count")
            ax3.set_title("Brick Count (Episode 0)")
            ax3.grid(True, alpha=0.3)

        # 4. CNN feature space (PCA)
        ax4 = fig.add_subplot(2, 3, 4)
        if "cnn" in results and results["cnn"] is not None:
            features_pca = results["cnn"]["features_pca"]
            phases = np.array(self.data["game_phases"])

            for phase in np.unique(phases):
                mask = phases == phase
                ax4.scatter(
                    features_pca[mask, 0],
                    features_pca[mask, 1],
                    label=phase,
                    alpha=0.5,
                    s=5,
                )

            ax4.set_xlabel("PC1")
            ax4.set_ylabel("PC2")
            ax4.set_title("CNN Feature Space")
            ax4.legend()

        # 5. Temporal patterns
        ax5 = fig.add_subplot(2, 3, 5)
        rewards = np.array(self.data["rewards"])
        actions = np.array(self.data["actions"])

        # Plot rewards and actions over first 200 steps
        n_steps = min(200, len(rewards))
        x = np.arange(n_steps)

        ax5.plot(x, rewards[:n_steps] * 10, "g-", alpha=0.7, label="Reward (x10)")
        ax5.plot(x, actions[:n_steps], "b-", alpha=0.5, label="Action")
        ax5.set_xlabel("Timestep")
        ax5.set_ylabel("Value")
        ax5.set_title("Temporal Patterns")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Game phase transitions
        ax6 = fig.add_subplot(2, 3, 6)
        phases_array = np.array(self.data["game_phases"])
        phase_codes = {p: i for i, p in enumerate(np.unique(phases_array))}
        phase_numeric = np.array([phase_codes[p] for p in phases_array])

        ax6.plot(phase_numeric[:500], "r-", linewidth=1, alpha=0.7)
        ax6.set_xlabel("Timestep")
        ax6.set_ylabel("Phase Code")
        ax6.set_title("Game Phase Transitions (first 500 steps)")
        ax6.set_yticks(list(phase_codes.values()))
        ax6.set_yticklabels(list(phase_codes.keys()))
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.config.save_plots:
            plt.savefig(
                f"{self.config.plot_dir}/breakout_analysis.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

        # Additional specialized plots
        self._plot_game_phase_details(results)
        self._plot_spatial_strata(results)

        if "topological" in results and results["topological"] is not None:
            self._plot_persistence(results["topological"])

    def _plot_game_phase_details(self, results: Dict):
        """Detailed game phase visualization"""
        if "phase_stats" not in results:
            return

        phase_stats = results["phase_stats"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Action distributions
        ax = axes[0]
        phases = list(phase_stats.keys())
        for i, phase in enumerate(phases):
            action_dist = phase_stats[phase]["action_distribution"]
            ax.bar(
                np.arange(len(action_dist)) + i * 0.25,
                action_dist,
                width=0.25,
                label=phase,
                alpha=0.7,
            )

        ax.set_xlabel("Action")
        ax.set_ylabel("Probability")
        ax.set_title("Action Distribution by Phase")
        ax.legend()
        ax.set_xticks(np.arange(4))

        # 2. Reward by phase
        ax = axes[1]
        rewards = [phase_stats[p]["mean_reward"] for p in phases]
        ax.bar(phases, rewards, color="green", alpha=0.7)
        ax.set_xlabel("Game Phase")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Reward by Phase")

        # 3. Brick count by phase
        ax = axes[2]
        bricks = [phase_stats[p]["mean_bricks"] for p in phases]
        ax.bar(phases, bricks, color="blue", alpha=0.7)
        ax.set_xlabel("Game Phase")
        ax.set_ylabel("Mean Brick Count")
        ax.set_title("Bricks Remaining by Phase")

        plt.tight_layout()

        if self.config.save_plots:
            plt.savefig(
                f"{self.config.plot_dir}/game_phase_details.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

    def _plot_spatial_strata(self, results: Dict):
        """Visualize spatial strata"""
        if "spatial" not in results:
            return

        spatial = results["spatial"]
        if "ball_positions" not in spatial:
            return

        ball_positions = spatial["ball_positions"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Ball positions with clusters
        ax = axes[0]
        if "ball_clusters" in spatial:
            labels = spatial["ball_clusters"]
            unique_labels = np.unique(labels)

            for label in unique_labels:
                mask = labels == label
                if label == -1:
                    ax.scatter(
                        ball_positions[mask, 0],
                        ball_positions[mask, 1],
                        c="gray",
                        alpha=0.3,
                        s=5,
                        label="Noise",
                    )
                else:
                    ax.scatter(
                        ball_positions[mask, 0],
                        ball_positions[mask, 1],
                        alpha=0.6,
                        s=5,
                        label=f"Cluster {label}",
                    )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Ball Position Clusters")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # 2. Spatial strata regions
        ax = axes[1]

        # Draw strata boundaries
        if "spatial_strata" in spatial:
            strata = spatial["spatial_strata"]

            # Draw all points as background
            ax.scatter(
                ball_positions[:, 0],
                ball_positions[:, 1],
                c="lightblue",
                alpha=0.3,
                s=3,
            )

            # Highlight different regions
            if "horizontal" in strata:
                x_thresholds = np.percentile(ball_positions[:, 0], [25, 75])
                ax.axvline(x=x_thresholds[0], color="red", linestyle="--", alpha=0.5)
                ax.axvline(x=x_thresholds[1], color="red", linestyle="--", alpha=0.5)
                ax.text(0.1, 0.9, "Left Zone", transform=ax.transAxes)
                ax.text(0.4, 0.9, "Center Zone", transform=ax.transAxes)
                ax.text(0.75, 0.9, "Right Zone", transform=ax.transAxes)

            if "vertical" in strata:
                y_thresholds = np.percentile(ball_positions[:, 1], [33, 66])
                ax.axhline(y=y_thresholds[0], color="blue", linestyle="--", alpha=0.5)
                ax.axhline(y=y_thresholds[1], color="blue", linestyle="--", alpha=0.5)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Spatial Strata")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if self.config.save_plots:
            plt.savefig(
                f"{self.config.plot_dir}/spatial_strata.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

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
                f"{self.config.plot_dir}/breakout_persistence.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

    def run_analysis(self) -> Dict:
        """Run complete Breakout stratification analysis"""
        print("\n" + "=" * 70)
        print("BREAKOUT STRATIFICATION ANALYSIS")
        print("=" * 70)

        # Collect data
        self.collect_data()

        results = {}

        # Run analyses
        if self.config.analyze_spatial:
            results["spatial"] = self.analyze_spatial_structure()

        if self.config.analyze_game_phase:
            results["phase_stats"] = self.analyze_game_phase_structure()

        if self.config.analyze_cnn_features and len(self.data["cnn_features"]) > 0:
            results["cnn"] = self.analyze_cnn_feature_space()

        if self.config.analyze_temporal:
            results["temporal"] = self.analyze_temporal_patterns()

        # Topological analysis (optional)
        results["topological"] = self.topological_analysis()

        # Visualize
        self.visualize_all(results)

        # Summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """Print stratification summary"""
        print("\n" + "=" * 70)
        print("STRATIFICATION SUMMARY")
        print("=" * 70)

        evidence = []

        # 1. Spatial evidence
        if "spatial" in results and results["spatial"]:
            if "ball_clusters" in results["spatial"]:
                labels = results["spatial"]["ball_clusters"]
                n_spatial = len(set(labels)) - (1 if -1 in labels else 0)
                if n_spatial >= 2:
                    evidence.append(
                        f"✓ Spatial: {n_spatial} distinct ball position clusters"
                    )

        # 2. Game phase evidence
        if "phase_stats" in results and results["phase_stats"]:
            n_phases = len(results["phase_stats"])
            if n_phases >= 3:
                evidence.append(
                    f"✓ Game phases: {n_phases} distinct phases with different behaviors"
                )

        # 3. CNN feature evidence
        if "cnn" in results and results["cnn"]:
            if "feature_labels" in results["cnn"]:
                labels = results["cnn"]["feature_labels"]
                n_feature = len(set(labels)) - (1 if -1 in labels else 0)
                if n_feature >= 2:
                    evidence.append(
                        f"✓ CNN features: {n_feature} distinct representation clusters"
                    )

        # 4. Temporal evidence
        if "temporal" in results and results["temporal"]:
            if "reward_intervals" in results["temporal"]:
                evidence.append("✓ Temporal: structured reward patterns detected")

        # 5. Topological evidence
        if "topological" in results and results["topological"]:
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

        if len(evidence) >= 4:
            print("Breakout exhibits CLEAR HIERARCHICAL STRATIFICATION!")
            print("\nThe environment has multiple interacting levels:")
            print("  • Spatial strata: ball positions cluster in distinct regions")
            print("  • Temporal strata: game progresses through clear phases")
            print("  • Behavioral strata: action patterns change with game state")
            print(
                "  • Representational strata: CNN learns distinct features for each phase"
            )

        elif len(evidence) >= 2:
            print("Breakout shows MODERATE STRATIFICATION.")
            print(
                "\nSome hierarchical structure is present, but boundaries may be fuzzy."
            )

        else:
            print("Limited evidence for strong stratification in this analysis.")
            print("The representation may be more continuous than discrete.")

        print("=" * 70)


def analyze_breakout(
    model_path: str,
    env_name: str = "BreakoutNoFrameskip-v4",
    model_type: str = "DQN",
    n_episodes: int = 50,
    **kwargs,
):
    """
    Convenience function to analyze Breakout policy

    Args:
        model_path: Path to saved model
        env_name: Gym environment name
        model_type: Type of model (DQN, PPO, etc.)
        n_episodes: Number of episodes to collect
        **kwargs: Additional config parameters
    """
    # Create environment with proper wrappers for Atari
    env = gym.make(env_name, render_mode="rgb_array")

    # Load model
    if model_type == "DQN":
        from stable_baselines3 import DQN

        model = DQN.load(model_path)
    elif model_type == "PPO":
        from stable_baselines3 import PPO

        model = PPO.load(model_path)
    elif model_type == "A2C":
        from stable_baselines3 import A2C

        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create config
    config = BreakoutConfig(n_episodes=n_episodes, **kwargs)

    # Run analysis
    analyzer = BreakoutStratificationAnalyzer(env, model, config)
    results = analyzer.run_analysis()

    return results, analyzer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Breakout policy stratification"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--env", type=str, default="Breakout-v5", help="Environment name"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="DQN",
        choices=["DQN", "PPO", "A2C"],
        help="Type of model",
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--no_spatial", action="store_true", help="Skip spatial analysis"
    )
    parser.add_argument(
        "--no_phase", action="store_true", help="Skip game phase analysis"
    )

    args = parser.parse_args()

    # Run analysis
    results, analyzer = analyze_breakout(
        model_path=args.model_path,
        env_name=args.env,
        model_type=args.model_type,
        n_episodes=args.episodes,
        analyze_spatial=not args.no_spatial,
        analyze_game_phase=not args.no_phase,
    )
