import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict
import json
import os
from datetime import datetime
import pickle
from pathlib import Path


@dataclass
class TrajectoryData:
    """Container for trajectory data with filtration support"""

    states: np.ndarray  # [timesteps, state_dim]
    actions: np.ndarray  # [timesteps, action_dim] or None
    rewards: np.ndarray  # [timesteps] or None
    dones: np.ndarray  # [timesteps] boolean
    infos: List[Dict]  # Additional environment info
    timesteps: np.ndarray  # Actual timestep indices

    # Filtration-specific fields
    distances: Optional[np.ndarray] = None  # Pairwise distances for persistence
    point_clouds: Optional[List[np.ndarray]] = None  # Point clouds at each timestep
    filtration_values: Optional[np.ndarray] = None  # Filtration values for each point

    def __post_init__(self):
        """Validate trajectory data"""
        if self.states is not None and len(self.states.shape) == 1:
            self.states = self.states.reshape(-1, 1)

    @property
    def n_timesteps(self) -> int:
        return len(self.states)

    @property
    def state_dim(self) -> int:
        return self.states.shape[1] if len(self.states.shape) > 1 else 1

    def compute_pairwise_distances(self, metric: str = "euclidean") -> np.ndarray:
        """Compute pairwise distances between states for persistence calculation"""
        from scipy.spatial.distance import pdist, squareform

        distances = pdist(self.states, metric=metric)
        self.distances = squareform(distances)
        return self.distances

    def create_point_clouds(
        self, window_size: int = 10, stride: int = 1
    ) -> List[np.ndarray]:
        """Create point clouds using sliding window for temporal filtrations"""
        self.point_clouds = []
        for i in range(0, self.n_timesteps - window_size + 1, stride):
            window = self.states[i : i + window_size]
            self.point_clouds.append(window)
        return self.point_clouds

    def compute_delay_embedding(self, delay: int = 1, dimension: int = 3) -> np.ndarray:
        """Compute time-delay embedding for persistent homology of dynamical systems"""
        n = self.n_timesteps - (dimension - 1) * delay
        embedding = np.zeros((n, dimension * self.state_dim))

        for i in range(dimension):
            start = i * delay
            end = start + n
            embedding[:, i * self.state_dim : (i + 1) * self.state_dim] = self.states[
                start:end
            ]

        return embedding


class RLLogger:
    """
    Enhanced logger for RL experiments with support for trajectory collection
    and persistent homology calculations.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        save_trajectories: bool = True,
        max_trajectories_per_buffer: int = 100,
        enable_filtration: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.save_trajectories = save_trajectories
        self.max_trajectories_per_buffer = max_trajectories_per_buffer
        self.enable_filtration = enable_filtration

        # Create experiment directory
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.trajectories: List[TrajectoryData] = []
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.episode_metrics: Dict[str, List[float]] = defaultdict(list)

        # Filtration-specific storage
        self.persistence_diagrams: List[Dict] = []
        self.filtration_metrics: Dict[str, List[float]] = defaultdict(list)

        # Current episode tracking
        self.current_episode = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
            "timesteps": [],
        }
        self.episode_step = 0
        self.total_steps = 0

        # Metadata
        self.metadata = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "config": {},
            "env_info": {},
        }

    def log_step(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        done: bool = False,
        info: Optional[Dict] = None,
    ):
        """Log a single step of interaction"""
        self.current_episode["states"].append(
            state.copy() if state is not None else None
        )
        self.current_episode["actions"].append(
            action.copy() if action is not None else None
        )
        self.current_episode["rewards"].append(reward)
        self.current_episode["dones"].append(done)
        self.current_episode["infos"].append(info or {})
        self.current_episode["timesteps"].append(self.total_steps)

        self.episode_step += 1
        self.total_steps += 1

        if done:
            self._end_episode()

    def _end_episode(self):
        """Finalize and store the current episode"""
        if len(self.current_episode["states"]) > 0:
            # Convert to numpy arrays
            trajectory = TrajectoryData(
                states=np.array(self.current_episode["states"]),
                actions=(
                    np.array(self.current_episode["actions"])
                    if self.current_episode["actions"][0] is not None
                    else None
                ),
                rewards=(
                    np.array(self.current_episode["rewards"])
                    if self.current_episode["rewards"][0] is not None
                    else None
                ),
                dones=np.array(self.current_episode["dones"]),
                infos=self.current_episode["infos"],
                timesteps=np.array(self.current_episode["timesteps"]),
            )

            # Store trajectory
            self.trajectories.append(trajectory)

            # Compute episode metrics
            episode_return = (
                np.sum(trajectory.rewards) if trajectory.rewards is not None else 0
            )
            episode_length = trajectory.n_timesteps

            self.episode_metrics["return"].append(episode_return)
            self.episode_metrics["length"].append(episode_length)

            # Log episode summary
            self.log_metric("episode_return", episode_return)
            self.log_metric("episode_length", episode_length)

            # Optional: compute filtration metrics
            if self.enable_filtration and trajectory.n_timesteps > 1:
                self._compute_filtration_metrics(trajectory)

            # Limit trajectories stored
            if len(self.trajectories) > self.max_trajectories_per_buffer:
                self.trajectories.pop(0)

            # Reset current episode
            self.current_episode = {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "infos": [],
                "timesteps": [],
            }
            self.episode_step = 0

    def _compute_filtration_metrics(self, trajectory: TrajectoryData):
        """Compute metrics based on persistent homology"""
        try:
            import gudhi as gd
            from gudhi.representations import PersistenceImage, Landscape

            # Compute pairwise distances
            distances = trajectory.compute_pairwise_distances()

            # Create Rips complex
            rips_complex = gd.RipsComplex(distance_matrix=distances)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

            # Compute persistence
            persistence = simplex_tree.persistence()

            # Store persistence diagram
            diagram = {
                "episode": len(self.trajectories) - 1,
                "timesteps": trajectory.n_timesteps,
                "diagram": persistence,
                "betti_numbers": simplex_tree.betti_numbers(),
            }
            self.persistence_diagrams.append(diagram)

            # Extract persistence metrics
            if persistence:
                # Average persistence lifetime
                lifetimes = [
                    death - birth
                    for _, (birth, death) in persistence
                    if death != float("inf")
                ]
                if lifetimes:
                    self.filtration_metrics["avg_lifetime"].append(np.mean(lifetimes))

                # Number of persistent features
                n_features = len([p for p in persistence if p[1][1] != float("inf")])
                self.filtration_metrics["n_persistent_features"].append(n_features)

        except ImportError:
            print("GUDHI not installed. Skipping persistence computation.")
        except Exception as e:
            print(f"Error computing persistence: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a scalar metric"""
        self.metrics[key].append(value)
        if step is not None:
            if f"{key}_steps" not in self.metrics:
                self.metrics[f"{key}_steps"] = []
            self.metrics[f"{key}_steps"].append(step)

    def log_filtration_metric(self, key: str, value: float):
        """Log a filtration-specific metric"""
        self.filtration_metrics[key].append(value)

    def get_trajectories_by_criteria(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_return: Optional[float] = None,
    ) -> List[TrajectoryData]:
        """Filter trajectories based on criteria"""
        filtered = self.trajectories

        if min_length is not None:
            filtered = [t for t in filtered if t.n_timesteps >= min_length]
        if max_length is not None:
            filtered = [t for t in filtered if t.n_timesteps <= max_length]
        if min_return is not None and filtered and filtered[0].rewards is not None:
            filtered = [t for t in filtered if np.sum(t.rewards) >= min_return]

        return filtered

    def get_latest_trajectories(self, n: int = 10) -> List[TrajectoryData]:
        """Get the n most recent trajectories"""
        return self.trajectories[-n:]

    def save(self, filename: Optional[str] = None):
        """Save all logged data to disk"""
        if filename is None:
            filename = f"experiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        save_path = self.exp_dir / filename

        # Prepare data for saving
        data = {
            "metadata": self.metadata,
            "metrics": dict(self.metrics),
            "episode_metrics": dict(self.episode_metrics),
            "filtration_metrics": dict(self.filtration_metrics),
            "total_steps": self.total_steps,
            "n_episodes": len(self.trajectories),
        }

        # Save trajectories if enabled
        if self.save_trajectories:
            data["trajectories"] = self.trajectories
            data["persistence_diagrams"] = self.persistence_diagrams

        # Save to file
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        # Also save metrics as CSV for easy viewing
        self._save_metrics_csv()

        print(f"Data saved to {save_path}")
        return save_path

    def _save_metrics_csv(self):
        """Save metrics to CSV files"""
        # Save scalar metrics
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            df.to_csv(self.exp_dir / "metrics.csv", index=False)

        # Save episode metrics
        if self.episode_metrics:
            df_ep = pd.DataFrame(self.episode_metrics)
            df_ep.to_csv(self.exp_dir / "episode_metrics.csv", index=False)

        # Save filtration metrics
        if self.filtration_metrics:
            df_filt = pd.DataFrame(self.filtration_metrics)
            df_filt.to_csv(self.exp_dir / "filtration_metrics.csv", index=False)

    def load(self, filepath: str):
        """Load previously saved experiment data"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.metadata = data.get("metadata", self.metadata)
        self.metrics = defaultdict(list, data.get("metrics", {}))
        self.episode_metrics = defaultdict(list, data.get("episode_metrics", {}))
        self.filtration_metrics = defaultdict(list, data.get("filtration_metrics", {}))
        self.total_steps = data.get("total_steps", 0)

        if "trajectories" in data:
            self.trajectories = data["trajectories"]
        if "persistence_diagrams" in data:
            self.persistence_diagrams = data["persistence_diagrams"]

    def set_config(self, config: Dict):
        """Set experiment configuration"""
        self.metadata["config"] = config

    def set_env_info(self, env_info: Dict):
        """Set environment information"""
        self.metadata["env_info"] = env_info


# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = RLLogger(
        log_dir="./rl_experiments",
        experiment_name="mujoco_persistence_test",
        save_trajectories=True,
        enable_filtration=True,
    )

    # Set metadata
    logger.set_config({"algorithm": "PPO", "learning_rate": 3e-4, "batch_size": 64})

    logger.set_env_info(
        {"env_name": "HalfCheetah-v4", "state_dim": 17, "action_dim": 6}
    )

    # Simulate some trajectories
    for episode in range(5):
        state = np.random.randn(17)  # Simulated state

        for step in range(100):
            action = np.random.randn(6)
            reward = np.random.randn()
            done = step == 99

            logger.log_step(state, action, reward, done, {"step_info": step})

            if done:
                break

    # Save data
    logger.save()

    # Retrieve trajectories for persistence analysis
    trajectories = logger.get_latest_trajectories(n=3)
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i}: {traj.n_timesteps} steps, state_dim={traj.state_dim}")

        # Compute delay embedding for dynamical systems analysis
        if traj.n_timesteps > 10:
            embedding = traj.compute_delay_embedding(delay=2, dimension=3)
            print(f"  Delay embedding shape: {embedding.shape}")

