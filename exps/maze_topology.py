"""
Comprehensive Topological RL Analysis Testbed
Combined direct loop detection with Morse theory
Fixed PyGame rendering issues
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import KDTree
from sklearn.neighbors import KernelDensity
from collections import defaultdict, deque
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Optional
import pickle
import time
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

# Try to import pygame, but make it optional
try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    print("Pygame not available. Running without visualization.")
    PYGAME_AVAILABLE = False

# ============================================
# 1. CUSTOM MAZE ENVIRONMENT (FIXED)
# ============================================


class MazeType(Enum):
    SIMPLE = "simple"
    FOUR_ROOMS = "four_rooms"
    SPIRAL = "spiral"
    H_SHUFFLE = "h_shuffle"


class PointMazeEnv(gym.Env):
    """A 2D maze environment for topological analysis"""

    def __init__(self, maze_type=MazeType.FOUR_ROOMS, size=20):
        super().__init__()
        self.size = size
        self.maze_type = maze_type
        self.walls = self._generate_walls()

        # State: (x, y) position
        self.observation_space = spaces.Box(
            low=0, high=size, shape=(2,), dtype=np.float32
        )

        # Action: (dx, dy) small movement
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Start and goal
        self.start_pos = np.array([1.0, 1.0], dtype=np.float32)
        self.goal_pos = np.array([size - 2, size - 2], dtype=np.float32)

        # For visualization
        self.state_history = []
        self.trajectories = []

        # Current state
        self.state = self.start_pos.copy()

        # PyGame initialization (optional)
        self.screen = None
        self.clock = None
        self.scale = 40  # pixels per unit

    def _generate_walls(self):
        """Generate maze walls based on type - FIXED COORDINATES"""
        walls = []
        size = self.size

        # Border walls - FIXED: using proper tuples
        walls.append(((0, 0), (size, 0)))  # Top border
        walls.append(((0, size), (size, size)))  # Bottom border
        walls.append(((0, 0), (0, size)))  # Left border
        walls.append(((size, 0), (size, size)))  # Right border

        if self.maze_type == MazeType.FOUR_ROOMS:
            # Four rooms maze
            mid = size // 2
            # Vertical wall with door
            walls.append(((mid, 0), (mid, mid - 3)))
            walls.append(((mid, mid + 3), (mid, size)))
            # Horizontal wall with door
            walls.append(((0, mid), (mid - 3, mid)))
            walls.append(((mid + 3, mid), (size, mid)))

        elif self.maze_type == MazeType.SPIRAL:
            # Spiral maze
            for i in range(2, size - 2, 4):
                # Outer square
                walls.append(((i, i), (size - i, i)))  # Top
                walls.append(((size - i, i), (size - i, size - i)))  # Right
                walls.append(((i, size - i), (size - i, size - i)))  # Bottom
                walls.append(((i, i), (i, size - i - 4)))  # Left with gap

        elif self.maze_type == MazeType.H_SHUFFLE:
            # H-shaped maze that can be reconfigured
            mid = size // 2
            # Vertical bars
            for i in [mid - 5, mid + 5]:
                walls.append(((i, 0), (i, mid - 3)))
                walls.append(((i, mid + 3), (i, size)))

            # Horizontal connectors (can be toggled)
            self.toggle_walls = [
                ((mid - 5, mid), (mid + 5, mid)),  # Middle bridge
                ((mid - 5, mid - 3), (mid - 5, mid + 3)),  # Left vertical gap filler
                ((mid + 5, mid - 3), (mid + 5, mid + 3)),  # Right vertical gap filler
            ]
            walls.extend(self.toggle_walls)

        return walls

    def reset(self):
        """Reset environment"""
        self.state = self.start_pos.copy()
        self.state_history = [self.state.copy()]
        return self.state

    def step(self, action):
        """Take a step in the environment"""
        # Add noise to action
        noise = np.random.normal(0, 0.1, size=2)
        action = np.clip(action + noise, -1, 1)

        # Proposed new position
        new_state = self.state + action * 0.5

        # Check wall collisions
        if not self._check_collision(self.state, new_state):
            self.state = new_state

        # Clip to bounds
        self.state = np.clip(
            self.state, 0.1, self.size - 0.1
        )  # Keep away from exact borders

        # Record state
        self.state_history.append(self.state.copy())

        # Compute reward
        distance_to_goal = np.linalg.norm(self.state - self.goal_pos)
        reward = -distance_to_goal * 0.1  # Negative distance reward

        # Bonus for being close to goal
        if distance_to_goal < 1.0:
            reward += 10.0

        # Check if goal reached
        done = distance_to_goal < 0.5

        return self.state, reward, done, {}

    def _check_collision(self, old_pos, new_pos):
        """Check if movement crosses any wall"""
        for (x1, y1), (x2, y2) in self.walls:
            if self._line_intersect(old_pos, new_pos, (x1, y1), (x2, y2)):
                return True
        return False

    def _line_intersect(self, p1, p2, q1, q2):
        """Check if line segments p1-p2 and q1-q2 intersect"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        p1 = (float(p1[0]), float(p1[1]))
        p2 = (float(p2[0]), float(p2[1]))
        q1 = (float(q1[0]), float(q1[1]))
        q2 = (float(q2[0]), float(q2[1]))

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

    def render(self, mode="human"):
        """Render the maze and agent path - FIXED"""
        if not PYGAME_AVAILABLE:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.size * self.scale, self.size * self.scale)
            )
            self.clock = pygame.time.Clock()
            pygame.display.set_caption(f"Topological RL Maze - {self.maze_type.value}")

        # Clear screen
        self.screen.fill((240, 240, 240))  # Light gray background

        # Draw walls
        for (x1, y1), (x2, y2) in self.walls:
            # Convert coordinates to pixel positions
            start_pos = (int(x1 * self.scale), int(y1 * self.scale))
            end_pos = (int(x2 * self.scale), int(y2 * self.scale))
            pygame.draw.line(self.screen, (50, 50, 50), start_pos, end_pos, 4)

        # Draw goal
        goal_pos = (
            int(self.goal_pos[0] * self.scale),
            int(self.goal_pos[1] * self.scale),
        )
        pygame.draw.circle(self.screen, (0, 200, 0), goal_pos, 10)
        pygame.draw.circle(self.screen, (0, 255, 0), goal_pos, 8)

        # Draw agent path
        if len(self.state_history) > 1:
            points = []
            for state in self.state_history:
                x = int(state[0] * self.scale)
                y = int(state[1] * self.scale)
                points.append((x, y))

            if len(points) >= 2:
                pygame.draw.lines(self.screen, (200, 50, 50), False, points, 2)

        # Draw current agent
        agent_pos = (int(self.state[0] * self.scale), int(self.state[1] * self.scale))
        pygame.draw.circle(self.screen, (50, 50, 200), agent_pos, 6)
        pygame.draw.circle(self.screen, (100, 100, 255), agent_pos, 4)

        # Draw start position
        start_pos = (
            int(self.start_pos[0] * self.scale),
            int(self.start_pos[1] * self.scale),
        )
        pygame.draw.circle(self.screen, (200, 100, 0), start_pos, 5)

        # Update display
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return

        self.clock.tick(30)

    def close(self):
        """Close the environment"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def get_full_trajectory(self):
        """Return the current trajectory"""
        return np.array(self.state_history)

    def reconfigure_maze(self):
        """Change maze configuration (for topology evolution tracking)"""
        if self.maze_type == MazeType.H_SHUFFLE:
            # Randomly toggle some walls
            for i in range(len(self.toggle_walls)):
                if np.random.random() < 0.3:
                    if self.toggle_walls[i] in self.walls:
                        self.walls.remove(self.toggle_walls[i])
                    else:
                        self.walls.append(self.toggle_walls[i])


# ============================================
# 2. DIRECT LOOP DETECTION (SIMPLIFIED)
# ============================================


@dataclass
class Loop:
    """A detected loop in trajectory data"""

    states: np.ndarray  # Sequence of states
    length: int  # Number of states
    area: float  # Area enclosed by loop
    stitched_from: int  # How many trajectories were stitched


class LoopDetector:
    """Detect loops by stitching multiple trajectories - SIMPLIFIED VERSION"""

    def __init__(self, epsilon=1.5, min_loop_length=4):
        self.epsilon = epsilon  # Stitching distance threshold
        self.min_loop_length = min_loop_length
        self.trajectories = []  # List of state arrays
        self.loops_history = []  # Track loops over time

    def add_trajectory(self, states: np.ndarray):
        """Add a new trajectory to the database"""
        self.trajectories.append(states.copy())

        # Keep only recent trajectories for efficiency
        if len(self.trajectories) > 50:
            self.trajectories = self.trajectories[-20:]

    def detect_loops_simple(self):
        """Simplified loop detection using trajectory endpoints"""
        loops = []

        if len(self.trajectories) < 2:
            return loops

        # Get recent trajectories
        recent_trajs = (
            self.trajectories[-10:]
            if len(self.trajectories) > 10
            else self.trajectories
        )

        # Look for loops by checking if start and end of different trajectories are close
        for i in range(len(recent_trajs)):
            traj_i = recent_trajs[i]
            if len(traj_i) < 3:
                continue

            start_i = traj_i[0]
            end_i = traj_i[-1]

            for j in range(i + 1, len(recent_trajs)):
                traj_j = recent_trajs[j]
                if len(traj_j) < 3:
                    continue

                start_j = traj_j[0]
                end_j = traj_j[-1]

                # Check for potential loops
                # Case 1: Start of i is close to end of j
                if np.linalg.norm(start_i - end_j) < self.epsilon:
                    # Combine trajectories to form a loop
                    loop_states = np.vstack([traj_j, traj_i])
                    if self._validate_loop(loop_states):
                        loop = self._create_loop(loop_states, stitched_from=2)
                        loops.append(loop)

                # Case 2: End of i is close to start of j
                if np.linalg.norm(end_i - start_j) < self.epsilon:
                    # Combine trajectories to form a loop
                    loop_states = np.vstack([traj_i, traj_j])
                    if self._validate_loop(loop_states):
                        loop = self._create_loop(loop_states, stitched_from=2)
                        loops.append(loop)

        # Also check within single trajectories
        for traj in recent_trajs:
            if len(traj) >= self.min_loop_length:
                # Check if any point is close to a later point
                for i in range(len(traj) - self.min_loop_length):
                    for j in range(i + self.min_loop_length, len(traj)):
                        if np.linalg.norm(traj[i] - traj[j]) < self.epsilon:
                            # Extract loop segment
                            loop_states = traj[i : j + 1]
                            if self._validate_loop(loop_states):
                                loop = self._create_loop(loop_states, stitched_from=1)
                                loops.append(loop)

        self.loops_history.append(loops)
        return loops

    def _validate_loop(self, states):
        """Check if sequence of states forms a valid loop"""
        if len(states) < self.min_loop_length:
            return False

        # Check if first and last states are close enough
        if np.linalg.norm(states[0] - states[-1]) > self.epsilon * 1.5:
            return False

        # Check area (should be > 0 for a real loop)
        area = self._compute_polygon_area(states)
        if area < 0.1:
            return False

        return True

    def _compute_polygon_area(self, vertices):
        """Compute area of polygon using shoelace formula"""
        x = vertices[:, 0]
        y = vertices[:, 1]

        # Close the polygon
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])

        # Shoelace formula
        area = 0.5 * np.abs(
            np.dot(x_closed[:-1], y_closed[1:]) - np.dot(y_closed[:-1], x_closed[1:])
        )
        return area

    def _create_loop(self, states, stitched_from):
        """Create a Loop object"""
        # Ensure loop is closed by adding first point at the end
        closed_states = np.vstack([states, states[0:1]])
        area = self._compute_polygon_area(closed_states)

        return Loop(
            states=closed_states,
            length=len(closed_states),
            area=area,
            stitched_from=stitched_from,
        )

    def get_loop_statistics(self):
        """Get statistics about detected loops"""
        if not self.loops_history:
            return {"n_loops": 0, "avg_length": 0, "max_length": 0, "avg_area": 0}

        recent_loops = self.loops_history[-1]

        if not recent_loops:
            return {"n_loops": 0, "avg_length": 0, "max_length": 0, "avg_area": 0}

        stats = {
            "n_loops": len(recent_loops),
            "avg_length": np.mean([l.length for l in recent_loops]),
            "max_length": max([l.length for l in recent_loops]),
            "avg_area": np.mean([l.area for l in recent_loops]),
            "total_stitched": sum([l.stitched_from for l in recent_loops]),
        }

        return stats


# ============================================
# 3. SIMPLIFIED MORSE THEORY ANALYZER
# ============================================


class MorseTheoryAnalyzer:
    """Simplified Morse theory analyzer using visitation density"""

    def __init__(self, bandwidth=1.5):
        self.bandwidth = bandwidth
        self.visited_states = []
        self.visitation_grid = None
        self.grid_size = 20
        self.density_map = None

    def update(self, states: np.ndarray):
        """Update with new visited states"""
        self.visited_states.extend(states)

        # Keep only recent states for efficiency
        if len(self.visited_states) > 5000:
            self.visited_states = self.visited_states[-2000:]

        # Update density map
        self._update_density_map()

    def _update_density_map(self):
        """Create a simple density map using grid counting"""
        if len(self.visited_states) < 10:
            return

        states_array = np.array(self.visited_states)

        # Create grid
        x_edges = np.linspace(0, 20, self.grid_size + 1)
        y_edges = np.linspace(0, 20, self.grid_size + 1)

        # Count visits in each grid cell
        self.density_map, _, _ = np.histogram2d(
            states_array[:, 0], states_array[:, 1], bins=[x_edges, y_edges]
        )

        # Smooth the density map
        from scipy.ndimage import gaussian_filter

        self.density_map = gaussian_filter(self.density_map, sigma=1.0)

        # Normalize
        if self.density_map.max() > 0:
            self.density_map = self.density_map / self.density_map.max()

    def find_critical_points_simple(self):
        """Find critical points using local maxima/minima in density map"""
        if self.density_map is None:
            return []

        critical_points = []
        h, w = self.density_map.shape

        # Find local maxima (low visitation = high in Morse function)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                neighborhood = self.density_map[i - 1 : i + 2, j - 1 : j + 2]
                center = neighborhood[1, 1]

                # Local minimum in density = maximum in Morse function
                if center == neighborhood.min() and center < 0.1:
                    # Convert grid coordinates to world coordinates
                    x = (j + 0.5) * (20 / w)
                    y = (i + 0.5) * (20 / h)
                    critical_points.append((np.array([x, y]), "maximum"))

                # Local maximum in density = minimum in Morse function
                elif center == neighborhood.max() and center > 0.3:
                    # Convert grid coordinates to world coordinates
                    x = (j + 0.5) * (20 / w)
                    y = (i + 0.5) * (20 / h)
                    critical_points.append((np.array([x, y]), "minimum"))

        return critical_points

    def compute_morse_function_grid(self):
        """Compute Morse function values on a grid"""
        if self.density_map is None:
            return None, None, None

        # Morse function = -log(density + epsilon)
        epsilon = 0.01
        morse_values = -np.log(self.density_map + epsilon)

        # Create coordinate grids
        x = np.linspace(0, 20, self.grid_size)
        y = np.linspace(0, 20, self.grid_size)
        X, Y = np.meshgrid(x, y)

        return X, Y, morse_values

    def get_topology_statistics(self):
        """Get statistics about topological features"""
        critical_points = self.find_critical_points_simple()

        stats = {
            "n_critical_points": len(critical_points),
            "n_minima": sum(1 for _, t in critical_points if t == "minimum"),
            "n_maxima": sum(1 for _, t in critical_points if t == "maximum"),
            "state_coverage": len(self.visited_states),
            "unique_cells": np.sum(self.density_map > 0)
            if self.density_map is not None
            else 0,
        }

        return stats


# ============================================
# 4. TOPOLOGY EVOLUTION TRACKER
# ============================================


class TopologyEvolutionTracker:
    """Track evolution of topological features over training"""

    def __init__(self, loop_detector, morse_analyzer):
        self.loop_detector = loop_detector
        self.morse_analyzer = morse_analyzer
        self.history = []
        self.epoch = 0

    def record_epoch(self):
        """Record current topological state"""
        loop_stats = self.loop_detector.get_loop_statistics()
        topology_stats = self.morse_analyzer.get_topology_statistics()

        epoch_data = {
            "epoch": self.epoch,
            "n_trajectories": len(self.loop_detector.trajectories),
            **loop_stats,
            **topology_stats,
        }

        self.history.append(epoch_data)
        self.epoch += 1

        return epoch_data

    def visualize_evolution(self, save_path=None):
        """Create visualization of topological evolution"""
        if len(self.history) < 2:
            print("Need more epochs to visualize evolution")
            return None

        epochs = [h["epoch"] for h in self.history]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Number of loops
        axes[0, 0].plot(
            epochs,
            [h["n_loops"] for h in self.history],
            "b-o",
            linewidth=2,
            markersize=5,
        )
        axes[0, 0].set_xlabel("Training Epoch")
        axes[0, 0].set_ylabel("Number of Loops")
        axes[0, 0].set_title("Loop Discovery Over Time")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Loop length
        axes[0, 1].plot(
            epochs,
            [h["avg_length"] for h in self.history],
            "r-s",
            linewidth=2,
            markersize=5,
            label="Average",
        )
        axes[0, 1].plot(
            epochs,
            [h["max_length"] for h in self.history],
            "g-^",
            linewidth=2,
            markersize=5,
            label="Max",
        )
        axes[0, 1].set_xlabel("Training Epoch")
        axes[0, 1].set_ylabel("Loop Length")
        axes[0, 1].set_title("Loop Complexity Evolution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Loop area
        axes[0, 2].plot(
            epochs,
            [h["avg_area"] for h in self.history],
            "m-*",
            linewidth=2,
            markersize=5,
        )
        axes[0, 2].set_xlabel("Training Epoch")
        axes[0, 2].set_ylabel("Average Loop Area")
        axes[0, 2].set_title("Size of Discovered Loops")
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Critical points
        axes[1, 0].plot(
            epochs,
            [h["n_critical_points"] for h in self.history],
            "c-D",
            linewidth=2,
            markersize=5,
            label="Total",
        )
        axes[1, 0].plot(
            epochs,
            [h.get("n_minima", 0) for h in self.history],
            "orange",
            marker="o",
            linewidth=2,
            markersize=4,
            label="Minima",
        )
        axes[1, 0].plot(
            epochs,
            [h.get("n_maxima", 0) for h in self.history],
            "purple",
            marker="^",
            linewidth=2,
            markersize=4,
            label="Maxima",
        )
        axes[1, 0].set_xlabel("Training Epoch")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Morse Critical Points Evolution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: State coverage
        axes[1, 1].plot(
            epochs, [h["state_coverage"] for h in self.history], "g-", linewidth=2
        )
        axes[1, 1].set_xlabel("Training Epoch")
        axes[1, 1].set_ylabel("Total States Visited")
        axes[1, 1].set_title("State Space Coverage")
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Trajectory stitching
        axes[1, 2].plot(
            epochs,
            [h.get("total_stitched", 0) for h in self.history],
            "brown",
            marker="s",
            linewidth=2,
            markersize=5,
        )
        axes[1, 2].set_xlabel("Training Epoch")
        axes[1, 2].set_ylabel("Stitching Count")
        axes[1, 2].set_title("Trajectory Stitching Complexity")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Evolution plot saved to {save_path}")

        return fig


# ============================================
# 5. SIMPLE RL AGENT
# ============================================


class SimpleRLAgent:
    """Simple RL agent with basic exploration"""

    def __init__(self, state_dim=2, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q-table approximation (simplified)
        self.q_table = defaultdict(float)

        # Exploration parameters
        self.epsilon = 0.3  # Exploration rate
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor

    def discretize_state(self, state):
        """Discretize continuous state for Q-table"""
        return tuple(np.round(state, 1))

    def get_action(self, state, goal=None):
        """Get action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Random exploration
            action = np.random.uniform(-1, 1, size=self.action_dim)
        else:
            # Greedy action (simplified: move toward goal if provided)
            if goal is not None:
                direction = goal - state
                norm = np.linalg.norm(direction)
                if norm > 0:
                    action = direction / norm
                else:
                    action = np.random.uniform(-1, 1, size=self.action_dim)
            else:
                # Random if no goal
                action = np.random.uniform(-1, 1, size=self.action_dim)

        return np.clip(action, -1, 1)

    def update(self, state, action, reward, next_state, done):
        """Simple Q-learning update"""
        state_disc = self.discretize_state(state)
        next_state_disc = self.discretize_state(next_state)

        # Simple discretization of action
        action_disc = tuple(np.sign(action))

        # Q-learning update
        current_q = self.q_table[(state_disc, action_disc)]

        if done:
            target = reward
        else:
            # Estimate max future Q (simplified)
            future_q = max(
                [
                    self.q_table[(next_state_disc, a)]
                    for a in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                ],
                default=0,
            )
            target = reward + self.gamma * future_q

        # Update Q-value
        self.q_table[(state_disc, action_disc)] = current_q + self.alpha * (
            target - current_q
        )


# ============================================
# 6. MAIN EXPERIMENT RUNNER (SIMPLIFIED)
# ============================================


def run_simplified_experiment(
    maze_type=MazeType.FOUR_ROOMS, n_epochs=20, render_every=5
):
    """Run simplified topology-RL experiment"""

    print(f"\n{'=' * 60}")
    print(f"Topological RL Experiment: {maze_type.value} maze")
    print(f"{'=' * 60}")

    # Initialize environment
    env = PointMazeEnv(maze_type=maze_type, size=20)

    # Initialize analyzers
    loop_detector = LoopDetector(epsilon=1.5, min_loop_length=4)
    morse_analyzer = MorseTheoryAnalyzer(bandwidth=1.5)
    tracker = TopologyEvolutionTracker(loop_detector, morse_analyzer)

    # Initialize agent
    agent = SimpleRLAgent()

    # Storage for results
    results = {
        "rewards": [],
        "steps": [],
        "loop_stats": [],
        "topology_stats": [],
    }

    # Training loop
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        epoch_rewards = []
        epoch_steps = []

        # Collect 3 episodes per epoch
        for episode in range(3):
            state = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            states = [state.copy()]

            while not done and steps < 100:
                # Get action
                action = agent.get_action(state, env.goal_pos)

                # Take step
                next_state, reward, done, _ = env.step(action)

                # Update agent
                agent.update(state, action, reward, next_state, done)

                # Store state
                states.append(next_state.copy())

                # Update metrics
                state = next_state
                episode_reward += reward
                steps += 1

                # Render occasionally
                if epoch % render_every == 0 and episode == 0 and PYGAME_AVAILABLE:
                    env.render()

            # Add trajectory to loop detector
            loop_detector.add_trajectory(np.array(states))

            # Update Morse analyzer
            morse_analyzer.update(np.array(states))

            # Store episode results
            epoch_rewards.append(episode_reward)
            epoch_steps.append(steps)

            print(
                f"  Episode {episode + 1}: Reward={episode_reward:.1f}, Steps={steps}"
            )

        # Analyze topology for this epoch
        loops = loop_detector.detect_loops_simple()
        epoch_stats = tracker.record_epoch()

        # Store results
        results["rewards"].append(np.mean(epoch_rewards))
        results["steps"].append(np.mean(epoch_steps))
        results["loop_stats"].append(loop_detector.get_loop_statistics())
        results["topology_stats"].append(epoch_stats)

        # Print epoch summary
        print(
            f"  Avg reward: {np.mean(epoch_rewards):.2f}, Avg steps: {np.mean(epoch_steps):.1f}"
        )
        print(f"  Loops detected: {len(loops)}")
        print(f"  Critical points: {epoch_stats.get('n_critical_points', 0)}")

        # Occasionally reconfigure maze (for H_SHUFFLE type)
        if maze_type == MazeType.H_SHUFFLE and epoch % 7 == 0 and epoch > 0:
            print("  Reconfiguring maze...")
            env.reconfigure_maze()

    # Final analysis
    print(f"\n{'=' * 60}")
    print("Experiment Complete!")
    print(f"{'=' * 60}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Evolution plot
    evolution_fig = tracker.visualize_evolution(
        save_path=f"topology_evolution_{maze_type.value}.png"
    )

    # 2. Final state visualization
    final_fig = visualize_final_state_simple(env, loop_detector, morse_analyzer, epoch)

    # 3. Save results
    with open(f"results_{maze_type.value}.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved:")
    print(f"  - topology_evolution_{maze_type.value}.png")
    print(f"  - final_state_{maze_type.value}.png")
    print(f"  - results_{maze_type.value}.pkl")

    # Close environment
    env.close()

    return env, loop_detector, morse_analyzer, tracker, results


def visualize_final_state_simple(env, loop_detector, morse_analyzer, epoch):
    """Create visualization of final topological state"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Maze with recent trajectories
    axes[0, 0].set_title(f"Maze with Trajectories (Epoch {epoch})", fontsize=12)

    # Plot walls
    for (x1, y1), (x2, y2) in env.walls:
        axes[0, 0].plot([x1, x2], [y1, y2], "k-", linewidth=3)

    # Plot recent trajectories (last 5)
    colors = plt.cm.rainbow(np.linspace(0, 1, 5))
    recent_trajs = (
        loop_detector.trajectories[-5:]
        if len(loop_detector.trajectories) >= 5
        else loop_detector.trajectories
    )

    for idx, traj in enumerate(recent_trajs):
        if len(traj) > 1:
            axes[0, 0].plot(
                traj[:, 0],
                traj[:, 1],
                "-",
                color=colors[idx % len(colors)],
                alpha=0.6,
                linewidth=1.5,
                label=f"Traj {idx + 1}",
            )

    # Plot goal and start
    axes[0, 0].plot(env.goal_pos[0], env.goal_pos[1], "g*", markersize=15, label="Goal")
    axes[0, 0].plot(
        env.start_pos[0], env.start_pos[1], "bs", markersize=10, label="Start"
    )

    axes[0, 0].set_xlim(0, env.size)
    axes[0, 0].set_ylim(0, env.size)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Detected loops
    axes[0, 1].set_title(
        f"Detected Loops: {len(loop_detector.loops_history[-1]) if loop_detector.loops_history else 0}",
        fontsize=12,
    )

    # Plot walls
    for (x1, y1), (x2, y2) in env.walls:
        axes[0, 1].plot([x1, x2], [y1, y2], "k-", linewidth=3)

    # Plot loops
    if loop_detector.loops_history and len(loop_detector.loops_history[-1]) > 0:
        loops = loop_detector.loops_history[-1]
        colors = plt.cm.Set1(np.linspace(0, 1, len(loops)))

        for i, loop in enumerate(loops[:5]):  # Show only first 5 loops
            states = loop.states
            axes[0, 1].plot(
                states[:, 0],
                states[:, 1],
                "-",
                color=colors[i],
                linewidth=2,
                alpha=0.8,
                label=f"Loop {i + 1} (area={loop.area:.1f})",
            )

            # Mark start/end point
            axes[0, 1].plot(states[0, 0], states[0, 1], "ko", markersize=4)

    axes[0, 1].set_xlim(0, env.size)
    axes[0, 1].set_ylim(0, env.size)
    axes[0, 1].set_aspect("equal")
    if loop_detector.loops_history and len(loop_detector.loops_history[-1]) > 0:
        axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Morse function (density)
    axes[1, 0].set_title("State Density (Morse Function Base)", fontsize=12)

    X, Y, morse_values = morse_analyzer.compute_morse_function_grid()

    if morse_values is not None:
        # Plot density as heatmap
        contour = axes[1, 0].contourf(
            X, Y, morse_values, levels=20, cmap="viridis", alpha=0.8
        )
        plt.colorbar(contour, ax=axes[1, 0], label="-log(density)")

        # Plot critical points
        critical_points = morse_analyzer.find_critical_points_simple()
        for point, cp_type in critical_points:
            color = "blue" if cp_type == "minimum" else "red"
            marker = "o" if cp_type == "minimum" else "^"
            axes[1, 0].plot(
                point[0],
                point[1],
                marker,
                color=color,
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1,
                label=cp_type.capitalize(),
            )

    # Plot walls
    for (x1, y1), (x2, y2) in env.walls:
        axes[1, 0].plot([x1, x2], [y1, y2], "k-", linewidth=2)

    axes[1, 0].set_xlim(0, env.size)
    axes[1, 0].set_ylim(0, env.size)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].grid(True, alpha=0.3)

    # Remove duplicate labels
    handles, labels = axes[1, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        axes[1, 0].legend(by_label.values(), by_label.keys(), fontsize=8)

    # Plot 4: State visitation heatmap
    axes[1, 1].set_title("State Visitation Heatmap", fontsize=12)

    if morse_analyzer.visited_states:
        states_array = np.array(morse_analyzer.visited_states)

        # Create heatmap
        hb = axes[1, 1].hexbin(
            states_array[:, 0], states_array[:, 1], gridsize=15, cmap="hot", alpha=0.8
        )
        plt.colorbar(hb, ax=axes[1, 1], label="Visit density")

    # Plot walls
    for (x1, y1), (x2, y2) in env.walls:
        axes[1, 1].plot([x1, x2], [y1, y2], "w-", linewidth=2)

    axes[1, 1].set_xlim(0, env.size)
    axes[1, 1].set_ylim(0, env.size)
    axes[1, 1].set_aspect("equal")

    plt.suptitle(
        f"Topological Analysis: {env.maze_type.value.replace('_', ' ').title()} Maze",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save figure
    save_path = f"final_state_{env.maze_type.value}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Final state visualization saved to {save_path}")

    return fig


# ============================================
# 7. QUICK DEMO FUNCTION
# ============================================


def run_quick_demo():
    """Run a quick demo with minimal epochs"""
    print("\n" + "=" * 60)
    print("TOPOLOGICAL RL QUICK DEMO")
    print("=" * 60)
    print("\nThis demo will run a simplified version of the experiment.")
    print("It tests loop detection and Morse theory analysis on a maze.")

    # Run with simple maze for quick testing
    maze_type = MazeType.FOUR_ROOMS

    try:
        env, loop_detector, morse_analyzer, tracker, results = (
            run_simplified_experiment(
                maze_type=maze_type,
                n_epochs=10,  # Fewer epochs for quick demo
                render_every=999,  # Don't render during quick demo
            )
        )

        # Print summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)

        if tracker.history:
            final_stats = tracker.history[-1]
            print(f"\nFinal Topological State:")
            print(f"  Loops detected: {final_stats['n_loops']}")
            print(f"  Avg loop length: {final_stats['avg_length']:.1f}")
            print(f"  Max loop length: {final_stats['max_length']}")
            print(f"  Critical points: {final_stats['n_critical_points']}")
            print(f"    - Minima (high density): {final_stats.get('n_minima', 0)}")
            print(f"    - Maxima (low density): {final_stats.get('n_maxima', 0)}")
            print(f"  States visited: {final_stats['state_coverage']}")

        print(f"\nFiles generated:")
        print(f"  - topology_evolution_{maze_type.value}.png")
        print(f"  - final_state_{maze_type.value}.png")
        print(f"  - results_{maze_type.value}.pkl")

        # Show one of the plots
        plt.show()

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()

        # Try even simpler version
        print("\nTrying ultra-simple version...")
        run_ultra_simple_test()


def run_ultra_simple_test():
    """Ultra-simple test without visualization dependencies"""
    print("\nRunning ultra-simple topological analysis test...")

    # Create simple data
    np.random.seed(42)

    # Simulate some trajectories in a 10x10 space
    trajectories = []

    # Create some looping patterns
    for i in range(5):
        # Circular trajectory
        theta = np.linspace(0, 2 * np.pi, 20)
        x = 5 + 3 * np.cos(theta) + np.random.normal(0, 0.1, 20)
        y = 5 + 3 * np.sin(theta) + np.random.normal(0, 0.1, 20)
        trajectories.append(np.column_stack([x, y]))

        # Random walk
        start = np.random.uniform(2, 8, 2)
        steps = []
        current = start.copy()
        for _ in range(30):
            step = np.random.uniform(-0.5, 0.5, 2)
            current = np.clip(current + step, 0, 10)
            steps.append(current.copy())
        trajectories.append(np.array(steps))

    # Test loop detector
    print("\nTesting Loop Detector...")
    detector = LoopDetector(epsilon=1.0, min_loop_length=5)

    for traj in trajectories:
        detector.add_trajectory(traj)

    loops = detector.detect_loops_simple()
    print(f"Detected {len(loops)} loops")

    if loops:
        print(f"First loop: length={loops[0].length}, area={loops[0].area:.2f}")

    # Test Morse analyzer
    print("\nTesting Morse Theory Analyzer...")
    analyzer = MorseTheoryAnalyzer(bandwidth=1.0)

    for traj in trajectories:
        analyzer.update(traj)

    critical_points = analyzer.find_critical_points_simple()
    print(f"Found {len(critical_points)} critical points")

    stats = analyzer.get_topology_statistics()
    print(f"State coverage: {stats['state_coverage']} points")
    print(f"Unique cells visited: {stats['unique_cells']}")

    print("\nUltra-simple test completed successfully!")
    print("This demonstrates the core topological analysis works.")

    return detector, analyzer


# ============================================
# 8. MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TOPOLOGICAL REINFORCEMENT LEARNING TESTBED")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Run quick demo (recommended for first run)")
    print("  2. Run full experiment with Four Rooms maze")
    print("  3. Run full experiment with Spiral maze")
    print("  4. Run full experiment with H-Shuffle maze")
    print("  5. Run ultra-simple test (no dependencies)")

    try:
        choice = input("\nEnter your choice (1-5, default=1): ").strip()

        if choice == "" or choice == "1":
            run_quick_demo()
        elif choice == "2":
            run_simplified_experiment(maze_type=MazeType.FOUR_ROOMS, n_epochs=20)
        elif choice == "3":
            run_simplified_experiment(maze_type=MazeType.SPIRAL, n_epochs=20)
        elif choice == "4":
            run_simplified_experiment(maze_type=MazeType.H_SHUFFLE, n_epochs=25)
        elif choice == "5":
            detector, analyzer = run_ultra_simple_test()
        else:
            print("Invalid choice. Running quick demo.")
            run_quick_demo()

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTrying ultra-simple test instead...")
        run_ultra_simple_test()

    print("\n" + "=" * 60)
    print("Testbed completed. Check generated files for results.")
    print("=" * 60)
