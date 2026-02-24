import numpy as np
import gymnasium as gym
import pickle
import os
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
import json
from dataclasses import dataclass, asdict
import time


@dataclass
class TrainingMetrics:
    """Store training metrics"""

    episodes: List[int]
    rewards: List[float]
    success_rates: List[float]
    steps_per_episode: List[int]
    epsilon_values: List[float]
    loss_values: List[float] = None


class ConnectedActionCluster:
    """Represents a connected cluster of states with same preferred action"""

    def __init__(self, cluster_id: int, action: int, states: List[int]):
        self.id = cluster_id
        self.action = action
        self.states = sorted(states)
        self.size = len(states)
        self.centroid = None
        self.value_sum = 0.0
        self.avg_value = 0.0
        self.boundary_states = []
        self.incoming_transitions = defaultdict(float)
        self.outgoing_transitions = defaultdict(float)

    def to_dict(self):
        return {
            "id": self.id,
            "action": self.action,
            "states": self.states,
            "size": self.size,
            "centroid": self.centroid,
            "value_sum": self.value_sum,
            "avg_value": self.avg_value,
            "boundary_states": self.boundary_states,
        }


class FrozenLakeSarsa:
    """
    SARSA (State-Action-Reward-State-Action) algorithm for FrozenLake
    On-policy TD control method
    """

    def __init__(
        self,
        env_name: str = "FrozenLake-v1",
        map_name: str = "4x4",
        is_slippery: bool = True,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        episodes: int = 10000,
        render: bool = False,
    ):
        self.env = gym.make(
            env_name,
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode="human" if render else None,
        )
        self.env_name = env_name
        self.map_name = map_name
        self.is_slippery = is_slippery

        # SARSA parameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes

        # Initialize Q-table
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # Tracking metrics
        self.metrics = TrainingMetrics(
            episodes=[],
            rewards=[],
            success_rates=[],
            steps_per_episode=[],
            epsilon_values=[],
        )

        # Grid information
        self.grid_size = 4 if map_name == "4x4" else 8
        self.state_coords = {
            s: (s // self.grid_size, s % self.grid_size) for s in range(self.n_states)
        }
        self.coord_to_state = {
            (r, c): r * self.grid_size + c
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        }

        # Action mappings
        self.action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # left, down, right, up

        self.training_time = 0

    def choose_action(self, state: int) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def train(self, verbose: bool = True) -> Dict:
        """Train using SARSA algorithm"""
        print(f"\n{'=' * 60}")
        print(f"SARSA Training on {self.map_name} FrozenLake")
        print(f"{'=' * 60}")
        print(f"Slippery: {self.is_slippery}")
        print(f"Learning rate: {self.lr}, Discount: {self.gamma}")

        start_time = time.time()
        success_window = deque(maxlen=100)

        for episode in range(self.episodes):
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False
            total_reward = 0
            steps = 0

            while not done:
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Choose next action (for SARSA update)
                if not done:
                    next_action = self.choose_action(next_state)
                else:
                    next_action = 0

                # SARSA update: Q(s,a) = Q(s,a) + lr * (r + gamma * Q(s',a') - Q(s,a))
                current_q = self.q_table[state, action]
                next_q = self.q_table[next_state, next_action] if not done else 0
                td_target = reward + self.gamma * next_q
                td_error = td_target - current_q

                self.q_table[state, action] += self.lr * td_error

                state = next_state
                action = next_action
                total_reward += reward
                steps += 1

            # Track metrics
            success = 1 if total_reward > 0 else 0
            success_window.append(success)
            success_rate = np.mean(success_window)

            self.metrics.episodes.append(episode)
            self.metrics.rewards.append(total_reward)
            self.metrics.success_rates.append(success_rate)
            self.metrics.steps_per_episode.append(steps)
            self.metrics.epsilon_values.append(self.epsilon)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (episode + 1) % 1000 == 0:
                print(
                    f"Episode {episode + 1}/{self.episodes} | "
                    f"Success Rate: {success_rate:.3f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        self.training_time = time.time() - start_time
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        print(f"Final success rate: {success_rate:.3f}")

        return {
            "q_table": self.q_table,
            "final_success_rate": success_rate,
            "training_time": self.training_time,
        }

    def get_stochastic_policy(self, temperature: float = 1.0) -> Dict[int, np.ndarray]:
        """Get stochastic policy using softmax"""
        policy = {}
        for state in range(self.n_states):
            q_values = self.q_table[state]
            q_values = q_values - np.max(q_values)  # for numerical stability
            exp_q = np.exp(q_values / temperature)
            policy[state] = exp_q / np.sum(exp_q)
        return policy

    def get_deterministic_policy(self) -> Dict[int, int]:
        """Get deterministic policy (greedy)"""
        return {s: int(np.argmax(self.q_table[s])) for s in range(self.n_states)}

    def get_state_values(self) -> np.ndarray:
        """Get state values V(s) = max_a Q(s,a)"""
        return np.max(self.q_table, axis=1)

    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            "q_table": self.q_table,
            "metrics": {
                "rewards": self.metrics.rewards,
                "success_rates": self.metrics.success_rates,
                "epsilon_values": self.metrics.epsilon_values,
            },
            "parameters": {
                "algorithm": "SARSA",
                "env_name": self.env_name,
                "map_name": self.map_name,
                "is_slippery": self.is_slippery,
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "epsilon_min": self.epsilon_min,
                "episodes": self.episodes,
                "training_time": self.training_time,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.q_table = model_data["q_table"]
        print(f"Model loaded from {filepath}")
        return model_data


class FrozenLakeMonteCarlo:
    """
    Monte Carlo methods for FrozenLake
    Includes both first-visit and every-visit MC control
    """

    def __init__(
        self,
        env_name: str = "FrozenLake-v1",
        map_name: str = "4x4",
        is_slippery: bool = True,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        episodes: int = 10000,
        first_visit: bool = True,
        render: bool = False,
    ):
        self.env = gym.make(
            env_name,
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode="human" if render else None,
        )
        self.env_name = env_name
        self.map_name = map_name
        self.is_slippery = is_slippery

        # MC parameters
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.first_visit = first_visit

        # Initialize Q-table and returns
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.returns = defaultdict(list)  # Store returns for each (state, action)

        # Tracking metrics
        self.metrics = TrainingMetrics(
            episodes=[],
            rewards=[],
            success_rates=[],
            steps_per_episode=[],
            epsilon_values=[],
        )

        # Grid information (same as SARSA)
        self.grid_size = 4 if map_name == "4x4" else 8
        self.state_coords = {
            s: (s // self.grid_size, s % self.grid_size) for s in range(self.n_states)
        }
        self.coord_to_state = {
            (r, c): r * self.grid_size + c
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        }
        self.action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        self.training_time = 0

    def choose_action(self, state: int) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def generate_episode(self) -> List[Tuple]:
        """Generate one episode following current policy"""
        episode = []
        state, _ = self.env.reset()
        done = False

        while not done:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state

        return episode

    def train(self, verbose: bool = True) -> Dict:
        """Train using Monte Carlo control"""
        print(f"\n{'=' * 60}")
        print(f"Monte Carlo Training on {self.map_name} FrozenLake")
        print(f"{'=' * 60}")
        print(f"Slippery: {self.is_slippery}")
        print(f"First-visit MC: {self.first_visit}")

        start_time = time.time()
        success_window = deque(maxlen=100)

        for episode in range(self.episodes):
            # Generate episode
            episode_data = self.generate_episode()

            # Calculate returns
            G = 0
            visited_state_actions = set()

            # Work backwards through episode
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = self.gamma * G + reward

                # Check for first-visit condition
                sa_pair = (state, action)
                if not self.first_visit or sa_pair not in visited_state_actions:
                    visited_state_actions.add(sa_pair)
                    self.returns[sa_pair].append(G)

                    # Update Q-value with average return
                    self.q_table[state, action] = np.mean(self.returns[sa_pair])

            # Calculate episode statistics
            total_reward = sum(r for _, _, r in episode_data)
            steps = len(episode_data)
            success = 1 if total_reward > 0 else 0
            success_window.append(success)
            success_rate = np.mean(success_window)

            # Track metrics
            self.metrics.episodes.append(episode)
            self.metrics.rewards.append(total_reward)
            self.metrics.success_rates.append(success_rate)
            self.metrics.steps_per_episode.append(steps)
            self.metrics.epsilon_values.append(self.epsilon)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (episode + 1) % 1000 == 0:
                print(
                    f"Episode {episode + 1}/{self.episodes} | "
                    f"Success Rate: {success_rate:.3f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        self.training_time = time.time() - start_time
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        print(f"Final success rate: {success_rate:.3f}")

        return {
            "q_table": self.q_table,
            "final_success_rate": success_rate,
            "training_time": self.training_time,
        }

    def get_stochastic_policy(self, temperature: float = 1.0) -> Dict[int, np.ndarray]:
        """Get stochastic policy using softmax"""
        policy = {}
        for state in range(self.n_states):
            q_values = self.q_table[state]
            q_values = q_values - np.max(q_values)
            exp_q = np.exp(q_values / temperature)
            policy[state] = exp_q / np.sum(exp_q)
        return policy

    def get_deterministic_policy(self) -> Dict[int, int]:
        """Get deterministic policy"""
        return {s: int(np.argmax(self.q_table[s])) for s in range(self.n_states)}

    def get_state_values(self) -> np.ndarray:
        """Get state values V(s)"""
        return np.max(self.q_table, axis=1)

    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            "q_table": self.q_table,
            "returns": dict(self.returns),  # Convert defaultdict to dict
            "metrics": {
                "rewards": self.metrics.rewards,
                "success_rates": self.metrics.success_rates,
                "epsilon_values": self.metrics.epsilon_values,
            },
            "parameters": {
                "algorithm": "MonteCarlo",
                "first_visit": self.first_visit,
                "env_name": self.env_name,
                "map_name": self.map_name,
                "is_slippery": self.is_slippery,
                "discount_factor": self.gamma,
                "epsilon_min": self.epsilon_min,
                "episodes": self.episodes,
                "training_time": self.training_time,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.q_table = model_data["q_table"]
        if "returns" in model_data:
            self.returns = defaultdict(list, model_data["returns"])
        print(f"Model loaded from {filepath}")
        return model_data


class ClusterAnalyzer:
    """
    Analyze clusters from trained policies
    """

    def __init__(self, agent, name: str = "policy"):
        self.agent = agent
        self.name = name
        self.q_table = agent.q_table
        self.n_states = agent.n_states
        self.n_actions = agent.n_actions
        self.grid_size = agent.grid_size
        self.state_coords = agent.state_coords
        self.coord_to_state = agent.coord_to_state
        self.action_symbols = agent.action_symbols
        self.action_names = agent.action_names

        # Will be populated
        self.clusters = []
        self.state_to_cluster = {}
        self.cluster_dict = {}
        self.transition_graph = None

    def get_neighbors(self, state: int) -> List[int]:
        """Get 4-directional neighbors"""
        r, c = self.state_coords[state]
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append(self.coord_to_state[(nr, nc)])
        return neighbors

    def find_connected_clusters(self) -> Tuple[List[ConnectedActionCluster], Dict]:
        """
        Find connected components where states share the same preferred action
        """
        # Get deterministic policy
        policy = self.agent.get_deterministic_policy()
        state_values = self.agent.get_state_values()

        # Group by action first
        action_states = {a: [] for a in range(self.n_actions)}
        for state, action in policy.items():
            action_states[action].append(state)

        # Find connected components within each action group
        clusters = []
        state_to_cluster = {}
        cluster_id = 0

        for action, states in action_states.items():
            if not states:
                continue

            state_set = set(states)
            visited = set()

            for start_state in states:
                if start_state in visited:
                    continue

                # BFS to find connected component
                component = []
                queue = deque([start_state])
                visited.add(start_state)

                while queue:
                    current = queue.popleft()
                    component.append(current)

                    for neighbor in self.get_neighbors(current):
                        if neighbor in state_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                # Create cluster
                cluster = ConnectedActionCluster(cluster_id, action, component)

                # Compute centroid
                coords = [self.state_coords[s] for s in component]
                centroid_r = np.mean([c[0] for c in coords])
                centroid_c = np.mean([c[1] for c in coords])
                cluster.centroid = (float(centroid_r), float(centroid_c))

                # Compute values
                values = [state_values[s] for s in component]
                cluster.value_sum = float(np.sum(values))
                cluster.avg_value = float(np.mean(values))

                # Find boundary states
                boundary = []
                for state in component:
                    for neighbor in self.get_neighbors(state):
                        if neighbor not in state_set:
                            boundary.append(state)
                            break
                cluster.boundary_states = boundary

                clusters.append(cluster)

                for s in component:
                    state_to_cluster[s] = cluster_id

                cluster_id += 1

        self.clusters = clusters
        self.state_to_cluster = state_to_cluster
        self.cluster_dict = {c.id: c for c in clusters}

        return clusters, state_to_cluster

    def build_transition_graph(self, trajectories: List[Dict]) -> nx.DiGraph:
        """
        Build transition graph between clusters from trajectories
        """
        G = nx.DiGraph()

        # Add nodes
        for cluster in self.clusters:
            G.add_node(
                cluster.id,
                action=cluster.action,
                states=cluster.states,
                size=cluster.size,
                value_sum=cluster.value_sum,
                avg_value=cluster.avg_value,
                centroid=cluster.centroid,
                boundary_states=cluster.boundary_states,
            )

        # Count transitions
        transitions = defaultdict(
            lambda: defaultdict(lambda: {"count": 0, "actions": []})
        )

        for traj in trajectories:
            for i in range(len(traj["states"]) - 1):
                s = traj["states"][i]
                a = traj["actions"][i]
                s_next = traj["next_states"][i]

                if s in self.state_to_cluster and s_next in self.state_to_cluster:
                    c_from = self.state_to_cluster[s]
                    c_to = self.state_to_cluster[s_next]

                    if c_from != c_to:  # Only count cross-cluster transitions
                        transitions[c_from][c_to]["count"] += 1
                        transitions[c_from][c_to]["actions"].append(a)

        # Add edges with probabilities
        for c_from, targets in transitions.items():
            total = sum(d["count"] for d in targets.values())

            for c_to, data in targets.items():
                prob = data["count"] / total

                # Get action distribution
                action_counts = defaultdict(int)
                for a in data["actions"]:
                    action_counts[a] += 1

                action_dist = {
                    int(a): float(cnt / data["count"])
                    for a, cnt in action_counts.items()
                }

                G.add_edge(
                    int(c_from),
                    int(c_to),
                    probability=float(prob),
                    weight=float(prob * 3),
                    count=int(data["count"]),
                    action_distribution=action_dist,
                    dominant_action=int(max(action_dist, key=action_dist.get)),
                )

        self.transition_graph = G
        return G

    def collect_trajectories(
        self, n_trajectories: int = 1000, use_stochastic: bool = True
    ) -> List[Dict]:
        """
        Collect trajectories using the learned policy
        """
        trajectories = []
        attempts = 0
        max_attempts = n_trajectories * 10

        stochastic_policy = (
            self.agent.get_stochastic_policy() if use_stochastic else None
        )
        deterministic_policy = self.agent.get_deterministic_policy()

        while len(trajectories) < n_trajectories and attempts < max_attempts:
            attempts += 1
            state, _ = self.agent.env.reset()
            done = False
            trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "success": False,
            }

            while not done:
                if use_stochastic:
                    probs = stochastic_policy[state]
                    action = np.random.choice(self.n_actions, p=probs)
                else:
                    action = deterministic_policy[state]

                next_state, reward, terminated, truncated, _ = self.agent.env.step(
                    action
                )
                done = terminated or truncated

                trajectory["states"].append(int(state))
                trajectory["actions"].append(int(action))
                trajectory["rewards"].append(float(reward))
                trajectory["next_states"].append(int(next_state))

                state = next_state

            trajectory["success"] = bool(trajectory["rewards"][-1] > 0)

            if trajectory["success"]:
                trajectories.append(trajectory)

        print(
            f"Collected {len(trajectories)} successful trajectories ({attempts} attempts)"
        )
        return trajectories

    def find_optimal_path(
        self, start_state: int = 0, goal_state: int = 15
    ) -> Tuple[List[int], List[float]]:
        """
        Find most probable path through clusters
        """
        if not self.transition_graph:
            return [], []

        start_cluster = self.state_to_cluster.get(start_state)
        goal_cluster = self.state_to_cluster.get(goal_state)

        if start_cluster is None or goal_cluster is None:
            return [], []

        try:
            # Use negative log probability as cost
            G = self.transition_graph.copy()
            for u, v, d in G.edges(data=True):
                G[u][v]["cost"] = -np.log(d["probability"])

            path = nx.shortest_path(G, start_cluster, goal_cluster, weight="cost")
            path_probs = [
                G[path[i]][path[i + 1]]["probability"] for i in range(len(path) - 1)
            ]

            return path, path_probs
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], []

    def visualize_clusters(self, save_path: Optional[str] = None):
        """
        Visualize clusters on the grid
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Cluster map
        grid = np.full((self.grid_size, self.grid_size), -1)
        action_grid = np.full((self.grid_size, self.grid_size), -1)

        for cluster in self.clusters:
            for state in cluster.states:
                r, c = self.state_coords[state]
                grid[r, c] = cluster.id
                action_grid[r, c] = cluster.action

        # Create colormap for clusters
        unique_clusters = np.unique(grid[grid >= 0])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        color_map = {cid: colors[i] for i, cid in enumerate(unique_clusters)}

        # Draw grid
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r, c] >= 0:
                    color = color_map[grid[r, c]]
                    rect = plt.Rectangle(
                        (c, self.grid_size - 1 - r),
                        1,
                        1,
                        facecolor=color,
                        alpha=0.6,
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax1.add_patch(rect)

                    # Add action symbol
                    action = action_grid[r, c]
                    ax1.text(
                        c + 0.5,
                        self.grid_size - 0.5 - r,
                        self.action_symbols[action],
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                    )

        ax1.set_xlim(0, self.grid_size)
        ax1.set_ylim(0, self.grid_size)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(
            f"{self.name}: Connected Action Clusters\n({len(self.clusters)} clusters)"
        )
        ax1.set_aspect("equal")

        # Plot 2: Cluster graph
        if self.transition_graph and len(self.transition_graph.nodes) > 0:
            pos = {
                node: (data["centroid"][1], -data["centroid"][0])
                for node, data in self.transition_graph.nodes(data=True)
            }

            # Draw nodes
            for node in self.transition_graph.nodes():
                data = self.transition_graph.nodes[node]
                size = data["size"] * 500

                nx.draw_networkx_nodes(
                    self.transition_graph,
                    pos,
                    nodelist=[node],
                    node_size=size,
                    node_color=[color_map[node]],
                    edgecolors="black",
                    linewidths=2,
                    alpha=0.7,
                    ax=ax2,
                )

                ax2.text(
                    pos[node][0],
                    pos[node][1],
                    f"C{node}\n({self.action_symbols[data['action']]})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            # Draw edges
            for u, v, d in self.transition_graph.edges(data=True):
                nx.draw_networkx_edges(
                    self.transition_graph,
                    pos,
                    edgelist=[(u, v)],
                    width=d["weight"],
                    edge_color="gray",
                    alpha=0.5,
                    arrows=True,
                    arrowsize=15,
                    ax=ax2,
                )

                # Add edge label
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax2.text(
                    mid_x,
                    mid_y,
                    f"{d['probability']:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
                )

            ax2.set_title("Cluster Transition Graph")
        else:
            ax2.text(
                0.5,
                0.5,
                "No transition graph available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        ax2.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()

    def save_cluster_data(self, base_filename: str):
        """
        Save all cluster analysis data
        """
        os.makedirs("cluster_analysis", exist_ok=True)

        # Prepare serializable data
        cluster_data = {
            "name": self.name,
            "n_clusters": len(self.clusters),
            "clusters": [c.to_dict() for c in self.clusters],
            "state_to_cluster": {
                str(k): int(v) for k, v in self.state_to_cluster.items()
            },
            "policy": self.agent.get_deterministic_policy(),
            "state_values": {
                str(k): float(v) for k, v in enumerate(self.agent.get_state_values())
            },
            "grid_size": self.grid_size,
            "action_mapping": self.action_names,
        }

        # Save as JSON
        with open(f"cluster_analysis/{base_filename}_clusters.json", "w") as f:
            json.dump(cluster_data, f, indent=2)

        # Save graph if exists
        if self.transition_graph:
            with open(f"cluster_analysis/{base_filename}_graph.pkl", "wb") as f:
                pickle.dump(self.transition_graph, f)

        print(f"Cluster data saved to cluster_analysis/{base_filename}_*")


def run_comparison_experiment():
    """
    Run comprehensive comparison between SARSA and Monte Carlo
    """
    print("\n" + "=" * 70)
    print("SARSA vs MONTE CARLO COMPARISON ON FROZENLAKE")
    print("=" * 70)

    results = {}

    # 1. Train SARSA agent
    print("\n" + "-" * 70)
    print("TRAINING SARSA AGENT")
    print("-" * 70)

    sarsa_agent = FrozenLakeSarsa(
        map_name="4x4", is_slippery=True, learning_rate=0.1, episodes=10000
    )
    sarsa_results = sarsa_agent.train(verbose=True)
    sarsa_agent.save_model("models/sarsa_frozenlake.pkl")

    # 2. Train Monte Carlo agent
    print("\n" + "-" * 70)
    print("TRAINING MONTE CARLO AGENT")
    print("-" * 70)

    mc_agent = FrozenLakeMonteCarlo(
        map_name="4x4", is_slippery=True, first_visit=True, episodes=10000
    )
    mc_results = mc_agent.train(verbose=True)
    mc_agent.save_model("models/mc_frozenlake.pkl")

    # 3. Analyze clusters for SARSA
    print("\n" + "-" * 70)
    print("ANALYZING SARSA CLUSTERS")
    print("-" * 70)

    sarsa_analyzer = ClusterAnalyzer(sarsa_agent, "SARSA")
    sarsa_clusters, sarsa_state_to_cluster = sarsa_analyzer.find_connected_clusters()

    # Collect trajectories
    sarsa_trajs = sarsa_analyzer.collect_trajectories(n_trajectories=500)

    # Build transition graph
    sarsa_graph = sarsa_analyzer.build_transition_graph(sarsa_trajs)

    # Find optimal path
    sarsa_path, sarsa_probs = sarsa_analyzer.find_optimal_path()

    # Visualize
    sarsa_analyzer.visualize_clusters(save_path="cluster_analysis/sarsa_clusters.png")

    # Save data
    sarsa_analyzer.save_cluster_data("sarsa")

    # 4. Analyze clusters for Monte Carlo
    print("\n" + "-" * 70)
    print("ANALYZING MONTE CARLO CLUSTERS")
    print("-" * 70)

    mc_analyzer = ClusterAnalyzer(mc_agent, "MonteCarlo")
    mc_clusters, mc_state_to_cluster = mc_analyzer.find_connected_clusters()

    # Collect trajectories
    mc_trajs = mc_analyzer.collect_trajectories(n_trajectories=500)

    # Build transition graph
    mc_graph = mc_analyzer.build_transition_graph(mc_trajs)

    # Find optimal path
    mc_path, mc_probs = mc_analyzer.find_optimal_path()

    # Visualize
    mc_analyzer.visualize_clusters(save_path="cluster_analysis/mc_clusters.png")

    # Save data
    mc_analyzer.save_cluster_data("mc")

    # 5. Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    results["sarsa"] = {
        "final_success_rate": sarsa_results["final_success_rate"],
        "training_time": sarsa_results["training_time"],
        "n_clusters": len(sarsa_clusters),
        "cluster_sizes": [c.size for c in sarsa_clusters],
        "avg_cluster_size": np.mean([c.size for c in sarsa_clusters]),
        "optimal_path": [int(c) for c in sarsa_path] if sarsa_path else [],
        "path_probability": np.prod(sarsa_probs) if sarsa_probs else 0,
    }

    results["mc"] = {
        "final_success_rate": mc_results["final_success_rate"],
        "training_time": mc_results["training_time"],
        "n_clusters": len(mc_clusters),
        "cluster_sizes": [c.size for c in mc_clusters],
        "avg_cluster_size": np.mean([c.size for c in mc_clusters]),
        "optimal_path": [int(c) for c in mc_path] if mc_path else [],
        "path_probability": np.prod(mc_probs) if mc_probs else 0,
    }

    # Print comparison table
    print("\n{:<20} {:>15} {:>15}".format("Metric", "SARSA", "Monte Carlo"))
    print("-" * 52)
    print(
        "{:<20} {:>15.3f} {:>15.3f}".format(
            "Success Rate",
            results["sarsa"]["final_success_rate"],
            results["mc"]["final_success_rate"],
        )
    )
    print(
        "{:<20} {:>15.2f} {:>15.2f}".format(
            "Training Time (s)",
            results["sarsa"]["training_time"],
            results["mc"]["training_time"],
        )
    )
    print(
        "{:<20} {:>15} {:>15}".format(
            "Number of Clusters",
            results["sarsa"]["n_clusters"],
            results["mc"]["n_clusters"],
        )
    )
    print(
        "{:<20} {:>15.2f} {:>15.2f}".format(
            "Avg Cluster Size",
            results["sarsa"]["avg_cluster_size"],
            results["mc"]["avg_cluster_size"],
        )
    )

    if results["sarsa"]["optimal_path"]:
        print(
            "\nSARSA Optimal Path: "
            + " → ".join([f"C{c}" for c in results["sarsa"]["optimal_path"]])
        )
        print(f"Path Probability: {results['sarsa']['path_probability']:.3f}")

    if results["mc"]["optimal_path"]:
        print(
            "\nMC Optimal Path: "
            + " → ".join([f"C{c}" for c in results["mc"]["optimal_path"]])
        )
        print(f"Path Probability: {results['mc']['path_probability']:.3f}")

    # 6. Compare Q-value distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Q-value histograms
    axes[0, 0].hist(sarsa_agent.q_table.flatten(), bins=30, alpha=0.7, label="SARSA")
    axes[0, 0].hist(mc_agent.q_table.flatten(), bins=30, alpha=0.7, label="MC")
    axes[0, 0].set_xlabel("Q-values")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Q-Value Distribution")
    axes[0, 0].legend()

    # Success rate over time
    axes[0, 1].plot(sarsa_agent.metrics.success_rates, label="SARSA", alpha=0.7)
    axes[0, 1].plot(mc_agent.metrics.success_rates, label="MC", alpha=0.7)
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Success Rate")
    axes[0, 1].set_title("Learning Curves")
    axes[0, 1].legend()

    # State values heatmap comparison
    sarsa_values = sarsa_agent.get_state_values().reshape(4, 4)
    mc_values = mc_agent.get_state_values().reshape(4, 4)

    im1 = axes[1, 0].imshow(sarsa_values, cmap="viridis", aspect="equal")
    axes[1, 0].set_title("SARSA State Values")
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_yticks(range(4))
    plt.colorbar(im1, ax=axes[1, 0])

    im2 = axes[1, 1].imshow(mc_values, cmap="viridis", aspect="equal")
    axes[1, 1].set_title("Monte Carlo State Values")
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_yticks(range(4))
    plt.colorbar(im2, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(
        "cluster_analysis/sarsa_vs_mc_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.show()

    # Save results
    with open("cluster_analysis/comparison_results.json", "w") as f:
        # Convert numpy types to Python natives
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {convert(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - models/sarsa_frozenlake.pkl")
    print("  - models/mc_frozenlake.pkl")
    print("  - cluster_analysis/sarsa_clusters.json")
    print("  - cluster_analysis/mc_clusters.json")
    print("  - cluster_analysis/sarsa_vs_mc_comparison.png")
    print("  - cluster_analysis/comparison_results.json")

    return {
        "sarsa": (sarsa_agent, sarsa_analyzer, sarsa_trajs),
        "mc": (mc_agent, mc_analyzer, mc_trajs),
        "results": results,
    }


def load_and_replay(model_path: str, n_episodes: int = 10, render: bool = False):
    """
    Load a saved model and replay episodes
    """
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    # Determine algorithm type from parameters
    params = model_data.get("parameters", {})
    algo = params.get("algorithm", "Unknown")

    print(f"\nLoading {algo} model from {model_path}")

    if algo == "SARSA":
        agent = FrozenLakeSarsa(
            **{
                k: v
                for k, v in params.items()
                if k in ["env_name", "map_name", "is_slippery"]
            }
        )
    else:
        agent = FrozenLakeMonteCarlo(
            **{
                k: v
                for k, v in params.items()
                if k in ["env_name", "map_name", "is_slippery"]
            }
        )

    agent.q_table = model_data["q_table"]

    print(f"\nReplaying {n_episodes} episodes:")
    print("-" * 50)

    for episode in range(n_episodes):
        state, _ = agent.env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = np.argmax(agent.q_table[state])
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1

            if render:
                agent.env.render()

        print(
            f"Episode {episode + 1}: Steps={steps}, Reward={total_reward}, Success={'✓' if total_reward > 0 else '✗'}"
        )

    return agent


if __name__ == "__main__":
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("cluster_analysis", exist_ok=True)

    # Run comparison experiment
    results = run_comparison_experiment()

    # Example: Load and replay a saved model
    print("\n" + "=" * 70)
    print("LOADING AND REPLAYING SARSA MODEL")
    print("=" * 70)
    load_and_replay("models/sarsa_frozenlake.pkl", n_episodes=5)
