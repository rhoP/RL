import numpy as np
import gymnasium as gym
import pickle
import os
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
import json
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import ndimage
from sklearn.cluster import DBSCAN


class ConnectedActionCluster:
    """
    Represents a connected cluster of states that share the same preferred action
    """

    def __init__(self, cluster_id: int, action: int, states: List[int]):
        self.id = cluster_id
        self.action = action
        self.states = sorted(states)
        self.size = len(states)
        self.centroid = None
        self.value_sum = 0.0
        self.avg_value = 0.0
        self.boundary_states = []  # States on the edge of the cluster
        self.incoming_transitions = defaultdict(float)  # {cluster_id: probability}
        self.outgoing_transitions = defaultdict(float)  # {cluster_id: probability}

    def __repr__(self):
        return f"Cluster({self.id}, action={self.action}, size={self.size})"


class StochasticPolicyLearner:
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
        episodes: int = 20000,
        temperature: float = 1.0,
    ):
        """
        Initialize learner for stochastic policy on FrozenLake
        """
        self.env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)
        self.env_name = env_name
        self.map_name = map_name
        self.is_slippery = is_slippery

        # Q-learning parameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.temperature = temperature

        # Initialize Q-table
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # Tracking metrics
        self.rewards_history = []
        self.success_rate_history = []

        # Grid dimensions
        self.grid_size = 4
        self.n_rows = self.grid_size
        self.n_cols = self.grid_size
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
        self.action_vectors = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # (dr, dc)

        # Neighbor directions (4-directional connectivity)
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    def softmax_policy(self, state: int) -> np.ndarray:
        """
        Compute softmax policy probabilities for a state
        """
        q_values = self.q_table[state]
        # Subtract max for numerical stability
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / self.temperature)
        return exp_q / np.sum(exp_q)

    def choose_action_softmax(self, state: int) -> int:
        """
        Choose action according to softmax policy
        """
        probs = self.softmax_policy(state)
        return np.random.choice(self.n_actions, p=probs)

    def choose_action_epsilon_greedy(self, state: int) -> int:
        """
        Choose action using epsilon-greedy for training
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def train(self, verbose: bool = True) -> Dict:
        """
        Train using Q-learning with epsilon-greedy exploration
        """
        print(f"Training stochastic policy on {self.map_name} FrozenLake...")
        print(f"Slippery: {self.is_slippery}")
        print("-" * 50)

        success_window = []

        for episode in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action_epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Q-learning update
                best_next = np.max(self.q_table[next_state])
                self.q_table[state, action] += self.lr * (
                    reward + self.gamma * best_next - self.q_table[state, action]
                )

                state = next_state
                total_reward += reward

            # Track success
            success = 1 if total_reward > 0 else 0
            success_window.append(success)
            if len(success_window) > 100:
                success_window.pop(0)

            success_rate = np.mean(success_window)
            self.success_rate_history.append(float(success_rate))
            self.rewards_history.append(float(total_reward))

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (episode + 1) % 2000 == 0:
                print(
                    f"Episode {episode + 1}/{self.episodes} | "
                    f"Success Rate: {success_rate:.3f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        print("-" * 50)
        print(f"Training completed! Final success rate: {success_rate:.3f}")

        return {
            "q_table": self.q_table,
            "success_rate_history": self.success_rate_history,
            "final_success_rate": float(success_rate),
        }

    def get_stochastic_policy(self) -> Dict[int, np.ndarray]:
        """
        Get the learned stochastic policy (softmax probabilities)
        """
        policy = {}
        for state in range(self.n_states):
            policy[state] = self.softmax_policy(state)
        return policy

    def get_deterministic_policy(self) -> Dict[int, int]:
        """
        Get deterministic policy (max probability action)
        """
        policy = {}
        for state in range(self.n_states):
            probs = self.softmax_policy(state)
            policy[state] = int(np.argmax(probs))
        return policy

    def get_state_values(self) -> np.ndarray:
        """
        Get state values V(s) = max_a Q(s,a)
        """
        return np.max(self.q_table, axis=1)

    def get_neighbors(self, state: int) -> List[int]:
        """
        Get 4-directional neighbors of a state that exist in the grid
        """
        r, c = self.state_coords[state]
        neighbors = []

        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                neighbors.append(self.coord_to_state[(nr, nc)])

        return neighbors

    def find_connected_components(
        self, action_states: Dict[int, List[int]]
    ) -> Dict[int, List[ConnectedActionCluster]]:
        """
        Find connected components within each action group using BFS
        Returns: {action: [list of ConnectedActionCluster objects]}
        """
        action_clusters = {action: [] for action in range(self.n_actions)}
        cluster_id = 0

        for action, states in action_states.items():
            if not states:
                continue

            # Create set of states for this action
            state_set = set(states)
            visited = set()

            # Find connected components using BFS
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

                    # Check neighbors that share the same action
                    for neighbor in self.get_neighbors(current):
                        if neighbor in state_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                # Create cluster object
                cluster = ConnectedActionCluster(cluster_id, action, component)
                action_clusters[action].append(cluster)
                cluster_id += 1

        return action_clusters

    def cluster_states_by_action_and_connectivity(
        self,
    ) -> Tuple[Dict[int, List[ConnectedActionCluster]], Dict[int, int]]:
        """
        Cluster states by both preferred action and connectivity
        Returns: (action_clusters, state_to_cluster_id mapping)
        """
        # First group by action
        deterministic_policy = self.get_deterministic_policy()
        action_states = {action: [] for action in range(self.n_actions)}

        for state, action in deterministic_policy.items():
            action_states[action].append(int(state))

        # Find connected components within each action group
        action_clusters = self.find_connected_components(action_states)

        # Create state to cluster mapping
        state_to_cluster_id = {}
        for action, clusters in action_clusters.items():
            for cluster in clusters:
                for state in cluster.states:
                    state_to_cluster_id[state] = cluster.id

        return action_clusters, state_to_cluster_id

    def compute_cluster_centroids(
        self, action_clusters: Dict[int, List[ConnectedActionCluster]]
    ):
        """
        Compute centroids for each cluster based on state coordinates
        """
        for action, clusters in action_clusters.items():
            for cluster in clusters:
                if cluster.states:
                    coords = [self.state_coords[s] for s in cluster.states]
                    centroid_r = np.mean([c[0] for c in coords])
                    centroid_c = np.mean([c[1] for c in coords])
                    cluster.centroid = (centroid_r, centroid_c)

    def compute_cluster_values(
        self, action_clusters: Dict[int, List[ConnectedActionCluster]]
    ):
        """
        Compute sum and average of state values for each cluster
        """
        state_values = self.get_state_values()

        for action, clusters in action_clusters.items():
            for cluster in clusters:
                values = [state_values[s] for s in cluster.states]
                cluster.value_sum = float(np.sum(values))
                cluster.avg_value = float(np.mean(values))

    def identify_boundary_states(
        self, action_clusters: Dict[int, List[ConnectedActionCluster]]
    ):
        """
        Identify states on the boundary of each cluster (adjacent to different action or outside)
        """
        for action, clusters in action_clusters.items():
            for cluster in clusters:
                boundary = []
                cluster_states = set(cluster.states)

                for state in cluster.states:
                    # Check if any neighbor is in a different cluster or is invalid
                    for neighbor in self.get_neighbors(state):
                        if neighbor not in cluster_states:
                            boundary.append(state)
                            break

                cluster.boundary_states = boundary

    def build_clustered_transition_graph(
        self, trajectories: List[Dict]
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Build a directed graph where nodes are connected action clusters
        """
        # Get clusters
        action_clusters, state_to_cluster_id = (
            self.cluster_states_by_action_and_connectivity()
        )

        # Compute cluster properties
        self.compute_cluster_centroids(action_clusters)
        self.compute_cluster_values(action_clusters)
        self.identify_boundary_states(action_clusters)

        # Create mapping from cluster ID to cluster object
        cluster_dict = {}
        for action, clusters in action_clusters.items():
            for cluster in clusters:
                cluster_dict[cluster.id] = cluster

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for cluster in cluster_dict.values():
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

        # Count transitions between clusters from trajectories
        cluster_transitions = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "count": 0,
                    "actions": [],
                    "from_boundary": [],
                    "to_boundary": [],
                }
            )
        )

        for traj in trajectories:
            for i in range(len(traj["states"]) - 1):
                s = traj["states"][i]
                a = traj["actions"][i]
                s_next = traj["next_states"][i]

                cluster_from = state_to_cluster_id[s]
                cluster_to = state_to_cluster_id[s_next]

                # Record transition
                trans_data = cluster_transitions[cluster_from][cluster_to]
                trans_data["count"] += 1
                trans_data["actions"].append(a)

                # Track if transition starts/ends at boundary
                if s in cluster_dict[cluster_from].boundary_states:
                    trans_data["from_boundary"].append(True)
                if s_next in cluster_dict[cluster_to].boundary_states:
                    trans_data["to_boundary"].append(True)

        # Add edges with probabilities and metadata
        for cluster_from, targets in cluster_transitions.items():
            total_from = sum(data["count"] for data in targets.values())

            for cluster_to, data in targets.items():
                prob = data["count"] / total_from

                # Calculate action distribution
                action_counts = defaultdict(int)
                for a in data["actions"]:
                    action_counts[a] += 1

                action_dist = {
                    int(a): float(cnt / data["count"])
                    for a, cnt in action_counts.items()
                }

                # Calculate boundary transition probability
                p_from_boundary = (
                    np.mean(data["from_boundary"]) if data["from_boundary"] else 0
                )
                p_to_boundary = (
                    np.mean(data["to_boundary"]) if data["to_boundary"] else 0
                )

                # Add directed edge
                G.add_edge(
                    int(cluster_from),
                    int(cluster_to),
                    probability=float(prob),
                    weight=float(prob * 3),  # Scale for visualization
                    count=int(data["count"]),
                    action_distribution=action_dist,
                    dominant_action=max(action_dist, key=action_dist.get),
                    p_from_boundary=float(p_from_boundary),
                    p_to_boundary=float(p_to_boundary),
                )

        return G, cluster_dict

    def find_optimal_path_clusters(
        self,
        G: nx.DiGraph,
        cluster_dict: Dict,
        start_state: int = 0,
        goal_state: int = 15,
    ):
        """
        Find the most probable path through clusters from start to goal
        """
        start_cluster = None
        goal_cluster = None

        # Find clusters containing start and goal
        for cluster_id, cluster in cluster_dict.items():
            if start_state in cluster.states:
                start_cluster = cluster_id
            if goal_state in cluster.states:
                goal_cluster = cluster_id

        if start_cluster is None or goal_cluster is None:
            print("Warning: Start or goal state not found in clusters")
            return []

        # Use Dijkstra-like algorithm to find highest probability path
        try:
            # Convert probabilities to costs for shortest path
            for u, v, d in G.edges(data=True):
                G[u][v]["cost"] = (
                    -np.log(d["probability"]) if d["probability"] > 0 else float("inf")
                )

            path = nx.shortest_path(G, start_cluster, goal_cluster, weight="cost")

            # Get path probabilities
            path_probs = []
            for i in range(len(path) - 1):
                prob = G[path[i]][path[i + 1]]["probability"]
                path_probs.append(prob)

            return path, path_probs
        except nx.NetworkXNoPath:
            print("No path found from start to goal cluster")
            return []

    def visualize_clustered_graph(
        self,
        G: nx.DiGraph,
        cluster_dict: Dict,
        highlight_path: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize the connected action cluster graph with flow
        """
        plt.figure(figsize=(16, 12))

        # Create colormap for actions
        action_colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

        # Create colormap for node values
        values = [cluster_dict[n].value_sum for n in G.nodes()]
        if max(values) > min(values):
            norm = Normalize(vmin=min(values), vmax=max(values))
        else:
            norm = Normalize(vmin=0, vmax=1)
        cmap = cm.viridis

        # Calculate node positions based on actual grid centroids
        pos = {}
        for node in G.nodes():
            centroid = cluster_dict[node].centroid
            if centroid:
                # Transform centroid to plot coordinates
                r, c = centroid
                # Flip r for proper orientation (0 at top)
                pos[node] = (c, -r)
            else:
                pos[node] = (0, 0)

        # Draw nodes
        for node in G.nodes():
            cluster = cluster_dict[node]
            size = cluster.size * 1000  # Scale node size

            # Color based on sum of values
            node_color = cmap(norm(cluster.value_sum))

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_size=size,
                node_color=[node_color],
                edgecolors="black",
                linewidths=2,
                alpha=0.9,
            )

            # Add cluster label
            plt.text(
                pos[node][0],
                pos[node][1] + 0.2,
                f"C{node}\n({self.action_symbols[cluster.action]})",
                horizontalalignment="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

            # Add value info
            plt.text(
                pos[node][0],
                pos[node][1] - 0.2,
                f"Σ={cluster.value_sum:.1f}",
                horizontalalignment="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.7),
            )

        # Draw edges
        edges = G.edges(data=True)
        if edges:
            for u, v, d in edges:
                # Edge width based on probability
                width = d["weight"]

                # Edge color based on dominant action
                edge_color = action_colors[d["dominant_action"]]

                # Draw edge with curvature
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=edge_color,
                    alpha=0.6,
                    arrows=True,
                    arrowsize=20,
                    # arrowstyle="->",
                    connectionstyle="arc3,rad=0.1",
                )

                # Add edge label
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2

                # Add slight offset for multiple edges
                offset = 0.1 if (v, u) in G.edges() else 0

                label = f"{d['probability']:.2f}\n({self.action_symbols[d['dominant_action']]})"
                plt.text(
                    mid_x + offset,
                    mid_y + offset,
                    label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9),
                )

        # Highlight optimal path if provided
        if highlight_path:
            path_edges = [
                (highlight_path[i], highlight_path[i + 1])
                for i in range(len(highlight_path) - 1)
            ]

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=path_edges,
                width=4,
                edge_color="red",
                alpha=0.8,
                arrows=True,
                arrowsize=25,
                # arrowstyle="->",
                connectionstyle="arc3,rad=0.1",
            )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
        cbar.set_label("Sum of State Values in Cluster", fontsize=12)

        # Add legend for actions
        legend_elements = []
        for action, color in enumerate(action_colors):
            legend_elements.append(
                plt.Line2D(
                    [0], [0], color=color, lw=2, label=f"{self.action_names[action]}"
                )
            )

        plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

        plt.title(
            "Connected Action Cluster Graph\n(Nodes = Connected regions with same preferred action)",
            fontsize=14,
            pad=20,
        )
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Clustered graph saved to {save_path}")

        plt.show()

    def visualize_cluster_flow(
        self,
        G: nx.DiGraph,
        cluster_dict: Dict,
        trajectories: List[Dict],
        save_path: Optional[str] = None,
    ):
        """
        Visualize the flow through clusters from start to goal
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Cluster grid with flow arrows
        ax1.set_title("Cluster Flow Map", fontsize=14)

        # Create grid visualization
        grid = np.full((self.n_rows, self.n_cols), -1, dtype=int)
        action_grid = np.full((self.n_rows, self.n_cols), -1, dtype=int)

        # Fill grid with cluster IDs and actions
        for cluster in cluster_dict.values():
            for state in cluster.states:
                r, c = self.state_coords[state]
                grid[r, c] = cluster.id
                action_grid[r, c] = cluster.action

        # Create cluster color map
        unique_clusters = list(cluster_dict.keys())
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        cluster_color_map = {
            cid: cluster_colors[i] for i, cid in enumerate(unique_clusters)
        }

        # Draw grid cells
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                cluster_id = grid[r, c]
                if cluster_id >= 0:
                    color = cluster_color_map[cluster_id]
                    rect = plt.Rectangle(
                        (c, self.n_rows - 1 - r),
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
                        self.n_rows - 0.5 - r,
                        self.action_symbols[action],
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                    )

        # Add flow arrows based on trajectory statistics
        # Calculate cluster transition counts
        cluster_flow = defaultdict(lambda: defaultdict(int))
        for traj in trajectories:
            prev_cluster = None
            for state in traj["states"]:
                curr_cluster = None
                for cid, cluster in cluster_dict.items():
                    if state in cluster.states:
                        curr_cluster = cid
                        break

                if (
                    prev_cluster is not None
                    and curr_cluster is not None
                    and prev_cluster != curr_cluster
                ):
                    cluster_flow[prev_cluster][curr_cluster] += 1
                prev_cluster = curr_cluster

        # Draw flow arrows between cluster centroids
        for cid_from, targets in cluster_flow.items():
            total = sum(targets.values())
            centroid_from = cluster_dict[cid_from].centroid
            if centroid_from:
                x_from, y_from = centroid_from[1], self.n_rows - 1 - centroid_from[0]

                for cid_to, count in targets.items():
                    prob = count / total
                    if prob > 0.05:  # Only show significant flows
                        centroid_to = cluster_dict[cid_to].centroid
                        if centroid_to:
                            x_to, y_to = (
                                centroid_to[1],
                                self.n_rows - 1 - centroid_to[0],
                            )

                            # Draw arrow
                            ax1.annotate(
                                "",
                                xy=(x_to + 0.5, y_to + 0.5),
                                xytext=(x_from + 0.5, y_from + 0.5),
                                arrowprops=dict(
                                    arrowstyle="->", color="red", lw=2 * prob, alpha=0.7
                                ),
                            )

        ax1.set_xlim(0, self.n_cols)
        ax1.set_ylim(0, self.n_rows)
        ax1.set_xticks(range(self.n_cols + 1))
        ax1.set_yticks(range(self.n_rows + 1))
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        # Plot 2: Cluster transition graph with flow weights
        ax2.set_title("Cluster Transition Graph with Flow", fontsize=14)

        # Recreate graph with flow weights
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        for node in G.nodes():
            cluster = cluster_dict[node]
            size = cluster.size * 1000
            nx.draw_networkx_nodes(
                G,
                pos,
                [node],
                node_size=size,
                node_color=[cluster_color_map[node]],
                edgecolors="black",
                linewidths=2,
                alpha=0.7,
                ax=ax2,
            )

            ax2.text(
                pos[node][0],
                pos[node][1],
                f"C{node}\n({self.action_symbols[cluster.action]})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Draw edges with flow-based widths
        for u, v, d in G.edges(data=True):
            flow_weight = d["probability"] * 3
            nx.draw_networkx_edges(
                G,
                pos,
                [(u, v)],
                width=flow_weight,
                edge_color="gray",
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                ax=ax2,
            )

        ax2.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Flow visualization saved to {save_path}")

        plt.show()

    def collect_successful_trajectories(self, n_trajectories: int = 1000) -> List[Dict]:
        """
        Collect successful trajectories using the stochastic policy
        """
        trajectories = []
        attempts = 0
        max_attempts = n_trajectories * 10

        while len(trajectories) < n_trajectories and attempts < max_attempts:
            attempts += 1
            state, _ = self.env.reset()
            done = False
            trajectory = {
                "states": [],
                "actions": [],
                "action_probs": [],
                "rewards": [],
                "next_states": [],
                "success": False,
            }

            while not done:
                probs = self.softmax_policy(state)
                action = np.random.choice(self.n_actions, p=probs)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                trajectory["states"].append(int(state))
                trajectory["actions"].append(int(action))
                trajectory["action_probs"].append([float(p) for p in probs])
                trajectory["rewards"].append(float(reward))
                trajectory["next_states"].append(int(next_state))

                state = next_state

            trajectory["success"] = bool(trajectory["rewards"][-1] > 0)

            if trajectory["success"]:
                trajectories.append(trajectory)

        print(
            f"Collected {len(trajectories)} successful trajectories out of {attempts} attempts"
        )
        return trajectories

    def save_all_data(self, base_filename: str = "frozen_lake_connected_clusters"):
        """
        Save all learned data and analysis
        """
        os.makedirs("cluster_data", exist_ok=True)

        # Get basic data
        stochastic_policy = self.get_stochastic_policy()
        deterministic_policy = self.get_deterministic_policy()
        state_values = self.get_state_values()

        # Collect trajectories
        trajectories = self.collect_successful_trajectories(n_trajectories=1000)

        # Build clustered graph
        G, cluster_dict = self.build_clustered_transition_graph(trajectories)

        # Find optimal path
        optimal_path, path_probs = self.find_optimal_path_clusters(G, cluster_dict)

        # Convert to serializable format
        cluster_data = {}
        for cid, cluster in cluster_dict.items():
            cluster_data[str(cid)] = {
                "id": int(cluster.id),
                "action": int(cluster.action),
                "states": [int(s) for s in cluster.states],
                "size": int(cluster.size),
                "value_sum": float(cluster.value_sum),
                "avg_value": float(cluster.avg_value),
                "centroid": [float(cluster.centroid[0]), float(cluster.centroid[1])]
                if cluster.centroid
                else None,
                "boundary_states": [int(s) for s in cluster.boundary_states],
            }

        # Save data
        data = {
            "clusters": cluster_data,
            "deterministic_policy": {
                str(k): int(v) for k, v in deterministic_policy.items()
            },
            "state_values": {str(k): float(v) for k, v in enumerate(state_values)},
            "optimal_path": [int(c) for c in optimal_path] if optimal_path else [],
            "path_probabilities": [float(p) for p in path_probs] if path_probs else [],
            "parameters": {
                "map_name": self.map_name,
                "is_slippery": bool(self.is_slippery),
                "temperature": float(self.temperature),
            },
        }

        with open(f"cluster_data/{base_filename}_data.json", "w") as f:
            json.dump(data, f, indent=2)

        # Save trajectories
        serializable_trajectories = []
        for traj in trajectories:
            serializable_traj = {
                "states": [int(s) for s in traj["states"]],
                "actions": [int(a) for a in traj["actions"]],
                "rewards": [float(r) for r in traj["rewards"]],
                "next_states": [int(s) for s in traj["next_states"]],
            }
            serializable_trajectories.append(serializable_traj)

        with open(f"cluster_data/{base_filename}_trajectories.json", "w") as f:
            json.dump(serializable_trajectories, f, indent=2)

        # Save graph
        with open(f"cluster_data/{base_filename}_graph.pkl", "wb") as f:
            pickle.dump(G, f)

        # Create visualizations
        self.visualize_clustered_graph(
            G,
            cluster_dict,
            optimal_path,
            save_path=f"cluster_data/{base_filename}_graph.png",
        )
        self.visualize_cluster_flow(
            G,
            cluster_dict,
            trajectories,
            save_path=f"cluster_data/{base_filename}_flow.png",
        )

        print(f"\nAll data saved in 'cluster_data/' directory")

        return G, cluster_dict, trajectories, optimal_path


def analyze_clusters(G: nx.DiGraph, cluster_dict: Dict):
    """
    Print detailed cluster analysis
    """
    print("\n" + "=" * 70)
    print("CONNECTED ACTION CLUSTER ANALYSIS")
    print("=" * 70)

    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    print(f"\nTotal clusters: {len(cluster_dict)}")

    print("\n" + "-" * 70)
    print("CLUSTER DETAILS:")
    print("-" * 70)

    for cid, cluster in sorted(cluster_dict.items()):
        print(f"\nCluster {cid} ({action_names[cluster.action]} action):")
        print(f"  States: {cluster.states}")
        print(f"  Size: {cluster.size}")
        print(f"  Sum of values: {cluster.value_sum:.3f}")
        print(f"  Avg value: {cluster.avg_value:.3f}")
        print(f"  Boundary states: {cluster.boundary_states}")
        print(f"  Centroid: {cluster.centroid}")

        # Get outgoing edges
        outgoing = G.out_edges(cid, data=True)
        if outgoing:
            print(f"  Outgoing transitions:")
            for _, target, data in outgoing:
                print(
                    f"    → C{target}: {data['probability']:.3f} "
                    f"(action: {action_names[data['dominant_action']]})"
                )

        # Get incoming edges
        incoming = G.in_edges(cid, data=True)
        if incoming:
            print(f"  Incoming transitions:")
            for source, _, data in incoming:
                print(f"    ← C{source}: {data['probability']:.3f}")

    print("\n" + "-" * 70)
    print("CLUSTER TRANSITION MATRIX:")
    print("-" * 70)

    # Create transition matrix
    n_clusters = len(cluster_dict)
    matrix = np.zeros((n_clusters, n_clusters))

    for u, v, d in G.edges(data=True):
        matrix[u, v] = d["probability"]

    # Print matrix
    print("\n      To:")
    print("      " + " ".join([f"C{i:2}" for i in range(n_clusters)]))
    print("From: " + "-" * (n_clusters * 4))
    for i in range(n_clusters):
        row_str = f"C{i:2}:  "
        for j in range(n_clusters):
            row_str += f"{matrix[i, j]:.2f} "
        print(row_str)


def main():
    """
    Main experiment function
    """
    print("=" * 70)
    print("CONNECTED ACTION CLUSTER LEARNING ON FROZENLAKE")
    print("=" * 70)

    # Initialize learner
    learner = StochasticPolicyLearner(
        map_name="4x4", is_slippery=True, episodes=20000, temperature=1.0
    )

    # Train
    results = learner.train(verbose=True)

    # Get clusters
    action_clusters, state_to_cluster = (
        learner.cluster_states_by_action_and_connectivity()
    )

    print("\n" + "=" * 70)
    print("CONNECTED CLUSTERS BY ACTION")
    print("=" * 70)

    for action, clusters in action_clusters.items():
        print(f"\n{learner.action_names[action]} (Action {action}):")
        for cluster in clusters:
            print(f"  Cluster {cluster.id}: {cluster.states}")

    # Collect trajectories and build graph
    trajectories = learner.collect_successful_trajectories(n_trajectories=1000)
    G, cluster_dict = learner.build_clustered_transition_graph(trajectories)

    # Find optimal path
    optimal_path, path_probs = learner.find_optimal_path_clusters(G, cluster_dict)

    if optimal_path:
        print("\n" + "=" * 70)
        print("OPTIMAL CLUSTER PATH FROM START TO GOAL")
        print("=" * 70)

        path_str = " → ".join(
            [
                f"C{c}({learner.action_symbols[cluster_dict[c].action]})"
                for c in optimal_path
            ]
        )
        print(f"\nPath: {path_str}")

        prob_product = np.prod(path_probs)
        print(f"Path probabilities: {[f'{p:.3f}' for p in path_probs]}")
        print(f"Overall probability: {prob_product:.3f}")

    # Analyze clusters
    analyze_clusters(G, cluster_dict)

    # Visualize
    learner.visualize_clustered_graph(G, cluster_dict, optimal_path)
    learner.visualize_cluster_flow(G, cluster_dict, trajectories)

    # Save data
    G, cluster_dict, trajectories, optimal_path = learner.save_all_data()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return learner, G, cluster_dict, trajectories


if __name__ == "__main__":
    learner, G, cluster_dict, trajectories = main()
