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
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import warnings
from sarsa_monte import FrozenLakeSarsa, FrozenLakeMonteCarlo, ClusterAnalyzer

warnings.filterwarnings("ignore")


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


class ValueSublevelSet:
    """
    Represents a sub-level set of the value function
    V(s) <= threshold
    """

    def __init__(
        self, threshold: float, states: List[int], value_range: Tuple[float, float]
    ):
        self.threshold = threshold
        self.states = sorted(states)
        self.size = len(states)
        self.value_range = value_range  # (min_value, max_value) in this set
        self.connected_components = []  # Will store connected components within this sublevel

    def to_dict(self):
        return {
            "threshold": self.threshold,
            "states": self.states,
            "size": self.size,
            "value_range": self.value_range,
            "n_components": len(self.connected_components),
        }


class ValueLevelSetCluster:
    """
    Cluster based on value function level sets
    Combines both value threshold and connectivity
    """

    def __init__(
        self,
        cluster_id: int,
        level: float,
        states: List[int],
        value_range: Tuple[float, float],
        is_high_value: bool,
    ):
        self.id = cluster_id
        self.level = level  # The value threshold
        self.states = sorted(states)
        self.size = len(states)
        self.value_range = value_range
        self.is_high_value = (
            is_high_value  # True if this is a high-value region (above threshold)
        )
        self.centroid = None
        self.boundary_states = []
        self.avg_value = np.mean(value_range)

    def to_dict(self):
        return {
            "id": self.id,
            "level": self.level,
            "states": self.states,
            "size": self.size,
            "value_range": self.value_range,
            "is_high_value": self.is_high_value,
            "centroid": self.centroid,
            "boundary_states": self.boundary_states,
            "avg_value": self.avg_value,
        }


class ValueClusterAnalyzer:
    """
    Analyze value function topology using sub-level sets
    """

    def __init__(self, agent, name: str = "policy"):
        self.agent = agent
        self.name = name
        self.q_table = agent.q_table
        self.state_values = agent.get_state_values()
        self.n_states = agent.n_states
        self.n_actions = agent.n_actions
        self.grid_size = agent.grid_size
        self.state_coords = agent.state_coords
        self.coord_to_state = agent.coord_to_state

        # Value statistics
        self.v_min = np.min(self.state_values)
        self.v_max = np.max(self.state_values)
        self.v_mean = np.mean(self.state_values)
        self.v_std = np.std(self.state_values)

        # Will store level set clusters
        self.level_clusters = []  # List of ValueLevelSetCluster
        self.state_to_level_cluster = {}  # Map state to level cluster ID

    def get_neighbors(self, state: int) -> List[int]:
        """Get 4-directional neighbors"""
        r, c = self.state_coords[state]
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append(self.coord_to_state[(nr, nc)])
        return neighbors

    def compute_sublevel_sets(self, n_levels: int = 5) -> List[ValueSublevelSet]:
        """
        Compute sub-level sets at different thresholds
        V(s) <= threshold
        """
        # Create thresholds from min to max
        thresholds = np.linspace(self.v_min, self.v_max, n_levels + 1)[
            1:
        ]  # Exclude min

        sublevel_sets = []

        for threshold in thresholds:
            # States with value <= threshold
            states = [
                s for s in range(self.n_states) if self.state_values[s] <= threshold
            ]

            if states:
                values_in_set = [self.state_values[s] for s in states]
                value_range = (min(values_in_set), max(values_in_set))

                sublevel_set = ValueSublevelSet(
                    threshold=float(threshold), states=states, value_range=value_range
                )

                # Find connected components within this sublevel set
                self._find_connected_components_in_sublevel(sublevel_set)

                sublevel_sets.append(sublevel_set)

        return sublevel_sets

    def compute_superlevel_sets(self, n_levels: int = 5) -> List[ValueSublevelSet]:
        """
        Compute super-level sets (V(s) >= threshold)
        """
        # Create thresholds from min to max
        thresholds = np.linspace(self.v_min, self.v_max, n_levels + 1)[
            :-1
        ]  # Exclude max

        superlevel_sets = []

        for threshold in thresholds:
            # States with value >= threshold
            states = [
                s for s in range(self.n_states) if self.state_values[s] >= threshold
            ]

            if states:
                values_in_set = [self.state_values[s] for s in states]
                value_range = (min(values_in_set), max(values_in_set))

                superlevel_set = ValueSublevelSet(
                    threshold=float(threshold), states=states, value_range=value_range
                )

                # Find connected components
                self._find_connected_components_in_sublevel(superlevel_set)

                superlevel_sets.append(superlevel_set)

        return superlevel_sets

    def _find_connected_components_in_sublevel(self, level_set: ValueSublevelSet):
        """
        Find connected components within a level set
        """
        state_set = set(level_set.states)
        visited = set()
        components = []

        for start_state in level_set.states:
            if start_state in visited:
                continue

            # BFS to find component
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

            components.append(component)

        level_set.connected_components = components

    def cluster_by_value_levels(
        self, n_levels: int = 5, use_superlevel: bool = True
    ) -> List[ValueLevelSetCluster]:
        """
        Create clusters based on value levels and connectivity
        """
        if use_superlevel:
            level_sets = self.compute_superlevel_sets(n_levels)
        else:
            level_sets = self.compute_sublevel_sets(n_levels)

        clusters = []
        cluster_id = 0

        for level_set in level_sets:
            is_high_value = (
                use_superlevel  # Superlevel = high value, Sublevel = low value
            )

            for component in level_set.connected_components:
                values = [self.state_values[s] for s in component]

                cluster = ValueLevelSetCluster(
                    cluster_id=cluster_id,
                    level=float(level_set.threshold),
                    states=component,
                    value_range=(float(min(values)), float(max(values))),
                    is_high_value=is_high_value,
                )

                # Compute centroid
                coords = [self.state_coords[s] for s in component]
                centroid_r = np.mean([c[0] for c in coords])
                centroid_c = np.mean([c[1] for c in coords])
                cluster.centroid = (float(centroid_r), float(centroid_c))

                # Find boundary states
                boundary = []
                component_set = set(component)
                for state in component:
                    for neighbor in self.get_neighbors(state):
                        if neighbor not in component_set:
                            boundary.append(state)
                            break
                cluster.boundary_states = boundary

                clusters.append(cluster)

                for s in component:
                    self.state_to_level_cluster[s] = cluster_id

                cluster_id += 1

        self.level_clusters = clusters
        return clusters

    def compute_value_topology_features(self) -> Dict:
        """
        Compute topological features of the value function
        """
        features = {}

        # Create value grid
        value_grid = np.zeros((self.grid_size, self.grid_size))
        for state in range(self.n_states):
            r, c = self.state_coords[state]
            value_grid[r, c] = self.state_values[state]

        # Find local maxima (potential subgoals)
        local_maxima = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = value_grid[r, c]
                neighbors = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        neighbors.append(value_grid[nr, nc])

                if all(val > n for n in neighbors):
                    state = self.coord_to_state[(r, c)]
                    local_maxima.append(
                        {"state": state, "value": float(val), "position": (r, c)}
                    )

        features["local_maxima"] = local_maxima

        # Find ridges (where gradient changes)
        gy, gx = np.gradient(value_grid)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        features["avg_gradient"] = float(np.mean(gradient_magnitude))
        features["max_gradient"] = float(np.max(gradient_magnitude))

        # Compute value contours (simplified)
        from skimage import measure

        try:
            contours = measure.find_contours(value_grid, level=self.v_mean)
            features["n_contours_at_mean"] = len(contours)
        except:
            features["n_contours_at_mean"] = 0

        # Value range and distribution
        features["value_range"] = [float(self.v_min), float(self.v_max)]
        features["value_mean"] = float(self.v_mean)
        features["value_std"] = float(self.v_std)

        # Entropy of value distribution (discretized)
        hist, _ = np.histogram(self.state_values, bins=10)
        hist = hist / hist.sum()
        features["value_entropy"] = float(-np.sum(hist * np.log(hist + 1e-10)))

        return features

    def build_value_transition_graph(self, trajectories: List[Dict]) -> nx.DiGraph:
        """
        Build transition graph between value-based clusters
        """
        if not self.level_clusters:
            self.cluster_by_value_levels()

        G = nx.DiGraph()

        # Add nodes
        for cluster in self.level_clusters:
            G.add_node(
                cluster.id,
                level=cluster.level,
                states=cluster.states,
                size=cluster.size,
                value_range=cluster.value_range,
                is_high_value=cluster.is_high_value,
                centroid=cluster.centroid,
                boundary_states=cluster.boundary_states,
                avg_value=cluster.avg_value,
            )

        # Count transitions
        transitions = defaultdict(lambda: defaultdict(int))

        for traj in trajectories:
            for i in range(len(traj["states"]) - 1):
                s = traj["states"][i]
                s_next = traj["next_states"][i]

                if (
                    s in self.state_to_level_cluster
                    and s_next in self.state_to_level_cluster
                ):
                    c_from = self.state_to_level_cluster[s]
                    c_to = self.state_to_level_cluster[s_next]

                    if c_from != c_to:
                        transitions[c_from][c_to] += 1

        # Add edges
        for c_from, targets in transitions.items():
            total = sum(targets.values())
            for c_to, count in targets.items():
                prob = count / total
                G.add_edge(
                    int(c_from),
                    int(c_to),
                    probability=float(prob),
                    weight=float(prob * 3),
                    count=int(count),
                )

        return G

    def find_value_pathway(
        self, start_state: int = 0, goal_state: int = 15
    ) -> List[int]:
        """
        Find pathway through value clusters from start to goal
        """
        if not self.level_clusters:
            return []

        start_cluster = self.state_to_level_cluster.get(start_state)
        goal_cluster = self.state_to_level_cluster.get(goal_state)

        if start_cluster is None or goal_cluster is None:
            return []

        # Sort clusters by value level
        cluster_values = {c.id: c.avg_value for c in self.level_clusters}

        # Simple pathway: move through increasing value clusters
        current_cluster = start_cluster
        pathway = [current_cluster]

        visited = set([current_cluster])

        while current_cluster != goal_cluster and len(pathway) < len(
            self.level_clusters
        ):
            # Find neighbor with highest value
            neighbors = []
            for c in self.level_clusters:
                if c.id != current_cluster and c.id not in visited:
                    # Check if clusters are adjacent in grid space
                    # Simplified: check if any states are adjacent
                    states_current = set(self.level_clusters[current_cluster].states)
                    states_candidate = set(c.states)

                    adjacent = False
                    for s in states_current:
                        for n in self.get_neighbors(s):
                            if n in states_candidate:
                                adjacent = True
                                break
                        if adjacent:
                            break

                    if adjacent:
                        neighbors.append(c.id)

            if not neighbors:
                break

            # Move to neighbor with highest value
            next_cluster = max(neighbors, key=lambda x: cluster_values[x])
            visited.add(next_cluster)
            pathway.append(next_cluster)
            current_cluster = next_cluster

        return pathway if current_cluster == goal_cluster else []

    def visualize_value_clusters(self, save_path: Optional[str] = None):
        """
        Visualize value-based clusters
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Value heatmap
        ax = axes[0]
        value_grid = np.zeros((self.grid_size, self.grid_size))
        for state in range(self.n_states):
            r, c = self.state_coords[state]
            value_grid[r, c] = self.state_values[state]

        im = ax.imshow(
            value_grid, cmap="viridis", aspect="equal", vmin=self.v_min, vmax=self.v_max
        )
        ax.set_title(f"Value Function Heatmap\n{self.name}")
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))

        # Add value labels
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ax.text(
                    c,
                    r,
                    f"{value_grid[r, c]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value_grid[r, c] > self.v_mean else "black",
                    fontsize=8,
                )

        plt.colorbar(im, ax=ax)

        # Plot 2: Value level clusters
        ax = axes[1]
        if self.level_clusters:
            # Create grid coloring by cluster
            cluster_grid = np.full((self.grid_size, self.grid_size), -1)
            for cluster in self.level_clusters:
                for state in cluster.states:
                    r, c = self.state_coords[state]
                    cluster_grid[r, c] = cluster.id

            # Create colormap
            unique_clusters = np.unique(cluster_grid[cluster_grid >= 0])
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            color_map = {cid: colors[i] for i, cid in enumerate(unique_clusters)}

            # Draw grid
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if cluster_grid[r, c] >= 0:
                        cid = cluster_grid[r, c]
                        color = color_map[cid]
                        rect = plt.Rectangle(
                            (c, r),
                            1,
                            1,
                            facecolor=color,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=1,
                        )
                        ax.add_patch(rect)

                        # Add value level indicator
                        cluster = self.level_clusters[cid]
                        level_text = f"{cluster.level:.1f}\n({len(cluster.states)})"
                        ax.text(
                            c + 0.5,
                            r + 0.5,
                            level_text,
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

            ax.set_title(f"Value Level Clusters\n({len(self.level_clusters)} clusters)")
        else:
            ax.text(
                0.5,
                0.5,
                "No clusters",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        # Plot 3: Value topology (contours and maxima)
        ax = axes[2]

        # Create smoothed value grid for contours
        from scipy.ndimage import gaussian_filter

        smoothed = gaussian_filter(value_grid, sigma=0.5)

        # Draw contour lines
        levels = np.linspace(self.v_min, self.v_max, 8)
        contour = ax.contour(
            smoothed, levels=levels, colors="blue", alpha=0.5, linewidths=1
        )
        ax.clabel(contour, inline=True, fontsize=8)

        # Mark local maxima
        maxima = self.compute_value_topology_features()["local_maxima"]
        for m in maxima:
            r, c = m["position"]
            ax.plot(
                c,
                r,
                "r*",
                markersize=10,
                label="Local Max" if r == 0 and c == 0 else "",
            )
            ax.text(c, r - 0.2, f"{m['value']:.2f}", ha="center", fontsize=7)

        ax.set_title("Value Topology\n(Contours & Local Maxima)")
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        if maxima:
            ax.legend(loc="lower right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Value cluster visualization saved to {save_path}")

        plt.show()

    def compare_with_action_clusters(self, action_analyzer) -> Dict:
        """
        Compare value-based clusters with action-based clusters
        """
        comparison = {
            "cluster_counts": {
                "value_clusters": len(self.level_clusters),
                "action_clusters": len(action_analyzer.clusters),
            },
            "agreement": {},
            "value_composition": {},
            "action_composition": {},
        }

        # For each value cluster, see what actions are present
        for vc in self.level_clusters:
            actions_in_cluster = defaultdict(int)
            for state in vc.states:
                action = action_analyzer.agent.get_deterministic_policy()[state]
                actions_in_cluster[action] += 1

            total = len(vc.states)
            action_dist = {
                int(a): float(cnt / total) for a, cnt in actions_in_cluster.items()
            }

            comparison["action_composition"][f"VC{vc.id}"] = {
                "dominant_action": max(action_dist, key=action_dist.get)
                if action_dist
                else None,
                "action_distribution": action_dist,
                "value_level": vc.level,
                "is_high_value": vc.is_high_value,
            }

        # For each action cluster, see value statistics
        for ac in action_analyzer.clusters:
            values = [self.state_values[s] for s in ac.states]

            comparison["value_composition"][f"AC{ac.id}"] = {
                "action": ac.action,
                "mean_value": float(np.mean(values)),
                "std_value": float(np.std(values)),
                "min_value": float(np.min(values)),
                "max_value": float(np.max(values)),
            }

        # Calculate agreement (states where both clusterings agree on boundaries)
        # This is complex - simplified version:
        agreement_count = 0
        total_pairs = 0

        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                same_action = action_analyzer.state_to_cluster.get(
                    i
                ) == action_analyzer.state_to_cluster.get(j)
                same_value = self.state_to_level_cluster.get(
                    i
                ) == self.state_to_level_cluster.get(j)

                if same_action == same_value:
                    agreement_count += 1
                total_pairs += 1

        comparison["agreement"]["pairwise_coherence"] = (
            agreement_count / total_pairs if total_pairs > 0 else 0
        )

        return comparison

    def save_value_cluster_data(self, base_filename: str):
        """
        Save value cluster analysis data
        """
        os.makedirs("value_analysis", exist_ok=True)

        # Get features
        features = self.compute_value_topology_features()

        # Prepare data
        data = {
            "name": self.name,
            "value_statistics": {
                "min": float(self.v_min),
                "max": float(self.v_max),
                "mean": float(self.v_mean),
                "std": float(self.v_std),
            },
            "topology_features": features,
            "value_clusters": [c.to_dict() for c in self.level_clusters],
            "state_to_cluster": {
                str(k): int(v) for k, v in self.state_to_level_cluster.items()
            },
        }

        # Save
        with open(f"value_analysis/{base_filename}_value_clusters.json", "w") as f:
            json.dump(data, f, indent=2)

        print(
            f"Value cluster data saved to value_analysis/{base_filename}_value_clusters.json"
        )


class EnhancedClusterAnalyzer:
    """
    Combined analyzer that integrates both action-based and value-based clustering
    """

    def __init__(self, agent, name: str = "policy"):
        self.agent = agent
        self.name = name

        # Initialize both analyzers
        self.action_analyzer = ClusterAnalyzer(agent, f"{name}_action")
        self.value_analyzer = ValueClusterAnalyzer(agent, f"{name}_value")

    def analyze_all(self, trajectories: List[Dict], n_value_levels: int = 5):
        """
        Perform comprehensive analysis using both methods
        """
        # Action-based clustering
        print(f"\nAnalyzing action-based clusters for {self.name}...")
        self.action_analyzer.find_connected_clusters()
        self.action_analyzer.build_transition_graph(trajectories)

        # Value-based clustering
        print(f"Analyzing value-based clusters for {self.name}...")
        self.value_analyzer.cluster_by_value_levels(
            n_levels=n_value_levels, use_superlevel=True
        )
        self.value_analyzer.build_value_transition_graph(trajectories)

        # Compare approaches
        comparison = self.value_analyzer.compare_with_action_clusters(
            self.action_analyzer
        )

        return comparison

    def visualize_comprehensive(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization comparing both approaches
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1: Action-based clustering
        # Column 1: Action cluster map
        ax = axes[0, 0]
        self._plot_action_cluster_map(ax)
        ax.set_title(
            f"{self.name}: Action Clusters\n({len(self.action_analyzer.clusters)} clusters)"
        )

        # Column 2: Action transition graph
        ax = axes[0, 1]
        self._plot_action_graph(ax)
        ax.set_title("Action Cluster Graph")

        # Column 3: Action cluster sizes
        ax = axes[0, 2]
        sizes = [c.size for c in self.action_analyzer.clusters]
        actions = [c.action for c in self.action_analyzer.clusters]
        colors = [plt.cm.tab10(a / 4) for a in actions]
        ax.bar(range(len(sizes)), sizes, color=colors)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Size")
        ax.set_title("Action Cluster Sizes")

        # Row 2: Value-based clustering
        # Column 1: Value cluster map
        ax = axes[1, 0]
        self._plot_value_cluster_map(ax)
        ax.set_title(
            f"Value Clusters\n({len(self.value_analyzer.level_clusters)} clusters)"
        )

        # Column 2: Value transition graph
        ax = axes[1, 1]
        self._plot_value_graph(ax)
        ax.set_title("Value Cluster Graph")

        # Column 3: Value levels
        ax = axes[1, 2]
        levels = [c.level for c in self.value_analyzer.level_clusters]
        sizes = [c.size for c in self.value_analyzer.level_clusters]
        colors = [
            "red" if c.is_high_value else "blue"
            for c in self.value_analyzer.level_clusters
        ]
        scatter = ax.scatter(levels, sizes, c=colors, s=100, alpha=0.6)
        ax.set_xlabel("Value Level")
        ax.set_ylabel("Cluster Size")
        ax.set_title("Value Clusters: Level vs Size")

        # Add legend for high/low value
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", alpha=0.6, label="High Value (≥ level)"),
            Patch(facecolor="blue", alpha=0.6, label="Low Value (< level)"),
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Comprehensive visualization saved to {save_path}")

        plt.show()

    def _plot_action_cluster_map(self, ax):
        """Helper to plot action cluster map"""
        grid = np.full((self.agent.grid_size, self.agent.grid_size), -1)
        action_grid = np.full((self.agent.grid_size, self.agent.grid_size), -1)

        for cluster in self.action_analyzer.clusters:
            for state in cluster.states:
                r, c = self.agent.state_coords[state]
                grid[r, c] = cluster.id
                action_grid[r, c] = cluster.action

        # Create colormap
        unique_clusters = np.unique(grid[grid >= 0])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        color_map = {cid: colors[i] for i, cid in enumerate(unique_clusters)}

        # Draw grid
        for r in range(self.agent.grid_size):
            for c in range(self.agent.grid_size):
                if grid[r, c] >= 0:
                    color = color_map[grid[r, c]]
                    rect = plt.Rectangle(
                        (c, r),
                        1,
                        1,
                        facecolor=color,
                        alpha=0.6,
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax.add_patch(rect)

                    # Add action symbol
                    action = action_grid[r, c]
                    ax.text(
                        c + 0.5,
                        r + 0.5,
                        self.agent.action_symbols[action],
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

        ax.set_xlim(0, self.agent.grid_size)
        ax.set_ylim(0, self.agent.grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def _plot_value_cluster_map(self, ax):
        """Helper to plot value cluster map"""
        if not self.value_analyzer.level_clusters:
            ax.text(
                0.5,
                0.5,
                "No value clusters",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Create grid coloring by cluster
        cluster_grid = np.full((self.agent.grid_size, self.agent.grid_size), -1)
        for cluster in self.value_analyzer.level_clusters:
            for state in cluster.states:
                r, c = self.agent.state_coords[state]
                cluster_grid[r, c] = cluster.id

        # Create colormap
        unique_clusters = np.unique(cluster_grid[cluster_grid >= 0])
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        color_map = {cid: colors[i] for i, cid in enumerate(unique_clusters)}

        # Draw grid
        for r in range(self.agent.grid_size):
            for c in range(self.agent.grid_size):
                if cluster_grid[r, c] >= 0:
                    cid = cluster_grid[r, c]
                    color = color_map[cid]
                    rect = plt.Rectangle(
                        (c, r),
                        1,
                        1,
                        facecolor=color,
                        alpha=0.7,
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax.add_patch(rect)

                    # Add level indicator
                    cluster = self.value_analyzer.level_clusters[cid]
                    level_text = f"{cluster.level:.2f}"
                    ax.text(
                        c + 0.5,
                        r + 0.5,
                        level_text,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        ax.set_xlim(0, self.agent.grid_size)
        ax.set_ylim(0, self.agent.grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def _plot_action_graph(self, ax):
        """Helper to plot action cluster graph"""
        if not self.action_analyzer.transition_graph:
            ax.text(
                0.5, 0.5, "No graph", ha="center", va="center", transform=ax.transAxes
            )
            return

        G = self.action_analyzer.transition_graph
        pos = {
            node: (data["centroid"][1], -data["centroid"][0])
            for node, data in G.nodes(data=True)
        }

        # Draw nodes
        for node in G.nodes():
            data = G.nodes[node]
            size = data["size"] * 500

            nx.draw_networkx_nodes(
                G,
                pos,
                [node],
                node_size=size,
                node_color="lightblue",
                edgecolors="black",
                linewidths=2,
                alpha=0.7,
                ax=ax,
            )

            ax.text(
                pos[node][0],
                pos[node][1],
                f"C{node}",
                ha="center",
                va="center",
                fontsize=8,
            )

        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            width=1,
            edge_color="gray",
            alpha=0.5,
            arrows=True,
            arrowsize=15,
            ax=ax,
        )

        ax.axis("off")

    def _plot_value_graph(self, ax):
        """Helper to plot value cluster graph"""
        # Similar to action graph but for value clusters
        ax.text(
            0.5, 0.5, "Value graph", ha="center", va="center", transform=ax.transAxes
        )
        ax.axis("off")

    def save_all_analysis(self, base_filename: str):
        """
        Save all analysis data
        """
        # Save action-based analysis
        self.action_analyzer.save_cluster_data(f"{base_filename}_action")

        # Save value-based analysis
        self.value_analyzer.save_value_cluster_data(base_filename)

        # Save comparison
        comparison = self.value_analyzer.compare_with_action_clusters(
            self.action_analyzer
        )

        with open(f"value_analysis/{base_filename}_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"All analysis saved for {base_filename}")


# Enhanced comparison experiment
def run_enhanced_comparison_experiment():
    """
    Run comprehensive comparison with value-based analysis
    """
    print("\n" + "=" * 80)
    print("ENHANCED COMPARISON: SARSA vs MONTE CARLO with VALUE TOPOLOGY ANALYSIS")
    print("=" * 80)

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("cluster_analysis", exist_ok=True)
    os.makedirs("value_analysis", exist_ok=True)

    results = {}

    # 1. Train SARSA agent
    print("\n" + "-" * 80)
    print("TRAINING SARSA AGENT")
    print("-" * 80)

    sarsa_agent = FrozenLakeSarsa(
        map_name="4x4", is_slippery=True, learning_rate=0.1, episodes=10000
    )
    sarsa_results = sarsa_agent.train(verbose=True)
    sarsa_agent.save_model("models/sarsa_frozenlake.pkl")

    # 2. Train Monte Carlo agent
    print("\n" + "-" * 80)
    print("TRAINING MONTE CARLO AGENT")
    print("-" * 80)

    mc_agent = FrozenLakeMonteCarlo(
        map_name="4x4", is_slippery=True, first_visit=True, episodes=10000
    )
    mc_results = mc_agent.train(verbose=True)
    mc_agent.save_model("models/mc_frozenlake.pkl")

    # 3. Collect trajectories
    print("\n" + "-" * 80)
    print("COLLECTING TRAJECTORIES")
    print("-" * 80)

    # SARSA trajectories
    sarsa_analyzer = ClusterAnalyzer(sarsa_agent, "SARSA")
    sarsa_trajs = sarsa_analyzer.collect_trajectories(n_trajectories=500)

    # MC trajectories
    mc_analyzer = ClusterAnalyzer(mc_agent, "MonteCarlo")
    mc_trajs = mc_analyzer.collect_trajectories(n_trajectories=500)

    # 4. Enhanced analysis for SARSA
    print("\n" + "-" * 80)
    print("ENHANCED ANALYSIS FOR SARSA")
    print("-" * 80)

    sarsa_enhanced = EnhancedClusterAnalyzer(sarsa_agent, "SARSA")
    sarsa_comparison = sarsa_enhanced.analyze_all(sarsa_trajs, n_value_levels=5)
    sarsa_enhanced.visualize_comprehensive(
        save_path="value_analysis/sarsa_comprehensive.png"
    )
    sarsa_enhanced.save_all_analysis("sarsa")

    # 5. Enhanced analysis for Monte Carlo
    print("\n" + "-" * 80)
    print("ENHANCED ANALYSIS FOR MONTE CARLO")
    print("-" * 80)

    mc_enhanced = EnhancedClusterAnalyzer(mc_agent, "MonteCarlo")
    mc_comparison = mc_enhanced.analyze_all(mc_trajs, n_value_levels=5)
    mc_enhanced.visualize_comprehensive(save_path="value_analysis/mc_comprehensive.png")
    mc_enhanced.save_all_analysis("mc")

    # 6. Value topology features
    print("\n" + "-" * 80)
    print("VALUE TOPOLOGY FEATURES")
    print("-" * 80)

    sarsa_value_features = (
        sarsa_enhanced.value_analyzer.compute_value_topology_features()
    )
    mc_value_features = mc_enhanced.value_analyzer.compute_value_topology_features()

    print("\nSARSA Value Topology:")
    for k, v in sarsa_value_features.items():
        print(f"  {k}: {v}")

    print("\nMonte Carlo Value Topology:")
    for k, v in mc_value_features.items():
        print(f"  {k}: {v}")

    # 7. Comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'SARSA':>15} {'Monte Carlo':>15}")
    print("-" * 62)

    # Basic metrics
    print(
        f"{'Success Rate':<30} {sarsa_results['final_success_rate']:>15.3f} {mc_results['final_success_rate']:>15.3f}"
    )
    print(
        f"{'Training Time (s)':<30} {sarsa_results['training_time']:>15.2f} {mc_results['training_time']:>15.2f}"
    )

    # Action clusters
    print(
        f"{'Action Clusters':<30} {len(sarsa_enhanced.action_analyzer.clusters):>15} {len(mc_enhanced.action_analyzer.clusters):>15}"
    )

    # Value clusters
    print(
        f"{'Value Clusters':<30} {len(sarsa_enhanced.value_analyzer.level_clusters):>15} {len(mc_enhanced.value_analyzer.level_clusters):>15}"
    )

    # Value statistics
    print(
        f"{'Value Range':<30} {sarsa_value_features['value_range'][1] - sarsa_value_features['value_range'][0]:>15.3f} {mc_value_features['value_range'][1] - mc_value_features['value_range'][0]:>15.3f}"
    )
    print(
        f"{'Value Entropy':<30} {sarsa_value_features['value_entropy']:>15.3f} {mc_value_features['value_entropy']:>15.3f}"
    )

    # Local maxima
    print(
        f"{'Local Maxima':<30} {len(sarsa_value_features['local_maxima']):>15} {len(mc_value_features['local_maxima']):>15}"
    )

    # Cluster coherence
    print(
        f"{'Action-Value Coherence':<30} {sarsa_comparison['agreement']['pairwise_coherence']:>15.3f} {mc_comparison['agreement']['pairwise_coherence']:>15.3f}"
    )

    # 8. Generate report
    report = f"""
ENHANCED CLUSTER ANALYSIS REPORT
{"=" * 80}

Environment: FrozenLake-v1 (4x4, slippery=True)

ALGORITHM COMPARISON
{"-" * 80}

SARSA:
  - Success Rate: {sarsa_results["final_success_rate"]:.3f}
  - Training Time: {sarsa_results["training_time"]:.2f}s
  - Action Clusters: {len(sarsa_enhanced.action_analyzer.clusters)}
  - Value Clusters: {len(sarsa_enhanced.value_analyzer.level_clusters)}
  - Value Range: [{sarsa_value_features["value_range"][0]:.3f}, {sarsa_value_features["value_range"][1]:.3f}]
  - Value Entropy: {sarsa_value_features["value_entropy"]:.3f}
  - Local Maxima: {len(sarsa_value_features["local_maxima"])}
  
Monte Carlo:
  - Success Rate: {mc_results["final_success_rate"]:.3f}
  - Training Time: {mc_results["training_time"]:.2f}s
  - Action Clusters: {len(mc_enhanced.action_analyzer.clusters)}
  - Value Clusters: {len(mc_enhanced.value_analyzer.level_clusters)}
  - Value Range: [{mc_value_features["value_range"][0]:.3f}, {mc_value_features["value_range"][1]:.3f}]
  - Value Entropy: {mc_value_features["value_entropy"]:.3f}
  - Local Maxima: {len(mc_value_features["local_maxima"])}

VALUE TOPOLOGY INSIGHTS
{"-" * 80}

The value function landscape reveals:
1. Local maxima indicate potential subgoal states
2. Value contours show the gradient of state importance
3. Cluster coherence measures how well action preferences align with value levels

Files Generated:
  - models/sarsa_frozenlake.pkl
  - models/mc_frozenlake.pkl
  - cluster_analysis/* (action-based clusters)
  - value_analysis/* (value-based clusters)
  - value_analysis/*_comprehensive.png (visualizations)
"""

    with open("value_analysis/comprehensive_report.txt", "w") as f:
        f.write(report)

    print(report)

    return {
        "sarsa": (sarsa_agent, sarsa_enhanced),
        "mc": (mc_agent, mc_enhanced),
        "comparison": {
            "sarsa_comparison": sarsa_comparison,
            "mc_comparison": mc_comparison,
            "sarsa_features": sarsa_value_features,
            "mc_features": mc_value_features,
        },
    }


if __name__ == "__main__":
    # Run enhanced comparison
    results = run_enhanced_comparison_experiment()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(
        "\nCheck the 'value_analysis/' directory for detailed results and visualizations."
    )
