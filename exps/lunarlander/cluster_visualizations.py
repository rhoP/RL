import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from typing import Dict, List, Optional
import os
from collections import defaultdict
import warnings
import cluster
from stable_baselines3 import PPO, A2C

warnings.filterwarnings("ignore")


class SingleAgentClusterVisualizer:
    """
    Visualize clusters from a single agent (PPO or A2C)
    """

    def __init__(
        self, env, agent, analyzer, agent_name: str = "PPO", save_dir: str = "lunar_viz"
    ):
        self.env = env
        self.agent = agent
        self.analyzer = analyzer
        self.agent_name = agent_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Lunar Lander visualization constants
        self.SCALE = 30.0
        self.VIEWPORT_W = 600
        self.VIEWPORT_H = 400

        # Colors
        self.terrain_color = (0.4, 0.2, 0.0)
        self.sky_color = (0.2, 0.2, 0.4)
        self.lander_color = (0.8, 0.8, 0.8)
        self.flame_color = (1.0, 0.5, 0.0)

        # Action colors
        self.action_colors = {
            0: (0.8, 0.2, 0.2, 0.6),  # Red - do nothing
            1: (0.2, 0.8, 0.2, 0.6),  # Green - left engine
            2: (0.2, 0.2, 0.8, 0.6),  # Blue - main engine
            3: (0.8, 0.8, 0.2, 0.6),  # Yellow - right engine
        }

        # Get clusters from analyzer
        self.clusters = analyzer.clusters

        # Create cluster_labels array from the clusters
        # We need to know which state index belongs to which cluster
        # The analyzer's state_to_cluster dictionary maps state indices to cluster ids
        if hasattr(analyzer, "state_to_cluster"):
            self.state_to_cluster = analyzer.state_to_cluster
            # Create labels array for all states
            n_states = (
                len(analyzer.collected_states)
                if hasattr(analyzer, "collected_states")
                else 0
            )
            self.cluster_labels = np.full(n_states, -1)
            for state_idx, cluster_id in self.state_to_cluster.items():
                if state_idx < n_states:
                    self.cluster_labels[state_idx] = cluster_id
        else:
            # If no state_to_cluster mapping, we'll create one on the fly
            self.state_to_cluster = {}
            self.cluster_labels = (
                np.full(len(analyzer.collected_states), -1)
                if hasattr(analyzer, "collected_states")
                else np.array([])
            )

            # For each cluster, mark its state indices
            for cluster in self.clusters:
                if hasattr(cluster, "indices") and cluster.indices is not None:
                    for idx in cluster.indices:
                        self.state_to_cluster[idx] = cluster.id
                        if idx < len(self.cluster_labels):
                            self.cluster_labels[idx] = cluster.id
                elif hasattr(cluster, "states") and cluster.states is not None:
                    # If cluster stores actual state vectors, we need to find which indices they correspond to
                    # This is more complex - we'll handle it in visualization methods
                    pass

        self.states = (
            analyzer.collected_states if hasattr(analyzer, "collected_states") else None
        )
        self.actions = (
            analyzer.collected_actions
            if hasattr(analyzer, "collected_actions")
            else None
        )
        self.values = (
            analyzer.collected_values if hasattr(analyzer, "collected_values") else None
        )

        print(f"Initialized visualizer with {len(self.clusters)} clusters")
        if self.states is not None:
            print(f"  States shape: {self.states.shape}")
            print(
                f"  Cluster labels range: [{np.min(self.cluster_labels)}, {np.max(self.cluster_labels)}]"
            )

    def visualize_xy_clusters(self, save_name: str = None):
        """
        Visualize clusters in X-Y position space
        """
        if save_name is None:
            save_name = f"{self.agent_name.lower()}_xy_clusters.png"

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Create base environment for each subplot
        for ax in axes:
            self._draw_environment_background(ax)

        # Extract X,Y positions
        x_pos = self.states[:, 0]
        y_pos = self.states[:, 1]

        # Plot 1: States colored by cluster
        ax = axes[0]
        n_clusters = len(self.clusters)
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

        for i, cluster in enumerate(self.clusters):
            mask = self.cluster_labels == cluster.id
            ax.scatter(
                x_pos[mask],
                y_pos[mask],
                c=[cluster_colors[i % len(cluster_colors)]],
                s=10,
                alpha=0.6,
                label=f"C{cluster.id}" if i < 5 else "",
            )

        ax.set_title(
            f"{self.agent_name}: X-Y Position by Cluster\n({n_clusters} clusters)"
        )
        if n_clusters <= 5:
            ax.legend()

        # Plot 2: States colored by action
        ax = axes[1]
        for action in range(4):
            mask = self.actions == action
            color = self.action_colors[action][:3]
            action_name = ["None", "Left", "Main", "Right"][action]
            ax.scatter(
                x_pos[mask], y_pos[mask], c=[color], s=10, alpha=0.6, label=action_name
            )

        ax.set_title(f"{self.agent_name}: X-Y Position by Action")
        ax.legend()

        # Plot 3: States colored by value
        ax = axes[2]
        scatter = ax.scatter(
            x_pos,
            y_pos,
            c=self.values,
            cmap="viridis",
            s=10,
            alpha=0.6,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"{self.agent_name}: X-Y Position by Value")
        plt.colorbar(scatter, ax=ax, label="State Value")

        plt.suptitle(f"{self.agent_name} Cluster Analysis - X-Y Space", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved to {self.save_dir}/{save_name}")

    def visualize_cluster_gallery(
        self,
        n_clusters_to_show: int = 6,
        samples_per_cluster: int = 4,
        save_name: str = None,
    ):
        """
        Create a gallery showing representative states from each cluster
        """
        if save_name is None:
            save_name = f"{self.agent_name.lower()}_cluster_gallery.png"

        # Select largest clusters to show
        clusters_sorted = sorted(self.clusters, key=lambda c: c.size, reverse=True)
        clusters_to_show = clusters_sorted[:n_clusters_to_show]

        # Create figure
        n_rows = len(clusters_to_show)
        n_cols = samples_per_cluster + 1  # +1 for cluster info
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row, cluster in enumerate(clusters_to_show):
            # Column 0: Cluster info
            ax = axes[row, 0]
            ax.axis("off")

            # Get states in this cluster
            cluster_mask = self.cluster_labels == cluster.id
            cluster_states = self.states[cluster_mask]
            cluster_actions = self.actions[cluster_mask]

            # Calculate action distribution
            unique_actions, counts = np.unique(cluster_actions, return_counts=True)
            action_dist = {
                a: c / len(cluster_actions) for a, c in zip(unique_actions, counts)
            }

            # Display cluster statistics
            info_text = f"Cluster {cluster.id}\n"
            info_text += f"Size: {cluster.size}\n"
            info_text += f"Dominant: {self._action_name(cluster.dominant_action)}\n"
            info_text += f"Value: μ={cluster.value_mean:.2f} ±{cluster.value_std:.2f}\n"
            info_text += f"Spread: {cluster.spread:.2f}\n\n"
            info_text += "Action distribution:\n"

            for a in range(4):
                pct = action_dist.get(a, 0) * 100
                info_text += f"  {self._action_name(a)}: {pct:.0f}%\n"

            ax.text(
                0.1,
                0.5,
                info_text,
                transform=ax.transAxes,
                verticalalignment="center",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

            # Columns 1+: Sample states from this cluster
            n_samples = min(samples_per_cluster, len(cluster_states))
            sample_indices = np.random.choice(
                len(cluster_states), n_samples, replace=False
            )

            for col, idx in enumerate(sample_indices):
                ax = axes[row, col + 1]
                state = cluster_states[idx]
                action = cluster_actions[idx]

                self._draw_lander_state(
                    ax,
                    state,
                    title=f"Action: {self._action_name(action)}",
                    show_info=False,
                )

        plt.suptitle(f"{self.agent_name} Cluster Gallery", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved to {self.save_dir}/{save_name}")

    def visualize_contact_states(self, save_name: str = None):
        """
        Visualize how clusters correspond to leg contact states
        """
        if save_name is None:
            save_name = f"{self.agent_name.lower()}_contact_states.png"

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Contact states: left_contact (index 6), right_contact (index 7)
        left_contact = self.states[:, 6] > 0.5
        right_contact = self.states[:, 7] > 0.5

        contact_modes = [
            (left_contact & right_contact, "Both Legs", "green"),
            (left_contact & ~right_contact, "Left Only", "blue"),
            (~left_contact & right_contact, "Right Only", "orange"),
            (~left_contact & ~right_contact, "No Contact", "red"),
        ]

        # Plot 1: Contact states in X-Y space
        ax = axes[0, 0]
        self._draw_environment_background(ax)

        for mask, label, color in contact_modes:
            if np.any(mask):
                ax.scatter(
                    self.states[mask, 0],
                    self.states[mask, 1],
                    c=color,
                    label=label,
                    alpha=0.5,
                    s=10,
                )

        ax.set_title(f"{self.agent_name}: Contact States in X-Y Space")
        ax.legend()

        # Plot 2: Contact distribution across clusters
        ax = axes[0, 1]

        cluster_ids = []
        contact_data = []

        for cluster in self.clusters:
            mask = self.cluster_labels == cluster.id
            if np.any(mask):
                left_rate = np.mean(left_contact[mask])
                right_rate = np.mean(right_contact[mask])
                both_rate = np.mean(left_contact[mask] & right_contact[mask])
                none_rate = np.mean(~left_contact[mask] & ~right_contact[mask])

                cluster_ids.append(cluster.id)
                contact_data.append(
                    {
                        "none": none_rate,
                        "left": left_rate - both_rate,
                        "right": right_rate - both_rate,
                        "both": both_rate,
                    }
                )

        # Stacked bar chart
        bottom = np.zeros(len(cluster_ids))
        for contact_type, color, label in [
            ("none", "red", "No Contact"),
            ("left", "blue", "Left Only"),
            ("right", "orange", "Right Only"),
            ("both", "green", "Both"),
        ]:
            values = [d[contact_type] for d in contact_data]
            bars = ax.bar(
                cluster_ids, values, bottom=bottom, color=color, alpha=0.7, label=label
            )
            bottom += values

        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Proportion")
        ax.set_title("Contact State Distribution by Cluster")
        ax.legend()

        # Plot 3: Cluster sizes
        ax = axes[0, 2]
        sizes = [c.size for c in self.clusters]
        ax.bar(range(len(sizes)), sizes, color="steelblue", alpha=0.7)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Size")
        ax.set_title("Cluster Sizes")

        # Plot 4: Example of no-contact state
        ax = axes[1, 0]
        no_contact_mask = ~left_contact & ~right_contact
        if np.any(no_contact_mask):
            idx = np.random.choice(np.where(no_contact_mask)[0])
            self._draw_lander_state(ax, self.states[idx], title="No Contact (Flying)")

        # Plot 5: Example of left contact
        ax = axes[1, 1]
        left_only_mask = left_contact & ~right_contact
        if np.any(left_only_mask):
            idx = np.random.choice(np.where(left_only_mask)[0])
            self._draw_lander_state(ax, self.states[idx], title="Left Leg Contact")

        # Plot 6: Example of both legs contact
        ax = axes[1, 2]
        both_mask = left_contact & right_contact
        if np.any(both_mask):
            idx = np.random.choice(np.where(both_mask)[0])
            self._draw_lander_state(
                ax, self.states[idx], title="Both Legs Contact (Landed)"
            )

        plt.suptitle(f"{self.agent_name}: Leg Contact Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved to {self.save_dir}/{save_name}")

    def visualize_cluster_transitions(
        self, trajectory_idx: int = 0, save_name: str = None
    ):
        """
        Visualize a trajectory through cluster space
        """
        if save_name is None:
            save_name = f"{self.agent_name.lower()}_transitions.png"

        if (
            not hasattr(self.analyzer, "collected_trajectories")
            or not self.analyzer.collected_trajectories
        ):
            print("No trajectories available")
            return

        traj = self.analyzer.collected_trajectories[trajectory_idx]
        states = np.array(traj["states"])
        actions = np.array(traj["actions"])

        # Determine cluster for each state
        cluster_sequence = []
        for state in states:
            # Find closest cluster centroid
            min_dist = float("inf")
            closest = None
            for cluster in self.clusters:
                dist = np.linalg.norm(state - cluster.centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest = cluster.id
            cluster_sequence.append(closest)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Trajectory in X-Y space
        ax = axes[0, 0]
        self._draw_environment_background(ax)

        # Plot trajectory line
        ax.plot(states[:, 0], states[:, 1], "b-", alpha=0.5, linewidth=2)

        # Mark points by cluster
        unique_clusters = list(set(cluster_sequence))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        color_map = {c: colors[i] for i, c in enumerate(unique_clusters)}

        for i, (state, cluster) in enumerate(zip(states, cluster_sequence)):
            ax.scatter(
                state[0],
                state[1],
                c=[color_map[cluster]],
                s=50,
                alpha=0.7,
                edgecolors="black",
                linewidth=1,
            )

        ax.set_title(f"Trajectory through Cluster Space\nStart → End")

        # Plot 2: Cluster membership over time
        ax = axes[0, 1]
        ax.plot(cluster_sequence, "b-", linewidth=2, alpha=0.7)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cluster ID")
        ax.set_title("Cluster Transitions Over Time")
        ax.grid(True, alpha=0.3)

        # Plot 3: Actions over time
        ax = axes[0, 2]
        ax.plot(actions, "r-", linewidth=2, alpha=0.7)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action")
        ax.set_yticks(range(4))
        ax.set_yticklabels(["None", "Left", "Main", "Right"])
        ax.set_title("Actions Over Time")
        ax.grid(True, alpha=0.3)

        # Plot 4: Start state
        ax = axes[1, 0]
        self._draw_lander_state(ax, states[0], title=f"Start (C{cluster_sequence[0]})")

        # Plot 5: Middle state
        ax = axes[1, 1]
        mid_idx = len(states) // 2
        self._draw_lander_state(
            ax, states[mid_idx], title=f"Middle (C{cluster_sequence[mid_idx]})"
        )

        # Plot 6: End state
        ax = axes[1, 2]
        self._draw_lander_state(ax, states[-1], title=f"End (C{cluster_sequence[-1]})")

        plt.suptitle(f"{self.agent_name}: Trajectory Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved to {self.save_dir}/{save_name}")

    def visualize_value_landscape(self, save_name: str = None):
        """
        Visualize the value function landscape across clusters
        """
        if save_name is None:
            save_name = f"{self.agent_name.lower()}_value_landscape.png"

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Value by X-Y position
        ax = axes[0]
        self._draw_environment_background(ax)

        scatter = ax.scatter(
            self.states[:, 0],
            self.states[:, 1],
            c=self.values,
            cmap="viridis",
            s=10,
            alpha=0.6,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"{self.agent_name}: Value Landscape")
        plt.colorbar(scatter, ax=ax, label="State Value")

        # Plot 2: Value distribution by cluster
        ax = axes[1]

        cluster_values = []
        cluster_ids = []
        for cluster in self.clusters:
            mask = self.cluster_labels == cluster.id
            if np.any(mask):
                cluster_values.append(self.values[mask])
                cluster_ids.append(cluster.id)

        # Box plot
        bp = ax.boxplot(cluster_values, positions=cluster_ids, widths=0.6)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("State Value")
        ax.set_title("Value Distribution by Cluster")
        ax.grid(True, alpha=0.3)

        # Plot 3: Highest and lowest value states
        ax = axes[2]

        # Find highest value state
        high_idx = np.argmax(self.values)
        low_idx = np.argmin(self.values)

        # Create subplots for high and low
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(2, 1, hspace=0.5)
        ax.remove()

        ax_high = fig.add_subplot(gs[0, 0])
        self._draw_lander_state(
            ax_high,
            self.states[high_idx],
            title=f"Highest Value: {self.values[high_idx]:.2f}",
        )

        ax_low = fig.add_subplot(gs[1, 0])
        self._draw_lander_state(
            ax_low,
            self.states[low_idx],
            title=f"Lowest Value: {self.values[low_idx]:.2f}",
        )

        plt.suptitle(f"{self.agent_name}: Value Function Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved to {self.save_dir}/{save_name}")

    def create_dashboard(self, save_name: str = None):
        """
        Create a comprehensive dashboard with all visualizations
        """
        if save_name is None:
            save_name = f"{self.agent_name.lower()}_dashboard.png"

        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. X-Y Cluster Map (top left, spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self._draw_environment_background(ax1)

        # Color by cluster
        n_clusters = len(self.clusters)
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

        for i, cluster in enumerate(self.clusters):
            mask = self.cluster_labels == cluster.id
            ax1.scatter(
                self.states[mask, 0],
                self.states[mask, 1],
                c=[cluster_colors[i % len(cluster_colors)]],
                s=5,
                alpha=0.6,
            )

        ax1.set_title(f"{self.agent_name}: X-Y Cluster Map")

        # 2. Cluster Size Distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        sizes = [c.size for c in self.clusters]
        ax2.bar(range(len(sizes)), sizes, color="steelblue", alpha=0.7)
        ax2.set_xlabel("Cluster ID")
        ax2.set_ylabel("Size")
        ax2.set_title("Cluster Sizes")

        # 3. Action Distribution (middle right)
        ax3 = fig.add_subplot(gs[1, 2])
        action_counts = [np.sum(self.actions == a) for a in range(4)]
        colors = [self.action_colors[a][:3] for a in range(4)]
        ax3.pie(
            action_counts,
            labels=["None", "Left", "Main", "Right"],
            colors=colors,
            autopct="%1.0f%%",
        )
        ax3.set_title("Overall Action Distribution")

        # 4. Value Distribution (bottom right)
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.hist(self.values, bins=30, color="purple", alpha=0.7)
        ax4.set_xlabel("State Value")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Value Distribution")

        # 5. Sample states from different clusters (bottom row)
        for i, cluster in enumerate(self.clusters[:4]):  # First 4 clusters
            ax = fig.add_subplot(gs[2, i])
            mask = self.cluster_labels == cluster.id
            if np.any(mask):
                idx = np.random.choice(np.where(mask)[0])
                self._draw_lander_state(
                    ax,
                    self.states[idx],
                    title=f"C{cluster.id}\n{self._action_name(cluster.dominant_action)}",
                    show_info=False,
                )

        plt.suptitle(
            f"{self.agent_name} Cluster Analysis Dashboard", fontsize=16, y=1.02
        )
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved to {self.save_dir}/{save_name}")

    def _draw_environment_background(self, ax):
        """Draw the Lunar Lander environment background"""
        # Sky
        ax.add_patch(Rectangle((-1, -0.5), 2, 2, facecolor=(0.2, 0.2, 0.4), alpha=0.3))

        # Ground/terrain
        ax.add_patch(
            Rectangle((-1, -0.5), 2, 0.3, facecolor=(0.4, 0.2, 0.0), alpha=0.5)
        )

        # Landing pads
        ax.add_patch(Rectangle((-0.3, -0.5), 0.2, 0.05, facecolor="yellow", alpha=0.8))
        ax.add_patch(Rectangle((0.1, -0.5), 0.2, 0.05, facecolor="yellow", alpha=0.8))

        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, alpha=0.3)

    def _draw_lander_state(self, ax, state, title="", show_info=True):
        """Draw the lunar lander in a specific state"""
        if state is None:
            ax.text(
                0.5, 0.5, "No state", ha="center", va="center", transform=ax.transAxes
            )
            return

        x, y, vx, vy, angle, ang_vel, left_contact, right_contact = state

        ax.clear()
        self._draw_environment_background(ax)

        # Lander body
        body_length = 0.3
        body_width = 0.2

        def rotate_point(px, py, cx, cy, angle):
            s = np.sin(angle)
            c = np.cos(angle)
            px -= cx
            py -= cy
            xnew = px * c - py * s
            ynew = px * s + py * c
            return xnew + cx, ynew + cy

        # Draw body
        body_points = [
            (x, y + body_length / 2),
            (x - body_width / 2, y - body_length / 2),
            (x + body_width / 2, y - body_length / 2),
        ]

        rotated_body = [rotate_point(px, py, x, y, angle) for px, py in body_points]
        body_poly = Polygon(
            rotated_body, facecolor=(0.8, 0.8, 0.8), edgecolor="black", linewidth=1
        )
        ax.add_patch(body_poly)

        # Draw legs
        leg_length = 0.15

        # Left leg
        leg_start = rotate_point(x - body_width / 2, y - body_length / 2, x, y, angle)
        leg_end = rotate_point(
            x - body_width / 2 - leg_length,
            y - body_length / 2 - leg_length,
            x,
            y,
            angle,
        )
        leg_color = "green" if left_contact > 0.5 else "gray"
        ax.plot(
            [leg_start[0], leg_end[0]],
            [leg_start[1], leg_end[1]],
            color=leg_color,
            linewidth=2,
        )

        # Right leg
        leg_start = rotate_point(x + body_width / 2, y - body_length / 2, x, y, angle)
        leg_end = rotate_point(
            x + body_width / 2 + leg_length,
            y - body_length / 2 - leg_length,
            x,
            y,
            angle,
        )
        leg_color = "green" if right_contact > 0.5 else "gray"
        ax.plot(
            [leg_start[0], leg_end[0]],
            [leg_start[1], leg_end[1]],
            color=leg_color,
            linewidth=2,
        )

        # Draw engine flames if velocity indicates firing
        if abs(vx) > 0.1 or abs(vy) > 0.1:
            flame_points = [
                (x - body_width / 4, y - body_length / 2),
                (x + body_width / 4, y - body_length / 2),
                (x, y - body_length / 2 - 0.1),
            ]
            rotated_flame = [
                rotate_point(px, py, x, y, angle) for px, py in flame_points
            ]
            flame_poly = Polygon(rotated_flame, facecolor=(1.0, 0.5, 0.0), alpha=0.7)
            ax.add_patch(flame_poly)

        # Draw velocity vector
        ax.arrow(
            x,
            y,
            vx * 0.5,
            vy * 0.5,
            head_width=0.03,
            head_length=0.03,
            fc="blue",
            ec="blue",
            alpha=0.5,
        )

        if show_info:
            info_text = f"vx={vx:.2f}, vy={vy:.2f}\n"
            info_text += f"L={int(left_contact)}, R={int(right_contact)}"

            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

        ax.set_title(title, fontsize=8)

    def _action_name(self, action: int) -> str:
        """Convert action number to readable name"""
        names = {0: "None", 1: "Left", 2: "Main", 3: "Right"}
        return names.get(action, f"A{action}")


# Simple usage example
def visualize_single_agent(agent, analyzer, agent_name="PPO"):
    """
    Create all visualizations for a single agent
    """
    # Create visualizer
    visualizer = SingleAgentClusterVisualizer(
        env=analyzer.env, agent=agent, analyzer=analyzer, agent_name=agent_name
    )

    # Run all visualizations
    print(f"\nCreating visualizations for {agent_name}...")

    # 1. X-Y cluster map
    visualizer.visualize_xy_clusters()

    # 2. Cluster gallery
    visualizer.visualize_cluster_gallery(n_clusters_to_show=6)

    # 3. Contact state analysis
    visualizer.visualize_contact_states()

    # 4. Value landscape
    visualizer.visualize_value_landscape()

    # 5. Trajectory visualization (if trajectories exist)
    if hasattr(analyzer, "collected_trajectories") and analyzer.collected_trajectories:
        visualizer.visualize_cluster_transitions(trajectory_idx=0)

    # 6. Dashboard
    visualizer.create_dashboard()

    return visualizer


# Even simpler: just visualize clusters from an existing analyzer
def quick_cluster_viz(analyzer, agent_name="PPO"):
    """
    Quick visualization of clusters
    """
    visualizer = SingleAgentClusterVisualizer(
        env=analyzer.env,
        agent=None,  # Not needed for basic visualization
        analyzer=analyzer,
        agent_name=agent_name,
    )

    # Just show X-Y cluster map and contact states
    visualizer.visualize_xy_clusters()
    visualizer.visualize_contact_states()

    return visualizer


if __name__ == "__main__":
    env = cluster.create_lunar_lander_env()
    model = PPO.load("models/ppo_lunar_lander_v3")
    analyzer = cluster.LunarLanderClusterAnalyzer(model, env, "PPO")
    analyzer.collect_trajectories(n_episodes=200)
    analyzer.cluster_by_action_preference(n_clusters=12, method="kmeans", use_pca=True)
    analyzer.visualize_clusters_2d(save_path="lunar_viz/ppo_clusters_2d.png")
    viz = SingleAgentClusterVisualizer(env, model, analyzer, "PPO")
    # Generate all visualizations
    viz.visualize_xy_clusters()
    viz.visualize_cluster_gallery()
    viz.visualize_contact_states()
    viz.visualize_value_landscape()
    viz.create_dashboard()
