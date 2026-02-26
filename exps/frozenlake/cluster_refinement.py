import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque
import pickle
import os


class FrozenLakeClusterRefinement:
    """
    Find coarsest common refinement of two clusterings and use Bellman optimality
    """

    def __init__(self, env_name: str = "FrozenLake-v1", map_name: str = "4x4"):
        self.env = gym.make(env_name, map_name=map_name)
        self.map_name = map_name
        self.grid_size = 4 if map_name == "4x4" else 8
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        # State coordinates
        self.state_coords = {
            s: (s // self.grid_size, s % self.grid_size) for s in range(self.n_states)
        }

        # Action mappings
        self.action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

        # Storage
        self.clustering1 = None  # First clustering (e.g., from SARSA)
        self.clustering2 = None  # Second clustering (e.g., from MC)
        self.refinement = None  # Coarsest common refinement
        self.q_refinement = None  # Q-values on refinement
        self.superior_policy = None  # Improved policy
        self.state_to_cluster = None

    def load_clustering_from_agent(self, agent, name: str = "agent") -> Dict:
        """
        Extract clustering from a trained agent
        Clusters are connected states with same preferred action
        """
        # Get deterministic policy
        policy = agent.get_deterministic_policy()

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

                    for neighbor in self._get_neighbors(current):
                        if neighbor in state_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                clusters.append(
                    {
                        "id": cluster_id,
                        "action": action,
                        "states": component,
                        "size": len(component),
                    }
                )

                for s in component:
                    state_to_cluster[s] = cluster_id

                cluster_id += 1

        return {
            "name": name,
            "clusters": clusters,
            "state_to_cluster": state_to_cluster,
            "n_clusters": len(clusters),
        }

    def _get_neighbors(self, state: int) -> List[int]:
        """Get 4-directional neighbors"""
        r, c = self.state_coords[state]
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append(nr * self.grid_size + nc)
        return neighbors

    def set_clusterings(self, clustering1: Dict, clustering2: Dict):
        """Set the two clusterings to refine"""
        self.clustering1 = clustering1
        self.clustering2 = clustering2
        print(
            f"\nClustering 1 ({clustering1['name']}): {clustering1['n_clusters']} clusters"
        )
        print(
            f"Clustering 2 ({clustering2['name']}): {clustering2['n_clusters']} clusters"
        )

    def compute_coarsest_common_refinement(self) -> Dict:
        """
        Find the coarsest clustering that refines both input clusterings
        Two states are in the same refined cluster iff they are in the same cluster
        in both original clusterings
        """
        if self.clustering1 is None or self.clustering2 is None:
            raise ValueError("Both clusterings must be set first")

        print("\n" + "=" * 60)
        print("COMPUTING COARSEST COMMON REFINEMENT")
        print("=" * 60)

        # Create equivalence relation: states equivalent if they're in same cluster in both
        equivalence = {}
        refined_id = 0
        refined_clusters = []
        state_to_refined = {}

        # Group states by (cluster1_id, cluster2_id) pairs
        pair_to_states = defaultdict(list)

        for state in range(self.n_states):
            c1 = self.clustering1["state_to_cluster"][state]
            c2 = self.clustering2["state_to_cluster"][state]
            pair_to_states[(c1, c2)].append(state)

        print(f"\nFound {len(pair_to_states)} equivalence classes")

        # Create refined clusters
        for (c1, c2), states in pair_to_states.items():
            # Get actions from both clusterings for this refined cluster
            action1 = None
            action2 = None
            for cluster in self.clustering1["clusters"]:
                if cluster["id"] == c1:
                    action1 = cluster["action"]
                    break

            for cluster in self.clustering2["clusters"]:
                if cluster["id"] == c2:
                    action2 = cluster["action"]
                    break

            refined_clusters.append(
                {
                    "id": refined_id,
                    "c1_id": c1,
                    "c2_id": c2,
                    "action1": action1,
                    "action2": action2,
                    "states": states,
                    "size": len(states),
                    "agreement": action1 == action2,
                }
            )

            for s in states:
                state_to_refined[s] = refined_id

            refined_id += 1

        self.refinement = {
            "clusters": refined_clusters,
            "state_to_cluster": state_to_refined,
            "n_clusters": len(refined_clusters),
            "agreement_rate": np.mean([c["agreement"] for c in refined_clusters]),
        }
        self.state_to_cluster = state_to_refined

        print(f"Created {self.refinement['n_clusters']} refined clusters")
        print(f"Action agreement rate: {self.refinement['agreement_rate']:.3f}")

        # Print interesting refined clusters
        print("\nRefined clusters with action disagreement:")
        for cluster in refined_clusters:
            if not cluster["agreement"]:
                action1_name = self.action_names[cluster["action1"]]
                action2_name = self.action_names[cluster["action2"]]
                print(
                    f"  Cluster {cluster['id']}: {cluster['states']} "
                    f"(C1:{action1_name} vs C2:{action2_name})"
                )

        return self.refinement

    def compute_refinement_q_values(
        self, agent1, agent2, gamma: float = 0.99
    ) -> np.ndarray:
        """
        Compute Q-values for refined clusters using Bellman optimality
        Q(C, a) = max over both policies' Q-values for states in C
        """
        if self.refinement is None:
            raise ValueError("Must compute refinement first")

        print("\n" + "=" * 60)
        print("COMPUTING REFINEMENT Q-VALUES")
        print("=" * 60)

        n_refined = self.refinement["n_clusters"]
        n_actions = self.n_actions

        # Initialize Q-values for refined clusters
        self.q_refinement = np.zeros((n_refined, n_actions))

        # For each refined cluster, take the best Q-value from either agent
        for cluster in self.refinement["clusters"]:
            cid = cluster["id"]

            for action in range(n_actions):
                # Get max Q-value over states in this cluster from both agents
                max_q = -np.inf

                for state in cluster["states"]:
                    q1 = agent1.q_table[state, action]
                    q2 = agent2.q_table[state, action]
                    max_q = max(max_q, q1, q2)

                self.q_refinement[cid, action] = max_q

        print(f"Computed Q-values for {n_refined} refined clusters")

        return self.q_refinement

    def compute_superior_policy(self) -> Dict[int, int]:
        """
        Compute policy on refined clusters using Bellman optimality
        π*(C) = argmax_a Q_refinement(C, a)
        Then map back to individual states
        """
        if self.q_refinement is None:
            raise ValueError("Must compute refinement Q-values first")

        print("\n" + "=" * 60)
        print("COMPUTING SUPERIOR POLICY")
        print("=" * 60)

        # Policy on refined clusters
        cluster_policy = {}
        for cid in range(self.refinement["n_clusters"]):
            cluster_policy[cid] = np.argmax(self.q_refinement[cid])

        # Map to individual states
        state_policy = {}
        policy_changes = 0

        for state in range(self.n_states):
            cid = self.refinement["state_to_cluster"][state]
            new_action = cluster_policy[cid]

            # Check if this differs from original policies
            orig_action1 = None
            orig_action2 = None
            for cluster in self.clustering1["clusters"]:
                if state in cluster["states"]:
                    orig_action1 = cluster["action"]
                    break

            for cluster in self.clustering2["clusters"]:
                if state in cluster["states"]:
                    orig_action2 = cluster["action"]
                    break

            state_policy[state] = new_action

            if new_action != orig_action1 or new_action != orig_action2:
                policy_changes += 1

        self.superior_policy = state_policy

        print(f"Policy changes: {policy_changes}/{self.n_states} states")

        # Compare with original policies
        agreement1 = np.mean(
            [
                1
                for s in range(self.n_states)
                if state_policy[s] == self.clustering1["clusters"][0]["action"]
                for c in self.clustering1["clusters"]
                if s in c["states"]
            ]
        )
        agreement2 = np.mean(
            [
                1
                for s in range(self.n_states)
                if state_policy[s] == self.clustering2["clusters"][0]["action"]
                for c in self.clustering2["clusters"]
                if s in c["states"]
            ]
        )

        print(f"Agreement with policy 1: {agreement1:.3f}")
        print(f"Agreement with policy 2: {agreement2:.3f}")

        return state_policy

    def evaluate_policy(self, policy: Dict[int, int], n_episodes: int = 1000) -> float:
        """Evaluate a policy"""
        successes = 0

        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False

            while not done:
                action = policy[state]
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state

            if reward > 0:
                successes += 1

        return successes / n_episodes

    def visualize_refinement(self, save_path: Optional[str] = None):
        """
        Visualize the refinement process
        """
        if self.refinement is None:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: First clustering
        ax = axes[0, 0]
        self._plot_clustering(
            ax, self.clustering1, f"Clustering 1: {self.clustering1['name']}"
        )

        # Plot 2: Second clustering
        ax = axes[0, 1]
        self._plot_clustering(
            ax, self.clustering2, f"Clustering 2: {self.clustering2['name']}"
        )

        # Plot 3: Refinement
        ax = axes[0, 2]
        self._plot_refinement(ax, "Coarsest Common Refinement")

        # Plot 4: Action agreement map
        ax = axes[1, 0]
        self._plot_agreement_map(ax, "Action Agreement\n(Green=Agree, Red=Disagree)")

        # Plot 5: Refinement Q-values
        ax = axes[1, 1]
        if self.q_refinement is not None:
            self._plot_q_values(ax, "Refinement Q-values")

        # Plot 6: Policy improvement
        ax = axes[1, 2]
        if self.superior_policy is not None:
            self._plot_superior_policy(ax, "Superior Policy")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        plt.show()

    def _plot_clustering(self, ax, clustering, title):
        """Helper to plot a clustering"""
        grid = np.full((self.grid_size, self.grid_size), -1)
        action_grid = np.full((self.grid_size, self.grid_size), -1)

        for cluster in clustering["clusters"]:
            for state in cluster["states"]:
                r, c = self.state_coords[state]
                grid[r, c] = cluster["id"]
                action_grid[r, c] = cluster["action"]

        # Create colormap
        unique_clusters = np.unique(grid[grid >= 0])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r, c] >= 0:
                    color = colors[grid[r, c] % len(colors)]
                    rect = plt.Rectangle(
                        (c, self.grid_size - 1 - r),
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
                        self.grid_size - 0.5 - r,
                        self.action_symbols[action],
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\n({clustering['n_clusters']} clusters)")
        ax.set_aspect("equal")

    def _plot_refinement(self, ax, title):
        """Plot the refinement"""
        grid = np.full((self.grid_size, self.grid_size), -1)

        for cluster in self.refinement["clusters"]:
            for state in cluster["states"]:
                r, c = self.state_coords[state]
                grid[r, c] = cluster["id"]

        unique_clusters = np.unique(grid[grid >= 0])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r, c] >= 0:
                    color = colors[grid[r, c] % len(colors)]
                    rect = plt.Rectangle(
                        (c, self.grid_size - 1 - r),
                        1,
                        1,
                        facecolor=color,
                        alpha=0.6,
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax.add_patch(rect)

                    # Show if actions agreed
                    for cluster in self.refinement["clusters"]:
                        if cluster["id"] == grid[r, c]:
                            if not cluster["agreement"]:
                                ax.text(
                                    c + 0.5,
                                    self.grid_size - 0.5 - r,
                                    "?",
                                    ha="center",
                                    va="center",
                                    fontsize=14,
                                    color="red",
                                )

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\n({self.refinement['n_clusters']} clusters)")
        ax.set_aspect("equal")

    def _plot_agreement_map(self, ax, title):
        """Plot where the two clusterings agree on actions"""
        grid = np.full((self.grid_size, self.grid_size), 0)

        for state in range(self.n_states):
            r, c = self.state_coords[state]

            # Find actions from both clusterings
            action1 = None
            action2 = None

            for cluster in self.clustering1["clusters"]:
                if state in cluster["states"]:
                    action1 = cluster["action"]
                    break

            for cluster in self.clustering2["clusters"]:
                if state in cluster["states"]:
                    action2 = cluster["action"]
                    break

            if action1 == action2:
                # Agreement - color by the agreed action
                colors = ["red", "blue", "green", "purple"]
                rect = plt.Rectangle(
                    (c, self.grid_size - 1 - r),
                    1,
                    1,
                    facecolor=colors[action1],
                    alpha=0.6,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)
                ax.text(
                    c + 0.5,
                    self.grid_size - 0.5 - r,
                    self.action_symbols[action1],
                    ha="center",
                    va="center",
                    fontsize=12,
                )
            else:
                # Disagreement - white with X
                rect = plt.Rectangle(
                    (c, self.grid_size - 1 - r),
                    1,
                    1,
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)
                ax.plot(
                    [c, c + 1],
                    [self.grid_size - 1 - r, self.grid_size - r],
                    "r-",
                    linewidth=2,
                )
                ax.plot(
                    [c, c + 1],
                    [self.grid_size - r, self.grid_size - 1 - r],
                    "r-",
                    linewidth=2,
                )

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_aspect("equal")

    def _plot_q_values(self, ax, title):
        """Plot Q-values for each refined cluster"""
        if self.q_refinement is None:
            return

        n_clusters = self.refinement["n_clusters"]
        x = np.arange(n_clusters)
        width = 0.2

        for action in range(self.n_actions):
            offset = (action - 1.5) * width
            ax.bar(
                x + offset,
                self.q_refinement[:, action],
                width,
                label=self.action_names[action],
                alpha=0.7,
            )

        ax.set_xlabel("Refined Cluster ID")
        ax.set_ylabel("Q-value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_superior_policy(self, ax, title):
        """Plot the superior policy"""
        grid = np.full((self.grid_size, self.grid_size), -1)

        for state in range(self.n_states):
            r, c = self.state_coords[state]
            action = self.superior_policy[state]

            colors = ["red", "blue", "green", "purple"]
            rect = plt.Rectangle(
                (c, self.grid_size - 1 - r),
                1,
                1,
                facecolor=colors[action],
                alpha=0.6,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)
            ax.text(
                c + 0.5,
                self.grid_size - 0.5 - r,
                self.action_symbols[action],
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_aspect("equal")

    def print_refinement_analysis(self):
        """Print detailed analysis of the refinement"""
        print("\n" + "=" * 60)
        print("REFINEMENT ANALYSIS")
        print("=" * 60)

        print(f"\nClustering 1: {self.clustering1['n_clusters']} clusters")
        print(f"Clustering 2: {self.clustering2['n_clusters']} clusters")
        print(f"Refinement: {self.refinement['n_clusters']} clusters")

        print(
            f"\nRefinement reduces ambiguity by: "
            f"{(self.clustering1['n_clusters'] + self.clustering2['n_clusters']) / 2 - self.refinement['n_clusters']:.1f} "
            f"clusters on average"
        )

        # Analyze refined clusters
        print("\nRefined clusters by size:")
        sizes = [c["size"] for c in self.refinement["clusters"]]
        print(f"  Min size: {min(sizes)}")
        print(f"  Max size: {max(sizes)}")
        print(f"  Avg size: {np.mean(sizes):.2f}")
        print(f"  Std size: {np.std(sizes):.2f}")

        # Action agreement
        print(f"\nAction agreement rate: {self.refinement['agreement_rate']:.3f}")
        print(
            f"Disagreement clusters: {sum(1 for c in self.refinement['clusters'] if not c['agreement'])}"
        )

        if self.q_refinement is not None:
            print(
                f"\nRefinement Q-value range: [{np.min(self.q_refinement):.3f}, {np.max(self.q_refinement):.3f}]"
            )

        if self.superior_policy is not None:
            # Evaluate policies
            print("\n" + "=" * 60)
            print("POLICY EVALUATION")
            print("=" * 60)

            # Get original policies
            policy1 = {}
            for cluster in self.clustering1["clusters"]:
                for state in cluster["states"]:
                    policy1[state] = cluster["action"]

            policy2 = {}
            for cluster in self.clustering2["clusters"]:
                for state in cluster["states"]:
                    policy2[state] = cluster["action"]

            # Evaluate
            print("\nEvaluating policies on FrozenLake...")
            success1 = self.evaluate_policy(policy1)
            success2 = self.evaluate_policy(policy2)
            success_superior = self.evaluate_policy(self.superior_policy)

            print(f"\n{'Policy':<20} {'Success Rate':>15}")
            print("-" * 37)
            print(f"{self.clustering1['name']:<20} {success1:>15.3f}")
            print(f"{self.clustering2['name']:<20} {success2:>15.3f}")
            print(f"{'Superior Policy':<20} {success_superior:>15.3f}")

            improvement1 = (
                (success_superior - success1) / success1 * 100 if success1 > 0 else 0
            )
            improvement2 = (
                (success_superior - success2) / success2 * 100 if success2 > 0 else 0
            )

            print(
                f"\nImprovement over {self.clustering1['name']}: {improvement1:+.1f}%"
            )
            print(f"Improvement over {self.clustering2['name']}: {improvement2:+.1f}%")


def run_refinement_experiment():
    """
    Run complete refinement experiment with SARSA and Monte Carlo
    """
    print("\n" + "=" * 70)
    print("CLUSTER REFINEMENT WITH BELLMAN OPTIMALITY")
    print("=" * 70)

    # Create refinement analyzer
    analyzer = FrozenLakeClusterRefinement(map_name="4x4")

    # Load or train agents (using your existing implementations)
    # For demonstration, I'll create synthetic agents with different policies

    # agent1 = sarsa_monte.FrozenLakeSarsa()
    # agent2 = sarsa_monte.FrozenLakeMonteCarlo()

    # Extract clusterings
    print("\nExtracting clusterings from agents...")
    clustering1 = analyzer.load_clustering_from_agent(agent1, "SARSA")
    clustering2 = analyzer.load_clustering_from_agent(agent2, "MC")

    analyzer.set_clusterings(clustering1, clustering2)

    # Compute refinement
    refinement = analyzer.compute_coarsest_common_refinement()

    # Compute Q-values for refinement
    q_refinement = analyzer.compute_refinement_q_values(agent1, agent2)

    # Compute superior policy
    superior_policy = analyzer.compute_superior_policy()

    # Visualize
    analyzer.visualize_refinement(save_path="refinement_analysis.png")

    # Print analysis
    analyzer.print_refinement_analysis()

    return analyzer


if __name__ == "__main__":
    analyzer = run_refinement_experiment()
