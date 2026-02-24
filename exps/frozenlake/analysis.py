import numpy as np
import gymnasium as gym
import pickle
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
import json
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import warnings

warnings.filterwarnings("ignore")


class ClusterComparisonAnalyzer:
    """
    Analyze and compare clusters from different policies trained on the same environment
    """

    def __init__(self, env_name: str = "FrozenLake-v1", map_name: str = "4x4"):
        self.env = gym.make(env_name, map_name=map_name, is_slippery=True)
        self.env_name = env_name
        self.map_name = map_name

        self.grid_size = 4
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        self.action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

        # Store policies and their clusters
        self.policies = {}  # {policy_name: {policy_data}}
        self.clusterings = {}  # {policy_name: {cluster_data}}

    def load_policy_clusters(self, policy_name: str, filepath: str):
        """
        Load cluster data from a saved policy
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            # Handle different file formats
            if isinstance(data, dict):
                if "clusters" in data:
                    self.clusterings[policy_name] = data
                else:
                    # Try to reconstruct clusters from q_table
                    q_table = data.get("q_table", data)
                    clusters = self._extract_clusters_from_q(q_table)
                    self.clusterings[policy_name] = clusters
            elif hasattr(data, "q_table"):
                clusters = self._extract_clusters_from_q(data.q_table)
                self.clusterings[policy_name] = clusters
            else:
                print(f"Unknown data format for {policy_name}")
                return False

            print(
                f"Loaded {policy_name} with {len(self.clusterings[policy_name]['clusters'])} clusters"
            )
            return True

        except Exception as e:
            print(f"Error loading {policy_name}: {e}")
            return False

    def _extract_clusters_from_q(self, q_table: np.ndarray) -> Dict:
        """
        Extract connected action clusters from Q-table
        """
        # Get deterministic policy
        policy = np.argmax(q_table, axis=1)

        # Group by action
        action_states = {a: [] for a in range(self.n_actions)}
        for s, a in enumerate(policy):
            action_states[a].append(s)

        # Find connected components
        clusters = []
        cluster_id = 0
        state_to_cluster = {}

        for action, states in action_states.items():
            if not states:
                continue

            # Create adjacency for BFS
            state_set = set(states)
            visited = set()

            for start_state in states:
                if start_state in visited:
                    continue

                # BFS to find connected component
                component = []
                queue = [start_state]
                visited.add(start_state)

                while queue:
                    current = queue.pop(0)
                    component.append(current)

                    # Check neighbors
                    r, c = divmod(current, self.grid_size)
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            neighbor = nr * self.grid_size + nc
                            if neighbor in state_set and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

                # Create cluster
                cluster = {
                    "id": cluster_id,
                    "action": int(action),
                    "states": component,
                    "size": len(component),
                }
                clusters.append(cluster)

                for s in component:
                    state_to_cluster[s] = cluster_id

                cluster_id += 1

        return {
            "clusters": clusters,
            "state_to_cluster": state_to_cluster,
            "policy": policy.tolist(),
        }

    def compute_cluster_similarity(self, policy1: str, policy2: str) -> Dict:
        """
        Compute similarity metrics between two clusterings
        """
        if policy1 not in self.clusterings or policy2 not in self.clusterings:
            return {}

        c1 = self.clusterings[policy1]
        c2 = self.clusterings[policy2]

        # Get state assignments
        s2c1 = np.array([c1["state_to_cluster"][s] for s in range(self.n_states)])
        s2c2 = np.array([c2["state_to_cluster"][s] for s in range(self.n_states)])

        # Compute metrics
        metrics = {
            "nmi": normalized_mutual_info_score(s2c1, s2c2),
            "adjusted_rand": adjusted_rand_score(s2c1, s2c2),
            "n_clusters_1": len(c1["clusters"]),
            "n_clusters_2": len(c2["clusters"]),
            "policy_agreement": np.mean(c1["policy"] == c2["policy"]),
        }

        # Compute cluster alignment
        alignment = self._find_cluster_alignment(c1["clusters"], c2["clusters"])
        metrics["alignment"] = alignment

        return metrics

    def _find_cluster_alignment(self, clusters1: List, clusters2: List) -> Dict:
        """
        Find best matching between clusters from two policies
        """
        # Create overlap matrix
        n1, n2 = len(clusters1), len(clusters2)
        overlap = np.zeros((n1, n2))

        for i, c1 in enumerate(clusters1):
            states1 = set(c1["states"])
            for j, c2 in enumerate(clusters2):
                states2 = set(c2["states"])
                overlap[i, j] = len(states1 & states2) / len(states1 | states2)

        # Find best matches (greedy)
        matches = []
        used_i, used_j = set(), set()

        for _ in range(min(n1, n2)):
            max_val = -1
            max_ij = None
            for i in range(n1):
                if i in used_i:
                    continue
                for j in range(n2):
                    if j in used_j:
                        continue
                    if overlap[i, j] > max_val:
                        max_val = overlap[i, j]
                        max_ij = (i, j)

            if max_ij:
                matches.append(
                    {
                        "cluster1_id": clusters1[max_ij[0]]["id"],
                        "cluster2_id": clusters2[max_ij[1]]["id"],
                        "action1": clusters1[max_ij[0]]["action"],
                        "action2": clusters2[max_ij[1]]["action"],
                        "overlap": float(max_val),
                    }
                )
                used_i.add(max_ij[0])
                used_j.add(max_ij[1])

        return {
            "matches": matches,
            "avg_overlap": np.mean([m["overlap"] for m in matches]) if matches else 0,
            "n_matches": len(matches),
        }

    def analyze_policy_diversity(self) -> Dict:
        """
        Analyze diversity across all loaded policies
        """
        policies = list(self.clusterings.keys())
        if len(policies) < 2:
            return {}

        results = {"pairwise": {}, "consensus": None, "action_distributions": {}}

        # Pairwise comparisons
        for i, p1 in enumerate(policies):
            for p2 in policies[i + 1 :]:
                key = f"{p1} vs {p2}"
                results["pairwise"][key] = self.compute_cluster_similarity(p1, p2)

        # Action distribution across policies
        for policy in policies:
            actions = self.clusterings[policy]["policy"]
            dist = [actions.count(a) / self.n_states for a in range(self.n_actions)]
            results["action_distributions"][policy] = dist

        # Find consensus clustering (if possible)
        results["consensus"] = self._find_consensus_clustering(policies)

        return results

    def _find_consensus_clustering(self, policies: List[str]) -> Dict:
        """
        Attempt to find consensus clustering across policies
        """
        # Build co-occurrence matrix
        co_occurrence = np.zeros((self.n_states, self.n_states))

        for policy in policies:
            s2c = self.clusterings[policy]["state_to_cluster"]
            for i in range(self.n_states):
                for j in range(i + 1, self.n_states):
                    if s2c[i] == s2c[j]:
                        co_occurrence[i, j] += 1
                        co_occurrence[j, i] += 1

        # Normalize
        co_occurrence /= len(policies)

        # Find clusters using hierarchical clustering on co-occurrence
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Convert to distance
        distance = 1 - co_occurrence
        np.fill_diagonal(distance, 0)

        # Perform hierarchical clustering
        condensed = squareform(distance)
        Z = linkage(condensed, method="average")

        # Find natural number of clusters
        # Look for elbow in inconsistency
        from scipy.cluster.hierarchy import inconsistent

        depth = 5
        inc = inconsistent(Z, depth)
        last = inc[-depth:, 3]

        # Simple heuristic: use average cluster size from original policies
        avg_n_clusters = np.mean(
            [len(self.clusterings[p]["clusters"]) for p in policies]
        )
        consensus_clusters = fcluster(Z, t=int(avg_n_clusters), criterion="maxclust")

        return {
            "n_clusters": int(avg_n_clusters),
            "assignments": consensus_clusters.tolist(),
            "co_occurrence": co_occurrence.tolist(),
        }

    def identify_stable_regions(self, min_agreement: float = 0.7) -> Dict:
        """
        Identify states that are consistently clustered together across policies
        """
        policies = list(self.clusterings.keys())
        if len(policies) < 2:
            return {}

        n_policies = len(policies)
        stable_regions = []

        # Track which states are consistently together
        state_pairs = defaultdict(int)

        for policy in policies:
            s2c = self.clusterings[policy]["state_to_cluster"]
            for i in range(self.n_states):
                for j in range(i + 1, self.n_states):
                    if s2c[i] == s2c[j]:
                        state_pairs[(i, j)] += 1

        # Find maximal sets of states that are consistently together
        from itertools import combinations

        # Build graph of consistently co-clustered states
        G = nx.Graph()
        for (i, j), count in state_pairs.items():
            if count / n_policies >= min_agreement:
                G.add_edge(i, j)

        # Find connected components (stable regions)
        stable_regions = list(nx.connected_components(G))

        # Filter out singletons
        stable_regions = [
            sorted(region) for region in stable_regions if len(region) > 1
        ]

        return {
            "stable_regions": stable_regions,
            "n_regions": len(stable_regions),
            "agreement_threshold": min_agreement,
        }

    def analyze_hierarchical_potential(self) -> Dict:
        """
        Analyze the potential for hierarchical RL based on cluster structure
        """
        results = {
            "subgoal_candidates": [],
            "bottleneck_states": [],
            "cluster_hierarchy": None,
            "abstract_transitions": {},
        }

        policies = list(self.clusterings.keys())
        if not policies:
            return results

        # Use consensus clustering as base for hierarchy
        consensus = self._find_consensus_clustering(policies)
        assignments = consensus["assignments"]

        # Identify bottleneck states (states that connect different clusters)
        for policy in policies:
            G = nx.DiGraph()

            # Build transition graph from policy?
            # For now, use grid connectivity
            for state in range(self.n_states):
                r, c = divmod(state, self.grid_size)
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        neighbor = nr * self.grid_size + nc
                        if assignments[state] != assignments[neighbor]:
                            # This is a cross-cluster edge
                            if state not in results["bottleneck_states"]:
                                results["bottleneck_states"].append(state)

        # Identify subgoal candidates (states with high value that are bottlenecks)
        # This would require loading value functions from policies

        # Build abstract transition probabilities between clusters
        abstract_transitions = defaultdict(lambda: defaultdict(list))

        for policy in policies:
            # We would need trajectory data for this
            pass

        results["bottleneck_states"] = list(set(results["bottleneck_states"]))

        return results

    def visualize_cluster_comparison(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization comparing clusters across policies
        """
        policies = list(self.clusterings.keys())
        if not policies:
            print("No policies loaded")
            return

        n_policies = len(policies)
        fig, axes = plt.subplots(2, max(2, n_policies), figsize=(5 * n_policies, 10))

        # Row 1: Cluster maps for each policy
        for i, policy in enumerate(policies):
            ax = axes[0, i]

            # Create grid visualization
            grid = np.full((self.grid_size, self.grid_size), -1)
            action_grid = np.full((self.grid_size, self.grid_size), -1)

            for cluster in self.clusterings[policy]["clusters"]:
                for state in cluster["states"]:
                    r, c = divmod(state, self.grid_size)
                    grid[r, c] = cluster["id"]
                    action_grid[r, c] = cluster["action"]

            # Create cluster colormap
            unique_clusters = np.unique(grid[grid >= 0])
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
            color_map = {cid: colors[j] for j, cid in enumerate(unique_clusters)}

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
            ax.set_title(
                f"{policy}\n({len(self.clusterings[policy]['clusters'])} clusters)"
            )
            ax.set_aspect("equal")

        # Fill empty subplots
        for i in range(n_policies, axes.shape[1]):
            axes[0, i].axis("off")

        # Row 2: Similarity metrics
        ax = axes[1, 0]
        if n_policies >= 2:
            # Pairwise similarity matrix
            similarity_matrix = np.zeros((n_policies, n_policies))
            for i, p1 in enumerate(policies):
                for j, p2 in enumerate(policies):
                    if i < j:
                        metrics = self.compute_cluster_similarity(p1, p2)
                        similarity_matrix[i, j] = metrics["nmi"]
                        similarity_matrix[j, i] = metrics["nmi"]
                    elif i == j:
                        similarity_matrix[i, j] = 1.0

            im = ax.imshow(similarity_matrix, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xticks(range(n_policies))
            ax.set_yticks(range(n_policies))
            ax.set_xticklabels(policies, rotation=45, ha="right")
            ax.set_yticklabels(policies)
            ax.set_title("Pairwise NMI Similarity")
            plt.colorbar(im, ax=ax)
        else:
            ax.text(
                0.5,
                0.5,
                "Need at least 2 policies\nfor similarity matrix",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

        ax = axes[1, 1]
        # Action distribution across policies
        x = np.arange(self.n_actions)
        width = 0.8 / n_policies

        for i, policy in enumerate(policies):
            actions = self.clusterings[policy]["policy"]
            dist = [actions.count(a) / self.n_states for a in range(self.n_actions)]
            offset = (i - n_policies / 2 + 0.5) * width
            bars = ax.bar(x + offset, dist, width, label=policy, alpha=0.7)

        ax.set_xlabel("Action")
        ax.set_ylabel("Frequency")
        ax.set_title("Action Distribution by Policy")
        ax.set_xticks(x)
        ax.set_xticklabels([self.action_names[a] for a in range(self.n_actions)])
        ax.legend()

        # Row 2, columns 2+ (if any) - cluster statistics
        if axes.shape[1] > 2:
            ax = axes[1, 2]
            # Cluster size distributions
            data = []
            labels = []
            for policy in policies:
                sizes = [c["size"] for c in self.clusterings[policy]["clusters"]]
                data.append(sizes)
                labels.append(policy)

            ax.boxplot(data, labels=labels)
            ax.set_ylabel("Cluster Size")
            ax.set_title("Cluster Size Distribution")
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle("Policy Cluster Comparison", fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Comparison plot saved to {save_path}")

        plt.show()

    def generate_hierarchical_rl_schema(self) -> Dict:
        """
        Generate a hierarchical RL schema based on cluster analysis
        """
        policies = list(self.clusterings.keys())
        if not policies:
            return {}

        # Use consensus clustering for hierarchy
        consensus = self._find_consensus_clustering(policies)
        assignments = consensus["assignments"]

        # Create abstract MDP
        abstract_mdp = {
            "states": list(set(assignments)),
            "actions": list(range(self.n_actions)),
            "transitions": defaultdict(dict),
            "subgoals": [],
            "options": [],
        }

        # Estimate abstract transitions (simplified)
        # In practice, this would come from trajectory data
        for cluster in abstract_mdp["states"]:
            abstract_mdp["transitions"][cluster] = {}
            for action in range(self.n_actions):
                # Estimate probability of moving to other clusters
                # This is a placeholder - would need real transition data
                abstract_mdp["transitions"][cluster][action] = {}

        # Identify subgoal candidates (clusters containing goal or high-value states)
        goal_state = self.n_states - 1  # Bottom-right corner
        goal_cluster = assignments[goal_state]
        abstract_mdp["subgoals"].append(goal_cluster)

        # Define options (temporally extended actions)
        # Each option is a policy to reach a subgoal
        for subgoal in abstract_mdp["subgoals"]:
            option = {
                "target_cluster": subgoal,
                "initiation_set": [c for c in abstract_mdp["states"] if c != subgoal],
                "termination": lambda s, g=subgoal: s == g,
                "policy": None,  # Would be learned
            }
            abstract_mdp["options"].append(option)

        return abstract_mdp

    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report
        """
        policies = list(self.clusterings.keys())
        report = []

        report.append("=" * 80)
        report.append("CLUSTER COMPARISON ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nEnvironment: {self.env_name}, Map: {self.map_name}")
        report.append(f"Number of policies analyzed: {len(policies)}")

        if policies:
            report.append("\n" + "-" * 80)
            report.append("POLICY SUMMARY")
            report.append("-" * 80)

            for policy in policies:
                clusters = self.clusterings[policy]["clusters"]
                actions = self.clusterings[policy]["policy"]
                action_dist = [actions.count(a) for a in range(self.n_actions)]

                report.append(f"\n{policy}:")
                report.append(f"  Clusters: {len(clusters)}")
                report.append(
                    f"  Avg cluster size: {np.mean([c['size'] for c in clusters]):.2f}"
                )
                report.append(
                    f"  Action distribution: "
                    + ", ".join(
                        [
                            f"{self.action_names[a]}:{action_dist[a]}"
                            for a in range(self.n_actions)
                        ]
                    )
                )

            if len(policies) >= 2:
                report.append("\n" + "-" * 80)
                report.append("PAIRWISE COMPARISONS")
                report.append("-" * 80)

                diversity = self.analyze_policy_diversity()
                for pair, metrics in diversity["pairwise"].items():
                    report.append(f"\n{pair}:")
                    report.append(f"  NMI: {metrics['nmi']:.3f}")
                    report.append(f"  Adjusted Rand: {metrics['adjusted_rand']:.3f}")
                    report.append(
                        f"  Policy agreement: {metrics['policy_agreement']:.3f}"
                    )
                    report.append(
                        f"  Avg cluster overlap: {metrics['alignment']['avg_overlap']:.3f}"
                    )

                report.append("\n" + "-" * 80)
                report.append("STABLE REGIONS")
                report.append("-" * 80)

                stable = self.identify_stable_regions(min_agreement=0.7)
                report.append(
                    f"\nFound {stable['n_regions']} stable regions (≥70% agreement):"
                )
                for i, region in enumerate(stable["stable_regions"]):
                    report.append(f"  Region {i + 1}: {region}")

                report.append("\n" + "-" * 80)
                report.append("HIERARCHICAL RL POTENTIAL")
                report.append("-" * 80)

                hier = self.analyze_hierarchical_potential()
                report.append(f"\nBottleneck states: {hier['bottleneck_states']}")

                abstract_mdp = self.generate_hierarchical_rl_schema()
                report.append(
                    f"\nAbstract MDP has {len(abstract_mdp['states'])} states"
                )
                report.append(f"Identified subgoals: {abstract_mdp['subgoals']}")
                report.append(f"Defined {len(abstract_mdp['options'])} options")

        return "\n".join(report)


class HierarchicalRLTrainer:
    """
    Train a hierarchical RL agent using discovered clusters
    """

    def __init__(self, env, cluster_analyzer: ClusterComparisonAnalyzer):
        self.env = env
        self.analyzer = cluster_analyzer

        # Use consensus clustering for hierarchy
        policies = list(cluster_analyzer.clusterings.keys())
        if policies:
            self.consensus = cluster_analyzer._find_consensus_clustering(policies)
            self.assignments = self.consensus["assignments"]
        else:
            self.consensus = None
            self.assignments = list(range(env.observation_space.n))

        # Abstract MDP
        self.abstract_states = list(set(self.assignments))
        self.n_abstract = len(self.abstract_states)

        # Initialize abstract Q-table
        self.abstract_q = np.zeros((self.n_abstract, env.action_space.n))

        # Options for each abstract state
        self.options = {}

    def get_abstract_state(self, state: int) -> int:
        """Map ground state to abstract state"""
        return self.assignments[state]

    def train_abstract_policy(self, episodes: int = 5000):
        """
        Train high-level policy on abstract MDP
        """
        print("\nTraining abstract policy...")

        for episode in range(episodes):
            state, _ = self.env.reset()
            abstract_state = self.get_abstract_state(state)
            done = False
            total_reward = 0

            while not done:
                # Choose action using epsilon-greedy on abstract Q
                if np.random.random() < 0.1:  # epsilon
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.abstract_q[abstract_state])

                # Take action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_abstract = self.get_abstract_state(next_state)
                done = terminated or truncated

                # Q-learning update on abstract level
                best_next = np.max(self.abstract_q[next_abstract])
                self.abstract_q[abstract_state, action] += 0.1 * (
                    reward + 0.99 * best_next - self.abstract_q[abstract_state, action]
                )

                abstract_state = next_abstract
                total_reward += reward

            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}, Reward: {total_reward}")

    def get_option_policy(self, target_abstract: int) -> Dict[int, int]:
        """
        Get a policy to reach a target abstract state
        """
        # This would be learned, but for now use ground policy
        # that maximizes probability of reaching target
        policy = {}

        # Use first available ground policy as base
        if self.analyzer.clusterings:
            first_policy = list(self.analyzer.clusterings.keys())[0]
            ground_policy = self.analyzer.clusterings[first_policy]["policy"]

            for state in range(self.env.observation_space.n):
                if self.assignments[state] == target_abstract:
                    # Already in target, any action is fine
                    policy[state] = ground_policy[state]
                else:
                    # Try to move towards target abstract state
                    # Simplified: use ground policy
                    policy[state] = ground_policy[state]

        return policy


def run_cluster_comparison_experiment():
    """
    Run comprehensive cluster comparison experiment
    """
    print("=" * 80)
    print("CLUSTER COMPARISON AND HIERARCHICAL RL ANALYSIS")
    print("=" * 80)

    # Initialize analyzer
    analyzer = ClusterComparisonAnalyzer(map_name="4x4")

    # Generate or load multiple policies
    # For demonstration, we'll create synthetic variations

    print("\nGenerating synthetic policy variations...")

    # Policy 1: Standard trained policy (simulated)
    q1 = np.random.randn(16, 4)  # Replace with actual trained Q
    q1[15, :] = 0  # Goal state
    analyzer.load_policy_clusters("Policy_Standard", {"q_table": q1})

    # Policy 2: More conservative (higher temperature)
    q2 = q1.copy()
    analyzer.load_policy_clusters("Policy_Conservative", {"q_table": q2})

    # Policy 3: More exploratory (different random seed)
    q3 = np.random.randn(16, 4)
    q3[15, :] = 0
    analyzer.load_policy_clusters("Policy_Exploratory", {"q_table": q3})

    # Visualize comparison
    analyzer.visualize_cluster_comparison(save_path="cluster_comparison.png")

    # Generate report
    report = analyzer.generate_report()
    print(report)

    # Save report
    with open("cluster_comparison_report.txt", "w") as f:
        f.write(report)

    # Hierarchical RL potential
    hier_schema = analyzer.generate_hierarchical_rl_schema()
    print("\n" + "=" * 80)
    print("HIERARCHICAL RL SCHEMA")
    print("=" * 80)
    print(f"Abstract states: {hier_schema['states']}")
    print(f"Subgoals: {hier_schema['subgoals']}")
    print(f"Options defined: {len(hier_schema['options'])}")

    return analyzer


if __name__ == "__main__":
    analyzer = run_cluster_comparison_experiment()
