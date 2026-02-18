"""
Enhanced Topological Analysis for Distinguishing Stratified vs Hierarchical Structures
Works with CT-Graph, Minecraft-style, and Ant Maze environments
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings("ignore")

# For TDA
try:
    from ripser import ripser
    from persim import plot_diagrams

    RIPSER_AVAILABLE = True
except ImportError:
    print("Install ripser: pip install ripser persim")
    RIPSER_AVAILABLE = False


class HierarchicalStructureAnalyzer:
    """
    Analyzes RL agent representations to distinguish between:
    - Stratified structure: flat partitioning with boundaries
    - Hierarchical structure: multi-level partial order with subgoal dependencies
    """

    def __init__(self, model, env_name, env_config=None):
        """
        Args:
            model: Trained SB3 model (DQN, PPO, etc.)
            env_name: Environment name or custom env
            env_config: Configuration for custom environments
        """
        self.env_name = env_name
        self.model = model
        self.env = self._make_env(env_name, env_config)

        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.hidden_features = []
        self.episode_starts = [0]  # Indices where episodes start

        # Analysis results
        self.stratification_score = None
        self.hierarchy_score = None
        self.structure_type = None

    def _make_env(self, env_name, config):
        """Create environment with support for custom envs"""
        if env_name == "CT-Graph":
            # You'd need to implement or import CT-Graph
            # For now, using a placeholder

            return gym.make("FrozenLake-v1", map_name="8x8")
        elif env_name == "MiniGrid":
            try:
                from minigrid.wrappers import ImgObsWrapper

                env = gym.make("MiniGrid-FourRooms-v0")
                return ImgObsWrapper(env)
            except ImportError:
                print("Install minigrid: pip install gym-minigrid")
                return gym.make("CartPole-v1")
        else:
            return gym.make(env_name)

    def collect_experience(self, n_episodes=200, use_hidden=True):
        """
        Collect trajectories with hidden representations

        Args:
            n_episodes: Number of episodes to collect
            use_hidden: Extract hidden layer features
        """
        print(f"Collecting {n_episodes} episodes from {self.env_name}...")

        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            self.episode_starts.append(len(self.states))

            while not done:
                # Store state
                self.states.append(obs.copy() if hasattr(obs, "copy") else obs)

                # Get action and hidden features
                with torch.no_grad():
                    if hasattr(obs, "__len__") and len(obs) > 0:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    else:
                        obs_tensor = torch.FloatTensor([obs]).unsqueeze(0)

                    # Extract features if model supports it
                    if (
                        use_hidden
                        and hasattr(self.model, "policy")
                        and hasattr(self.model.policy, "mlp_extractor")
                    ):
                        features = self.model.policy.mlp_extractor(obs_tensor)[0]
                        action, _ = self.model.predict(obs, deterministic=False)
                        self.hidden_features.append(features.numpy().flatten())
                    else:
                        action, _ = self.model.predict(obs, deterministic=False)

                # Step environment
                next_obs, reward, done, _ = self.env.step(action)

                # Store data
                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                self.next_states.append(
                    next_obs.copy() if hasattr(next_obs, "copy") else next_obs
                )

                obs = next_obs

            if (episode + 1) % 20 == 0:
                print(f"  Episode {episode + 1}: {len(self.states)} total transitions")

        # Convert to arrays where possible
        self.states = (
            np.array(self.states)
            if isinstance(self.states[0], np.ndarray)
            else self.states
        )
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)

        if self.hidden_features:
            self.hidden_features = np.array(self.hidden_features)

        print(f"Collected {len(self.states)} transitions across {n_episodes} episodes")
        return self

    def detect_decision_points(self, window=5, threshold=0.3):
        """
        Identify potential decision points where agent must choose between options

        Returns:
            decision_indices: indices of decision states
            decision_entropy: action entropy at each state
        """
        print("\n=== Detecting Decision Points ===")

        # Compute action entropy in sliding window
        action_entropy = np.zeros(len(self.actions))

        for i in range(len(self.actions)):
            start = max(0, i - window)
            end = min(len(self.actions), i + window + 1)
            window_actions = self.actions[start:end]

            # Compute entropy of action distribution
            counts = np.bincount(window_actions, minlength=self.model.action_space.n)
            probs = counts / len(window_actions)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            action_entropy[i] = entropy

        # Find peaks (decision points)
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(action_entropy, height=threshold)

        print(f"Found {len(peaks)} decision points (action entropy peaks)")
        print(f"  Mean entropy at decisions: {np.mean(action_entropy[peaks]):.3f}")
        print(
            f"  Mean entropy elsewhere: {np.mean(action_entropy[~np.isin(np.arange(len(action_entropy)), peaks)]):.3f}"
        )

        return peaks, action_entropy

    def extract_subgoal_sequences(self, decision_points, min_gap=10):
        """
        Extract subgoal sequences between decision points

        For hierarchical environments, subgoals should form a partial order
        """
        print("\n=== Extracting Subgoal Sequences ===")

        # Group decision points by episode
        episode_decisions = []
        for i in range(len(self.episode_starts) - 1):
            ep_start = self.episode_starts[i]
            ep_end = self.episode_starts[i + 1]

            ep_decisions = [d for d in decision_points if ep_start <= d < ep_end]
            episode_decisions.append(ep_decisions)

        # Extract subgoal states (states at decision points)
        subgoal_states = []
        for ep_dec in episode_decisions:
            ep_subgoals = [self.states[d] for d in ep_dec]
            subgoal_states.append(ep_subgoals)

        # Analyze subgoal ordering
        if len(subgoal_states) < 5:
            print("  Not enough episodes with decision points")
            return None, None

        # Check for consistent ordering across episodes
        # For true hierarchy, subgoals should appear in consistent order
        if all(len(sg) >= 2 for sg in subgoal_states if sg):
            # Compare first and second subgoal across episodes
            first_subgoals = [sg[0] for sg in subgoal_states if len(sg) >= 1]
            second_subgoals = [sg[1] for sg in subgoal_states if len(sg) >= 2]

            if first_subgoals and second_subgoals:
                # Check if first subgoals are similar across episodes
                first_similarity = self._compute_state_similarity(first_subgoals)
                second_similarity = self._compute_state_similarity(second_subgoals)

                print(f"  First subgoal similarity: {first_similarity:.3f}")
                print(f"  Second subgoal similarity: {second_similarity:.3f}")

                # Hierarchy indicator: subgoals are consistent across episodes
                if first_similarity > 0.5 and second_similarity > 0.3:
                    print("  ✓ Consistent subgoal ordering detected (hierarchical)")
                else:
                    print("  ✗ No consistent subgoal ordering (possibly stratified)")

        return subgoal_states, episode_decisions

    def _compute_state_similarity(self, states):
        """Compute average pairwise similarity between states"""
        if len(states) < 2:
            return 0.0

        # Flatten if needed
        if hasattr(states[0], "shape") and len(states[0].shape) > 1:
            states_flat = [s.flatten() for s in states]
        else:
            states_flat = states

        states_flat = np.array(states_flat)

        # Compute pairwise cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(states_flat)

        # Return mean of upper triangle (excluding diagonal)
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        return np.mean(upper_tri)

    def build_state_transition_graph(self, subsample=1000):
        """
        Build directed graph of state transitions

        For hierarchical structure, graph should have:
        - Communities corresponding to subgoals
        - Bottlenecks at decision points
        """
        print("\n=== Building State Transition Graph ===")

        # Subsample for efficiency
        if len(self.states) > subsample:
            indices = np.random.choice(len(self.states), subsample, replace=False)
            indices.sort()
            states_sub = self.states[indices]
            actions_sub = self.actions[indices]
            next_idx_map = {idx: i for i, idx in enumerate(indices)}
        else:
            indices = np.arange(len(self.states))
            states_sub = self.states
            actions_sub = self.actions

        # Build graph
        G = nx.DiGraph()

        # Add nodes
        for i, idx in enumerate(indices):
            if hasattr(states_sub[i], "shape") and len(states_sub[i].shape) > 1:
                # For image observations, use hash of flattened array
                node_id = hash(states_sub[i].tobytes()) % (10**8)
            else:
                # For vector observations, use tuple
                node_id = (
                    tuple(states_sub[i].flatten())
                    if hasattr(states_sub[i], "flatten")
                    else states_sub[i]
                )
            G.add_node(i, state=states_sub[i], original_idx=idx, node_id=node_id)

        # Add edges for transitions
        for i, idx in enumerate(indices[:-1]):
            next_idx = idx + 1
            if next_idx in next_idx_map:
                j = next_idx_map[next_idx]
                G.add_edge(i, j, action=actions_sub[i])

        # Compute graph metrics
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Detect communities (potential subgoal regions)
        try:
            from community import community_louvain

            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            partition = community_louvain.best_partition(G_undirected)

            n_communities = len(set(partition.values()))
            print(f"  Detected {n_communities} communities in state space")

            # Compute modularity
            modularity = community_louvain.modularity(partition, G_undirected)
            print(f"  Modularity: {modularity:.3f}")

            # Hierarchy indicator: high modularity with bottleneck structure
            if modularity > 0.3 and n_communities > 2:
                print("  ✓ Strong community structure detected")

            return G, partition, modularity
        except ImportError:
            print("  Install python-louvain for community detection")
            return G, None, None

    def analyze_temporal_abstraction(self):
        """
        Analyze if agent uses temporal abstraction (skills/options)

        Key for hierarchy: same action sequences repeated in different contexts
        """
        print("\n=== Analyzing Temporal Abstraction ===")

        # Extract action sequences of varying lengths
        max_seq_len = 10
        sequence_counts = defaultdict(int)

        for i in range(len(self.actions) - max_seq_len):
            for length in range(2, max_seq_len + 1):
                seq = tuple(self.actions[i : i + length])
                sequence_counts[seq] += 1

        # Find repeated sequences (potential skills)
        min_repetitions = max(3, len(self.actions) // 100)
        repeated_seqs = {
            seq: count
            for seq, count in sequence_counts.items()
            if count >= min_repetitions
        }

        print(f"  Found {len(repeated_seqs)} repeated action sequences")

        if repeated_seqs:
            # Show top 5 most repeated sequences
            top_seqs = sorted(repeated_seqs.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            print("  Top repeated sequences:")
            for seq, count in top_seqs:
                print(f"    {seq}: {count} times")

        # Hierarchy indicator: presence of repeated multi-action sequences
        hierarchy_score_seq = len(repeated_seqs) / (max_seq_len * 2)

        return repeated_seqs, hierarchy_score_seq

    def analyze_causal_dependencies(self):
        """
        Analyze causal relationships between subgoals

        Based on recent causal HRL research [citation:1][citation:7]
        """
        print("\n=== Analyzing Causal Dependencies ===")

        # Use decision points as potential subgoals
        decision_points, entropy = self.detect_decision_points()

        if len(decision_points) < 5:
            print("  Not enough decision points for causal analysis")
            return None, 0

        # Check if reaching certain states enables others
        # For true hierarchy, subgoals have causal dependencies

        # Group by episode
        subgoals_by_episode = []
        for i in range(len(self.episode_starts) - 1):
            ep_start = self.episode_starts[i]
            ep_end = self.episode_starts[i + 1]

            ep_decisions = [d for d in decision_points if ep_start <= d < ep_end]
            if len(ep_decisions) >= 2:
                # Get states at decision points
                states_at_decisions = [self.states[d] for d in ep_decisions]
                subgoals_by_episode.append(states_at_decisions)

        # Check for consistent ordering
        # In true hierarchy, subgoal A should consistently precede subgoal B

        if len(subgoals_by_episode) < 5:
            print("  Insufficient episodes with multiple subgoals")
            return None, 0

        # Simplified causal score: proportion of episodes where subgoal order is consistent
        # This is a placeholder - real causal discovery would use proper methods [citation:7]

        # For now, check if first and second subgoals are similar across episodes
        first_subgoals = [sg[0] for sg in subgoals_by_episode if len(sg) >= 1]
        second_subgoals = [sg[1] for sg in subgoals_by_episode if len(sg) >= 2]

        if first_subgoals and second_subgoals:
            # Cluster first subgoals
            if hasattr(first_subgoals[0], "shape") and len(first_subgoals[0].shape) > 1:
                # For images, use simple flatten
                first_flat = np.array([s.flatten()[:100] for s in first_subgoals])
                second_flat = np.array([s.flatten()[:100] for s in second_subgoals])
            else:
                first_flat = np.array(first_subgoals)
                second_flat = np.array(second_subgoals)

            from sklearn.cluster import KMeans

            kmeans_first = KMeans(n_clusters=min(3, len(first_flat))).fit(first_flat)
            kmeans_second = KMeans(n_clusters=min(3, len(second_flat))).fit(second_flat)

            # Check if clusters are distinct (different subgoal types)
            first_clusters = kmeans_first.labels_
            second_clusters = kmeans_second.labels_

            # If clusters are well-separated, suggests different subgoal types
            from sklearn.metrics import silhouette_score

            if len(set(first_clusters)) > 1:
                first_sil = silhouette_score(first_flat, first_clusters)
                print(f"  First subgoal silhouette: {first_sil:.3f}")
            if len(set(second_clusters)) > 1:
                second_sil = silhouette_score(second_flat, second_clusters)
                print(f"  Second subgoal silhouette: {second_sil:.3f}")

            # Hierarchy indicator: different subgoal types appear in consistent order
            return (first_clusters, second_clusters), 0.5

        return None, 0

    def compute_hierarchy_metrics(self):
        """
        Compute quantitative metrics to distinguish structure types

        Returns:
            stratification_score: 0-1 (higher = more stratified)
            hierarchy_score: 0-1 (higher = more hierarchical)
        """
        print("\n" + "=" * 60)
        print("COMPUTING STRUCTURE METRICS")
        print("=" * 60)

        metrics = {}

        # 1. Decision point density [stratification indicator]
        decision_points, entropy = self.detect_decision_points()
        decision_density = len(decision_points) / len(self.actions)
        metrics["decision_density"] = decision_density
        print(f"Decision density: {decision_density:.4f}")

        # 2. Subgoal consistency [hierarchy indicator]
        subgoals, ep_decisions = self.extract_subgoal_sequences(decision_points)
        if subgoals and ep_decisions:
            # Check if subgoals cluster across episodes
            subgoal_cluster_score = 0.0
            subgoal_list = [sg for ep_sg in subgoals for sg in ep_sg]
            if subgoal_list and len(subgoal_list) > 5:
                # Flatten states
                if hasattr(subgoal_list[0], "shape") and len(subgoal_list[0].shape) > 1:
                    subgoal_flat = np.array([s.flatten()[:50] for s in subgoal_list])
                else:
                    subgoal_flat = np.array(subgoal_list)

                # Cluster subgoals
                from sklearn.cluster import KMeans

                n_clusters = min(5, len(subgoal_flat))
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters).fit(subgoal_flat)
                    from sklearn.metrics import silhouette_score

                    subgoal_cluster_score = silhouette_score(
                        subgoal_flat, kmeans.labels_
                    )
                    print(f"Subgoal clustering quality: {subgoal_cluster_score:.3f}")
            metrics["subgoal_cluster_score"] = subgoal_cluster_score

        # 3. Action sequence repetition [hierarchy indicator]
        repeated_seqs, seq_score = self.analyze_temporal_abstraction()
        metrics["sequence_repetition"] = seq_score
        print(f"Sequence repetition score: {seq_score:.3f}")

        # 4. Graph community structure [both]
        G, partition, modularity = self.build_state_transition_graph()
        metrics["graph_modularity"] = modularity if modularity else 0
        print(f"Graph modularity: {metrics['graph_modularity']:.3f}")

        # 5. Boundary sharpness [stratification indicator]
        # Use Q-value differences or value function
        if hasattr(self, "q_values") and self.q_values is not None:
            q_diff = (
                np.abs(self.q_values[:, 0] - self.q_values[:, 1])
                if self.q_values.shape[1] >= 2
                else np.zeros(len(self.q_values))
            )
            # Compute gradient of q_diff
            q_gradient = np.abs(np.diff(q_diff))
            boundary_sharpness = np.mean(q_gradient > np.percentile(q_gradient, 90))
            metrics["boundary_sharpness"] = boundary_sharpness
            print(f"Boundary sharpness: {boundary_sharpness:.3f}")
        else:
            metrics["boundary_sharpness"] = 0

        # 6. Causal dependency strength [hierarchy indicator]
        causal_result, causal_score = self.analyze_causal_dependencies()
        metrics["causal_strength"] = causal_score
        print(f"Causal dependency strength: {causal_score:.3f}")

        # Compute combined scores
        # Stratification indicators: decision density, boundary sharpness, modularity
        stratification_score = np.mean(
            [
                metrics["decision_density"] * 5,  # Scale up
                metrics["boundary_sharpness"],
                metrics["graph_modularity"],
            ]
        )

        # Hierarchy indicators: subgoal clustering, sequence repetition, causal strength
        hierarchy_score = np.mean(
            [
                metrics.get("subgoal_cluster_score", 0),
                metrics["sequence_repetition"],
                metrics.get("causal_strength", 0),
            ]
        )

        self.stratification_score = min(1.0, stratification_score)
        self.hierarchy_score = min(1.0, hierarchy_score)

        print(f"\nFinal Scores:")
        print(f"  Stratification score: {self.stratification_score:.3f}")
        print(f"  Hierarchy score: {self.hierarchy_score:.3f}")

        # Determine structure type
        if self.hierarchy_score > self.stratification_score + 0.2:
            self.structure_type = "HIERARCHICAL"
            print("\n✅ Structure type: HIERARCHICAL (multi-level partial order)")
            print("   Evidence: Consistent subgoal ordering, temporal abstraction,")
            print("             causal dependencies between subgoals")
        elif self.stratification_score > self.hierarchy_score + 0.2:
            self.structure_type = "STRATIFIED"
            print("\n✅ Structure type: STRATIFIED (flat regions with boundaries)")
            print("   Evidence: Sharp decision boundaries, region-based partitioning,")
            print("             no consistent subgoal ordering")
        else:
            self.structure_type = "MIXED or UNCLEAR"
            print("\n⚠️ Structure type: MIXED or UNCLEAR")
            print("   Evidence shows both stratification and hierarchy indicators")

        return {
            "metrics": metrics,
            "stratification_score": self.stratification_score,
            "hierarchy_score": self.hierarchy_score,
            "structure_type": self.structure_type,
        }

    def visualize_structure(self):
        """
        Comprehensive visualization of detected structure
        """
        print("\n=== Creating Structure Visualization ===")

        fig = plt.figure(figsize=(20, 15))

        # 1. State space with decision points
        ax1 = fig.add_subplot(2, 3, 1)
        if len(self.states.shape) == 2 and self.states.shape[1] >= 2:
            # Use first two dimensions for visualization
            scatter = ax1.scatter(
                self.states[:, 0],
                self.states[:, 1],
                c=self.actions,
                cmap="tab10",
                s=5,
                alpha=0.5,
            )

            # Highlight decision points
            decision_points, _ = self.detect_decision_points()
            if len(decision_points) > 0:
                ax1.scatter(
                    self.states[decision_points, 0],
                    self.states[decision_points, 1],
                    c="red",
                    s=30,
                    marker="x",
                    label="Decision points",
                )

            ax1.set_xlabel("State dim 1")
            ax1.set_ylabel("State dim 2")
            ax1.set_title("State Space with Decision Points")
            ax1.legend()
        else:
            ax1.text(
                0.5, 0.5, "Cannot visualize high-dim states", ha="center", va="center"
            )

        # 2. Action entropy over time
        ax2 = fig.add_subplot(2, 3, 2)
        decision_points, entropy = self.detect_decision_points()
        ax2.plot(entropy, "b-", alpha=0.7, linewidth=1)
        ax2.scatter(decision_points, entropy[decision_points], c="red", s=20, zorder=5)
        ax2.axhline(y=np.mean(entropy), color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Action Entropy")
        ax2.set_title("Decision Points (High Action Entropy)")

        # 3. Subgoal consistency heatmap
        ax3 = fig.add_subplot(2, 3, 3)
        if hasattr(self, "hidden_features") and len(self.hidden_features) > 0:
            # Use hidden features for subgoal clustering
            from sklearn.manifold import TSNE

            sample_idx = np.random.choice(
                len(self.hidden_features),
                min(500, len(self.hidden_features)),
                replace=False,
            )
            hidden_sample = self.hidden_features[sample_idx]

            tsne = TSNE(n_components=2, random_state=42)
            hidden_2d = tsne.fit_transform(hidden_sample)

            scatter = ax3.scatter(
                hidden_2d[:, 0],
                hidden_2d[:, 1],
                c=self.actions[sample_idx] if len(self.actions) > 0 else None,
                cmap="tab10",
                s=10,
                alpha=0.6,
            )
            ax3.set_title("Hidden Features (t-SNE)")
        else:
            ax3.text(0.5, 0.5, "No hidden features available", ha="center", va="center")

        # 4. Transition graph (simplified)
        ax4 = fig.add_subplot(2, 3, 4)
        G, partition, _ = self.build_state_transition_graph(subsample=200)

        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, iterations=50, seed=42)

            # Color by community if available
            if partition:
                colors = [partition[i] for i in G.nodes()]
                nx.draw_networkx_nodes(
                    G, pos, node_color=colors, node_size=20, cmap="tab10", ax=ax4
                )
            else:
                nx.draw_networkx_nodes(G, pos, node_size=20, ax=ax4)

            nx.draw_networkx_edges(
                G, pos, alpha=0.3, arrows=True, arrowstyle="->", ax=ax4
            )
            ax4.set_title("State Transition Graph")
            ax4.axis("off")

        # 5. Structure type indicator
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.axis("off")

        # Create structure summary
        summary_text = f"""
        STRUCTURE ANALYSIS RESULTS
        
        Environment: {self.env_name}
        
        Stratification Score: {self.stratification_score:.3f}
        Hierarchy Score: {self.hierarchy_score:.3f}
        
        Detected Structure: {self.structure_type}
        
        Key Indicators:
        • Decision Density: {self.stratification_score:.2f}
        • Subgoal Consistency: {self.hierarchy_score:.2f}
        • Temporal Abstraction: {"Yes" if self.hierarchy_score > 0.4 else "No"}
        
        Interpretation:
        {self._get_interpretation()}
        """

        ax5.text(
            0.1,
            0.9,
            summary_text,
            transform=ax5.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 6. Score comparison
        ax6 = fig.add_subplot(2, 3, 6)
        categories = ["Stratification", "Hierarchy"]
        scores = [self.stratification_score, self.hierarchy_score]
        colors = ["lightcoral", "lightblue"]
        bars = ax6.bar(categories, scores, color=colors)
        ax6.set_ylim(0, 1)
        ax6.set_ylabel("Score")
        ax6.set_title("Structure Type Comparison")

        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            f"structure_analysis_{self.env_name}.png", dpi=150, bbox_inches="tight"
        )
        plt.show()
        print(f"Visualization saved to structure_analysis_{self.env_name}.png")

    def _get_interpretation(self):
        """Generate interpretation text based on scores"""
        if self.structure_type == "HIERARCHICAL":
            return (
                "The environment exhibits true hierarchical structure with\n"
                "multiple levels of abstraction. Subgoals appear in consistent\n"
                "order across episodes, and the agent has learned reusable skills.\n"
                "This matches environments like Minecraft or CT-Graph where\n"
                "tasks decompose into prerequisite subgoals."
            )
        elif self.structure_type == "STRATIFIED":
            return (
                "The environment exhibits stratified (layered) structure with\n"
                "distinct regions separated by boundaries. Different regions have\n"
                "different dynamics, but there's no consistent subgoal ordering.\n"
                "This matches environments like Ant Maze or Four Rooms where\n"
                "space is partitioned into regions."
            )
        else:
            return (
                "The structure is mixed or unclear. The environment may have\n"
                "elements of both stratification and hierarchy, or the agent\n"
                "may not have learned to exploit hierarchical structure.\n"
                "Consider collecting more episodes or training longer."
            )

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("=" * 70)
        print(f"HIERARCHICAL VS STRATIFIED STRUCTURE ANALYSIS")
        print(f"Environment: {self.env_name}")
        print("=" * 70)

        # Run all analyses
        results = self.compute_hierarchy_metrics()

        # Visualize
        self.visualize_structure()

        # Save results
        import json

        with open(f"structure_results_{self.env_name}.json", "w") as f:
            # Convert numpy values to Python types
            serializable = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    serializable[k] = {
                        k2: float(v2) if hasattr(v2, "item") else v2
                        for k2, v2 in v.items()
                    }
                else:
                    serializable[k] = float(v) if hasattr(v, "item") else v
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to structure_results_{self.env_name}.json")

        return results


def compare_environments(env_configs):
    """
    Compare structure across multiple environments
    """
    print("\n" + "=" * 70)
    print("ENVIRONMENT STRUCTURE COMPARISON")
    print("=" * 70)

    results = {}

    for env_name, model_path in env_configs.items():
        print(f"\n--- Analyzing {env_name} ---")

        # Load model (you'd need to have trained models)
        from stable_baselines3 import DQN, PPO

        try:
            if "dqn" in model_path.lower():
                model = DQN.load(model_path)
            else:
                model = PPO.load(model_path)

            analyzer = HierarchicalStructureAnalyzer(model, env_name)
            analyzer.collect_experience(n_episodes=100)

            # For environments without Q-values, we can still analyze
            if not hasattr(analyzer, "q_values") or analyzer.q_values is None:
                # Use proxy from action probabilities if possible
                pass

            result = analyzer.run_full_analysis()
            results[env_name] = {
                "stratification": result["stratification_score"],
                "hierarchy": result["hierarchy_score"],
                "type": result["structure_type"],
            }

        except Exception as e:
            print(f"Error analyzing {env_name}: {e}")

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    env_names = list(results.keys())
    x = np.arange(len(env_names))
    width = 0.35

    strat_scores = [results[e]["stratification"] for e in env_names]
    hier_scores = [results[e]["hierarchy"] for e in env_names]

    ax.bar(
        x - width / 2, strat_scores, width, label="Stratification", color="lightcoral"
    )
    ax.bar(x + width / 2, hier_scores, width, label="Hierarchy", color="lightblue")

    ax.set_xlabel("Environment")
    ax.set_ylabel("Score")
    ax.set_title("Structure Type Comparison Across Environments")
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("environment_comparison.png", dpi=150)
    plt.show()

    return results


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("HIERARCHICAL VS STRATIFIED STRUCTURE DETECTION")
    print("=" * 70)
    print("\nThis script analyzes RL environments to distinguish between:")
    print("  • STRATIFIED: Flat regions with boundaries (e.g., Four Rooms, Ant Maze)")
    print("  • HIERARCHICAL: Multi-level partial order (e.g., Minecraft, CT-Graph)")
    print("\nTo use this script:")
    print("  1. Train models on different environments")
    print("  2. Pass model paths to the analyzer")
    print("  3. Compare the stratification vs hierarchy scores")

    # Example configuration (you'll need to provide actual model paths)
    env_configs = {
        # 'CartPole-v1': 'models/dqn_cartpole.zip',
        # 'AntMaze': 'models/ppo_antmaze.zip',
        # 'MiniGrid-FourRooms': 'models/dqn_minigrid.zip',
        # 'Minecraft-like': 'models/ppo_minecraft.zip',
    }

    if env_configs:
        compare_environments(env_configs)
    else:
        print("\nNo environments configured. Please add model paths to test.")
