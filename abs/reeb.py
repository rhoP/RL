import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, linkage
import networkx as nx
from sklearn.manifold import MDS
import warnings

warnings.filterwarnings("ignore")


class ValueFunctionReebGraph:
    """
    Construct and visualize the Reeb graph of a value function over the state space.
    The Reeb graph tracks connected components of level sets of the value function.
    """

    def __init__(self, env, policy, value_function=None, n_samples=1000, device="cpu"):
        self.env = env
        self.policy = policy
        self.device = device
        self.n_samples = n_samples

        # Determine state dimension
        if env.observation_space.shape:
            self.state_dim = env.observation_space.shape[0]
            # Get state bounds
            self.state_low = env.observation_space.low
            self.state_high = env.observation_space.high

            # Replace infinities with reasonable bounds
            self.state_low = np.where(np.isfinite(self.state_low), self.state_low, -10)
            self.state_high = np.where(
                np.isfinite(self.state_high), self.state_high, 10
            )
        else:
            self.state_dim = 1
            self.state_low = 0
            self.state_high = 15  # change this for discrete

        # Use provided value function or estimate from policy
        self.value_function = value_function or self._estimate_value_function

    def _estimate_value_function(self, states):
        """Estimate value of states using the policy."""
        values = []
        states = np.array(states).reshape(-1, self.state_dim)

        for state in states:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Get action from policy
                if hasattr(self.policy, "predict"):
                    action, _ = self.policy.predict(state, deterministic=True)
                    action_tensor = (
                        torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    )
                else:
                    action_tensor = self.policy(state_tensor)
                    action = action_tensor.cpu().numpy()[0]

                # Try different methods to get value
                value = None

                # Method 1: Use critic if available (for SAC/TD3)
                if hasattr(self.policy, "critic"):
                    try:
                        # For SAC with twin critics
                        if isinstance(self.policy.critic, nn.Module):
                            # Single critic
                            if hasattr(self.policy.critic, "forward"):
                                # Check if critic expects actions
                                try:
                                    value = (
                                        self.policy.critic(state_tensor, action_tensor)[
                                            0, 0
                                        ]
                                        .cpu()
                                        .numpy()
                                    )
                                except:
                                    # Try without actions
                                    value = (
                                        self.policy.critic(state_tensor)[0, 0]
                                        .cpu()
                                        .numpy()
                                    )
                        elif isinstance(self.policy.critic, tuple) or isinstance(
                            self.policy.critic, list
                        ):
                            # Twin critics (SAC)
                            values_list = []
                            for critic in self.policy.critic:
                                try:
                                    v = (
                                        critic(state_tensor, action_tensor)[0, 0]
                                        .cpu()
                                        .numpy()
                                    )
                                except:
                                    v = critic(state_tensor)[0, 0].cpu().numpy()
                                values_list.append(v)
                            value = np.min(
                                values_list
                            )  # Take minimum for conservative estimate
                    except Exception as e:
                        print(f"Critic error: {e}")
                        value = None

                # Method 2: Use value network if available (for PPO)
                if value is None and hasattr(self.policy, "value_net"):
                    try:
                        value = self.policy.value_net(state_tensor)[0, 0].cpu().numpy()
                    except:
                        try:
                            value = self.policy.value_net(state_tensor).cpu().numpy()[0]
                        except:
                            pass

                # Method 3: Use predict_values if available
                if value is None and hasattr(self.policy, "predict_values"):
                    try:
                        value = (
                            self.policy.predict_values(state_tensor).cpu().numpy()[0, 0]
                        )
                    except:
                        pass

                # Method 4: Fallback - use negative action magnitude (heuristic)
                if value is None:
                    # For control tasks, often want to minimize action effort
                    value = -np.linalg.norm(action)

                values.append(float(value))

        return np.array(values)

    def sample_state_space(self, n_samples=None):
        """Uniformly sample states from the environment's state space."""
        if n_samples is None:
            n_samples = self.n_samples

        samples = []
        for _ in range(n_samples):
            # Uniform sampling within bounds
            state = np.random.uniform(self.state_low, self.state_high)
            samples.append(state)

        return np.array(samples)

    def compute_reeb_graph(self, n_levels=20, connectivity_radius=None):
        """
        Compute the Reeb graph of the value function.

        Args:
            n_levels: Number of level sets to consider
            connectivity_radius: Radius for connecting nearby states

        Returns:
            graph: NetworkX graph object with Reeb structure
            node_data: Dictionary with node attributes
        """
        # Sample state space
        states = self.sample_state_space()

        # Compute values for all states
        print("Estimating values for sampled states...")
        values = self._estimate_value_function(states)

        # Handle NaN or Inf values
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

        # Normalize values to [0, 1] for level set partitioning
        v_min, v_max = values.min(), values.max()
        if v_max - v_min < 1e-6:
            v_max = v_min + 1.0
        values_norm = (values - v_min) / (v_max - v_min)
        values_norm = np.clip(values_norm, 0, 1)

        # Create level sets
        level_boundaries = np.linspace(0, 1, n_levels + 1)
        level_sets = []

        for i in range(n_levels):
            mask = (values_norm >= level_boundaries[i]) & (
                values_norm < level_boundaries[i + 1]
            )
            level_states = states[mask]
            level_values = values[mask]

            if len(level_states) > 0:
                level_sets.append(
                    {
                        "states": level_states,
                        "values": level_values,
                        "level": i,
                        "value_range": (
                            level_boundaries[i] * (v_max - v_min) + v_min,
                            level_boundaries[i + 1] * (v_max - v_min) + v_min,
                        ),
                    }
                )

        print(f"Created {len(level_sets)} non-empty level sets")

        # Cluster each level set into connected components
        if connectivity_radius is None:
            # Estimate connectivity radius based on sampling density
            if len(states) > 1:
                state_span = np.max(states, axis=0) - np.min(states, axis=0)
                state_span = np.where(state_span > 0, state_span, 1.0)
                connectivity_radius = 0.1 * np.mean(state_span)
            else:
                connectivity_radius = 0.1

        # Build graph nodes (connected components)
        nodes = []
        node_id = 0

        for level_idx, level_data in enumerate(level_sets):
            if len(level_data["states"]) < 2:
                # Single state or empty level set
                if len(level_data["states"]) == 1:
                    nodes.append(
                        {
                            "id": node_id,
                            "level": level_idx,
                            "centroid": level_data["states"][0],
                            "mean_value": level_data["values"][0],
                            "size": 1,
                            "states": level_data["states"],
                        }
                    )
                    node_id += 1
                continue

            # Cluster states in this level set
            try:
                # Use hierarchical clustering
                dist_matrix = pdist(level_data["states"])
                linkage_matrix = linkage(dist_matrix, method="single")

                # Determine clusters based on connectivity radius
                max_dist = connectivity_radius * np.sqrt(self.state_dim)
                clusters = fcluster(linkage_matrix, max_dist, criterion="distance")

                # Group by cluster
                unique_clusters = np.unique(clusters)

                for cluster_id in unique_clusters:
                    cluster_mask = clusters == cluster_id
                    comp_states = level_data["states"][cluster_mask]
                    comp_vals = level_data["values"][cluster_mask]

                    if len(comp_states) > 0:
                        # Node centroid and representative value
                        centroid = np.mean(comp_states, axis=0)
                        mean_value = np.mean(comp_vals)

                        nodes.append(
                            {
                                "id": node_id,
                                "level": level_idx,
                                "centroid": centroid,
                                "mean_value": mean_value,
                                "size": len(comp_states),
                                "states": comp_states,
                            }
                        )
                        node_id += 1
            except Exception as e:
                print(f"Clustering error at level {level_idx}: {e}")
                # Fallback: treat all states as one component
                centroid = np.mean(level_data["states"], axis=0)
                mean_value = np.mean(level_data["values"])

                nodes.append(
                    {
                        "id": node_id,
                        "level": level_idx,
                        "centroid": centroid,
                        "mean_value": mean_value,
                        "size": len(level_data["states"]),
                        "states": level_data["states"],
                    }
                )
                node_id += 1

        print(f"Created {len(nodes)} graph nodes")

        # Build graph edges (connect components between adjacent levels)
        G = nx.Graph()

        # Add nodes
        for node in nodes:
            G.add_node(
                node["id"],
                level=node["level"],
                centroid=node["centroid"],
                value=node["mean_value"],
                size=node["size"],
            )

        # Connect nodes between adjacent levels if they're close
        edge_count = 0
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if node2["level"] == node1["level"] + 1:
                    # Check if components are connected (close in state space)
                    dist = np.linalg.norm(node1["centroid"] - node2["centroid"])
                    if dist < 2 * connectivity_radius:
                        G.add_edge(node1["id"], node2["id"], weight=1.0 / (1.0 + dist))
                        edge_count += 1

        print(f"Created {edge_count} graph edges")

        return G, {"nodes": nodes, "values": values, "states": states}

    def plot_reeb_graph_2d(
        self, graph, node_data, ax=None, projection_2d=True, highlight_extrema=True
    ):
        """Plot the Reeb graph in 2D."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        if graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
            return ax

        # Get node positions for visualization
        pos = {}
        node_values = []
        node_sizes = []

        # Collect all centroids
        centroids = []
        node_list = list(graph.nodes())

        for node_id in node_list:
            node = graph.nodes[node_id]
            centroids.append(node["centroid"])
            node_values.append(node["value"])
            node_sizes.append(node["size"] * 50)

        centroids = np.array(centroids)

        # Project to 2D if needed
        if projection_2d and self.state_dim > 2:
            if len(centroids) > 1:
                try:
                    projector = MDS(n_components=2, random_state=42, n_init=1)
                    projected = projector.fit_transform(centroids)
                except:
                    # Fallback to first 2 dimensions
                    projected = centroids[:, :2]
            else:
                projected = centroids[:, :2]

            for i, node_id in enumerate(node_list):
                pos[node_id] = projected[i]
        else:
            # Use first 2 dimensions
            for i, node_id in enumerate(node_list):
                pos[node_id] = centroids[i, :2]

        # Normalize node values for coloring
        node_values = np.array(node_values)
        if len(node_values) > 1:
            v_min, v_max = node_values.min(), node_values.max()
            if v_max > v_min:
                node_colors = (node_values - v_min) / (v_max - v_min)
            else:
                node_colors = np.zeros_like(node_values)
        else:
            node_colors = [0.5]

        # Draw graph
        nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax, edge_color="gray", width=1)

        # Draw nodes
        nodes_draw = nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap="viridis",
            alpha=0.8,
            ax=ax,
        )

        # Add colorbar
        if nodes_draw is not None and len(node_values) > 1:
            plt.colorbar(nodes_draw, ax=ax, label="Normalized Value")

        # Highlight extremal nodes (min/max value)
        if highlight_extrema and len(node_values) > 1:
            max_idx = node_list[np.argmax(node_values)]
            min_idx = node_list[np.argmin(node_values)]

            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[max_idx],
                node_color="red",
                node_size=300,
                node_shape="*",
                ax=ax,
                label="Maximum Value",
            )

            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[min_idx],
                node_color="blue",
                node_size=300,
                node_shape="*",
                ax=ax,
                label="Minimum Value",
            )

            ax.legend()

        ax.set_title(f"Reeb Graph of Value Function ({self.env.spec.id})")
        ax.set_xlabel("Component 1" if projection_2d else "State Dimension 1")
        ax.set_ylabel("Component 2" if projection_2d else "State Dimension 2")
        ax.grid(True, alpha=0.3)

        return ax

    def analyze_value_topology(self, graph, node_data):
        """Analyze topological features of the value landscape."""
        analysis = {}

        if graph.number_of_nodes() == 0:
            analysis["n_nodes"] = 0
            analysis["n_edges"] = 0
            analysis["n_components"] = 0
            analysis["critical_points"] = 0
            analysis["local_extrema"] = 0
            analysis["cycles"] = 0
            return analysis

        # Number of connected components
        analysis["n_components"] = nx.number_connected_components(graph)

        # Graph characteristics
        analysis["n_nodes"] = graph.number_of_nodes()
        analysis["n_edges"] = graph.number_of_edges()

        if graph.number_of_nodes() > 1:
            # Value range
            node_values = [graph.nodes[n]["value"] for n in graph.nodes()]
            analysis["value_range"] = (
                float(np.min(node_values)),
                float(np.max(node_values)),
            )
            analysis["value_span"] = float(np.max(node_values) - np.min(node_values))

            # Find critical points (nodes with degree != 2 in Reeb graph)
            degrees = dict(graph.degree())
            critical_nodes = {n: d for n, d in degrees.items() if d != 2}
            analysis["critical_points"] = len(critical_nodes)

            # Local extrema (degree 1 nodes are leaves)
            leaves = [n for n, d in degrees.items() if d == 1]
            analysis["local_extrema"] = len(leaves)

            # Cycles in the graph indicate interesting topology
            try:
                cycles = nx.cycle_basis(graph)
                analysis["cycles"] = len(cycles)
            except:
                analysis["cycles"] = 0

        return analysis


def run_reeb_analysis(env_id, algorithm="SAC", n_samples=300, n_levels=10):
    """Run Reeb graph analysis for a trained policy."""
    print(f"\n{'=' * 60}")
    print(f"Reeb Graph Analysis: {algorithm} on {env_id}")
    print("=" * 60)

    # Train policy
    env = gym.make(env_id)

    if algorithm == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001)
    elif algorithm == "TD3":
        model = TD3("MlpPolicy", env, verbose=1)
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Quick training
    print("Training policy...")
    # model.learn(total_timesteps=10000)  # Reduced for faster execution

    model.load("sac_pendulum")
    print("saved sac model for pendulum")

    # Create Reeb graph analyzer
    reeb = ValueFunctionReebGraph(env, model.policy, n_samples=n_samples)

    # Compute Reeb graph
    print("Computing Reeb graph...")
    graph, node_data = reeb.compute_reeb_graph(n_levels=n_levels)

    # Analyze topology
    analysis = reeb.analyze_value_topology(graph, node_data)

    # Plot results
    fig = plt.figure(figsize=(15, 5))

    # 2D projection plot
    ax1 = fig.add_subplot(121)
    reeb.plot_reeb_graph_2d(graph, node_data, ax=ax1)

    # Topology summary
    ax2 = fig.add_subplot(122)
    ax2.axis("off")

    summary_text = f"""Topological Analysis:
    
Nodes: {analysis.get("n_nodes", "N/A")}
Edges: {analysis.get("n_edges", "N/A")}
Connected Components: {analysis.get("n_components", "N/A")}

Value Range: [{analysis.get("value_range", ("N/A", "N/A"))[0]:.2f}, 
             {analysis.get("value_range", ("N/A", "N/A"))[1]:.2f}]
Value Span: {analysis.get("value_span", "N/A"):.2f}

Critical Points: {analysis.get("critical_points", "N/A")}
Local Extrema: {analysis.get("local_extrema", "N/A")}
Cycles: {analysis.get("cycles", "N/A")}

Algorithm: {algorithm}
Environment: {env_id}
Samples: {n_samples}
Level Sets: {n_levels}
    """

    ax2.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax2.set_title("Topology Summary")

    plt.suptitle(f"Value Function Topology - {algorithm} on {env_id}")
    plt.tight_layout()
    plt.savefig(f"reeb_graph_{env_id}_{algorithm}.png", dpi=150, bbox_inches="tight")
    plt.show()

    env.close()

    return graph, node_data, analysis


# Main execution
if __name__ == "__main__":
    # Configuration - using environments that work well
    ENV_IDS = [
        "FrozenLake-v1",
        # "Pendulum-v1",
        # "MountainCarContinuous-v0",
    ]

    ALGORITHMS = [  # "SAC", "TD3",
        "PPO"
    ]

    # Run analyses
    for env_id in ENV_IDS:
        for algo in ALGORITHMS:
            try:
                print(f"\n{'=' * 60}")
                print(f"Running analysis for {algo} on {env_id}")
                print("=" * 60)

                graph, data, analysis = run_reeb_analysis(
                    env_id,
                    algorithm=algo,
                    n_samples=200,  # Reduced for speed
                    n_levels=8,  # Reduced for speed
                )

                print(f"\nResults for {algo} on {env_id}:")
                for key, value in analysis.items():
                    print(f"  {key}: {value}")

            except Exception as e:
                print(f"Error analyzing {algo} on {env_id}: {e}")
                import traceback

                traceback.print_exc()
