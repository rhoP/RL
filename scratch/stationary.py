import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gymnasium as gym
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arrow


def get_transition_dynamics(env):
    """
    Extract transition dynamics from Gymnasium FrozenLake environment.
    Returns a dictionary similar to the old env.P format.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    grid_size = int(np.sqrt(n_states))

    # Reconstruct the transition dynamics
    P = {s: {a: [] for a in range(n_actions)} for s in range(n_states)}

    # For each state and action, determine the transition probabilities
    for state in range(n_states):
        for action in range(n_actions):
            # Reset environment to current state
            env.unwrapped.s = state

            # Get all possible transitions for this state-action pair
            transitions = env.unwrapped.P[state][action]

            for prob, next_state, reward, terminated, truncated in transitions:
                P[state][action].append((prob, next_state, reward, terminated or truncated))

    return P


def create_mdp_graph(env, policy, stationary_dist=None):
    """
    Create a graph representation of the MDP with policy and stationary distribution.

    Args:
        env: FrozenLake environment
        policy: Deterministic policy (function or array mapping state -> action)
        stationary_dist: Stationary distribution (optional)

    Returns:
        G: NetworkX graph with node and edge attributes
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    grid_size = int(np.sqrt(n_states))

    # Get transition dynamics
    P = get_transition_dynamics(env)

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for state in range(n_states):
        node_attrs = {
            'state': state,
            'grid_pos': (state % grid_size, state // grid_size),
            'is_hole': False,
            'is_goal': False,
            'stationary_prob': stationary_dist[state] if stationary_dist is not None else 0.0
        }

        # Check if state is hole or goal
        desc = env.unwrapped.desc.flatten()
        if desc[state] == b'H':
            node_attrs['is_hole'] = True
        elif desc[state] == b'G':
            node_attrs['is_goal'] = True

        G.add_node(state, **node_attrs)

    # Add edges based on policy and transitions
    for state in range(n_states):
        if G.nodes[state]['is_hole'] or G.nodes[state]['is_goal']:
            continue  # Terminal states

        # Get action from policy
        if callable(policy):
            action = policy(state)
        elif isinstance(policy, (list, np.ndarray)):
            action = policy[state]
        else:
            raise ValueError("Policy must be callable or array-like")

        # Add edge for the chosen action
        for prob, next_state, reward, done in P[state][action]:
            if prob > 0:  # Only add edges with positive probability
                edge_attrs = {
                    'action': action,
                    'probability': prob,
                    'reward': reward,
                    'done': done
                }
                G.add_edge(state, next_state, **edge_attrs)

    return G


def plot_mdp_graph(env, policy, stationary_dist=None, figsize=(12, 10)):
    """
    Plot the MDP as a graph with stationary distribution as node colors.

    Args:
        env: FrozenLake environment
        policy: Deterministic policy
        stationary_dist: Stationary distribution
        figsize: Figure size
    """
    # Create the graph
    G = create_mdp_graph(env, policy, stationary_dist)
    n_states = env.observation_space.n
    grid_size = int(np.sqrt(n_states))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get node positions (grid layout)
    pos = {node: (data['grid_pos'][0], -data['grid_pos'][1]) for node, data in G.nodes(data=True)}

    # Create colormap for stationary distribution
    if stationary_dist is not None:
        cmap = plt.cm.viridis
        node_colors = [data['stationary_prob'] for _, data in G.nodes(data=True)]
        vmin, vmax = 0, max(node_colors) if max(node_colors) > 0 else 1
    else:
        cmap = plt.cm.Set3
        node_colors = [0] * n_states

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors if stationary_dist is not None else None,
        cmap=cmap,
        node_size=800,
        alpha=0.8,
        ax=ax
    )

    if stationary_dist is not None:
        nodes.set_norm(plt.Normalize(vmin=vmin, vmax=vmax))

    # Draw edges with arrows showing the policy
    for edge in G.edges():
        start, end = edge
        start_pos = pos[start]
        end_pos = pos[end]

        # Draw arrow
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        # Normalize for better arrow appearance
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            dx, dy = dx / length * 0.8, dy / length * 0.8  # Scale for better visibility

        arrow = Arrow(start_pos[0], start_pos[1], dx, dy,
                      width=0.1, color='red', alpha=0.7)
        ax.add_patch(arrow)

    # Add node labels (state numbers and stationary probabilities)
    labels = {}
    for node, data in G.nodes(data=True):
        label = f"S{node}\n"
        if stationary_dist is not None:
            label += f"p={data['stationary_prob']:.3f}"
        labels[node] = label

    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    # Add action labels on edges
    edge_labels = {}
    for start, end, data in G.edges(data=True):
        action_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        edge_labels[(start, end)] = action_map[data['action']]

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)

    # Customize appearance based on state type
    for node, data in G.nodes(data=True):
        if data['is_hole']:
            ax.text(pos[node][0], pos[node][1], 'HOLE',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="circle,pad=0.3", fc="red", alpha=0.5))
        elif data['is_goal']:
            ax.text(pos[node][0], pos[node][1], 'GOAL',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="circle,pad=0.3", fc="green", alpha=0.5))

    # Add colorbar if stationary distribution is provided
    if stationary_dist is not None:
        cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
        cbar.set_label('Stationary Probability', fontsize=12)

    # Set title and layout
    title = "Frozen Lake MDP with Policy"
    if stationary_dist is not None:
        title += f"\nStationary Distribution Entropy: {calculate_entropy(stationary_dist):.3f}"
    ax.set_title(title, fontsize=14, pad=20)

    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-grid_size, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    return fig, ax


def estimate_stationary_distribution(env, policy, num_episodes=1000, max_steps=100):
    """
    Estimate stationary distribution by running episodes and counting state visits.

    Args:
        env: The environment
        policy: The policy to evaluate
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        stationary_dist: Estimated stationary distribution
        state_visits: Raw state visit counts
    """
    n_states = env.observation_space.n
    state_visits = np.zeros(n_states)
    total_visits = 0

    for episode in range(num_episodes):
        state, _ = env.reset()

        for step in range(max_steps):
            state_visits[state] += 1
            total_visits += 1

            # Get action from policy
            if callable(policy):
                action = policy(state)
            elif isinstance(policy, (list, np.ndarray)):
                action = policy[state]
            else:
                raise ValueError("Policy must be callable or array-like")

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

            state = next_state

    stationary_dist = state_visits / total_visits
    return stationary_dist, state_visits


def calculate_entropy(distribution):
    """Calculate entropy of a probability distribution."""
    distribution = np.array(distribution)
    distribution = distribution[distribution > 0]  # Remove zeros
    return -np.sum(distribution * np.log(distribution))


def get_deterministic_policy_from_model(model, env):
    """Extract deterministic policy from SB3 model."""
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)

    for state in range(n_states):
        action, _ = model.predict(state, deterministic=True)
        policy[state] = action

    return policy


# Example usage and complete workflow
def full_workflow_example():
    # Create environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    # Example deterministic policy (you can replace this with your trained policy)
    # Simple policy: always go right
    def simple_policy(state):
        return 2  # Right action

    # Calculate stationary distribution
    stationary_dist, _ = estimate_stationary_distribution(
        env, simple_policy, num_episodes=5000, max_steps=100
    )

    print("Stationary distribution:", stationary_dist)
    print("Entropy:", calculate_entropy(stationary_dist))

    # Plot the MDP graph
    fig, ax = plot_mdp_graph(env, simple_policy, stationary_dist)
    plt.show()

    return stationary_dist


# Run the example
if __name__ == "__main__":
    full_workflow_example()
