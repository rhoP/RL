import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class BackwardGraphBuilder:
    """Builds a graph backwards from terminal states during training for vectorized environments."""

    def __init__(self, n_envs):
        self.graph = nx.DiGraph()
        self.episode_histories = [[] for _ in range(n_envs)]  # Separate history for each environment
        self.terminal_states = set()
        self.n_envs = n_envs

    def record_step(self, env_infos):
        """
        Record steps for all environments in the vectorized environment.

        Args:
            env_infos: List of dictionaries with info for each environment:
                [{'state': s, 'action': a, 'reward': r, 'next_state': ns, 'done': d, 'env_idx': i}, ...]
        """
        for info in env_infos:
            env_idx = info['env_idx']
            state = info['state']
            action = info['action']
            reward = info['reward']
            next_state = info['next_state']
            done = info['done']

            # Record step for this environment
            self.episode_histories[env_idx].append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

            if done:
                self._build_backward_graph(env_idx)
                self.episode_histories[env_idx] = []  # Reset for next episode

    def _build_backward_graph(self, env_idx):
        """Build graph backwards from terminal state for a specific environment."""
        episode_history = self.episode_histories[env_idx]
        if not episode_history:
            return

        # Start from the terminal state and work backwards
        for i in range(len(episode_history) - 1, -1, -1):
            step = episode_history[i]
            state = step['state']
            action = step['action']
            reward = step['reward']
            next_state = step['next_state']
            done = step['done']

            # Add nodes
            if state not in self.graph:
                self.graph.add_node(state, state=state)
            if next_state not in self.graph:
                self.graph.add_node(next_state, state=next_state)

            # Add edge backwards: from next_state to state
            self.graph.add_edge(
                next_state,  # Target node (where we came from)
                state,  # Source node (where we're going to)
                action=action,
                reward=reward,
                done=done,
                direction="backward",
                env_idx=env_idx
            )

            if done:
                self.terminal_states.add(next_state)

    def get_graph(self):
        """Get the current backward graph."""
        return self.graph

    def plot_backward_graph(self, figsize=(12, 10)):
        """Plot the backward graph with terminal states highlighted."""
        if self.graph.number_of_nodes() == 0:
            print("No graph data available")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, seed=42)

        # Color nodes: terminal states in green, others in blue
        node_colors = []
        for node in self.graph.nodes():
            if node in self.terminal_states:
                node_colors.append('lightgreen')  # Terminal state
            else:
                node_colors.append('lightblue')  # Regular state

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=800,
                               node_color=node_colors, ax=ax)

        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, ax=ax)

        # Draw edges with colors based on action
        edge_colors = []
        for u, v, data in self.graph.edges(data=True):
            action = data['action']
            edge_colors.append(plt.cm.Set1(action % 8))

        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors,
                               arrows=True, arrowsize=20, ax=ax)

        # Draw edge labels
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            action = data['action']
            reward = data['reward']
            done = data['done']
            edge_labels[(u, v)] = f"a:{action}\nr:{reward:.1f}\n{'T' if done else 'F'}"

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels,
                                     font_size=8, ax=ax)

        # Create legend
        action_legend = [Patch(color=plt.cm.Set1(i % 8), label=f'Action {i}')
                         for i in range(4)]  # Assuming 4 actions for FrozenLake

        node_legend = [
            Patch(facecolor='lightblue', label='State'),
            Patch(facecolor='lightgreen', label='Terminal State')
        ]

        ax.legend(handles=action_legend + node_legend, loc='upper right')
        ax.set_title("Backward Graph (Shows which states lead to which outcomes)")
        ax.axis('off')

        return fig


