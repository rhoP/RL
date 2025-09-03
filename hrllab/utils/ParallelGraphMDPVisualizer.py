import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from stable_baselines3.common.callbacks import BaseCallback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import Pool, cpu_count
import time
import copy
from hrllab.utils.custom_wrappers import CustomRewardWrapper


def make_env():
    """Create a function that returns a new environment instance."""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    env = TimeLimit(env, max_episode_steps=100)
    return CustomRewardWrapper(env)


def _process_state_action_pair(args):
    """Process a single state-action pair (for parallelization)."""
    state, action, env_class = args

    # Create a new environment instance without TimeLimit wrapper
    env = env_class()

    # Set the environment to the current state
    env.reset()
    if hasattr(env, 's'):  # Some environments use 's' for state
        env.s = state
    elif hasattr(env, 'state'):
        env.state = state

    # Take the action
    next_state, reward, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated

    return state, action, next_state, reward, done


def get_policy_trajectory(model, max_steps: int = 100) -> List[int]:
    """
    Get a trajectory following the current policy.

    Args:
        model: The RL model
        max_steps: Maximum number of steps in the trajectory

    Returns:
        List of states visited
    """
    # Create a separate environment for trajectory generation
    env = make_env()
    obs, _ = env.reset()
    trajectory = [int(obs)]

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        trajectory.append(int(obs))

        if terminated or truncated:
            break

    return trajectory


class ParallelGraphMDPVisualizer:
    """A class to visualize MDPs as graphs and track policy evolution during training with parallelization."""

    def __init__(self, env: gym.Env):
        """
        Initialize the Graph MDP Visualizer.
        
        Args:
            env: A Gymnasium environment
        """
        self.env = env
        self.graph = None
        self.policies = []  # Store policies at different training stages
        self.trajectories = []  # Store trajectories at different training stages
        self.env_class = self.env.unwrapped.__class__ if hasattr(self.env, 'unwrapped') else self.env.__class__

    def build_mdp_graph_parallel(self) -> nx.MultiDiGraph:
        """
        Build a graph representation of the MDP using parallel processing.
        
        Returns:
            A NetworkX MultiDiGraph representing the MDP
        """
        G = nx.MultiDiGraph()

        # Get state and action spaces from the unwrapped environment
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env

        if isinstance(unwrapped_env.observation_space, spaces.Discrete):
            n_states = unwrapped_env.observation_space.n
        else:
            raise ValueError("Only discrete state spaces are supported")

        if isinstance(unwrapped_env.action_space, spaces.Discrete):
            n_actions = unwrapped_env.action_space.n
        else:
            raise ValueError("Only discrete action spaces are supported")

        # Add nodes (states)
        for state in range(n_states):
            G.add_node(state, state=state)

        # Prepare tasks for parallel processing
        tasks = []

        for state in range(n_states):
            for action in range(n_actions):
                tasks.append((state, action, self.env_class))

        # Process tasks in parallel
        print(f"Processing {len(tasks)} state-action pairs with {min(cpu_count(), 8)} cores...")
        start_time = time.time()

        # Use a limited number of workers to avoid overloading the system
        max_workers = min(cpu_count(), 8)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_state_action_pair, tasks))

        end_time = time.time()
        print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")

        # Add edges to graph
        for state, action, next_state, reward, done in results:
            G.add_edge(
                state,
                next_state,
                action=action,
                reward=reward,
                done=done
            )

        self.graph = G
        return G

    class PolicyTrackingCallback(BaseCallback):
        """Callback to track policies during training."""

        def __init__(self, visualizer, save_freq: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.visualizer = visualizer
            self.save_freq = save_freq

        def _on_step(self) -> bool:
            # Save policy at regular intervals
            if self.n_calls % self.save_freq == 0:
                try:
                    # Save current policy state
                    policy_state = {
                        'policy': copy.deepcopy(self.model.policy.state_dict()),
                        'step': self.num_timesteps
                    }

                    # Get trajectory with current policy
                    trajectory = self.visualizer.get_policy_trajectory(self.model)

                    # Store policy and trajectory
                    self.visualizer.policies.append(policy_state)
                    self.visualizer.trajectories.append(trajectory)

                    print(f"Saved policy at step {self.num_timesteps}")

                except Exception as e:
                    print(f"Error saving policy: {e}")

            return True

    def plot_mdp_graph(self,
                       node_size: int = 1000,
                       font_size: int = 10,
                       figsize: Tuple[int, int] = (12, 10)):
        """
        Plot the MDP graph.
        
        Args:
            node_size: Size of nodes in the plot
            font_size: Font size for labels
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        if self.graph is None:
            print("Building MDP graph in parallel...")
            self.build_mdp_graph_parallel()

        fig, ax = plt.subplots(figsize=figsize)

        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size,
                               node_color='lightblue', ax=ax)

        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=font_size, ax=ax)

        # Calculate transition probabilities for state-action pairs
        transition_probs = {}
        for state in self.graph.nodes():
            # Count total transitions from this state for each action
            action_counts = {}
            for u, v, data in self.graph.edges(data=True):
                if u == state:
                    action = data['action']
                    action_counts[action] = action_counts.get(action, 0) + 1

            # Calculate probabilities
            for u, v, data in self.graph.edges(data=True):
                if u == state:
                    action = data['action']
                    total = action_counts[action]
                    prob = 1.0 / total if total > 0 else 0.0
                    transition_probs[(u, v, action)] = prob

        # Draw edges with colors based on action
        edge_colors = []
        for u, v, data in self.graph.edges(data=True):
            action = data['action']
            # Map action to color
            edge_colors.append(plt.cm.Set1(action % 8))

        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors,
                               arrows=True, arrowsize=20, ax=ax)

        # Draw edge labels (probability, action and reward)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            action = data['action']
            reward = data['reward']
            prob = transition_probs.get((u, v, action), 0.0)
            edge_labels[(u, v)] = f"a:{action}\nr:{reward:.1f}\np:{prob:.2f}"

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels,
                                     font_size=font_size - 2, ax=ax)

        # Create legend for actions
        action_legend = [Patch(color=plt.cm.Set1(i % 8), label=f'Action {i}')
                         for i in range(self.env.action_space.n)]
        ax.legend(handles=action_legend, loc='upper right')

        ax.set_title("MDP Graph Representation")
        ax.axis('off')

        return fig

    def plot_policy_evolution(self,
                              figsize: Tuple[int, int] = (12, 10)):
        """
        Plot the evolution of policies during training.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        if not self.policies or not self.trajectories:
            raise ValueError("No policies or trajectories saved. Train a model first.")

        fig, ax = plt.subplots(figsize=figsize)

        # Get graph layout
        pos = nx.spring_layout(self.graph, seed=42)

        # Draw the base graph
        nx.draw_networkx_nodes(self.graph, pos, node_size=800,
                               node_color='lightgray', ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=10, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edge_color='lightgray',
                               arrows=True, arrowsize=15, ax=ax)

        # Draw policy trajectories with different colors
        colors = list(mcolors.TABLEAU_COLORS.values())

        for i, trajectory in enumerate(self.trajectories):
            # Create edges from trajectory
            trajectory_edges = list(zip(trajectory[:-1], trajectory[1:]))

            # Draw the trajectory
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=trajectory_edges,
                edge_color=colors[i % len(colors)],
                width=3, arrows=True, arrowsize=25, ax=ax,
                connectionstyle="arc3,rad=0.1"  # Curve the edges slightly
            )

            # Draw nodes in the trajectory
            trajectory_nodes = list(set(trajectory))
            nx.draw_networkx_nodes(
                self.graph, pos, nodelist=trajectory_nodes,
                node_color=colors[i % len(colors)], node_size=800, ax=ax
            )

        # Create legend
        legend_elements = [
            Patch(facecolor=colors[i % len(colors)],
                  label=f'Step {self.policies[i]["step"]}')
            for i in range(len(self.policies))
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title("Policy Evolution During Training")
        ax.axis('off')

        return fig
