import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch as th
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 





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
        
    def _process_state_action_pair(self, args):
        """Process a single state-action pair (for parallelization)."""
        state, action, env_class = args
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
            results = list(executor.map(self._process_state_action_pair, tasks))
        
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
    
    def get_policy_trajectory(self, model, max_steps: int = 100) -> List[int]:
        """
        Get a trajectory following the current policy.
        
        Args:
            model: The RL model
            max_steps: Maximum number of steps in the trajectory
            
        Returns:
            List of states visited
        """
        obs, _ = self.env.reset()
        trajectory = [int(obs)]
        
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.env.step(int(action))
            trajectory.append(int(obs))
            
            if terminated or truncated:
                break
                
        return trajectory
    
    class PolicyTrackingCallback(BaseCallback):
        """Callback to track policies during training."""
        
        def __init__(self, visualizer, save_freq: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.visualizer = visualizer
            self.save_freq = save_freq
        
        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                # Save current policy
                policy_state = {
                    'policy': self.model.policy.state_dict(),
                    'step': self.n_calls
                }
                trajectory = self.visualizer.get_policy_trajectory(self.model)
                self.visualizer.policies.append(policy_state)
                self.visualizer.trajectories.append(trajectory)
                
            return True
    
    def plot_mdp_graph(self, 
                      node_size: int = 1000,
                      font_size: int = 10,
                      figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
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
        
        # Draw edges with colors based on action
        edge_colors = []
        for u, v, data in self.graph.edges(data=True):
            action = data['action']
            # Map action to color
            edge_colors.append(plt.cm.Set1(action % 8))
            
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, 
                              arrows=True, arrowsize=20, ax=ax)
        
        # Draw edge labels (action and reward)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            action = data['action']
            reward = data['reward']
            done = data['done']
            edge_labels[(u, v)] = f"a:{action}\nr:{reward:.1f}\n{'T' if done else 'F'}"
            
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, 
                                    font_size=font_size-2, ax=ax)
        
        # Create legend for actions
        action_legend = [Patch(color=plt.cm.Set1(i % 8), label=f'Action {i}') 
                        for i in range(self.env.action_space.n)]
        ax.legend(handles=action_legend, loc='upper right')
        
        ax.set_title("MDP Graph Representation")
        ax.axis('off')
        
        return fig
    
    def plot_policy_evolution(self,
                             figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
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



def make_env():
    """Create a function that returns a new environment instance."""
    env = TimeLimit(gym.make('FrozenLake-v1', is_slippery=False), max_episode_steps=100)
    return env


def main():
    # Create environment
    env = TimeLimit(gym.make('FrozenLake-v1', is_slippery=False, render_mode='human'), max_episode_steps=100)
    
    # Create visualizer
    visualizer = ParallelGraphMDPVisualizer(env)
    
    # Build MDP graph in parallel
    graph = visualizer.build_mdp_graph_parallel()
    
    # Plot initial MDP graph
    fig1 = visualizer.plot_mdp_graph()
    plt.savefig('mdp_graph_parallel.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create callback for tracking policies
    callback = visualizer.PolicyTrackingCallback(visualizer, save_freq=5000)
    
    # Create multiple environments for parallel training
    n_envs = 16  # Number of parallel environments
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Create and train model with GPU acceleration
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=0.001,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # Train the model
    print("Training model with parallel environments...")
    model.learn(total_timesteps=50000, callback=callback)
    
    # Plot policy evolution
    fig2 = visualizer.plot_policy_evolution()
    plt.savefig('policy_evolution_parallel.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test the final policy
    print("Testing final policy:")
    obs, _ = env.reset()
    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        env.render()
        print(f"Step {i}: Action={action}, Reward={reward}, Terminated={terminated}")
        if terminated or truncated:
            break

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    main()