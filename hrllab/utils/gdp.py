import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

class GraphMDPVisualizer:
    """A class to visualize MDPs as graphs and track policy evolution during training."""
    
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
        
    def build_mdp_graph(self) -> nx.MultiDiGraph:
        """
        Build a graph representation of the MDP.
        
        Returns:
            A NetworkX MultiDiGraph representing the MDP
        """
        G = nx.MultiDiGraph()
        
        # Get state and action spaces
        if isinstance(self.env.observation_space, spaces.Discrete):
            n_states = self.env.observation_space.n
        else:
            raise ValueError("Only discrete state spaces are supported")
            
        if isinstance(self.env.action_space, spaces.Discrete):
            n_actions = self.env.action_space.n
        else:
            raise ValueError("Only discrete action spaces are supported")
        
        # Add nodes (states)
        for state in range(n_states):
            G.add_node(state, state=state)
        
        # Add edges (actions)
        # For deterministic environments, we can step through each state-action pair
        for state in range(n_states):
            for action in range(n_actions):
                # Set the environment to the current state
                self.env.reset()
                if hasattr(self.env, 's'):  # Some environments use 's' for state
                    self.env.s = state
                elif hasattr(self.env, 'state'):
                    self.env.state = state
                
                # Take the action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Add edge
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
            obs, reward, terminated, truncated, _ = self.env.step(action)
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
            self.build_mdp_graph()
            
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
