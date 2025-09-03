import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import torch as th
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime

class TrainingMetricsCallback(BaseCallback):
    """
    Callback for collecting training metrics during RL training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Update current episode reward and length
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        return True

def plot_training_metrics(episode_rewards: List[float], 
                         episode_lengths: List[float],
                         window_size: int = 10,
                         figsize: Tuple[int, int] = (12, 8)):
    """
    Plot training metrics including rewards and episode lengths.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        window_size: Size of moving average window
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Calculate moving averages
    rewards_ma = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    lengths_ma = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
    
    # Plot episode rewards
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    ax1.plot(range(window_size-1, len(episode_rewards)), rewards_ma, 
            label=f'Moving Average (window={window_size})', color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.3, label='Episode Length')
    ax2.plot(range(window_size-1, len(episode_lengths)), lengths_ma, 
            label=f'Moving Average (window={window_size})', color='red', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_loss_curves(losses: Dict[str, List[float]], 
                    figsize: Tuple[int, int] = (12, 6)):
    """
    Plot loss curves from training.
    
    Args:
        losses: Dictionary of loss arrays (e.g., {'policy_loss': [...], 'value_loss': [...]})
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, len(losses), figsize=figsize)
    if len(losses) == 1:
        axes = [axes]
    
    for i, (loss_name, loss_values) in enumerate(losses.items()):
        axes[i].plot(loss_values)
        axes[i].set_xlabel('Update Step')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(f'{loss_name.title()} Loss')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_testing_metrics(test_episodes: List[Dict[str, Any]],
                        figsize: Tuple[int, int] = (12, 8)):
    """
    Plot testing metrics including rewards, lengths, and success rate.
    
    Args:
        test_episodes: List of dictionaries with test episode results
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    rewards = [ep['reward'] for ep in test_episodes]
    lengths = [ep['length'] for ep in test_episodes]
    successes = [1 if ep.get('success', False) else 0 for ep in test_episodes]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Plot test rewards
    ax1.bar(range(len(rewards)), rewards)
    ax1.set_xlabel('Test Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Test Episode Rewards')
    ax1.axhline(y=np.mean(rewards), color='r', linestyle='--', 
               label=f'Mean: {np.mean(rewards):.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot test lengths
    ax2.bar(range(len(lengths)), lengths)
    ax2.set_xlabel('Test Episode')
    ax2.set_ylabel('Length')
    ax2.set_title('Test Episode Lengths')
    ax2.axhline(y=np.mean(lengths), color='r', linestyle='--', 
               label=f'Mean: {np.mean(lengths):.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot success rate
    success_rate = np.mean(successes) * 100
    ax3.bar(['Success', 'Failure'], [success_rate, 100-success_rate], 
           color=['green', 'red'])
    ax3.set_ylabel('Percentage')
    ax3.set_title(f'Success Rate: {success_rate:.1f}%')
    
    plt.tight_layout()
    return fig

def plot_comparison_metrics(metrics_dict: Dict[str, Dict[str, List[float]]],
                           metric_name: str = 'rewards',
                           figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of metrics across different experiments or algorithms.
    
    Args:
        metrics_dict: Dictionary of metrics for different experiments
        metric_name: Name of metric to plot ('rewards', 'lengths', etc.)
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for exp_name, metrics in metrics_dict.items():
        if metric_name in metrics:
            values = metrics[metric_name]
            ax.plot(values, label=exp_name, alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric_name.title())
    ax.set_title(f'Comparison of {metric_name.title()} Across Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def save_training_plots(episode_rewards: List[float],
                       episode_lengths: List[float],
                       save_dir: str = './training_plots',
                       experiment_name: str = 'experiment'):
    """
    Save training plots to directory.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        save_dir: Directory to save plots
        experiment_name: Name of experiment for file naming
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot and save training metrics
    fig = plot_training_metrics(episode_rewards, episode_lengths)
    fig.savefig(f'{save_dir}/{experiment_name}_training_metrics_{timestamp}.png', 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Training plots saved to {save_dir}")

def test_policy(env: gym.Env, model, num_episodes: int = 10, 
               render: bool = False) -> List[Dict[str, Any]]:
    """
    Test a trained policy on the environment.
    
    Args:
        env: Gymnasium environment
        model: Trained RL model
        num_episodes: Number of test episodes
        render: Whether to render the environment
        
    Returns:
        List of test episode results
    """
    test_results = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        test_results.append({
            'episode': episode,
            'reward': total_reward,
            'length': steps,
            'success': terminated  # Assuming termination means success
        })
    
    return test_results

def plot_learning_curve(episode_rewards: List[float], 
                       confidence_interval: float = 0.95,
                       figsize: Tuple[int, int] = (10, 6)):
    """
    Plot learning curve with confidence interval.
    
    Args:
        episode_rewards: List of episode rewards
        confidence_interval: Confidence interval for shading (0-1)
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate moving average and confidence interval
    window_size = max(1, len(episode_rewards) // 20)
    rewards_ma = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    
    # Calculate confidence interval
    std_err = np.std(episode_rewards) / np.sqrt(window_size)
    ci = std_err * 1.96  # 95% confidence interval
    
    # Plot
    x_values = range(window_size-1, len(episode_rewards))
    ax.plot(x_values, rewards_ma, label='Moving Average', linewidth=2)
    ax.fill_between(x_values, rewards_ma - ci, rewards_ma + ci, 
                   alpha=0.3, label=f'{confidence_interval*100:.0f}% CI')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curve with Confidence Interval')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig