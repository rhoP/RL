import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from plots import TrainingMetricsCallback, test_policy, plot_testing_metrics

def setup_experiment(env_name: str, algorithm: str = 'PPO', 
                    policy_kwargs: Optional[Dict] = None,
                    **kwargs) -> Any:
    """
    Set up RL experiment with given environment and algorithm.
    
    Args:
        env_name: Name of Gymnasium environment
        algorithm: RL algorithm ('PPO', 'A2C', 'DQN')
        policy_kwargs: Policy network arguments
        **kwargs: Additional arguments for algorithm
        
    Returns:
        Environment and model
    """
    # Create environment
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    
    # Algorithm mapping
    algorithm_map = {
        'PPO': PPO,
        'A2C': A2C,
        'DQN': DQN
    }
    
    if algorithm not in algorithm_map:
        raise ValueError(f"Algorithm {algorithm} not supported. Choose from {list(algorithm_map.keys())}")
    
    # Default policy kwargs
    if policy_kwargs is None:
        policy_kwargs = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
        }
    
    # Create model
    model_class = algorithm_map[algorithm]
    model = model_class(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **kwargs
    )
    
    return env, model

def run_experiment(env_name: str, algorithm: str = 'PPO',
                  total_timesteps: int = 10000,
                  save_path: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
    """
    Run complete RL experiment with training and testing.
    
    Args:
        env_name: Name of Gymnasium environment
        algorithm: RL algorithm
        total_timesteps: Number of training timesteps
        save_path: Path to save model
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with experiment results
    """
    # Setup experiment
    env, model = setup_experiment(env_name, algorithm, **kwargs)
    
    # Setup callback for metrics
    metrics_callback = TrainingMetricsCallback()
    
    # Train model
    print(f"Training {algorithm} on {env_name} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback)
    
    # Save model if path provided
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")
    
    # Test model
    test_env = gym.make(env_name)
    test_results = test_policy(test_env, model, num_episodes=10)
    
    # Return results
    return {
        'model': model,
        'env': env,
        'training_metrics': {
            'episode_rewards': metrics_callback.episode_rewards,
            'episode_lengths': metrics_callback.episode_lengths
        },
        'test_results': test_results
    }

def compare_algorithms(env_name: str, algorithms: List[str] = ['PPO', 'A2C'],
                      total_timesteps: int = 10000,
                      **kwargs) -> Dict[str, Any]:
    """
    Compare multiple RL algorithms on the same environment.
    
    Args:
        env_name: Name of Gymnasium environment
        algorithms: List of algorithms to compare
        total_timesteps: Number of training timesteps
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for algorithm in algorithms:
        print(f"Running {algorithm}...")
        result = run_experiment(
            env_name=env_name,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            **kwargs
        )
        results[algorithm] = result
    
    return results

def plot_algorithm_comparison(comparison_results: Dict[str, Any],
                             metric: str = 'episode_rewards',
                             figsize: Tuple[int, int] = (12, 6)):
    """
    Plot comparison of different algorithms.
    
    Args:
        comparison_results: Results from compare_algorithms
        metric: Metric to compare ('episode_rewards', 'test_rewards', etc.)
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for algorithm, results in comparison_results.items():
        if metric in results['training_metrics']:
            values = results['training_metrics'][metric]
            ax.plot(values, label=algorithm, alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Algorithm Comparison: {metric.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig