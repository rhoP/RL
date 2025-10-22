import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt

class DiscreteStatePPOAnalyzer:
    def __init__(self, env_name, env_kwargs=None):
        self.env_name = env_name
        self.env = make_vec_env(env_name, n_envs=1, env_kwargs=env_kwargs or {})
        
        # Get environment properties
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        print(f"Environment: {env_name}")
        print(f"State space: {self.n_states} states")
        print(f"Action space: {self.n_actions} actions")
        
        # Use a simpler network architecture that works better with discrete states
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=dict(
                net_arch=[32, 32],  # Smaller network
                activation_fn=nn.ReLU,
            ),
            learning_rate=2.5e-4,
            n_steps=256,  # Smaller rollout
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=42,
            device='cpu'  
        )
    
    def train_policy(self, total_timesteps=50000):
        """Train PPO policy to optimal performance"""
        print(f"Training PPO for {total_timesteps} timesteps...")
        
        # Add callback to monitor training progress
        from stable_baselines3.common.callbacks import BaseCallback
        
        class TrainingCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_rewards = []
                self.current_episode_reward = 0
                
            def _on_step(self) -> bool:
                # Track episode rewards during training
                if 'episode' in self.locals['infos'][0]:
                    episode_info = self.locals['infos'][0]['episode']
                    self.episode_rewards.append(episode_info['r'])
                    if len(self.episode_rewards) % 10 == 0:
                        avg_reward = np.mean(self.episode_rewards[-10:])
                        print(f"Step {self.num_timesteps}: Recent avg reward = {avg_reward:.3f}")
                return True
        
        callback = TrainingCallback()
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
        
        # Manual evaluation since evaluate_policy might have issues
        print("Evaluating trained policy...")
        test_rewards = self._manual_evaluation(n_episodes=20)
        mean_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        
        print(f"Trained policy performance: {mean_reward:.3f} +/- {std_reward:.3f}")
        print(f"Success rate: {np.mean([r > 0 for r in test_rewards]):.3f}")
        
        return mean_reward, std_reward
    
    def _manual_evaluation(self, n_episodes=20):
        """Manual evaluation of the policy"""
        rewards = []
        for i in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
        return rewards
    
    def collect_episodes(self, num_episodes=50, max_steps=100, deterministic=True):
        """Collect episodes and create state-episode matrix"""
        state_episode_matrix = np.zeros((self.n_states, num_episodes))
        reward_vector = np.zeros(num_episodes)
        episode_data = []
        
        print(f"Collecting {num_episodes} episodes...")
        
        successful_episodes = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_states = set()
            state_sequence = []
            action_sequence = []
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                next_obs, reward, terminated, truncated = self.env.step(action)
                
                # Handle discrete observation properly
                if isinstance(obs, np.ndarray) and obs.size > 1:
                    current_state = obs[0]
                else:
                    current_state = obs
                
                episode_states.add(current_state)
                state_sequence.append(current_state)
                
                if isinstance(action, np.ndarray):
                    action_sequence.append(action[0])
                else:
                    action_sequence.append(action)
                
                total_reward += reward
                steps += 1
                obs = next_obs

                if terminated or truncated:
                    break
            
            # Mark states in state-episode matrix
            for state in episode_states:
                if state < self.n_states:  # Ensure state is within bounds
                    state_episode_matrix[state, episode] = 1
            
            reward_vector[episode] = total_reward
            if total_reward > 0:
                successful_episodes += 1
            
            episode_data.append({
                'states': list(episode_states),
                'state_sequence': state_sequence,
                'action_sequence': action_sequence,
                'total_reward': total_reward,
                'steps': steps
            })
            
            if (episode + 1) % 10 == 0:
                success = "✓" if total_reward > 0 else "✗"
                print(f"Episode {episode + 1}/{num_episodes} {success} (reward: {total_reward}, steps: {steps})")
        
        print(f"Collection complete: {successful_episodes}/{num_episodes} successful episodes")
        return state_episode_matrix, reward_vector, episode_data
    
    def create_block_triangular_form(self, M, episode_data):
        """Create block triangular form based on environment structure"""
        if self.env_name == "FrozenLake-v1":
            return self._frozenlake_block_triangular(M, episode_data)
        elif self.env_name == "Taxi-v3":
            return self._taxi_block_triangular(M, episode_data)
        elif self.env_name == "CliffWalking-v0":
            return self._cliffwalking_block_triangular(M, episode_data)
        else:
            # Default: order by state visitation frequency
            return self._frequency_block_triangular(M, episode_data)
    
    def _frozenlake_block_triangular(self, M, episode_data):
        """Block triangular form for FrozenLake (4x4 grid)"""
        # Group states by row
        blocks = []
        for row in range(4):
            block_states = list(range(row * 4, (row + 1) * 4))
            blocks.append(block_states)
        
        state_ordering = []
        for i in range(len(blocks)):
            for j in range(i + 1):
                state_ordering.extend(blocks[j])
        
        state_ordering = list(dict.fromkeys(state_ordering))
        M_triangular = M[state_ordering, :]
        
        return M_triangular, state_ordering, blocks
    
    def _taxi_block_triangular(self, M, episode_data):
        """Simplified block triangular form for Taxi environment"""
        # Use just the taxi position for block structure (25 states)
        blocks = []
        for row in range(5):
            block_states = list(range(row * 5, (row + 1) * 5))
            blocks.append(block_states)
        
        state_ordering = []
        for i in range(len(blocks)):
            for j in range(i + 1):
                state_ordering.extend(blocks[j])
        
        state_ordering = list(dict.fromkeys(state_ordering))
        M_triangular = M[state_ordering, :]
        
        return M_triangular, state_ordering, blocks
    
    def _cliffwalking_block_triangular(self, M, episode_data):
        """Block triangular form for CliffWalking (4x12 grid)"""
        blocks = []
        for row in range(4):
            block_states = list(range(row * 12, (row + 1) * 12))
            blocks.append(block_states)
        
        state_ordering = []
        for i in range(len(blocks)):
            for j in range(i + 1):
                state_ordering.extend(blocks[j])
        
        state_ordering = list(dict.fromkeys(state_ordering))
        M_triangular = M[state_ordering, :]
        
        return M_triangular, state_ordering, blocks
    
    def _frequency_block_triangular(self, M, episode_data):
        """Default: order by state visitation frequency"""
        state_frequencies = np.sum(M, axis=1)
        # Sort by frequency (most frequent first)
        state_ordering = np.argsort(state_frequencies)[::-1]
        
        # Create blocks based on frequency
        non_zero_freqs = state_frequencies[state_frequencies > 0]
        if len(non_zero_freqs) > 0:
            blocks = []
            # Split into 3 blocks based on frequency
            if len(non_zero_freqs) >= 3:
                freq_ranges = np.array_split(np.argsort(state_frequencies)[::-1], 3)
                blocks = [list(arr) for arr in freq_ranges]
            else:
                blocks = [list(range(len(state_ordering)))]
        else:
            blocks = [list(range(len(state_ordering)))]
        
        M_triangular = M[state_ordering, :]
        return M_triangular, state_ordering, blocks
    
    def analyze_results(self, M, r, episode_data, blocks, state_ordering):
        """Comprehensive analysis of results"""
        print("\n" + "="*60)
        print(f"RESULTS ANALYSIS: {self.env_name}")
        print("="*60)
        
        # Basic statistics
        success_rate = np.mean(r > 0)
        avg_reward = np.mean(r)
        avg_steps = np.mean([ep['steps'] for ep in episode_data])
        
        print(f"Success rate: {success_rate:.3f} ({np.sum(r > 0)}/{len(r)} episodes)")
        print(f"Average reward: {avg_reward:.3f}")
        print(f"Average steps: {avg_steps:.1f}")
        
        # State coverage analysis
        state_coverage = np.sum(M, axis=1)
        visited_states = np.sum(state_coverage > 0)
        
        print(f"\nState Space Coverage:")
        print(f"States visited: {visited_states}/{self.n_states} ({visited_states/self.n_states*100:.1f}%)")
        print(f"Average states per episode: {np.mean(np.sum(M, axis=0)):.1f}")
        
        # Matrix properties
        matrix_density = np.mean(M > 0)
        print(f"Matrix density: {matrix_density:.3f}")
        
        # Block analysis
        print(f"\nBlock Structure:")
        for i, block in enumerate(blocks):
            if block:  # Only non-empty blocks
                block_coverage = np.sum([state_coverage[s] > 0 for s in block if s < len(state_coverage)])
                print(f"Block {i}: {len(block)} states, {block_coverage} visited")
        
        # Action analysis
        all_actions = []
        for ep in episode_data:
            all_actions.extend(ep['action_sequence'])
        
        if all_actions:
            unique_actions, action_counts = np.unique(all_actions, return_counts=True)
            print(f"\nAction Distribution:")
            action_names = self._get_action_names()
            for action, count in zip(unique_actions, action_counts):
                action_name = action_names.get(action, f"Action_{action}")
                percentage = count / len(all_actions) * 100
                print(f"  {action_name}: {count} ({percentage:.1f}%)")
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'states_visited': visited_states,
            'matrix_density': matrix_density
        }
    
    def _get_action_names(self):
        """Get action names for each environment"""
        if self.env_name == "FrozenLake-v1":
            return {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        elif self.env_name == "Taxi-v3":
            return {0: "SOUTH", 1: "NORTH", 2: "EAST", 3: "WEST", 4: "PICKUP", 5: "DROPOFF"}
        elif self.env_name == "CliffWalking-v0":
            return {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        else:
            return {}
    
    def visualize_results(self, M, r, episode_data, blocks, state_ordering):
        """Visualize the state-episode matrix"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: State-Episode Matrix
            plt.subplot(2, 2, 1)
            plt.imshow(M > 0, cmap='Blues', aspect='auto', interpolation='none')
            plt.title(f'State-Episode Matrix: {self.env_name}\n(Block Triangular Form)')
            plt.xlabel('Episodes')
            plt.ylabel('States (Ordered)')
            plt.colorbar(label='State Present')
            
            # Plot 2: Reward Distribution
            plt.subplot(2, 2, 2)
            plt.hist(r, bins=20, alpha=0.7, color='green', edgecolor='black')
            plt.title('Reward Distribution')
            plt.xlabel('Total Reward')
            plt.ylabel('Frequency')
            if len(r) > 0:
                plt.axvline(np.mean(r), color='red', linestyle='--', label=f'Mean: {np.mean(r):.2f}')
                plt.legend()
            
            # Plot 3: State Visitation Frequency
            plt.subplot(2, 2, 3)
            state_frequencies = np.sum(M, axis=1)
            plt.bar(range(len(state_frequencies)), state_frequencies, alpha=0.7)
            plt.title('State Visitation Frequency')
            plt.xlabel('State (Ordered)')
            plt.ylabel('Number of Episodes')
            
            # Plot 4: Episode Length Distribution
            plt.subplot(2, 2, 4)
            episode_lengths = [ep['steps'] for ep in episode_data]
            plt.hist(episode_lengths, bins=20, alpha=0.7, color='purple', edgecolor='black')
            plt.title('Episode Length Distribution')
            plt.xlabel('Steps')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Visualization error: {e}")

def run_frozenlake_case_study():
    """Run FrozenLake case study with proper configuration"""
    print("\n" + "="*80)
    print("CASE STUDY: FrozenLake-v1")
    print("="*80)
    
    try:
        # Initialize analyzer
        analyzer = DiscreteStatePPOAnalyzer(
            "FrozenLake-v1", 
            env_kwargs={'is_slippery': False, 'map_name': '4x4'}  # Start with non-slippery for easier learning
        )
        
        # Train policy with more timesteps
        print("Phase 1: Training with non-slippery environment...")
        mean_reward, std_reward = analyzer.train_policy(total_timesteps=100000)
        
        # If performance is poor, try different hyperparameters
        if mean_reward < 0.5:
            print("Performance is low, trying alternative hyperparameters...")
            # Create new model with different hyperparameters
            analyzer.model = PPO(
                "MlpPolicy",
                analyzer.env,
                policy_kwargs=dict(
                    net_arch=[64, 64],
                    activation_fn=nn.Tanh,
                ),
                learning_rate=1e-3,
                n_steps=512,
                batch_size=128,
                gamma=0.95,  # Slightly lower discount
                ent_coef=0.1,  # Higher entropy for exploration
                vf_coef=0.5,
                max_grad_norm=0.8,
                verbose=1,
                seed=42,
                device='cpu'
            )
            mean_reward, std_reward = analyzer.train_policy(total_timesteps=50000)
        
        # Collect episodes
        M, r, episode_data = analyzer.collect_episodes(num_episodes=30, deterministic=True)
        
        # Create block triangular form
        M_triangular, state_ordering, blocks = analyzer.create_block_triangular_form(M, episode_data)
        
        # Analyze results
        results = analyzer.analyze_results(M_triangular, r, episode_data, blocks, state_ordering)
        
        # Visualize results
        analyzer.visualize_results(M_triangular, r, episode_data, blocks, state_ordering)
        
        return M_triangular, r, episode_data, results
        
    except Exception as e:
        print(f"Error in FrozenLake case study: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def run_simple_test():
    """Run a simple test to verify the environment works"""
    print("Running simple environment test...")
    
    # Test the environment directly
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    # Take a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: action={action}, obs={obs}, reward={reward}, done={done}")
        if done:
            break
    
    env.close()

if __name__ == "__main__":
    # First run a simple test
    run_simple_test()
    
    # Then run the main case study
    results = run_frozenlake_case_study()
    
    if results[0] is not None:
        M, r, episode_data, results_dict = results
        print("\n" + "="*80)
        print("CASE STUDY COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Success rate: {results_dict['success_rate']:.3f}")
        print(f"States visited: {results_dict['states_visited']}/{results_dict.get('total_states', 16)}")
        print(f"Matrix shape: {M.shape}")
    else:
        print("\nCase study failed. Check the error messages above.")