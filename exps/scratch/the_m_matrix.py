import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn

class RolloutCollectorCallback(BaseCallback):
    """Callback to collect rollout data during training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_data = []
        self.current_episode = []
        
    def _on_step(self) -> bool:
        # Get the current rollout buffer data
        if len(self.model.rollout_buffer.actions) > 0:
            # Store episode data when episode ends
            if self.locals.get('dones', [False])[0]:
                if self.current_episode:
                    self.episode_data.append(self.current_episode)
                    self.current_episode = []
        return True

class OptimalPPORolloutCollector:
    def __init__(self, env_name="LunarLander-v3", map_size=4):
        self.map_size = map_size
        self.n_states = map_size * map_size
        self.n_actions = 4
        
        # Create environment
        self.env = make_vec_env(env_name, n_envs=4)
                               # env_kwargs={'is_slippery': True, 'map_name': f"{map_size}x{map_size}"})
        
        # Optimal PPO configuration
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh),
            learning_rate=3e-4,
            n_steps=512,  # This determines rollout length
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=42
        )
    
    def train_optimal(self, total_timesteps=100000):
        """Train an optimal PPO policy"""
        print("Training optimal PPO policy...")
        
        # Create callback to collect data during training
        collector_callback = RolloutCollectorCallback()
        
        self.model.learn(
            total_timesteps=total_timesteps, 
            progress_bar=True,
            callback=collector_callback
        )
        
        # Evaluate the trained policy
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=100)
        print(f"Optimal policy evaluation:")
        print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
        
        return mean_reward, std_reward
    
    def collect_rollouts_direct(self, num_rollouts=10):
        """Directly collect rollouts using the PPO model's internal method"""
        print(f"Collecting {num_rollouts} rollouts using direct buffer access...")
        
        all_episodes = []
        
        for rollout_idx in range(num_rollouts):
            # Reset environment and buffer
            self.model._last_obs = None
            self.model._last_episode_starts = True
            self.model.rollout_buffer.reset()
            
            # Manually run the collection process
            self.model._update_current_progress_remaining(1.0, 1.0)
            
            # This is a simplified version of what happens in collect_rollouts
            episodes_from_rollout = self._collect_single_rollout()
            all_episodes.extend(episodes_from_rollout)
            
            print(f"Rollout {rollout_idx + 1}/{num_rollouts}: collected {len(episodes_from_rollout)} episodes")
        
        # Process all episodes to create matrices
        return self._create_matrices_from_episodes(all_episodes)
    
    def _collect_single_rollout(self):
        """Collect a single rollout using the model's policy"""
        episodes = []
        current_episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        # Reset environment
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle (obs, info) tuple
        
        done = False
        steps_collected = 0
        
        while steps_collected < self.model.n_steps and not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).to(self.model.device)
                actions, values, log_probs = self.model.policy(obs_tensor)
            
            # Take action in environment
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            
            next_obs, rewards, dones, infos = self.env.step(actions)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            
            # Store step data
            current_state = obs[0] if isinstance(obs, np.ndarray) and len(obs.shape) > 0 else obs
            current_episode['states'].append(current_state)
            current_episode['actions'].append(actions[0] if isinstance(actions, np.ndarray) else actions)
            current_episode['rewards'].append(rewards[0] if isinstance(rewards, np.ndarray) else rewards)
            current_episode['dones'].append(dones[0] if isinstance(dones, np.ndarray) else dones)
            
            # Check if episode ended
            if dones[0] if isinstance(dones, np.ndarray) else dones:
                episodes.append(current_episode)
                current_episode = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'dones': []
                }
                # Reset environment
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
            else:
                obs = next_obs
            
            steps_collected += 1
        
        # Add the final episode if it's not empty
        if current_episode['states']:
            episodes.append(current_episode)
        
        return episodes
    
    def _create_matrices_from_episodes(self, all_episodes, max_episodes=50):
        """Create state-episode matrix and reward vector from collected episodes"""
        if len(all_episodes) > max_episodes:
            all_episodes = all_episodes[:max_episodes]
        
        num_episodes = len(all_episodes)
        state_episode_matrix = np.zeros((self.n_states, num_episodes))
        reward_vector = np.zeros(num_episodes)
        episode_data = []
        
        for ep_idx, episode in enumerate(all_episodes):
            states = episode['states']
            rewards = episode['rewards']
            total_reward = sum(rewards)
            
            # Mark unique states in the matrix
            unique_states = set(states)
            for state in unique_states:
                state_episode_matrix[state, ep_idx] = 1
            
            reward_vector[ep_idx] = total_reward
            episode_data.append({
                'states': states,
                'unique_states': list(unique_states),
                'rewards': rewards,
                'total_reward': total_reward,
                'steps': len(states),
                'actions': episode['actions']
            })
        
        return state_episode_matrix, reward_vector, episode_data

    def collect_rollouts_block_triangular(self, num_rollouts=10, max_episodes=50):
        """Collect rollouts and organize in block triangular form"""
        M, r, episode_data = self.collect_rollouts_direct(num_rollouts)
        
        # If we have more episodes than needed, take the first max_episodes
        if M.shape[1] > max_episodes:
            M = M[:, :max_episodes]
            r = r[:max_episodes]
            episode_data = episode_data[:max_episodes]
        
        # Create blocks based on row positions
        blocks = []
        for row in range(self.map_size):
            block_states = list(range(row * self.map_size, (row + 1) * self.map_size))
            blocks.append(block_states)
        
        # Reorder to block triangular form
        M_triangular, state_ordering = self._arrange_block_triangular(M, blocks)
        
        return M_triangular, r, episode_data, state_ordering, blocks
    
    def _arrange_block_triangular(self, M, blocks):
        """Arrange the state-episode matrix in block triangular form"""
        state_ordering = []
        for i in range(len(blocks)):
            for j in range(i + 1):
                state_ordering.extend(blocks[j])
        
        # Remove duplicates while preserving order
        state_ordering = list(dict.fromkeys(state_ordering))
        
        # Reorder the matrix
        M_reordered = M[state_ordering, :]
        
        return M_reordered, state_ordering

def analyze_rollout_data(M, r, episode_data, blocks):
    """Analyze the rollout data collected from PPO"""
    print("\n" + "="*60)
    print("ROLLOUT DATA ANALYSIS")
    print("="*60)
    
    success_rate = np.mean(r > 0)
    avg_reward = np.mean(r)
    avg_steps = np.mean([ep['steps'] for ep in episode_data])
    
    print(f"Total episodes collected: {len(r)}")
    print(f"Success rate: {success_rate:.3f} ({np.sum(r > 0)}/{len(r)} successful episodes)")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average steps per episode: {avg_steps:.1f}")
    
    # State coverage analysis
    state_coverage = np.sum(M, axis=1)
    print(f"\nState coverage analysis:")
    print(f"States visited in at least one episode: {np.sum(state_coverage > 0)}/{M.shape[0]}")
    print(f"Average state coverage per episode: {np.mean(np.sum(M, axis=0)):.1f} states")
    
    # Episode length analysis
    episode_lengths = [ep['steps'] for ep in episode_data]
    print(f"Min steps: {np.min(episode_lengths)}, Max steps: {np.max(episode_lengths)}")
    
    # Action distribution
    all_actions = []
    for ep in episode_data:
        all_actions.extend(ep['actions'])
    
    if all_actions:
        unique_actions, action_counts = np.unique(all_actions, return_counts=True)
        print(f"\nAction distribution:")
        action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        for action, count in zip(unique_actions, action_counts):
            print(f"  {action_names[action]}: {count} times ({count/len(all_actions)*100:.1f}%)")
    
    return success_rate, avg_reward

class FeatureBasedPPOCollector:
    def __init__(self, env_name="Pendulum-v1", feature_dim=10):
        self.env = make_vec_env(env_name, n_envs=1)
        self.feature_dim = feature_dim
        
        # Autoencoder for state compression
        self.autoencoder = self._build_autoencoder()
        self.feature_scaler = StandardScaler()
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh),
            learning_rate=3e-4,
            verbose=1
        )
    
    def _build_autoencoder(self):
        """Build a simple autoencoder for state compression"""
        class StateAutoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        obs_dim = self.env.observation_space.shape[0]
        return StateAutoencoder(obs_dim, self.feature_dim)
    
    def extract_state_features(self, states):
        """Extract low-dimensional features from states"""
        states_tensor = torch.FloatTensor(states)
        with torch.no_grad():
            features, _ = self.autoencoder(states_tensor)
        return features.numpy()
    
    def create_feature_episode_matrix(self, episode_data):
        """Create episode matrix using state features"""
        num_episodes = len(episode_data)
        
        # Collect all state features
        all_features = []
        episode_features = []
        reward_vector = np.zeros(num_episodes)
        
        for ep_idx, episode in enumerate(episode_data):
            states = episode['states_continuous']
            features = self.extract_state_features(states)
            
            episode_features.append(features)
            all_features.extend(features)
            reward_vector[ep_idx] = episode['total_reward']
        
        all_features = np.array(all_features)
        
        # Create binary matrix based on feature similarity
        M_feature = self._create_feature_similarity_matrix(episode_features)
        
        return {
            'M_feature': M_feature,
            'state_features': all_features,
            'reward_vector': reward_vector,
            'episode_features': episode_features
        }
    
    def _create_feature_similarity_matrix(self, episode_features, similarity_threshold=0.7):
        """Create binary matrix based on feature space similarity"""
        # This is a simplified approach - in practice, you might use clustering
        # on the feature space
        
        # Flatten all features and find unique feature patterns
        all_unique_features = []
        for features in episode_features:
            for feature_vec in features:
                # Simple discretization of feature space
                discretized = tuple(np.round(feature_vec, 1))
                all_unique_features.append(discretized)
        
        unique_patterns = list(set(all_unique_features))
        num_patterns = len(unique_patterns)
        num_episodes = len(episode_features)
        
        M = np.zeros((num_patterns, num_episodes))
        
        pattern_to_idx = {pattern: i for i, pattern in enumerate(unique_patterns)}
        
        for ep_idx, features in enumerate(episode_features):
            for feature_vec in features:
                discretized = tuple(np.round(feature_vec, 1))
                pattern_idx = pattern_to_idx[discretized]
                M[pattern_idx, ep_idx] = 1
        
        return M

def main():
    # Create and train optimal PPO policy
    print("=== TRAINING OPTIMAL PPO POLICY ===")
    collector = OptimalPPORolloutCollector(map_size=4)
    
    # Train to optimal performance
    mean_reward, std_reward = collector.train_optimal(total_timesteps=50000)
    
    # Collect episodes using direct rollout collection
    print("\n=== COLLECTING EPISODES USING DIRECT ROLLOUT ACCESS ===")
    M_triangular, r, episode_data, state_ordering, blocks = \
        collector.collect_rollouts_block_triangular(num_rollouts=5, max_episodes=50)
    
    # Analyze the rollout data
    success_rate, avg_reward = analyze_rollout_data(M_triangular, r, episode_data, blocks)
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS - PPO ROLLOUT COLLECTION")
    print("="*60)
    print(f"Block triangular matrix M shape: {M_triangular.shape}")
    print(f"Reward vector r shape: {r.shape}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Average reward: {avg_reward:.3f}")
    
    # Show matrix properties
    print(f"\nMatrix M properties:")
    print(f"Density (non-zero entries): {np.mean(M_triangular > 0):.3f}")
    print(f"States per episode (mean): {np.mean(np.sum(M_triangular, axis=0)):.1f}")
    
    # Display sample data
    print(f"\nFirst 10 rewards: {r[:10]}")
    print(f"\nFirst 8x8 block of M (states x episodes):")
    print(M_triangular[:8, :8].astype(int))
    
    # Show detailed episode examples
    print(f"\nDetailed episode examples:")
    for i in range(min(2, len(episode_data))):
        ep = episode_data[i]
        print(f"\nEpisode {i}:")
        print(f"  Total reward: {ep['total_reward']}")
        print(f"  Steps: {ep['steps']}")
        print(f"  Unique states visited: {len(ep['unique_states'])}")
        print(f"  State sequence: {ep['states'][:8]}...")  # First 8 states
        print(f"  Action sequence: {ep['actions'][:8]}...")  # First 8 actions
    
    return M_triangular, r, state_ordering, blocks, episode_data

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ContinuousStatePPOCollector:
    def __init__(self, env_name="Pendulum-v1", n_state_bins=100, n_clusters=50):
        self.env = make_vec_env(env_name, n_envs=1)
        self.n_state_bins = n_state_bins
        self.n_clusters = n_clusters
        
        # State space properties
        self.observation_shape = self.env.observation_space.shape
        self.state_dim = self.observation_shape[0]
        
        # Discretization structures
        self.state_bins = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_centers_ = None
        
        # PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh),
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            verbose=1,
            seed=42
        )
    
    def discretize_state(self, state):
        """Convert continuous state to discrete representation"""
        if self.state_bins is None:
            raise ValueError("Must call fit_discretization first")
        
        # Method 1: Binning
        discretized_bins = []
        for i in range(self.state_dim):
            digitized = np.digitize(state[i], self.state_bins[i]) - 1
            digitized = np.clip(digitized, 0, self.n_state_bins - 1)
            discretized_bins.append(digitized)
        
        # Method 2: Clustering
        state_reshaped = state.reshape(1, -1) if len(state.shape) == 1 else state
        cluster_label = self.kmeans.predict(state_reshaped)[0]
        
        return {
            'binned': tuple(discretized_bins),
            'cluster': cluster_label,
            'continuous': state
        }
    
    def fit_discretization(self, n_samples=10000):
        """Fit discretization parameters by sampling from environment"""
        print("Fitting state discretization...")
        
        # Collect state samples
        states = []
        obs = self.env.reset()
        
        for _ in range(n_samples):
            action = self.env.action_space.sample()
            next_obs, _, done, _ = self.env.step([action])
            states.append(obs.flatten())
            obs = next_obs
            if done:
                obs = self.env.reset()
        
        states = np.array(states)
        
        # Fit scaler
        self.scaler.fit(states)
        states_scaled = self.scaler.transform(states)
        
        # Create bins for each dimension
        self.state_bins = []
        for i in range(self.state_dim):
            min_val = np.min(states_scaled[:, i])
            max_val = np.max(states_scaled[:, i])
            bins = np.linspace(min_val, max_val, self.n_state_bins + 1)
            self.state_bins.append(bins)
        
        # Fit K-means clustering
        self.kmeans.fit(states_scaled)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        print(f"Fitted discretization with {self.n_state_bins} bins and {self.n_clusters} clusters")
        return states
    
    def collect_episodes_continuous(self, num_episodes=50, max_steps=200):
        """Collect episodes in continuous state space"""
        print(f"Collecting {num_episodes} episodes in continuous state space...")
        
        # We'll use multiple representations for the state-episode matrix
        episode_data = []
        all_states = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_states = []
            episode_states_discrete = []
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                next_obs, reward, done, truncated, _ = self.env.step(action)
                
                current_state = obs.flatten()
                discrete_state = self.discretize_state(current_state)
                
                episode_states.append(current_state)
                episode_states_discrete.append(discrete_state)
                total_reward += reward
                steps += 1
                obs = next_obs
                
                if done or truncated:
                    break
            
            # Convert to numpy arrays
            episode_states = np.array(episode_states)
            
            episode_data.append({
                'states_continuous': episode_states,
                'states_discrete': episode_states_discrete,
                'total_reward': total_reward,
                'steps': steps,
                'state_clusters': [s['cluster'] for s in episode_states_discrete],
                'state_bins': [s['binned'] for s in episode_states_discrete]
            })
            all_states.extend(episode_states)
            
            if (episode + 1) % 10 == 0:
                print(f"Collected episode {episode + 1}/{num_episodes}")
        
        all_states = np.array(all_states)
        return episode_data, all_states
    
    def create_state_episode_matrices(self, episode_data):
        """Create multiple state-episode matrix representations"""
        num_episodes = len(episode_data)
        
        # Matrix 1: Cluster-based representation
        M_cluster = np.zeros((self.n_clusters, num_episodes))
        
        # Matrix 2: Binned representation (flattened)
        total_bins = self.n_state_bins ** self.state_dim
        M_binned = np.zeros((total_bins, num_episodes))
        
        # Matrix 3: Continuous feature matrix (states x features)
        all_states = []
        reward_vector = np.zeros(num_episodes)
        
        for ep_idx, episode in enumerate(episode_data):
            reward_vector[ep_idx] = episode['total_reward']
            
            # Cluster-based matrix
            unique_clusters = set(episode['state_clusters'])
            for cluster in unique_clusters:
                M_cluster[cluster, ep_idx] = 1
            
            # Binned matrix
            unique_binned_states = set(episode['state_bins'])
            for binned_state in unique_binned_states:
                # Convert multi-dimensional bin to single index
                flat_idx = self._binned_state_to_index(binned_state)
                M_binned[flat_idx, ep_idx] = 1
            
            # Collect all continuous states
            all_states.extend(episode['states_continuous'])
        
        # Continuous state matrix (not binary - actual state values)
        state_features = np.array(all_states)  # Shape: (total_steps_across_episodes, state_dim)
        
        return {
            'M_cluster': M_cluster,
            'M_binned': M_binned,
            'state_features': state_features,
            'reward_vector': reward_vector,
            'episode_data': episode_data
        }
    
    def _binned_state_to_index(self, binned_state):
        """Convert multi-dimensional bin tuple to flat index"""
        index = 0
        for i, bin_val in enumerate(binned_state):
            index += bin_val * (self.n_state_bins ** i)
        return index
    
    def create_block_triangular_continuous(self, matrices, method='cluster'):
        """Create block triangular organization for continuous states"""
        if method == 'cluster':
            M = matrices['M_cluster']
            n_states = self.n_clusters
        else:  # binned
            M = matrices['M_binned']
            n_states = self.n_state_bins ** self.state_dim
        
        # For continuous spaces, we can create blocks based on state properties
        # Example: Group states by their value ranges or cluster properties
        
        # Simple approach: sort clusters by their distance from origin
        if method == 'cluster' and hasattr(self, 'cluster_centers_'):
            cluster_distances = np.linalg.norm(self.cluster_centers_, axis=1)
            state_ordering = np.argsort(cluster_distances)
            M_ordered = M[state_ordering, :]
        else:
            # Default ordering
            state_ordering = np.arange(n_states)
            M_ordered = M
        
        return M_ordered, state_ordering

def continuous_space_workflow():
    """Complete workflow for continuous state spaces"""
    
    # Initialize collector
    collector = ContinuousStatePPOCollector(env_name="Pendulum-v1", n_state_bins=20, n_clusters=30)
    
    # Fit discretization
    collector.fit_discretization(n_samples=5000)
    
    # Train policy
    print("Training PPO policy...")
    collector.model.learn(total_timesteps=50000, progress_bar=True)
    
    # Collect episodes
    episode_data, all_states = collector.collect_episodes_continuous(num_episodes=50)
    
    # Create matrices
    matrices = collector.create_state_episode_matrices(episode_data)
    
    # Create block triangular forms
    M_cluster_triangular, cluster_ordering = collector.create_block_triangular_continuous(
        matrices, method='cluster'
    )
    
    M_binned_triangular, binned_ordering = collector.create_block_triangular_continuous(
        matrices, method='binned'
    )
    
    # Analysis
    print("\n" + "="*60)
    print("CONTINUOUS SPACE ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Cluster-based matrix shape: {M_cluster_triangular.shape}")
    print(f"Binned matrix shape: {M_binned_triangular.shape}")
    print(f"Average reward: {np.mean(matrices['reward_vector']):.3f}")
    
    print(f"\nCluster matrix density: {np.mean(M_cluster_triangular > 0):.3f}")
    print(f"Binned matrix density: {np.mean(M_binned_triangular > 0):.3f}")
    
    # State space coverage
    unique_clusters_visited = np.sum(np.any(M_cluster_triangular > 0, axis=1))
    print(f"Unique state clusters visited: {unique_clusters_visited}/{collector.n_clusters}")
    
    return matrices, M_cluster_triangular, M_binned_triangular, episode_data

if __name__ == "__main__":
    results = continuous_space_workflow()
