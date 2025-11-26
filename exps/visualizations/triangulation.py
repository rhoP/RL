import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from collections import defaultdict
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KernelDensity
import pandas as pd
from scipy import stats


class RLTrajectoryStorage:
    def __init__(self, max_trajectories=1000):
        self.trajectories = []
        self.max_trajectories = max_trajectories
        self.state_dim = None
        self.action_dim = None
        
    def add_trajectory(self, states, actions, rewards, dones):
        """Store a complete trajectory"""
        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones)
        }
        
        if self.state_dim is None and len(states) > 0:
            self.state_dim = states[0].shape[0] if hasattr(states[0], 'shape') else len(states[0])
        if self.action_dim is None and len(actions) > 0:
            self.action_dim = actions[0].shape[0] if hasattr(actions[0], 'shape') and not all(actions[0].shape) else len(actions[0])
        
        self.trajectories.append(trajectory)
        
        # Keep only the most recent trajectories if limit exceeded
        if len(self.trajectories) > self.max_trajectories:
            self.trajectories = self.trajectories[-self.max_trajectories:]
    
    def get_state_action_pairs(self, max_samples=10000):
        """Extract state-action pairs from all trajectories"""
        state_action_pairs = []
        
        for traj in self.trajectories:
            states = traj['states']
            actions = traj['actions']
            
            # Ensure states and actions have the same length
            min_len = min(len(states), len(actions))
            for i in range(min_len):
                state = states[i].flatten() if hasattr(states[i], 'flatten') else states[i]
                action = actions[i].flatten() if hasattr(actions[i], 'flatten') else actions[i]
                
                state_action = np.concatenate([state, action])
                state_action_pairs.append(state_action)
                
                if len(state_action_pairs) >= max_samples:
                    return np.array(state_action_pairs)
        
        return np.array(state_action_pairs)
    
    def save(self, filename):
        """Save trajectories to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'trajectories': self.trajectories,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }, f)
    
    def load(self, filename):
        """Load trajectories from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.trajectories = data['trajectories']
            self.state_dim = data['state_dim']
            self.action_dim = data['action_dim']

class TrajectoryCollectorCallback(BaseCallback):
    """Custom callback to collect trajectories during training"""
    def __init__(self, storage, verbose=0):
        super(TrajectoryCollectorCallback, self).__init__(verbose)
        self.storage = storage
        self.current_trajectory = defaultdict(list)
        
    def _on_step(self):
        # Get the current state, action, reward, done info
        state = self.locals["obs_tensor"][0]  # Get first env state
        action = self.locals["actions"][0]    # Get first env action
        reward = self.locals["rewards"][0]    # Get first env reward
        done = self.locals["dones"][0]        # Get first env done
        
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['dones'].append(done)
        
        # If episode is done, store the trajectory and reset
        if done:
            self.storage.add_trajectory(
                self.current_trajectory['states'],
                self.current_trajectory['actions'],
                self.current_trajectory['rewards'],
                self.current_trajectory['dones']
            )
            self.current_trajectory = defaultdict(list)
            
        return True

class ManifoldTriangulation:
    def __init__(self, method='pca', n_components=2):
        self.method = method
        self.n_components = n_components
        self.dimension_reducer = None
        self.triangulation = None
        self.projected_points = None
        
    def fit_dimension_reduction(self, state_action_pairs):
        """Fit dimension reduction for high-dimensional state-action space"""
        if self.method == 'pca':
            self.dimension_reducer = PCA(n_components=self.n_components)
        elif self.method == 'tsne':
            self.dimension_reducer = TSNE(n_components=self.n_components, 
                                         random_state=42,
                                         perplexity=min(30, len(state_action_pairs)-1))
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        self.projected_points = self.dimension_reducer.fit_transform(state_action_pairs)
        return self.projected_points
    
    def compute_triangulation(self, points=None):
        """Compute Delaunay triangulation of the points"""
        if points is None:
            if self.projected_points is None:
                raise ValueError("No points provided. Call fit_dimension_reduction first.")
            points = self.projected_points
        
        self.triangulation = Delaunay(points)
        return self.triangulation
    
    def get_simplices(self):
        """Get the simplices (triangles in 2D, tetrahedra in 3D, etc.)"""
        if self.triangulation is None:
            raise ValueError("No triangulation computed. Call compute_triangulation first.")
        return self.triangulation.simplices
    
    def find_simplex(self, point):
        """Find which simplex contains a given point"""
        if self.triangulation is None:
            raise ValueError("No triangulation computed. Call compute_triangulation first.")
        return self.triangulation.find_simplex(point)
    
    def visualize_triangulation_2d(self, state_action_pairs=None, 
                                 show_points=True, alpha=0.1,
                                 color_by_reward=False, storage=None):
        """Visualize the 2D triangulation with optional reward coloring"""
        if self.n_components != 2:
            print("Warning: Visualization only works for 2D projections")
            return
        
        if self.projected_points is None:
            raise ValueError("No projected points available")
        
        plt.figure(figsize=(12, 10))
        
        # Plot triangulation
        plt.triplot(self.projected_points[:, 0], 
                   self.projected_points[:, 1], 
                   self.triangulation.simplices.copy(), 
                   'b-', alpha=alpha)
        
        if show_points:
            if color_by_reward and storage is not None:
                # Color points by cumulative reward of their trajectory
                rewards = []
                for traj in storage.trajectories:
                    traj_rewards = np.sum(traj['rewards'])
                    for _ in range(len(traj['states'])):
                        rewards.append(traj_rewards)
                rewards = np.array(rewards[:len(self.projected_points)])
                
                scatter = plt.scatter(self.projected_points[:, 0], 
                                    self.projected_points[:, 1], 
                                    c=rewards, cmap='viridis', 
                                    alpha=0.6, s=10)
                plt.colorbar(scatter, label='Trajectory Cumulative Reward')
            elif state_action_pairs is not None:
                # Color points by their original state-action norm
                norms = np.linalg.norm(state_action_pairs, axis=1)
                scatter = plt.scatter(self.projected_points[:, 0], 
                                    self.projected_points[:, 1], 
                                    c=norms, cmap='viridis', 
                                    alpha=0.6, s=10)
                plt.colorbar(scatter, label='State-Action Norm')
            else:
                plt.scatter(self.projected_points[:, 0], 
                          self.projected_points[:, 1], 
                          alpha=0.6, s=10)
        
        plt.title(f'Delaunay Triangulation of State-Action Manifold ({self.method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_manifold_properties(self):
        """Analyze basic properties of the triangulated manifold"""
        if self.triangulation is None:
            raise ValueError("No triangulation computed")
        
        simplices = self.triangulation.simplices
        points = self.projected_points
        
        properties = {
            'num_points': len(points),
            'num_simplices': len(simplices),
            'avg_simplex_volume': np.mean([self._simplex_volume(points[s]) for s in simplices]),
            'manifold_dimensionality': self.n_components
        }
        
        return properties
    
    def _simplex_volume(self, simplex_points):
        """Calculate volume of a simplex"""
        if len(simplex_points) == 3:  # Triangle
            return 0.5 * np.abs(np.cross(simplex_points[1]-simplex_points[0], 
                                       simplex_points[2]-simplex_points[0]))
        else:
            # For higher dimensions, use determinant-based volume calculation
            vectors = simplex_points[1:] - simplex_points[0]
            return np.abs(np.linalg.det(vectors)) / np.math.factorial(len(vectors))

def train_ppo_and_collect_trajectories(env_name="Pendulum-v1", total_timesteps=50000):
    """Train a PPO agent and collect trajectories"""
    print(f"Training PPO on {env_name}...")
    
    # Create environment
    env = gym.make(env_name, continuous=True)
    env = DummyVecEnv([lambda: env])
    
    # Initialize trajectory storage
    storage = RLTrajectoryStorage()
    
    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=4,
                gamma=0.999,
                gae_lambda=0.98,
                clip_range=0.2,
                ent_coef=0.01)
    
    # Train with trajectory collection
    callback = TrajectoryCollectorCallback(storage)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Collect additional trajectories with the trained policy
    print("Collecting additional trajectories with trained policy...")
    collect_additional_trajectories(model, storage, env_name, num_trajectories=50)
    
    return model, storage, env

def collect_additional_trajectories(model, storage, env_name, num_trajectories=50):
    """Collect additional trajectories using the trained policy"""
    env = gym.make(env_name, continuous=True)
    collected = 0
    
    while collected < num_trajectories:
        state, _ = env.reset()
        states, actions, rewards, dones = [], [], [], []
        
        while True:
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            if done:
                break


        # Ignore weak trajectories
        if np.sum(rewards) > 200:
            storage.add_trajectory(states, actions, rewards, dones)
            collected += 1
        
    env.close()

def demonstrate_with_lunar_lander():
    """Demonstrate with LunarLander environment"""
    print("=== LunarLander-v3 Demonstration ===")
    model, storage, env = train_ppo_and_collect_trajectories(
        env_name="LunarLander-v3", 
        total_timesteps=100000
    )
    
    # Evaluate the trained policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Trained policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, storage

def demonstrate_with_pendulum():
    """Demonstrate with Pendulum environment"""
    print("=== Pendulum-v1 Demonstration ===")
    model, storage, env = train_ppo_and_collect_trajectories(
        env_name="Pendulum-v1", 
        total_timesteps=50000
    )
    
    # Evaluate the trained policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Trained policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, storage

def run_exp():
    env_choice = "lunar"      
    if env_choice == "pendulum":
        model, storage = demonstrate_with_pendulum()
    else:
        model, storage = demonstrate_with_lunar_lander()
    
    # Extract state-action pairs
    print("\nExtracting state-action pairs...")
    state_action_pairs = storage.get_state_action_pairs()
    print(f"Collected {len(state_action_pairs)} state-action pairs")
    print(f"State-action space dimension: {state_action_pairs.shape[1]}")
    
    # Compute manifold triangulation
    print("\nComputing manifold triangulation...")
    triangulator = ManifoldTriangulation(method='pca', n_components=2)
    
    # Reduce dimensionality
    projected_points = triangulator.fit_dimension_reduction(state_action_pairs)
    print(f"Reduced to {projected_points.shape[1]} dimensions")
    print(f"Explained variance ratio (PCA): {triangulator.dimension_reducer.explained_variance_ratio_}")
    
    # Compute triangulation
    triangulation = triangulator.compute_triangulation()
    print(f"Computed triangulation with {len(triangulation.simplices)} simplices")
    
    # Analyze manifold properties
    properties = triangulator.analyze_manifold_properties()
    print("\nManifold properties:")
    for key, value in properties.items():
        print(f"  {key}: {value}")
    
    # Visualize with reward coloring
    print("\nVisualizing triangulation...")
    triangulator.visualize_triangulation_2d(
        state_action_pairs=state_action_pairs,
        color_by_reward=True,
        storage=storage
    )
    
    # Demonstrate point location
    test_point = projected_points[0]  # Use first point as test
    simplex_idx = triangulator.find_simplex(test_point)
    print(f"\nTest point is in simplex: {simplex_idx}")
    
    # Save trajectories and model
    storage.save('rl_trajectories.pkl')
    model.save('ppo_model')
    print("\nTrajectories saved to 'rl_trajectories.pkl'")
    print("Model saved to 'ppo_model.zip'")

    # Example of analyzing different regions of the manifold
    print("\nAnalyzing different regions of the manifold...")
    
    # Find points with highest and lowest cumulative rewards
    all_rewards = []
    for traj in storage.trajectories:
        traj_reward = np.sum(traj['rewards'])
        all_rewards.append(traj_reward)
    
    best_traj_idx = np.argmax(all_rewards)
    worst_traj_idx = np.argmin(all_rewards)
    
    print(f"Best trajectory reward: {all_rewards[best_traj_idx]:.2f}")
    print(f"Worst trajectory reward: {all_rewards[worst_traj_idx]:.2f}")



class SubManifoldAnalyzer:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = None
        self.projected_points = None
        self.cluster_labels = None
        self.submanifolds = {}
        
    def identify_submanifolds(self, state_action_pairs, method='dbscan', **kwargs):
        """Identify distinct sub-manifolds in the state-action space"""
        
        # First, perform PCA
        self.pca = PCA(n_components=self.n_components)
        self.projected_points = self.pca.fit_transform(state_action_pairs)
        
        # Now identify clusters/submanifolds in the PCA space
        if method == 'dbscan':
            clustering = DBSCAN(eps=kwargs.get('eps', 0.5), 
                              min_samples=kwargs.get('min_samples', 10))
        elif method == 'kmeans':
            clustering = KMeans(n_clusters=kwargs.get('n_clusters', 5),
                              random_state=42)
        else:
            raise ValueError("Method must be 'dbscan' or 'kmeans'")
        
        self.cluster_labels = clustering.fit_predict(self.projected_points)
        
        # Extract submanifold information
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
                
            mask = self.cluster_labels == label
            submanifold_points = state_action_pairs[mask]
            projected_submanifold = self.projected_points[mask]
            
            self.submanifolds[label] = {
                'state_action_pairs': submanifold_points,
                'projected_points': projected_submanifold,
                'size': len(submanifold_points),
                'center': np.mean(submanifold_points, axis=0),
                'pca_center': np.mean(projected_submanifold, axis=0)
            }
        
        return self.cluster_labels
    
    def analyze_submanifold_characteristics(self, storage):
        """Analyze what each submanifold represents in terms of behavior"""
        characteristics = {}
        
        for label, manifold_data in self.submanifolds.items():
            # Get the original trajectories that contributed to this submanifold
            contributing_trajectories = []
            
            # We need to map back from state-action pairs to trajectories
            point_indices = []
            current_idx = 0
            
            for traj_idx, traj in enumerate(storage.trajectories):
                traj_length = min(len(traj['states']), len(traj['actions']))
                for step_idx in range(traj_length):
                    # Check if this point belongs to the current submanifold
                    if current_idx in self.get_point_indices_for_submanifold(label):
                        contributing_trajectories.append(traj_idx)
                        point_indices.append((traj_idx, step_idx))
                    current_idx += 1
            
            # Analyze state and action distributions for this submanifold
            state_action_pairs = manifold_data['state_action_pairs']
            state_dim = storage.state_dim
            action_dim = storage.action_dim
            
            # Split into states and actions
            states = state_action_pairs[:, :8]
            actions = state_action_pairs[:, 8:]

            
            characteristics[label] = {
                'size': manifold_data['size'],
                'mean_state': np.mean(states, axis=0),
                'std_state': np.std(states, axis=0),
                'mean_action': stats.mode(actions).mode,
                'contributing_trajectories': list(set(contributing_trajectories)),
                'point_indices': point_indices
            }
        
        return characteristics
    
    def get_point_indices_for_submanifold(self, label):
        """Get indices of points belonging to a specific submanifold"""
        return np.where(self.cluster_labels == label)[0]
    
    def visualize_submanifolds(self, state_action_pairs=None):
        """Visualize the identified submanifolds"""
        if self.projected_points is None:
            raise ValueError("No projected points available. Call identify_submanifolds first.")
        
        plt.figure(figsize=(15, 12))
        
        # Create a colormap for the clusters
        unique_labels = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Plot noise points in black
                mask = self.cluster_labels == label
                plt.scatter(self.projected_points[mask, 0], 
                          self.projected_points[mask, 1],
                          c='black', alpha=0.3, s=10, label='Noise')
            else:
                mask = self.cluster_labels == label
                plt.scatter(self.projected_points[mask, 0], 
                          self.projected_points[mask, 1],
                          c=[colors[i]], alpha=0.7, s=20, 
                          label=f'Submanifold {label} (n={np.sum(mask)})')
                
                # Plot cluster center
                center = np.mean(self.projected_points[mask], axis=0)
                plt.scatter(center[0], center[1], c=[colors[i]], 
                          marker='X', s=200, edgecolors='black', linewidth=2)
        
        plt.title('Identified Sub-manifolds in State-Action Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_submanifold_projection_weights(self):
        """Get the PCA loadings to understand what each submanifold represents"""
        if self.pca is None:
            raise ValueError("PCA not fitted yet.")
        
        loadings = self.pca.components_.T
        return loadings
    
    def extract_representative_trajectories(self, storage, n_trajectories=3):
        """Extract representative trajectories for each submanifold"""
        representative_trajectories = {}
        
        for label in self.submanifolds.keys():
            # Get trajectories that have points in this submanifold
            traj_scores = {}
            
            for traj_idx, traj in enumerate(storage.trajectories):
                # Count how many points from this trajectory are in the submanifold
                points_in_manifold = 0
                current_idx = 0
                
                for i in range(len(storage.trajectories)):
                    if i == traj_idx:
                        traj_length = min(len(traj['states']), len(traj['actions']))
                        try:
                            points_in_manifold = len([j for j in range(current_idx, current_idx + traj_length) 
                                                if self.cluster_labels[j] == label])
                        except:
                            break

                        break
                    current_idx += min(len(storage.trajectories[i]['states']), 
                                     len(storage.trajectories[i]['actions']))
            
                if points_in_manifold > 0:
                    traj_scores[traj_idx] = points_in_manifold
            
            # Get top n trajectories
            top_trajectories = sorted(traj_scores.items(), key=lambda x: x[1], reverse=True)[:n_trajectories]
            representative_trajectories[label] = top_trajectories
        
        return representative_trajectories

# Enhanced main analysis function
def analyze_lunar_lander_submanifolds(storage):
    """Complete analysis for Lunar Lander submanifolds"""
    
    
    # Extract state-action pairs
    state_action_pairs = storage.get_state_action_pairs()
    print(f"Analyzing {len(state_action_pairs)} state-action pairs")
    
    # Initialize and run submanifold analysis
    analyzer = SubManifoldAnalyzer(n_components=2)
    
    # Identify submanifolds using DBSCAN (good for arbitrary shapes)
    cluster_labels = analyzer.identify_submanifolds(
        state_action_pairs, 
        method='dbscan',
        eps=0.3,  # Adjust based on your PCA plot density
        min_samples=20
    )
    
    print(f"Identified {len(analyzer.submanifolds)} submanifolds")
    print(f"Number of noise points: {np.sum(cluster_labels == -1)}")
    
    # Visualize the submanifolds
    analyzer.visualize_submanifolds(state_action_pairs)
    
    # Analyze characteristics of each submanifold
    characteristics = analyzer.analyze_submanifold_characteristics(storage)
    
    # Print detailed analysis
    print("\n=== SUBMANIFOLD ANALYSIS ===")
    for label, chars in characteristics.items():
        print(f"\n--- Submanifold {label} (Size: {chars['size']}) ---")
        print(f"Contributing trajectories: {len(chars['contributing_trajectories'])}")
        print(f"Action mode: {chars['mean_action']}")
        
        # Interpret the state for Lunar Lander
        # Lunar Lander state: [x, y, vel_x, vel_y, angle, angular_vel, left_leg, right_leg]
        state_names = ['x', 'y', 'vel_x', 'vel_y', 'angle', 'angular_vel', 'left_leg', 'right_leg']
        mean_state = chars['mean_state']
        
        print("Key state characteristics:")
        print(f"  Position: ({mean_state[0]:.3f}, {mean_state[1]:.3f})")
        print(f"  Velocity: ({mean_state[2]:.3f}, {mean_state[3]:.3f})")
        print(f"  Angle: {mean_state[4]:.3f} rad")
        print(f"  Angular velocity: {mean_state[5]:.3f}")
    
    # Get PCA loadings to understand what drives the separation
    loadings = analyzer.get_submanifold_projection_weights()
    state_dim = 8 #storage.state_dim
    action_dim = 1 # storage.action_dim
    
    print("\n=== PCA LOADINGS ANALYSIS ===")
    print("What drives PC1 and PC2 separation:")
    
    # State components
    state_names = ['x', 'y', 'vel_x', 'vel_y', 'angle', 'angular_vel', 'left_leg', 'right_leg']
    for i, name in enumerate(state_names):
        print(f"  {name}: PC1={loadings[i, 0]:.3f}, PC2={loadings[i, 1]:.3f}")
    
    # Action components (Lunar Lander has 4 discrete actions)
    action_names = ['do_nothing', 'left_engine', 'main_engine', 'right_engine']
    for i, name in enumerate(action_names):
        idx = state_dim + i
        if idx < loadings.shape[0]:
            print(f"  {name}: PC1={loadings[idx, 0]:.3f}, PC2={loadings[idx, 1]:.3f}")
    
    # Extract representative trajectories
    representative_trajs = analyzer.extract_representative_trajectories(storage, n_trajectories=2)
    
    print("\n=== REPRESENTATIVE TRAJECTORIES ===")
    for label, trajs in representative_trajs.items():
        print(f"Submanifold {label}:")
        for traj_idx, score in trajs:
            traj_reward = np.sum(storage.trajectories[traj_idx]['rewards'])
            print(f"  Trajectory {traj_idx}: {score} points, total reward: {traj_reward:.2f}")
    
    return analyzer, characteristics

def plot_submanifold_behavior_comparison(analyzer, characteristics):
    """Create detailed comparison plots of different submanifolds"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Action distributions
    for label, chars in characteristics.items():
        axes[0, 0].bar([f"M{label}-A{i}" for i in range(len(chars['mean_action']))], 
                      chars['mean_action'], alpha=0.7, label=f'Submanifold {label}')
    axes[0, 0].set_title('Mean Action Values by Submanifold')
    axes[0, 0].set_ylabel('Action Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Key state variables
    key_state_indices = [1, 2, 3, 4]  # y, vel_x, vel_y, angle
    key_state_names = ['Altitude (y)', 'Horizontal Vel', 'Vertical Vel', 'Angle']

    for i, (idx, name) in enumerate(zip(key_state_indices, key_state_names)):
        values = [chars['mean_state'][idx] for chars in characteristics.values()]
        axes[0, 1].bar([f"M{label}" for label in characteristics.keys()], 
                      values, alpha=0.7, label=name)
    axes[0, 1].set_title('Key State Variables by Submanifold')
    axes[0, 1].set_ylabel('State Value')
    axes[0, 1].legend()
    
        
    # Plot 3: Submanifold sizes
    sizes = [chars['size'] for chars in characteristics.values()]
    axes[1, 0].bar([f"Submanifold {label}" for label in characteristics.keys()], sizes)
    axes[1, 0].set_title('Submanifold Sizes')
    axes[1, 0].set_ylabel('Number of Points')
    
    # Plot 4: Trajectory contributions
    traj_counts = [len(chars['contributing_trajectories']) for chars in characteristics.values()]
    axes[1, 1].bar([f"Submanifold {label}" for label in characteristics.keys()], traj_counts)
    axes[1, 1].set_title('Number of Contributing Trajectories')
    axes[1, 1].set_ylabel('Trajectory Count')
    
    plt.tight_layout()
    plt.show()

def extract_behavioral_modes(analyzer, storage):
    """Interpret what each submanifold represents in terms of agent behavior"""
    
    characteristics = analyzer.analyze_submanifold_characteristics(storage)
    behavioral_modes = {}
    
    for label, chars in characteristics.items():
        mean_state = chars['mean_state']
        mean_action = chars['mean_action']
        
        # Lunar Lander specific interpretation
        altitude = mean_state[1]  # y position
        horizontal_vel = mean_state[2]  # vel_x
        vertical_vel = mean_state[3]  # vel_y
        angle = mean_state[4]  # angle
        
        # Determine behavioral mode based on state and action patterns
        if altitude > 0.5:
            mode = "High Altitude Maneuvering"
        elif abs(horizontal_vel) > 0.3:
            mode = "Lateral Correction"
        elif vertical_vel < -0.2:
            mode = "Rapid Descent"
        elif abs(angle) > 0.3:
            mode = "Angle Correction"
        else:
            mode = "Stable Approach"
        
        # Refine based on actions
        dominant_action = np.argmax(np.abs(mean_action))
        action_names = ['Do Nothing', 'Left Engine', 'Main Engine', 'Right Engine']
        
        behavioral_modes[label] = {
            'mode': mode,
            'dominant_action': action_names[dominant_action],
            'altitude': altitude,
            'horizontal_speed': horizontal_vel,
            'vertical_speed': vertical_vel,
            'tilt': angle
        }
    
    return behavioral_modes


def run_anal(path=None):
    storage = RLTrajectoryStorage()
    if path is None:
        storage.load('ll_trajectories.pkl')
    else:
        storage.load(path)

    analyzer, characteristics = analyze_lunar_lander_submanifolds(storage)
    
    # Plot detailed comparisons
    plot_submanifold_behavior_comparison(analyzer, characteristics)
    
    # Extract behavioral interpretations
    behavioral_modes = extract_behavioral_modes(analyzer, storage)
    
    print("\n=== BEHAVIORAL MODES IDENTIFIED ===")
    for label, mode_info in behavioral_modes.items():
        print(f"\nSubmanifold {label}: {mode_info['mode']}")
        print(f"  Dominant Action: {mode_info['dominant_action']}")
        print(f"  Typical State: Altitude={mode_info['altitude']:.3f}, "
              f"HSpeed={mode_info['horizontal_speed']:.3f}, "
              f"VSpeed={mode_info['vertical_speed']:.3f}, "
              f"Tilt={mode_info['tilt']:.3f}")





# Run the complete analysis
if __name__ == "__main__":
   # run_exp() 
   run_anal(path="rl_trajectories.pkl")
    
