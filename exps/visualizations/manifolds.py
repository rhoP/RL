import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
import pickle
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.utils.extmath import randomized_svd
import warnings
warnings.filterwarnings('ignore')



class RandomizedPCA:
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, 
                      svd_solver='randomized', 
                      random_state=random_state)
    
    def fit_transform_fast(self, data):
        """Use randomized SVD for much faster PCA"""
        print(f"Fitting RandomizedPCA on {len(data)} points...")
        return self.pca.fit_transform(data)




class RLTrajectoryStorage:
    def __init__(self, max_trajectories=1000):
        self.max_trajectories = max_trajectories
        self.trajectories = []
        self.state_dim = None
        self.action_dim = None
        
    def add_trajectory(self, states: List[np.ndarray], actions: List[np.ndarray], 
                      rewards: List[float], dones: List[bool],
                      infos: Optional[List[Dict]] = None):
        """Store a complete trajectory"""
        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'infos': infos if infos is not None else [],
            'length': len(states) - 1,  # states include initial state
            'total_reward': np.sum(rewards)
        }
        
        if self.state_dim is None and len(states) > 0:
            self.state_dim = states[0].shape[0] if hasattr(states[0], 'shape') else len(states[0])
        if self.action_dim is None and len(actions) > 0:
            self.action_dim = actions[0].shape[0] if hasattr(actions[0], 'shape') and not all(actions[0].shape) else len(actions[0])
        
        self.trajectories.append(trajectory)
        
        if len(self.trajectories) > self.max_trajectories:
            self.trajectories = self.trajectories[-self.max_trajectories:]
    
    def get_state_action_pairs(self, max_samples=10000) -> np.ndarray:
        """Extract state-action pairs from all trajectories"""
        state_action_pairs = []
        
        for traj in self.trajectories:
            states = traj['states'][:-1]  # Exclude final state (no action taken from it)
            actions = traj['actions']
            
            min_len = min(len(states), len(actions))
            for i in range(min_len):
                state = states[i].flatten()
                action = actions[i].flatten()
                state_action = np.concatenate([state, action])
                state_action_pairs.append(state_action)
                
                if len(state_action_pairs) >= max_samples:
                    return np.array(state_action_pairs)
        
        return np.array(state_action_pairs)
    
    def get_all_states(self) -> np.ndarray:
        """Get all states from all trajectories"""
        states = []
        for traj in self.trajectories:
            states.extend(traj['states'])
        return np.array(states)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'trajectories': self.trajectories,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.trajectories = data['trajectories']
            self.state_dim = data['state_dim']
            self.action_dim = data['action_dim']

class MeshPropagator:
    def __init__(self, model, env, state_dim, action_dim):
        self.model = model
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def generate_mesh_around_trajectories(self, storage: RLTrajectoryStorage, 
                                        n_points_per_dim: int = 10,
                                        padding: float = 0.1) -> np.ndarray:
        """Generate a coarse mesh around the stored trajectories"""
        # Get all states from trajectories
        all_states = storage.get_all_states()
        
        # Create bounding box around trajectories with padding
        min_bounds = np.min(all_states, axis=0) * (1 + padding)
        max_bounds = np.max(all_states, axis=0) * (1 + padding)
        
        # Generate mesh grid
        mesh_points = []
        for dim in range(self.state_dim):
            dim_points = np.linspace(min_bounds[dim], max_bounds[dim], n_points_per_dim)
            mesh_points.append(dim_points)
        
        # Create mesh grid
        mesh_grid = np.meshgrid(*mesh_points)
        mesh_states = np.vstack([grid.ravel() for grid in mesh_grid]).T
        
        print(f"Generated mesh with {len(mesh_states)} points")
        return mesh_states
    
    def propagate_state_forward(self, state: np.ndarray, steps: int = 10) -> np.ndarray:
        """Propagate a single state forward in time using the trained policy"""
        propagated_states = [state]
        current_state = state.copy()
        
        for step in range(steps):
            action, _ = self.model.predict(current_state, deterministic=True)
            # Use the environment's dynamics (simulated step)
            # For LunarLander, we need to simulate the dynamics
            next_state = self._simulate_dynamics(current_state, action)
            propagated_states.append(next_state)
            current_state = next_state
            
        return np.array(propagated_states)
    
    def propagate_state_backward(self, state: np.ndarray, steps: int = 10) -> np.ndarray:
        """Propagate a single state backward in time (approximate inverse dynamics)"""
        propagated_states = [state]
        current_state = state.copy()
        
        for step in range(steps):
            # For backward propagation, we need to estimate the previous state
            # This is challenging without true inverse dynamics
            prev_state = self._estimate_previous_state(current_state)
            if prev_state is not None:
                propagated_states.insert(0, prev_state)
                current_state = prev_state
            else:
                break
                
        return np.array(propagated_states)
    
    def _simulate_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Simulate Lunar Lander dynamics (simplified)"""
        # Lunar Lander state: [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = state
        
        # Action: 0: none, 1: left, 2: main, 3: right
        action = action[0] if isinstance(action, np.ndarray) else action
        
        # Constants for Lunar Lander
        gravity = -0.05
        thrust_power = 0.15
        rotation_power = 0.1
        dt = 0.05
        
        # Apply action
        if action == 1:  # Left engine
            vx += -thrust_power * np.sin(angle) * dt
            vy += thrust_power * np.cos(angle) * dt
            ang_vel += rotation_power * dt
        elif action == 2:  # Main engine
            vx += thrust_power * np.sin(angle) * dt
            vy += thrust_power * np.cos(angle) * dt
        elif action == 3:  # Right engine
            vx += thrust_power * np.sin(angle) * dt
            vy += thrust_power * np.cos(angle) * dt
            ang_vel += -rotation_power * dt
        
        # Apply gravity
        vy += gravity * dt
        
        # Update position and angle
        x += vx * dt
        y += vy * dt
        angle += ang_vel * dt
        
        # Normalize angle
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi
        
        # Ground contact
        if y <= 0:
            y = 0
            vy = 0
            vx = vx * 0.8  # Friction
            if abs(angle) < 0.2:  # Leg contact
                if abs(x) < 0.2:
                    left_leg = 1
                    right_leg = 1
                elif x < 0:
                    left_leg = 1
                else:
                    right_leg = 1
        
        return np.array([x, y, vx, vy, angle, ang_vel, left_leg, right_leg])
    
    def _estimate_previous_state(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Estimate previous state (simplified inverse dynamics)"""
        # This is a rough approximation - in practice, you'd need proper inverse dynamics
        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = state
        
        dt = 0.05
        gravity = -0.05
        
        # Rough backward estimation
        prev_vx = vx * 0.9  # Assume some damping
        prev_vy = (vy - gravity * dt) * 0.9
        prev_x = x - prev_vx * dt
        prev_y = y - prev_vy * dt
        prev_angle = angle - ang_vel * dt
        prev_ang_vel = ang_vel * 0.9
        
        # Normalize angle
        prev_angle = ((prev_angle + np.pi) % (2 * np.pi)) - np.pi
        
        # Reset legs if we're above ground
        if prev_y > 0:
            prev_left_leg = 0
            prev_right_leg = 0
        else:
            prev_left_leg = left_leg
            prev_right_leg = right_leg
            
        return np.array([prev_x, prev_y, prev_vx, prev_vy, prev_angle, prev_ang_vel, 
                        prev_left_leg, prev_right_leg])
    
    def propagate_mesh_points(self, mesh_states: np.ndarray, 
                            forward_steps: int = 5, 
                            backward_steps: int = 5) -> Dict[str, np.ndarray]:
        """Propagate all mesh points forward and backward in time"""
        all_forward_trajectories = []
        all_backward_trajectories = []
        all_extended_trajectories = []
        
        for i, state in enumerate(mesh_states):
            if i % 100 == 0:
                print(f"Propagating point {i}/{len(mesh_states)}")
            
            # Forward propagation
            forward_traj = self.propagate_state_forward(state, forward_steps)
            all_forward_trajectories.append(forward_traj)
            
            # Backward propagation
            backward_traj = self.propagate_state_backward(state, backward_steps)
            all_backward_trajectories.append(backward_traj)
            
            # Combined trajectory (backward + original + forward)
            extended_traj = np.vstack([backward_traj, forward_traj[1:]])  # Avoid duplicate center point
            all_extended_trajectories.append(extended_traj)
        
        return {
            'forward_trajectories': all_forward_trajectories,
            'backward_trajectories': all_backward_trajectories,
            'extended_trajectories': all_extended_trajectories,
            'mesh_centers': mesh_states
        }


class TrajectoryVisualizer:
    def __init__(self, state_dim, action_dim, is_continuous=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_continuous = is_continuous
        self.pca = RandomizedPCA(n_components=2)
        
    def visualize_propagated_mesh(self, propagation_results: Dict[str, np.ndarray],
                                original_trajectories: RLTrajectoryStorage,
                                model=None,
                                show_original: bool = True):
        """Visualize the original trajectories and propagated mesh points"""
        
        # Fit PCA on original trajectories
        original_state_action_pairs = original_trajectories.get_state_action_pairs()
        self.pca.fit_transform_fast(original_state_action_pairs)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original trajectories in PCA space
        if show_original:
            self._plot_original_trajectories(axes[0, 0], original_trajectories)
        
        # Plot 2: Mesh centers with policy actions
        self._plot_mesh_centers(axes[0, 1], propagation_results['mesh_centers'], model)
        
        # Plot 3: Forward propagated trajectories
        self._plot_propagated_trajectories(axes[1, 0], propagation_results['forward_trajectories'], 
                                         'Forward Propagated', model)
        
        # Plot 4: Extended trajectories (backward + forward)
        self._plot_propagated_trajectories(axes[1, 1], propagation_results['extended_trajectories'],
                                         'Extended Trajectories', model)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_original_trajectories(self, ax, storage: RLTrajectoryStorage):
        """Plot original trajectories in PCA space"""
        state_action_pairs = storage.get_state_action_pairs()
        projected = self.pca.transform(state_action_pairs)
        
        # Color by trajectory or by action
        if self.is_continuous:
            # Color by action magnitude for continuous actions
            actions = state_action_pairs[:, self.state_dim:]
            action_magnitude = np.linalg.norm(actions, axis=1)
            scatter = ax.scatter(projected[:, 0], projected[:, 1], 
                               c=action_magnitude, cmap='viridis', 
                               alpha=0.6, s=10)
            plt.colorbar(scatter, ax=ax, label='Action Magnitude')
        else:
            # Color by trajectory for discrete actions
            colors = plt.cm.viridis(np.linspace(0, 1, len(storage.trajectories)))
            start_idx = 0
            for i, traj in enumerate(storage.trajectories):
                traj_length = len(traj['actions'])
                traj_points = projected[start_idx:start_idx + traj_length]
                ax.plot(traj_points[:, 0], traj_points[:, 1], color=colors[i], alpha=0.7, linewidth=1)
                start_idx += traj_length
        
        ax.set_title('Original Trajectories (PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
    

    def _plot_mesh_centers(self, ax, mesh_centers: np.ndarray, model=None):
        """Plot mesh centers in PCA space for continuous actions"""
        # For continuous actions, we need to get the actual actions from the policy
        if model is None:
            # If no model provided, use zero actions as fallback
            zero_actions = np.zeros((len(mesh_centers), 2))  # Lunar Lander continuous has 2 actions
            mesh_state_action = np.hstack([mesh_centers, zero_actions])
        else:
            # Get the actual actions the policy would take for each mesh state
            actions = []
            for state in mesh_centers:
                action, _ = model.predict(state, deterministic=True)
                actions.append(action)
            actions = np.array(actions)
            mesh_state_action = np.hstack([mesh_centers, actions])
        
        projected = self.pca.transform(mesh_state_action)
        
        # Color by action magnitude or specific action component
        if model is not None:
            action_magnitude = np.linalg.norm(actions, axis=1)
            scatter = ax.scatter(projected[:, 0], projected[:, 1], 
                            c=action_magnitude, cmap='viridis', 
                            alpha=0.8, s=30)
            plt.colorbar(scatter, ax=ax, label='Action Magnitude')
        else:
            ax.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=20, c='red')
        
        ax.set_title('Mesh Centers with Policy Actions')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
    
    def _plot_propagated_trajectories(self, ax, trajectories: List[np.ndarray], 
                                    title: str, model=None):
        """Plot propagated trajectories in PCA space for continuous actions"""
        if model is None:
            # Use zero actions if no model provided
            all_points = []
            for traj in trajectories:
                if len(traj) == 0:
                    continue
                zero_actions = np.zeros((len(traj), self.action_dim))
                traj_state_action = np.hstack([traj, zero_actions])
                all_points.append(traj_state_action)
            
            if all_points:
                all_points = np.vstack(all_points)
                projected_all = self.pca.transform(all_points)
                ax.scatter(projected_all[:, 0], projected_all[:, 1], 
                          alpha=0.3, s=5, c='blue')
        else:
            # Get actual actions for each point in each trajectory
            colors = plt.cm.plasma(np.linspace(0, 1, len(trajectories)))
            
            for i, traj in enumerate(trajectories):
                if len(traj) == 0:
                    continue
                
                # Get actions for this trajectory
                traj_actions = []
                for state in traj:
                    action, _ = model.predict(state, deterministic=True)
                    traj_actions.append(action)
                traj_actions = np.array(traj_actions)
                
                traj_state_action = np.hstack([traj, traj_actions])
                projected = self.pca.transform(traj_state_action)
                
                # Color by trajectory index
                ax.plot(projected[:, 0], projected[:, 1], 
                       color=colors[i], alpha=0.4, linewidth=1)
        
        ax.set_title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)


def train_ppo_lunar_lander(total_timesteps=100000):
    """Train a PPO agent on Lunar Lander"""
    print("Training PPO on LunarLander-v3...")
    
    env = gym.make('LunarLander-v3', continuous=True)
    env = DummyVecEnv([lambda: env])
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device='cpu'
    )
    
    model.learn(total_timesteps=total_timesteps)
    return model, env

def collect_trajectories(model, env_name, num_trajectories=50):
    """Collect trajectories using the trained policy"""
    storage = RLTrajectoryStorage()
    env = gym.make(env_name, continuous=True)
    
    for episode in range(num_trajectories):
        state, info = env.reset()
        states = [state]
        actions = []
        rewards = []
        dones = []
        infos = [info]
        
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)
            
            state = next_state
            total_reward += reward
        
        storage.add_trajectory(states, actions, rewards, dones, infos)
        print(f"Collected trajectory {episode + 1}: reward = {total_reward:.2f}")
    
    env.close()
    return storage


class FastMeshPropagator(MeshPropagator):
    def __init__(self, model, env, state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Extract the policy network for faster inference
        self.policy_net = self._extract_policy_network()
        self.policy_net.to(device)
        self.policy_net.eval()
        
        # Pre-compile dynamics function if possible
        self.dynamics_fn = self._create_dynamics_function()
        
    def _extract_policy_network(self):
        """Extract the policy network from the SB3 model for faster inference"""
        # SB3 stores the policy in model.policy
        if hasattr(self.model.policy, 'mlp_extractor'):
            # For MlpPolicy
            return self.model.policy.mlp_extractor
        else:
            # Fallback: create a wrapper for the full policy
            class PolicyWrapper(nn.Module):
                def __init__(self, sb3_policy):
                    super().__init__()
                    self.sb3_policy = sb3_policy
                
                def forward(self, x):
                    # This is a simplified version - you may need to adapt based on your SB3 version
                    return self.sb3_policy(x, deterministic=True)[0]
            
            return PolicyWrapper(self.model.policy)
    
    def _create_dynamics_function(self):
        """Create a vectorized dynamics function"""
        def vectorized_dynamics(states, actions):
            """Vectorized Lunar Lander dynamics"""
            # states: [batch_size, 8], actions: [batch_size, 2]
            batch_size = states.shape[0]
            
            # Extract state components
            x = states[:, 0]
            y = states[:, 1]
            vx = states[:, 2]
            vy = states[:, 3]
            angle = states[:, 4]
            ang_vel = states[:, 5]
            left_leg = states[:, 6]
            right_leg = states[:, 7]
            
            # Extract action components
            main_engine = actions[:, 0]  # Main engine power
            orientation = actions[:, 1]  # Orientation control
            
            # Constants
            gravity = -0.05
            thrust_power = 0.15
            rotation_power = 0.1
            dt = 0.05
            
            # Apply actions (continuous version)
            vx += thrust_power * torch.sin(angle) * main_engine * dt
            vy += thrust_power * torch.cos(angle) * main_engine * dt
            ang_vel += rotation_power * orientation * dt
            
            # Apply gravity
            vy += gravity * dt
            
            # Update position and angle
            x += vx * dt
            y += vy * dt
            angle += ang_vel * dt
            
            # Normalize angle
            angle = ((angle + np.pi) % (2 * np.pi)) - np.pi
            
            # Ground contact (vectorized)
            on_ground = y <= 0
            y = torch.where(on_ground, torch.zeros_like(y), y)
            vy = torch.where(on_ground, torch.zeros_like(vy), vy)
            vx = torch.where(on_ground, vx * 0.8, vx)  # Friction
            
            # Leg contact (simplified)
            left_leg = torch.where(on_ground & (x < 0.2), torch.ones_like(left_leg), left_leg)
            right_leg = torch.where(on_ground & (x > -0.2), torch.ones_like(right_leg), right_leg)
            
            # Combine new state
            new_states = torch.stack([x, y, vx, vy, angle, ang_vel, left_leg, right_leg], dim=1)
            return new_states
        
        return vectorized_dynamics
    
    def batch_predict_actions(self, states_batch: np.ndarray) -> np.ndarray:
        """Batch predict actions using GPU"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states_batch).to(self.device)
            
            # Use the policy network for faster inference
            if hasattr(self.policy_net, 'forward'):
                actions_tensor = self.policy_net(states_tensor)
            else:
                # Fallback to original model (slower)
                actions = []
                batch_size = 1024  # Process in smaller batches to avoid memory issues
                for i in range(0, len(states_batch), batch_size):
                    batch = states_batch[i:i+batch_size]
                    batch_actions = []
                    for state in batch:
                        action, _ = self.model.predict(state, deterministic=True)
                        batch_actions.append(action)
                    actions.extend(batch_actions)
                return np.array(actions)
            
            return actions_tensor.cpu().numpy()
    
    def propagate_states_batch(self, states_batch: np.ndarray, actions_batch: np.ndarray) -> np.ndarray:
        """Propagate a batch of states using vectorized dynamics"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states_batch).to(self.device)
            actions_tensor = torch.FloatTensor(actions_batch).to(self.device)
            
            next_states = self.dynamics_fn(states_tensor, actions_tensor)
            return next_states.cpu().numpy()
    
    def propagate_mesh_points_fast(self, mesh_states: np.ndarray, 
                                 forward_steps: int = 5, 
                                 backward_steps: int = 5,
                                 batch_size: int = 4096) -> Dict[str, np.ndarray]:
        """Fast propagation using batched operations and GPU"""
        print(f"Fast propagation with batch size {batch_size} on {self.device}")
        
        all_forward_trajectories = []
        all_backward_trajectories = []
        
        # Process in batches for memory efficiency
        num_batches = (len(mesh_states) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        # Forward propagation (batched)
        print("Forward propagation...")
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(mesh_states))
            batch_states = mesh_states[batch_start:batch_end]
            
            if batch_idx % 10 == 0:
                print(f"Forward batch {batch_idx+1}/{num_batches}")
            
            batch_forward_trajectories = self._propagate_batch_forward_fast(
                batch_states, forward_steps, sub_batch_size=1024)
            all_forward_trajectories.extend(batch_forward_trajectories)
        
        # Backward propagation (batched)
        print("Backward propagation...")
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(mesh_states))
            batch_states = mesh_states[batch_start:batch_end]
            
            if batch_idx % 10 == 0:
                print(f"Backward batch {batch_idx+1}/{num_batches}")
            
            batch_backward_trajectories = self._propagate_batch_backward_fast(
                batch_states, backward_steps, sub_batch_size=1024)
            all_backward_trajectories.extend(batch_backward_trajectories)
        
        # Create extended trajectories
        all_extended_trajectories = []
        for i in range(len(mesh_states)):
            extended_traj = np.vstack([all_backward_trajectories[i], 
                                     all_forward_trajectories[i][1:]])  # Avoid duplicate center
            all_extended_trajectories.append(extended_traj)
        
        end_time = time.time()
        print(f"Propagation completed in {end_time - start_time:.2f} seconds")
        
        return {
            'forward_trajectories': all_forward_trajectories,
            'backward_trajectories': all_backward_trajectories,
            'extended_trajectories': all_extended_trajectories,
            'mesh_centers': mesh_states
        }
    
    def _propagate_batch_forward_fast(self, batch_states: np.ndarray, steps: int, 
                                    sub_batch_size: int = 1024) -> List[np.ndarray]:
        """Fast forward propagation for a batch of states"""
        batch_trajectories = []
        
        # Process each state in the batch
        for i in range(len(batch_states)):
            trajectory = [batch_states[i]]
            current_state = batch_states[i].copy()
            
            for step in range(steps):
                # Get action for current state
                action = self.batch_predict_actions(current_state.reshape(1, -1))[0]
                
                # Propagate state
                next_state = self.propagate_states_batch(
                    current_state.reshape(1, -1), 
                    action.reshape(1, -1)
                )[0]
                
                trajectory.append(next_state)
                current_state = next_state
            
            batch_trajectories.append(np.array(trajectory))
        
        return batch_trajectories
    
    def _propagate_batch_backward_fast(self, batch_states: np.ndarray, steps: int,
                                     sub_batch_size: int = 1024) -> List[np.ndarray]:
        """Fast backward propagation for a batch of states"""
        batch_trajectories = []
        
        for i in range(len(batch_states)):
            trajectory = [batch_states[i]]
            current_state = batch_states[i].copy()
            
            for step in range(steps):
                prev_state = self._estimate_previous_state_batch(
                    current_state.reshape(1, -1)
                )[0]
                if prev_state is not None:
                    trajectory.insert(0, prev_state)
                    current_state = prev_state
                else:
                    break
            
            batch_trajectories.append(np.array(trajectory))
        
        return batch_trajectories
    
    def _estimate_previous_state_batch(self, states_batch: np.ndarray) -> np.ndarray:
        """Vectorized backward dynamics estimation"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states_batch).to(self.device)
            
            # Extract state components
            x = states_tensor[:, 0]
            y = states_tensor[:, 1]
            vx = states_tensor[:, 2]
            vy = states_tensor[:, 3]
            angle = states_tensor[:, 4]
            ang_vel = states_tensor[:, 5]
            left_leg = states_tensor[:, 6]
            right_leg = states_tensor[:, 7]
            
            dt = 0.05
            gravity = -0.05
            
            # Rough backward estimation (vectorized)
            prev_vx = vx * 0.9
            prev_vy = (vy - gravity * dt) * 0.9
            prev_x = x - prev_vx * dt
            prev_y = y - prev_vy * dt
            prev_angle = angle - ang_vel * dt
            prev_ang_vel = ang_vel * 0.9
            
            # Normalize angle
            prev_angle = ((prev_angle + np.pi) % (2 * np.pi)) - np.pi
            
            # Reset legs if above ground
            prev_left_leg = torch.where(prev_y > 0, torch.zeros_like(left_leg), left_leg)
            prev_right_leg = torch.where(prev_y > 0, torch.zeros_like(right_leg), right_leg)
            
            prev_states = torch.stack([prev_x, prev_y, prev_vx, prev_vy, prev_angle, 
                                     prev_ang_vel, prev_left_leg, prev_right_leg], dim=1)
            
            return prev_states.cpu().numpy()

# Even faster version using full vectorization across time steps
class UltraFastMeshPropagator(FastMeshPropagator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def propagate_mesh_points_ultrafast(self, mesh_states: np.ndarray,
                                      forward_steps: int = 5,
                                      backward_steps: int = 5) -> Dict[str, np.ndarray]:
        """Ultra-fast propagation using full vectorization across time and space"""
        print(f"Ultra-fast propagation on {self.device}")
        start_time = time.time()
        
        # Convert to tensor
        states_tensor = torch.FloatTensor(mesh_states).to(self.device)
        n_points = len(mesh_states)
        
        # Forward propagation (fully vectorized)
        print("Vectorized forward propagation...")
        forward_trajectories = self._vectorized_forward_propagation(
            states_tensor, forward_steps)
        
        # Backward propagation (fully vectorized)
        print("Vectorized backward propagation...")
        backward_trajectories = self._vectorized_backward_propagation(
            states_tensor, backward_steps)
        
        # Create extended trajectories
        extended_trajectories = []
        for i in range(n_points):
            extended_traj = torch.cat([backward_trajectories[i], 
                                     forward_trajectories[i][1:]], dim=0)
            extended_trajectories.append(extended_traj.cpu().numpy())
        
        end_time = time.time()
        print(f"Ultra-fast propagation completed in {end_time - start_time:.2f} seconds")
        
        return {
            'forward_trajectories': [traj.cpu().numpy() for traj in forward_trajectories],
            'backward_trajectories': [traj.cpu().numpy() for traj in backward_trajectories],
            'extended_trajectories': extended_trajectories,
            'mesh_centers': mesh_states
        } # type: ignore
    
    def _vectorized_forward_propagation(self, initial_states: torch.Tensor, 
                                      steps: int) -> List[torch.Tensor]:
        """Fully vectorized forward propagation"""
        n_points = initial_states.shape[0]
        
        # Initialize trajectories tensor [n_points, steps+1, state_dim]
        trajectories = torch.zeros(n_points, steps + 1, self.state_dim).to(self.device)
        trajectories[:, 0] = initial_states
        
        current_states = initial_states
        
        for step in range(steps):
            # Batch predict actions for all current states
            with torch.no_grad():
                actions = self.policy_net(current_states)
            
            # Vectorized dynamics step
            next_states = self.dynamics_fn(current_states, actions)
            
            # Store in trajectories
            trajectories[:, step + 1] = next_states
            current_states = next_states
        
        # Convert to list of trajectories
        return [trajectories[i] for i in range(n_points)]
    
    def _vectorized_backward_propagation(self, initial_states: torch.Tensor,
                                       steps: int) -> List[torch.Tensor]:
        """Fully vectorized backward propagation"""
        n_points = initial_states.shape[0]
        
        # Initialize trajectories tensor [n_points, steps+1, state_dim]
        trajectories = torch.zeros(n_points, steps + 1, self.state_dim).to(self.device)
        trajectories[:, -1] = initial_states  # Start from the end
        
        current_states = initial_states
        
        for step in range(steps):
            # Vectorized backward dynamics
            prev_states = self._vectorized_backward_dynamics(current_states)
            
            # Store in trajectories (working backwards)
            trajectories[:, steps - step - 1] = prev_states
            current_states = prev_states
        
        return [trajectories[i] for i in range(n_points)]
    
    def _vectorized_backward_dynamics(self, states: torch.Tensor) -> torch.Tensor:
        """Vectorized backward dynamics"""
        # Extract state components
        x = states[:, 0]
        y = states[:, 1]
        vx = states[:, 2]
        vy = states[:, 3]
        angle = states[:, 4]
        ang_vel = states[:, 5]
        left_leg = states[:, 6]
        right_leg = states[:, 7]
        
        dt = 0.05
        gravity = -0.05
        
        # Backward estimation
        prev_vx = vx * 0.9
        prev_vy = (vy - gravity * dt) * 0.9
        prev_x = x - prev_vx * dt
        prev_y = y - prev_vy * dt
        prev_angle = angle - ang_vel * dt
        prev_ang_vel = ang_vel * 0.9
        
        # Normalize angle
        prev_angle = ((prev_angle + np.pi) % (2 * np.pi)) - np.pi
        
        # Reset legs if above ground
        prev_left_leg = torch.where(prev_y > 0, torch.zeros_like(left_leg), left_leg)
        prev_right_leg = torch.where(prev_y > 0, torch.zeros_like(right_leg), right_leg)
        
        return torch.stack([prev_x, prev_y, prev_vx, prev_vy, prev_angle, 
                          prev_ang_vel, prev_left_leg, prev_right_leg], dim=1)


def main():
    # Step 1: Train or load PPO model
    try:
        model = PPO.load("ppo_lunar_lander")
        print("Loaded pre-trained model")
        env = gym.make('LunarLander-v3', continuous=True)
    except:
        print("Training new model...")
        model, vec_env = train_ppo_lunar_lander(total_timesteps=100000)
        model.save("ppo_lunar_lander")
        env = vec_env.envs[0]
    
    # Step 2: Collect trajectories
    print("\nCollecting trajectories...")
    storage = collect_trajectories(model, 'LunarLander-v3', num_trajectories=30)
    
    # Step 3: Generate mesh and propagate
    print("\nGenerating mesh and propagating...")
    propagator = UltraFastMeshPropagator(model, env, storage.state_dim, storage.action_dim)
    
    # Generate mesh around trajectories
    mesh_states = propagator.generate_mesh_around_trajectories(
        storage, 
        n_points_per_dim=8,  # You can increase this with fast propagation
        padding=0.2
    )
    
    # Fast propagation
    propagation_results = propagator.propagate_mesh_points_ultrafast(
        mesh_states,
        forward_steps=10,
        backward_steps=5
    )
    
    # Step 4: Visualize results with continuous action support
    print("\nVisualizing results...")
    visualizer = TrajectoryVisualizer(storage.state_dim, storage.action_dim, is_continuous=True)
    visualizer.visualize_propagated_mesh(propagation_results, storage, model=model)
    
    # Save everything
    storage.save('lunar_lander_continuous_trajectories.pkl')
    
    with open('propagation_results_continuous.pkl', 'wb') as f:
        pickle.dump(propagation_results, f)
    
    print("\nAnalysis complete!")
    print(f"Stored {len(storage.trajectories)} trajectories")
    print(f"Generated mesh with {len(mesh_states)} points")

if __name__ == "__main__":
    main()