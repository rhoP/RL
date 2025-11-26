import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import pandas as pd
from scipy import stats

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
            states = state_action_pairs[:, :state_dim]
            actions = state_action_pairs[:, state_dim:]
            
            characteristics[label] = {
                'size': manifold_data['size'],
                'mean_state': np.mean(states, axis=0),
                'std_state': np.std(states, axis=0),
                'mean_action': np.mean(actions, axis=0),
                'std_action': np.std(actions, axis=0),
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
                        points_in_manifold = len([j for j in range(current_idx, current_idx + traj_length) 
                                                if self.cluster_labels[j] == label])
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
def analyze_lunar_lander_submanifolds():
    """Complete analysis for Lunar Lander submanifolds"""
    
    # Load your existing data
    storage = RLTrajectoryStorage()
    storage.load('rl_trajectories.pkl')
    
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
        print(f"Mean action: {chars['mean_action']}")
        print(f"Action std: {chars['std_action']}")
        
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
    state_dim = storage.state_dim
    action_dim = storage.action_dim
    
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

# Additional function to extract specific behavioral modes
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

# Run the complete analysis
if __name__ == "__main__":
    analyzer, characteristics = analyze_lunar_lander_submanifolds()
    
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
