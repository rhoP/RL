import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from stable_baselines3 import PPO
import torch
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def train_ppo_frozenlake_stochastic(env_name="FrozenLake-v1", map_name="4x4", total_timesteps=100000):
    """Train a PPO agent on stochastic FrozenLake"""
    
    env = gym.make(
        env_name,
        desc=None,
        map_name=map_name,
        is_slippery=True,  
        render_mode=None
    )
    
    print(f"Training PPO on stochastic {map_name} FrozenLake...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=42
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    return model, env

def extract_stochastic_policy(model, env):
    """Extract the stochastic policy and state values"""
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    action_probs = {}
    state_values = {}
    q_values_approx = {}
    
    for state in range(state_size):
        # Get action probabilities (stochastic policy)
        obs = np.array([state])
        probs = model.policy.get_distribution(torch.tensor(obs)).distribution.probs.detach().numpy()[0]
        action_probs[state] = probs
        
        # Get state value
        state_values[state] = model.policy.predict_values(torch.tensor(obs.reshape(1, -1))).item()
        
        # Approximate Q-values using policy and state value
        for action in range(action_size):
            q_values_approx[(state, action)] = probs[action] * state_values[state] * 10
    
    return action_probs, state_values, q_values_approx

def collect_stochastic_transitions(env, action_probs, num_samples=2000):
    """Collect transition samples under the stochastic policy"""
    
    transition_counts = defaultdict(lambda: defaultdict(int))
    state_action_visits = defaultdict(int)
    
    print("Collecting stochastic transition samples...")
    
    for sample in range(num_samples):
        state, _ = env.reset()
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Sample action from stochastic policy
            action_probs_current = action_probs[state]
            action = np.random.choice(4, p=action_probs_current)
            
            state_action_visits[(state, action)] += 1
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Record transition
            transition_counts[(state, action)][next_state] += 1
            
            state = next_state
            
            if terminated or truncated:
                break
    
    # Convert counts to probabilities
    transition_probs = {}
    for (state, action), next_state_counts in transition_counts.items():
        total = sum(next_state_counts.values())
        transition_probs[(state, action)] = {
            next_state: count / total for next_state, count in next_state_counts.items()
        }
    
    return transition_probs, state_action_visits

def create_3d_state_action_lattice(env, action_probs, state_values, transition_probs, state_action_visits):
    """Create a 3D representation of the state-action lattice"""
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    grid_size = int(np.sqrt(state_size))
    
    # Create 3D positions for state-action pairs
    positions_3d = {}
    node_colors = {}
    node_sizes = {}
    node_alphas = {}
    
    action_offsets = {
        0: (-0.2, 0, 0),    # Left
        1: (0, -0.2, 0),    # Down  
        2: (0.2, 0, 0),     # Right
        3: (0, 0.2, 0)      # Up
    }
    
    for state in range(state_size):
        # Grid position
        row = state // grid_size
        col = state % grid_size
        
        base_x = col
        base_y = grid_size - 1 - row  # Flip y for visualization
        base_z = 0  # State layer
        
        for action in range(action_size):
            node_id = f"S{state}_A{action}"
            
            # Offset based on action
            offset_x, offset_y, offset_z = action_offsets[action]
            positions_3d[node_id] = (base_x + offset_x, base_y + offset_y, action)
            
            # Node properties
            prob = action_probs[state][action]
            node_colors[node_id] = prob  # Color by action probability
            node_sizes[node_id] = 100 + prob * 500  # Size by probability
            node_alphas[node_id] = 0.3 + prob * 0.7  # Alpha by probability
    
    return positions_3d, node_colors, node_sizes, node_alphas

def plot_3d_policy_lattice(env, action_probs, state_values, transition_probs, state_action_visits):
    """Plot the 3D state-action lattice with policy paths"""
    
    state_size = env.observation_space.n
    grid_size = int(np.sqrt(state_size))
    
    # Create 3D lattice data
    positions_3d, node_colors, node_sizes, node_alphas = create_3d_state_action_lattice(
        env, action_probs, state_values, transition_probs, state_action_visits
    )
    
    # Create the figure
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: 3D State-Action Lattice
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Plot all state-action nodes
    for node_id, (x, y, z) in positions_3d.items():
        state = int(node_id.split('_')[0][1:])
        action = int(node_id.split('_')[1][1:])
        prob = action_probs[state][action]
        
        color = plt.cm.viridis(prob)
        size = node_sizes[node_id]
        alpha = node_alphas[node_id]
        
        ax1.scatter(x, y, z, c=[color], s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
        
        # Add labels for significant actions
        if prob > 0.2:
            action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
            ax1.text(x, y, z, f"S{state}\n{action_symbols[action]}", 
                    fontsize=8, ha='center', va='center')
    
    # Plot transitions for the most probable actions
    plot_most_probable_paths(ax1, positions_3d, action_probs, transition_probs, state_values)
    
    ax1.set_xlabel('Grid X')
    ax1.set_ylabel('Grid Y')
    ax1.set_zlabel('Action')
    ax1.set_title('3D State-Action Policy Lattice\n(Color = Action Probability)', 
                  fontsize=12, fontweight='bold')
    
    # Plot 2: 2D Policy Projection (XY plane)
    ax2 = fig.add_subplot(2, 2, 2)
    plot_2d_policy_projection(ax2, positions_3d, action_probs, state_values, grid_size)
    
    # Plot 3: Action Probability Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    plot_action_probability_histogram(ax3, action_probs)
    
    # Plot 4: State Value Surface
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_state_value_surface(ax4, state_values, grid_size)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_most_probable_paths(ax, positions_3d, action_probs, transition_probs, state_values, num_paths=5):
    """Plot the most probable paths through the state-action lattice"""
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_paths))
    
    for path_idx in range(num_paths):
        path = []
        current_state = 0  # Start state
        visited_states = set()
        max_steps = 20
        
        for step in range(max_steps):
            if current_state in visited_states:
                break
            visited_states.add(current_state)
            
            # Choose action based on policy probabilities
            probs = action_probs[current_state]
            # Add some randomness to see different paths
            if path_idx == 0:
                action = np.argmax(probs)  # Most probable action for first path
            else:
                # Sample from top actions for variety
                top_actions = np.argsort(probs)[-2:]
                action = np.random.choice(top_actions)
            
            current_node = f"S{current_state}_A{action}"
            path.append(current_node)
            
            # Find most probable next state
            if (current_state, action) in transition_probs:
                next_state_probs = transition_probs[(current_state, action)]
                if next_state_probs:
                    next_state = max(next_state_probs.items(), key=lambda x: x[1])[0]
                    current_state = next_state
                else:
                    break
            else:
                break
            
            # Stop if reached terminal state or loop
            if current_state == 15 or current_state in [5, 7, 11, 12]:  # Goal or holes
                path.append(f"S{current_state}_A0")  # Add terminal state
                break
        
        # Plot the path
        if len(path) > 1:
            x_path = [positions_3d[node][0] for node in path]
            y_path = [positions_3d[node][1] for node in path]
            z_path = [positions_3d[node][2] for node in path]
            
            ax.plot(x_path, y_path, z_path, 
                   color=colors[path_idx], linewidth=3, alpha=0.8,
                   label=f'Path {path_idx+1}')
    
    ax.legend()

def plot_2d_policy_projection(ax, positions_3d, action_probs, state_values, grid_size):
    """Plot 2D projection of the policy"""
    
    # Create a 2D grid showing the most probable action at each state
    policy_grid = np.zeros((grid_size, grid_size))
    action_grid = np.empty((grid_size, grid_size), dtype=object)
    
    for state in range(grid_size * grid_size):
        row = state // grid_size
        col = state % grid_size
        
        # Find most probable action
        best_action = np.argmax(action_probs[state])
        policy_grid[grid_size - 1 - row, col] = action_probs[state][best_action]
        action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        action_grid[grid_size - 1 - row, col] = action_symbols[best_action]
    
    # Plot the policy grid
    im = ax.imshow(policy_grid, cmap='viridis', alpha=0.8)
    
    # Annotate with actions
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            text = f"S{state}\n{action_grid[i, j]}\n{policy_grid[i, j]:.2f}"
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.set_title('2D Policy Projection\n(Most Probable Action)', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label='Action Probability')

def plot_action_probability_histogram(ax, action_probs):
    """Plot histogram of action probabilities"""
    
    all_probs = []
    for state in action_probs:
        all_probs.extend(action_probs[state])
    
    ax.hist(all_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Action Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Probability Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_prob = np.mean(all_probs)
    ax.axvline(mean_prob, color='red', linestyle='--', label=f'Mean: {mean_prob:.3f}')
    ax.legend()

def plot_state_value_surface(ax, state_values, grid_size):
    """Plot state values as a 3D surface"""
    
    # Create value grid
    value_grid = np.zeros((grid_size, grid_size))
    for state in range(len(state_values)):
        row = state // grid_size
        col = state % grid_size
        value_grid[grid_size - 1 - row, col] = state_values[state]
    
    # Create meshgrid for surface plot
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, value_grid, cmap='plasma', 
                          alpha=0.8, linewidth=0, antialiased=True)
    
    # Add state labels
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            ax.text(j, i, value_grid[i, j], 
                   f'S{state}\n{value_grid[i, j]:.2f}', 
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_zlabel('State Value')
    ax.set_title('State Value Surface', fontsize=12, fontweight='bold')

def analyze_stochastic_policy(action_probs, state_values, transition_probs):
    """Analyze the stochastic policy characteristics"""
    
    print("\n" + "="*60)
    print("STOCHASTIC POLICY ANALYSIS")
    print("="*60)
    
    # Policy entropy analysis
    entropies = []
    for state in range(len(action_probs)):
        probs = action_probs[state]
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        entropies.append(entropy)
    
    print(f"Average policy entropy: {np.mean(entropies):.3f}")
    print(f"Maximum policy entropy: {np.max(entropies):.3f}")
    print(f"Minimum policy entropy: {np.min(entropies):.3f}")
    
    # Action probability statistics
    all_probs = np.concatenate([action_probs[state] for state in range(len(action_probs))])
    print(f"\nAction Probability Statistics:")
    print(f"Mean: {np.mean(all_probs):.3f}")
    print(f"Std: {np.std(all_probs):.3f}")
    print(f"Max: {np.max(all_probs):.3f}")
    print(f"Min: {np.min(all_probs):.3f}")
    
    # Most uncertain states
    print(f"\nMost uncertain states (highest entropy):")
    uncertain_states = np.argsort(entropies)[-5:][::-1]
    for state in uncertain_states:
        print(f"State S{state}: entropy = {entropies[state]:.3f}")
        print(f"  Action probs: {action_probs[state]}")

# Main execution
if __name__ == "__main__":
    print("Creating 3D State-Action Lattice for Stochastic FrozenLake...")
    
    # Train on stochastic environment
    model, env = train_ppo_frozenlake_stochastic(
        env_name="FrozenLake-v1",
        map_name="4x4",
        total_timesteps=100000
    )
    
    # Extract stochastic policy
    action_probs, state_values, q_values_approx = extract_stochastic_policy(model, env)
    
    # Collect stochastic transitions
    transition_probs, state_action_visits = collect_stochastic_transitions(
        env, action_probs, num_samples=2000
    )
    
    # Create 3D visualization
    print("Creating 3D state-action lattice visualization...")
    fig = plot_3d_policy_lattice(
        env, action_probs, state_values, transition_probs, state_action_visits
    )
    
    # Analyze the policy
    analyze_stochastic_policy(action_probs, state_values, transition_probs)
    
    env.close()
    
    print("\n3D Lattice Interpretation:")
    print("• X-Y plane: Grid positions (states)")
    print("• Z-axis: Actions (0=Left, 1=Down, 2=Right, 3=Up)")
    print("• Node color: Action probability (brighter = more probable)")
    print("• Node size: Action probability")
    print("• Colored paths: Sample trajectories through state-action space")
    print("• The policy forms a 'cloud' in state-action space due to stochasticity")
