import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
import torch
from collections import defaultdict
import networkx as nx


def extract_state_values(model, env):
    """Extract state values from the policy"""
    state_size = env.observation_space.n
    state_values = {}
    
    for state in range(state_size):
        obs = np.array([state])
        value = model.policy.predict_values(torch.tensor(obs.reshape(1, -1))).item()
        state_values[state] = value
    
    return state_values

def train_ppo_frozenlake_stochastic_simple(env_name="FrozenLake-v1", map_name="4x4", total_timesteps=50000):
    """Train a PPO agent on stochastic FrozenLake"""
    
    env = gym.make(
        env_name,
        desc=None,
        map_name=map_name,
        is_slippery=True,  # Stochastic environment
        render_mode=None
    )
    
    print(f"Training PPO on stochastic {map_name} FrozenLake...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=42
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    return model, env

def extract_policy_and_transitions(env, model, num_samples=1000):
    """Extract policy and transition probabilities"""
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # Get action probabilities for each state
    action_probs = {}
    state_values = {}  # ADD THIS
    for state in range(state_size):
        obs = np.array([state])
        probs = model.policy.get_distribution(torch.tensor(obs)).distribution.probs.detach().numpy()[0]
        action_probs[state] = probs
        # ADD THESE LINES:
        value = model.policy.predict_values(torch.tensor(obs.reshape(1, -1))).item()
        state_values[state] = value

    # Collect transition samples
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    print("Collecting transition samples...")
    for sample in range(num_samples):
        state, _ = env.reset()
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Sample action from policy
            probs = action_probs[state]
            action = np.random.choice(4, p=probs)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Record transition
            transition_counts[(state, action)][next_state] += 1
            
            state = next_state
    
    # Convert to probabilities
    transition_probs = {}
    for (state, action), next_counts in transition_counts.items():
        total = sum(next_counts.values())
        transition_probs[(state, action)] = {
            next_state: count / total for next_state, count in next_counts.items()
        }
    
    return action_probs, transition_probs, state_values

def create_clear_3d_lattice_plot(env, action_probs, transition_probs, state_values):  # ADD state_values parameter
    """Create a clear 3D lattice plot with transition arcs"""
    
    state_size = env.observation_space.n
    grid_size = int(np.sqrt(state_size))
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for actions (we'll use value-based coloring instead)
    action_colors = {
        0: 'red',    # Left
        1: 'blue',   # Down
        2: 'green',  # Right
        3: 'orange'  # Up
    }
    
    action_symbols = {
        0: '←', 1: '↓', 2: '→', 3: '↑'
    }
    
    action_names = {
        0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'
    }
    
    # Create positions for state-action pairs
    positions = {}
    
    # Normalize state values for colormap
    all_values = list(state_values.values())
    vmin, vmax = min(all_values), max(all_values)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.viridis  # Choose your preferred colormap
    
    for state in range(state_size):
        # Grid position
        row = state // grid_size
        col = state % grid_size
        
        base_x = col
        base_y = grid_size - 1 - row  # Flip y for better visualization
        
        # Create positions for each action at this state
        for action in range(4):
            node_id = f"S{state}_A{action}"
            
            # Position: (grid_x, grid_y, action_level)
            positions[node_id] = (base_x, base_y, action)
    
    # Plot nodes (state-action pairs) - MODIFIED COLORING
    for state in range(state_size):
        for action in range(4):
            node_id = f"S{state}_A{action}"
            x, y, z = positions[node_id]
            prob = action_probs[state][action]
            value = state_values[state]  # GET STATE VALUE
            
            # Node size based on action probability
            size = 50 + prob * 200
            # Color based on state value using colormap
            color = cmap(norm(value))
            alpha = 0.3 + prob * 0.7
            
            ax.scatter(x, y, z, s=size, c=[color], alpha=alpha, edgecolors='black', linewidths=1)
            
            # Label significant nodes
            if prob > 0.1:
                ax.text(x, y, z, f"S{state}\n{action_symbols[action]}\n{prob:.2f}", 
                       fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Plot transitions as arcs - MODIFIED COLORING
    print("Plotting transition arcs...")
    
    for (state, action), next_probs in transition_probs.items():
        from_node = f"S{state}_A{action}"
        from_x, from_y, from_z = positions[from_node]
        
        for next_state, trans_prob in next_probs.items():
            if trans_prob > 0.1:  # Only plot significant transitions
                # Find the most probable action for the next state
                next_action_probs = action_probs[next_state]
                next_action = np.argmax(next_action_probs)
                to_node = f"S{next_state}_A{next_action}"
                
                to_x, to_y, to_z = positions[to_node]
                
                # Color based on STATE VALUE of source state
                color = cmap(norm(state_values[state]))
                
                # Plot arc with arrow
                ax.plot([from_x, to_x], [from_y, to_y], [from_z, to_z], 
                       color=color, alpha=trans_prob, linewidth=1 + trans_prob * 3)
                
                # Add arrow head
                mid_x, mid_y, mid_z = (from_x + to_x)/2, (from_y + to_y)/2, (from_z + to_z)/2
                ax.quiver(mid_x, mid_y, mid_z, 
                         (to_x - from_x)*0.3, (to_y - from_y)*0.3, (to_z - from_z)*0.3,
                         color=color, alpha=trans_prob, 
                         arrow_length_ratio=0.2, linewidth=1)
                
                # Add transition probability label for significant transitions
                if trans_prob > 0.3:
                    label_x, label_y, label_z = (from_x + to_x)/2, (from_y + to_y)/2, (from_z + to_z)/2 + 0.1
                    ax.text(label_x, label_y, label_z, f'{trans_prob:.2f}', 
                           fontsize=7, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
    
    # ADD COLORBAR
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('State Value', fontsize=12, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Grid X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Grid Y Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Action', fontsize=12, fontweight='bold')
    
    # Set axis limits and ticks
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_zlim(-0.5, 3.5)
    
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_zticks(range(4))
    ax.set_zticklabels([f'{action_names[i]} ({action_symbols[i]})' for i in range(4)])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # UPDATE LEGEND to reflect value-based coloring
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=cmap(0.8), markersize=10, 
                  label='High Value State'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=cmap(0.2), markersize=10, 
                  label='Low Value State')
    ]
    
    # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # ax.set_title('3D State-Action Lattice with Value-Based Coloring\nFrozenLake Stochastic Policy', 
    #            fontsize=16, fontweight='bold', pad=20)
    
    # Set viewing angle for better visibility
    ax.view_init(elev=20, azim=45)
    
    return fig, ax

def create_2d_projection_plots(env, action_probs, transition_probs):
    """Create 2D projection plots for additional clarity"""
    
    state_size = env.observation_space.n
    grid_size = int(np.sqrt(state_size))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    action_names = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
    action_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
    
    # Plot 1: Action Probability Heatmap
    for action in range(4):
        ax = axes[action]
        
        # Create probability grid for this action
        prob_grid = np.zeros((grid_size, grid_size))
        for state in range(state_size):
            row = state // grid_size
            col = state % grid_size
            prob_grid[grid_size-1-row, col] = action_probs[state][action]
        
        im = ax.imshow(prob_grid, cmap='viridis', vmin=0, vmax=1)
        
        # Annotate with probabilities
        for i in range(grid_size):
            for j in range(grid_size):
                state = i * grid_size + j
                prob = action_probs[state][action]
                color = 'white' if prob > 0.5 else 'black'
                ax.text(j, i, f'{prob:.2f}', ha='center', va='center', 
                       color=color, fontweight='bold', fontsize=8)
        
        ax.set_title(f'{action_names[action]} Action Probabilities', fontsize=12, fontweight='bold')
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_xticklabels([f'Col{j}' for j in range(grid_size)])
        ax.set_yticklabels([f'Row{grid_size-1-i}' for i in range(grid_size)])
        plt.colorbar(im, ax=ax, label='Probability')
    
    plt.tight_layout()
    plt.show()
    
    # Plot transition network
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create a simplified transition graph
    G = nx.DiGraph()
    pos = {}
    
    for state in range(state_size):
        row = state // grid_size
        col = state % grid_size
        pos[state] = (col, grid_size-1-row)
        G.add_node(state, pos=pos[state])
    
    # Add edges for significant transitions
    edge_weights = {}
    for (state, action), next_probs in transition_probs.items():
        if action_probs[state][action] > 0.3:  # Only for likely actions
            for next_state, prob in next_probs.items():
                if prob > 0.2:
                    if (state, next_state) not in edge_weights:
                        edge_weights[(state, next_state)] = []
                    edge_weights[(state, next_state)].append(prob)
    
    # Average probabilities for multiple actions between same states
    for (state, next_state), probs in edge_weights.items():
        avg_prob = np.mean(probs)
        G.add_edge(state, next_state, weight=avg_prob, probability=avg_prob)
    
    # Draw the graph
    node_colors = ['lightblue' for _ in G.nodes()]
    node_sizes = [800 for _ in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, edgecolors='black')
    
    # Draw edges with weights
    edges = nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='red', 
                                  width=[G.edges[edge]['weight'] * 3 for edge in G.edges()],
                                  alpha=0.7, arrows=True, arrowsize=20, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10, font_weight='bold')
    
    # Draw edge labels
    edge_labels = {(u, v): f'{G.edges[(u, v)]["probability"]:.2f}' 
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, ax=ax2, edge_labels=edge_labels, font_size=8)
    
    ax2.set_title('Simplified State Transition Network\n(Edge weights = transition probabilities)', 
                 fontsize=14, fontweight='bold')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_policy_analysis(action_probs, transition_probs):
    """Print analysis of the policy"""
    
    print("\n" + "="*60)
    print("POLICY ANALYSIS")
    print("="*60)
    
    action_names = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
    
    # Calculate policy entropy
    entropies = []
    for state in range(len(action_probs)):
        probs = action_probs[state]
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        entropies.append(entropy)
    
    print(f"Average Policy Entropy: {np.mean(entropies):.3f}")
    print(f"Most certain state: S{np.argmin(entropies)} (entropy: {np.min(entropies):.3f})")
    print(f"Most uncertain state: S{np.argmax(entropies)} (entropy: {np.max(entropies):.3f})")
    
    print("\nKey State-Action Decisions:")
    for state in range(len(action_probs)):
        best_action = np.argmax(action_probs[state])
        best_prob = action_probs[state][best_action]
        if best_prob > 0.6:  # Show only confident decisions
            print(f"  S{state}: {action_names[best_action]} (p={best_prob:.3f})")

# Main execution
if __name__ == "__main__":
    
    # Train policy
    model, env = train_ppo_frozenlake_stochastic_simple(
        env_name="FrozenLake-v1",
        map_name="4x4", 
        total_timesteps=50000
    )
    
    # Extract policy and transitions
    action_probs, transition_probs, state_values = extract_policy_and_transitions(env, model, num_samples=1000)
    
    # Create 3D lattice plot
    print("Creating 3D lattice visualization...")
    fig, ax = create_clear_3d_lattice_plot(env, action_probs, transition_probs, state_values)
    plt.tight_layout()
    plt.show()
    
    # Create 2D projection plots
    print("Creating 2D projection plots...")
    create_2d_projection_plots(env, action_probs, transition_probs)
    
    # Print analysis
    print_policy_analysis(action_probs, transition_probs)
    
    env.close()
    
    
