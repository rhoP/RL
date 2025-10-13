import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from stable_baselines3 import PPO
import torch
from collections import defaultdict

def train_ppo_frozenlake(env_name="FrozenLake-v1", map_name="4x4", total_timesteps=30000):
    """Train a PPO agent on FrozenLake using Stable Baselines3"""
    
    env = gym.make(
        env_name,
        desc=None,
        map_name=map_name,
        is_slippery=False,  # Use deterministic for clearer transitions
        render_mode=None
    )
    
    print(f"Training PPO on {map_name} FrozenLake environment...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        policy_kwargs=dict(net_arch=[32, 32]),
        seed=42
    )
    
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model

def get_transition_graph(env, model):
    """Create a graph of state-action pairs and their transitions"""
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for all state-action pairs
    for state in range(state_size):
        for action in range(action_size):
            node_id = f"S{state}_A{action}"
            G.add_node(node_id, state=state, action=action)
    
    # Add edges based on transitions
    for state in range(state_size):
        for action in range(action_size):
            # Get the transition for this state-action pair
            env.unwrapped.s = state  # Set current state
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Get the optimal action from the policy for the next state
            if terminated:
                # Terminal state - self-loop or no action
                next_action = None
                next_node = f"S{next_state}_A0"  # Use action 0 as placeholder
            else:
                # Get the predicted action from policy
                next_action_probs = model.policy.get_distribution(
                    torch.tensor(np.array([next_state]))
                ).distribution.probs.detach().numpy()[0]
                next_action = np.argmax(next_action_probs)
                next_node = f"S{next_state}_A{next_action}"
            
            current_node = f"S{state}_A{action}"
            
            # Add edge with transition information
            G.add_edge(
                current_node, 
                next_node,
                reward=reward,
                terminated=terminated
            )
    
    return G

def get_state_action_values(model, env):
    """Get Q-values for all state-action pairs"""
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    q_values = {}
    
    for state in range(state_size):
        # Get state value
        state_val = model.policy.predict_values(
            torch.tensor(np.array([state]).reshape(1, -1))
        ).item()
        
        # Get action probabilities
        action_probs = model.policy.get_distribution(
            torch.tensor(np.array([state]))
        ).distribution.probs.detach().numpy()[0]
        
        for action in range(action_size):
            node_id = f"S{state}_A{action}"
            # Approximate Q-value using policy and state value
            q_values[node_id] = action_probs[action] * state_val * 10
    
    return q_values

def create_state_action_graph(env, model, map_name="4x4"):
    """Create the complete state-action transition graph visualization"""
    
    # Get the transition graph and Q-values
    G = get_transition_graph(env, model)
    q_values = get_state_action_values(model, env)
    
    # Set node attributes for visualization
    for node in G.nodes():
        G.nodes[node]['q_value'] = q_values.get(node, 0)
        state = int(node.split('_')[0][1:])
        action = int(node.split('_')[1][1:])
        G.nodes[node]['state'] = state
        G.nodes[node]['action'] = action
    
    # Create the visualization
    plt.figure(figsize=(20, 15))
    
    # Use spring layout for better node positioning
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Define node colors based on Q-values
    node_colors = [G.nodes[node]['q_value'] for node in G.nodes()]
    
    # Define node sizes based on importance (higher Q-value = larger node)
    node_sizes = [300 + G.nodes[node]['q_value'] * 500 for node in G.nodes()]
    
    # Define edge colors based on reward
    edge_colors = ['green' if G.edges[edge]['reward'] > 0 else 
                  'red' if G.edges[edge]['terminated'] else 
                  'blue' for edge in G.edges()]
    
    # Define edge widths based on transition probability (simplified)
    edge_widths = [2.0 for edge in G.edges()]
    
    # Draw the graph
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap='viridis',
        alpha=0.8,
        edgecolors='black',
        linewidths=1
    )
    
    # Draw edges
    edges = nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )
    
    # Draw labels
    labels = {}
    for node in G.nodes():
        state = G.nodes[node]['state']
        action = G.nodes[node]['action']
        action_names = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}
        labels[node] = f"S{state}\n{action_names[action]}\n{G.nodes[node]['q_value']:.2f}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Draw edge labels (rewards)
    edge_labels = {}
    for edge in G.edges():
        reward = G.edges[edge]['reward']
        if reward != 0:
            edge_labels[edge] = f"R:{reward}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    
    # Add colorbar for Q-values
    plt.colorbar(nodes, label='Q-Value', shrink=0.8)
    
    # Customize the plot
    plt.title(f'State-Action Transition Graph\nFrozenLake {map_name} - PPO Policy', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                  markersize=10, label='High Q-value'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                  markersize=10, label='Low Q-value'),
        plt.Line2D([0], [0], color='green', lw=2, label='Positive Reward'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Normal Transition'),
        plt.Line2D([0], [0], color='red', lw=2, label='Terminal/Hole')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G

def create_grid_layout_graph(env, model, map_name="4x4"):
    """Create a grid-based layout for better interpretability"""
    
    G = get_transition_graph(env, model)
    q_values = get_state_action_values(model, env)
    
    # Set node attributes
    for node in G.nodes():
        G.nodes[node]['q_value'] = q_values.get(node, 0)
        state = int(node.split('_')[0][1:])
        action = int(node.split('_')[1][1:])
        G.nodes[node]['state'] = state
        G.nodes[node]['action'] = action
    
    # Create grid layout based on state positions
    pos = {}
    grid_size = int(np.sqrt(env.observation_space.n))
    
    action_offsets = {
        0: (-0.2, 0),   # Left
        1: (0, -0.2),   # Down
        2: (0.2, 0),    # Right
        3: (0, 0.2)     # Up
    }
    
    for node in G.nodes():
        state = G.nodes[node]['state']
        action = G.nodes[node]['action']
        
        # Calculate grid position
        row = state // grid_size
        col = state % grid_size
        
        # Base position for the state
        base_x = col
        base_y = grid_size - 1 - row  # Flip y-axis for better visualization
        
        # Add offset based on action
        offset_x, offset_y = action_offsets[action]
        pos[node] = (base_x + offset_x, base_y + offset_y)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Node colors and sizes
    node_colors = [G.nodes[node]['q_value'] for node in G.nodes()]
    node_sizes = [200 + G.nodes[node]['q_value'] * 300 for node in G.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap='plasma',
        alpha=0.9,
        edgecolors='black',
        linewidths=1
    )
    
    # Draw edges
    edge_colors = ['green' if G.edges[edge]['reward'] > 0 else 
                  'red' if G.edges[edge]['terminated'] else 
                  'gray' for edge in G.edges()]
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=1.5,
        alpha=0.7,
        arrows=True,
        arrowsize=15,
        arrowstyle='->'
    )
    
    # Draw labels
    action_names = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    labels = {}
    for node in G.nodes():
        state = G.nodes[node]['state']
        action = G.nodes[node]['action']
        labels[node] = f"S{state}\n{action_names[action]}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold')
    
    # Add state grid background
    for state in range(env.observation_space.n):
        row = state // grid_size
        col = state % grid_size
        y = grid_size - 1 - row
        plt.gca().add_patch(plt.Rectangle((col-0.4, y-0.4), 0.8, 0.8, 
                                        fill=False, edgecolor='gray', linestyle='--', alpha=0.3))
        plt.text(col, y, f'S{state}', ha='center', va='center', alpha=0.5)
    
    plt.colorbar(nodes, label='Q-Value')
    plt.title(f'Grid Layout: State-Action Transition Graph\nFrozenLake {map_name} - PPO Policy', 
              fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    return G, pos

def analyze_transition_paths(G, start_state=0):
    """Analyze optimal paths through the state-action graph"""
    
    print("Analyzing optimal paths...")
    
    # Find all terminal nodes (states with termination)
    terminal_nodes = []
    for node in G.nodes():
        state = G.nodes[node]['state']
        # Check if this node leads to termination
        for edge in G.out_edges(node):
            if G.edges[edge]['terminated']:
                terminal_nodes.append(node)
    
    # Find paths to successful terminal states (reward > 0)
    successful_paths = []
    for terminal in terminal_nodes:
        # Check if this terminal gives positive reward
        for edge in G.in_edges(terminal):
            if G.edges[edge]['reward'] > 0:
                # Find path from start to this terminal
                start_node = f"S{start_state}_A{np.argmax([G.nodes[f'S{start_state}_A{a}']['q_value'] for a in range(4)])}"
                
                try:
                    path = nx.shortest_path(G, start_node, terminal)
                    successful_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
    
    print(f"Found {len(successful_paths)} successful path(s) from start state {start_state}")
    
    if successful_paths:
        print("\nShortest successful path:")
        shortest_path = min(successful_paths, key=len)
        for i, node in enumerate(shortest_path):
            state = G.nodes[node]['state']
            action = G.nodes[node]['action']
            action_names = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
            print(f"Step {i}: State S{state} -> Action {action_names[action]}")
    
    return successful_paths

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Creating State-Action Transition Graph...")
    
    # Create environment and train model
    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,  # Deterministic for clearer graph
        render_mode=None
    )
    env.reset()
    
    model = train_ppo_frozenlake(total_timesteps=30000)
    
    print("Creating spring layout graph...")
    G1 = create_state_action_graph(env, model, "4x4")
    
    print("Creating grid layout graph...")
    G2, grid_pos = create_grid_layout_graph(env, model, "4x4")
    
    # Analyze paths
    analyze_transition_paths(G1)
    
    env.close()
    
    print("\nGraph Interpretation:")
    print("• Each node represents a state-action pair (e.g., 'S0_A1' = State 0, Action 1)")
    print("• Node color intensity shows Q-value (brighter = higher value)")
    print("• Node size indicates importance in the policy")
    print("• Green edges: transitions with positive reward")
    print("• Red edges: transitions to terminal states (holes)")
    print("• Blue edges: normal transitions")
    print("• Arrows show the flow of state-action transitions under the optimal policy")
