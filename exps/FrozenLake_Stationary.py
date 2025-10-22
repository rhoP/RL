import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from stable_baselines3 import PPO
import torch
from collections import defaultdict

def train_ppo_frozenlake_deterministic(env_name="FrozenLake-v1", map_name="4x4", total_timesteps=50000):
    """Train a PPO agent on deterministic FrozenLake"""
    
    env = gym.make(
        env_name,
        desc=None,
        map_name=map_name,
        is_slippery=True,  # Deterministic environment
        render_mode=None
    )
    
    print(f"Training PPO on deterministic {map_name} FrozenLake...")
    
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
        seed=42,
        device='cpu'
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    return model, env

def evaluate_policy_deterministic(model, env, n_episodes=100):
    """Evaluate the policy deterministically"""
    successes = 0
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
        
        total_rewards.append(total_reward)
        if total_reward > 0:
            successes += 1
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = successes / n_episodes
    
    print(f"Success rate: {success_rate:.2f}")
    return mean_reward, std_reward

def extract_optimal_policy(model, env):
    """Extract the optimal deterministic policy and state values"""
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    optimal_actions = {}
    state_values = {}
    action_probs = {}
    
    for state in range(state_size):
        # Get action probabilities
        obs = np.array([state])
        probs = model.policy.get_distribution(torch.tensor(obs)).distribution.probs.detach().numpy()[0]
        action_probs[state] = probs
        
        # Get optimal action (deterministic)
        optimal_action = np.argmax(probs)
        optimal_actions[state] = optimal_action
        
        # Get state value
        state_values[state] = model.policy.predict_values(torch.tensor(obs.reshape(1, -1))).item()
    
    return optimal_actions, state_values, action_probs

def get_deterministic_transitions(env, optimal_actions):
    """Get deterministic transitions for the optimal policy"""
    
    transition_probs = {}
    state_size = env.observation_space.n
    
    # For deterministic environments, we can compute transitions directly
    for state in range(state_size):
        optimal_action = optimal_actions[state]
        
        # Use the environment's internal dynamics to get the next state
        # For FrozenLake, we need to simulate the step
        env.unwrapped.s = state  # Set the current state
        next_state, reward, terminated, truncated, _ = env.step(optimal_action)
        
        # In deterministic FrozenLake, each state-action leads to exactly one next state
        transition_probs[(state, optimal_action)] = {
            next_state: 1.0  # 100% probability to this next state
        }
    
    return transition_probs

def collect_transition_samples(env, optimal_actions, num_samples=1000):
    """Collect transition samples by actually running the environment"""
    
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    print("Collecting transition samples...")
    
    for sample in range(num_samples):
        state, _ = env.reset()
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Follow optimal policy
            action = optimal_actions[state]
            
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
    
    return transition_probs

def create_deterministic_policy_graph(env, optimal_actions, state_values, transition_probs, map_name="4x4"):
    """Create a graph where nodes are state-optimal-action pairs"""
    
    G = nx.DiGraph()
    state_size = env.observation_space.n
    grid_size = int(np.sqrt(state_size))
    
    # Action symbols for visualization
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    action_names = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
    
    # Add nodes for each state with its optimal action
    for state in range(state_size):
        optimal_action = optimal_actions[state]
        node_id = f"S{state}"
        
        G.add_node(
            node_id,
            state=state,
            optimal_action=optimal_action,
            action_symbol=action_symbols[optimal_action],
            action_name=action_names[optimal_action],
            state_value=state_values[state],
            row=state // grid_size,
            col=state % grid_size
        )
    
    # Add edges based on transition probabilities
    for state in range(state_size):
        optimal_action = optimal_actions[state]
        from_node = f"S{state}"
        
        if (state, optimal_action) in transition_probs:
            for next_state, prob in transition_probs[(state, optimal_action)].items():
                to_node = f"S{next_state}"
                
                if prob > 0.01:  # Only show transitions with significant probability
                    G.add_edge(
                        from_node,
                        to_node,
                        probability=prob,
                        label=f"{prob:.2f}"
                    )
    
    return G

def plot_deterministic_policy_grid(G, env, map_name="4x4"):
    """Plot the deterministic policy on the FrozenLake grid"""
    
    state_size = env.observation_space.n
    grid_size = int(np.sqrt(state_size))
    
    # Create grid layout
    pos = {}
    for node in G.nodes():
        state = G.nodes[node]['state']
        row = state // grid_size
        col = state % grid_size
        # Flip y-axis for better visualization
        pos[node] = (col, grid_size - 1 - row)
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Policy graph with transitions
    node_colors = [G.nodes[node]['state_value'] for node in G.nodes()]
    node_sizes = [800 + G.nodes[node]['state_value'] * 500 for node in G.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax1,
        node_color=node_colors,
        node_size=node_sizes,
        cmap='viridis',
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw edges with weights
    edge_colors = []
    edge_widths = []
    for edge in G.edges():
        prob = G.edges[edge]['probability']
        edge_widths.append(2 + prob * 5)
        
        # Color based on value improvement
        from_value = G.nodes[edge[0]]['state_value']
        to_value = G.nodes[edge[1]]['state_value']
        if to_value > from_value:
            edge_colors.append('red')
        else:
            edge_colors.append('blue')
    
    nx.draw_networkx_edges(
        G, pos, ax=ax1,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.7,
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )
    
    # Draw node labels (state + optimal action)
    labels = {node: f"{node}\n{G.nodes[node]['action_symbol']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=10, font_weight='bold')
    
    # Draw edge labels (transition probabilities)
    edge_labels = {(u, v): f"{G.edges[(u, v)]['probability']:.2f}" 
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=8)
    
    ax1.set_title(f'Deterministic Policy Graph\nFrozenLake {map_name}', 
                  fontsize=14, fontweight='bold')
    ax1.axis('equal')
    
    # Add grid background
    for i in range(grid_size + 1):
        ax1.axhline(y=i - 0.5, color='gray', linestyle='--', alpha=0.3)
        ax1.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Plot 2: Policy value heatmap
    value_grid = np.zeros((grid_size, grid_size))
    action_grid = np.empty((grid_size, grid_size), dtype=object)
    
    for node in G.nodes():
        state = G.nodes[node]['state']
        row = state // grid_size
        col = state % grid_size
        value_grid[grid_size - 1 - row, col] = G.nodes[node]['state_value']
        action_grid[grid_size - 1 - row, col] = G.nodes[node]['action_symbol']
    
    im = ax2.imshow(value_grid, cmap='viridis', alpha=0.8)
    
    # Annotate with state numbers and actions
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            text = f"S{state}\n{action_grid[i, j]}"
            ax2.text(j, i, text, ha='center', va='center', 
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax2.set_title('Policy Value Heatmap with Optimal Actions', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='State Value')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_policy_path(G, start_state=0, goal_state=15):
    """Analyze the path taken by the deterministic policy"""
    
    print("\nAnalyzing policy path from start to goal...")
    
    current_node = f"S{start_state}"
    path = [current_node]
    states_visited = set([start_state])
    
    max_steps = 20  # Prevent infinite loops
    step = 0
    
    while step < max_steps:
        # Find outgoing edges from current node
        outgoing_edges = list(G.out_edges(current_node))
        
        if not outgoing_edges:
            print(f"  No outgoing edges from {current_node}")
            break
        
        # Follow the highest probability transition
        next_node = None
        max_prob = 0
        
        for u, v in outgoing_edges:
            prob = G.edges[(u, v)]['probability']
            next_state = int(v[1:])  # Extract state number from "SX"
            
            if prob > max_prob and next_state not in states_visited:
                max_prob = prob
                next_node = v
        
        if next_node is None:
            # All next states already visited, take any
            for u, v in outgoing_edges:
                prob = G.edges[(u, v)]['probability']
                if prob > max_prob:
                    max_prob = prob
                    next_node = v
        
        if next_node is None:
            break
        
        path.append(next_node)
        current_state = int(next_node[1:])
        states_visited.add(current_state)
        current_node = next_node
        
        step += 1
        
        if current_state == goal_state:
            print(f"  ✓ Reached goal state S{goal_state}!")
            break
    
    print("Policy path:", " → ".join(path))
    return path

def print_policy_summary(optimal_actions, state_values, env):
    """Print a summary of the deterministic policy"""
    
    state_size = env.observation_space.n
    grid_size = int(np.sqrt(state_size))
    
    print(f"\nDeterministic Policy Summary ({grid_size}x{grid_size} Grid):")
    print("=" * 50)
    
    action_names = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
    
    # Create policy grid
    policy_grid = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            state = i * grid_size + j
            action = optimal_actions[state]
            row.append(action_names[action])
        policy_grid.append(row)
    
    print("\nOptimal Actions:")
    for i, row in enumerate(policy_grid):
        print(f"Row {i}: {row}")
    
    print(f"\nState Values:")
    for state in range(state_size):
        print(f"S{state}: {state_values[state]:.3f} → {action_names[optimal_actions[state]]}")

# Main execution
if __name__ == "__main__":
    print("Creating Deterministic Policy Graph for FrozenLake...")
    
    # Train model on deterministic environment
    model, env = train_ppo_frozenlake_deterministic(
        env_name="FrozenLake-v1",
        map_name="4x4", 
        total_timesteps=50000
    )
    
    # Evaluate policy first
    print("Evaluating policy...")
    mean_reward, std_reward = evaluate_policy_deterministic(model, env)
    
    # Extract optimal policy
    optimal_actions, state_values, action_probs = extract_optimal_policy(model, env)
    
    # Collect transition samples using the correct method
    transition_probs = collect_transition_samples(
        env, optimal_actions, num_samples=500
    )
    
    # Create the deterministic policy graph
    G = create_deterministic_policy_graph(
        env, optimal_actions, state_values, transition_probs, "4x4"
    )
    
    # Visualize
    print("Creating policy visualization...")
    plot_deterministic_policy_grid(G, env, "4x4")
    
    # Analyze policy path
    path = analyze_policy_path(G, start_state=0, goal_state=15)
    
    # Print summary
    print_policy_summary(optimal_actions, state_values, env)
    
    env.close()
    
    print("\nGraph Interpretation:")
    print("• Each node: State (S0-S15) + Optimal Action (arrow)")
    print("• Node color: State value (brighter = higher value)") 
    print("• Node size: Importance in policy")
    print("• Edges: Transitions under optimal policy")
    print("• Edge labels: Transition probabilities")
    print("• Red edges: Moving to higher-value states")
    print("• Blue edges: Moving to lower-value states")