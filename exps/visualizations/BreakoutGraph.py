import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space
import torch
import torch.nn as nn
from collections import defaultdict
import cv2

def preprocess_breakout_observation(obs):
    """Preprocess Breakout observation to reduce state space"""
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # Convert to grayscale and resize
    if len(obs.shape) == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    
    # Normalize
    obs = obs.astype(np.float32) / 255.0
    
    return obs

def discretize_state(obs, grid_size=8):
    """Discretize the continuous state space for graph representation"""
    if len(obs.shape) == 2:  # Already 2D
        processed = obs
    else:
        processed = preprocess_breakout_observation(obs)
    
    # Further reduce resolution for state discretization
    state_repr = cv2.resize(processed, (grid_size, grid_size))
    
    # Create a hashable state representation
    state_id = hash(state_repr.tobytes())
    
    return state_id, state_repr

def train_ppo_breakout(total_timesteps=100000):
    """Train a PPO agent on Breakout"""
    
    env = make_vec_env(
        "BreakoutNoFrameskip-v4",
        n_envs=1,
        env_kwargs={
            'frameskip': 4,
            'repeat_action_probability': 0.0
        }
    )
    
    print("Training PPO on Breakout...")
    
    # Use CNN policy for pixel observations
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        policy_kwargs=dict(
            net_arch=[],
            activation_fn=nn.ReLU,
            normalize_images=False
        ),
        verbose=1
    )
    
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model

class BreakoutStateTracker:
    """Track states and transitions in Breakout"""
    
    def __init__(self, grid_size=8):
        self.grid_size = grid_size
        self.state_mapping = {}  # state_id -> state_index
        self.state_features = {}  # state_id -> feature_vector
        self.transitions = defaultdict(list)
        self.state_counter = 0
    
    def get_state_id(self, obs):
        """Get or create state ID for observation"""
        state_id, state_repr = discretize_state(obs, self.grid_size)
        
        if state_id not in self.state_mapping:
            self.state_mapping[state_id] = self.state_counter
            self.state_features[state_id] = state_repr.flatten()
            self.state_counter += 1
        
        return state_id
    
    def record_transition(self, state_id, action, next_state_id, reward, done):
        """Record a state-action transition"""
        self.transitions[(state_id, action)].append({
            'next_state': next_state_id,
            'reward': reward,
            'done': done
        })

def create_breakout_transition_graph(model, num_episodes=10, max_steps=1000):
    """Create a transition graph for Breakout by sampling episodes"""
    
    env = gym.make("BreakoutNoFrameskip-v4", frameskip=4)
    tracker = BreakoutStateTracker(grid_size=6)  # Smaller grid for tractability
    
    print("Sampling episodes to build transition graph...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        current_state_id = tracker.get_state_id(obs)
        episode_steps = 0
        
        while episode_steps < max_steps:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=False)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state_id = tracker.get_state_id(next_obs)
            
            # Record transition
            tracker.record_transition(
                current_state_id, action, next_state_id, reward, done
            )
            
            current_state_id = next_state_id
            episode_steps += 1
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Recorded {episode_steps} steps, {len(tracker.state_mapping)} unique states")
    
    env.close()
    return tracker

def build_graph_from_tracker(tracker, model, top_k_states=50):
    """Build a NetworkX graph from the state tracker"""
    
    G = nx.DiGraph()
    
    # Get the most frequently visited states
    state_visit_count = defaultdict(int)
    for (state_id, action), transitions in tracker.transitions.items():
        state_visit_count[state_id] += len(transitions)
    
    # Select top K states for visualization
    top_states = sorted(state_visit_count.items(), key=lambda x: x[1], reverse=True)[:top_k_states]
    top_state_ids = [state_id for state_id, count in top_states]
    
    print(f"Building graph with {len(top_state_ids)} states...")
    
    # Add nodes for top states and their actions
    action_names = {0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT'}
    
    for state_id in top_state_ids:
        state_idx = tracker.state_mapping[state_id]
        
        # Add nodes for each possible action in this state
        for action in range(4):  # Breakout has 4 actions
            node_id = f"S{state_idx}_A{action}"
            
            # Estimate Q-value (simplified)
            # In practice, you'd need to compute this properly
            q_value = np.random.random()  # Placeholder
            
            G.add_node(
                node_id,
                state_id=state_id,
                state_idx=state_idx,
                action=action,
                action_name=action_names[action],
                q_value=q_value,
                visit_count=state_visit_count[state_id]
            )
    
    # Add edges based on recorded transitions
    edge_count = 0
    for (state_id, action), transitions in tracker.transitions.items():
        if state_id not in top_state_ids:
            continue
            
        state_idx = tracker.state_mapping[state_id]
        from_node = f"S{state_idx}_A{action}"
        
        for transition in transitions:
            next_state_id = transition['next_state']
            
            if next_state_id in tracker.state_mapping and next_state_id in top_state_ids:
                next_state_idx = tracker.state_mapping[next_state_id]
                
                # Find the most likely next action from policy
                # This is simplified - in practice, you'd sample from the policy
                next_action = np.random.randint(0, 4)  # Placeholder
                to_node = f"S{next_state_idx}_A{next_action}"
                
                G.add_edge(
                    from_node,
                    to_node,
                    reward=transition['reward'],
                    done=transition['done']
                )
                edge_count += 1
    
    print(f"Graph built with {len(G.nodes())} nodes and {edge_count} edges")
    return G

def visualize_breakout_graph(G, layout_type='spring'):
    """Visualize the Breakout state-action graph"""
    
    plt.figure(figsize=(20, 15))
    
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G, seed=42)
    
    # Node colors and sizes based on Q-values and visit counts
    node_colors = [G.nodes[node].get('q_value', 0.5) for node in G.nodes()]
    node_sizes = [300 + G.nodes[node].get('visit_count', 0) for node in G.nodes()]
    
    # Edge colors based on reward
    edge_colors = []
    for edge in G.edges():
        reward = G.edges[edge].get('reward', 0)
        if reward > 0:
            edge_colors.append('green')
        elif G.edges[edge].get('done', False):
            edge_colors.append('red')
        else:
            edge_colors.append('blue')
    
    # Draw nodes
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
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=1.5,
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        arrowstyle='->'
    )
    
    # Draw labels
    labels = {}
    for node in G.nodes():
        state_idx = G.nodes[node]['state_idx']
        action_name = G.nodes[node]['action_name']
        q_value = G.nodes[node].get('q_value', 0)
        labels[node] = f"S{state_idx}\n{action_name}\n{q_value:.2f}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight='bold')
    
    plt.colorbar(nodes, label='Q-Value')
    plt.title('Breakout State-Action Transition Graph\n(Sampled Top States)', 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_simplified_breakout_analysis(model, num_episodes=5):
    """Create a simplified analysis focusing on key game states"""
    
    env = gym.make("BreakoutNoFrameskip-v4", frameskip=4, render_mode='rgb_array')
    
    # Track key metrics
    ball_positions = []
    paddle_positions = []
    brick_states = []
    actions_taken = []
    rewards_earned = []
    
    print("Analyzing Breakout gameplay...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step = 0
        
        while step < 1000:  # Limit steps per episode
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Store frame and action for visualization
            if step % 10 == 0:  # Sample every 10th step
                actions_taken.append(action)
                rewards_earned.append(episode_reward)
            
            # Take action
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            step += 1
            if done:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {step}")
    
    env.close()
    
    # Create simplified visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    action_counts = [list(actions_taken).count(i) for i in range(4)]
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    plt.bar(action_names, action_counts, color=['blue', 'red', 'green', 'orange'])
    plt.title('Action Distribution')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.plot(rewards_earned)
    plt.title('Cumulative Reward Over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 3)
    # Show sample gameplay frames (conceptual)
    plt.text(0.5, 0.5, 'Breakout Gameplay Analysis\n\n• Ball tracking\n• Paddle movement\n• Brick patterns\n• Reward timing', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.axis('off')
    
    plt.suptitle('Breakout PPO Policy Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Alternative: Use a simplified Breakout environment
def create_simplified_breakout_graph():
    """Create a conceptual graph for Breakout using known game mechanics"""
    
    G = nx.DiGraph()
    
    # Define key game states conceptually
    states = {
        'START': 'Game start',
        'BALL_LEFT': 'Ball moving left',
        'BALL_RIGHT': 'Ball moving right', 
        'BRICK_HIT': 'Brick being hit',
        'PADDLE_LEFT': 'Paddle on left',
        'PADDLE_RIGHT': 'Paddle on right',
        'PADDLE_CENTER': 'Paddle centered',
        'WALL_BOUNCE': 'Ball bouncing off wall',
        'LIFE_LOST': 'Ball missed',
        'LEVEL_CLEAR': 'All bricks cleared'
    }
    
    actions = {
        'NOOP': 'No operation',
        'LEFT': 'Move paddle left', 
        'RIGHT': 'Move paddle right',
        'FIRE': 'Start game/launch ball'
    }
    
    # Add state-action nodes
    for state, state_desc in states.items():
        for action, action_desc in actions.items():
            node_id = f"{state}_{action}"
            G.add_node(node_id, state=state, action=action, 
                      description=f"{state_desc} + {action_desc}")
    
    # Add transitions based on game logic
    transitions = [
        ('START_FIRE', 'BALL_RIGHT_NOOP'),
        ('BALL_LEFT_LEFT', 'BALL_LEFT_NOOP'),
        ('BALL_LEFT_RIGHT', 'BALL_RIGHT_NOOP'), 
        ('BALL_RIGHT_LEFT', 'BALL_LEFT_NOOP'),
        ('BALL_RIGHT_RIGHT', 'BALL_RIGHT_NOOP'),
        ('BALL_LEFT_NOOP', 'WALL_BOUNCE_NOOP'),
        ('BALL_RIGHT_NOOP', 'BRICK_HIT_NOOP'),
        ('BRICK_HIT_NOOP', 'BALL_LEFT_NOOP'),
        ('BRICK_HIT_NOOP', 'BALL_RIGHT_NOOP'),
        ('BALL_LEFT_NOOP', 'LIFE_LOST_NOOP'),  # Missed ball
        ('BRICK_HIT_NOOP', 'LEVEL_CLEAR_NOOP')  # All bricks cleared
    ]
    
    for from_node, to_node in transitions:
        if from_node in G.nodes() and to_node in G.nodes():
            G.add_edge(from_node, to_node, reward=1.0 if 'BRICK_HIT' in to_node else 0.0)
    
    # Visualize conceptual graph
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title('Conceptual Breakout State-Action Graph\n(Game Mechanics)', 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G

# Main execution
if __name__ == "__main__":
    print("Breakout State-Action Graph Analysis")
    print("Note: Breakout is more complex than FrozenLake")
    print("We'll use sampling and simplification techniques...")
    
    # Option 1: Train and analyze (computationally intensive)
    # model = train_ppo_breakout(total_timesteps=50000)
    # tracker = create_breakout_transition_graph(model, num_episodes=3)
    # G = build_graph_from_tracker(tracker, model, top_k_states=30)
    # visualize_breakout_graph(G)
    
    # Option 2: Simplified analysis
    # create_simplified_breakout_analysis(model, num_episodes=3)
    
    # Option 3: Conceptual graph (immediate visualization)
    print("Creating conceptual Breakout state-action graph...")
    G_conceptual = create_simplified_breakout_graph()
    
    print("\nBreakout Analysis Challenges:")
    print("1. High-dimensional state space (pixels)")
    print("2. Continuous state representations") 
    print("3. Large number of possible states")
    print("4. Complex transition dynamics")
    print("\nSolutions implemented:")
    print("• State discretization and hashing")
    print("• Sampling-based state space exploration")
    print("• Focus on frequently visited states")
    print("• Conceptual graph for game mechanics")
