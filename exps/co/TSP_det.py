import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import defaultdict

class TSPEnvironment:
    """A simple TSP environment for combinatorial optimization"""
    
    def __init__(self, num_cities=8, seed=42):
        self.num_cities = num_cities
        self.seed = seed
        self.reset()
        
    def reset(self):
        np.random.seed(self.seed)
        # Generate random city coordinates
        self.cities = np.random.rand(self.num_cities, 2) * 10
        self.current_city = 0  # Start from city 0
        self.visited = set([0])
        self.route = [0]
        self.distance_traveled = 0.0
        self.steps = 0
        self.done = False
        
        # Precompute distances
        self.distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.distances[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        
        return self.get_state()
    
    def get_state(self):
        """Return current state representation"""
        return {
            'current_city': self.current_city,
            'visited': self.visited.copy(),
            'cities': self.cities.copy(),
            'route': self.route.copy()
        }
    
    def step(self, action):
        """Take a step in the environment"""
        if self.done:
            return self.get_state(), 0.0, True, {}
        
        next_city = action
        reward = -self.distances[self.current_city][next_city]  # Negative distance as reward
        
        self.current_city = next_city
        self.visited.add(next_city)
        self.route.append(next_city)
        self.distance_traveled += self.distances[self.route[-2]][self.route[-1]]
        self.steps += 1
        
        # Check if all cities visited
        if len(self.visited) == self.num_cities:
            # Return to start city
            return_reward = -self.distances[self.current_city][0]
            reward += return_reward
            self.route.append(0)
            self.distance_traveled += self.distances[self.current_city][0]
            self.done = True
        
        return self.get_state(), reward, self.done, {}

class TSPPolicyNetwork(nn.Module):
    """Policy network for TSP with proper tensor handling"""
    
    def __init__(self, input_size, hidden_size, num_cities):
        super(TSPPolicyNetwork, self).__init__()
        self.num_cities = num_cities
        
        # City encoder
        self.city_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Current city context
        self.context_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)  # Single output per city
        
    def forward(self, city_features, current_city_idx, visited_mask):
        batch_size = city_features.size(0)
        
        # Encode all cities
        city_embeddings = self.city_encoder(city_features)  # [batch_size, num_cities, hidden_size]
        
        # Get current city embedding
        current_city_emb = city_embeddings[torch.arange(batch_size), current_city_idx]  # [batch_size, hidden_size]
        
        # Repeat current city embedding for all cities
        current_city_expanded = current_city_emb.unsqueeze(1).repeat(1, self.num_cities, 1)  # [batch_size, num_cities, hidden_size]
        
        # Combine city embeddings with current city context
        combined = torch.cat([city_embeddings, current_city_expanded], dim=-1)  # [batch_size, num_cities, hidden_size*2]
        context_aware = self.context_net(combined)  # [batch_size, num_cities, hidden_size]
        
        # Compute logits
        logits = self.output_layer(context_aware).squeeze(-1)  # [batch_size, num_cities]
        
        # Apply visited mask
        logits = logits.masked_fill(visited_mask, -1e9)
        
        return torch.softmax(logits, dim=-1), logits

def train_tsp_policy(env, num_episodes=3000, lr=1e-3):
    """Train a policy for TSP using REINFORCE with proper batching"""
    
    num_cities = env.num_cities
    policy_net = TSPPolicyNetwork(input_size=2, hidden_size=128, num_cities=num_cities)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    episode_rewards = []
    episode_distances = []
    
    print("Training TSP Policy...")
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        # Convert cities to tensor
        cities_tensor = torch.FloatTensor(state['cities']).unsqueeze(0)  # [1, num_cities, 2]
        current_city = torch.LongTensor([state['current_city']])
        visited_mask = torch.BoolTensor([[i in state['visited'] for i in range(num_cities)]])
        
        episode_reward = 0
        steps = 0
        
        while True:
            # Get action probabilities
            probs, logits = policy_net(cities_tensor, current_city, visited_mask)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Take action
            next_state, reward, done, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            
            # Update state for next step
            current_city = torch.LongTensor([next_state['current_city']])
            visited_mask = torch.BoolTensor([[i in next_state['visited'] for i in range(num_cities)]])
            
            steps += 1
            
            if done:
                break
        
        # Compute returns and update policy
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        episode_rewards.append(episode_reward)
        episode_distances.append(-episode_reward)  # Convert back to distance
        
        if (episode + 1) % 500 == 0:
            avg_distance = np.mean(episode_distances[-100:])
            print(f"Episode {episode + 1}, Average Distance (last 100): {avg_distance:.3f}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_distances)
    plt.title('Episode Distances')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return policy_net, episode_rewards, episode_distances

def extract_deterministic_tsp_policy(policy_net, env):
    """Extract deterministic policy by always choosing highest probability action"""
    
    num_cities = env.num_cities
    state = env.reset()
    
    optimal_route = [0]
    current_city = 0
    visited = set([0])
    route_decisions = {}
    state_values = {}
    
    cities_tensor = torch.FloatTensor(state['cities']).unsqueeze(0)
    
    step = 0
    max_steps = num_cities * 2  # Prevent infinite loops
    
    while len(visited) < num_cities and step < max_steps:
        current_city_tensor = torch.LongTensor([current_city])
        visited_mask = torch.BoolTensor([[i in visited for i in range(num_cities)]])
        
        with torch.no_grad():
            probs, logits = policy_net(cities_tensor, current_city_tensor, visited_mask)
        
        probs_np = probs.squeeze(0).numpy()
        
        # Choose highest probability unvisited city
        available_cities = [i for i in range(num_cities) if i not in visited]
        if not available_cities:
            break
            
        next_city = available_cities[np.argmax([probs_np[i] for i in available_cities])]
        
        # Store decision
        route_decisions[current_city] = {
            'action': next_city,
            'probability': probs_np[next_city],
            'available_actions': {i: probs_np[i] for i in available_cities}
        }
        
        state_values[current_city] = -np.log(probs_np[next_city] + 1e-8)
        
        current_city = next_city
        visited.add(next_city)
        optimal_route.append(next_city)
        step += 1
    
    # Return to start if we have a complete tour
    if len(visited) == num_cities:
        route_decisions[current_city] = {
            'action': 0,
            'probability': 1.0,
            'available_actions': {0: 1.0}
        }
        optimal_route.append(0)
    
    return optimal_route, route_decisions, state_values

def create_tsp_policy_graph(env, optimal_route, route_decisions, state_values):
    """Create a graph visualization of the TSP policy"""
    
    G = nx.DiGraph()
    num_cities = env.num_cities
    cities = env.cities
    
    # Add city nodes
    for i in range(num_cities):
        G.add_node(
            f"City{i}",
            city_id=i,
            x=cities[i][0],
            y=cities[i][1],
            value=state_values.get(i, 0),
            is_start=(i == 0)
        )
    
    # Add edges for the optimal route
    for i in range(len(optimal_route) - 1):
        from_city = optimal_route[i]
        to_city = optimal_route[i + 1]
        
        distance = np.linalg.norm(cities[from_city] - cities[to_city])
        prob = route_decisions.get(from_city, {}).get('probability', 1.0)
        
        G.add_edge(
            f"City{from_city}",
            f"City{to_city}",
            distance=distance,
            probability=prob,
            weight=distance
        )
    
    return G

def plot_tsp_policy_solution(env, optimal_route, route_decisions, state_values):
    """Plot the TSP policy solution with multiple visualizations"""
    
    num_cities = env.num_cities
    cities = env.cities
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Optimal Route
    ax1.set_title('Optimal TSP Route\n(Deterministic Policy)', fontsize=14, fontweight='bold')
    
    # Plot cities
    for i, (x, y) in enumerate(cities):
        color = 'red' if i == 0 else 'blue'
        marker = 's' if i == 0 else 'o'
        size = 100 if i == 0 else 80
        ax1.scatter(x, y, c=color, marker=marker, s=size, zorder=5)
        ax1.text(x + 0.1, y + 0.1, f'City{i}', fontsize=9, fontweight='bold')
    
    # Plot route
    route_x = [cities[i][0] for i in optimal_route]
    route_y = [cities[i][1] for i in optimal_route]
    ax1.plot(route_x, route_y, 'g-', linewidth=2, alpha=0.7, label='Optimal Route')
    
    # Add arrows to show direction
    for i in range(len(optimal_route) - 1):
        from_city = optimal_route[i]
        to_city = optimal_route[i + 1]
        
        dx = cities[to_city][0] - cities[from_city][0]
        dy = cities[to_city][1] - cities[from_city][1]
        
        ax1.arrow(cities[from_city][0], cities[from_city][1], 
                 dx * 0.8, dy * 0.8, head_width=0.2, head_length=0.2, 
                 fc='green', ec='green', alpha=0.7)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.axis('equal')
    
    # Plot 2: Policy Decision Graph
    ax2.set_title('Policy Decision Graph\n(Nodes = Cities, Edges = Decisions)', 
                  fontsize=14, fontweight='bold')
    
    G = create_tsp_policy_graph(env, optimal_route, route_decisions, state_values)
    
    # Create positions from city coordinates
    pos = {f"City{i}": (cities[i][0], cities[i][1]) for i in range(num_cities)}
    
    # Draw nodes
    node_colors = ['red' if G.nodes[node]['is_start'] else 'skyblue' for node in G.nodes()]
    node_sizes = [200 if G.nodes[node]['is_start'] else 150 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, edgecolors='black')
    
    # Draw edges with weights
    edge_colors = ['green' for edge in G.edges()]
    edge_widths = [2 + G.edges[edge]['probability'] * 3 for edge in G.edges()]
    
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color=edge_colors, 
                          width=edge_widths, alpha=0.7, arrows=True, 
                          arrowsize=20, arrowstyle='->')
    
    # Draw labels
    labels = {node: f"City{G.nodes[node]['city_id']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, ax=ax2, labels=labels, font_size=9, font_weight='bold')
    
    # Draw edge labels
    edge_labels = {}
    for u, v in G.edges():
        dist = G.edges[(u, v)]['distance']
        prob = G.edges[(u, v)]['probability']
        edge_labels[(u, v)] = f"d={dist:.2f}\np={prob:.2f}"
    
    nx.draw_networkx_edge_labels(G, pos, ax=ax2, edge_labels=edge_labels, font_size=7)
    
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action Probabilities Heatmap
    ax3.set_title('Action Probability Matrix\n(Rows = Current City, Columns = Next City)', 
                  fontsize=14, fontweight='bold')
    
    prob_matrix = np.zeros((num_cities, num_cities))
    for from_city in range(num_cities):
        if from_city in route_decisions:
            available_actions = route_decisions[from_city]['available_actions']
            for to_city in available_actions:
                prob_matrix[from_city, to_city] = available_actions[to_city]
    
    # Create custom colormap
    cmap = plt.cm.viridis
    im = ax3.imshow(prob_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(num_cities):
        for j in range(num_cities):
            if prob_matrix[i, j] > 0:
                color = "white" if prob_matrix[i, j] > 0.5 else "black"
                ax3.text(j, i, f'{prob_matrix[i, j]:.2f}',
                        ha="center", va="center", color=color, fontweight='bold', fontsize=8)
    
    ax3.set_xlabel('Next City')
    ax3.set_ylabel('Current City')
    ax3.set_xticks(range(num_cities))
    ax3.set_yticks(range(num_cities))
    ax3.set_xticklabels([f'C{i}' for i in range(num_cities)])
    ax3.set_yticklabels([f'C{i}' for i in range(num_cities)])
    plt.colorbar(im, ax=ax3, label='Action Probability')
    
    # Plot 4: Policy Value and Route Statistics
    ax4.set_title('Policy Value and Route Analysis', fontsize=14, fontweight='bold')
    
    # Calculate route statistics
    total_distance = 0
    for i in range(len(optimal_route) - 1):
        from_city = optimal_route[i]
        to_city = optimal_route[i + 1]
        total_distance += np.linalg.norm(cities[from_city] - cities[to_city])
    
    decision_certainties = [route_decisions[i]['probability'] for i in route_decisions if i in route_decisions]
    avg_certainty = np.mean(decision_certainties) if decision_certainties else 0
    
    # Create summary text
    summary_text = f"""
    TSP SOLUTION SUMMARY
    {'='*30}
    Number of Cities: {num_cities}
    Total Distance: {total_distance:.3f}
    Route: {' → '.join([f'City{i}' for i in optimal_route])}
    
    POLICY ANALYSIS
    {'='*30}
    Average Decision Certainty: {avg_certainty:.3f}
    Decisions Made: {len(route_decisions)}
    """
    
    if route_decisions:
        summary_text += "\nDECISION POINTS:\n"
        for i, city_id in enumerate(optimal_route[:-1]):
            if city_id in route_decisions:
                decision = route_decisions[city_id]
                summary_text += f"  City{city_id} → City{decision['action']}: p={decision['probability']:.3f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10, 
             family='monospace', verticalalignment='top')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return total_distance, avg_certainty

# Main execution
if __name__ == "__main__":
    print("TSP Combinatorial Optimization with Deterministic Policy")
    print("=" * 60)
    
    # Create TSP environment with fewer cities for stability
    env = TSPEnvironment(num_cities=6, seed=42)
    
    # Train policy
    policy_net, rewards, distances = train_tsp_policy(env, num_episodes=2000)
    
    # Extract deterministic policy
    print("\nExtracting deterministic policy...")
    optimal_route, route_decisions, state_values = extract_deterministic_tsp_policy(policy_net, env)
    
    # Create visualizations
    print("Creating policy visualizations...")
    total_distance, avg_certainty = plot_tsp_policy_solution(env, optimal_route, route_decisions, state_values)
    
    print(f"\nFinal Results:")
    print(f"Optimal Route: {' → '.join([f'City{i}' for i in optimal_route])}")
    print(f"Total Distance: {total_distance:.3f}")
    print(f"Average Decision Certainty: {avg_certainty:.3f}")
