import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from collections import defaultdict, deque
import random

def train_lunar_lander_policy(total_timesteps=500000):
    """Train an optimal policy for LunarLander"""
    
    env = gym.make("LunarLander-v3")
    
    print("Training PPO on LunarLander...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=42,
        device='cpu'  
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Trained policy - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, env

def collect_successful_trajectories(model, env, num_episodes=1000, success_threshold=200):
    """Collect unique successful trajectories"""
    
    trajectories = []
    state_action_sequences = []
    success_count = 0
    
    print(f"Collecting successful trajectories (reward > {success_threshold})...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()  # Get just the observation, ignore info
        state = obs  # Use the observation directly
        trajectory = []
        state_actions = []
        
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < 1000:  # Limit steps
            action, _ = model.predict(state, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs  # Use the observation directly
            
            trajectory.append({
                'state': state.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_state.copy()
            })
            
            state_actions.append((state.copy(), action))
            
            state = next_state
            total_reward += reward
            steps += 1
        
        if total_reward > success_threshold:
            success_count += 1
            trajectories.append(trajectory)
            state_action_sequences.append(state_actions)
            
            if success_count % 50 == 0:
                print(f"Collected {success_count} successful trajectories...")
    
    print(f"Total successful trajectories collected: {success_count}/{num_episodes}")
    return trajectories, state_action_sequences

def create_2d_trajectory_plot(state_action_sequences):
    """Create 2D plot of successful trajectories using actual x-y positions"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for different actions
    action_colors = {
        0: 'red',    # Do nothing
        1: 'blue',   # Fire left orientation engine
        2: 'green',  # Fire main engine
        3: 'orange'  # Fire right orientation engine
    }
    
    action_names = {
        0: 'None',
        1: 'Left',
        2: 'Main',
        3: 'Right'
    }
    
    # Plot 1: X-Y Position Trajectories
    print("Plotting X-Y trajectories...")
    
    all_x_positions = []
    all_y_positions = []
    
    for i, trajectory in enumerate(state_action_sequences):
        if i % 10 == 0:  # Plot every 10th trajectory to reduce clutter
            states = [sa[0] for sa in trajectory]
            actions = [int(sa[1]) for sa in trajectory]
            
            # Extract x and y positions (indices 0 and 1 in LunarLander state)
            x_positions = [state[0] for state in states]
            y_positions = [state[1] for state in states]
            
            all_x_positions.extend(x_positions)
            all_y_positions.extend(y_positions)
            
            # Plot trajectory line
            ax1.plot(x_positions, y_positions, 'gray', alpha=0.3, linewidth=1)
            
            # Plot state-action points
            for j, (state, action) in enumerate(zip(states, actions)):
                color = action_colors[action]
                size = 20 + j * 1  # Increase size along trajectory
                alpha = 0.3 + (j / len(actions)) * 0.7  # Increase alpha along trajectory
                
                ax1.scatter(state[0], state[1], c=color, s=size, alpha=alpha, 
                           edgecolors='black', linewidth=0.5)
    
    # Add starting and ending points
    if state_action_sequences:
        start_states = [traj[0][0] for traj in state_action_sequences[::10]]
        end_states = [traj[-1][0] for traj in state_action_sequences[::10]]
        
        start_x = [state[0] for state in start_states]
        start_y = [state[1] for state in start_states]
        end_x = [state[0] for state in end_states]
        end_y = [state[1] for state in end_states]
        
        ax1.scatter(start_x, start_y, c='purple', s=80, alpha=0.8, 
                   label='Start States', marker='^')
        ax1.scatter(end_x, end_y, c='yellow', s=80, alpha=0.8, 
                   label='End States', marker='s')
    
    # Add landing pad (always at x=0, y=0)
    ax1.scatter(0, 0, c='black', s=100, marker='X', label='Landing Pad')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Successful Landing Trajectories\n(X-Y Position Space)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Ground Level')
    ax1.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Center Line')
    
    # Plot 2: X-Velocity vs Y-Velocity
    ax2.set_title('Velocity Space\n(X-Velocity vs Y-Velocity)', fontsize=12, fontweight='bold')
    
    for i, trajectory in enumerate(state_action_sequences):
        if i % 10 == 0:
            states = [sa[0] for sa in trajectory]
            actions = [int(sa[1]) for sa in trajectory]
            
            # Extract velocities (indices 2 and 3 in LunarLander state)
            x_velocities = [state[2] for state in states]
            y_velocities = [state[3] for state in states]
            
            # Plot trajectory in velocity space
            ax2.plot(x_velocities, y_velocities, 'gray', alpha=0.3, linewidth=1)
            
            for j, (state, action) in enumerate(zip(states, actions)):
                color = action_colors[action]
                ax2.scatter(state[2], state[3], c=color, s=10, alpha=0.5)
    
    ax2.set_xlabel('X Velocity')
    ax2.set_ylabel('Y Velocity')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='green', linestyle='--', alpha=0.5)
    
    # Plot 3: Action Distribution by Y Position
    ax3.set_title('Action Distribution by Altitude', fontsize=12, fontweight='bold')
    
    all_actions = []
    all_y_positions = []
    
    for trajectory in state_action_sequences:
        states = [sa[0] for sa in trajectory]
        actions = [sa[1] for sa in trajectory]
        
        y_positions = [state[1] for state in states]
        all_actions.extend(actions)
        all_y_positions.extend(y_positions)
    
    # Create action density plots by altitude
    if all_y_positions:
        y_bins = np.linspace(min(all_y_positions), max(all_y_positions), 20)
        
        for action in range(4):
            action_y = [y for a, y in zip(all_actions, all_y_positions) if a == action]
            if action_y:
                ax3.hist(action_y, bins=y_bins, alpha=0.6, color=action_colors[action], 
                        label=action_names[action], density=True, stacked=True)
    
    ax3.set_xlabel('Y Position (Altitude)')
    ax3.set_ylabel('Action Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Ground')
    
    # Plot 4: Angle and Angular Velocity
    ax4.set_title('Orientation Space\n(Angle vs Angular Velocity)', fontsize=12, fontweight='bold')
    
    for i, trajectory in enumerate(state_action_sequences):
        if i % 10 == 0:
            states = [sa[0] for sa in trajectory]
            actions = [int(sa[1]) for sa in trajectory]
            
            # Extract angle and angular velocity (indices 4 and 5 in LunarLander state)
            angles = [state[4] for state in states]  # Angle in radians
            ang_velocities = [state[5] for state in states]  # Angular velocity
            
            # Convert angles to degrees for better interpretation
            angles_deg = [np.degrees(angle) for angle in angles]
            
            # Plot trajectory in orientation space
            ax4.plot(angles_deg, ang_velocities, 'gray', alpha=0.3, linewidth=1)
            
            for j, (state, action) in enumerate(zip(states, actions)):
                color = action_colors[action]
                angle_deg = np.degrees(state[4])
                ax4.scatter(angle_deg, state[5], c=color, s=10, alpha=0.5)
    
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('Angular Velocity')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_action_timing_plot(state_action_sequences):
    """Plot action timing and sequences"""
    
    if not state_action_sequences:
        print("No trajectories to plot")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    action_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
    action_names = {0: 'None', 1: 'Left', 2: 'Main', 3: 'Right'}
    
    # Plot 1: Action sequences over time
    ax1.set_title('Action Sequences Over Time\n(Sample Trajectories)', fontsize=12, fontweight='bold')
    
    num_trajectories_to_plot = min(10, len(state_action_sequences))
    for i, trajectory in enumerate(state_action_sequences[:num_trajectories_to_plot]):
        states = [sa[0] for sa in trajectory]
        actions = [int(sa[1]) for sa in trajectory]
        times = list(range(len(actions)))
        
        # Plot action sequence
        for t, action in enumerate(actions):
            ax1.scatter(t, i, c=action_colors[action], s=50, alpha=0.7, 
                       marker='s', edgecolors='black')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Trajectory Index')
    ax1.set_yticks(range(num_trajectories_to_plot))
    ax1.grid(True, alpha=0.3)
    
    # Create custom legend
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor=action_colors[i], markersize=8,
                                label=action_names[i]) for i in range(4)]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Action frequency by phase of landing
    ax2.set_title('Action Frequency by Landing Phase', fontsize=12, fontweight='bold')
    
    # Divide trajectories into phases
    phase_actions = {0: [], 1: [], 2: []}  # Start, Middle, End
    
    for trajectory in state_action_sequences:
        traj_length = len(trajectory)
        if traj_length >= 3:
            # Start phase (first third)
            start_actions = [int(sa[1]) for sa in trajectory[:traj_length//3]]
            phase_actions[0].extend(start_actions)
            
            # Middle phase
            middle_actions = [int(sa[1]) for sa in trajectory[traj_length//3:2*traj_length//3]]
            phase_actions[1].extend(middle_actions)
            
            # End phase (last third)
            end_actions = [int(sa[1]) for sa in trajectory[2*traj_length//3:]]
            phase_actions[2].extend(end_actions)
    
    phases = ['Start', 'Middle', 'End']
    x_pos = np.arange(len(phases))
    width = 0.2
    
    for action in range(4):
        action_counts = []
        for phase in range(3):
            if phase_actions[phase]:
                count = phase_actions[phase].count(action) / len(phase_actions[phase])
                action_counts.append(count)
            else:
                action_counts.append(0)
        
        ax2.bar(x_pos + (action - 1.5) * width, action_counts, width,
               color=action_colors[action], label=action_names[action], alpha=0.8)
    
    ax2.set_xlabel('Landing Phase')
    ax2.set_ylabel('Action Frequency')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(phases)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_trajectory_patterns(trajectories, state_action_sequences):
    """Analyze patterns in successful trajectories"""
    
    if not trajectories:
        print("No trajectories to analyze")
        return
    
    print("\n" + "="*60)
    print("TRAJECTORY PATTERN ANALYSIS")
    print("="*60)
    
    # Basic statistics
    traj_lengths = [len(traj) for traj in trajectories]
    final_rewards = [sum(step['reward'] for step in traj) for traj in trajectories]
    
    print(f"Number of successful trajectories: {len(trajectories)}")
    print(f"Average trajectory length: {np.mean(traj_lengths):.1f} steps")
    print(f"Average final reward: {np.mean(final_rewards):.1f}")
    print(f"Max reward: {np.max(final_rewards):.1f}")
    print(f"Min reward: {np.min(final_rewards):.1f}")
    
    # Landing precision analysis
    final_positions = [traj[-1]['state'][:2] for traj in trajectories]  # Final (x, y)
    final_velocities = [traj[-1]['state'][2:4] for traj in trajectories]  # Final (vx, vy)
    
    landing_errors = [np.sqrt(x**2 + y**2) for x, y in final_positions]
    landing_speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in final_velocities]
    
    print(f"\nLanding Precision:")
    print(f"Average landing error: {np.mean(landing_errors):.3f}")
    print(f"Average landing speed: {np.mean(landing_speeds):.3f}")
    print(f"Perfect landings (error < 0.1): {sum(e < 0.1 for e in landing_errors)}")
    
    # Action frequency analysis
    all_actions = []
    for trajectory in state_action_sequences:
        for _, action in trajectory:
            all_actions.append(int(action))
    
    action_counts = {i: all_actions.count(i) for i in range(4)}
    total_actions = len(all_actions)
    
    print(f"\nAction Frequencies:")
    action_names = {0: 'None', 1: 'Left', 2: 'Main', 3: 'Right'}
    for action in range(4):
        freq = action_counts[action] / total_actions
        print(f"  {action_names[action]}: {freq:.3f} ({action_counts[action]} times)")

# Main execution
if __name__ == "__main__":
    print("LunarLander Optimal Policy Analysis")
    print("=" * 60)
    
    # Train optimal policy
    model, env = train_lunar_lander_policy(total_timesteps=200000)  # Reduced for faster testing
    
    # Collect successful trajectories
    trajectories, state_action_sequences = collect_successful_trajectories(
        model, env, num_episodes=100, success_threshold=100  # Reduced for faster testing
    )
    
    if not trajectories:
        print("No successful trajectories found! Lowering threshold...")
        trajectories, state_action_sequences = collect_successful_trajectories(
            model, env, num_episodes=100, success_threshold=50
        )
    
    if trajectories:
        # Create 2D trajectory plots
        print("Creating 2D trajectory visualizations...")
        fig = create_2d_trajectory_plot(state_action_sequences)
        
        # Create action timing plots
        print("Creating action timing plots...")
        create_action_timing_plot(state_action_sequences)
        
        # Analyze patterns
        analyze_trajectory_patterns(trajectories, state_action_sequences)
        
        print(f"\nVisualization Interpretation:")
        print("• Plot 1: Actual X-Y flight paths with actions colored")
        print("• Plot 2: Velocity space showing control strategies") 
        print("• Plot 3: Which actions are used at different altitudes")
        print("• Plot 4: Orientation control during landing")
        print("• Action timing: How action sequences evolve over time")
        
    else:
        print("No successful trajectories collected for analysis.")
    
    env.close()