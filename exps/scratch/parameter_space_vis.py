import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from tqdm import tqdm
import seaborn as sns
import torch

def extract_policy_vectors(model, num_states=16):
    """
    Alternative method using direct policy access
    """
    policy_vector = []
    
    # Get the policy network
    policy_net = model.policy
    
    for state in range(num_states):
        # Convert to tensor properly
        observation = torch.tensor([state], dtype=torch.float32).to(model.device)
        
        # Use the policy to get action distribution
        with torch.no_grad():
            # Get features from the policy network
            features = policy_net.extract_features(observation.unsqueeze(0))
            
            if hasattr(policy_net, 'mlp_extractor'):
                latent_pi = policy_net.mlp_extractor.policy_net(features)
            else:
                # For simpler architectures, use the policy head directly
                latent_pi = policy_net.action_net(features)
            
            # Get action distribution

            action_logits = policy_net.action_net(latent_pi)
            probabilities = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]
        
        policy_vector.extend(probabilities)
    
    return np.array(policy_vector)

def collect_policy_trajectory(env_name="FrozenLake-v1", num_policies=100, 
                            training_steps=5000, eval_episodes=50):
    """
    Train multiple PPO policies and collect their policy vectors
    Reduced defaults for faster execution
    """
    policy_vectors = []
    success_rates = []
    
    for i in tqdm(range(num_policies)):
        try:
            # Create environment
            env = make_vec_env(env_name, n_envs=1, env_kwargs={"is_slippery": False})
            
            # Train PPO with different random seeds
            model = PPO(
                "MlpPolicy", 
                env, 
                learning_rate=0.003,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                seed=i,
                verbose=0,
                device='cpu',
                policy_kwargs=dict(net_arch=[16, 16])  # Smaller network
            )
            
            # Train for a short period
            model.learn(total_timesteps=training_steps)
            
            # Extract policy vector
            policy_vec = extract_policy_vectors(model)
            policy_vectors.append(policy_vec)
            
            # Evaluate policy success rate
            success_rate = evaluate_policy_success(model, env_name, eval_episodes)
            success_rates.append(success_rate)
            
            env.close()
            
        except Exception as e:
            print(f"Error training policy {i}: {e}")
            continue
    
    return np.array(policy_vectors), np.array(success_rates)

def evaluate_policy_success(model, env_name, num_episodes=50):
    """Evaluate policy success rate"""
    env = gym.make(env_name, is_slippery=False)
    successes = 0
    
    for _ in range(num_episodes):
        obs = env.reset()
        if not isinstance(obs, int):
            obs = obs[0]
        done = False
        
        while not done:
            # Use deterministic evaluation for consistent results
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            
            if not isinstance(next_obs, int):
                next_obs = next_obs[0]
            obs = next_obs
            
            if reward > 0:
                successes += 1
                break
            if done and reward == 0:
                break
    
    env.close()
    return successes / num_episodes


def analyze_policy_manifold(policy_vectors, success_rates):
    """
    Perform PCA on policy space and analyze the manifold structure
    """
    # Apply PCA
    pca = PCA(n_components=2)
    policy_pca = pca.fit_transform(policy_vectors)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D PCA plot colored by success rate
    scatter1 = axes[0].scatter(policy_pca[:, 0], policy_pca[:, 1], 
                              c=success_rates, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter1, ax=axes[0], label='Success Rate')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Policy Space PCA\n(Color = Success Rate)')
    
    # Binary success/failure plot
    success_binary = success_rates > 0.5
    colors = ['red' if not success else 'green' for success in success_binary]
    axes[1].scatter(policy_pca[:, 0], policy_pca[:, 1], c=colors, alpha=0.7, s=50)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('Policy Space PCA\n(Red = Failure, Green = Success)')
    axes[1].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Success'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Failure')
    ])
    
    plt.tight_layout()
    plt.show()
    
    return pca, policy_pca

def analyze_policy_characteristics(policy_vectors, success_rates):
    """
    Analyze what makes policies successful
    """
    # Reshape to (num_policies, num_states, num_actions)
    policy_matrix = policy_vectors.reshape(len(policy_vectors), 16, 4)
    
    # Find most successful and least successful policies
    success_idx = np.argmax(success_rates)
    failure_idx = np.argmin(success_rates)
    
    print(f"Most successful policy: {success_rates[success_idx]:.3f}")
    print(f"Least successful policy: {success_rates[failure_idx]:.3f}")
    
    # Plot policy heatmaps for best and worst
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Best policy
    best_policy = policy_matrix[success_idx]
    im1 = axes[0, 0].imshow(best_policy, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Best Policy (Success: {success_rates[success_idx]:.2f})')
    axes[0, 0].set_xlabel('Action')
    axes[0, 0].set_ylabel('State')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Worst policy  
    worst_policy = policy_matrix[failure_idx]
    im2 = axes[0, 1].imshow(worst_policy, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Worst Policy (Success: {success_rates[failure_idx]:.2f})')
    axes[0, 1].set_xlabel('Action')
    axes[0, 1].set_ylabel('State')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Policy entropy (measure of randomness)
    policy_entropy = -np.sum(policy_matrix * np.log(policy_matrix + 1e-8), axis=2)
    avg_entropy = np.mean(policy_entropy, axis=1)
    
    axes[1, 0].scatter(avg_entropy, success_rates, alpha=0.6)
    axes[1, 0].set_xlabel('Average Policy Entropy')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('Policy Determinism vs Success')
    
    # Policy consistency (how deterministic are the policies)
    max_probs = np.max(policy_matrix, axis=2)
    avg_max_prob = np.mean(max_probs, axis=1)
    
    axes[1, 1].scatter(avg_max_prob, success_rates, alpha=0.6)
    axes[1, 1].set_xlabel('Average Max Action Probability')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Policy Determinism vs Success')
    
    plt.tight_layout()
    plt.show()
    
    return policy_matrix


def analyze_policy_clusters(policy_vectors, success_rates, threshold=0.8):
    """
    Analyze clusters of successful vs unsuccessful policies
    """
    successful_policies = policy_vectors[success_rates > threshold]
    unsuccessful_policies = policy_vectors[success_rates < 0.2]
    
    print(f"Successful policies: {len(successful_policies)}")
    print(f"Unsuccessful policies: {len(unsuccessful_policies)}")
    
    # Analyze policy variance
    successful_var = np.var(successful_policies, axis=0)
    unsuccessful_var = np.var(unsuccessful_policies, axis=0)
    
    # Compare policy distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Average successful policy
    avg_successful = np.mean(successful_policies, axis=0)
    avg_unsuccessful = np.mean(unsuccessful_policies, axis=0)
    
    # Reshape to state-action matrix
    successful_matrix = avg_successful.reshape(16, 4)
    unsuccessful_matrix = avg_unsuccessful.reshape(16, 4)
    
    # Plot average policy heatmaps
    axes[0, 0].imshow(successful_matrix, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Average Successful Policy')
    axes[0, 0].set_xlabel('Action')
    axes[0, 0].set_ylabel('State')
    
    axes[0, 1].imshow(unsuccessful_matrix, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Average Unsuccessful Policy')
    axes[0, 1].set_xlabel('Action')
    axes[0, 1].set_ylabel('State')
    
    # Policy divergence
    policy_divergence = np.abs(successful_matrix - unsuccessful_matrix)
    axes[1, 0].imshow(policy_divergence, cmap='Reds', aspect='auto')
    axes[1, 0].set_title('Policy Difference (Successful - Unsuccessful)')
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('State')
    
    # Success rate histogram
    axes[1, 1].hist(success_rates, bins=20, alpha=0.7, color='skyblue')
    axes[1, 1].axvline(threshold, color='red', linestyle='--', label='Success threshold')
    axes[1, 1].set_xlabel('Success Rate')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].set_title('Policy Success Rate Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return successful_policies, unsuccessful_policies

def analyze_policy_evolution(env_name="FrozenLake-v1", num_training_steps=10000, checkpoints=20):
    """
    Analyze how policy evolves during training in the policy space
    """
    env = make_vec_env(env_name, n_envs=1, env_kwargs={"is_slippery": False})
    
    policy_trajectory = []
    success_trajectory = []
    
    model = PPO("MlpPolicy", env, verbose=0)
    
    steps_per_checkpoint = num_training_steps // checkpoints
    
    for step in range(checkpoints):
        model.learn(total_timesteps=steps_per_checkpoint, reset_num_timesteps=False)
        
        # Extract current policy
        policy_vec = extract_policy_vectors(model)
        policy_trajectory.append(policy_vec)
        
        # Evaluate current success rate
        success_rate = evaluate_policy_success(model, env_name, 50)
        success_trajectory.append(success_rate)
    
    env.close()
    
    # Project trajectory to PCA space
    policy_trajectory = np.array(policy_trajectory)
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(policy_trajectory)
    
    # Plot training trajectory
    plt.figure(figsize=(10, 8))
    plt.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                c=success_trajectory, cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(label='Success Rate')
    
    # Add trajectory lines with arrows
    for i in range(len(trajectory_2d)-1):
        plt.arrow(trajectory_2d[i, 0], trajectory_2d[i, 1],
                 trajectory_2d[i+1, 0]-trajectory_2d[i, 0],
                 trajectory_2d[i+1, 1]-trajectory_2d[i, 1],
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
    
    # Annotate start and end
    plt.annotate('Start', xy=trajectory_2d[0, :], xytext=(5, 5), 
                 textcoords='offset points', color='red', weight='bold')
    plt.annotate('End', xy=trajectory_2d[-1, :], xytext=(5, 5), 
                 textcoords='offset points', color='blue', weight='bold')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Policy Evolution During Training')
    plt.show()
    
    return policy_trajectory, success_trajectory

def main():
    # Quick test with fewer policies
    print("Training PPO policies...")
    policy_vectors, success_rates = collect_policy_trajectory(
        num_policies=50,    # Small number for testing
        training_steps=2000, # Reduced training
        eval_episodes=20    # Reduced evaluation
    )
    
    if len(policy_vectors) == 0:
        print("No policies were successfully trained!")
        return
    
    print(f"Successfully collected {len(policy_vectors)} policies")
    print(f"Success rates: min={np.min(success_rates):.3f}, "
          f"max={np.max(success_rates):.3f}, mean={np.mean(success_rates):.3f}")
    
    # Analyze policy manifold
    print("\nPerforming PCA on policy space...")
    pca, policy_pca = analyze_policy_manifold(policy_vectors, success_rates)
    
    # Analyze policy characteristics
    print("\nAnalyzing policy characteristics...")
    policy_matrix = analyze_policy_characteristics(policy_vectors, success_rates)
    
    # Print insights
    print("\n" + "="*50)
    print("KEY INSIGHTS:")
    print("="*50)
    print(f"1. Policy space dimension: {policy_vectors.shape[1]} -> 2 principal components")
    print(f"2. PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
    print(f"3. Success rate distribution shape: {success_rates.shape}")

if __name__ == "__main__":
    main()