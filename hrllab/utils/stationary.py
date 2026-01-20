import numpy as np
from scipy.linalg import null_space

def calculate_stationary_distribution(env, policy):
    """
    Calculate the stationary distribution of a policy using transition matrix.
    Only suitable for small environments like FrozenLake.
    
    Args:
        env: The FrozenLake environment
        policy: The policy (can be SB3 policy or Q-table)
    
    Returns:
        stationary_dist: Stationary distribution as a probability vector
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Build transition matrix P for the given policy
    P = np.zeros((n_states, n_states))
    
    if hasattr(policy, 'predict'):  # SB3 policy
        for s in range(n_states):
            action, _ = policy.predict(s, deterministic=False)
            for prob, next_state, _, _ in env.P[s][action]:
                P[next_state, s] += prob
    else:  # Assuming Q-table or deterministic policy
        for s in range(n_states):
            if isinstance(policy, np.ndarray):  # Q-table
                action = np.argmax(policy[s])
            else:  # Deterministic policy function
                action = policy(s)
            
            for prob, next_state, _, _ in env.P[s][action]:
                P[next_state, s] += prob
    
    # Transpose to get the right format: P[i,j] = prob(j -> i)
    P = P.T
    
    # Find stationary distribution (eigenvector with eigenvalue 1)
    eigenvalues, eigenvectors = np.linalg.eig(P)
    idx = np.where(np.isclose(eigenvalues, 1.0))[0]
    
    if len(idx) > 0:
        stationary_dist = np.real(eigenvectors[:, idx[0]])
        stationary_dist = stationary_dist / stationary_dist.sum()
        return stationary_dist
    else:
        return np.ones(n_states) / n_states  # Uniform distribution as fallback


def estimate_stationary_distribution(env, policy, num_episodes=1000, max_steps=100):
    """
    Estimate stationary distribution by running episodes and counting state visits.

    Args:
        env: The environment
        policy: The policy to evaluate
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        stationary_dist: Estimated stationary distribution
        state_visits: Raw state visit counts
    """
    n_states = env.observation_space.n
    state_visits = np.zeros(n_states)
    total_visits = 0

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle new gym API
            state = state[0]

        for step in range(max_steps):
            state_visits[state] += 1
            total_visits += 1

            # Get action from policy
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(state, deterministic=True)
            else:
                action = policy(state)

            # Take step
            next_state, reward, done, truncated, info = env.step(int(action))

            if done or truncated:
                break

            state = next_state

    stationary_dist = state_visits / total_visits
    return stationary_dist, state_visits