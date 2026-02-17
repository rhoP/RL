import gymnasium as gym
import numpy as np
from collections import defaultdict
from scipy.linalg import eigvals

# -----------------------------
# Configuration
# -----------------------------
ENV_NAME = "FrozenLake-v1"  # Try: FrozenLake-v1, Taxi-v3, CliffWalking-v0
NUM_EPISODES = 5000
MAX_STEPS = 200
EPS = 1e-12


# -----------------------------
# Collect Transitions
# -----------------------------
env = gym.make(ENV_NAME)
nS = env.observation_space.n
nA = env.action_space.n

counts = np.zeros((nS, nS))

for _ in range(NUM_EPISODES):
    s, _ = env.reset()
    for _ in range(MAX_STEPS):
        a = env.action_space.sample()
        s_next, r, terminated, truncated, _ = env.step(a)

        counts[s, s_next] += 1
        s = s_next

        if terminated or truncated:
            break

env.close()

# -----------------------------
# Estimate Transition Matrix
# -----------------------------
P = counts / (counts.sum(axis=1, keepdims=True) + EPS)

# Remove states never visited
visited = counts.sum(axis=1) > 0
P = P[visited][:, visited]
nS_eff = P.shape[0]

# -----------------------------
# Local Transition Entropy
# -----------------------------
local_entropy = -np.sum(P * np.log(P + EPS), axis=1)

print("Mean local entropy:", local_entropy.mean())
print("Min local entropy:", local_entropy.min())
print("Max local entropy:", local_entropy.max())

# -----------------------------
# Stationary Distribution
# -----------------------------
eigvals_full = eigvals(P.T)
idx = np.argmax(np.real(eigvals_full))
stat_dist = np.real(eigvals_full[idx])

# Compute stationary via eigenvector
eigvals_full, eigvecs = np.linalg.eig(P.T)
stat_idx = np.argmax(np.real(eigvals_full))
mu = np.real(eigvecs[:, stat_idx])
mu = np.abs(mu)
mu = mu / mu.sum()

# -----------------------------
# Entropy Rate
# -----------------------------
entropy_rate = -np.sum(mu[:, None] * P * np.log(P + EPS))
print("Entropy rate:", entropy_rate)

# -----------------------------
# Spectral Gap
# -----------------------------
evals = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
lambda1 = evals[0]
lambda2 = evals[1] if len(evals) > 1 else 0
spectral_gap = lambda1 - lambda2

print("Top eigenvalues:", evals[:5])
print("Spectral gap:", spectral_gap)

# -----------------------------
# Simple Diagnostic
# -----------------------------
if local_entropy.mean() < 0.3:
    print("\nEnvironment appears near-deterministic.")
elif spectral_gap > 0.2:
    print("\nEnvironment likely has metastable structure (good for abstraction).")
else:
    print("\nEnvironment likely highly mixing / stochastic.")
