import tomlkit


def create_config(env, total_timesteps=50000):
    """Create a configuration dictionary for the experiment."""
    return {
        "environment": {
            "name": "CustomFrozenLake",
            "map_name": "4x4",
            "n_states": env.n_states,
            "n_actions": env.action_space.n,
            "is_slippery": False,
            "reward_structure": "Regular: -1, Goal: +100"
        },
        # ... rest of the config remains the same
    }


