import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MaskablePokeEnvWrapper(gym.Wrapper):
    """
    Wrapper that adds action_masks() method required by MaskablePPO.

    Also handles:
    - Action history tracking for oscillation detection
    - Skipping turns where the server needs no decision from us
    - Reporting the real battle result as info["battle_won"]
    - Dynamic observation/action space from rl_env
    """

    def __init__(self, env, rl_env):
        super().__init__(env)
        self.rl_env = rl_env
        self._current_battle = None
        self._last_action = None

        # Get spaces from rl_env
        agent = list(self.rl_env.possible_agents)[0]
        obs_space = self.rl_env.observation_spaces[agent]
        
        self.observation_space = spaces.Box(
            low=obs_space.low,
            high=obs_space.high,
            shape=obs_space.shape,
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.rl_env.ACTION_SPACE_SIZE)
    
    def reset(self, **kwargs):
        # rl_env resets its per-battle state itself when it sees a new battle tag
        obs, info = self.env.reset(**kwargs)
        self._current_battle = self.rl_env.battle1
        self._last_action = None
        return obs, info

    def step(self, action):
        # Update action history BEFORE taking the step (for oscillation detection)
        if self._current_battle is not None:
            self.rl_env.update_action_history(action, self._current_battle)

        self._last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        # When only the opponent has to act (e.g. their forced switch) the server ignores our
        # action, so play through those turns instead of handing PPO a decision that did nothing.
        while not (terminated or truncated) and not self.rl_env.agent1_to_move:
            obs, extra_reward, terminated, truncated, info = self.env.step(0)
            reward += extra_reward
        self._current_battle = self.rl_env.battle1
        if self._current_battle is not None and self._current_battle.finished:
            # Report the real result; the episode return is not a reliable win signal
            info = dict(info)
            info["battle_won"] = bool(self._current_battle.won)
        return obs, reward, terminated, truncated, info
    
    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions for MaskablePPO."""
        if self._current_battle is None:
            return np.ones(self.rl_env.ACTION_SPACE_SIZE, dtype=bool)
        return self.rl_env.get_action_mask(self._current_battle).astype(bool)
