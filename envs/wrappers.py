import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper


class MaskablePokeEnvWrapper(gym.Wrapper):
    """
    Wrapper that adds action_masks() method required by MaskablePPO.
    
    Also handles:
    - Battle state reset between episodes
    - Action history tracking for oscillation detection
    - Dynamic observation/action space from rl_env
    """
    
    def __init__(self, env, rl_env):
        super().__init__(env)
        self.rl_env = rl_env
        self._current_battle = None
        self._last_action = None
        
        # Get spaces from rl_env (handles both v13 and v15 observation sizes)
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
        obs, info = self.env.reset(**kwargs)
        self._current_battle = self.rl_env.battle1
        self._last_action = None
        # Reset battle-specific state in rl_env
        self.rl_env._reset_battle_state()
        return obs, info
    
    def step(self, action):
        # Update action history BEFORE taking the step (for oscillation detection)
        if self._current_battle is not None:
            self.rl_env.update_action_history(action, self._current_battle)
        
        self._last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_battle = self.rl_env.battle1
        return obs, reward, terminated, truncated, info
    
    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions for MaskablePPO."""
        if self._current_battle is None:
            return np.ones(self.rl_env.ACTION_SPACE_SIZE, dtype=bool)
        return self.rl_env.get_action_mask(self._current_battle).astype(bool)


class IndexPreservingVecNormalize:
    """
    A wrapper that normalizes only the float portion of observations,
    preserving integer indices for embedding lookup.
    
    Use this instead of VecNormalize when using embedding-based observations.
    
    Args:
        venv: Vectorized environment
        n_indices: Number of index dimensions at the start of observation (default 84 for v15)
        norm_obs: Whether to normalize float observations
        norm_reward: Whether to normalize rewards
        clip_obs: Clipping value for normalized observations
        clip_reward: Clipping value for normalized rewards
        gamma: Discount factor for reward normalization
        epsilon: Small constant for numerical stability
    """
    
    def __init__(
        self,
        venv,
        n_indices: int = 84,
        norm_obs: bool = True,
        norm_reward: bool = False,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        self.venv = venv
        self.n_indices = n_indices
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Get observation shape
        obs_shape = venv.observation_space.shape[0]
        self.n_floats = obs_shape - n_indices
        
        # Running statistics for float features only
        self.obs_mean = np.zeros(self.n_floats, dtype=np.float32)
        self.obs_var = np.ones(self.n_floats, dtype=np.float32)
        self.obs_count = epsilon
        
        # Reward statistics
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = epsilon
        self.returns = np.zeros(venv.num_envs)
        
        # Training flag
        self.training = True
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        return getattr(self.venv, name)
    
    def _update_obs_stats(self, obs_floats: np.ndarray):
        """Update running mean and variance for float observations."""
        batch_mean = np.mean(obs_floats, axis=0)
        batch_var = np.var(obs_floats, axis=0)
        batch_count = obs_floats.shape[0]
        
        # Welford's online algorithm
        delta = batch_mean - self.obs_mean
        tot_count = self.obs_count + batch_count
        
        self.obs_mean = self.obs_mean + delta * batch_count / tot_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.obs_count * batch_count / tot_count
        self.obs_var = M2 / tot_count
        self.obs_count = tot_count
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation, preserving indices."""
        if not self.norm_obs:
            return obs
        
        # Split indices and floats
        indices = obs[:, :self.n_indices]
        floats = obs[:, self.n_indices:]
        
        # Update stats if training
        if self.training:
            self._update_obs_stats(floats)
        
        # Normalize floats
        normalized_floats = (floats - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        normalized_floats = np.clip(normalized_floats, -self.clip_obs, self.clip_obs)
        
        # Recombine
        return np.concatenate([indices, normalized_floats], axis=1).astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.venv.reset(**kwargs)
        self.returns = np.zeros(self.venv.num_envs)
        return self._normalize_obs(obs), info
    
    def step(self, actions):
        obs, rewards, dones, truncated, infos = self.venv.step(actions)
        
        # Normalize observations
        obs = self._normalize_obs(obs)
        
        # Normalize rewards if enabled
        if self.norm_reward:
            self.returns = self.returns * self.gamma + rewards
            # Update reward stats
            if self.training:
                batch_mean = np.mean(self.returns)
                batch_var = np.var(self.returns)
                delta = batch_mean - self.ret_mean
                self.ret_count += 1
                self.ret_mean += delta / self.ret_count
                self.ret_var += (batch_var - self.ret_var) / self.ret_count
            
            rewards = rewards / np.sqrt(self.ret_var + self.epsilon)
            rewards = np.clip(rewards, -self.clip_reward, self.clip_reward)
        
        # Reset returns for done episodes
        self.returns[dones] = 0.0
        
        return obs, rewards, dones, truncated, infos
    
    def set_training(self, training: bool):
        """Set training mode (updates running statistics when True)."""
        self.training = training
    
    def save(self, path: str):
        """Save normalization statistics."""
        np.savez(
            path,
            obs_mean=self.obs_mean,
            obs_var=self.obs_var,
            obs_count=self.obs_count,
            ret_mean=self.ret_mean,
            ret_var=self.ret_var,
            ret_count=self.ret_count,
        )
    
    def load(self, path: str):
        """Load normalization statistics."""
        data = np.load(path)
        self.obs_mean = data["obs_mean"]
        self.obs_var = data["obs_var"]
        self.obs_count = data["obs_count"]
        self.ret_mean = data["ret_mean"]
        self.ret_var = data["ret_var"]
        self.ret_count = data["ret_count"]


class ProgressiveCurriculumWrapper(gym.Env):
    """
    Wrapper for curriculum learning with multiple opponents.
    
    Gradually shifts from easier to harder opponents based on training progress.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, rl_env, opponents, schedule):
        """
        Args:
            rl_env: The RL player environment
            opponents: List of opponent players
            schedule: List of (step_threshold, weights) tuples
                      weights is a list of probabilities for each opponent
        """
        self.rl_env = rl_env
        self.opponents = opponents
        self.schedule = schedule
        self._step_count = 0
        self._current_opponent_idx = 0
        self._create_env(0)
        
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
    
    def _create_env(self, opponent_idx):
        """Create environment with specified opponent."""
        self.wrapped_env = SingleAgentWrapper(
            self.rl_env,
            self.opponents[opponent_idx]
        )
        self.env = MaskablePokeEnvWrapper(self.wrapped_env, self.rl_env)
    
    def _get_weights_at_step(self, step):
        """Interpolate opponent weights based on current step."""
        for i, (threshold, weights) in enumerate(self.schedule):
            if step < threshold:
                if i == 0:
                    return self.schedule[0][1]
                prev_threshold, prev_weights = self.schedule[i-1]
                progress = (step - prev_threshold) / (threshold - prev_threshold)
                return [
                    prev_weights[j] + progress * (weights[j] - prev_weights[j])
                    for j in range(len(weights))
                ]
        return self.schedule[-1][1]
    
    def _select_opponent(self):
        """Select opponent based on current weights."""
        weights = self._get_weights_at_step(self._step_count)
        self._current_opponent_idx = np.random.choice(len(self.opponents), p=weights)
        return self._current_opponent_idx
    
    def reset(self, seed=None, options=None):
        """Reset and potentially switch opponent."""
        super().reset(seed=seed)
        opponent_idx = self._select_opponent()
        self._create_env(opponent_idx)
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action):
        self._step_count += 1
        return self.env.step(action)
    
    def action_masks(self):
        """Required for MaskablePPO."""
        return self.env.action_masks()
    
    def render(self):
        pass
    
    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def get_current_weights(self):
        """Get current opponent selection weights."""
        return self._get_weights_at_step(self._step_count)
    
    def get_step_count(self):
        """Get total steps taken."""
        return self._step_count


class MixedOpponentWrapper(gym.Wrapper):
    """
    Wrapper that randomly picks opponent each game.
    
    Simpler than ProgressiveCurriculumWrapper - just uniform random selection.
    """
    
    def __init__(self, env, rl_env, opponents):
        super().__init__(env)
        self.rl_env = rl_env
        self.opponents = opponents
        self._current_battle = None
        
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
        # Pick random opponent each game
        opponent = random.choice(self.opponents)
        self.env.opponent = opponent
        obs, info = self.env.reset(**kwargs)
        self._current_battle = self.rl_env.battle1
        # Reset battle-specific state
        self.rl_env._reset_battle_state()
        return obs, info
    
    def step(self, action):
        # Update action history BEFORE taking the step
        if self._current_battle is not None:
            self.rl_env.update_action_history(action, self._current_battle)
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_battle = self.rl_env.battle1
        return obs, reward, terminated, truncated, info
    
    def action_masks(self):
        """Return boolean mask of valid actions for MaskablePPO."""
        if self._current_battle is None:
            return np.ones(self.rl_env.ACTION_SPACE_SIZE, dtype=bool)
        return self.rl_env.get_action_mask(self._current_battle).astype(bool)