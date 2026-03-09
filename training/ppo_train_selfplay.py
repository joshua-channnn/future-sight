import os
import asyncio
import uuid
import numpy as np
import math
import pickle
import random
from typing import Callable, List, Optional, Tuple
from collections import deque

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO

from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

# IMPORTANT: Use the v13 RLPlayer with 157 features
from envs.rl_player_6v6_v13 import RL6v6Player
from envs.wrappers import MaskablePokeEnvWrapper


# Use explicit IPv4 localhost to avoid ::1 resolution issues.
SERVER_CONFIGURATION = ServerConfiguration(
    "ws://127.0.0.1:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)

class MaskableVecNormalize(VecNormalize):
    """VecNormalize that passes through action_masks() calls."""
    def action_masks(self) -> np.ndarray:
        return np.array(self.venv.env_method("action_masks"))


def cosine_schedule(initial_value: float, final_value: float, warmup_fraction: float = 0.02) -> Callable:
    """Cosine annealing learning rate schedule with warmup."""
    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if progress < warmup_fraction:
            return initial_value
        decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return final_value + (initial_value - final_value) * cosine_decay
    return func


class PPOPlayer(Player):
    """Lightweight PPO player for self-play opponents."""
    
    def __init__(self, model: MaskablePPO, rl_env: RL6v6Player, obs_rms=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.rl_env = rl_env
        self.obs_rms = obs_rms
        self.clip_obs = 10.0
    
    def choose_move(self, battle: AbstractBattle):
        obs = self.rl_env.embed_battle(battle)
        
        if self.obs_rms is not None:
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        
        mask = self.rl_env.get_action_mask(battle)
        action, _ = self.model.predict(
            obs.reshape(1, -1).astype(np.float32),
            deterministic=False,  # Stochastic for diversity
            action_masks=mask.reshape(1, -1)
        )
        return self.rl_env.action_to_order(int(action[0]), battle)


class SelfPlayManager:
    """
    Manages self-play opponent pool with Fictitious Self-Play.
    
    Key features:
    - Maintains pool of past checkpoints
    - Gradually reduces heuristic opponent over time
    - Caches loaded models to avoid repeated disk reads
    """
    
    def __init__(
        self,
        rl_env: RL6v6Player,
        heuristic_opponent: Player,
        server_configuration: ServerConfiguration,
        max_pool_size: int = 10,
        max_cached: int = 5,
    ):
        self.rl_env = rl_env
        self.heuristic_opponent = heuristic_opponent
        self.server_configuration = server_configuration
        self.max_pool_size = max_pool_size
        self.max_cached = max_cached
        
        # Checkpoint pool: list of (model_path, vec_path)
        self.checkpoint_pool: List[Tuple[str, Optional[str]]] = []
        
        # LRU cache for loaded models
        self.model_cache = {}  # path -> (model, obs_rms)
        self.cache_order = deque()
        
        self.total_steps = 0
        
        # Current opponent
        self._current_opponent = heuristic_opponent
        self._rng = random.Random(42)
    
    def _get_heuristic_probability(self) -> float:
        """
        Dynamic heuristic probability based on training progress.
        - Early training: Higher heuristic % to maintain baseline skills
        - Late training: Lower heuristic % for self-play robustness
        """
        if self.total_steps < 500_000:
            return 0.5  # 50% heuristic early
        elif self.total_steps < 1_500_000:
            return 0.3  # 30% mid-training
        elif self.total_steps < 3_000_000:
            return 0.2  # 20% later
        else:
            return 0.1  # 10% late game (mostly self-play)
    
    def _load_model(self, model_path: str, vec_path: Optional[str] = None):
        """Load model with LRU caching."""
        if model_path in self.model_cache:
            # Move to end of cache order (most recently used)
            self.cache_order.remove(model_path)
            self.cache_order.append(model_path)
            return self.model_cache[model_path]
        
        # Evict oldest if cache full
        while len(self.model_cache) >= self.max_cached:
            old_path = self.cache_order.popleft()
            del self.model_cache[old_path]
        
        model = MaskablePPO.load(model_path)
        obs_rms = None
        
        if vec_path and os.path.exists(vec_path):
            try:
                with open(vec_path, "rb") as f:
                    vec_data = pickle.load(f)
                if hasattr(vec_data, 'obs_rms'):
                    obs_rms = vec_data.obs_rms
            except Exception as e:
                print(f"  Warning: Could not load vec_norm from {vec_path}: {e}")
        
        self.model_cache[model_path] = (model, obs_rms)
        self.cache_order.append(model_path)
        
        return model, obs_rms
    
    def add_checkpoint(self, model_path: str, vec_path: Optional[str] = None):
        """Add a checkpoint to the self-play pool."""
        if not model_path.endswith('.zip'):
            model_path = model_path + '.zip'
        
        if not os.path.exists(model_path):
            print(f"  Warning: Checkpoint not found: {model_path}")
            return
        
        self.checkpoint_pool.append((model_path, vec_path))
        
        if len(self.checkpoint_pool) > self.max_pool_size:
            removed = self.checkpoint_pool.pop(0)
            # Also remove from cache if present
            if removed[0] in self.model_cache:
                self.cache_order.remove(removed[0])
                del self.model_cache[removed[0]]
        
        print(f"  ✓ Added checkpoint to pool (size: {len(self.checkpoint_pool)})")
    
    def select_opponent(self) -> Player:
        """Select opponent for next episode."""
        heuristic_prob = self._get_heuristic_probability()
        
        if self._rng.random() < heuristic_prob or not self.checkpoint_pool:
            self._current_opponent = self.heuristic_opponent
            return self._current_opponent
        
        # Sample from checkpoint pool (uniform for now)
        model_path, vec_path = self._rng.choice(self.checkpoint_pool)
        
        try:
            model, obs_rms = self._load_model(model_path, vec_path)
            self._current_opponent = PPOPlayer(
                model=model,
                rl_env=self.rl_env,
                obs_rms=obs_rms,
                battle_format="gen9randombattle",
                server_configuration=self.server_configuration,
                start_listening=False,
            )
        except Exception as e:
            print(f"  Warning: Failed to load opponent from {model_path}: {e}")
            self._current_opponent = self.heuristic_opponent
        
        return self._current_opponent
    
    def get_current_opponent(self) -> Player:
        return self._current_opponent
    
    def update_steps(self, steps: int):
        """Update total step count for dynamic scheduling."""
        self.total_steps = steps


class SelfPlayEnvWrapper(MaskablePokeEnvWrapper):
    """
    Environment wrapper that uses self-play opponent pool.
    Selects a new opponent at each episode reset.
    """
    
    def __init__(self, rl_env: RL6v6Player, self_play_manager: SelfPlayManager):
        self.rl_env = rl_env
        self.sp_manager = self_play_manager
        
        opponent = self_play_manager.select_opponent()
        wrapped = SingleAgentWrapper(rl_env, opponent)
        super().__init__(wrapped, rl_env)
    
    def reset(self, **kwargs):
        opponent = self.sp_manager.select_opponent()
        self.env = SingleAgentWrapper(self.rl_env, opponent)
        
        return super().reset(**kwargs)


class SelfPlayCallback(BaseCallback):
    """
    Callback for self-play training:
    - Saves checkpoints to the self-play pool
    - Runs evaluation against SimpleHeuristics (fixed benchmark)
    - Logs training metrics
    """
    
    def __init__(
        self,
        sp_manager: SelfPlayManager,
        eval_freq: int = 100_000,
        n_eval: int = 200,
        checkpoint_freq: int = 200_000,
        save_path: str = "models/",
        version: str = "v15",
        training_vec_env=None,
    ):
        super().__init__(verbose=1)
        self.sp_manager = sp_manager
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.checkpoint_freq = checkpoint_freq
        self.save_path = save_path
        self.version = version
        self.training_vec_env = training_vec_env
        
        self.checkpoint_count = 0
        self.best_win_rate = 0.0
        self.results = []
        
        os.makedirs(f"{save_path}/checkpoints", exist_ok=True)
        os.makedirs(f"{save_path}/state", exist_ok=True)

    def save_training_state(self, path: str):
        state = {
            "num_timesteps": self.num_timesteps,
            "checkpoint_count": self.checkpoint_count,
            "best_win_rate": self.best_win_rate,
            "results": self.results,
            "checkpoint_pool": list(self.sp_manager.checkpoint_pool),
            "sp_total_steps": self.sp_manager.total_steps,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_training_state(self, path: str) -> int:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.checkpoint_count = state.get("checkpoint_count", 0)
        self.best_win_rate = state.get("best_win_rate", 0.0)
        self.results = state.get("results", [])
        self.sp_manager.total_steps = state.get("sp_total_steps", 0)

        self.sp_manager.checkpoint_pool = []
        self.sp_manager.model_cache = {}
        self.sp_manager.cache_order = deque()
        for model_path, vec_path in state.get("checkpoint_pool", []):
            self.sp_manager.add_checkpoint(model_path, vec_path)
        return int(state.get("num_timesteps", 0))
    
    def _make_eval_env(self):
        """Create fresh evaluation environment."""
        eval_name = f"EvalBot-{uuid.uuid4().hex[:8]}"
        opp_name = f"EvalOpp-{uuid.uuid4().hex[:8]}"
        
        eval_rl_env = RL6v6Player(
            battle_format="gen9randombattle",
            challenge_timeout=60.0,
            account_configuration1=AccountConfiguration.generate(eval_name, rand=True),
            account_configuration2=AccountConfiguration.generate(opp_name, rand=True),
            server_configuration=SERVER_CONFIGURATION,
        )
        eval_wrapped = SingleAgentWrapper(
            eval_rl_env,
            SimpleHeuristicsPlayer(
                battle_format="gen9randombattle",
                server_configuration=SERVER_CONFIGURATION,
                start_listening=False,
            )
        )
        eval_env = MaskablePokeEnvWrapper(eval_wrapped, eval_rl_env)
        eval_vec_env = DummyVecEnv([lambda: eval_env])
        eval_vec_env = MaskableVecNormalize(
            eval_vec_env, norm_obs=True, norm_reward=False,
            clip_obs=10.0, gamma=0.995
        )
        
        if self.training_vec_env is not None:
            eval_vec_env.obs_rms = self.training_vec_env.obs_rms
        eval_vec_env.training = False
        
        return eval_vec_env
    
    def _run_evaluation(self) -> Tuple[float, float]:
        """Run evaluation games and return (win_rate, avg_reward)."""
        eval_vec_env = self._make_eval_env()
        
        wins = 0
        total_reward = 0
        completed = 0
        
        for _ in range(self.n_eval):
            obs = None
            for attempt in range(3):
                try:
                    obs = eval_vec_env.reset()
                    break
                except (asyncio.TimeoutError, TimeoutError) as e:
                    if "Agent is not challenging" in str(e):
                        eval_vec_env.close()
                        eval_vec_env = self._make_eval_env()
                        continue
                    raise
            
            if obs is None:
                print("  Eval reset failed. Skipping remaining games.")
                break
            
            done = False
            ep_reward = 0
            
            while not done:
                masks = eval_vec_env.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
                obs, reward, done, _ = eval_vec_env.step(action)
                ep_reward += reward[0]
            
            total_reward += ep_reward
            if ep_reward > 0:
                wins += 1
            completed += 1
        
        eval_vec_env.close()
        
        win_rate = wins / max(1, completed)
        avg_reward = total_reward / max(1, completed)
        
        return win_rate, avg_reward
    
    def _on_step(self):
        self.sp_manager.update_steps(self.num_timesteps)
        
        if self.num_timesteps % self.checkpoint_freq == 0 and self.num_timesteps > 0:
            self.checkpoint_count += 1
            ckpt_path = f"{self.save_path}/checkpoints/{self.version}_ckpt_{self.checkpoint_count}"
            vec_path = f"{ckpt_path}_vec.pkl"
            
            self.model.save(ckpt_path)
            if self.training_vec_env:
                self.training_vec_env.save(vec_path)
            
            self.sp_manager.add_checkpoint(ckpt_path, vec_path)

            latest_path = f"{self.save_path}/ppo_pokemon_6v6_{self.version}_latest"
            self.model.save(latest_path)
            if self.training_vec_env:
                self.training_vec_env.save(
                    f"{self.save_path}/vec_normalize_pokemon_6v6_{self.version}_latest.pkl"
                )
            self.save_training_state(f"{self.save_path}/state/training_state_{self.version}.pkl")
            
            heur_prob = self.sp_manager._get_heuristic_probability()
            print(f"\n>>> Checkpoint {self.checkpoint_count} @ {self.num_timesteps:,} steps")
            print(f"    Pool size: {len(self.sp_manager.checkpoint_pool)}")
            print(f"    Heuristic probability: {heur_prob:.0%}")
        
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            win_rate, avg_reward = self._run_evaluation()
            
            self.results.append({
                'step': self.num_timesteps,
                'win_rate': win_rate,
                'avg_reward': avg_reward,
            })
            
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/best", self.best_win_rate)
            self.logger.record("eval/avg_reward", avg_reward)
            self.logger.record("selfplay/pool_size", len(self.sp_manager.checkpoint_pool))
            self.logger.record("selfplay/heuristic_prob", self.sp_manager._get_heuristic_probability())
            
            print(f"\n{'='*60}")
            print(f">>> Eval @ {self.num_timesteps:,} steps")
            print(f"    Win rate vs SimpleHeuristics: {win_rate:.1%}")
            print(f"    Avg reward: {avg_reward:.3f}")
            print(f"    Best: {self.best_win_rate:.1%}")
            
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.model.save(f"{self.save_path}/ppo_pokemon_6v6_{self.version}_best")
                if self.training_vec_env:
                    self.training_vec_env.save(
                        f"{self.save_path}/vec_normalize_pokemon_6v6_{self.version}_best.pkl"
                    )
                print(f"    ★ New best model saved!")
            
            print(f"{'='*60}\n")
        
        return True


def train():
    VERSION = "v14"
    TOTAL_STEPS = 6_000_000
    RESUME_PATH = f"models/state/training_state_{VERSION}.pkl"
    
    V13_MODEL = "models/ppo_pokemon_6v6_v13"
    V13_VEC = "models/vec_normalize_pokemon_6v6_v13.pkl"
    
    print("=" * 70)
    print(f"PPO {VERSION}: Self-Play Training (from v13)")
    print("=" * 70)
    print("\nStrategy:")
    print("  • Load v13 best model (72% vs SimpleHeuristics)")
    print("  • Fictitious Self-Play with checkpoint pool")
    print("  • Dynamic heuristic probability:")
    print("      0-500k:   50% heuristic (maintain baseline)")
    print("      500k-1.5M: 30% heuristic")
    print("      1.5M-3M:  20% heuristic")
    print("      3M+:      10% heuristic (mostly self-play)")
    print("  • Checkpoint every 200k steps")
    print("  • Pool size: 10 checkpoints")
    print("  • LR: 5e-5 → 1e-5 (cosine)")
    print("  • Entropy: 0.03 (higher for self-play diversity)")
    print("=" * 70 + "\n")
    
    if not os.path.exists(V13_MODEL + ".zip"):
        print(f"ERROR: v13 model not found at {V13_MODEL}.zip")
        print("Please ensure v13 model exists before running self-play.")
        return None, []
    
    test_env = RL6v6Player(
        battle_format="gen9randombattle",
        server_configuration=SERVER_CONFIGURATION,
    )
    print(f"Observation space: {test_env.OBSERVATION_SIZE}")
    assert test_env.OBSERVATION_SIZE == 157, f"Expected 157, got {test_env.OBSERVATION_SIZE}"
    print("  ✓ Observation space verified (157 features)\n")
    
    rl_env = RL6v6Player(
        battle_format="gen9randombattle",
        challenge_timeout=60.0,
        server_configuration=SERVER_CONFIGURATION,
    )
    
    heuristic_opponent = SimpleHeuristicsPlayer(
        battle_format="gen9randombattle",
        server_configuration=SERVER_CONFIGURATION,
        start_listening=False,
    )
    
    sp_manager = SelfPlayManager(
        rl_env=rl_env,
        heuristic_opponent=heuristic_opponent,
        server_configuration=SERVER_CONFIGURATION,
        max_pool_size=10,
        max_cached=5,
    )
    
    # Add v13 to the pool as the first self-play opponent
    print("Adding v13 to self-play pool...")
    sp_manager.add_checkpoint(V13_MODEL, V13_VEC)
    
    env = SelfPlayEnvWrapper(rl_env, sp_manager)
    
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    vec_env = MaskableVecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.995
    )
    
    if os.path.exists(V13_VEC):
        try:
            with open(V13_VEC, "rb") as f:
                v13_vec = pickle.load(f)
            if hasattr(v13_vec, 'obs_rms'):
                vec_env.obs_rms = v13_vec.obs_rms
                print("  ✓ Loaded v13 normalization stats\n")
        except Exception as e:
            print(f"  Warning: Could not load v13 vec_norm: {e}\n")
    
    callback = SelfPlayCallback(
        sp_manager=sp_manager,
        eval_freq=100_000,
        n_eval=200,
        checkpoint_freq=200_000,  # More frequent checkpoints
        save_path="models/",
        version=VERSION,
        training_vec_env=vec_env,
    )

    resume_timesteps = 0
    if os.path.exists(RESUME_PATH):
        print(f"Resuming from {RESUME_PATH}...")
        model = MaskablePPO.load(
            f"models/ppo_pokemon_6v6_{VERSION}_latest",
            env=vec_env,
            learning_rate=cosine_schedule(5e-5, 1e-5),
            ent_coef=0.03,
            tensorboard_log=f"./logs/ppo_{VERSION}/",
        )
        resume_timesteps = callback.load_training_state(RESUME_PATH)
        remaining_steps = max(0, TOTAL_STEPS - resume_timesteps)

        progress_already_done = resume_timesteps / TOTAL_STEPS
        model.learning_rate = cosine_schedule(
            initial_value=5e-5 * (1 - progress_already_done),  # Approximate current LR
            final_value=1e-5,
            warmup_fraction=0.0  # No warmup on resume
            )
    else:
        print(f"Loading v13 model from {V13_MODEL}...")
        model = MaskablePPO.load(
            V13_MODEL,
            env=vec_env,
            # Override hyperparameters for self-play fine-tuning
            learning_rate=cosine_schedule(5e-5, 1e-5),
            ent_coef=0.03,  # Higher entropy for diversity in self-play
            tensorboard_log=f"./logs/ppo_{VERSION}/",
        )
        print("  ✓ Model loaded")
        print(f"    Starting from: v13 (72% baseline)")
        print(f"    Learning rate: 5e-5 → 1e-5")
        print(f"    Entropy coefficient: 0.03\n")
        remaining_steps = TOTAL_STEPS
    
    print("Starting self-play training...\n")
    
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=not os.path.exists(RESUME_PATH),
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
    finally:
        print("Saving resumable state...")
        callback.save_training_state(RESUME_PATH)
        model.save(f"models/ppo_pokemon_6v6_{VERSION}_latest")
        vec_env.save(f"models/vec_normalize_pokemon_6v6_{VERSION}_latest.pkl")
    
    model.save(f"models/ppo_pokemon_6v6_{VERSION}")
    vec_env.save(f"models/vec_normalize_pokemon_6v6_{VERSION}.pkl")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    print("\nWin Rate Progression vs SimpleHeuristics:")
    for r in callback.results:
        marker = " ★" if r['win_rate'] >= callback.best_win_rate else ""
        print(f"  Step {r['step']:>9,}: {r['win_rate']:>5.1%}  (reward: {r['avg_reward']:>6.3f}){marker}")
    
    print(f"\nBest win rate: {callback.best_win_rate:.1%}")
    print(f"v13 baseline: 72%")
    
    if callback.best_win_rate > 0.72:
        print(f"✓ Improvement: +{(callback.best_win_rate - 0.72)*100:.1f}%")
    elif callback.best_win_rate < 0.72:
        print(f"⚠ Regression: {(callback.best_win_rate - 0.72)*100:.1f}%")
        print("  This may be okay - self-play can temporarily reduce heuristic performance")
        print("  while improving robustness against diverse opponents.")
    
    print(f"\nModels saved:")
    print(f"  • models/ppo_pokemon_6v6_{VERSION}_best.zip")
    print(f"  • models/ppo_pokemon_6v6_{VERSION}.zip")
    
    return model, callback.results


if __name__ == "__main__":
    model, results = train()