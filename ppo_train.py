import os
import asyncio
import uuid
import numpy as np
import math
import time
import random
from typing import Callable, List, Optional
from collections import deque

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO

from poke_env.player import Player, SimpleHeuristicsPlayer, RandomPlayer, MaxBasePowerPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.ps_client import AccountConfiguration

from RLPlayer6v6 import RL6v6Player
from wrappers import MaskablePokeEnvWrapper, ProgressiveCurriculumWrapper


class MaskableVecNormalize(VecNormalize):
    """VecNormalize that passes through action_masks() calls."""
    def action_masks(self) -> np.ndarray:
        return np.array(self.venv.env_method("action_masks"))


def cosine_schedule(initial_value: float, final_value: float, warmup_fraction: float = 0.05) -> Callable:
    """Cosine annealing learning rate schedule with warmup."""
    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if progress < warmup_fraction:
            # Linear warmup
            return initial_value * (progress / warmup_fraction)
        decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return final_value + (initial_value - final_value) * cosine_decay
    return func


class EvalCallback(BaseCallback):
    """Evaluation callback with detailed logging."""
    
    def __init__(
        self,
        eval_freq: int = 100_000,
        n_eval: int = 200,
        save_path: str = "models/",
        training_vec_env=None,
        version: str = "v13",
    ):
        super().__init__(verbose=1)
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.save_path = save_path
        self.training_vec_env = training_vec_env
        self.version = version
        
        self.results = []
        self.best_win_rate = 0.0
        
        os.makedirs(save_path, exist_ok=True)

    def _make_eval_env(self):
        """Create fresh evaluation environment."""
        eval_name = f"EvalBot-{uuid.uuid4().hex[:8]}"
        opp_name = f"EvalOpp-{uuid.uuid4().hex[:8]}"
        eval_rl_env = RL6v6Player(
            battle_format="gen9randombattle",
            challenge_timeout=60.0,
            account_configuration1=AccountConfiguration.generate(eval_name, rand=True),
            account_configuration2=AccountConfiguration.generate(opp_name, rand=True),
        )
        eval_wrapped = SingleAgentWrapper(
            eval_rl_env,
            SimpleHeuristicsPlayer(battle_format="gen9randombattle")
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
    
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            eval_vec_env = self._make_eval_env()
            
            wins = 0
            total_reward = 0
            completed = 0
            
            for _ in range(self.n_eval):
                obs = None
                for _ in range(3):
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
                    print("Eval reset failed. Skipping eval batch.")
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
            
            self.results.append({
                'step': self.num_timesteps,
                'win_rate': win_rate,
                'avg_reward': avg_reward
            })
            
            # Log to tensorboard
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/best", self.best_win_rate)
            self.logger.record("eval/avg_reward", avg_reward)
            
            print(f"\n{'='*60}")
            print(f">>> Eval @ {self.num_timesteps:,} steps")
            print(f"    Win rate vs SimpleHeuristics: {win_rate:.1%} ({wins}/{completed})")
            print(f"    Avg reward: {avg_reward:.3f}")
            print(f"    Best: {self.best_win_rate:.1%}")
            
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.model.save(f"{self.save_path}/ppo_pokemon_6v6_{self.version}_best")
                if self.training_vec_env:
                    self.training_vec_env.save(f"{self.save_path}/vec_normalize_pokemon_6v6_{self.version}_best.pkl")
                print(f"    ★ New best model saved!")
            
            print(f"{'='*60}\n")
        
        return True


class CheckpointCallback(BaseCallback):
    """Save checkpoints at regular intervals."""
    
    def __init__(self, save_freq: int, save_path: str, version: str, training_vec_env=None):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.version = version
        self.training_vec_env = training_vec_env
        self.checkpoint_count = 0
        
        os.makedirs(f"{save_path}/checkpoints", exist_ok=True)
    
    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            self.checkpoint_count += 1
            ckpt_path = f"{self.save_path}/checkpoints/{self.version}_ckpt_{self.checkpoint_count}"
            self.model.save(ckpt_path)
            if self.training_vec_env:
                self.training_vec_env.save(f"{ckpt_path}_vec.pkl")
            print(f"\n>>> Checkpoint {self.checkpoint_count} saved at {self.num_timesteps:,} steps\n")
        return True


class MetricsCallback(BaseCallback):
    """Log training metrics for monitoring."""
    
    def __init__(self, log_freq: int = 10_000):
        super().__init__()
        self.log_freq = log_freq
    
    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0 and self.num_timesteps > 0:
            # Log key PPO metrics
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # These come from the rollout buffer
                pass  # SB3 logs these automatically
        return True


def train():
    VERSION = "v14"
    TOTAL_STEPS = 8_000_000  # More steps since training from scratch
    
    print("=" * 70)
    print(f"PPO {VERSION}: Fresh Training with Enhanced Features")
    print("=" * 70)
    print("\nNew features (157 total):")
    print("  • Turn number + game phase (4) - fixes Tera timing")
    print("  • Team advantage context (3) - resource awareness")
    print("  • Switch context + oscillation detection (5) - fixes over-switching")
    print("  • Setup opportunity detection (3) - encourages setup moves")
    print("  • Healing context (3) - encourages recovery moves")
    print("\nTraining strategy:")
    print("  • Curriculum: MaxDamage → SimpleHeuristics")
    print("  • Higher gamma (0.995) for long-horizon credit assignment")
    print("  • Longer rollouts (4096 steps) for better value estimates")
    print("  • Oscillation penalty in reward shaping")
    print("=" * 70 + "\n")
    
    # === VERIFY OBSERVATION SPACE ===
    test_env = RL6v6Player(battle_format="gen9randombattle")
    print(f"Observation space: {test_env.OBSERVATION_SIZE}")
    assert test_env.OBSERVATION_SIZE == 184, f"Expected 184, got {test_env.OBSERVATION_SIZE}"
    print("  ✓ Observation space verified (184 features)\n")
    
    # === CREATE ENVIRONMENT WITH CURRICULUM ===
    rl_env = RL6v6Player(
        battle_format="gen9randombattle",
        challenge_timeout=60.0,
    )
    
    # Curriculum opponents: start easy, get harder
    opponents = [
        MaxBasePowerPlayer(battle_format="gen9randombattle"),      # 0: Easy
        SimpleHeuristicsPlayer(battle_format="gen9randombattle"), # 1: Hard
    ]
    
    # Curriculum schedule: (step_threshold, [weight_easy, weight_hard])
    # Gradually shift from easy to hard opponent
    schedule = [
        (0,         [1.0, 0.0]),   # Start: 100% MaxDamage
        (500_000,   [0.8, 0.2]),   # 500k: 80/20
        (1_000_000, [0.5, 0.5]),   # 1M: 50/50
        (2_000_000, [0.2, 0.8]),   # 2M: 20/80
        (3_000_000, [0.0, 1.0]),   # 3M+: 100% SimpleHeuristics
    ]
    
    env = ProgressiveCurriculumWrapper(rl_env, opponents, schedule)
    
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    vec_env = MaskableVecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward - we have careful shaping
        clip_obs=10.0,
        gamma=0.995
    )
    
    # === CREATE MODEL ===
    print("Creating new model...")
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        # Learning rate with warmup and cosine decay
        learning_rate=cosine_schedule(1e-4, 1e-5, warmup_fraction=0.02),
        
        # Rollout settings - longer for better value estimates
        n_steps=8192,           # Steps per rollout (was 2048)
        batch_size=256,         # Minibatch size
        n_epochs=4,            # PPO epochs per update
        
        # Discount and GAE - high gamma for long-horizon learning
        gamma=0.995,            # Discount factor (high for delayed rewards)
        gae_lambda=0.95,        # GAE lambda
        
        # PPO clipping
        clip_range=0.2,
        clip_range_vf=None,     # No value function clipping
        
        # Loss coefficients
        ent_coef=0.02,          # Entropy bonus for exploration
        vf_coef=0.5,            # Value function loss weight
        max_grad_norm=0.5,      # Gradient clipping
        
        # Network architecture
        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],        # Policy network
                "vf": [512, 512, 256]    # Value network (larger)
            },
            "activation_fn": torch.nn.ReLU,
        },
        
        tensorboard_log=f"./logs/ppo_{VERSION}/",
        verbose=1,
        seed=42,
    )
    
    print("  ✓ Model created")
    print(f"    Policy network: [256, 256]")
    print(f"    Value network: [512, 512, 256]")
    print(f"    Total parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}")
    
    # === CALLBACKS ===
    callbacks = [
        EvalCallback(
            eval_freq=100_000,
            n_eval=200,
            save_path="models/",
            training_vec_env=vec_env,
            version=VERSION,
        ),
        CheckpointCallback(
            save_freq=500_000,
            save_path="models/",
            version=VERSION,
            training_vec_env=vec_env,
        ),
    ]
    
    # === TRAIN ===
    print(f"\nStarting training for {TOTAL_STEPS:,} steps...\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # === SAVE FINAL MODEL ===
    model.save(f"models/ppo_pokemon_6v6_{VERSION}")
    vec_env.save(f"models/vec_normalize_pokemon_6v6_{VERSION}.pkl")
    
    # === PRINT RESULTS ===
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    eval_callback = callbacks[0]
    if eval_callback.results:
        print("\nWin Rate Progression vs SimpleHeuristics:")
        for r in eval_callback.results:
            marker = " ★" if r['win_rate'] >= eval_callback.best_win_rate else ""
            print(f"  Step {r['step']:>9,}: {r['win_rate']:>5.1%}  (reward: {r['avg_reward']:>6.3f}){marker}")
        
        print(f"\nBest win rate: {eval_callback.best_win_rate:.1%}")
    
    print(f"\nModels saved to: models/")
    print(f"  • ppo_pokemon_6v6_{VERSION}_best.zip")
    print(f"  • ppo_pokemon_6v6_{VERSION}_final.zip")
    
    return model, eval_callback.results


if __name__ == "__main__":
    import torch  # Import here for policy_kwargs
    model, results = train()