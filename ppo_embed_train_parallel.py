"""
ppo_v17_difflr.py — v17.2 Training with Differential Learning Rates + Enriched Features

Two key changes from v17.1:

1. Differential LR: 10-30x higher LR for embedding tables vs FC layers.
   v17.1's embedding norms were completely frozen at LR 3e-5.

2. Enriched features (623 -> 640 dims):
   - Per-Pokemon: +move_knowledge (how many moves revealed, 0-1) [30->31 floats]
   - Global: +bench_move_effectiveness (5 bench slots' best move eff vs opp) [55->60]
   - Global: speed_comparison changed from binary to continuous ratio

NOTE: Because observation size changed (623->640), this CANNOT load v17.1 models.
      Must train from scratch with --new flag.

Usage:
  python ppo_v17_difflr.py --new                     # Default settings
  python ppo_v17_difflr.py --new --embed_lr 1e-3     # More aggressive embedding LR
  python ppo_v17_difflr.py --new --gamma 0.99         # Test shorter discount horizon
"""

import os
import sys
import pickle
import math
import uuid
import multiprocessing
import numpy as np
from datetime import datetime
from typing import Callable, Optional
from functools import partial

import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sb3_contrib import MaskablePPO

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from RLPlayer6v6 import RL6v6PlayerV17
from embedding_extractor import V17FeatureExtractor
from wrappers import MaskablePokeEnvWrapper


# ============================================================================
# Configuration
# ============================================================================

BASE_PORT = 8000

def get_server_config(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://127.0.0.1:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

CONFIG = {
    "total_timesteps": 20_000_000,
    "n_envs": 4,

    # === Differential LR ===
    "base_lr": 3e-5,           # Final LR for FC layers, heads, final MLP
    "base_lr_initial": 5e-4,   # Starting LR for FC layers (Wang decay)
    "embed_lr": 3e-4,          # Final LR for embedding tables (10x base)
    "embed_lr_initial": 5e-3,  # Starting LR for embeddings (10x base initial)
    "use_lr_decay": True,      # Use Wang-style decay (True for fresh, False for resume)

    # === PPO hyperparameters ===
    "gamma": 0.9999,           # Override with --gamma flag for ablation
    "gae_lambda": 0.754,
    "clip_range": 0.08,
    "n_epochs": 4,
    "batch_size": 1024,
    "n_steps": 512,
    "max_grad_norm": 0.5,
    "ent_coef": 0.05,
    "vf_coef": 0.4375,

    # === Architecture (unchanged from v17.1) ===
    "features_dim": 256,
    "pokemon_repr_dim": 128,
    "embed_dropout": 0.05,
    "net_arch_pi": [256, 128],
    "net_arch_vf": [256, 128],
    "share_features_extractor": False,

    # === Evaluation ===
    "eval_freq": 100_000,
    "n_eval_episodes": 200,
    "save_freq": 200_000,

    # === Paths ===
    "data_dir": "data",
    "log_dir": "logs",
    "model_dir": "models",
    "version": "v17.2",
}


# ============================================================================
# Differential LR Optimizer Setup
# ============================================================================

# Names of embedding modules in the feature extractor
EMBEDDING_LAYER_NAMES = [
    "pokemon_embed",
    "move_embed",
    "ability_embed",
    "item_embed",
    "type_embed",
]


def is_embedding_param(name: str) -> bool:
    """Check if a parameter name belongs to an embedding table."""
    for embed_name in EMBEDDING_LAYER_NAMES:
        if embed_name in name and "weight" in name:
            return True
    return False


def setup_differential_optimizer(model: MaskablePPO, base_lr: float, embed_lr: float,
                                  preserve_adam_state: bool = True):
    """
    Replace the model's optimizer with one that uses different LRs for
    embedding parameters vs everything else.
    
    This must be called AFTER model creation but BEFORE training.
    
    When resuming from a checkpoint, set preserve_adam_state=True to copy
    Adam's momentum buffers (m, v) from the old single-group optimizer
    into the new two-group optimizer. This avoids a training disruption.
    
    Args:
        model: The MaskablePPO model
        base_lr: Learning rate for FC layers, heads, norms
        embed_lr: Learning rate for embedding tables (should be 10-30x base_lr)
        preserve_adam_state: If True, copy Adam state from old optimizer
    """
    policy = model.policy
    old_optimizer = policy.optimizer
    
    # Build a map from parameter id -> old optimizer state
    old_state_map = {}
    if preserve_adam_state and old_optimizer is not None:
        for group in old_optimizer.param_groups:
            for param in group["params"]:
                pid = id(param)
                if pid in old_optimizer.state:
                    old_state_map[pid] = old_optimizer.state[pid]
    
    embedding_params = []
    other_params = []
    embedding_param_names = []
    other_param_names = []
    
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if is_embedding_param(name):
            embedding_params.append(param)
            embedding_param_names.append(name)
        else:
            other_params.append(param)
            other_param_names.append(name)
    
    # Create new optimizer with parameter groups
    optimizer = torch.optim.Adam([
        {"params": embedding_params, "lr": embed_lr, "name": "embeddings"},
        {"params": other_params, "lr": base_lr, "name": "network"},
    ], eps=1e-5)
    
    # Restore Adam state (exp_avg, exp_avg_sq, step) from old optimizer
    restored = 0
    if old_state_map:
        for group in optimizer.param_groups:
            for param in group["params"]:
                pid = id(param)
                if pid in old_state_map:
                    optimizer.state[param] = old_state_map[pid]
                    restored += 1
    
    # Replace the optimizer in the policy
    policy.optimizer = optimizer
    
    # === Patch SB3's LR update to be a no-op ===
    # SB3's OnPolicyAlgorithm.train() calls:
    #   self._update_learning_rate(self.policy.optimizer)
    # which sets ALL param_group["lr"] to the same value.
    # We make this a no-op because DifferentialLRScheduleCallback
    # handles LR updates with proper per-group scheduling.
    
    def _patched_update_learning_rate(self_model, optimizers):
        """No-op: LR is managed by DifferentialLRScheduleCallback."""
        pass
    
    import types
    model._update_learning_rate = types.MethodType(
        _patched_update_learning_rate, model
    )
    print(f"  ✓ Patched model._update_learning_rate (no-op, callback controls LR)")
    
    # Print summary
    n_embed = sum(p.numel() for p in embedding_params)
    n_other = sum(p.numel() for p in other_params)
    print(f"\n{'='*60}")
    print(f"Differential Optimizer Setup")
    print(f"{'='*60}")
    print(f"  Embedding params: {n_embed:,} ({len(embedding_params)} tensors) @ LR {embed_lr:.1e}")
    for name in embedding_param_names:
        print(f"    - {name}")
    print(f"  Network params:   {n_other:,} ({len(other_params)} tensors) @ LR {base_lr:.1e}")
    print(f"  LR ratio: {embed_lr/base_lr:.0f}x")
    if old_state_map:
        print(f"  Adam state: restored {restored}/{len(old_state_map)} param states")
    else:
        print(f"  Adam state: fresh (no previous optimizer state)")
    print(f"{'='*60}\n")
    
    return optimizer


def update_differential_lr(optimizer: torch.optim.Adam, base_lr: float, embed_lr: float):
    """
    Update learning rates in the differential optimizer.
    Call this if you want to implement LR scheduling.
    """
    for param_group in optimizer.param_groups:
        if param_group.get("name") == "embeddings":
            param_group["lr"] = embed_lr
        else:
            param_group["lr"] = base_lr


def safe_save(model, path):
    """
    Save model without pickling SubprocVecEnv or logger tty handles.
    
    Python 3.14 is strict about pickling:
    - AuthenticationString in SubprocVecEnv processes
    - tty file objects in HumanOutputFormat logger
    
    We temporarily detach both, save, then restore.
    """
    # Ensure .zip extension (SB3 expects it for loading)
    if not path.endswith(".zip"):
        path = path + ".zip"
    
    env_backup = model.env
    logger_backup = model._logger
    model.env = None
    model._logger = None
    try:
        model.save(path)
    finally:
        model.env = env_backup
        model._logger = logger_backup


# ============================================================================
# Callbacks
# ============================================================================

class WinRateCallback(BaseCallback):
    """Track win rate during training."""
    def __init__(self, check_freq: int = 10000):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self.wins = 0
        self.losses = 0
        self._last_log_step = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                if info["episode"]["r"] > 0:
                    self.wins += 1
                else:
                    self.losses += 1

        if self.num_timesteps - self._last_log_step >= self.check_freq:
            self._last_log_step = self.num_timesteps
            total = self.wins + self.losses
            if total > 0:
                self.logger.record("train/win_rate", self.wins / total)
                self.logger.record("train/episodes", total)
            self.wins = 0
            self.losses = 0
        return True


class EnhancedEmbeddingStatsCallback(BaseCallback):
    """
    Enhanced embedding monitoring that tracks:
    - Frobenius norm (overall magnitude)
    - Gradient norm (how much they're actually being updated)
    - Per-step delta (actual movement between checkpoints)
    - Cosine similarity to initial embeddings (how much structure has changed)
    """
    def __init__(self, check_freq: int = 25000):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self._last_log_step = 0
        self._initial_norms = {}
        self._last_weights = {}
        self._initialized = False

    def _on_step(self):
        if self.num_timesteps - self._last_log_step >= self.check_freq:
            self._last_log_step = self.num_timesteps
            
            # Access both policy and value feature extractors
            # (share_features_extractor=False means separate extractors)
            for prefix, extractor in self._get_extractors():
                for name, embed in [
                    ("pokemon", extractor.pokemon_embed),
                    ("move", extractor.move_embed),
                    ("ability", extractor.ability_embed),
                    ("item", extractor.item_embed),
                    ("type", extractor.type_embed),
                ]:
                    key = f"{prefix}/{name}"
                    weight = embed.weight.data
                    
                    # Current norm
                    norm = weight.norm().item()
                    self.logger.record(f"embeddings/{key}_norm", norm)
                    
                    # Gradient norm (if available)
                    if embed.weight.grad is not None:
                        grad_norm = embed.weight.grad.norm().item()
                        self.logger.record(f"embeddings/{key}_grad_norm", grad_norm)
                    
                    # Delta from last checkpoint
                    if key in self._last_weights:
                        delta = (weight - self._last_weights[key]).norm().item()
                        self.logger.record(f"embeddings/{key}_delta", delta)
                    
                    # Store for next comparison
                    self._last_weights[key] = weight.clone()
                    
                    # Store initial norms
                    if not self._initialized:
                        self._initial_norms[key] = norm
                    
                    # Relative change from initial
                    if key in self._initial_norms and self._initial_norms[key] > 0:
                        rel_change = abs(norm - self._initial_norms[key]) / self._initial_norms[key]
                        self.logger.record(f"embeddings/{key}_rel_change", rel_change)
            
            if not self._initialized:
                self._initialized = True
                
            # Also log the actual LR being used for each param group
            optimizer = self.model.policy.optimizer
            for group in optimizer.param_groups:
                group_name = group.get("name", "unknown")
                self.logger.record(f"lr/{group_name}", group["lr"])
        
        return True
    
    def _get_extractors(self):
        """
        Get feature extractors from the policy.
        
        With share_features_extractor=False, SB3's MlpPolicy creates:
          - policy.pi_features_extractor (for policy network)
          - policy.vf_features_extractor (for value network) 
        With share_features_extractor=True:
          - policy.features_extractor (shared)
        
        We need to handle both cases and also the MaskablePPO variant
        which may use mlp_extractor differently.
        """
        policy = self.model.policy
        extractors = []
        
        # Check for separate pi/vf extractors first (share_features_extractor=False)
        if hasattr(policy, 'pi_features_extractor') and policy.pi_features_extractor is not None:
            extractors.append(("pi", policy.pi_features_extractor))
        if hasattr(policy, 'vf_features_extractor') and policy.vf_features_extractor is not None:
            extractors.append(("vf", policy.vf_features_extractor))
        
        # Fall back to shared extractor
        if not extractors and hasattr(policy, 'features_extractor'):
            extractors.append(("shared", policy.features_extractor))
        
        # Deduplicate by object identity (in case pi and vf point to same object)
        seen_ids = set()
        unique = []
        for prefix, ext in extractors:
            if id(ext) not in seen_ids:
                seen_ids.add(id(ext))
                unique.append((prefix, ext))
        
        return unique if unique else [("policy", policy.features_extractor)]


class DifferentialLRScheduleCallback(BaseCallback):
    """
    Manages differential LR scheduling with Wang-style decay.
    
    Both base_lr and embed_lr decay from initial to final values using
    Wang's aggressive schedule: lr(x) = final + (initial - final) / (8x + 1)^1.5
    
    The embed/base ratio is maintained throughout training.
    """
    def __init__(self, config: dict, total_timesteps: int):
        super().__init__(verbose=0)
        self.base_lr_initial = config.get("base_lr_initial", config["base_lr"])
        self.base_lr_final = config["base_lr"]
        self.embed_lr_initial = config.get("embed_lr_initial", config["embed_lr"])
        self.embed_lr_final = config["embed_lr"]
        self.use_decay = config.get("use_lr_decay", True)
        self.total_timesteps = total_timesteps
        self._last_log = 0
        self._fix_count = 0
    
    def _get_lrs(self) -> tuple:
        """Compute current base and embed LRs based on training progress."""
        if not self.use_decay:
            return self.base_lr_final, self.embed_lr_final
        
        # progress: 0.0 at start, 1.0 at end
        progress = self.num_timesteps / max(self.total_timesteps, 1)
        progress = min(progress, 1.0)
        
        # Wang decay: fast initial drop, then slow
        decay = 1.0 / (8.0 * progress + 1.0) ** 1.5
        
        base_lr = self.base_lr_final + (self.base_lr_initial - self.base_lr_final) * decay
        embed_lr = self.embed_lr_final + (self.embed_lr_initial - self.embed_lr_final) * decay
        
        return base_lr, embed_lr
    
    def _on_step(self):
        base_lr, embed_lr = self._get_lrs()
        
        # Apply to all param groups
        for group in self.model.policy.optimizer.param_groups:
            if group.get("name") == "embeddings":
                group["lr"] = embed_lr
            elif group.get("name") == "network":
                group["lr"] = base_lr
        
        # Log actual LRs periodically
        if self.num_timesteps - self._last_log >= 50000:
            self._last_log = self.num_timesteps
            for group in self.model.policy.optimizer.param_groups:
                name = group.get("name", "default")
                self.logger.record(f"lr/{name}", group["lr"])
        
        return True


class PeriodicSaveCallback(BaseCallback):
    """Save model + training state + optimizer state periodically."""
    def __init__(self, save_freq, save_path, version, eval_callback_ref=None):
        super().__init__(verbose=0)
        self.save_freq = save_freq
        self.save_path = save_path
        self.version = version
        self.eval_callback_ref = eval_callback_ref
        self._last_save_step = 0

    def _on_step(self):
        if self.num_timesteps - self._last_save_step >= self.save_freq and self.num_timesteps > 0:
            self._last_save_step = self.num_timesteps
            
            latest_path = f"{self.save_path}/ppo_pokemon_6v6_{self.version}_latest"
            safe_save(self.model, latest_path)
            
            # Save optimizer state separately (SB3 doesn't save param group metadata)
            optim_state_path = f"{self.save_path}/optimizer_state_{self.version}.pt"
            torch.save(self.model.policy.optimizer.state_dict(), optim_state_path)
            
            state = {
                'timesteps': self.num_timesteps,
                'eval_state': self.eval_callback_ref.get_state() if self.eval_callback_ref else {},
                'saved_at': datetime.now().isoformat(),
            }
            state_path = f"{self.save_path}/training_state_{self.version}.pkl"
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
            
            print(f"\n>>> Checkpoint @ {self.num_timesteps:,} steps (model + optimizer state)")
        return True


class EvalWinRateCallback(BaseCallback):
    """Evaluate against SimpleHeuristics and save best model."""
    def __init__(self, eval_env, eval_freq=100_000, n_eval=200,
                 save_path="models", version="v17.2"):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.save_path = save_path
        self.version = version
        self.best_win_rate = 0.0
        self.results = []
        self._last_eval_step = 0

    def _on_step(self):
        current_step = self.num_timesteps
        if current_step - self._last_eval_step >= self.eval_freq and current_step > 0:
            self._last_eval_step = current_step
            wins = 0
            total_reward = 0

            print(f"\n>>> Eval ({self.n_eval} games)...")
            for ep in range(self.n_eval):
                obs = self.eval_env.reset()
                done = False
                ep_reward = 0
                while not done:
                    masks = self.eval_env.env_method("action_masks")[0]
                    action, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
                    obs, reward, done, info = self.eval_env.step(action)
                    ep_reward += reward[0]
                    done = done[0]
                total_reward += ep_reward
                if ep_reward > 0:
                    wins += 1
                if (ep + 1) % 50 == 0:
                    print(f"    {ep + 1}/{self.n_eval}...")

            win_rate = wins / self.n_eval
            avg_reward = total_reward / self.n_eval

            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/avg_reward", avg_reward)
            self.logger.record("eval/best_win_rate", self.best_win_rate)
            self.results.append({
                'step': current_step, 'win_rate': win_rate, 'avg_reward': avg_reward,
            })

            marker = ""
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                safe_save(self.model, f"{self.save_path}/ppo_pokemon_6v6_{self.version}_best")
                marker = " ★ NEW BEST"

            print(f"\n{'='*60}")
            print(f">>> Eval @ {current_step:,} steps")
            print(f"    Win rate: {win_rate:.1%} ({wins}/{self.n_eval}){marker}")
            print(f"    Avg reward: {avg_reward:.3f}")
            print(f"    Best: {self.best_win_rate:.1%}")
            print(f"{'='*60}\n")
        return True

    def get_state(self):
        return {
            'best_win_rate': self.best_win_rate,
            'results': self.results,
            'last_eval_step': self._last_eval_step,
        }

    def set_state(self, state):
        self.best_win_rate = state.get('best_win_rate', 0.0)
        self.results = state.get('results', [])
        self._last_eval_step = state.get('last_eval_step', 0)


# ============================================================================
# Environment (same as v17 training)
# ============================================================================

def make_env(env_idx: int, config: dict, is_eval: bool = False):
    def _init():
        port = BASE_PORT + env_idx
        server_config = get_server_config(port)
        uid = uuid.uuid4().hex[:6]

        rl_env = RL6v6PlayerV17(
            battle_format="gen9randombattle",
            data_dir=config["data_dir"],
            server_configuration=server_config,
            account_configuration1=AccountConfiguration.generate(f"Bot{env_idx}-{uid}", rand=True),
            account_configuration2=AccountConfiguration.generate(f"Opp{env_idx}-{uid}", rand=True),
        )

        opponent = SimpleHeuristicsPlayer(
            battle_format="gen9randombattle",
            server_configuration=server_config,
            start_listening=False,
        )

        wrapped = SingleAgentWrapper(rl_env, opponent)
        env = MaskablePokeEnvWrapper(wrapped, rl_env)
        env = TimeLimit(env, max_episode_steps=500)
        env = Monitor(env)
        return env
    return _init


def create_training_env(config):
    n_envs = config["n_envs"]
    if n_envs == 1:
        return DummyVecEnv([make_env(0, config)])
    
    if sys.platform == "darwin":
        try:
            multiprocessing.set_start_method("forkserver", force=True)
        except RuntimeError:
            pass
    
    return SubprocVecEnv(
        [make_env(i, config) for i in range(n_envs)],
        start_method="forkserver" if sys.platform == "darwin" else "fork",
    )


def create_eval_env(config):
    return DummyVecEnv([make_env(config["n_envs"], config, is_eval=True)])


# ============================================================================
# Main Training
# ============================================================================

def _get_all_extractors(model):
    """Get all feature extractors from model, handling shared/separate cases."""
    policy = model.policy
    extractors = []
    
    if hasattr(policy, 'pi_features_extractor') and policy.pi_features_extractor is not None:
        extractors.append(("pi", policy.pi_features_extractor))
    if hasattr(policy, 'vf_features_extractor') and policy.vf_features_extractor is not None:
        extractors.append(("vf", policy.vf_features_extractor))
    
    if not extractors and hasattr(policy, 'features_extractor'):
        extractors.append(("shared", policy.features_extractor))
    
    # Deduplicate
    seen = set()
    unique = []
    for prefix, ext in extractors:
        if id(ext) not in seen:
            seen.add(id(ext))
            unique.append((prefix, ext))
    
    return unique if unique else [("policy", policy.features_extractor)]


def train(config=None, force_new=False, from_model=None):
    if config is None:
        config = CONFIG.copy()
    
    version = config["version"]
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)

    base_lr = config["base_lr"]
    embed_lr = config["embed_lr"]

    print("=" * 70)
    print(f"Pokemon RL Bot {version} — Differential LR + Enriched Features (640-dim)")
    print("=" * 70)
    print(f"\n  Training from scratch (obs size changed: 623 -> 640)")
    print(f"  Embedding LR: {config.get('embed_lr_initial', embed_lr):.1e} → {embed_lr:.1e} (Wang decay)")
    print(f"  Network LR:   {config.get('base_lr_initial', base_lr):.1e} → {base_lr:.1e} (Wang decay)")
    print(f"  LR ratio:     {embed_lr/base_lr:.0f}x (maintained throughout)")
    print(f"  Gamma:         {config['gamma']}")
    print(f"  Opponent:      SimpleHeuristics (fixed)")
    print(f"  Target:        {config['total_timesteps']:,} steps")
    print(f"\n  New features vs v17.1:")
    print(f"    - move_knowledge: fraction of moves revealed per Pokemon")
    print(f"    - speed_ratio: continuous effective speed ratio (was binary)")
    print(f"    - bench_move_eff: best move effectiveness per bench slot vs opp")
    print(f"\n  Key diagnostic: Watch embeddings/*_delta in TensorBoard.\n")

    # Check for resume
    state_path = f"{config['model_dir']}/training_state_{version}.pkl"
    latest_path = f"{config['model_dir']}/ppo_pokemon_6v6_{version}_latest"
    latest_exists = (os.path.exists(latest_path + ".zip") or os.path.exists(latest_path))
    resuming = (not force_new 
                and os.path.exists(state_path) 
                and latest_exists)
    
    # Resolve actual path for loading
    if os.path.exists(latest_path + ".zip"):
        latest_load_path = latest_path  # SB3 will find the .zip
    elif os.path.exists(latest_path):
        latest_load_path = latest_path  # SB3 will find it without .zip
    else:
        latest_load_path = latest_path

    # Create environments
    print("Creating environments...")
    train_env = create_training_env(config)
    eval_env = create_eval_env(config)
    print(f"  Training: {config['n_envs']} envs (ports {BASE_PORT}-{BASE_PORT + config['n_envs'] - 1})")
    print(f"  Eval: port {BASE_PORT + config['n_envs']}")

    # Load or create model
    model_path = from_model or config.get("base_model")
    
    if resuming and not from_model:
        # Resume a v17.2 training run
        # We need to handle the optimizer state carefully because safe_save
        # saves a 2-group optimizer (from differential LR), but SB3's load()
        # creates a fresh 1-group optimizer then tries to load the 2-group state.
        # Solution: Create a fresh model with matching architecture, then load
        # only the policy weights (not optimizer) from the saved file.
        print(f"\nResuming from {latest_load_path}...")
        
        # First create a fresh model with correct architecture
        policy_kwargs = {
            "features_extractor_class": V17FeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": config["features_dim"],
                "config_path": f"{config['data_dir']}/embedding_config.json",
                "pokemon_repr_dim": config.get("pokemon_repr_dim", 128),
                "embed_dropout": config["embed_dropout"],
            },
            "net_arch": dict(
                pi=config["net_arch_pi"],
                vf=config["net_arch_vf"],
            ),
            "share_features_extractor": config.get("share_features_extractor", False),
            "activation_fn": torch.nn.ReLU,
        }
        
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=base_lr,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config["log_dir"],
        )
        
        # Load saved weights into the fresh model (skip optimizer)
        import zipfile, io
        save_path = latest_load_path
        if not save_path.endswith(".zip"):
            if os.path.exists(save_path + ".zip"):
                save_path = save_path + ".zip"
        
        with zipfile.ZipFile(save_path, "r") as zip_f:
            # Load policy state dict
            with zip_f.open("policy.pth") as f:
                buffer = io.BytesIO(f.read())
                policy_state = torch.load(buffer, map_location="cpu")
            # Load other data (num_timesteps, etc.)
            if "data" in zip_f.namelist():
                with zip_f.open("data") as f:
                    import json as json_mod
                    data = json_mod.loads(f.read().decode())
        
        # Restore policy weights
        model.policy.load_state_dict(policy_state)
        
        # Restore num_timesteps from training state
        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)
        saved_timesteps = saved_state.get('timesteps', 0)
        model.num_timesteps = saved_timesteps
        model._num_timesteps_at_start = saved_timesteps
        
        print(f"  ✓ Loaded policy weights from {save_path}")
        print(f"  ✓ Restored timestep count: {saved_timesteps:,}")
    elif from_model:
        # Load a specific model (must have matching observation size!)
        # Same approach as resume — create fresh model, load weights only
        print(f"\nLoading model from {model_path}...")
        
        policy_kwargs = {
            "features_extractor_class": V17FeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": config["features_dim"],
                "config_path": f"{config['data_dir']}/embedding_config.json",
                "pokemon_repr_dim": config.get("pokemon_repr_dim", 128),
                "embed_dropout": config["embed_dropout"],
            },
            "net_arch": dict(
                pi=config["net_arch_pi"],
                vf=config["net_arch_vf"],
            ),
            "share_features_extractor": config.get("share_features_extractor", False),
            "activation_fn": torch.nn.ReLU,
        }
        
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=base_lr,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config["log_dir"],
        )
        
        load_path = model_path
        if not load_path.endswith(".zip"):
            if os.path.exists(load_path + ".zip"):
                load_path = load_path + ".zip"
        
        import zipfile, io
        with zipfile.ZipFile(load_path, "r") as zip_f:
            with zip_f.open("policy.pth") as f:
                buffer = io.BytesIO(f.read())
                policy_state = torch.load(buffer, map_location="cpu")
        model.policy.load_state_dict(policy_state)
        print(f"  ✓ Loaded policy weights from {load_path}")
    else:
        # Fresh training — create new model from scratch
        print(f"\nCreating new model from scratch (640-dim observations)...")
        policy_kwargs = {
            "features_extractor_class": V17FeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": config["features_dim"],
                "config_path": f"{config['data_dir']}/embedding_config.json",
                "pokemon_repr_dim": config.get("pokemon_repr_dim", 128),
                "embed_dropout": config["embed_dropout"],
            },
            "net_arch": dict(
                pi=config["net_arch_pi"],
                vf=config["net_arch_vf"],
            ),
            "share_features_extractor": config.get("share_features_extractor", False),
            "activation_fn": torch.nn.ReLU,
        }

        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=base_lr,  # Placeholder — will be overridden
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config["log_dir"],
        )

    # === THE KEY CHANGE: Replace optimizer with differential LR ===
    optimizer = setup_differential_optimizer(model, base_lr, embed_lr)
    
    # Restore optimizer state if resuming (preserves Adam momentum/variance)
    optim_state_path = f"{config['model_dir']}/optimizer_state_{version}.pt"
    if resuming and not from_model and os.path.exists(optim_state_path):
        try:
            saved_optim_state = torch.load(optim_state_path, map_location="cpu")
            # Validate structure matches (same number of param groups and params)
            if (len(saved_optim_state["state"]) > 0 and
                len(saved_optim_state["param_groups"]) == len(optimizer.param_groups)):
                # Restore state but keep our new LRs
                current_lrs = [(g["name"], g["lr"]) for g in optimizer.param_groups]
                optimizer.load_state_dict(saved_optim_state)
                # Re-apply our LRs (load_state_dict overwrites them)
                for group, (name, lr) in zip(optimizer.param_groups, current_lrs):
                    group["lr"] = lr
                    group["name"] = name
                print(f"  ✓ Restored optimizer state (Adam momentum/variance preserved)")
            else:
                print(f"  ⚠ Optimizer state structure mismatch, starting fresh optimizer")
        except Exception as e:
            print(f"  ⚠ Could not restore optimizer state: {e}")
    elif resuming and not from_model:
        print(f"  ⚠ No saved optimizer state found, Adam momentum will cold-start")

    # Print model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Total parameters: {total_params:,}")

    # Snapshot initial embedding norms for comparison
    print("\nInitial embedding norms:")
    for prefix, extractor in _get_all_extractors(model):
        for name, embed in [("pokemon", extractor.pokemon_embed),
                            ("move", extractor.move_embed),
                            ("ability", extractor.ability_embed),
                            ("item", extractor.item_embed),
                            ("type", extractor.type_embed)]:
            norm = embed.weight.data.norm().item()
            print(f"  {prefix}/{name}: {norm:.4f}")

    # Setup callbacks
    eval_callback = EvalWinRateCallback(
        eval_env=eval_env,
        eval_freq=config["eval_freq"],
        n_eval=config["n_eval_episodes"],
        save_path=config["model_dir"],
        version=version,
    )

    # Restore eval state if resuming
    if resuming and not from_model:
        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)
        eval_callback.set_state(saved_state.get('eval_state', {}))
        start_timesteps = saved_state.get('timesteps', 0)
        remaining = config["total_timesteps"] - start_timesteps
        print(f"Restored: {start_timesteps:,} steps done, {remaining:,} remaining")
        print(f"  Best: {eval_callback.best_win_rate:.1%}")
    else:
        start_timesteps = 0
        remaining = config["total_timesteps"]

    callbacks = CallbackList([
        WinRateCallback(check_freq=10000),
        EnhancedEmbeddingStatsCallback(check_freq=25000),
        # IMPORTANT: total_timesteps must be the GLOBAL total (30M), not remaining.
        # num_timesteps in callbacks reflects global count when reset_num_timesteps=False.
        DifferentialLRScheduleCallback(config=config, total_timesteps=config["total_timesteps"]),
        PeriodicSaveCallback(
            save_freq=config["save_freq"],
            save_path=config["model_dir"],
            version=version,
            eval_callback_ref=eval_callback,
        ),
        eval_callback,
    ])

    # Train
    print(f"\nTraining for {remaining:,} steps...")
    print("=" * 70)

    interrupted = False
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(not resuming or from_model is not None),
        )
    except KeyboardInterrupt:
        print("\n>>> Paused (Ctrl+C)")
        interrupted = True

    # Save
    actual = model.num_timesteps
    safe_save(model, f"{config['model_dir']}/ppo_pokemon_6v6_{version}_latest")
    
    # Save optimizer state for resume
    optim_state_path = f"{config['model_dir']}/optimizer_state_{version}.pt"
    torch.save(model.policy.optimizer.state_dict(), optim_state_path)
    
    state = {
        'timesteps': actual,
        'eval_state': eval_callback.get_state(),
        'saved_at': datetime.now().isoformat(),
        'config': {k: v for k, v in config.items() if not callable(v)},
        # Save LR config so resume uses the same differential LRs
        'base_lr': base_lr,
        'embed_lr': embed_lr,
    }
    with open(f"{config['model_dir']}/training_state_{version}.pkl", 'wb') as f:
        pickle.dump(state, f)

    if not interrupted:
        safe_save(model, f"{config['model_dir']}/ppo_pokemon_6v6_{version}")

    # Final embedding norms
    print("\n" + "=" * 70)
    print("Final embedding norms:")
    for prefix, extractor in _get_all_extractors(model):
        for name, embed in [("pokemon", extractor.pokemon_embed),
                            ("move", extractor.move_embed),
                            ("ability", extractor.ability_embed),
                            ("item", extractor.item_embed),
                            ("type", extractor.type_embed)]:
            norm = embed.weight.data.norm().item()
            print(f"  {prefix}/{name}: {norm:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if eval_callback.results:
        for r in eval_callback.results:
            m = " ★" if r['win_rate'] >= eval_callback.best_win_rate else ""
            print(f"  {r['step']:>9,}: {r['win_rate']:>5.1%}  (rew: {r['avg_reward']:>6.3f}){m}")
        print(f"\nBest: {eval_callback.best_win_rate:.1%}")
        print(f"v17.1 baseline: 69.5% peak / ~60% smoothed")
        print(f"v13 baseline:   72%")

    if interrupted:
        print(f"\nResume: python ppo_v17_difflr.py")
        print(f"Fresh:  python ppo_v17_difflr.py --new")

    try:
        train_env.close()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    try:
        eval_env.close()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    return model


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v17.2 Training with Differential LR")
    parser.add_argument("--new", action="store_true", help="Start fresh (don't resume)")
    parser.add_argument("--from_model", type=str, default=None,
                        help="Path to model to start from (without .zip)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps")
    parser.add_argument("--embed_lr", type=float, default=None,
                        help="Embedding learning rate (default: 3e-4)")
    parser.add_argument("--base_lr", type=float, default=None,
                        help="Base learning rate (default: 3e-5)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor (default: 0.9999, try 0.99)")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Path to model to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        config = CONFIG.copy()
        eval_env = create_eval_env(config)
        model = MaskablePPO.load(args.evaluate, env=eval_env)
        wins = 0
        for ep in range(200):
            obs = eval_env.reset()
            done = False
            while not done:
                masks = eval_env.env_method("action_masks")[0]
                action, _ = model.predict(obs, deterministic=True, action_masks=masks)
                obs, reward, done, _ = eval_env.step(action)
                done = done[0]
            if reward[0] > 0:
                wins += 1
            if (ep + 1) % 20 == 0:
                print(f"{ep+1}/200: {wins/(ep+1):.1%}")
        print(f"\nFinal: {wins/200:.1%}")
        eval_env.close()
    else:
        config = CONFIG.copy()
        if args.timesteps:
            config["total_timesteps"] = args.timesteps
        if args.embed_lr:
            config["embed_lr"] = args.embed_lr
        if args.base_lr:
            config["base_lr"] = args.base_lr
        if args.gamma:
            config["gamma"] = args.gamma
        
        train(config, force_new=args.new, from_model=args.from_model)