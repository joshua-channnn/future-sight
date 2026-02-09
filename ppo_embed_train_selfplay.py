"""
ppo_v17_selfplay.py — v17.3 Self-Play Training

Architecture: v17.3 (max+mean pooling, 700-dim final MLP input)
Observation: 640 dims (unchanged from v17.2)
Warm-start: Embeddings + encoder + matchup head from v17.2 best model

Self-play design (inspired by VGC-Bench fictitious play + Huang & Lee):
- Checkpoint pool: Save snapshot every POOL_SAVE_FREQ steps
- Opponent sampling: Mix of SimpleHeuristics + pool checkpoints
- Schedule: Start 80% heuristic / 20% self-play, shift to 50/50
- Fixed eval against SimpleHeuristics as stable benchmark

Usage:
  python ppo_v17_selfplay.py --base_model models/ppo_pokemon_6v6_v17.2_best
  python ppo_v17_selfplay.py --base_model models/ppo_pokemon_6v6_v17.2_best --timesteps 30000000
"""

import os
import sys
import pickle
import math
import uuid
import copy
import random
import multiprocessing
import shutil
import numpy as np
from datetime import datetime
from typing import Callable, Optional, List
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
from embedding_extractor_sp import V17SPFeatureExtractor
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

# Embedding layer names (same as v17.2 — used for differential LR)
EMBEDDING_LAYER_NAMES = [
    "pokemon_embed",
    "move_embed",
    "ability_embed",
    "item_embed",
    "type_embed",
]

CONFIG = {
    "total_timesteps": 20_000_000,
    "n_envs": 4,

    # === Differential LR (same as v17.2) ===
    "base_lr": 3e-5,
    "base_lr_initial": 3e-4,       # Lower initial than v17.2 since weights are pretrained
    "embed_lr": 3e-4,
    "embed_lr_initial": 3e-3,      # Lower initial — embeddings are already trained
    "use_lr_decay": True,

    # === PPO hyperparameters (unchanged) ===
    "gamma": 0.9999,
    "gae_lambda": 0.754,
    "clip_range": 0.08,
    "n_epochs": 4,
    "batch_size": 1024,
    "n_steps": 512,
    "max_grad_norm": 0.5,
    "ent_coef": 0.06,              # Slightly higher for self-play diversity
    "vf_coef": 0.4375,

    # === Architecture (v17.3 — max+mean pooling) ===
    "features_dim": 256,
    "pokemon_repr_dim": 128,
    "embed_dropout": 0.05,
    "net_arch_pi": [256, 128],
    "net_arch_vf": [256, 128],
    "share_features_extractor": False,

    # === Self-play ===
    "selfplay_ratio_initial": 1.0,    # 100% self-play from the start (Huang & Lee, Wang)
    "selfplay_ratio_final": 1.0,      # Stay at 100%
    "pool_save_freq": 1_000_000,      # Save checkpoint every 1M steps (faster pool buildup)
    "max_pool_size": 10,              # Keep at most 10 checkpoints in pool

    # === Evaluation ===
    "eval_freq": 100_000,
    "n_eval_episodes": 200,
    "save_freq": 200_000,

    # === Paths ===
    "data_dir": "data",
    "log_dir": "logs",
    "model_dir": "models",
    "pool_dir": "models/selfplay_pool",
    "version": "v17.3sp",
}


# ============================================================================
# Differential LR (same as v17.2)
# ============================================================================

def is_embedding_param(name: str) -> bool:
    for embed_name in EMBEDDING_LAYER_NAMES:
        if embed_name in name and "weight" in name:
            return True
    return False


def setup_differential_optimizer(model, base_lr, embed_lr):
    policy = model.policy
    embedding_params, other_params = [], []
    embedding_names, other_names = [], []

    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if is_embedding_param(name):
            embedding_params.append(param)
            embedding_names.append(name)
        else:
            other_params.append(param)
            other_names.append(name)

    optimizer = torch.optim.Adam([
        {"params": embedding_params, "lr": embed_lr, "name": "embeddings"},
        {"params": other_params, "lr": base_lr, "name": "network"},
    ], eps=1e-5)

    policy.optimizer = optimizer

    # Patch SB3's LR update to no-op
    import types
    def _patched_update_learning_rate(self_model, optimizers):
        pass
    model._update_learning_rate = types.MethodType(_patched_update_learning_rate, model)

    n_embed = sum(p.numel() for p in embedding_params)
    n_other = sum(p.numel() for p in other_params)
    print(f"\n{'='*60}")
    print(f"Differential Optimizer Setup")
    print(f"{'='*60}")
    print(f"  Embedding params: {n_embed:,} ({len(embedding_params)} tensors) @ LR {embed_lr:.1e}")
    print(f"  Network params:   {n_other:,} ({len(other_params)} tensors) @ LR {base_lr:.1e}")
    print(f"  LR ratio: {embed_lr/base_lr:.0f}x")
    print(f"{'='*60}\n")
    return optimizer


def safe_save(model, path):
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
# Warm-start: Load v17.2 weights into v17.3 architecture
# ============================================================================

def warm_start_from_v17_2(model, v17_2_path: str):
    """
    Transfer weights from v17.2 model into v17.3 (max+mean pooling).

    What transfers (exact match):
    - All 5 embedding tables (pokemon, move, ability, item, type)
    - SharedPokemonEncoder MLP
    - ActiveMatchupHead MLP

    What does NOT transfer (different shapes):
    - final_mlp (input changed: 444 → 700)
    - Policy/value heads downstream of features_dim are fine (256 → 256)

    We load the v17.2 state dict and do a partial match.
    """
    import zipfile, io

    load_path = v17_2_path
    if not load_path.endswith(".zip"):
        if os.path.exists(load_path + ".zip"):
            load_path = load_path + ".zip"

    print(f"  Loading v17.2 weights from {load_path}...")

    with zipfile.ZipFile(load_path, "r") as zip_f:
        with zip_f.open("policy.pth") as f:
            buffer = io.BytesIO(f.read())
            v17_2_state = torch.load(buffer, map_location="cpu", weights_only=False)

    # Get current model state
    current_state = model.policy.state_dict()

    transferred = 0
    skipped = 0
    skipped_names = []

    for name, param in v17_2_state.items():
        if name in current_state:
            if current_state[name].shape == param.shape:
                current_state[name] = param
                transferred += 1
            else:
                skipped += 1
                skipped_names.append(
                    f"    {name}: {param.shape} → {current_state[name].shape}"
                )
        else:
            skipped += 1
            skipped_names.append(f"    {name}: not found in v17.3")

    model.policy.load_state_dict(current_state)

    print(f"  ✓ Transferred {transferred} parameter tensors from v17.2")
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} tensors (shape mismatch or missing):")
        for s in skipped_names:
            print(s)

    return transferred, skipped


# ============================================================================
# Self-Play Opponent Manager
# ============================================================================

class SelfPlayOpponentManager:
    """
    Manages a pool of past model checkpoints for self-play training.

    The opponent for each game is sampled from:
    - SimpleHeuristics (probability = 1 - selfplay_ratio)
    - Random checkpoint from pool (probability = selfplay_ratio)

    The selfplay_ratio increases linearly from initial to final over training.
    """

    def __init__(self, config: dict):
        self.pool_dir = config["pool_dir"]
        self.max_pool_size = config["max_pool_size"]
        self.selfplay_ratio_initial = config["selfplay_ratio_initial"]
        self.selfplay_ratio_final = config["selfplay_ratio_final"]
        self.total_timesteps = config["total_timesteps"]
        self._current_timestep_estimate = 0  # Updated by callback

        os.makedirs(self.pool_dir, exist_ok=True)
        self.pool_paths: List[str] = []
        self._scan_existing_pool()

    def _scan_existing_pool(self):
        """Load/reload checkpoint paths from pool directory."""
        self.pool_paths = []
        if os.path.exists(self.pool_dir):
            for f in sorted(os.listdir(self.pool_dir)):
                if f.endswith(".zip"):
                    self.pool_paths.append(os.path.join(self.pool_dir, f))
        if self.pool_paths:
            print(f"  Found {len(self.pool_paths)} checkpoints in pool")

    def save_checkpoint(self, model, timestep: int):
        """Save a model checkpoint to the pool."""
        name = f"checkpoint_{timestep:010d}.zip"
        path = os.path.join(self.pool_dir, name)
        safe_save(model, path)

        self.pool_paths.append(path)
        print(f"  → Saved self-play checkpoint: {name} (pool size: {len(self.pool_paths)})")

        # Evict oldest if pool is full
        while len(self.pool_paths) > self.max_pool_size:
            oldest = self.pool_paths.pop(0)
            try:
                os.remove(oldest)
                print(f"  → Evicted oldest checkpoint: {os.path.basename(oldest)}")
            except OSError:
                pass

    def get_selfplay_ratio(self, timestep: int) -> float:
        """Get current self-play ratio based on training progress."""
        progress = min(timestep / max(self.total_timesteps, 1), 1.0)
        return (self.selfplay_ratio_initial +
                (self.selfplay_ratio_final - self.selfplay_ratio_initial) * progress)

    def should_use_selfplay(self, timestep: int) -> bool:
        """Decide whether this game should use a self-play opponent."""
        if not self.pool_paths:
            return False
        ratio = self.get_selfplay_ratio(timestep)
        return random.random() < ratio

    def get_random_checkpoint_path(self) -> Optional[str]:
        """Get a random checkpoint path from the pool."""
        if not self.pool_paths:
            return None
        return random.choice(self.pool_paths)

    def get_pool_size(self) -> int:
        return len(self.pool_paths)


# ============================================================================
# Self-Play Environment
# ============================================================================

# ============================================================================
# Callbacks
# ============================================================================

class WinRateCallback(BaseCallback):
    def __init__(self, check_freq=10000):
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
    def __init__(self, check_freq=25000):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self._last_log_step = 0
        self._initial_norms = {}
        self._last_weights = {}
        self._initialized = False

    def _on_step(self):
        if self.num_timesteps - self._last_log_step >= self.check_freq:
            self._last_log_step = self.num_timesteps
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
                    norm = weight.norm().item()
                    self.logger.record(f"embeddings/{key}_norm", norm)
                    if embed.weight.grad is not None:
                        self.logger.record(f"embeddings/{key}_grad_norm",
                                           embed.weight.grad.norm().item())
                    if key in self._last_weights:
                        delta = (weight - self._last_weights[key]).norm().item()
                        self.logger.record(f"embeddings/{key}_delta", delta)
                    self._last_weights[key] = weight.clone()
                    if not self._initialized:
                        self._initial_norms[key] = norm
                    if key in self._initial_norms and self._initial_norms[key] > 0:
                        rel = abs(norm - self._initial_norms[key]) / self._initial_norms[key]
                        self.logger.record(f"embeddings/{key}_rel_change", rel)
            if not self._initialized:
                self._initialized = True
            for group in self.model.policy.optimizer.param_groups:
                self.logger.record(f"lr/{group.get('name', 'unknown')}", group["lr"])
        return True

    def _get_extractors(self):
        policy = self.model.policy
        extractors = []
        if hasattr(policy, 'pi_features_extractor') and policy.pi_features_extractor is not None:
            extractors.append(("pi", policy.pi_features_extractor))
        if hasattr(policy, 'vf_features_extractor') and policy.vf_features_extractor is not None:
            extractors.append(("vf", policy.vf_features_extractor))
        if not extractors and hasattr(policy, 'features_extractor'):
            extractors.append(("shared", policy.features_extractor))
        seen = set()
        unique = []
        for p, e in extractors:
            if id(e) not in seen:
                seen.add(id(e))
                unique.append((p, e))
        return unique if unique else [("policy", policy.features_extractor)]


class DifferentialLRScheduleCallback(BaseCallback):
    def __init__(self, config: dict, total_timesteps: int):
        super().__init__(verbose=0)
        self.base_lr_initial = config.get("base_lr_initial", config["base_lr"])
        self.base_lr_final = config["base_lr"]
        self.embed_lr_initial = config.get("embed_lr_initial", config["embed_lr"])
        self.embed_lr_final = config["embed_lr"]
        self.use_decay = config.get("use_lr_decay", True)
        self.total_timesteps = total_timesteps
        self._last_log = 0

    def _get_lrs(self):
        if not self.use_decay:
            return self.base_lr_final, self.embed_lr_final
        progress = min(self.num_timesteps / max(self.total_timesteps, 1), 1.0)
        decay = 1.0 / (8.0 * progress + 1.0) ** 1.5
        base_lr = self.base_lr_final + (self.base_lr_initial - self.base_lr_final) * decay
        embed_lr = self.embed_lr_final + (self.embed_lr_initial - self.embed_lr_final) * decay
        return base_lr, embed_lr

    def _on_step(self):
        base_lr, embed_lr = self._get_lrs()
        for group in self.model.policy.optimizer.param_groups:
            if group.get("name") == "embeddings":
                group["lr"] = embed_lr
            elif group.get("name") == "network":
                group["lr"] = base_lr
        if self.num_timesteps - self._last_log >= 50000:
            self._last_log = self.num_timesteps
            for group in self.model.policy.optimizer.param_groups:
                self.logger.record(f"lr/{group.get('name', 'default')}", group["lr"])
        return True


class SelfPlayPoolCallback(BaseCallback):
    """
    Save checkpoints to the self-play pool and sync state to subprocesses.

    - Saves model checkpoints every pool_save_freq steps
    - Updates timestep in each subprocess env (for opponent scheduling)
    - Triggers pool refresh in subprocesses after saving new checkpoint
    """
    def __init__(self, opponent_manager: SelfPlayOpponentManager, save_freq: int):
        super().__init__(verbose=0)
        self.opponent_manager = opponent_manager
        self.save_freq = save_freq
        self._last_save = 0
        self._last_log = 0
        self._last_timestep_sync = 0
        self._timestep_sync_freq = 10000  # Sync timestep every 10K steps

    def _on_step(self):
        # Update timestep in main process manager
        self.opponent_manager._current_timestep_estimate = self.num_timesteps

        # Periodically sync timestep to subprocess environments
        if self.num_timesteps - self._last_timestep_sync >= self._timestep_sync_freq:
            self._last_timestep_sync = self.num_timesteps
            try:
                self.training_env.env_method("set_timestep", self.num_timesteps)
            except (AttributeError, Exception):
                pass  # DummyVecEnv or env doesn't support env_method

        # Save checkpoint to pool periodically
        if self.num_timesteps - self._last_save >= self.save_freq and self.num_timesteps > 0:
            self._last_save = self.num_timesteps
            self.opponent_manager.save_checkpoint(self.model, self.num_timesteps)

            # Tell subprocess environments to rescan the pool directory
            try:
                self.training_env.env_method("refresh_pool")
            except (AttributeError, Exception):
                pass

        # Log self-play stats periodically
        if self.num_timesteps - self._last_log >= 50000:
            self._last_log = self.num_timesteps
            ratio = self.opponent_manager.get_selfplay_ratio(self.num_timesteps)
            self.logger.record("selfplay/target_ratio", ratio)
            self.logger.record("selfplay/pool_size", self.opponent_manager.get_pool_size())

            # Try to get actual self-play usage stats from envs
            try:
                stats = self.training_env.env_method("get_selfplay_stats")
                total_games = sum(s['games'] for s in stats)
                total_sp = sum(s['selfplay_games'] for s in stats)
                if total_games > 0:
                    self.logger.record("selfplay/actual_ratio", total_sp / total_games)
                    self.logger.record("selfplay/total_games", total_games)
            except (AttributeError, Exception):
                pass

        return True


class PeriodicSaveCallback(BaseCallback):
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
            safe_save(self.model, f"{self.save_path}/ppo_pokemon_6v6_{self.version}_latest")
            torch.save(self.model.policy.optimizer.state_dict(),
                       f"{self.save_path}/optimizer_state_{self.version}.pt")
            state = {
                'timesteps': self.num_timesteps,
                'eval_state': self.eval_callback_ref.get_state() if self.eval_callback_ref else {},
                'saved_at': datetime.now().isoformat(),
            }
            with open(f"{self.save_path}/training_state_{self.version}.pkl", 'wb') as f:
                pickle.dump(state, f)
            print(f"\n>>> Checkpoint @ {self.num_timesteps:,} steps")
        return True


class EvalWinRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=100_000, n_eval=200,
                 save_path="models", version="v17.3sp"):
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
        if self.num_timesteps - self._last_eval_step >= self.eval_freq and self.num_timesteps > 0:
            self._last_eval_step = self.num_timesteps
            wins = 0
            total_reward = 0
            print(f"\n>>> Eval ({self.n_eval} games vs SimpleHeuristics)...")
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
                    print(f"    {ep+1}/{self.n_eval}...")

            win_rate = wins / self.n_eval
            avg_reward = total_reward / self.n_eval
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/avg_reward", avg_reward)
            self.logger.record("eval/best_win_rate", self.best_win_rate)
            self.results.append({
                'step': self.num_timesteps, 'win_rate': win_rate, 'avg_reward': avg_reward,
            })

            marker = ""
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                safe_save(self.model, f"{self.save_path}/ppo_pokemon_6v6_{self.version}_best")
                marker = " ★ NEW BEST"

            print(f"\n{'='*60}")
            print(f">>> Eval @ {self.num_timesteps:,} steps")
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
# Environment Creation (Self-Play Aware)
# ============================================================================

def make_selfplay_env(env_idx: int, config: dict, opponent_manager: SelfPlayOpponentManager,
                      is_eval: bool = False):
    """
    Create an environment that swaps between SimpleHeuristics and self-play
    opponents between episodes.

    For SubprocVecEnv: each subprocess gets its own copy of this function,
    so opponent selection happens independently per env.

    For eval: always uses SimpleHeuristics (stable benchmark).
    """
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

        heuristic_opponent = SimpleHeuristicsPlayer(
            battle_format="gen9randombattle",
            server_configuration=server_config,
            start_listening=False,
        )

        if is_eval:
            # Eval always uses SimpleHeuristics
            wrapped = SingleAgentWrapper(rl_env, heuristic_opponent)
            env = MaskablePokeEnvWrapper(wrapped, rl_env)
        else:
            # Training uses MixedSelfPlayWrapper
            wrapped = SingleAgentWrapper(rl_env, heuristic_opponent)
            env = MixedSelfPlayEnvWrapper(
                wrapped, rl_env,
                heuristic_opponent=heuristic_opponent,
                opponent_manager=opponent_manager,
                server_config=server_config,
                data_dir=config["data_dir"],
            )

        env = TimeLimit(env, max_episode_steps=500)
        env = Monitor(env)
        return env
    return _init


class MixedSelfPlayEnvWrapper(MaskablePokeEnvWrapper):
    """
    Extends MaskablePokeEnvWrapper to swap opponents between episodes.

    On each reset(), decides whether to use SimpleHeuristics or a
    self-play checkpoint based on training progress.

    NOTE: With SubprocVecEnv, each subprocess has its own copy of this
    wrapper. The opponent_manager in each subprocess is independent.
    The SelfPlayPoolCallback writes checkpoint paths to a shared file
    that each subprocess reads.
    """

    def __init__(self, env, rl_env, heuristic_opponent,
                 opponent_manager: SelfPlayOpponentManager,
                 server_config, data_dir="data"):
        super().__init__(env, rl_env)
        self.heuristic_opponent = heuristic_opponent
        self.opponent_manager = opponent_manager
        self.server_config = server_config
        self.data_dir = data_dir

        # Cache for loaded self-play opponents
        self._selfplay_cache = {}
        self._max_cache_size = 3

        # Stats
        self._games_played = 0
        self._selfplay_games = 0
        self._current_opponent_type = "heuristic"

        # Timestep tracking (updated via set_timestep)
        self._current_timestep = 0

    def set_timestep(self, timestep: int):
        """Called by callback via env_method to update training progress."""
        self._current_timestep = timestep
        self.opponent_manager._current_timestep_estimate = timestep

    def refresh_pool(self):
        """Called by callback via env_method to rescan checkpoint directory."""
        self.opponent_manager._scan_existing_pool()

    def get_selfplay_stats(self):
        """Called by callback via env_method to get stats."""
        return {
            'games': self._games_played,
            'selfplay_games': self._selfplay_games,
            'ratio': self._selfplay_games / max(self._games_played, 1),
        }

    def reset(self, **kwargs):
        self._games_played += 1

        # Decide opponent for this episode
        use_selfplay = (
            self.opponent_manager.pool_paths and
            self.opponent_manager.should_use_selfplay(self._current_timestep)
        )

        if use_selfplay:
            checkpoint_path = self.opponent_manager.get_random_checkpoint_path()
            if checkpoint_path:
                opponent = self._get_or_load_selfplay(checkpoint_path)
                if opponent is not None:
                    self.env.opponent = opponent
                    self._selfplay_games += 1
                    self._current_opponent_type = "selfplay"
                else:
                    self.env.opponent = self.heuristic_opponent
                    self._current_opponent_type = "heuristic"
            else:
                self.env.opponent = self.heuristic_opponent
                self._current_opponent_type = "heuristic"
        else:
            self.env.opponent = self.heuristic_opponent
            self._current_opponent_type = "heuristic"

        obs, info = super().reset(**kwargs)
        return obs, info

    def _get_or_load_selfplay(self, checkpoint_path: str):
        """Load a self-play opponent, using cache when possible."""
        if checkpoint_path in self._selfplay_cache:
            return self._selfplay_cache[checkpoint_path]

        try:
            from ppo_player import PPOPlayerV17
            opponent = PPOPlayerV17(
                model_path=checkpoint_path,
                data_dir=self.data_dir,
                deterministic=False,
                battle_format="gen9randombattle",
                server_configuration=self.server_config,
                start_listening=False,
            )

            if len(self._selfplay_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._selfplay_cache))
                del self._selfplay_cache[oldest_key]

            self._selfplay_cache[checkpoint_path] = opponent
            return opponent

        except Exception as e:
            print(f"  ⚠ Failed to load self-play opponent {checkpoint_path}: {e}")
            return None


def create_training_env(config, opponent_manager: SelfPlayOpponentManager):
    n_envs = config["n_envs"]
    if n_envs == 1:
        return DummyVecEnv([make_selfplay_env(0, config, opponent_manager)])
    if sys.platform == "darwin":
        try:
            multiprocessing.set_start_method("forkserver", force=True)
        except RuntimeError:
            pass
        start_method = "forkserver"
    else:
        # Avoid forked event-loop deadlocks in poke-env
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        start_method = "spawn"
    return SubprocVecEnv(
        [make_selfplay_env(i, config, opponent_manager) for i in range(n_envs)],
        start_method=start_method,
    )


def create_eval_env(config):
    # Eval env doesn't need opponent_manager — always uses SimpleHeuristics
    dummy_manager = SelfPlayOpponentManager(config)
    return DummyVecEnv([make_selfplay_env(config["n_envs"], config, dummy_manager, is_eval=True)])


# ============================================================================
# Main Training
# ============================================================================

def _get_all_extractors(model):
    policy = model.policy
    extractors = []
    if hasattr(policy, 'pi_features_extractor') and policy.pi_features_extractor is not None:
        extractors.append(("pi", policy.pi_features_extractor))
    if hasattr(policy, 'vf_features_extractor') and policy.vf_features_extractor is not None:
        extractors.append(("vf", policy.vf_features_extractor))
    if not extractors and hasattr(policy, 'features_extractor'):
        extractors.append(("shared", policy.features_extractor))
    seen = set()
    unique = []
    for p, e in extractors:
        if id(e) not in seen:
            seen.add(id(e))
            unique.append((p, e))
    return unique if unique else [("policy", policy.features_extractor)]


def train(config=None, base_model=None):
    if config is None:
        config = CONFIG.copy()

    version = config["version"]
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)
    os.makedirs(config["pool_dir"], exist_ok=True)

    base_lr = config["base_lr"]
    embed_lr = config["embed_lr"]

    print("=" * 70)
    print(f"Pokemon RL Bot {version} — Self-Play + Max+Mean Pooling")
    print("=" * 70)
    print(f"\n  Architecture: v17.3 (max+mean pooling, 700-dim final MLP)")
    print(f"  Observation:  640 dims (unchanged from v17.2)")
    print(f"  Warm-start:   Embeddings + encoder from v17.2")
    print(f"  Self-play:    100% (pool seeded with base model, Huang & Lee approach)")
    print(f"  Pool save:    Every {config['pool_save_freq']:,} steps")
    print(f"  Embedding LR: {config.get('embed_lr_initial', embed_lr):.1e} → {embed_lr:.1e}")
    print(f"  Network LR:   {config.get('base_lr_initial', base_lr):.1e} → {base_lr:.1e}")
    print(f"  Entropy:      {config['ent_coef']} (slightly higher for self-play)")
    print(f"  Target:       {config['total_timesteps']:,} steps\n")

    # Check for resume
    state_path = f"{config['model_dir']}/training_state_{version}.pkl"
    latest_path = f"{config['model_dir']}/ppo_pokemon_6v6_{version}_latest"
    latest_exists = (os.path.exists(latest_path + ".zip") or os.path.exists(latest_path))
    resuming = os.path.exists(state_path) and latest_exists

    # Setup self-play opponent manager (before creating envs)
    opponent_manager = SelfPlayOpponentManager(config)

    # Seed pool with the base model so self-play can start from step 0
    if base_model and opponent_manager.get_pool_size() == 0:
        import shutil
        seed_path = base_model
        if not seed_path.endswith(".zip"):
            if os.path.exists(seed_path + ".zip"):
                seed_path = seed_path + ".zip"
        if os.path.exists(seed_path):
            seed_dest = os.path.join(config["pool_dir"], "checkpoint_0000000000.zip")
            shutil.copy2(seed_path, seed_dest)
            opponent_manager._scan_existing_pool()
            print(f"  → Seeded pool with base model: {os.path.basename(seed_path)}")

    # Create environments
    print("Creating environments...")
    train_env = create_training_env(config, opponent_manager)
    eval_env = create_eval_env(config)
    print(f"  Training: {config['n_envs']} envs (mixed opponents)")
    print(f"  Eval: 1 env (SimpleHeuristics only)")

    # Create model with v17.3 architecture
    policy_kwargs = {
        "features_extractor_class": V17SPFeatureExtractor,
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

    if resuming and not base_model:
        # Resume a v17.3 training run (Ctrl+C recovery)
        print(f"\nResuming v17.3 training...")
        import zipfile, io
        save_path = latest_path
        if not save_path.endswith(".zip"):
            if os.path.exists(save_path + ".zip"):
                save_path = save_path + ".zip"
        with zipfile.ZipFile(save_path, "r") as zip_f:
            with zip_f.open("policy.pth") as f:
                buffer = io.BytesIO(f.read())
                policy_state = torch.load(buffer, map_location="cpu", weights_only=False)
        model.policy.load_state_dict(policy_state)

        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)
        saved_timesteps = saved_state.get('timesteps', 0)
        model.num_timesteps = saved_timesteps
        model._num_timesteps_at_start = saved_timesteps
        print(f"  ✓ Loaded policy weights")
        print(f"  ✓ Restored timestep count: {saved_timesteps:,}")

    elif base_model:
        # Warm-start from v17.2 best model (fresh v17.3 run)
        print(f"\nWarm-starting from v17.2 model...")
        transferred, skipped = warm_start_from_v17_2(model, base_model)
        saved_timesteps = 0
        resuming = False  # Ensure we treat this as a fresh run
    else:
        print("\nWARNING: No base model specified. Training from scratch.")
        print("  Use --base_model models/ppo_pokemon_6v6_v17.2_best for warm-start.")
        saved_timesteps = 0
        resuming = False

    # Setup differential optimizer
    optimizer = setup_differential_optimizer(model, base_lr, embed_lr)

    # Restore optimizer state if resuming
    if resuming:
        optim_path = f"{config['model_dir']}/optimizer_state_{version}.pt"
        if os.path.exists(optim_path):
            try:
                saved_optim = torch.load(optim_path, map_location="cpu", weights_only=False)
                if len(saved_optim["param_groups"]) == len(optimizer.param_groups):
                    current_lrs = [(g["name"], g["lr"]) for g in optimizer.param_groups]
                    optimizer.load_state_dict(saved_optim)
                    for group, (name, lr) in zip(optimizer.param_groups, current_lrs):
                        group["lr"] = lr
                        group["name"] = name
                    print(f"  ✓ Restored optimizer state (Adam momentum/variance preserved)")
                else:
                    print(f"  ⚠ Optimizer state mismatch, fresh optimizer")
            except Exception as e:
                print(f"  ⚠ Could not restore optimizer: {e}")
        else:
            print(f"  ⚠ No saved optimizer state found, Adam will cold-start")

        # Fix LR to correct value for current timestep (avoids resume LR spike bug)
        progress = saved_timesteps / config["total_timesteps"]
        decay = 1.0 / (8.0 * progress + 1.0) ** 1.5
        correct_embed_lr = config["embed_lr"] + (config.get("embed_lr_initial", config["embed_lr"]) - config["embed_lr"]) * decay
        correct_base_lr = config["base_lr"] + (config.get("base_lr_initial", config["base_lr"]) - config["base_lr"]) * decay
        for group in optimizer.param_groups:
            if group.get("name") == "embeddings":
                group["lr"] = correct_embed_lr
            elif group.get("name") == "network":
                group["lr"] = correct_base_lr
        print(f"  ✓ Set resume LRs: embed={correct_embed_lr:.2e}, network={correct_base_lr:.2e}")

    # Print model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Embedding norms
    print("\nInitial embedding norms:")
    for prefix, ext in _get_all_extractors(model):
        for name, embed in [("pokemon", ext.pokemon_embed), ("move", ext.move_embed),
                            ("ability", ext.ability_embed), ("item", ext.item_embed),
                            ("type", ext.type_embed)]:
            print(f"  {prefix}/{name}: {embed.weight.data.norm().item():.4f}")

    # Setup callbacks
    eval_callback = EvalWinRateCallback(
        eval_env=eval_env,
        eval_freq=config["eval_freq"],
        n_eval=config["n_eval_episodes"],
        save_path=config["model_dir"],
        version=version,
    )

    if resuming:
        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)
        eval_callback.set_state(saved_state.get('eval_state', {}))
        remaining = config["total_timesteps"] - saved_timesteps
        print(f"\nResuming: {saved_timesteps:,} done, {remaining:,} remaining")
        print(f"  Best: {eval_callback.best_win_rate:.1%}")
    else:
        remaining = config["total_timesteps"]
        print(f"\nFresh training: {remaining:,} total steps")

    callbacks = CallbackList([
        WinRateCallback(check_freq=10000),
        EnhancedEmbeddingStatsCallback(check_freq=25000),
        DifferentialLRScheduleCallback(config=config, total_timesteps=config["total_timesteps"]),
        SelfPlayPoolCallback(
            opponent_manager=opponent_manager,
            save_freq=config["pool_save_freq"],
        ),
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
    print(f"Self-play: 100% (fictitious play with checkpoint pool)")
    print(f"Eval: Always vs SimpleHeuristics (fixed benchmark)")
    print("=" * 70)

    interrupted = False
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            progress_bar=True,
            # resuming=True: keep num_timesteps from checkpoint (don't reset)
            # resuming=False: fresh run or warm-start, reset to 0
            reset_num_timesteps=(not resuming),
        )
    except KeyboardInterrupt:
        print("\n>>> Paused (Ctrl+C)")
        interrupted = True

    # Save
    actual = model.num_timesteps
    safe_save(model, f"{config['model_dir']}/ppo_pokemon_6v6_{version}_latest")
    torch.save(model.policy.optimizer.state_dict(),
               f"{config['model_dir']}/optimizer_state_{version}.pt")
    state = {
        'timesteps': actual,
        'eval_state': eval_callback.get_state(),
        'saved_at': datetime.now().isoformat(),
        'config': {k: v for k, v in config.items() if not callable(v)},
    }
    with open(f"{config['model_dir']}/training_state_{version}.pkl", 'wb') as f:
        pickle.dump(state, f)

    if not interrupted:
        safe_save(model, f"{config['model_dir']}/ppo_pokemon_6v6_{version}")

    # Final norms
    print("\n" + "=" * 70)
    print("Final embedding norms:")
    for prefix, ext in _get_all_extractors(model):
        for name, embed in [("pokemon", ext.pokemon_embed), ("move", ext.move_embed),
                            ("ability", ext.ability_embed), ("item", ext.item_embed),
                            ("type", ext.type_embed)]:
            print(f"  {prefix}/{name}: {embed.weight.data.norm().item():.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if eval_callback.results:
        for r in eval_callback.results:
            m = " ★" if r['win_rate'] >= eval_callback.best_win_rate else ""
            print(f"  {r['step']:>9,}: {r['win_rate']:>5.1%}  (rew: {r['avg_reward']:>6.3f}){m}")
        print(f"\nBest: {eval_callback.best_win_rate:.1%}")
        print(f"v17.2 baseline: 74% peak / ~67% smoothed")

    if interrupted:
        print(f"\nResume: python ppo_v17_selfplay.py")
        print(f"Fresh:  python ppo_v17_selfplay.py --base_model models/ppo_pokemon_6v6_v17.2_best")

    try:
        train_env.close()
    except:
        pass
    try:
        eval_env.close()
    except:
        pass
    return model


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v17.3 Self-Play Training")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to v17.2 model for warm-start")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps")
    parser.add_argument("--selfplay_initial", type=float, default=None,
                        help="Initial self-play ratio (default: 0.2)")
    parser.add_argument("--selfplay_final", type=float, default=None,
                        help="Final self-play ratio (default: 0.5)")
    parser.add_argument("--ent_coef", type=float, default=None,
                        help="Entropy coefficient (default: 0.06)")
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.timesteps:
        config["total_timesteps"] = args.timesteps
    if args.selfplay_initial is not None:
        config["selfplay_ratio_initial"] = args.selfplay_initial
    if args.selfplay_final is not None:
        config["selfplay_ratio_final"] = args.selfplay_final
    if args.ent_coef is not None:
        config["ent_coef"] = args.ent_coef

    train(config, base_model=args.base_model)