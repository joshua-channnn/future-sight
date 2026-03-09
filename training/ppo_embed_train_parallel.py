import os
import sys
import pickle
import math
import uuid
import multiprocessing
import numpy as np
from datetime import datetime
from typing import Callable, Optional

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

from envs.rl_player_6v6_v18 import RL6v6PlayerV18
from networks.embedding_extractor_v18 import V18FeatureExtractor
from envs.wrappers import MaskablePokeEnvWrapper


BASE_PORT = 8000


def get_server_config(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://127.0.0.1:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


EMBEDDING_LAYER_NAMES = [
    "pokemon_embed", "move_embed", "ability_embed", "item_embed", "type_embed",
]

CONFIG = {
    "total_timesteps": 20_000_000,
    "n_envs": 4,

    # Differential LR — CHANGED: slower decay, higher floors
    "base_lr": 5e-5,                # was 3e-5 — higher floor
    "base_lr_initial": 1e-4,        # same as v18
    "embed_lr": 5e-4,               # was 3e-4 — higher floor
    "embed_lr_initial": 1e-3,       # same as v18
    "lr_decay_exponent": 1.0,       # was 1.5 — gentler decay
    "use_lr_decay": True,

    # PPO hyperparameters (same as v18)
    "gamma": 0.9999,
    "gae_lambda": 0.754,
    "clip_range": 0.08,
    "n_epochs": 4,
    "batch_size": 1024,
    "n_steps": 512,
    "max_grad_norm": 0.5,
    "ent_coef": 0.06,
    "vf_coef": 0.5,

    # Architecture (same as v18)
    "features_dim": 256,
    "pokemon_repr_dim": 128,
    "embed_dropout": 0.05,
    "net_arch_pi": [256, 128],
    "net_arch_vf": [256, 128],
    "share_features_extractor": False,

    # Evaluation
    "eval_freq": 100_000,
    "n_eval_episodes": 200,
    "save_freq": 200_000,

    # Paths
    "data_dir": "data",
    "log_dir": "logs",
    "model_dir": "models",
    "version": "v18.1",
}


def is_embedding_param(name: str) -> bool:
    for embed_name in EMBEDDING_LAYER_NAMES:
        if embed_name in name and "weight" in name:
            return True
    return False


def setup_differential_optimizer(model, base_lr, embed_lr):
    policy = model.policy
    embedding_params, other_params = [], []

    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if is_embedding_param(name):
            embedding_params.append(param)
        else:
            other_params.append(param)

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


def warm_start_from_v17_2(model, v17_2_path: str):
    """
    Transfer weights from v17.2 into v18.1.
    
    Transfers all matching tensors. Skips mismatched shapes
    (ActiveMatchupHead MLP, final_mlp, etc. due to obs dim changes).
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
            v17_state = torch.load(buffer, map_location="cpu", weights_only=False)

    current_state = model.policy.state_dict()
    transferred, skipped = 0, 0
    skipped_names = []

    for name, param in v17_state.items():
        if name in current_state:
            if current_state[name].shape == param.shape:
                current_state[name] = param
                transferred += 1
            else:
                skipped += 1
                skipped_names.append(f"    {name}: {param.shape} -> {current_state[name].shape}")
        else:
            skipped += 1
            skipped_names.append(f"    {name}: not found in v18.1")

    model.policy.load_state_dict(current_state)
    print(f"  Transferred {transferred} parameter tensors from v17.2")
    if skipped > 0:
        print(f"  Skipped {skipped} tensors (shape mismatch or missing):")
        for s in skipped_names:
            print(s)
    return transferred, skipped


class WinRateCallback(BaseCallback):
    def __init__(self, check_freq=10000):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self.wins, self.losses = 0, 0
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
            self.wins, self.losses = 0, 0
        return True


class EnhancedEmbeddingStatsCallback(BaseCallback):
    def __init__(self, check_freq=25000):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self._last_log_step = 0
        self._initial_norms, self._last_weights = {}, {}
        self._initialized = False

    def _on_step(self):
        if self.num_timesteps - self._last_log_step < self.check_freq:
            return True
        self._last_log_step = self.num_timesteps
        for prefix, extractor in self._get_extractors():
            for name, embed in [("pokemon", extractor.pokemon_embed), ("move", extractor.move_embed),
                                ("ability", extractor.ability_embed), ("item", extractor.item_embed),
                                ("type", extractor.type_embed)]:
                key = f"{prefix}/{name}"
                weight = embed.weight.data
                norm = weight.norm().item()
                self.logger.record(f"embeddings/{key}_norm", norm)
                if embed.weight.grad is not None:
                    self.logger.record(f"embeddings/{key}_grad_norm", embed.weight.grad.norm().item())
                if key in self._last_weights:
                    self.logger.record(f"embeddings/{key}_delta", (weight - self._last_weights[key]).norm().item())
                self._last_weights[key] = weight.clone()
                if not self._initialized:
                    self._initial_norms[key] = norm
                if key in self._initial_norms and self._initial_norms[key] > 0:
                    self.logger.record(f"embeddings/{key}_rel_change",
                                       abs(norm - self._initial_norms[key]) / self._initial_norms[key])
        if not self._initialized:
            self._initialized = True
        for group in self.model.policy.optimizer.param_groups:
            self.logger.record(f"lr/{group.get('name', 'unknown')}", group["lr"])
        return True

    def _get_extractors(self):
        policy = self.model.policy
        extractors = []
        if hasattr(policy, 'pi_features_extractor'):
            extractors.append(("pi", policy.pi_features_extractor))
        if hasattr(policy, 'vf_features_extractor'):
            extractors.append(("vf", policy.vf_features_extractor))
        if not extractors and hasattr(policy, 'features_extractor'):
            extractors.append(("shared", policy.features_extractor))
        seen, unique = set(), []
        for p, e in extractors:
            if id(e) not in seen:
                seen.add(id(e))
                unique.append((p, e))
        return unique if unique else [("policy", self.model.policy.features_extractor)]


class DifferentialLRScheduleCallback(BaseCallback):
    def __init__(self, config: dict, total_timesteps: int):
        super().__init__(verbose=0)
        self.base_lr_initial = config.get("base_lr_initial", config["base_lr"])
        self.base_lr_final = config["base_lr"]
        self.embed_lr_initial = config.get("embed_lr_initial", config["embed_lr"])
        self.embed_lr_final = config["embed_lr"]
        self.use_decay = config.get("use_lr_decay", True)
        self.decay_exponent = config.get("lr_decay_exponent", 1.0)
        self.total_timesteps = total_timesteps
        self._last_log = 0

    def _on_step(self):
        if self.use_decay:
            progress = min(self.num_timesteps / max(self.total_timesteps, 1), 1.0)
            # CHANGED: configurable exponent (1.0 for v18.1, was 1.5 for v18)
            decay = 1.0 / (8.0 * progress + 1.0) ** self.decay_exponent
            base_lr = self.base_lr_final + (self.base_lr_initial - self.base_lr_final) * decay
            embed_lr = self.embed_lr_final + (self.embed_lr_initial - self.embed_lr_final) * decay
        else:
            base_lr, embed_lr = self.base_lr_final, self.embed_lr_final

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
    def __init__(self, eval_env, eval_freq=100_000, n_eval=200, save_path="models", version="v18.1"):
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
        if self.num_timesteps - self._last_eval_step < self.eval_freq or self.num_timesteps == 0:
            return True
        self._last_eval_step = self.num_timesteps

        wins, total_reward = 0, 0
        print(f"\n>>> Eval ({self.n_eval} games vs SimpleHeuristics)...")
        for ep in range(self.n_eval):
            obs = self.eval_env.reset()
            done, ep_reward = False, 0
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
        self.results.append({'step': self.num_timesteps, 'win_rate': win_rate, 'avg_reward': avg_reward})

        marker = ""
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            safe_save(self.model, f"{self.save_path}/ppo_pokemon_6v6_{self.version}_best")
            marker = " NEW BEST"

        print(f"\n{'='*60}")
        print(f">>> Eval @ {self.num_timesteps:,}: {win_rate:.1%} ({wins}/{self.n_eval}){marker} | Best: {self.best_win_rate:.1%}")
        print(f"{'='*60}\n")
        return True

    def get_state(self):
        return {'best_win_rate': self.best_win_rate, 'results': self.results,
                'last_eval_step': self._last_eval_step}

    def set_state(self, state):
        self.best_win_rate = state.get('best_win_rate', 0.0)
        self.results = state.get('results', [])
        self._last_eval_step = state.get('last_eval_step', 0)


def make_heuristic_env(env_idx, config, is_eval=False):
    def _init():
        port = BASE_PORT + env_idx
        server_config = get_server_config(port)
        uid = uuid.uuid4().hex[:6]

        rl_env = RL6v6PlayerV18(
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
        return DummyVecEnv([make_heuristic_env(0, config)])
    if sys.platform == "darwin":
        try:
            multiprocessing.set_start_method("forkserver", force=True)
        except RuntimeError:
            pass
    return SubprocVecEnv(
        [make_heuristic_env(i, config) for i in range(n_envs)],
        start_method="forkserver" if sys.platform == "darwin" else "fork",
    )


def create_eval_env(config):
    return DummyVecEnv([make_heuristic_env(config["n_envs"], config, is_eval=True)])


def _get_all_extractors(model):
    policy = model.policy
    extractors = []
    if hasattr(policy, 'pi_features_extractor'):
        extractors.append(("pi", policy.pi_features_extractor))
    if hasattr(policy, 'vf_features_extractor'):
        extractors.append(("vf", policy.vf_features_extractor))
    if not extractors and hasattr(policy, 'features_extractor'):
        extractors.append(("shared", policy.features_extractor))
    seen, unique = set(), []
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

    base_lr, embed_lr = config["base_lr"], config["embed_lr"]

    print("=" * 70)
    print(f"Pokemon RL Bot {version} — Heuristic-Only Training")
    print("=" * 70)
    print(f"  Architecture: v18 (cat. eff, def bench, opp history, self history)")
    print(f"  Observation:  677 dims")
    print(f"  Training:     100% SimpleHeuristics (no self-play)")
    print(f"  Immune mask:  Active (damaging moves with 0x eff masked)")
    print(f"  LR decay:     exponent={config['lr_decay_exponent']} (was 1.5 in v18)")
    print(f"  Embedding LR: {config.get('embed_lr_initial', embed_lr):.1e} -> {embed_lr:.1e}")
    print(f"  Network LR:   {config.get('base_lr_initial', base_lr):.1e} -> {base_lr:.1e}")
    print(f"  vf_coef:      {config['vf_coef']}")
    print(f"  Target:       {config['total_timesteps']:,} steps\n")

    state_path = f"{config['model_dir']}/training_state_{version}.pkl"
    latest_path = f"{config['model_dir']}/ppo_pokemon_6v6_{version}_latest"
    latest_exists = os.path.exists(latest_path + ".zip") or os.path.exists(latest_path)
    resuming = os.path.exists(state_path) and latest_exists

    print("Creating environments...")
    train_env = create_training_env(config)
    eval_env = create_eval_env(config)
    print(f"  Training: {config['n_envs']} envs | Eval: 1 env (SimpleHeuristics)")

    policy_kwargs = {
        "features_extractor_class": V18FeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": config["features_dim"],
            "config_path": f"{config['data_dir']}/embedding_config.json",
            "pokemon_repr_dim": config.get("pokemon_repr_dim", 128),
            "embed_dropout": config["embed_dropout"],
        },
        "net_arch": dict(pi=config["net_arch_pi"], vf=config["net_arch_vf"]),
        "share_features_extractor": config.get("share_features_extractor", False),
        "activation_fn": torch.nn.ReLU,
    }

    model = MaskablePPO(
        "MlpPolicy", train_env, learning_rate=base_lr,
        n_steps=config["n_steps"], batch_size=config["batch_size"],
        n_epochs=config["n_epochs"], gamma=config["gamma"],
        gae_lambda=config["gae_lambda"], clip_range=config["clip_range"],
        ent_coef=config["ent_coef"], vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"], policy_kwargs=policy_kwargs,
        verbose=1, tensorboard_log=config["log_dir"],
    )

    if resuming and not base_model:
        print(f"\nResuming {version} training...")
        import zipfile, io
        save_path = latest_path
        if not save_path.endswith(".zip"):
            if os.path.exists(save_path + ".zip"):
                save_path = save_path + ".zip"
        with zipfile.ZipFile(save_path, "r") as zip_f:
            with zip_f.open("policy.pth") as f:
                policy_state = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
        model.policy.load_state_dict(policy_state)
        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)
        saved_timesteps = saved_state.get('timesteps', 0)
        model.num_timesteps = saved_timesteps
        model._num_timesteps_at_start = saved_timesteps
        print(f"  Loaded policy, restored timestep: {saved_timesteps:,}")
    elif base_model:
        print(f"\nWarm-starting from v17.2...")
        transferred, skipped = warm_start_from_v17_2(model, base_model)
        saved_timesteps = 0
        resuming = False
    else:
        print("\nWARNING: No base model. Training from scratch.")
        saved_timesteps = 0
        resuming = False

    optimizer = setup_differential_optimizer(model, base_lr, embed_lr)

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
                    print(f"  Restored optimizer state")
            except Exception as e:
                print(f"  Could not restore optimizer: {e}")

        # Fix LR for resume point
        progress = saved_timesteps / config["total_timesteps"]
        decay = 1.0 / (8.0 * progress + 1.0) ** config["lr_decay_exponent"]
        correct_embed = config["embed_lr"] + (config.get("embed_lr_initial", config["embed_lr"]) - config["embed_lr"]) * decay
        correct_base = config["base_lr"] + (config.get("base_lr_initial", config["base_lr"]) - config["base_lr"]) * decay
        for group in optimizer.param_groups:
            if group.get("name") == "embeddings":
                group["lr"] = correct_embed
            elif group.get("name") == "network":
                group["lr"] = correct_base

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("\nInitial embedding norms:")
    for prefix, ext in _get_all_extractors(model):
        for name, embed in [("pokemon", ext.pokemon_embed), ("move", ext.move_embed),
                            ("ability", ext.ability_embed), ("item", ext.item_embed),
                            ("type", ext.type_embed)]:
            print(f"  {prefix}/{name}: {embed.weight.data.norm().item():.4f}")

    eval_callback = EvalWinRateCallback(
        eval_env=eval_env, eval_freq=config["eval_freq"],
        n_eval=config["n_eval_episodes"], save_path=config["model_dir"],
        version=version,
    )

    if resuming:
        with open(state_path, 'rb') as f:
            eval_callback.set_state(pickle.load(f).get('eval_state', {}))
        remaining = config["total_timesteps"] - saved_timesteps
        print(f"\nResuming: {saved_timesteps:,} done, {remaining:,} remaining | Best: {eval_callback.best_win_rate:.1%}")
    else:
        remaining = config["total_timesteps"]
        print(f"\nFresh training: {remaining:,} total steps")

    callbacks = CallbackList([
        WinRateCallback(check_freq=10000),
        EnhancedEmbeddingStatsCallback(check_freq=25000),
        DifferentialLRScheduleCallback(config=config, total_timesteps=config["total_timesteps"]),
        PeriodicSaveCallback(
            save_freq=config["save_freq"], save_path=config["model_dir"],
            version=version, eval_callback_ref=eval_callback,
        ),
        eval_callback,
    ])

    print(f"\nTraining for {remaining:,} steps (100% SimpleHeuristics)...")
    print("=" * 70)

    interrupted = False
    try:
        model.learn(
            total_timesteps=remaining, callback=callbacks,
            progress_bar=True, reset_num_timesteps=(not resuming),
        )
    except KeyboardInterrupt:
        print("\n>>> Paused (Ctrl+C)")
        interrupted = True

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

    print("\n" + "=" * 70)
    print("Final embedding norms:")
    for prefix, ext in _get_all_extractors(model):
        for name, embed in [("pokemon", ext.pokemon_embed), ("move", ext.move_embed),
                            ("ability", ext.ability_embed), ("item", ext.item_embed),
                            ("type", ext.type_embed)]:
            print(f"  {prefix}/{name}: {embed.weight.data.norm().item():.4f}")

    print("\n" + "=" * 70 + "\nSummary\n" + "=" * 70)
    if eval_callback.results:
        for r in eval_callback.results:
            m = " *" if r['win_rate'] >= eval_callback.best_win_rate else ""
            print(f"  {r['step']:>9,}: {r['win_rate']:>5.1%}  (rew: {r['avg_reward']:>6.3f}){m}")
        print(f"\nBest: {eval_callback.best_win_rate:.1%}")
        print(f"v17.2 baseline: 74% peak / 67% smoothed (target to beat)")
        print(f"v18 baseline:   71% peak / 62% smoothed")

    if interrupted:
        print(f"\nResume: python ppo_v18_1_heuristic.py")
        print(f"Fresh:  python ppo_v18_1_heuristic.py --base_model models/ppo_pokemon_6v6_v17.2_best")

    try:
        train_env.close()
    except Exception:
        pass
    try:
        eval_env.close()
    except Exception:
        pass
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v18.1 Heuristic-Only Training")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to v17.2 model for warm-start")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--ent_coef", type=float, default=None)
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.timesteps:
        config["total_timesteps"] = args.timesteps
    if args.ent_coef is not None:
        config["ent_coef"] = args.ent_coef

    train(config, base_model=args.base_model)