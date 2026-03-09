import asyncio
import numpy as np
import os
import zipfile
import io

import torch
import gymnasium as gym
from poke_env.player import Player, SingleBattleOrder, DefaultBattleOrder, BattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.spaces import Box, Discrete


ARCH_REGISTRY = {
    444: {
        "name": "v17.2",
        "extractor_module": "networks.embedding_extractor",
        "extractor_class": "V17FeatureExtractor",
        "rl_env_module": "envs.rl_player_6v6",
        "rl_env_class": "RL6v6PlayerV17",
    },
    700: {
        "name": "v17.3sp",
        "extractor_module": "networks.embedding_extractor_sp",
        "extractor_class": "V17SPFeatureExtractor",
        "rl_env_module": "envs.rl_player_6v6",
        "rl_env_class": "RL6v6PlayerV17",
    },
    725: {
        "name": "v18",
        "extractor_module": "networks.embedding_extractor_v18",
        "extractor_class": "V18FeatureExtractor",
        "rl_env_module": "envs.rl_player_6v6_v18",
        "rl_env_class": "RL6v6PlayerV18",
    },
}


def _detect_architecture(state_dict):
    for prefix in ["features_extractor", "pi_features_extractor"]:
        ln_key = f"{prefix}.final_mlp.0.weight"
        if ln_key in state_dict:
            return state_dict[ln_key].shape[0]
    return 444


def _import_class(module_name, class_name):
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class PPOPlayer(Player):
    """
    A poke-env Player that uses a trained MaskablePPO model.

    Auto-detects architecture version from saved weights.
    Uses model.predict() for inference — the same code path as evaluate.py.

    Dispatches between sync (training) and async (live play) modes
    automatically based on context.
    """

    def __init__(self, model_path, data_dir="data", deterministic=True, **kwargs):
        super().__init__(**kwargs)
        self.deterministic = deterministic
        self.data_dir = data_dir

        self._state_dict, self._arch_info = self._load_state_dict(model_path)
        arch_name = self._arch_info["name"]

        self._rl_env_class = _import_class(
            self._arch_info["rl_env_module"],
            self._arch_info["rl_env_class"],
        )
        self._extractor_class = _import_class(
            self._arch_info["extractor_module"],
            self._arch_info["extractor_class"],
        )

        self._rl_env = self._rl_env_class(
            battle_format=kwargs.get("battle_format", "gen9randombattle"),
            data_dir=data_dir,
            start_listening=False,
        )

        self.model = self._build_and_load_model()
        self.model.policy.eval()
       

    def _load_state_dict(self, model_path):
        load_path = model_path
        if not load_path.endswith(".zip"):
            if os.path.exists(load_path + ".zip"):
                load_path = load_path + ".zip"
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found: {load_path}")

        with zipfile.ZipFile(load_path, "r") as zip_f:
            with zip_f.open("policy.pth") as f:
                state_dict = torch.load(
                    io.BytesIO(f.read()),
                    map_location="cpu",
                    weights_only=False,
                )

        final_mlp_size = _detect_architecture(state_dict)
        if final_mlp_size not in ARCH_REGISTRY:
            raise ValueError(
                f"Unknown architecture with final_mlp size {final_mlp_size}. "
                f"Known: {list(ARCH_REGISTRY.keys())}"
            )

        return state_dict, ARCH_REGISTRY[final_mlp_size]

    def _build_and_load_model(self):
        obs_space = Box(
            low=-1.0, high=510.0,
            shape=(self._rl_env_class.OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        act_space = Discrete(self._rl_env_class.ACTION_SPACE_SIZE)

        policy_kwargs = {
            "features_extractor_class": self._extractor_class,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "config_path": f"{self.data_dir}/embedding_config.json",
                "pokemon_repr_dim": 128,
                "embed_dropout": 0.05,
            },
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),
            "share_features_extractor": False,
            "activation_fn": torch.nn.ReLU,
        }

        class _DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self, **kwargs):
                return np.zeros(obs_space.shape, dtype=np.float32), {}
            def step(self, action):
                return np.zeros(obs_space.shape, dtype=np.float32), 0.0, True, False, {}
            def action_masks(self):
                return np.ones(act_space.n, dtype=bool)

        dummy_env = DummyVecEnv([lambda: _DummyEnv()])
        model = MaskablePPO(
            "MlpPolicy", dummy_env,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        dummy_env.close()

        model.policy.load_state_dict(self._state_dict)
        return model

    def _predict_action(self, battle):
        """Core prediction logic shared by sync and async paths."""
        obs = self._rl_env.embed_battle(battle)
        mask = self._rl_env.get_action_mask(battle)

        if battle.turn < 2:
            tera_mask_backup = mask[4:8].copy()
            mask[4:8] = 0.0
            if mask.sum() == 0:
                mask[4:8] = tera_mask_backup

        obs_vec = np.expand_dims(obs, axis=0)
        mask_vec = np.expand_dims(mask.astype(bool), axis=0)

        action, _ = self.model.predict(
            obs_vec,
            deterministic=self.deterministic,
            action_masks=mask_vec,
        )
        action = int(action[0])

        self._rl_env.update_action_history(action, battle)

        return self._rl_env.action_to_order(action, battle)

    def choose_move(self, battle: AbstractBattle):
        """
        Smart dispatch: returns an Awaitable when called from poke-env's
        async Player framework (live play), or a direct BattleOrder when
        called synchronously (as a training opponent in SingleAgentWrapper).
        """
        try:
            asyncio.get_running_loop()
            # We're in an async context — return awaitable for live play
            return self._async_choose_move(battle)
        except RuntimeError:
            # No running loop — synchronous call from training
            return self._sync_choose_move(battle)

    def _sync_choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Synchronous move selection for use as training opponent.
        State is already current when called through SingleAgentWrapper."""
        return self._predict_action(battle)

    async def _async_choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Async move selection for live play. Yields control to the event loop
        so turn resolution messages (switches, damage, etc.) can be processed
        before we read the battle state.

        Showdown sends messages in separate batches:
          Batch 1: |request|{...}     <- triggers choose_move (stale state)
          Batch 2: |switch|...|turn|  <- actual turn resolution

        By awaiting here, we let poke-env process Batch 2. When we resume,
        the battle state reflects the current turn.
        """
        initial_turn = battle.turn

        for _ in range(20):
            await asyncio.sleep(0.025)
            if battle.turn > initial_turn:
                break
            if battle.opponent_active_pokemon is not None and initial_turn == 0:
                await asyncio.sleep(0.05)
                break

        return self._predict_action(battle)


class PPOPlayerV17(PPOPlayer):
    """Backward-compatible alias. Use PPOPlayer instead."""
    pass


def load_selfplay_opponent(checkpoint_path, data_dir="data",
                           deterministic=True, **player_kwargs):
    """Create a self-play opponent from a checkpoint."""
    return PPOPlayer(
        model_path=checkpoint_path,
        data_dir=data_dir,
        deterministic=deterministic,
        **player_kwargs,
    )