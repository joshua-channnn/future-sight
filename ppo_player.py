"""
PPOPlayerV17 - Uses trained v17.2/v17.3 PPO model for battle decisions.

Designed to serve as a self-play opponent during training.
Delegates to RL6v6PlayerV17 for embed_battle/get_action_mask/action_to_order
to ensure training and inference use identical state representations.

No VecNormalize needed — v17 uses embeddings that handle their own scaling.

Handles differential LR model loading (creates fresh model architecture,
loads only policy weights from zip file).
"""

import numpy as np
import os
import zipfile
import io

import torch
from poke_env.player import Player, SingleBattleOrder, DefaultBattleOrder, BattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from sb3_contrib import MaskablePPO

from RLPlayer6v6 import RL6v6PlayerV17


class PPOPlayerV17(Player):
    """
    A poke-env Player that uses a trained v17 MaskablePPO model.

    Can be used as an opponent in SingleAgentWrapper for self-play training.

    Args:
        model_path: Path to saved model (with or without .zip)
        data_dir: Path to embedding data directory
        deterministic: If True, always pick the greedy action.
                      If False, sample from the policy distribution.
                      For self-play opponents, False is better (more diversity).
        **kwargs: Passed to poke-env Player (battle_format, etc.)
    """

    # Shared RL env instance for embed_battle / action_mask / action_to_order
    # All PPOPlayerV17 instances can share this since it's stateless per-call
    _shared_rl_env = None
    _shared_data_dir = None

    def __init__(
        self,
        model_path: str,
        data_dir: str = "data",
        deterministic: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.deterministic = deterministic
        self.data_dir = data_dir

        # Initialize shared RL env for observation/action methods
        if (PPOPlayerV17._shared_rl_env is None or
                PPOPlayerV17._shared_data_dir != data_dir):
            PPOPlayerV17._shared_rl_env = RL6v6PlayerV17(
                battle_format=kwargs.get("battle_format", "gen9randombattle"),
                data_dir=data_dir,
            )
            PPOPlayerV17._shared_data_dir = data_dir

        # Load model weights
        self.policy = self._load_policy(model_path)
        self.policy.eval()  # Disable dropout

    def _load_policy(self, model_path: str):
        """
        Load policy weights from a saved model file.

        We only need the policy network for inference (no optimizer, no env).
        Load the state dict directly into a fresh policy to avoid SB3's
        optimizer restoration issues with differential LR.
        """
        load_path = model_path
        if not load_path.endswith(".zip"):
            if os.path.exists(load_path + ".zip"):
                load_path = load_path + ".zip"

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found: {load_path}")

        # Detect which feature extractor to use based on file contents
        # v17.3 models have final_mlp with 700-dim input, v17.2 has 444-dim
        with zipfile.ZipFile(load_path, "r") as zip_f:
            with zip_f.open("policy.pth") as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

        # Check final_mlp input size to determine architecture
        # LayerNorm weight size tells us: 444 = v17.2, 700 = v17.3
        ln_key = "features_extractor.final_mlp.0.weight"
        if ln_key in state_dict:
            final_mlp_size = state_dict[ln_key].shape[0]
        else:
            # Try pi-specific key
            ln_key_pi = "pi_features_extractor.final_mlp.0.weight"
            if ln_key_pi in state_dict:
                final_mlp_size = state_dict[ln_key_pi].shape[0]
            else:
                final_mlp_size = 444  # Default to v17.2

        if final_mlp_size == 700:
            from embedding_extractor_sp import V17SPFeatureExtractor
            extractor_class = V17SPFeatureExtractor
        else:
            from embedding_extractor import V17FeatureExtractor
            extractor_class = V17FeatureExtractor

        # Create a temporary model with correct architecture to get the policy
        from gymnasium.spaces import Box, Discrete
        obs_space = Box(low=-1.0, high=510.0,
                        shape=(RL6v6PlayerV17.OBSERVATION_SIZE,), dtype=np.float32)
        act_space = Discrete(RL6v6PlayerV17.ACTION_SPACE_SIZE)

        # Build policy kwargs
        policy_kwargs = {
            "features_extractor_class": extractor_class,
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

        # Create temporary dummy env for MaskablePPO initialization
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym

        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self, **kwargs):
                return np.zeros(obs_space.shape, dtype=np.float32), {}
            def step(self, action):
                return np.zeros(obs_space.shape, dtype=np.float32), 0.0, True, False, {}
            def action_masks(self):
                return np.ones(act_space.n, dtype=bool)

        dummy_env = DummyVecEnv([lambda: DummyEnv()])
        temp_model = MaskablePPO(
            "MlpPolicy", dummy_env,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        dummy_env.close()

        # Load weights
        temp_model.policy.load_state_dict(state_dict)
        return temp_model.policy

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Main decision method called by poke-env."""
        rl_env = PPOPlayerV17._shared_rl_env

        # Get observation using the shared RL env's encoding
        obs = rl_env.embed_battle(battle)

        # Get action mask
        action_mask = rl_env.get_action_mask(battle)

        # Predict action
        device = next(self.policy.parameters()).device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get action distribution from policy
            dist = self.policy.get_distribution(obs_tensor)
            # Apply mask: set invalid actions to -inf
            logits = dist.distribution.logits
            logits[~mask_tensor] = float('-inf')

            if self.deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                # Re-normalize and sample
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

        return rl_env.action_to_order(action, battle)


def load_selfplay_opponent(checkpoint_path: str, data_dir: str = "data",
                           deterministic: bool = False,
                           **player_kwargs) -> PPOPlayerV17:
    """
    Convenience function to create a self-play opponent from a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to embedding data
        deterministic: Whether to use greedy actions
        **player_kwargs: Passed to Player (battle_format, server_configuration, etc.)

    Returns:
        PPOPlayerV17 instance ready to use as an opponent
    """
    return PPOPlayerV17(
        model_path=checkpoint_path,
        data_dir=data_dir,
        deterministic=deterministic,
        **player_kwargs,
    )


if __name__ == "__main__":
    print("Testing PPOPlayerV17...")

    # Test with v17.2 model
    model_path = "models/ppo_pokemon_6v6_v17.2_best"
    if os.path.exists(model_path + ".zip"):
        player = PPOPlayerV17(
            model_path=model_path,
            data_dir="data",
            battle_format="gen9randombattle",
        )
        print(f"✓ Created PPOPlayerV17 successfully!")
        print(f"  Model: {model_path}")
        print(f"  Observation size: {RL6v6PlayerV17.OBSERVATION_SIZE}")
        print(f"  Action space: {RL6v6PlayerV17.ACTION_SPACE_SIZE}")
    else:
        print(f"Model not found: {model_path}")
        print("Train v17.2 first with: python ppo_v17_difflr.py --new")