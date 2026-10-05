"""Inference for torch checkpoints (v20, v21): V20Policy (network + featurizer, used by training
workers, evaluation and self-play pools) and V20Player (a poke-env Player for live play and as an
evaluation opponent).

Checkpoints are torch files ({"arch": "v20" or "v21", "net_config", "layout", "model", ...}),
unlike the SB3 zips of v17-v19; networks/registry.py maps each arch to its featurizer and network.
Both archs share the 14-action space, so one policy class serves both. `load_opponent` returns the
right player for a torch checkpoint or an SB3 zip.
"""
import asyncio

import numpy as np
import torch

from poke_env.player import DefaultBattleOrder, Player

from envs.rl_player_v20 import Featurizer, action_to_order, hold_requests_until_log
from networks import registry


def is_v20_checkpoint(path):
    """A torch checkpoint (any arch), as opposed to an SB3 zip."""
    return str(path).endswith(".pt")


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch")
    if arch not in registry.ARCHS:
        raise ValueError(f"{path} has an unknown arch {arch!r} (known: {sorted(registry.ARCHS)})")
    saved = ckpt.get("layout")
    if saved is not None and dict(saved) != dict(registry.expected_layout(arch, saved)):
        raise ValueError(f"{path} was saved with a different observation layout; "
                         "it needs the env code it was trained with")
    return ckpt


class V20Policy:
    """Chooses actions for a battle, from that battle's player's point of view, with a v20 or v21
    network and the matching featurizer."""

    def __init__(self, net, deterministic=False, featurizer=None, arch="v20"):
        self.net = net.eval()
        self.deterministic = deterministic
        self.featurizer = featurizer or Featurizer()
        self.arch = arch

    @classmethod
    def load(cls, path, deterministic=False, featurizer=None):
        ckpt = load_checkpoint(path)
        arch = ckpt["arch"]
        spec = registry.get(arch)
        layout = ckpt.get("layout") or spec.layout()
        net = spec.net(ckpt.get("net_config"), layout)
        net.load_state_dict(ckpt["model"])
        return cls(net, deterministic, featurizer or spec.featurizer(layout), arch)

    def action(self, battle):
        idx, flt, mask = self.featurizer.featurize(battle)
        # training stores features as float16; round the same way here
        flt = flt.astype(np.float16).astype(np.float32)
        with torch.inference_mode():
            a, _, _ = self.net.act(torch.as_tensor(idx)[None], torch.as_tensor(flt)[None],
                                   torch.as_tensor(mask)[None], self.deterministic)
        return int(a[0])

    def order(self, battle):
        return action_to_order(self.action(battle), battle)


class V20Player(Player):
    """poke-env Player driven by a v20 or v21 checkpoint.

    choose_move returns an awaitable inside poke-env's event loop (live play) and an order when
    called synchronously (as an opponent stepped by an env, e.g. in eval/ppo_test.py).
    hold_requests: against this repo's Showdown fork, which sends each request before the turn's
    log, handle requests only after the log (see envs.rl_player_v20.hold_requests_until_log).
    Leave it off on the public server, which sends the log first.
    """

    def __init__(self, model_path, deterministic=True, hold_requests=False, **kwargs):
        super().__init__(**kwargs)
        self.policy = V20Policy.load(model_path, deterministic)
        if hold_requests:
            hold_requests_until_log(self)

    def choose_move(self, battle):
        try:
            asyncio.get_running_loop()
            return self._async_choose_move(battle)
        except RuntimeError:
            return self._sync_choose_move(battle)

    def _sync_choose_move(self, battle):
        if getattr(battle, "_wait", False):
            return DefaultBattleOrder()
        return self.policy.order(battle)

    async def _async_choose_move(self, battle):
        return self.policy.order(battle)


def load_opponent(path, deterministic=False, data_dir="data", **player_kwargs):
    """An opponent for env-stepped play: V20Policy for torch checkpoints of any arch (returns int
    actions), PPOPlayer for SB3 zips (returns orders)."""
    if is_v20_checkpoint(path):
        return V20Policy.load(path, deterministic)
    from players.ppo_player import PPOPlayer
    player_kwargs.setdefault("battle_format", "gen9randombattle")
    player_kwargs.setdefault("start_listening", False)
    return PPOPlayer(model_path=path, data_dir=data_dir, deterministic=deterministic, **player_kwargs)


def opponent_decision(opponent, battle):
    """What an env-stepped opponent plays: an int action (v20/v21 policies) or a BattleOrder
    (anything else)."""
    if isinstance(opponent, V20Policy):
        return opponent.action(battle)
    return opponent.choose_move(battle)
