import os
import math
import time
import uuid
import argparse
import zipfile
import io
import requests
import numpy as np
from typing import Optional

import torch
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.rl_player_6v6_v18 import RL6v6PlayerV18
from networks.embedding_extractor_v18 import V18FeatureExtractor
from utils.battle_cloner import battle_to_search_request, state_to_observation


class MCTSNode:
    __slots__ = ['action', 'action_idx', 'prior', 'visit_count', 'total_value', 'mean_value']

    def __init__(self, action: str, action_idx: int, prior: float = 0.0):
        self.action = action
        self.action_idx = action_idx
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

    def update(self, value: float):
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count


ACTION_TO_MOVE = {
    0: "move 1", 1: "move 2", 2: "move 3", 3: "move 4",
    4: "move 1 terastallize", 5: "move 2 terastallize",
    6: "move 3 terastallize", 7: "move 4 terastallize",
    8: "switch 2", 9: "switch 3", 10: "switch 4", 11: "switch 5", 12: "switch 6",
}


def load_model(model_path: str, env, data_dir: str = "data"):
    """Load trained MaskablePPO model matching evaluate.py's approach exactly.
    Takes a DummyVecEnv so observation/action spaces match the real env."""
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    with zipfile.ZipFile(model_path, "r") as zf:
        with zf.open("policy.pth") as f:
            state_dict = torch.load(io.BytesIO(f.read()), map_location="cpu",
                                    weights_only=False)

    policy_kwargs = {
        "features_extractor_class": V18FeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "config_path": f"{data_dir}/embedding_config.json",
            "pokemon_repr_dim": 128,
            "embed_dropout": 0.05,
        },
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),
        "share_features_extractor": False,
        "activation_fn": torch.nn.ReLU,
    }

    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=3e-5,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    model.policy.load_state_dict(state_dict)
    model.policy.eval()
    return model


class WangMCTSPlayer(RL6v6PlayerV18):
    """
    Pokemon battle player using Wang-style MCTS.

    Inherits from RL6v6PlayerV18 so that `self` IS the battle participant.
    embed_battle, get_action_mask, action_to_order all work on `self`
    with correct state tracking (opponent history, switch tracking, etc).

    Model is loaded separately and used for action selection + value eval.
    """

    def __init__(
        self,
        model_path: str,
        search_server_url: str = "http://localhost:9002",
        n_opponent_samples: int = 5,
        override_threshold: float = 0.15,
        min_visits_to_override: int = 2,
        min_tera_turn: int = 2,
        debug: bool = False,
        **kwargs,
    ):
        # RL6v6PlayerV18 handles data loading, poke-env Player registration, etc.
        super().__init__(**kwargs)

        # Load the trained policy
        data_dir = kwargs.get("data_dir", "data")
        self.policy = load_policy(model_path, data_dir)

        self.search_server_url = search_server_url
        self.n_opponent_samples = n_opponent_samples
        self.override_threshold = override_threshold
        self.min_visits_to_override = min_visits_to_override
        self.min_tera_turn = min_tera_turn
        self.debug = debug

        self._stats = {
            'total_decisions': 0, 'search_decisions': 0,
            'fallback_decisions': 0, 'total_search_time_ms': 0,
            'override_count': 0,
        }

    @torch.no_grad()
    def _get_values_batch(self, observations: list) -> list:
        if not observations:
            return []
        obs_t = torch.as_tensor(np.stack(observations), dtype=torch.float32)
        values = self.policy.predict_values(obs_t)
        result = values.squeeze(-1).tolist()
        return result if isinstance(result, list) else [result]

    def _call_search_server(self, state: dict) -> Optional[dict]:
        try:
            resp = requests.post(
                f"{self.search_server_url}/simulate-batch",
                json=state, timeout=5.0,
            )
            return resp.json() if resp.status_code == 200 else None
        except (requests.exceptions.RequestException, ValueError):
            return None

    def _get_opponent_moves(self, battle: AbstractBattle) -> list:
        opp = battle.opponent_active_pokemon
        if not opp or opp.fainted:
            return ["move 1"]
        n = min(len(opp.moves), 4) if opp.moves else 4
        return [f"move {i+1}" for i in range(max(n, 1))]

    def _run_search(self, battle, policy_probs, action_mask, policy_greedy) -> Optional[int]:
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) <= 1:
            return None

        nodes = {}
        action_to_sim = {}
        for aidx in legal_actions:
            a = int(aidx)
            move_str = ACTION_TO_MOVE.get(a, "move 1")
            nodes[a] = MCTSNode(action=move_str, action_idx=a, prior=policy_probs[a])
            action_to_sim[a] = move_str

        unique_sim_moves = list(set(action_to_sim.values()))

        opp_moves = self._get_opponent_moves(battle)
        if len(opp_moves) > self.n_opponent_samples:
            opp_sample = list(np.random.choice(
                opp_moves, self.n_opponent_samples, replace=False))
        else:
            opp_sample = opp_moves

        search_state = battle_to_search_request(battle)
        search_state["movePairs"] = [
            {"p1": p1, "p2": p2}
            for p1 in unique_sim_moves for p2 in opp_sample
        ]

        response = self._call_search_server(search_state)
        if not response:
            return None

        results = response.get("results", [])
        observations, result_map = [], []
        for i, r in enumerate(results):
            if r.get("success"):
                observations.append(
                    state_to_observation(r["state"], original_battle=battle))
                result_map.append(i)

        if not observations:
            return None

        values = self._get_values_batch(observations)

        value_lookup = {}
        for obs_idx, res_idx in enumerate(result_map):
            r = results[res_idx]
            st = r.get("state", {})
            if st.get("ended"):
                w = st.get("winner", "")
                v = 1.0 if "SearchP1" in str(w) else (-1.0 if w else 0.0)
            else:
                v = values[obs_idx]
            value_lookup[(r["p1Move"], r["p2Move"])] = v

        for aidx, node in nodes.items():
            sim_move = action_to_sim[aidx]
            for p2 in opp_sample:
                if (sim_move, p2) in value_lookup:
                    node.update(value_lookup[(sim_move, p2)])

        policy_node = nodes.get(policy_greedy)
        if policy_node is None or policy_node.visit_count == 0:
            return None

        best_action = max(nodes.keys(), key=lambda a: nodes[a].mean_value)
        best_q = nodes[best_action].mean_value
        policy_q = policy_node.mean_value

        should_override = (
            best_action != policy_greedy
            and (best_q - policy_q) > self.override_threshold
            and nodes[best_action].visit_count >= self.min_visits_to_override
        )

        if self.debug:
            print(f"\n  MCTS (turn {battle.turn}):")
            for aidx in sorted(nodes.keys()):
                n = nodes[aidx]
                tags = ""
                if aidx == best_action and should_override: tags += " ←OVERRIDE"
                if aidx == policy_greedy: tags += " ←POLICY"
                print(f"    {n.action:25s}  Q={n.mean_value:+.4f}  "
                      f"N={n.visit_count:2d}  P={n.prior:.3f}{tags}")
            if not should_override and best_action != policy_greedy:
                print(f"    [no override: Δ={best_q - policy_q:+.3f} "
                      f"< threshold={self.override_threshold}]")

        return best_action if should_override else None

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Uses self.embed_battle (inherited from RL6v6PlayerV18, with correct
        state tracking since self IS the battle participant), then runs
        the SB3 policy for action selection, optionally overriding with MCTS.
        """
        self._stats['total_decisions'] += 1

        # embed_battle on SELF — correct state tracking
        obs = self.embed_battle(battle)
        action_mask = self.get_action_mask(battle)

        # Apply min_tera_turn
        if battle.turn < self.min_tera_turn:
            backup = action_mask[4:8].copy()
            action_mask[4:8] = 0.0
            if action_mask.sum() == 0:
                action_mask[4:8] = backup

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask.astype(bool), dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            # Use SB3's forward with action_masks — same path as ppo_test.py
            action_tensor, _, _ = self.policy.forward(
                obs_t, deterministic=False, action_masks=mask_t,
            )
            policy_action = action_tensor.item()

            # Get policy probs for MCTS priors
            dist = self.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.clone()
            logits[~mask_t] = float('-inf')
            policy_probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        policy_greedy = int(np.argmax(policy_probs))

        # Run search if multiple legal actions
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) <= 1:
            self._stats['fallback_decisions'] += 1
            return self.action_to_order(policy_action, battle)

        t0 = time.time()
        override = self._run_search(battle, policy_probs, action_mask, policy_greedy)
        ms = (time.time() - t0) * 1000

        if override is not None:
            self._stats['search_decisions'] += 1
            self._stats['total_search_time_ms'] += ms
            self._stats['override_count'] += 1
            if self.debug:
                print(f"  Final: {ACTION_TO_MOVE.get(policy_greedy)} → "
                      f"{ACTION_TO_MOVE.get(override)}")
            return self.action_to_order(override, battle)

        self._stats['fallback_decisions'] += 1
        return self.action_to_order(policy_action, battle)

    def get_search_stats(self) -> dict:
        t = self._stats['total_decisions']
        s = self._stats['search_decisions']
        return {
            **self._stats,
            'avg_search_time_ms': self._stats['total_search_time_ms'] / max(s, 1),
            'search_rate': s / max(t, 1),
            'override_rate': self._stats['override_count'] / max(s, 1),
        }


def run_evaluate(args):
    """Evaluation matching evaluate.py's exact approach."""
    try:
        resp = requests.get(f"{args.server}/health", timeout=2)
        print(f"Search server: {'OK' if resp.status_code == 200 else resp.status_code}")
    except Exception:
        print(f"WARNING: Search server unavailable at {args.server}")

    server_config = ServerConfiguration(
        f"ws://127.0.0.1:{args.ps_port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    if args.opponent == "simple":
        opp = SimpleHeuristicsPlayer(
            battle_format="gen9randombattle",
            server_configuration=server_config,
            start_listening=False,
        )
    else:
        opp = RandomPlayer(
            battle_format="gen9randombattle",
            server_configuration=server_config,
            start_listening=False,
        )

    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
    from envs.wrappers import MaskablePokeEnvWrapper
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.monitor import Monitor

    uid = uuid.uuid4().hex[:6]

    rl_env = RL6v6PlayerV18(
        battle_format="gen9randombattle",
        data_dir="data",
        server_configuration=server_config,
        account_configuration1=AccountConfiguration.generate(f"MCTS-{uid}", rand=True),
        account_configuration2=AccountConfiguration.generate(f"Opp-{uid}", rand=True),
    )

    wrapped = SingleAgentWrapper(rl_env, opp)
    masked_env = MaskablePokeEnvWrapper(wrapped, rl_env)
    tl_env = TimeLimit(masked_env, max_episode_steps=500)
    mon_env = Monitor(tl_env)
    vec_env = DummyVecEnv([lambda: mon_env])

    model = load_model(args.model, vec_env, "data")

    print(f"\nWang-MCTS vs {args.opponent} ({args.battles} battles)")
    print(f"  Model: {args.model}")
    print(f"  PS server: ws://127.0.0.1:{args.ps_port}")
    print(f"  Override threshold: {args.override_threshold}, Min visits: {args.min_visits}\n")

    # Search setup
    search_url = args.server
    n_opp_samples = args.opponent_samples
    override_threshold = args.override_threshold
    min_visits = args.min_visits
    do_search = override_threshold < 900

    stats = {'total': 0, 'overrides': 0}

    wins = 0
    total = 0
    for i in range(args.battles):
        obs = vec_env.reset()
        done = False
        while not done:
            # action_masks via env_method (matching evaluate.py)
            masks = vec_env.env_method("action_masks")[0]
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            action_int = int(action[0])

            # MCTS override (if enabled)
            if do_search and np.sum(masks) > 1:
                stats['total'] += 1
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                mask_t = torch.as_tensor(masks.astype(bool), dtype=torch.bool).unsqueeze(0)
                with torch.no_grad():
                    dist = model.policy.get_distribution(obs_t)
                    logits = dist.distribution.logits.clone()
                    logits[~mask_t] = float('-inf')
                    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                greedy = int(np.argmax(probs))

                battle = rl_env.battle1
                if battle is not None:
                    override = _run_search_standalone(
                        battle, probs, masks.astype(float),
                        greedy, model.policy, search_url, n_opp_samples,
                        override_threshold, min_visits,
                        debug=args.debug,
                    )
                    if override is not None:
                        action_int = override
                        action = np.array([action_int])
                        stats['overrides'] += 1

            obs, reward, done, info = vec_env.step(action)
            done = done[0]

        total += 1
        if reward[0] > 0:
            wins += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.battles} — running win rate: {wins/total*100:.1f}%")

    print(f"\n{'='*50}")
    print(f"Result: {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"{'='*50}")
    if do_search:
        print(f"\nSearch decisions: {stats['total']}")
        print(f"Overrides: {stats['overrides']} "
              f"({stats['overrides']/max(stats['total'],1)*100:.1f}%)")


def _run_search_standalone(battle, policy_probs, action_mask, policy_greedy,
                           nn_policy, search_url, n_opp_samples,
                           threshold, min_visits, debug=False):
    """Standalone search function — not tied to any class."""
    legal_actions = np.where(action_mask)[0]
    if len(legal_actions) <= 1:
        return None

    nodes = {}
    action_to_sim = {}
    for aidx in legal_actions:
        a = int(aidx)
        move_str = ACTION_TO_MOVE.get(a, "move 1")
        nodes[a] = MCTSNode(action=move_str, action_idx=a, prior=policy_probs[a])
        action_to_sim[a] = move_str

    unique_sim_moves = list(set(action_to_sim.values()))

    opp = battle.opponent_active_pokemon
    if opp and not opp.fainted and opp.moves:
        n = min(len(opp.moves), 4)
        opp_moves = [f"move {i+1}" for i in range(max(n, 1))]
    else:
        opp_moves = ["move 1"]

    if len(opp_moves) > n_opp_samples:
        opp_sample = list(np.random.choice(opp_moves, n_opp_samples, replace=False))
    else:
        opp_sample = opp_moves

    search_state = battle_to_search_request(battle)
    search_state["movePairs"] = [
        {"p1": p1, "p2": p2}
        for p1 in unique_sim_moves for p2 in opp_sample
    ]

    try:
        resp = requests.post(f"{search_url}/simulate-batch",
                             json=search_state, timeout=5.0)
        if resp.status_code != 200:
            return None
        response = resp.json()
    except Exception:
        return None

    results = response.get("results", [])
    observations, result_map = [], []
    for i, r in enumerate(results):
        if r.get("success"):
            observations.append(state_to_observation(r["state"]))
            result_map.append(i)

    if not observations:
        return None

    with torch.no_grad():
        obs_t = torch.as_tensor(np.stack(observations), dtype=torch.float32)
        values = nn_policy.predict_values(obs_t).squeeze(-1).tolist()
        if not isinstance(values, list):
            values = [values]

    value_lookup = {}
    for obs_idx, res_idx in enumerate(result_map):
        r = results[res_idx]
        st = r.get("state", {})
        if st.get("ended"):
            w = st.get("winner", "")
            v = 1.0 if "SearchP1" in str(w) else (-1.0 if w else 0.0)
        else:
            v = values[obs_idx]
        value_lookup[(r["p1Move"], r["p2Move"])] = v

    for aidx, node in nodes.items():
        sim_move = action_to_sim[aidx]
        for p2 in opp_sample:
            if (sim_move, p2) in value_lookup:
                node.update(value_lookup[(sim_move, p2)])

    policy_node = nodes.get(policy_greedy)
    if policy_node is None or policy_node.visit_count == 0:
        return None

    # Standardize Q-values across all actions so the threshold operates
    # on a meaningful scale regardless of the value network's raw range.
    # threshold=1.0 means "search best must be 1 std dev above policy action"
    q_values = [n.mean_value for n in nodes.values() if n.visit_count > 0]
    if len(q_values) < 2:
        return None

    q_mean = np.mean(q_values)
    q_std = np.std(q_values)
    if q_std < 1e-6:
        return None  # All Q-values identical — no signal

    best_action = max(nodes.keys(), key=lambda a: nodes[a].mean_value)
    best_q = nodes[best_action].mean_value
    policy_q = policy_node.mean_value

    # Normalized advantage: how many std devs is best above policy?
    normalized_advantage = (best_q - policy_q) / q_std

    should_override = (
        best_action != policy_greedy
        and normalized_advantage > threshold
        and nodes[best_action].visit_count >= min_visits
    )

    if debug:
        print(f"\n  MCTS (turn {battle.turn}) [Q range: {min(q_values):.2f}..{max(q_values):.2f}, "
              f"std={q_std:.3f}]:")
        for aidx in sorted(nodes.keys()):
            n = nodes[aidx]
            tags = ""
            if aidx == best_action and should_override: tags += " ←OVERRIDE"
            if aidx == policy_greedy: tags += " ←POLICY"
            print(f"    {n.action:25s}  Q={n.mean_value:+.4f}  "
                  f"N={n.visit_count:2d}  P={n.prior:.3f}{tags}")
        if not should_override and best_action != policy_greedy:
            print(f"    [no override: normalized Δ={normalized_advantage:+.2f}σ "
                  f"< threshold={threshold}σ]")

    return best_action if should_override else None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_pokemon_6v6_v18.2")
    p.add_argument("--battles", type=int, default=100)
    p.add_argument("--opponent", default="simple", choices=["simple", "random"])
    p.add_argument("--server", default="http://localhost:9002")
    p.add_argument("--ps-port", type=int, default=8000)
    p.add_argument("--opponent-samples", type=int, default=5)
    p.add_argument("--override-threshold", type=float, default=0.15)
    p.add_argument("--min-visits", type=int, default=2)
    p.add_argument("--debug", action="store_true")
    run_evaluate(p.parse_args())