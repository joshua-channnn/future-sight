"""Evaluation games for v20 checkpoints: a V20Policy on one side of an RLEnvV20, any opponent on
the other. Used by the training loop's evals and by eval/ppo_test.py. Wins are battle.won."""
import uuid
from collections import Counter

import numpy as np
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from envs.rl_player_v20 import RLEnvV20, disable_turn_snapshots
from players.v20_player import opponent_decision

ACTION_NAMES = ([f"Move {i + 1}" for i in range(4)] + [f"Move {i + 1}+Tera" for i in range(4)]
                + [f"Switch to slot {k + 1}" for k in range(6)])


def server_config(port):
    return ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket",
                               "https://play.pokemonshowdown.com/action.php?")


def play_games(policy, opponent, n_games, port, seat=1, name="ev", progress=None):
    """policy: V20Policy. opponent: anything opponent_decision accepts (V20Policy, PPOPlayer,
    SimpleHeuristicsPlayer, RandomPlayer, ...). seat 1: policy is agent1 (the challenger, p1);
    seat 2: agent2. Returns a dict with wins/losses/ties (policy's side), games, mean_len
    (policy decisions per game) and action_counts."""
    disable_turn_snapshots()
    uid = uuid.uuid4().hex[:4]
    env = RLEnvV20(server_configuration=server_config(port),
                   account_configuration1=AccountConfiguration.generate(f"{name}a{uid}", rand=True),
                   account_configuration2=AccountConfiguration.generate(f"{name}b{uid}", rand=True))
    wins = losses = ties = 0
    lens = []
    actions = Counter()
    try:
        for g in range(n_games):
            env.reset()
            a1, a2 = env.agent1.username, env.agent2.username
            n = 0
            while not env.battle1.finished:
                acts = {a1: 0, a2: 0}
                for agent, battle, to_move, mine in ((a1, env.battle1, env.agent1_to_move, seat == 1),
                                                     (a2, env.battle2, env.agent2_to_move, seat == 2)):
                    if not to_move or battle.finished:
                        continue
                    if mine:
                        a = policy.action(battle)
                        actions[a] += 1
                        acts[agent] = a
                        n += 1
                    else:
                        acts[agent] = opponent_decision(opponent, battle)
                env.step(acts)
            lens.append(n)
            won = (env.battle1 if seat == 1 else env.battle2).won
            wins += won is True
            losses += won is False
            ties += won is None
            if progress and (g + 1) % progress == 0:
                print(f"    {g + 1}/{n_games} - running win rate: {wins / (g + 1):.1%}", flush=True)
    finally:
        # without waiting, so a stuck battle can't block the caller
        env.close(force=True, wait=False)
    games = wins + losses + ties
    return {"wins": wins, "losses": losses, "ties": ties, "games": games,
            "win_rate": wins / max(games, 1),
            "ci_95": 1.96 * np.sqrt(max(wins / max(games, 1) * (1 - wins / max(games, 1)), 0) / max(games, 1)),
            "mean_len": float(np.mean(lens)) if lens else 0.0, "action_counts": dict(actions)}


def print_actions(action_counts):
    total = sum(action_counts.values())
    if not total:
        return
    moves = sum(action_counts.get(i, 0) for i in range(4))
    tera = sum(action_counts.get(i, 0) for i in range(4, 8))
    sw = sum(action_counts.get(i, 0) for i in range(8, 14))
    print(f"\n  Action distribution ({total:,} decisions): moves {100 * moves / total:.1f}%, "
          f"moves with tera {100 * tera / total:.1f}%, switches {100 * sw / total:.1f}%")
    for a in sorted(action_counts):
        c = action_counts[a]
        print(f"    {ACTION_NAMES[a]:>18}: {c:>7,} ({100 * c / total:5.1f}%) {'█' * int(40 * c / total)}")
