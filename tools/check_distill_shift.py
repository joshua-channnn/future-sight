"""Do a network's moves get better by the search's measure on the network's own positions?
(agent_docs/step7.md, distillation.) A network plays v20 (greedy); at each of its decisions with more
than one legal action the search (the best bot's settings, fixed budget) labels the position; then
for the playing network and every --judge network, the regret of its greedy pick by the search's own
per-action values (q of the best visited action minus q of the pick) and its agreement with the
search's most visited action.

    python3 -m tools.check_distill_shift --model models/v21_distill_pA.pt --games 100 \\
        --judge models/ppo_pokemon_6v6_v21.pt --port 8123
"""
import argparse

import numpy as np
import torch

from envs.rl_player_v20 import action_mask
from eval.v20_eval import play_games
from players.v20_player import V20Policy, load_opponent
from search.net import NetModel
from search.policy import SearchPolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the v21 .pt that plays")
    ap.add_argument("--judge", nargs="*", default=[], help="other v21 .pt files scored on the same positions")
    ap.add_argument("--opponent", default="models/ppo_pokemon_6v6_v20.pt")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    torch.set_num_threads(1)
    player = V20Policy.load(args.model, deterministic=True)
    judges = {p: V20Policy.load(p, deterministic=True) for p in args.judge}
    search = SearchPolicy(net=NetModel(), prior_weight=0.15, set_gamma=0.5, infer_items=True, iters_per_ms=600)
    records = []

    class Labeled:
        def action(self, battle):
            a = int(player.action(battle))
            if action_mask(battle).sum() > 1:
                search.last = None
                search.search(battle)
                last = search.last
                if last and last["q"]:
                    q = np.full(14, np.nan)
                    for k, v in last["q"].items():
                        q[k] = v
                    top = max(last["shares"], key=last["shares"].get)
                    picks = {args.model: a, **{p: int(j.action(battle)) for p, j in judges.items()}}
                    records.append((q, top, picks))
            return a

    r = play_games(Labeled(), load_opponent(args.opponent, deterministic=True), args.games, args.port, seat=1)
    print(f"{args.model} vs {args.opponent}: {r['wins']}/{r['games']} won | {len(records)} labeled decisions")
    for name in [args.model] + args.judge:
        reg = np.array([np.nanmax(q) - q[p[name]] for q, _, p in records])   # NaN: an unvisited pick
        agree = np.mean([p[name] == top for _, top, p in records])
        print(f"  {name:<44} regret mean {np.nanmean(reg):.4f} | > 0.05 {np.nanmean(reg > 0.05):5.1%} | "
              f"most visited {agree:5.1%} | unvisited picks {np.mean(np.isnan(reg)):.2%}")


if __name__ == "__main__":
    main()
