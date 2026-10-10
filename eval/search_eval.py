"""Evaluate the search bot (search/policy.py) against SimpleHeuristics, RandomPlayer, a model or
another search bot, with eval/v20_eval.play_games (the same env and win counting as v20 models).
Against Foul Play: eval/foulplay_eval.py --model search.

    python3 -m eval.search_eval --opponents simple --battles 1000 --port 8123
    python3 -m eval.search_eval --opponents model --opp-model models/ppo_pokemon_6v6_v20.pt --battles 1000 --seat both --port 8123
    # a Phase 2 A/B: the network's prior against the Phase 1 bot, both on a fixed iteration budget
    python3 -m eval.search_eval --opponents search --config '{"prior_weight": 0.5}' \
        --budget-iters-per-ms 600 --battles 500 --seat both --port 8123

--config / --opp-config: JSON of SearchPolicy arguments for our bot / the opponent search bot
(Phase 1 defaults when empty); "net": true loads v21 (search/net.py); "net_path": "<a v21 .pt>" uses
that network instead (e.g. a distilled one, training/distill_v21.py).

The search runs for a fixed wall-clock time per decision: run one eval at a time on an otherwise idle
Mac, or the budget buys fewer iterations.
"""
import argparse
import time

import numpy as np
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from eval.v20_eval import play_games, print_actions
from players.v20_player import load_opponent
from search.policy import SearchPolicy


_NET = {}


def make_policy(args, config, seed):
    cfg = dict(samples=args.samples, ms=args.ms, early_samples=args.early_samples, early_ms=args.early_ms,
               threads=args.threads, deterministic=args.deterministic, seed=seed,
               iters_per_ms=args.budget_iters_per_ms)
    cfg.update(config)
    path = cfg.pop("net_path", None)
    if cfg.pop("net", False) or path or cfg.get("prior_weight") or cfg.get("set_gamma") or \
            (cfg.get("algo") == "emm" and cfg.get("worst_weight", 1.0) < 1):
        from search.net import V21_PATH
        path = path or V21_PATH
        if path not in _NET:
            import torch
            torch.set_num_threads(1)  # batch-1 inference; several evals share the Mac
            from search.net import NetModel
            _NET[path] = NetModel(path)
        cfg["net"] = _NET[path]
    return SearchPolicy(**cfg)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the search bot")
    ap.add_argument("--opponents", choices=["simple", "random", "model", "search"], default="simple")
    ap.add_argument("--config", default="{}", help="JSON of SearchPolicy arguments for our bot")
    ap.add_argument("--opp-config", default="{}", help="JSON of SearchPolicy arguments for --opponents search")
    ap.add_argument("--budget-iters-per-ms", type=float, default=0.0,
                    help="fixed MCTS iterations per ms of budget instead of wall-clock time (both bots)")
    ap.add_argument("--opp-model", default=None, help="a v20/v21 .pt (greedy) or an SB3 zip")
    ap.add_argument("--battles", type=int, default=200, help="per seat")
    ap.add_argument("--seat", choices=["1", "2", "both"], default="1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--ms", type=int, default=100)
    ap.add_argument("--early-samples", type=int, default=4)
    ap.add_argument("--early-ms", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--deterministic", action="store_true", help="always the most visited option")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import json
    policy = make_policy(args, json.loads(args.config), args.seed)
    if args.opponents == "search":
        opp_seed = None if args.seed is None else args.seed + 1
        name, opp = f"Search({args.opp_config})", make_policy(args, json.loads(args.opp_config), opp_seed)
    elif args.opponents == "simple":
        name, opp = "SimpleHeuristics", SimpleHeuristicsPlayer(battle_format="gen9randombattle", start_listening=False)
    elif args.opponents == "random":
        name, opp = "RandomPlayer", RandomPlayer(battle_format="gen9randombattle", start_listening=False)
    else:
        if not args.opp_model:
            raise SystemExit("--opp-model is required with --opponents model")
        name, opp = f"Model({args.opp_model})", load_opponent(args.opp_model, deterministic=True)
    seats = {"1": [1], "2": [2], "both": [1, 2]}[args.seat]
    budget = f"{args.budget_iters_per_ms:g} iterations per ms" if args.budget_iters_per_ms else "wall clock"
    print(f"Search bot {args.config}: {args.samples} x {args.ms} ms ({args.early_samples} x {args.early_ms} ms early, "
          f"{budget}), threads {args.threads}, {'deterministic' if args.deterministic else 'near-best sampling'} | "
          f"vs {name} | {args.battles} games per seat, seats {seats}", flush=True)
    wins = games = 0
    t0 = time.time()
    for seat in seats:
        r = play_games(policy, opp, args.battles, args.port, seat=seat, name="se", progress=50)
        wins += r["wins"]
        games += r["games"]
        print(f"  seat {seat}: {r['win_rate']:.1%} ± {r['ci_95']:.1%} | W/L/T {r['wins']}/{r['losses']}/{r['ties']} | "
              f"{r['mean_len']:.1f} decisions per game", flush=True)
        print_actions(r["action_counts"])
    p = wins / max(games, 1)
    print(f"\nvs {name}: {wins}/{games} = {p:.1%} ± {1.96 * np.sqrt(p * (1 - p) / max(games, 1)):.1%} (95% CI) "
          f"in {(time.time() - t0) / 60:.1f} min")
    print("search:", policy.report())
    if args.opponents == "search":
        print("opponent search:", opp.report())


if __name__ == "__main__":
    main()
