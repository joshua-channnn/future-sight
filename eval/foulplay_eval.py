"""Evaluate a trained model against Foul Play (pmariglia's MCTS bot) on a local Showdown server.

Foul Play lives in ../foul-play and runs in Docker (its Rust engine is built inside the image).
It targets current Showdown; against an older server (like the Feb 2026 fork here) it needs the
local patches in ../foul-play: request/log reordering in fp/websocket_client.py and the
Snow -> Snowscape rename in fp/battle/protocol.py.

1. Start a local server:
     cd pokemon-showdown && node pokemon-showdown start --no-security --port 8005
2. Build Foul Play once:
     cd foul-play && docker build . -t foul-play:latest
3. Start Foul Play waiting for challenges. On macOS, Docker reaches the host through
   host.docker.internal. Pick a username that isn't registered on the main Showdown server.
     docker run --rm foul-play:latest \
       --websocket-uri ws://host.docker.internal:8005/showdown/websocket \
       --ps-username FoulPlayLocal7291 --bot-mode accept_challenge \
       --pokemon-format gen9randombattle --run-count 100 --log-level INFO
   Foul Play searches 100 ms per sampled state by default; raise --search-time-ms and
   --search-parallelism for a stronger (slower) opponent.
4. From pokemon-rl/:
     python -m eval.foulplay_eval --games 100 --port 8005 --opponent FoulPlayLocal7291
   --model takes a v20 checkpoint (.pt, the default), an SB3 zip (v17-v19), or "search" for the
   search bot (search/policy.py; --samples / --ms set its budget, Foul Play's default by default).
Foul Play's engine panics now and then ("Invalid PokemonIndex: 6", about once per 500 games) and
its container exits; the script then stops after --game-timeout with the games so far.
tools/foulplay_loop.sh restarts Foul Play until a number of games have finished.
"""
import argparse
import asyncio
import uuid

import numpy as np

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from players.ppo_player import PPOPlayer
from players.v20_player import V20Player
from search.policy import SearchPlayer, SearchPolicy


async def wait_for_battles(player, n_finished, timeout):
    """Wait until player has finished n_finished battles; False on timeout."""
    for _ in range(timeout):
        if player.n_finished_battles >= n_finished:
            return True
        await asyncio.sleep(1)
    return player.n_finished_battles >= n_finished


async def main():
    parser = argparse.ArgumentParser(description="Evaluate a model against a local Foul Play bot")
    parser.add_argument("--model", default="models/ppo_pokemon_6v6_v20.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--opponent", default="FoulPlayLocal7291",
                        help="The --ps-username Foul Play was started with")
    parser.add_argument("--game-timeout", type=int, default=600,
                        help="Seconds to wait for one game before giving up")
    parser.add_argument("--samples", type=int, default=2, help="search bot: sampled teams per decision")
    parser.add_argument("--ms", type=int, default=100, help="search bot: MCTS time per sample")
    parser.add_argument("--early-samples", type=int, default=4)
    parser.add_argument("--early-ms", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="{}",
                        help="search bot: JSON of further SearchPolicy arguments (eval/search_eval.py --config)")
    args = parser.parse_args()

    common = dict(
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration(f"FPEval{uuid.uuid4().hex[:6]}", None),
        server_configuration=ServerConfiguration(
            f"ws://127.0.0.1:{args.port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        max_concurrent_battles=1,
    )
    if args.model == "search":
        import json
        cfg = dict(samples=args.samples, ms=args.ms, early_samples=args.early_samples,
                   early_ms=args.early_ms, seed=args.seed)
        cfg.update(json.loads(args.config))
        if cfg.pop("net", False) or cfg.get("prior_weight") or cfg.get("set_gamma") or \
                (cfg.get("algo") == "emm" and cfg.get("worst_weight", 1.0) < 1):
            import torch
            torch.set_num_threads(1)
            from search.net import NetModel
            cfg["net"] = NetModel()
        policy = SearchPolicy(**cfg)
        player = SearchPlayer(policy=policy, hold_requests=True, **common)
    elif args.model.endswith(".pt"):
        # this local Showdown fork sends each request before the turn's log
        player = V20Player(model_path=args.model, deterministic=True, hold_requests=True, **common)
    else:
        player = PPOPlayer(model_path=args.model, data_dir="data", deterministic=True, **common)

    print(f"Challenging {args.opponent} to {args.games} gen9randombattle games with {args.model}"
          f"{' ' + args.config if args.model == 'search' else ''}...")
    # Foul Play plays one battle at a time and ignores challenges that arrive mid-battle, while
    # send_challenges(n) fires the next challenge as soon as a battle starts. Challenge one at a time.
    for game in range(1, args.games + 1):
        try:   # a challenge to a user who left (Foul Play crashed) is never answered
            await asyncio.wait_for(player.send_challenges(args.opponent, n_challenges=1), args.game_timeout)
        except asyncio.TimeoutError:
            print(f"Challenge {game} wasn't accepted within {args.game_timeout}s (is Foul Play still running?); "
                  f"stopping early.", flush=True)
            break
        if not await wait_for_battles(player, game, args.game_timeout):
            print(f"Game {game} didn't finish within {args.game_timeout}s "
                  f"(is Foul Play still running?); stopping early.")
            break
        if game % 10 == 0:
            print(f"  {game}/{args.games}: {player.n_won_battles} wins", flush=True)
        await asyncio.sleep(2)  # let Foul Play get back to waiting for challenges

    n, wins = player.n_finished_battles, player.n_won_battles
    p = wins / max(n, 1)
    ci = 1.96 * np.sqrt(p * (1 - p) / max(n, 1))
    print(f"vs Foul Play: {wins}/{n} wins = {p:.1%} (95% CI ±{ci:.1%})")
    if args.model == "search":
        print("search:", player.policy.report())


if __name__ == "__main__":
    asyncio.run(main())
