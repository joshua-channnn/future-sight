"""players/ladder_bot.py with several ladder games at once (poke-env's max_concurrent_battles), for
the search bot's ladder runs (agent_docs/ladder.md). The account and the bot are ladder_bot.py's;
this only sets how many games are in progress at once and prints that every minute.

    python3 -m players.ladder_parallel --concurrent 3 --games 100

The search is single-threaded, so overlapping games wait for each other's decisions (~0.6 s each),
which is nothing against people's pace. Stopping it mid-run forfeits the games in progress: stop it
right after a game ends (the log's "Battle #" lines).
"""
import argparse
import asyncio
import sys
import time

import players.ladder_bot as ladder_bot


async def _monitor(player):
    while True:
        await asyncio.sleep(60)
        active = sum(not b.finished for b in player.battles.values())
        print(f"[{time.strftime('%H:%M:%S')}] battles in progress: {active}, "
              f"finished: {len(player.battles) - active}")


def main():
    ap = argparse.ArgumentParser(description="players/ladder_bot.py with several games at once; "
                                             "other arguments (--games, --model) go to it")
    ap.add_argument("--concurrent", type=int, default=3, help="ladder games in progress at once")
    args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    make_player = ladder_bot.make_player

    def make_concurrent_player(*a, **kw):
        kw.setdefault("max_concurrent_battles", args.concurrent)
        player = make_player(*a, **kw)
        asyncio.get_running_loop().create_task(_monitor(player))
        return player

    ladder_bot.make_player = make_concurrent_player   # ladder_bot.main() looks it up when it runs
    asyncio.run(ladder_bot.main())


if __name__ == "__main__":
    main()
