"""players/ladder_bot.py with several ladder games at once (poke-env's max_concurrent_battles), for
the search bot's ladder runs (agent_docs/ladder.md). The account and the bot are ladder_bot.py's;
this sets how many games are in progress at once, prints that every minute and stops queuing new
games when asked to or when the Mac's battery runs low.

    python3 -m players.ladder_parallel --concurrent 3 --games 100
    python3 -m players.ladder_parallel --games 200 --stop-file logs/ladder.stop

--set-prior: the search's set prior. Default: data/gen9_randbats_set_counts_live.json, built by
tools/build_set_prior.py from a current Showdown checkout, whose sets match the public server's
(agent_docs/ladder.md), else the fork's (data/gen9_randbats_set_counts.json, which
players/ladder_bot.py run on its own uses).
--stop-file: once this file exists, no new games are queued; the games in progress finish and the
run ends with its results. Stopping the process instead forfeits the games in progress.
--min-battery: on battery power below this charge (%), no new games are queued until the Mac is on
AC power again; the games in progress finish (default 30; 0: off).

If the server drops the connection (the Mac slept: the lid closed), the games in progress are lost
and it exits with 3; so it does when no game has been in progress for 10 minutes (the server lost or
refused the search), with nothing lost. tools/ladder_loop.sh starts it again once the lid is open.

The search is single-threaded, so overlapping games wait for each other's decisions (~0.6 s each),
which is nothing against people's pace. Messages to the server go at least 0.65 s apart: it takes a
regular account's messages one per 0.6 s and drops any that come while 5 are waiting, which is what
happened when three games ended at once (each game's /leave, the next /search, /timer on and the
moves of the others): the /search was dropped and the run waited for a game that never came.
"""
import argparse
import asyncio
import os
import re
import subprocess
import sys
import time

from poke_env.concurrency import create_in_poke_loop, handle_threaded_coroutines

import players.ladder_bot as ladder_bot
from search.sampling import DATA_DIR, use_prior

LIVE_PRIOR = DATA_DIR / "gen9_randbats_set_counts_live.json"
MESSAGE_GAP = 0.65   # s between messages to the server (its throttle: one per 0.6 s, 5 waiting)
IDLE_LIMIT = 600     # s with no game in progress (not paused or stopping) before it exits with 3


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def battery():
    """(on battery power, charge %) from pmset; (False, 100) where there's no pmset."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False, 100
    m = re.search(r"(\d+)%", out)
    return "Battery Power" in out, int(m.group(1)) if m else 100


class _Run:
    """Whether to queue new games: not once stop_file exists, nor on battery below min_battery %."""

    def __init__(self, stop_file, min_battery):
        self.stop_file, self.min_battery = stop_file, min_battery
        self.paused = None   # why new games are on hold, None when they aren't

    def stop_requested(self):
        return bool(self.stop_file) and os.path.exists(self.stop_file)

    def low_battery(self):
        """True while on battery power below min_battery %; logs when that starts and ends."""
        reason = None
        if self.min_battery > 0:
            on_battery, charge = battery()
            if on_battery and charge < self.min_battery:
                reason = f"on battery at {charge}%"
        if bool(reason) != bool(self.paused):
            _log(f"paused ({reason}): no new games until the Mac is on AC power" if reason else "resumed")
        self.paused = reason
        return reason is not None


def _pace_messages(client, gap=MESSAGE_GAP):
    """Sends `client`'s messages (moves, /search, /leave, ...) at least `gap` s apart, in order."""
    send = client.send_message
    lock = create_in_poke_loop(asyncio.Lock)
    last = -gap

    async def send_message(*args, **kwargs):
        nonlocal last
        async with lock:
            wait = last + gap - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            last = time.monotonic()
            return await send(*args, **kwargs)

    client.send_message = send_message


async def _ladder(player, n_games, run):
    """poke-env's Player._ladder, checking `run` before queuing each game."""
    await player.ps_client.logged_in.wait()
    for _ in range(n_games):
        while not run.stop_requested() and run.low_battery():
            await asyncio.sleep(60)
        if run.stop_requested():
            _log(f"{run.stop_file} found: no new games; finishing the ones in progress")
            break
        async with player._battle_start_condition:
            await player.ps_client.search_ladder_game(player._format, player.next_team)
            await player._battle_start_condition.wait()
            while player._battle_count_queue.full():
                async with player._battle_end_condition:
                    await player._battle_end_condition.wait()
            await player._battle_semaphore.acquire()
    await player._battle_count_queue.join()


async def _monitor(player, run):
    listening = getattr(player.ps_client, "_listening_coroutine", None)
    idle_since = time.monotonic()
    while True:
        await asyncio.sleep(60)
        battles = list(player.battles.values())   # poke-env's thread adds to the dict
        active = sum(not b.finished for b in battles)
        if listening is not None and listening.done():
            # e.g. the Mac slept (lid closed) and the server dropped the connection; poke-env
            # doesn't reconnect, so the run can't go on
            _log(f"connection to the server lost: {active} games in progress are lost; exiting (3)")
            os._exit(3)
        if active or run.paused or run.stop_requested():
            idle_since = time.monotonic()
        elif time.monotonic() - idle_since >= IDLE_LIMIT:
            # the server dropped the /search (throttled) or refused it (e.g. before a restart), and
            # poke-env waits for its game forever; a new run searches again
            _log(f"no game in progress for {IDLE_LIMIT // 60} minutes: the search was lost; exiting (3)")
            os._exit(3)
        paused = f" (paused: {run.paused})" if run.paused else ""
        _log(f"battles in progress: {active}, finished: {len(battles) - active}{paused}")


def main():
    ap = argparse.ArgumentParser(description="players/ladder_bot.py with several games at once; "
                                             "other arguments (--games, --model) go to it")
    ap.add_argument("--concurrent", type=int, default=3, help="ladder games in progress at once")
    ap.add_argument("--set-prior", default=None,
                    help=f"the search's set prior (default: {LIVE_PRIOR.name}, else the fork's)")
    ap.add_argument("--stop-file", default=None,
                    help="no new games once this file exists; the games in progress finish")
    ap.add_argument("--min-battery", type=int, default=30,
                    help="no new games on battery below this charge (%%) until on AC power; 0: off")
    args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    path = args.set_prior or (LIVE_PRIOR if LIVE_PRIOR.exists() else None)
    if path:
        prior = use_prior(path)
        print(f"Set prior: {path} ({prior.meta.get('teams', 0):,} teams, {len(prior.sets)} species)")
    else:
        print("Set prior: the fork's (data/gen9_randbats_set_counts.json)")
    run = _Run(args.stop_file, args.min_battery)
    make_player = ladder_bot.make_player

    def make_concurrent_player(*a, **kw):
        kw.setdefault("max_concurrent_battles", args.concurrent)
        player = make_player(*a, **kw)
        _pace_messages(player.ps_client)
        player.ladder = lambda n_games: handle_threaded_coroutines(_ladder(player, n_games, run))
        asyncio.get_running_loop().create_task(_monitor(player, run))
        return player

    ladder_bot.make_player = make_concurrent_player   # ladder_bot.main() looks it up when it runs
    asyncio.run(ladder_bot.main())


if __name__ == "__main__":
    main()
