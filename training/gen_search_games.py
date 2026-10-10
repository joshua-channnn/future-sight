"""Search self-play data (agent_docs/step7_brief.md, step 1): the step 6 search bot plays itself, both
sides recorded, for distilling the search into the v21 network (step 2) and for a learned leaf
evaluator (step 3). Data collection only: nothing learns while it runs.

W worker processes (forkserver, as training/ppo_v20.py), each with its own RLEnvV20 on a local
Showdown server and two SearchPolicy objects with the best bot's settings (agent_docs/step6.md: the
v21 prior at 0.15, the aux-head set model at gamma 0.5, item inference) on a fixed budget of 600 MCTS
iterations per ms (equal to Foul Play's wall-clock budget, so W workers sharing the Mac don't weaken
the search), sharing one NetModel. Per decision, per side, one sample:

  idx int16[188], flt float16[1755], mask bool[14]   the v21 observation (NetModel's featurizer)
  target float32[14]  the search's visit shares summed over the sampled teams, taken before the
                      network's prior is added, normalized over the legal actions; one-hot on the
                      action played when the decision wasn't searched (searched False: one legal
                      action, Revival Blessing, or the search fell back)
  searched bool, action int8 (the near-best pick actually played), value float32 (the root's mean
  MCTS score for the decider, 0..1; NaN when not searched), q float32[14] (each action's mean MCTS
  score over its visits; NaN without visits), iterations int32, samples int8
  lab int16[43]       v21's aux labels: set_labels and, when both sides chose at this step,
                      next_action_label at LAB_NEXT; all -1 in Illusion games
  outcome int8        +1 / -1 / 0 from the decider's battle.won
  game int32 (row of the shard's game table g_*), side int8 (1 = p1, the challenger), turn int16,
  step int16 (the env step within the game: both sides decided at the same step when they share it)
and in the shard's .states.gz one line per sample: the true engine state (both sides exact, the
decider as side one: poke_engine.State.from_string), or an empty line (--state-every, or the build
failed).

Output: datasets/search_selfplay/<run>_w<worker>-<start>_<seq>.npz + .states.gz of --shard-games games
each (fewer when a worker stops), written atomically, the .npz last (a shard is complete once its
.npz exists); meta_<run>.json holds the settings and the git commit. A restart is a new run that only
adds shards.

Robustness for multi-day runs: no websocket keepalive pings (poke-engine holds the GIL during a
search, and a long one starved poke-env's ping handling: the connection dropped and one step 6 eval
crashed with EnvTimeout); a game that raises anyway (EnvTimeout: no battle update for 60 s) is dropped
and the env rebuilt; a dead worker is restarted; between games the run pauses while the Mac is on
battery (pmset) and stops when free disk space falls below --min-free-gb; SIGTERM / Ctrl+C: every
worker finishes its game, writes its shard and exits.

    cd ../pokemon-showdown && node pokemon-showdown start --no-security --port 8123
    python3 -m training.gen_search_games --workers 1 --games-per-worker 2 --out datasets/test   # test
    nohup python3 -m training.gen_search_games --workers 5 --hours 60 > logs/gen_search.log 2>&1 &
    python3 -m tools.check_search_games datasets/search_selfplay                               # check
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import queue as _queue
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime

import numpy as np

OUT_DIR = "datasets/search_selfplay"
# the best bot of step 6 (agent_docs/step6.md), on the fixed budget (iters_per_ms) instead of wall clock
SEARCH_CONFIG = {"prior_weight": 0.15, "set_gamma": 0.5, "infer_items": True, "iters_per_ms": 600}
NET_PATH = "models/ppo_pokemon_6v6_v21.pt"
N_ACTIONS = 14


def server_config(port):
    from poke_env.ps_client.server_configuration import ServerConfiguration
    return ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket",
                               "https://play.pokemonshowdown.com/action.php?")


class GameTooLong(Exception):
    pass


# =============================================================================================
# Worker
class Worker:
    def __init__(self, wid, start_n, cfg, queue):
        import torch
        torch.set_num_threads(1)
        from envs.rl_player_v20 import disable_turn_snapshots
        disable_turn_snapshots()
        from search.net import NetModel
        from search.policy import SearchPolicy
        self.wid, self.cfg, self.queue = wid, cfg, queue
        self.port = cfg["ports"][wid % len(cfg["ports"])]
        self.net = NetModel(cfg["net_path"])
        self.fz = self.net.fz
        self.policies = {side: SearchPolicy(net=self.net, **cfg["search"]) for side in (1, 2)}
        self.name = f"{cfg['run']}_w{wid}-{start_n}"
        self.seq = 0
        self.env = None
        self.errors = Counter()
        self.games = []     # finished games not written yet: (game info, [sample dicts])
        self.n_played = 0
        self.k = 0          # decisions so far (for --state-every)

    def new_env(self):
        from envs.rl_player_v20 import RLEnvV20
        from poke_env.ps_client import AccountConfiguration
        if self.env is not None:
            try:
                self.env.close(force=True, wait=False)
            except Exception:
                pass
        uid = uuid.uuid4().hex[:4]
        # no keepalive pings: a search holds the GIL, and a late pong closed the connection
        self.env = RLEnvV20(
            server_configuration=server_config(self.port), step_timeout=self.cfg["step_timeout"],
            ping_interval=None, ping_timeout=None,
            account_configuration1=AccountConfiguration.generate(f"g{self.wid}a{uid}", rand=True),
            account_configuration2=AccountConfiguration.generate(f"g{self.wid}b{uid}", rand=True))

    def decide(self, side, battle, other, illusion):
        """Featurize, label, search and play one decision of `side`; returns the sample (dict)."""
        from envs.rl_player_v21 import N_LAB, set_labels
        from search.state import build_true_state
        idx, flt, mask = self.fz.featurize(battle)
        state = ""
        if self.k % self.cfg["state_every"] == 0:
            try:
                state = build_true_state(battle, other)[0].to_string()
            except BaseException as e:  # noqa: BLE001 - poke-engine panics surface as PanicException
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                self.errors["true state failed"] += 1
        self.k += 1
        lab = np.full(N_LAB, -1, dtype=np.int16) if illusion else set_labels(battle, other)
        pol = self.policies[side]
        pol.last = None
        a = int(pol.action(battle))
        last = pol.last
        shares = (last or {}).get("shares") or {}
        tot = float(sum(shares.values()))
        searched = last is not None and tot > 0
        target = np.zeros(N_ACTIONS, dtype=np.float32)
        q = np.full(N_ACTIONS, np.nan, dtype=np.float32)
        if searched:
            for act, s in shares.items():
                target[act] = s / tot
            for act, v in last["q"].items():
                q[act] = v
        else:
            target[a] = 1.0
        return {"idx": idx.astype(np.int16), "flt": flt.astype(np.float16), "mask": mask,
                "target": target, "searched": searched, "action": a,
                "value": float(last["value"]) if searched else float("nan"), "q": q,
                "iterations": int(last["iterations"]) if searched else 0,
                "samples": int(last["samples"]) if searched else 0,
                "lab": lab, "side": side, "turn": battle.turn, "state": state}

    def play_game(self):
        """One complete game, both sides recorded. Returns (game info, samples)."""
        from envs.rl_player_v21 import LAB_NEXT, illusion_game, next_action_label
        env = self.env
        t0 = time.time()
        env.reset()
        a1, a2 = env.agent1.username, env.agent2.username
        illusion = illusion_game(env.battle1, env.battle2)
        samples = []
        n_dec = {1: 0, 2: 0}
        step = 0
        while not env.battle1.finished:
            b1, b2 = env.battle1, env.battle2
            choosing = {1: env.agent1_to_move and not b1.finished, 2: env.agent2_to_move and not b2.finished}
            acts = {a1: 0, a2: 0}
            new = []
            for side, agent, b, other in ((1, a1, b1, b2), (2, a2, b2, b1)):
                if choosing[side]:
                    s = self.decide(side, b, other, illusion)
                    acts[agent] = s["action"]
                    new.append((b, other, s))
                    n_dec[side] += 1
            if not illusion:
                # only now is every choice of this step known (the other side's is the label)
                for b, other, s in new:
                    if choosing[3 - s["side"]]:
                        s["lab"][LAB_NEXT] = next_action_label(b, other, acts[a2 if s["side"] == 1 else a1])
            for _, _, s in new:
                s["step"] = step
                samples.append(s)
            if max(n_dec.values()) > self.cfg["max_decisions"]:
                raise GameTooLong(f"more than {self.cfg['max_decisions']} decisions per side")
            env.step(acts)
            step += 1
        won = {1: env.battle1.won, 2: env.battle2.won}
        outcome = {side: 1 if w is True else (-1 if w is False else 0) for side, w in won.items()}
        for s in samples:
            s["outcome"] = outcome[s["side"]]
        info = {"tag": env.battle1.battle_tag, "turns": env.battle1.turn,
                "winner": 1 if won[1] is True else (2 if won[2] is True else 0),
                "n1": n_dec[1], "n2": n_dec[2], "illusion": illusion, "seconds": time.time() - t0,
                "time": t0}
        return info, samples

    def write_shard(self):
        if not self.games:
            return
        out = self.cfg["out"]
        base = os.path.join(out, f"{self.name}_{self.seq:04d}")
        rows = [(g, s) for g, (_, samples) in enumerate(self.games) for s in samples]
        arrays = {
            "idx": np.stack([s["idx"] for _, s in rows]),
            "flt": np.stack([s["flt"] for _, s in rows]),
            "mask": np.stack([s["mask"] for _, s in rows]),
            "target": np.stack([s["target"] for _, s in rows]),
            "searched": np.array([s["searched"] for _, s in rows], dtype=bool),
            "action": np.array([s["action"] for _, s in rows], dtype=np.int8),
            "value": np.array([s["value"] for _, s in rows], dtype=np.float32),
            "q": np.stack([s["q"] for _, s in rows]),
            "iterations": np.array([s["iterations"] for _, s in rows], dtype=np.int32),
            "samples": np.array([s["samples"] for _, s in rows], dtype=np.int8),
            "lab": np.stack([s["lab"] for _, s in rows]),
            "outcome": np.array([s["outcome"] for _, s in rows], dtype=np.int8),
            "game": np.array([g for g, _ in rows], dtype=np.int32),
            "side": np.array([s["side"] for _, s in rows], dtype=np.int8),
            "turn": np.array([s["turn"] for _, s in rows], dtype=np.int16),
            "step": np.array([s["step"] for _, s in rows], dtype=np.int16),
        }
        infos = [g for g, _ in self.games]
        arrays.update({
            "g_tag": np.array([g["tag"] for g in infos]),
            "g_worker": np.full(len(infos), self.wid, dtype=np.int16),
            "g_turns": np.array([g["turns"] for g in infos], dtype=np.int16),
            "g_winner": np.array([g["winner"] for g in infos], dtype=np.int8),
            "g_n1": np.array([g["n1"] for g in infos], dtype=np.int16),
            "g_n2": np.array([g["n2"] for g in infos], dtype=np.int16),
            "g_illusion": np.array([g["illusion"] for g in infos], dtype=bool),
            "g_seconds": np.array([g["seconds"] for g in infos], dtype=np.float32),
            "g_time": np.array([g["time"] for g in infos], dtype=np.float64),
        })
        tmp_s, tmp_n = base + ".states.gz.tmp", base + ".npz.tmp"
        with gzip.open(tmp_s, "wt", compresslevel=6) as f:
            for _, s in rows:
                f.write(s["state"].replace("\n", " "))
                f.write("\n")
        with open(tmp_n, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp_s, base + ".states.gz")
        os.replace(tmp_n, base + ".npz")
        self.seq += 1
        size = os.path.getsize(base + ".npz"), os.path.getsize(base + ".states.gz")
        self.queue.put(("shard", self.wid, os.path.basename(base), len(infos), len(rows), size))
        self.games = []

    def run(self, stop, pause, term, parent_pid):
        cfg = self.cfg
        fails = 0
        try:
            while not (stop.is_set() or term["flag"]):
                if cfg["games_per_worker"] and self.n_played >= cfg["games_per_worker"]:
                    break
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    break  # the main process is gone
                if pause.is_set():
                    while pause.is_set() and not (stop.is_set() or term["flag"]):
                        time.sleep(2)
                    if self.env is not None:
                        self.new_env()  # connections may not survive the Mac sleeping
                    continue
                if self.env is None:
                    self.new_env()
                errors_before = Counter(self.errors)
                try:
                    info, samples = self.play_game()
                except Exception as e:
                    name = type(e).__name__
                    self.errors[f"game abandoned ({name})"] += 1
                    n = self.errors[f"game abandoned ({name})"]
                    if n <= 3 or n % 50 == 0:
                        print(f"[worker {self.wid}] game abandoned ({name}: {str(e)[:200]}); rebuilding env",
                              flush=True)
                    self.queue.put(("abandoned", self.wid, name))
                    self.new_env()
                    fails += 1
                    time.sleep(min(60, 2 ** min(fails, 6)))  # back off while e.g. the server is down
                    continue
                fails = 0
                self.n_played += 1
                self.games.append((info, samples))
                stats = Counter()
                for pol in self.policies.values():
                    stats.update(pol.stats)
                    pol.stats.clear()
                    pol.latency.clear()
                    pol.sample_iters.clear()
                new_errors = self.errors - errors_before
                self.queue.put(("game", self.wid, {
                    "samples": len(samples), "searched": sum(s["searched"] for s in samples),
                    "iters": sum(s["iterations"] for s in samples), "winner": info["winner"],
                    "n1": info["n1"], "n2": info["n2"], "seconds": info["seconds"], "turns": info["turns"],
                    "illusion": info["illusion"], "states": sum(bool(s["state"]) for s in samples),
                    "policy_stats": {k: v for k, v in stats.items() if k != "searched"},
                    "errors": dict(new_errors)}))
                if len(self.games) >= cfg["shard_games"]:
                    self.write_shard()
        finally:
            try:
                self.write_shard()
            except Exception:
                self.queue.put(("error", self.wid, "shard write failed:\n" + traceback.format_exc()))
            if self.env is not None:
                try:
                    self.env.close(force=True, wait=False)
                except Exception:
                    pass


def worker_main(wid, start_n, cfg, queue, stop, pause, parent_pid):
    # the main process stops the workers through `stop` (each finishes its game and writes its
    # shard); Ctrl+C reaches the whole process group, so workers ignore it
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    term = {"flag": False}
    signal.signal(signal.SIGTERM, lambda signum, frame: term.__setitem__("flag", True))
    os.chdir(cfg["cwd"])
    try:
        w = Worker(wid, start_n, cfg, queue)
    except Exception:
        queue.put(("error", wid, "worker failed to start:\n" + traceback.format_exc()))
        queue.put(("exit", wid, 1))
        return
    try:
        w.run(stop, pause, term, parent_pid)
    except Exception:
        queue.put(("error", wid, traceback.format_exc()))
    queue.put(("exit", wid, 0))


# =============================================================================================
# Main process: workers, power, disk, progress
def on_ac_power():
    fake = os.environ.get("GEN_FAKE_BATTERY_FILE")   # tests: "on battery" while this file exists
    if fake:
        return not os.path.exists(fake)
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return True
    return "AC Power" in out or "Battery Power" not in out


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def git_info():
    def run(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def file_sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Progress:
    def __init__(self):
        self.games = self.samples = self.searched = self.iters = self.states = 0
        self.wins = Counter()
        self.dec = Counter()
        self.game_s = 0.0
        self.illusion = 0
        self.policy_stats = Counter()
        self.errors = Counter()
        self.shards = 0
        self.shard_bytes = [0, 0]
        self.shard_samples = 0
        self.window = []   # (time, samples) for the recent rate

    def add_game(self, g):
        self.games += 1
        self.samples += g["samples"]
        self.searched += g["searched"]
        self.iters += g["iters"]
        self.states += g["states"]
        self.wins[g["winner"]] += 1
        self.dec[1] += g["n1"]
        self.dec[2] += g["n2"]
        self.game_s += g["seconds"]
        self.illusion += g["illusion"]
        self.policy_stats.update(g["policy_stats"])
        self.errors.update(g["errors"])

    def line(self, active_s, n_alive, paused):
        now = time.time()
        self.window.append((now, self.samples))
        while len(self.window) > 2 and now - self.window[0][0] > 3600:
            self.window.pop(0)
        t_w, s_w = self.window[0]
        recent = (self.samples - s_w) / max(now - t_w, 1e-6) if now - t_w > 1 else 0.0
        g = max(self.games, 1)
        decided = self.wins[1] + self.wins[2]
        p1 = self.wins[1] / max(decided, 1)
        ci = 1.96 * np.sqrt(p1 * (1 - p1) / max(decided, 1))
        size = sum(self.shard_bytes) / max(self.shard_samples, 1)
        ps = self.policy_stats
        extra = {k: v for k, v in ps.items() if "fell back" in k or "error" in k or "failed" in k}
        extra["prior changed the top option"] = f"{ps['prior: changed the top option'] / max(ps['prior: decisions'], 1):.1%}"
        return (f"[{datetime.now():%m-%d %H:%M}] {active_s / 3600:.2f} h active | {n_alive} workers"
                f"{' PAUSED (battery)' if paused else ''} | games {self.games:,} | samples {self.samples:,} "
                f"({self.searched / max(self.samples, 1):.1%} searched) | {self.samples / max(active_s, 1e-6):.1f} "
                f"samples/s overall, {recent:.1f} last hour | p1 wins {p1:.1%} ± {ci:.1%} (ties {self.wins[0]}) | "
                f"decisions/game {self.dec[1] / g:.1f} + {self.dec[2] / g:.1f} | {self.game_s / g:.1f} s/game | "
                f"{self.iters / max(self.searched, 1) / 1000:.0f}K iterations/search | Illusion games "
                f"{self.illusion} | shards {self.shards} ({sum(self.shard_bytes) / 1e9:.2f} GB, {size / 1024:.2f} "
                f"KB/sample) | errors {dict(self.errors)} | search {extra}")


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ctx = mp.get_context("forkserver")
        self.queue = self.ctx.Queue()
        self.stop = self.ctx.Event()
        self.pause = self.ctx.Event()
        self.procs = {}      # wid -> Process
        self.starts = Counter()
        self.exited = set()
        self.deaths = 0
        self.restart_at = {}
        self.progress = Progress()
        self.stop_reason = None
        self.caffeinate = None

    def start_worker(self, wid):
        self.starts[wid] += 1
        p = self.ctx.Process(target=worker_main, name=f"gen-worker-{wid}", daemon=True,
                             args=(wid, self.starts[wid], self.cfg, self.queue, self.stop, self.pause, os.getpid()))
        p.start()
        self.procs[wid] = p
        self.exited.discard(wid)

    def request_stop(self, reason):
        if not self.stop.is_set():
            self.stop_reason = reason
            print(f">>> stopping ({reason}): each worker finishes its game and writes its shard", flush=True)
            self.stop.set()

    def set_caffeinate(self, on):
        if on and (self.caffeinate is None or self.caffeinate.poll() is not None):
            try:
                self.caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            except Exception:
                self.caffeinate = None
        elif not on and self.caffeinate is not None:
            self.caffeinate.terminate()
            self.caffeinate = None

    def handle(self, msg):
        kind, wid = msg[0], msg[1]
        if kind == "game":
            self.progress.add_game(msg[2])
            if self.cfg["max_samples"] and self.progress.samples >= self.cfg["max_samples"]:
                self.request_stop(f"{self.progress.samples:,} samples")
        elif kind == "abandoned":
            self.progress.errors[f"game abandoned ({msg[2]})"] += 1
        elif kind == "shard":
            _, _, name, n_games, n_samples, size = msg
            p = self.progress
            p.shards += 1
            p.shard_bytes[0] += size[0]
            p.shard_bytes[1] += size[1]
            p.shard_samples += n_samples
            print(f"  [shard] {name}: {n_games} games, {n_samples:,} samples, npz {size[0] / 1e6:.1f} MB + states "
                  f"{size[1] / 1e6:.1f} MB ({(size[0] + size[1]) / max(n_samples, 1) / 1024:.2f} KB/sample)", flush=True)
        elif kind == "error":
            print(f"[worker {wid}] {msg[2]}", flush=True)
        elif kind == "exit":
            self.exited.add(wid)

    def run(self):
        cfg = self.cfg
        t_start = time.time()
        for wid in range(cfg["workers"]):
            self.start_worker(wid)
        active_s, last_t = 0.0, time.time()
        next_power = next_progress = 0.0
        stop_t = None
        paused = False
        while True:
            try:
                self.handle(self.queue.get(timeout=1.0))
                while True:
                    self.handle(self.queue.get_nowait())
            except _queue.Empty:
                pass
            now = time.time()
            if not paused:
                active_s += now - last_t
            last_t = now
            if now >= next_power:
                next_power = now + 60
                ac = on_ac_power() or cfg["ignore_battery"]
                if not ac and not paused:
                    paused = True
                    self.pause.set()
                    self.set_caffeinate(False)
                    print(f"[{datetime.now():%m-%d %H:%M}] on battery: pausing after the current games", flush=True)
                elif ac and paused:
                    paused = False
                    self.pause.clear()
                    print(f"[{datetime.now():%m-%d %H:%M}] on AC power: resuming", flush=True)
                if not paused:
                    self.set_caffeinate(True)
                gb = free_gb(cfg["out"])
                if gb < cfg["min_free_gb"]:
                    self.request_stop(f"{gb:.1f} GB free on disk, below --min-free-gb {cfg['min_free_gb']}")
            if cfg["hours"] and active_s >= 3600 * cfg["hours"]:
                self.request_stop(f"{cfg['hours']} h of generation")
            alive = {wid for wid, p in self.procs.items() if p.is_alive()}
            if now >= next_progress:
                next_progress = now + cfg["progress_every"]
                print(self.progress.line(active_s, len(alive), paused) +
                      f" | disk {free_gb(cfg['out']):.1f} GB free", flush=True)
            if self.stop.is_set():
                stop_t = stop_t or now
                if not alive:
                    break
                if now - stop_t > cfg["stop_timeout"]:
                    print(f">>> workers {sorted(alive)} still running after {cfg['stop_timeout']} s: killing them",
                          flush=True)
                    for wid in alive:
                        self.procs[wid].kill()
                    break
                continue
            # restart workers that died (or exited without being asked to, e.g. after an error)
            for wid, p in list(self.procs.items()):
                if p.is_alive():
                    continue
                if cfg["games_per_worker"] and wid in self.exited:
                    continue  # done with its games
                if wid not in self.restart_at:
                    self.deaths += 1
                    print(f"[{datetime.now():%m-%d %H:%M}] worker {wid} exited (code {p.exitcode}); "
                          f"restarting in 30 s (deaths so far: {self.deaths})", flush=True)
                    self.restart_at[wid] = now + 30
                    if self.deaths > cfg["max_deaths"]:
                        self.request_stop(f"{self.deaths} worker deaths")
                elif now >= self.restart_at[wid]:
                    del self.restart_at[wid]
                    p.join(0)
                    self.start_worker(wid)
            if cfg["games_per_worker"] and all(not p.is_alive() for p in self.procs.values()) and \
                    self.exited >= set(self.procs):
                break
        for p in self.procs.values():
            p.join(5)
        # messages sent just before the workers exited
        try:
            while True:
                self.handle(self.queue.get(timeout=0.5))
        except _queue.Empty:
            pass
        self.set_caffeinate(False)
        print(self.progress.line(active_s, 0, False) + f" | disk {free_gb(cfg['out']):.1f} GB free", flush=True)
        print(f">>> done in {(time.time() - t_start) / 3600:.2f} h ({self.stop_reason or 'all workers finished'})",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description="Search self-play data (agent_docs/step7_brief.md, step 1)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--ports", default="8123", help="comma-separated Showdown server ports (workers share them)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--hours", type=float, default=0.0, help="stop after this much generation (pauses excluded)")
    ap.add_argument("--max-samples", type=int, default=0, help="stop after this many samples")
    ap.add_argument("--games-per-worker", type=int, default=0, help="each worker stops after this many games (tests)")
    ap.add_argument("--shard-games", type=int, default=100)
    ap.add_argument("--state-every", type=int, default=1, help="store the true state of every k-th decision")
    ap.add_argument("--config", default="{}", help="JSON of SearchPolicy arguments over the best bot's")
    ap.add_argument("--net", default=NET_PATH, help="v21 checkpoint for the search's network")
    ap.add_argument("--max-decisions", type=int, default=500, help="per side; longer games are dropped")
    ap.add_argument("--step-timeout", type=float, default=60.0)
    ap.add_argument("--progress-every", type=float, default=600.0, help="seconds between progress lines")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--stop-timeout", type=float, default=600.0, help="seconds to wait for workers on stop")
    ap.add_argument("--max-deaths", type=int, default=50, help="stop after this many worker deaths")
    ap.add_argument("--ignore-battery", action="store_true", help="don't pause on battery")
    args = ap.parse_args()

    run = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    search = dict(SEARCH_CONFIG)
    search.update(json.loads(args.config))
    cfg = {"run": run, "cwd": os.getcwd(), "out": args.out, "workers": args.workers,
           "ports": [int(p) for p in args.ports.split(",")], "hours": args.hours, "max_samples": args.max_samples,
           "games_per_worker": args.games_per_worker, "shard_games": args.shard_games,
           "state_every": max(1, args.state_every), "search": search, "net_path": args.net,
           "max_decisions": args.max_decisions, "step_timeout": args.step_timeout,
           "progress_every": args.progress_every, "min_free_gb": args.min_free_gb,
           "stop_timeout": args.stop_timeout, "max_deaths": args.max_deaths, "ignore_battery": args.ignore_battery}
    os.makedirs(args.out, exist_ok=True)
    import importlib.metadata
    meta = {"run": run, "started": datetime.now().isoformat(), "argv": sys.argv, "git": git_info(),
            "poke_engine": importlib.metadata.version("poke-engine"),
            "poke_env": importlib.metadata.version("poke-env"), "python": sys.version.split()[0],
            "net_sha1": file_sha1(args.net), "config": cfg,
            "format": "training/gen_search_games.py docstring"}
    with open(os.path.join(args.out, f"meta_{run}.json"), "w") as f:
        json.dump(meta, f, indent=1)

    gen = Generator(cfg)

    def on_signal(signum, frame):
        gen.request_stop(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    print(f"Search self-play {run}: {args.workers} workers on ports {cfg['ports']}, search {search}, net "
          f"{args.net} | shards of {args.shard_games} games in {args.out} | states every {cfg['state_every']} "
          f"decision(s) | git {meta['git']['commit'][:8]}{' (dirty)' if meta['git']['dirty'] else ''} | "
          f"{free_gb(args.out):.1f} GB free", flush=True)
    gen.run()


if __name__ == "__main__":
    main()
