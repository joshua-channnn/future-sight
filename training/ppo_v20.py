"""v20/v21 training: a CleanRL-style PPO loop with worker processes, two-sided self-play and the
network of `CONFIG["arch"]` (networks/registry.py): v20 (envs/rl_player_v20.py,
networks/v20_policy.py) or v21 (envs/rl_player_v21.py, networks/v21_policy.py). Replaces SB3.

Collection: W worker processes, each with one env (two agents) on a Showdown server, a CPU copy
of the current weights and one torch thread. Each update the learner sends the weights and a
sample quota; a worker plays whole games until it has its quota, then returns the samples with
GAE already computed. Workers never wait on each other within an update, only at its end.

Opponents per game: `heuristic_ratio` SimpleHeuristics; the rest are split between the live
policy (both sides recorded: two-sided self-play) and a uniformly chosen pool member (snapshots
of this run, plus the old SB3 seed models); only the learner's side is recorded against those.

PPO follows SB3's MaskablePPO (clipped surrogate, SB3-style value clipping, per-minibatch
advantage normalization, entropy bonus, grad clipping), plus the zero-avoiding term:
KL(uniform over legal actions || policy). v21 adds auxiliary losses (`aux_*_coef`): every sample
carries labels read from the other side's battle object (the opponent's hidden set and its action
at the same decision), and the update adds the weighted cross-entropies to the PPO loss.

    python3 -m training.ppo_v20 --workers 6 --servers 3          # train (resumes automatically)
    python3 -m training.ppo_v20 --arch v21 --workers 6 --servers 3 --stop-at 10000000   # version v21
    python3 -m training.ppo_v20 --opponents heuristic --version v20sh --stop-at 2000000
"""
import argparse
import math
import multiprocessing as mp
import os
import pickle
import random
import shutil
import sys
import time
import traceback
import uuid
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

BASE_PORT = 8000

CONFIG = {
    "version": "v20",                # defaults to the arch when --arch is given without --version
    "arch": "v20",                   # networks/registry.py: v20 or v21
    "total_timesteps": 20_000_000,   # learning-rate schedule horizon (Wang's schedule over 20M)
    "stop_at": None,                 # stop once this many steps are done (None: total_timesteps)

    # Collection
    "n_workers": 6,
    "n_servers": 3,                  # Showdown servers on base_port.. (workers share them)
    "base_port": BASE_PORT,
    "eval_port": None,               # default: base_port + n_servers
    "steps_per_update": 40_960,
    "max_decisions": 500,            # per side; longer games are dropped
    "step_timeout": 60.0,            # a battle with no progress this long is abandoned
    "worker_timeout": 1800.0,        # a worker silent this long in one update is restarted

    # PPO: v19's settings (Wang 2024, Table A.3), kept for the first like-for-like comparison
    "lr_initial": 10 ** -4.23,
    "lr_final": 0.0,
    "lr_decay_exponent": 1.5,        # lr = lr_initial / (8x + 1)^1.5 over progress x
    "embed_lr_mult": 10.0,           # embeddings at 10x, as in v19
    "adam_eps": 1e-5,
    "gamma": 0.9999,
    "gae_lambda": 0.754,
    "clip_range": 0.0829,
    "clip_range_vf": 0.0184,         # None disables (SB3 semantics: clamps each value's change)
    "n_epochs": 7,
    "batch_size": 1024,
    "max_grad_norm": 0.543,
    "ent_coef": 0.0588,
    "vf_coef": 0.4375,
    "zero_avoid_coef": 0.005,        # KL(uniform over legal || pi); Jaxcalibur gives no value

    # v21 only: history tokens and auxiliary losses (weights of each head's cross-entropy)
    "history_len": 16,               # battle events in the observation
    "aux_next_coef": 0.1,            # the opponent's action at the same decision
    "aux_item_coef": 0.05,           # per revealed opponent: its item, base ability, tera type, moves
    "aux_ability_coef": 0.05,
    "aux_tera_coef": 0.05,
    "aux_moves_coef": 0.05,

    "update_threads": None,          # torch threads for a CPU update (None: all cores)
    "update_device": "auto",         # cuda, else mps (Apple GPU: 2.7x the M3's CPU), else cpu

    # Opponents
    "opponents": "mix",              # "mix" or "heuristic" (SimpleHeuristics only)
    "heuristic_ratio": 0.10,
    "latest_opponent_ratio": 0.5,    # of non-heuristic games: vs the live policy (two-sided)
    "pool_save_freq": 1_000_000,
    "max_pool_size": 30,
    "seed_models": [
        "models/ppo_pokemon_6v6_v17.2_best.zip",
        "models/ppo_pokemon_6v6_v17.3sp_best.zip",
        "models/ppo_pokemon_6v6_v18.1_best.zip",
        "models/ppo_pokemon_6v6_v18.2_best.zip",
        "models/ppo_pokemon_6v6_v18.3_best.zip",
    ],

    # Evaluation: true wins over deterministic games vs SimpleHeuristics, in its own process
    "eval_freq": 1_000_000,
    "n_eval_episodes": 500,
    "save_freq": 200_000,

    "allow_config_change": False,    # resume even if the settings below differ from the checkpoint's
    "net_config": None,              # the arch's DEFAULT_CONFIG (networks/v2x_policy.py) when None
    "model_dir": "models",
    "log_dir": "logs",
    "data_dir": "data",
    "seed": None,
}


# Settings a resumed run must share with its checkpoint (pass the same --set flags when resuming)
RESUME_CHECK_KEYS = (
    "total_timesteps", "lr_initial", "lr_final", "lr_decay_exponent", "embed_lr_mult", "adam_eps",
    "gamma", "gae_lambda", "clip_range", "clip_range_vf", "n_epochs", "batch_size",
    "steps_per_update", "max_grad_norm", "ent_coef", "vf_coef", "zero_avoid_coef", "opponents",
    "heuristic_ratio", "latest_opponent_ratio", "pool_save_freq", "max_pool_size", "net_config",
    "arch", "history_len", "aux_next_coef", "aux_item_coef", "aux_ability_coef", "aux_tera_coef",
    "aux_moves_coef")
AUX_HEADS = ("next", "item", "ability", "tera", "moves")


def server_config(port):
    from poke_env.ps_client.server_configuration import ServerConfiguration
    return ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket",
                               "https://play.pokemonshowdown.com/action.php?")


def paths(cfg):
    v = cfg["version"]
    md = cfg["model_dir"]
    return {
        "latest": os.path.join(md, f"ppo_pokemon_6v6_{v}_latest.pt"),
        "best": os.path.join(md, f"ppo_pokemon_6v6_{v}_best.pt"),
        "final": os.path.join(md, f"ppo_pokemon_6v6_{v}.pt"),
        "pool": os.path.join(md, f"selfplay_pool_{v}"),
        "evals": os.path.join(md, f"eval_snapshots_{v}"),
        "tb": os.path.join(cfg["log_dir"], f"{v}"),
    }


def lr_at(cfg, step):
    x = min(step / max(cfg["total_timesteps"], 1), 1.0)
    decay = 1.0 / (8.0 * x + 1.0) ** cfg["lr_decay_exponent"]
    return cfg["lr_final"] + (cfg["lr_initial"] - cfg["lr_final"]) * decay


# =============================================================================================
# Worker side
def gae(values, final_reward, gamma, lam):
    """Advantages and returns for one side's complete trajectory with a single terminal reward."""
    T = len(values)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            delta = final_reward - values[t]
        else:
            delta = gamma * values[t + 1] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


class Worker:
    def __init__(self, wid, cfg):
        torch.set_num_threads(1)
        from envs.rl_player_v20 import disable_turn_snapshots
        disable_turn_snapshots()
        from networks import registry
        from poke_env.player import SimpleHeuristicsPlayer
        self.wid, self.cfg = wid, cfg
        self.rng = random.Random((cfg.get("seed") or random.randrange(1 << 30)) + 7919 * wid)
        if cfg.get("seed") is not None:
            torch.manual_seed(cfg["seed"] + 7919 * wid)
        self.port = cfg["base_port"] + wid % cfg["n_servers"]
        spec = registry.get(cfg["arch"])
        layout = spec.layout(cfg)
        self.fz = spec.featurizer(layout)
        self.net = spec.net(cfg.get("net_config"), layout).eval()
        self.labels = spec.has_labels
        self.game_labels = []
        self.sh = SimpleHeuristicsPlayer(battle_format="gen9randombattle", start_listening=False)
        self.pool_cache = OrderedDict()
        self.env = None
        self.errors = Counter()

    def new_env(self):
        from envs.rl_player_v20 import RLEnvV20
        from poke_env.ps_client import AccountConfiguration
        if self.env is not None:
            try:
                self.env.close(force=True, wait=False)
            except Exception:
                pass
        uid = uuid.uuid4().hex[:4]
        self.env = RLEnvV20(
            server_configuration=server_config(self.port), step_timeout=self.cfg["step_timeout"],
            account_configuration1=AccountConfiguration.generate(f"w{self.wid}a{uid}", rand=True),
            account_configuration2=AccountConfiguration.generate(f"w{self.wid}b{uid}", rand=True))

    def opponent(self, path):
        from players.v20_player import load_opponent
        if path in self.pool_cache:
            self.pool_cache.move_to_end(path)
            return self.pool_cache[path]
        opp = load_opponent(path, deterministic=False, data_dir=self.cfg["data_dir"])
        self.pool_cache[path] = opp
        while len(self.pool_cache) > self.cfg["max_pool_size"]:
            self.pool_cache.popitem(last=False)
        return opp

    def pick_opponent(self, pool):
        cfg = self.cfg
        if cfg["opponents"] == "heuristic" or self.rng.random() < cfg["heuristic_ratio"]:
            return "heuristic", self.sh
        if not pool or self.rng.random() < cfg["latest_opponent_ratio"]:
            return "live", None
        path = self.rng.choice(pool)
        try:
            opp = self.opponent(path)
            return f"pool_{getattr(opp, 'arch', 'sb3')}", opp
        except Exception as e:
            self.errors["opponent load failed"] += 1
            print(f"[worker {self.wid}] could not load {path}: {e}", flush=True)
            return "live", None

    def play_game(self, pool):
        """One complete game. Returns (kind, [(samples, won)] per recorded side, learner decisions)
        or None if the game was abandoned. With aux labels (v21), self.game_labels then holds one
        list of label rows per returned side, one row per sample."""
        from players.v20_player import opponent_decision
        kind, opp = self.pick_opponent(pool)
        env = self.env
        env.reset()
        a1, a2 = env.agent1.username, env.agent2.username
        record = {1: True, 2: kind == "live"}
        traj = {1: [], 2: []}
        labels = {1: [], 2: []}
        n_dec = {1: 0, 2: 0}
        if self.labels:
            from envs.rl_player_v21 import LAB_NEXT, N_LAB, illusion_game, next_action_label, set_labels
            no_labels = illusion_game(env.battle1, env.battle2)
        while not env.battle1.finished:
            b1, b2 = env.battle1, env.battle2
            acts = {a1: 0, a2: 0}
            choosing = {1: env.agent1_to_move and not b1.finished, 2: env.agent2_to_move and not b2.finished}
            live = []
            if choosing[1]:
                live.append((1, a1, b1))
            if choosing[2]:
                if kind == "live":
                    live.append((2, a2, b2))
                else:
                    acts[a2] = opponent_decision(opp, b2)
            if live:
                # features are stored as float16; round them before acting so the stored sample is
                # exactly what the policy saw (the PPO ratio starts at 1)
                feats = [(i, f.astype(np.float16), m) for i, f, m in
                         (self.fz.featurize(b) for _, _, b in live)]
                idx = torch.as_tensor(np.stack([f[0] for f in feats]))
                flt = torch.as_tensor(np.stack([f[1] for f in feats]).astype(np.float32))
                mask = torch.as_tensor(np.stack([f[2] for f in feats]))
                with torch.inference_mode():
                    act, logp, val = self.net.act(idx, flt, mask)
                for j, (side, agent, _) in enumerate(live):
                    a = int(act[j])
                    acts[agent] = a
                    n_dec[side] += 1
                    if record[side]:
                        traj[side].append((feats[j][0].astype(np.int16), feats[j][1], feats[j][2], a,
                                           float(logp[j]), float(val[j])))
                if self.labels:
                    # only now is every choice of this step known (the other side's is the label)
                    for side, agent, b in live:
                        if not record[side]:
                            continue
                        if no_labels:
                            row = np.full(N_LAB, -1, dtype=np.int16)
                        else:
                            other = b2 if side == 1 else b1
                            row = set_labels(b, other)
                            if choosing[3 - side]:
                                row[LAB_NEXT] = next_action_label(b, other, acts[a2 if side == 1 else a1])
                        labels[side].append(row)
            if max(n_dec.values()) > self.cfg["max_decisions"]:
                self.errors["game too long (dropped)"] += 1
                self.new_env()
                return None
            env.step(acts)
        out, self.game_labels = [], []
        for side, battle in ((1, env.battle1), (2, env.battle2)):
            if record[side] and traj[side]:
                out.append((traj[side], battle.won))
                self.game_labels.append(labels[side])
        return kind, out, n_dec[1]

    def collect(self, weights, quota, pool):
        self.net.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})
        self.net.eval()
        cfg = self.cfg
        buf = defaultdict(list)
        stats = {"games": 0, "samples": 0, "by_kind": Counter(), "wins": Counter(),
                 "losses": Counter(), "ties": Counter(), "ep_len": [], "two_sided_games": 0}
        t0 = time.time()
        n = 0
        while n < quota:
            if self.env is None:
                self.new_env()
            try:
                res = self.play_game(pool)
            except Exception as e:
                name = type(e).__name__
                self.errors[name] += 1
                if self.errors[name] <= 3 or self.errors[name] % 50 == 0:
                    print(f"[worker {self.wid}] game abandoned ({name}: {e}); rebuilding env", flush=True)
                self.new_env()
                continue
            if res is None:
                continue
            kind, sides, learner_len = res
            stats["games"] += 1
            stats["by_kind"][kind] += 1
            stats["ep_len"].append(learner_len)
            if len(sides) == 2:
                stats["two_sided_games"] += 1
            for j, (steps, won) in enumerate(sides):
                reward = 1.0 if won is True else (-1.0 if won is False else 0.0)
                if j == 0:  # the learner's (side 1) result, per opponent kind
                    (stats["wins"] if won is True else stats["losses"] if won is False
                     else stats["ties"])[kind] += 1
                values = [s[5] for s in steps]
                adv, ret = gae(values, reward, cfg["gamma"], cfg["gae_lambda"])
                for s in steps:
                    buf["idx"].append(s[0])
                    buf["flt"].append(s[1])
                    buf["mask"].append(s[2])
                    buf["act"].append(s[3])
                    buf["logp"].append(s[4])
                    buf["val"].append(s[5])
                buf["adv"].append(adv)
                buf["ret"].append(ret)
                if self.labels:
                    buf["lab"] += self.game_labels[j]
                n += len(steps)
        stats["samples"] = n
        stats["collect_s"] = time.time() - t0
        stats["errors"] = dict(self.errors)
        data = {
            "idx": np.stack(buf["idx"]),
            "flt": np.stack(buf["flt"]),
            "mask": np.stack(buf["mask"]),
            "act": np.asarray(buf["act"], dtype=np.int64),
            "logp": np.asarray(buf["logp"], dtype=np.float32),
            "val": np.asarray(buf["val"], dtype=np.float32),
            "adv": np.concatenate(buf["adv"]).astype(np.float32),
            "ret": np.concatenate(buf["ret"]).astype(np.float32),
        }
        if self.labels:
            data["lab"] = np.stack(buf["lab"])
        return data, stats


def worker_main(wid, cfg, conn):
    try:
        w = Worker(wid, cfg)
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg[0] == "close":
            break
        if msg[0] == "collect":
            _, weights, quota, pool = msg
            try:
                conn.send(("rollout",) + w.collect(weights, quota, pool))
            except Exception:
                conn.send(("error", traceback.format_exc()))
    try:
        if w.env is not None:
            w.env.close(force=True, wait=False)
    except Exception:
        pass


# =============================================================================================
# Evaluation (its own process, a fresh env each time: eval/v20_eval.py)
def eval_main(snapshot_path, step, n_games, port, queue):
    torch.set_num_threads(1)
    try:
        from envs.rl_player_v20 import disable_turn_snapshots
        disable_turn_snapshots()
        from eval.v20_eval import play_games
        from players.v20_player import V20Policy
        from poke_env.player import SimpleHeuristicsPlayer
        policy = V20Policy.load(snapshot_path, deterministic=True)
        sh = SimpleHeuristicsPlayer(battle_format="gen9randombattle", start_listening=False)
        t0 = time.time()
        r = play_games(policy, sh, n_games, port, seat=1, name="ev")
        queue.put(("eval", step, r["wins"], r["games"], r["mean_len"], time.time() - t0))
    except Exception:
        queue.put(("eval_error", step, traceback.format_exc()))


# =============================================================================================
# Learner side
class Pool:
    """Self-play pool directory: seed_* SB3 zips (oldest) and checkpoint_* v20 snapshots."""

    def __init__(self, directory, max_size):
        self.dir, self.max = directory, max_size
        os.makedirs(directory, exist_ok=True)

    def paths(self):
        files = [f for f in os.listdir(self.dir) if f.endswith((".zip", ".pt"))]
        files.sort(key=lambda f: (not f.startswith("seed_"), f))
        return [os.path.join(self.dir, f) for f in files]

    def seed(self, models):
        if self.paths():
            return
        for i, src in enumerate(models):
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.dir, f"seed_{i:03d}_{os.path.basename(src)}"))
                print(f"  seeded pool with {os.path.basename(src)}")

    def add(self, state, step):
        path = os.path.join(self.dir, f"checkpoint_{step:010d}.pt")
        torch.save(state, path)
        ps = self.paths()
        while len(ps) > self.max:
            os.remove(ps.pop(0))
        return path


def model_state(model, arch, extra=None):
    state = {"arch": arch, "net_config": model.config, "layout": dict(model.layout),
             "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    state.update(extra or {})
    return state


def explained_variance(y_pred, y_true):
    var = np.var(y_true)
    return float("nan") if var == 0 else float(1 - np.var(y_true - y_pred) / var)


def update_device(cfg):
    want = cfg.get("update_device", "auto")
    if want == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(want)


def _grad_norm(model):
    return torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in model.parameters()
                                                 if p.grad is not None]))


def ppo_update(model, optimizer, data, cfg, device, baseline=None):
    """SB3-style PPO epochs over the rollout. Minibatches are gathered on the CPU and all have
    batch_size samples (each epoch drops a random remainder of under batch_size): the MPS backend
    caches a compiled graph per input shape, about 57 MB each, and a partial last minibatch of a
    new size every update grew the learner by that much per update (7.6 GB after 150 updates).
    SB3's v19 rollouts were exactly 40 x 1024, so it never had partial minibatches either.

    With labels (data["lab"], v21) the loss adds sum(aux_<head>_coef * aux loss). Aux accuracies are
    measured in the first epoch, each minibatch before its own gradient step (so on samples the
    network hasn't trained on), next to the per-species base rates of `baseline` (AuxBaseline).
    The gradient norms of the PPO and aux losses alone are measured on the middle minibatch of
    every epoch (the first one of an update is dominated by the value loss): they share
    max_grad_norm and Adam's moments, so a large aux gradient would shrink the PPO step.
    `grad_ppo_kept` is the share of the PPO step the shared clip keeps relative to clipping the PPO
    gradient alone (1 = the aux losses cost the PPO step nothing at the clip)."""
    idx = torch.as_tensor(data["idx"])      # int16
    flt = torch.as_tensor(data["flt"])      # float16
    mask = torch.as_tensor(data["mask"])
    act = torch.as_tensor(data["act"])
    old_logp = torch.as_tensor(data["logp"])
    old_val = torch.as_tensor(data["val"])
    advs = torch.as_tensor(data["adv"])
    rets = torch.as_tensor(data["ret"])
    use_aux = "lab" in data and getattr(model, "has_aux", False)
    if use_aux:
        from networks.v21_policy import aux_obs, aux_terms
        lab = torch.as_tensor(data["lab"])  # int16
        coefs = {h: cfg[f"aux_{h}_coef"] for h in AUX_HEADS}
        base_tables = baseline.device_tables(device) if baseline is not None else None
        acc, base_acc = defaultdict(list), defaultdict(list)
    N = len(act)
    bs = min(cfg["batch_size"], N)
    n_mb = N // bs
    clip, clip_vf = cfg["clip_range"], cfg["clip_range_vf"]
    logs = defaultdict(list)  # device tensors, read once at the end (no per-step syncs)
    model.train()
    for epoch in range(cfg["n_epochs"]):
        perm = torch.randperm(N)
        for k in range(n_mb):
            mb = perm[k * bs:(k + 1) * bs]
            idx_mb, flt_mb = idx[mb].to(device).long(), flt[mb].to(device).float()
            aux_loss = None
            if use_aux:
                logp, ent, val, kl_u, aux_out = model.evaluate_actions(
                    idx_mb, flt_mb, mask[mb].to(device), act[mb].to(device), aux=True)
                obs = aux_obs(idx_mb, flt_mb, model.layout)
                lab_mb = lab[mb].to(device).long()
                losses, mets = aux_terms(aux_out, lab_mb, obs, metrics=epoch == 0)
                aux_loss = sum(coefs[h] * losses[h] for h in AUX_HEADS)
                for h in AUX_HEADS:
                    logs[f"aux_loss_{h}"].append(losses[h].detach())
                for name, (c, n) in mets.items():
                    acc[name].append(torch.stack([c, n]).float())
                if epoch == 0 and base_tables is not None:
                    with torch.no_grad():
                        _, bmets = aux_terms(baseline.logits(obs, base_tables), lab_mb, obs, metrics=True)
                    for name, (c, n) in bmets.items():
                        base_acc[name].append(torch.stack([c, n]).float())
            else:
                logp, ent, val, kl_u = model.evaluate_actions(
                    idx_mb, flt_mb, mask[mb].to(device), act[mb].to(device))
            adv = advs[mb].to(device)
            if bs > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            log_ratio = logp - old_logp[mb].to(device)
            ratio = torch.exp(log_ratio)
            pg = -torch.min(adv * ratio, adv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
            if clip_vf is not None:
                ov = old_val[mb].to(device)
                val_pred = ov + torch.clamp(val - ov, -clip_vf, clip_vf)
            else:
                val_pred = val
            v_loss = F.mse_loss(rets[mb].to(device), val_pred)
            ent_loss = -ent.mean()
            za = kl_u.mean()
            loss = pg + cfg["ent_coef"] * ent_loss + cfg["vf_coef"] * v_loss + cfg["zero_avoid_coef"] * za
            measure = aux_loss is not None and k == n_mb // 2
            if aux_loss is not None:
                if measure:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=True)
                    g_ppo = _grad_norm(model)
                    optimizer.zero_grad(set_to_none=True)
                    aux_loss.backward(retain_graph=True)
                    g_aux = _grad_norm(model)
                logs["ppo_loss"].append(loss.detach())
                loss = loss + aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            if measure:
                with torch.no_grad():
                    mx = cfg["max_grad_norm"]
                    logs["grad_norm_ppo"].append(g_ppo)
                    logs["grad_norm_aux"].append(g_aux)
                    logs["grad_norm_measured"].append(grad_norm.detach())
                    # |ppo + aux|^2 = |ppo|^2 + |aux|^2 + 2 ppo.aux
                    logs["grad_cos_ppo_aux"].append(
                        (grad_norm ** 2 - g_ppo ** 2 - g_aux ** 2) / (2 * g_ppo * g_aux + 1e-12))
                    # share of the PPO step the shared clip keeps, relative to clipping PPO alone
                    logs["grad_ppo_kept"].append(torch.clamp(mx / grad_norm, max=1.0) /
                                                 torch.clamp(mx / g_ppo, max=1.0))
            with torch.no_grad():
                logs["approx_kl"].append(((ratio - 1) - log_ratio).mean())
                logs["clip_fraction"].append(((ratio - 1).abs() > clip).float().mean())
                logs["policy_gradient_loss"].append(pg.detach())
                logs["value_loss"].append(v_loss.detach())
                logs["entropy_loss"].append(ent_loss.detach())
                logs["zero_avoid_kl"].append(za.detach())
                logs["loss"].append(loss.detach())
                logs["grad_norm"].append(grad_norm.detach())
                logs["grad_clipped"].append((grad_norm > cfg["max_grad_norm"]).float())
    model.eval()
    out = {k: float(torch.stack(v).mean()) for k, v in logs.items()}
    out["explained_variance"] = explained_variance(data["val"], data["ret"])
    if use_aux:
        for prefix, src in (("aux_acc_", acc), ("aux_base_", base_acc)):
            for name, v in src.items():
                c, n = torch.stack(v).sum(0).tolist()
                out[prefix + name] = c / n if n else float("nan")
                if prefix == "aux_acc_":
                    out["aux_n_" + name] = n
        del acc, base_acc
    del logs
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return out


class Trainer:
    def __init__(self, cfg):
        from networks import registry
        self.cfg = cfg
        self.spec = registry.get(cfg["arch"])
        if cfg.get("net_config") is None:
            cfg["net_config"] = self.spec.default_net_config()
        if cfg.get("eval_port") is None:
            cfg["eval_port"] = cfg["base_port"] + cfg["n_servers"]
        self.p = paths(cfg)
        os.makedirs(cfg["model_dir"], exist_ok=True)
        os.makedirs(self.p["evals"], exist_ok=True)
        self.device = update_device(cfg)
        self.model = self.spec.net(cfg["net_config"], self.spec.layout(cfg)).to(self.device)
        self.baseline = None
        if self.spec.has_labels:
            from networks.v21_policy import AuxBaseline
            self.baseline = AuxBaseline(self.model.layout)
        self.optimizer = torch.optim.Adam(
            [{"params": self.model.embedding_parameters(), "lr": cfg["lr_initial"] * cfg["embed_lr_mult"],
              "name": "embeddings"},
             {"params": self.model.other_parameters(), "lr": cfg["lr_initial"], "name": "network"}],
            eps=cfg["adam_eps"])
        self.step = 0
        self.games = 0
        self.update_n = 0
        self.eval_state = {"best": -1.0, "results": [], "done_steps": [], "pending": []}
        self.pool = Pool(self.p["pool"], cfg["max_pool_size"])
        self.ctx = mp.get_context("forkserver")
        self.workers = {}
        self.eval_queue = self.ctx.Queue()
        self.eval_procs = {}
        self.writer = None

    # ---- checkpoints ------------------------------------------------------------------------
    def save(self, path=None):
        state = model_state(self.model, self.cfg["arch"], {
            "optimizer": self.optimizer.state_dict(), "config": self.cfg, "step": self.step,
            "games": self.games, "update": self.update_n, "eval_state": self.eval_state,
            "saved_at": datetime.now().isoformat()})
        if self.baseline is not None:
            state["aux_baseline"] = self.baseline.state_dict()
        path = path or self.p["latest"]
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    def try_resume(self):
        if not os.path.exists(self.p["latest"]):
            return False
        ckpt = torch.load(self.p["latest"], map_location="cpu", weights_only=False)
        if ckpt.get("arch", "v20") != self.cfg["arch"]:
            raise SystemExit(f"{self.p['latest']} is a {ckpt.get('arch')!r} checkpoint, this run is "
                             f"{self.cfg['arch']!r}: pass the matching --arch, or another --version")
        saved = ckpt.get("config", {})
        changed = {k: (saved.get(k), self.cfg.get(k)) for k in RESUME_CHECK_KEYS
                   if k in saved and saved.get(k) != self.cfg.get(k)}
        if changed:
            lines = "\n".join(f"    {k}: checkpoint {a!r}, now {b!r}" for k, (a, b) in changed.items())
            if not self.cfg.get("allow_config_change"):
                raise SystemExit(f"{self.p['latest']} was trained with different settings:\n{lines}\n"
                                 "Pass the same --set flags as the original run, or add "
                                 "--set allow_config_change=True to resume with the new ones.")
            print(f"  WARNING: resuming with changed settings:\n{lines}", flush=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])  # moves its state to the params' device
        self.step, self.games, self.update_n = ckpt["step"], ckpt["games"], ckpt["update"]
        self.eval_state = ckpt.get("eval_state", self.eval_state)
        if self.baseline is not None and "aux_baseline" in ckpt:
            self.baseline.load_state_dict(ckpt["aux_baseline"])
        print(f"Resumed {self.cfg['version']} at {self.step:,} steps, {self.games:,} games, "
              f"update {self.update_n}", flush=True)
        return True

    # ---- workers ----------------------------------------------------------------------------
    def start_worker(self, wid):
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=worker_main, args=(wid, self.cfg, child), daemon=True,
                                name=f"v20-worker-{wid}")
        proc.start()
        child.close()
        self.workers[wid] = (proc, parent)

    def stop_worker(self, wid):
        proc, conn = self.workers.pop(wid)
        try:
            conn.close()
        except Exception:
            pass
        proc.kill()
        proc.join(5)

    def collect(self):
        cfg = self.cfg
        quota = math.ceil(cfg["steps_per_update"] / cfg["n_workers"])
        weights = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        pool = [] if cfg["opponents"] == "heuristic" else self.pool.paths()
        for wid in range(cfg["n_workers"]):
            if wid not in self.workers or not self.workers[wid][0].is_alive():
                if wid in self.workers:
                    self.stop_worker(wid)
                self.start_worker(wid)
            self.workers[wid][1].send(("collect", weights, quota, pool))
        results, pending = [], set(range(cfg["n_workers"]))
        t0 = time.time()
        while pending:
            for wid in list(pending):
                proc, conn = self.workers[wid]
                try:
                    ready = conn.poll(0.05)
                except (EOFError, OSError):
                    print(f"[learner] worker {wid} pipe broke; restarting next update", flush=True)
                    pending.discard(wid)
                    self.stop_worker(wid)
                    continue
                if ready:
                    try:
                        msg = conn.recv()
                    except (EOFError, OSError):
                        msg = ("error", "worker pipe closed")
                    pending.discard(wid)
                    if msg[0] == "rollout":
                        results.append((wid, msg[1], msg[2]))
                    else:
                        print(f"[learner] worker {wid} failed:\n{msg[1]}", flush=True)
                        self.stop_worker(wid)
                elif not proc.is_alive():
                    print(f"[learner] worker {wid} died (exit {proc.exitcode}); restarting next update",
                          flush=True)
                    pending.discard(wid)
                    self.stop_worker(wid)
                elif time.time() - t0 > cfg["worker_timeout"]:
                    print(f"[learner] worker {wid} silent for {cfg['worker_timeout']:.0f} s; restarting",
                          flush=True)
                    pending.discard(wid)
                    self.stop_worker(wid)
        return results, time.time() - t0

    # ---- evaluation -------------------------------------------------------------------------
    def launch_eval(self, step):
        snap = os.path.join(self.p["evals"], f"step_{step:010d}.pt")
        if not os.path.exists(snap):
            torch.save(model_state(self.model, self.cfg["arch"], {"step": step}), snap)
        if step not in self.eval_state["pending"]:
            self.eval_state["pending"].append(step)
        proc = self.ctx.Process(target=eval_main, daemon=True, name=f"v20-eval-{step}",
                                args=(snap, step, self.cfg["n_eval_episodes"], self.cfg["eval_port"],
                                      self.eval_queue))
        proc.start()
        self.eval_procs[step] = proc
        print(f">>> Eval started for step {step:,} ({self.cfg['n_eval_episodes']} games vs "
              f"SimpleHeuristics, port {self.cfg['eval_port']})", flush=True)

    def poll_evals(self, block=False):
        """Log finished evals. block=True waits until every running eval has reported (or died)."""
        import queue as _queue
        while True:
            try:
                msg = self.eval_queue.get(timeout=5) if block else self.eval_queue.get_nowait()
            except _queue.Empty:
                if block and any(p.is_alive() for p in self.eval_procs.values()):
                    continue
                break
            self.handle_eval(msg)
            if block and not self.eval_procs:
                break
        for step, proc in list(self.eval_procs.items()):
            if not proc.is_alive() and self.eval_queue.empty():
                proc.join(0)
                if proc.exitcode not in (0, None):
                    print(f">>> Eval @ {step:,} process died (exit {proc.exitcode})", flush=True)
                    self.eval_procs.pop(step, None)

    def handle_eval(self, msg):
        kind, step = msg[0], msg[1]
        proc = self.eval_procs.pop(step, None)
        if proc is not None:
            proc.join(5)
        if step in self.eval_state["pending"]:
            self.eval_state["pending"].remove(step)
        if kind == "eval_error":
            print(f">>> Eval @ {step:,} failed:\n{msg[2]}", flush=True)
            return
        _, _, wins, n, mean_len, secs = msg
        wr = wins / max(n, 1)
        marker = ""
        if wr > self.eval_state["best"]:
            self.eval_state["best"] = wr
            shutil.copy2(os.path.join(self.p["evals"], f"step_{step:010d}.pt"), self.p["best"])
            marker = " NEW BEST"
        self.eval_state["results"].append({"step": step, "win_rate": wr, "wins": wins, "games": n,
                                           "mean_len": mean_len})
        self.eval_state["done_steps"].append(step)
        if self.writer:
            self.writer.add_scalar("eval/win_rate", wr, step)
            self.writer.add_scalar("eval/ep_len_mean", mean_len, step)
        print(f"\n{'=' * 60}\n>>> Eval @ {step:,}: {wr:.1%} ({wins}/{n}){marker} | "
              f"Best: {self.eval_state['best']:.1%} | {secs:.0f} s\n{'=' * 60}\n", flush=True)

    # ---- main loop --------------------------------------------------------------------------
    def train(self):
        from torch.utils.tensorboard import SummaryWriter
        cfg = self.cfg
        resumed = self.try_resume()
        if cfg["opponents"] == "mix":
            self.pool.seed([m for m in cfg["seed_models"]])
        self.writer = SummaryWriter(self.p["tb"])
        stop_at = cfg["stop_at"] or cfg["total_timesteps"]
        n_params = sum(p.numel() for p in self.model.parameters())
        print("=" * 70)
        print(f"Future Sight {cfg['version']}: {cfg['arch']} network, {n_params:,} parameters")
        print(f"  workers {cfg['n_workers']} on ports {cfg['base_port']}..{cfg['base_port'] + cfg['n_servers'] - 1}, "
              f"eval port {cfg['eval_port']}")
        print(f"  opponents: {cfg['opponents']} | steps/update {cfg['steps_per_update']:,} | "
              f"LR horizon {cfg['total_timesteps']:,} | stop at {stop_at:,}")
        print(f"  update device: {self.device} | lr {cfg['lr_initial']:.3e} (embeddings x{cfg['embed_lr_mult']:g}), "
              f"decay (8x+1)^-{cfg['lr_decay_exponent']:g}")
        print(f"  PPO: gamma {cfg['gamma']} lambda {cfg['gae_lambda']} clip {cfg['clip_range']} "
              f"vf-clip {cfg['clip_range_vf']} epochs {cfg['n_epochs']} batch {cfg['batch_size']} "
              f"ent {cfg['ent_coef']} vf {cfg['vf_coef']} zero-avoid {cfg['zero_avoid_coef']}")
        if self.spec.has_labels:
            print(f"  net {cfg['net_config']} | history {cfg['history_len']} | aux coefs "
                  + " ".join(f"{h} {cfg[f'aux_{h}_coef']}" for h in AUX_HEADS))
        print("=" * 70, flush=True)
        # evals that were running when a previous run stopped
        for s in list(self.eval_state.get("pending", [])):
            if os.path.exists(os.path.join(self.p["evals"], f"step_{s:010d}.pt")):
                self.launch_eval(s)
        threads = cfg["update_threads"] or os.cpu_count()
        interrupted = False
        try:
            while self.step < stop_at:
                lr = lr_at(cfg, self.step)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr * (cfg["embed_lr_mult"] if g.get("name") == "embeddings" else 1.0)
                results, collect_s = self.collect()
                if not results:
                    print("[learner] no rollouts this update; retrying", flush=True)
                    time.sleep(5)
                    continue
                data = {k: np.concatenate([r[1][k] for r in results]) for k in results[0][1]}
                stats = [r[2] for r in results]
                del results
                n = len(data["act"])
                t1 = time.time()
                torch.set_num_threads(threads)
                logs = ppo_update(self.model, self.optimizer, data, cfg, self.device, self.baseline)
                if self.baseline is not None:
                    self.baseline.update(data["idx"], data["flt"], data["lab"])
                update_s = time.time() - t1
                prev = self.step
                self.step += n
                self.update_n += 1
                games = sum(s["games"] for s in stats)
                self.games += games
                self.log(logs, stats, n, games, lr, collect_s, update_s)
                del data
                if self.step // cfg["save_freq"] > prev // cfg["save_freq"]:
                    self.save()
                    print(f">>> Checkpoint @ {self.step:,} steps", flush=True)
                if cfg["opponents"] == "mix" and self.step // cfg["pool_save_freq"] > prev // cfg["pool_save_freq"]:
                    path = self.pool.add(model_state(self.model, cfg["arch"], {"step": self.step}), self.step)
                    print(f"  -> pool snapshot {os.path.basename(path)} (pool: {len(self.pool.paths())})",
                          flush=True)
                if cfg["eval_freq"] and self.step // cfg["eval_freq"] > prev // cfg["eval_freq"]:
                    self.launch_eval(self.step)
                    self.save()
                self.poll_evals()
        except KeyboardInterrupt:
            interrupted = True
            print("\n>>> Interrupted (Ctrl+C or SIGTERM); saving", flush=True)
        self.save()
        if not interrupted:
            if self.step >= cfg["total_timesteps"]:
                torch.save(model_state(self.model, cfg["arch"], {"step": self.step}), self.p["final"])
            if self.eval_procs:
                print(">>> Waiting for running evals...", flush=True)
                self.poll_evals(block=True)
                self.save()
        for wid in list(self.workers):
            try:
                self.workers[wid][1].send(("close",))
            except Exception:
                pass
        time.sleep(1)
        for wid in list(self.workers):
            self.stop_worker(wid)
        self.writer.close()
        print("\nEval results:")
        for r in self.eval_state["results"]:
            print(f"  {r['step']:>11,}: {r['win_rate']:.1%} ({r['wins']}/{r['games']})")

    def log(self, logs, stats, n, games, lr, collect_s, update_s):
        w = self.writer
        step = self.step
        for k, v in logs.items():
            w.add_scalar(f"train/{k}", v, step)
        w.add_scalar("train/learning_rate", lr, step)
        wins, losses, ties, kinds = Counter(), Counter(), Counter(), Counter()
        ep_lens = []
        errors = Counter()
        for s in stats:
            wins.update(s["wins"])
            losses.update(s["losses"])
            ties.update(s["ties"])
            kinds.update(s["by_kind"])
            ep_lens += s["ep_len"]
            errors.update(s["errors"])
        tot = sum(wins.values()) + sum(losses.values()) + sum(ties.values())
        wr = sum(wins.values()) / max(tot, 1)
        w.add_scalar("rollout/win_rate", wr, step)
        for k in kinds:
            nk = wins[k] + losses[k] + ties[k]
            if nk:
                w.add_scalar(f"rollout/win_rate_vs_{k}", wins[k] / nk, step)
                w.add_scalar(f"rollout/games_{k}", nk, step)
        ep_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        w.add_scalar("rollout/ep_len_mean", ep_len, step)
        w.add_scalar("rollout/games", games, step)
        w.add_scalar("rollout/games_total", self.games, step)
        two = sum(s["two_sided_games"] for s in stats)
        w.add_scalar("rollout/two_sided_share", two / max(games, 1), step)
        fps = n / max(collect_s, 1e-6)
        w.add_scalar("time/fps", fps, step)
        w.add_scalar("time/collect_s", collect_s, step)
        w.add_scalar("time/update_s", update_s, step)
        w.add_scalar("time/total_fps", n / max(collect_s + update_s, 1e-6), step)
        w.add_scalar("selfplay/pool_size", len(self.pool.paths()), step)
        err = f" | errors {dict(errors)}" if errors else ""
        print(f"update {self.update_n} | total_timesteps {step:,} | games {self.games:,} (+{games}) | "
              f"fps {fps:,.0f} collect, {n / max(collect_s + update_s, 1e-6):,.0f} total | "
              f"collect {collect_s:.0f}s update {update_s:.0f}s | lr {lr:.2e} | "
              f"kl {logs['approx_kl']:.4f} clip {logs['clip_fraction']:.3f} "
              f"ent {-logs['entropy_loss']:.3f} ev {logs['explained_variance']:.3f} "
              f"za {logs['zero_avoid_kl']:.3f} | win {wr:.1%} {dict(kinds)} | len {ep_len:.1f}{err}",
              flush=True)
        if "aux_loss_next" in logs:
            def acc(name):
                return f"{logs.get('aux_acc_' + name, float('nan')):.2f}/{logs.get('aux_base_' + name, float('nan')):.2f}"
            print("  aux | loss " + " ".join(f"{h} {logs['aux_loss_' + h]:.2f}" for h in AUX_HEADS)
                  + f" | acc (model/base) next {acc('next')} kind {acc('next_kind')} | item {acc('item')}"
                  f" ({acc('item_unrev')}) ability {acc('ability')} ({acc('ability_unrev')}) tera {acc('tera')}"
                  f" ({acc('tera_unrev')}) moves {acc('moves')} ({acc('moves_unrev')}) [unrevealed in ()]"
                  f" | grad ppo {logs['grad_norm_ppo']:.2f} aux {logs['grad_norm_aux']:.2f}"
                  f" (cos {logs['grad_cos_ppo_aux']:.2f}) both {logs['grad_norm_measured']:.2f}"
                  f" ppo kept {logs['grad_ppo_kept']:.2f} | all minibatches: mean {logs['grad_norm']:.2f}"
                  f" clipped {logs['grad_clipped']:.0%}", flush=True)


def _stop_on_sigterm(signum, frame):
    raise KeyboardInterrupt


def main():
    import signal
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    ap = argparse.ArgumentParser(description="v20/v21 PPO training")
    ap.add_argument("--version", default=None, help="run name (default: the arch)")
    ap.add_argument("--arch", choices=["v20", "v21"], default=None)
    ap.add_argument("--timesteps", type=int, default=None, help="LR schedule horizon")
    ap.add_argument("--stop-at", type=int, default=None, help="stop after this many steps")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--servers", type=int, default=None)
    ap.add_argument("--base-port", type=int, default=None)
    ap.add_argument("--eval-port", type=int, default=None)
    ap.add_argument("--opponents", choices=["mix", "heuristic"], default=None)
    ap.add_argument("--steps-per-update", type=int, default=None)
    ap.add_argument("--eval-freq", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--save-freq", type=int, default=None)
    ap.add_argument("--update-threads", type=int, default=None)
    ap.add_argument("--update-device", default=None, help="auto (default), cpu, mps or cuda")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any CONFIG entry (repeatable), e.g. --set lr_initial=1.766e-4 "
                         "--set clip_range_vf=None; a resumed run needs the same flags")
    args = ap.parse_args()
    cfg = dict(CONFIG)
    for arg, key in (("version", "version"), ("arch", "arch"), ("timesteps", "total_timesteps"), ("stop_at", "stop_at"),
                     ("workers", "n_workers"), ("servers", "n_servers"), ("base_port", "base_port"),
                     ("eval_port", "eval_port"), ("opponents", "opponents"),
                     ("steps_per_update", "steps_per_update"), ("eval_freq", "eval_freq"),
                     ("eval_episodes", "n_eval_episodes"), ("save_freq", "save_freq"),
                     ("update_threads", "update_threads"), ("update_device", "update_device"),
                     ("seed", "seed")):
        v = getattr(args, arg)
        if v is not None:
            cfg[key] = v
    import ast
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep or key not in CONFIG:
            raise SystemExit(f"--set {item!r}: expected KEY=VALUE with KEY one of CONFIG's keys")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw  # a plain string
        cfg[key] = value
        print(f"  --set {key} = {value!r}", flush=True)
    if args.version is None and not any(item.startswith("version=") for item in args.set):
        cfg["version"] = cfg["arch"]  # a v21 run never resumes v20's checkpoint by accident
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
