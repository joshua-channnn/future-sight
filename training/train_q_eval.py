"""Train the leaf evaluation on the search's own per-option values (agent_docs/step8.md section 4):
from each stored decision, the engine plays every legal option of ours one turn ahead (against
the opponent's recorded choice at that turn, weighted by its own search's visit shares, and over
the turn's chance outcomes, as MCTS expands a node: pe.option_features_from_strings), and the
evaluation is fit so that this one-turn lookahead values the options as the 120K-iteration search
did. The targets are the search's per-option mean scores q, in logits of the evaluation it used
(round 0: the hand-written evaluation, score sigmoid(0.0125 * points) = sigmoid(1.44 * logits);
round 1: lin1_86 at scale 1.4): t = logit(q) / 1.44 or / 1.4. Only differences between the options
of a decision are fit (the root's own value cancels), so unplayed options count as much as the
played one, and memorizing a game's result can't help.

The model is lin's (110 features, antisymmetric: v = a . (x_s1 - x_s2) + b . x_cross). --mode ls: least
squares on the options' centered values (closed form, streamed over the shards), then rescaled so
its one-turn values spread like --lin-init's (the search's leaf scale is tuned to that spread; the
least-squares values are compressed like the targets). --mode combined: the outcome loss of
train_leaf_eval.py plus lam times the targets' loss with a free scale. The engine's features of
the successors are cached per shard (--cache).

    python3 -m training.train_q_eval                                   # linq_ls (least squares)
    python3 -m training.train_q_eval --mode combined --lam 1 10 --out models/leaf_eval/linq
Result (agent_docs/step8.md section 4): the one-turn lookahead agrees better with the search, the
search with it doesn't play better.
"""
import argparse
import glob
import gzip
import json
import os
import time
import zlib

import numpy as np
import poke_engine as pe

N_X = 110
N_SIDE = 52
FS1, FS2 = 51, N_SIDE + 51   # s1_force_switch, s2_force_switch in lin's features
KAPPA = {"search_selfplay": 1 / 1.437, "search_selfplay_r1": 1 / 1.4}   # logit(q) -> logits
MIN_SHARE = 0.005   # options the search visited less are left out (q too noisy)


def split_shards(paths, val_mod):
    files = sorted(f for p in paths for f in glob.glob(os.path.join(p, "*.npz")))
    val = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % val_mod == 0]
    return [f for f in files if f not in set(val)], val


def option_strings(state, mask):
    """[(action, engine option string)] for side one's legal actions of a v20 mask (moves in
    request order, which is the engine's move order with the disabled ones left out; switches by
    team slot, which is the engine's slot), or None when the two don't match."""
    sd = state.side_one
    act = sd.pokemon[int(sd.active_index)]
    usable = [m.id.lower() for m in act.moves if m.id.lower() != "none" and not m.disabled]
    out = []
    for a in np.flatnonzero(mask):
        a = int(a)
        if a < 8:
            j = a % 4
            if j >= len(usable):
                return None
            out.append((a, usable[j] + ("-tera" if a >= 4 else "")))
        else:
            p = sd.pokemon[a - 8]
            if p.hp <= 0 or a - 8 == int(sd.active_index):
                return None
            out.append((a, p.id.lower()))
    return out


def valid_options(sd):
    """The option strings poke-engine accepts for a side (MoveChoice::from_string): the active's
    moves that aren't disabled (and their tera versions), the living benched Pokemon, "none"."""
    act = sd.pokemon[int(sd.active_index)]
    moves = [m.id.lower() for m in act.moves if m.id.lower() != "none" and not m.disabled]
    bench = [p.id.lower() for k, p in enumerate(sd.pokemon) if p.hp > 0 and k != int(sd.active_index)]
    return set(moves) | {m + "-tera" for m in moves} | set(bench) | {"none"}


def shard_cache(path, cache_dir, max_opp=4, min_opp_share=0.05, matchup=False, keep_frac=1.0, seed=0):
    """The successor features of every usable decision of a shard (cached as an npz). matchup: also
    the per-Pokemon leaf features and active pair of the successors (opt_phi: 12 x M_LEAF + 36) and
    the root's matchup inputs and pair features (for search/matchup_eval.py); keep_frac: a random
    subset of the decisions."""
    out_path = os.path.join(cache_dir, os.path.basename(os.path.dirname(path)) + "__" + os.path.basename(path))
    if os.path.exists(out_path):
        return out_path
    with np.load(path) as d:
        mask, target, q, searched = d["mask"], d["target"], d["q"], d["searched"]
        game, side, turn = d["game"], d["side"], d["turn"]
    with gzip.open(path[:-4] + ".states.gz", "rt") as f:
        states = f.read().splitlines()
    x = np.asarray(pe.features_from_strings(states), np.float32).reshape(len(states), N_X)
    fs1, fs2 = x[:, FS1] > 0.5, x[:, FS2] > 0.5
    rows = {}
    for i in range(len(states)):
        rows.setdefault((int(game[i]), int(side[i]), int(turn[i]), bool(fs1[i])), i)
    parsed = {}

    def state_of(i):
        if i not in parsed:
            parsed[i] = pe.State.from_string(states[i])
        return parsed[i]

    roots, s1, s2, w2, opts = [], [], [], [], []
    pick = np.random.default_rng(seed + zlib.crc32(os.path.basename(path).encode())).random(len(states)) < keep_frac
    skipped = {"one option": 0, "not searched": 0, "no opponent row": 0, "mapping": 0, "waiting": 0,
               "opponent row doesn't fit": 0}
    for i in range(len(states)):
        if not searched[i]:
            skipped["not searched"] += 1
            continue
        if not pick[i]:
            continue
        if mask[i].sum() < 2:
            skipped["one option"] += 1
            continue
        if not fs1[i] and fs2[i]:
            skipped["waiting"] += 1
            continue
        mine = option_strings(state_of(i), mask[i])
        if mine is None:
            skipped["mapping"] += 1
            continue
        keep = [(a, s) for a, s in mine if target[i, a] >= MIN_SHARE and np.isfinite(q[i, a])]
        if len(keep) < 2:
            skipped["one option"] += 1
            continue
        if fs1[i] and not fs2[i]:
            theirs, wts = ["none"], [1.0]
        else:
            j = rows.get((int(game[i]), 3 - int(side[i]), int(turn[i]), bool(fs1[i])))
            if j is None:
                skipped["no opponent row"] += 1
                continue
            m2 = option_strings(state_of(j), mask[j])
            if m2 is None:
                skipped["mapping"] += 1
                continue
            ok = valid_options(state_of(i).side_two)
            if not all(s in ok for _, s in m2):   # its row is another moment of the turn (U-turn, ...)
                skipped["opponent row doesn't fit"] += 1
                continue
            cand = sorted(((float(target[j, a]), s) for a, s in m2), reverse=True)
            cand = [c for c in cand if c[0] >= min_opp_share][:max_opp] or cand[:1]
            theirs, wts = [c[1] for c in cand], [c[0] for c in cand]
        roots.append(i)
        s1.append([s for _, s in keep])
        s2.append(theirs)
        w2.append(wts)
        opts.append([a for a, _ in keep])
    n_opt = np.array([len(o) for o in opts], np.int16)
    n_mu = 12 * 16 + 36 if matchup else 0
    nf = N_X + n_mu
    feats = np.zeros((int(n_opt.sum()), nf + 3), np.float32)
    o = 0
    for b in range(0, len(roots), 400):
        sl = slice(b, b + 400)
        raw = pe.option_features_from_strings([states[i] for i in roots[sl]], s1[sl], s2[sl], w2[sl], matchup)
        arr = np.frombuffer(raw, np.float32).reshape(-1, nf + 3)
        feats[o:o + len(arr)] = arr
        o += len(arr)
    assert o == len(feats)
    extra = {}
    if matchup:
        rs = [states[i] for i in roots]
        mi, mf = pe.matchup_inputs_from_strings(rs)
        extra = {"opt_phi": feats[:, N_X:nf].astype(np.float16),
                 "root_ints": np.asarray(mi, np.int16).reshape(len(rs), -1),
                 "root_floats": np.asarray(mf, np.float32).reshape(len(rs), -1).astype(np.float16),
                 "root_pairs": np.asarray(pe.pair_features_from_strings(rs), np.float32).reshape(len(rs), -1).astype(np.float16)}
    flat = np.concatenate([np.array(a, np.int8) for a in opts]) if opts else np.zeros(0, np.int8)
    r = np.array(roots, np.int32)
    rep = np.repeat(r, n_opt)
    np.savez(out_path + ".tmp.npz", root=r, n_opt=n_opt, action=flat, share=target[rep, flat].astype(np.float32),
             q=q[rep, flat].astype(np.float32), x=feats[:, :N_X].astype(np.float16), term=feats[:, nf:], **extra,
             root_x=x[r], game=game[r].astype(np.int32), turn=turn[r].astype(np.int16),
             skipped=np.array(json.dumps(skipped)))
    os.replace(out_path + ".tmp.npz", out_path)
    return out_path


class QData:
    """The cached decisions of some shards: per option its successor features z (antisymmetric:
    x_s1 - x_s2, cross), target t, the root index and the game key; per root the offset."""

    def __init__(self, files, cache_dir, label, max_term=0.02, keep_frac=1.0, seed=0):
        t0 = time.time()
        zs, ts, roots, games, shares = [], [], [], [], []
        n_roots = 0
        skipped = {}
        rng = np.random.default_rng(seed)
        for si, f in enumerate(files):
            c = np.load(shard_cache(f, cache_dir))
            for k, v in json.loads(str(c["skipped"])).items():
                skipped[k] = skipped.get(k, 0) + v
            n_opt = c["n_opt"].astype(np.int64)
            if not len(n_opt):
                continue
            start = np.r_[0, np.cumsum(n_opt)[:-1]]
            rid = np.repeat(np.arange(len(n_opt)), n_opt)
            term = c["term"]
            bad = np.zeros(len(n_opt), bool)
            np.logical_or.at(bad, rid, term[:, 0] < 1 - max_term)   # the turn may end the battle
            if keep_frac < 1:
                bad |= rng.random(len(n_opt)) >= keep_frac
            keep = ~bad[rid]
            x = c["x"].astype(np.float32)
            z = np.concatenate([x[:, :N_SIDE] - x[:, N_SIDE:2 * N_SIDE], x[:, 2 * N_SIDE:]], 1)[keep]
            kappa = KAPPA[os.path.basename(os.path.dirname(f))]
            t = kappa * np.log(np.clip(c["q"], 0.01, 0.99) / (1 - np.clip(c["q"], 0.01, 0.99)))
            new_id = np.cumsum(~bad) - 1 + n_roots
            zs.append(z), ts.append(t[keep]), roots.append(new_id[rid][keep])
            games.append((si * 100_000 + c["game"].astype(np.int64))[rid][keep]), shares.append(c["share"][keep])
            n_roots += int((~bad).sum())
            if (si + 1) % 25 == 0:
                print(f"  {label}: {si + 1}/{len(files)} shards, {n_roots:,} decisions, {time.time() - t0:.0f} s", flush=True)
        self.z, self.t, self.root = np.concatenate(zs), np.concatenate(ts), np.concatenate(roots)
        self.game, self.share = np.concatenate(games), np.concatenate(shares)
        self.n_roots = n_roots
        print(f"  {label}: {len(files)} shards, {n_roots:,} decisions, {len(self.t):,} options "
              f"({len(self.t) / max(n_roots, 1):.1f} per decision), {time.time() - t0:.0f} s; skipped {skipped}", flush=True)

    def centered(self, v):
        """v (per option) minus its decision's mean."""
        s = np.bincount(self.root, v, self.n_roots)
        n = np.bincount(self.root, minlength=self.n_roots)
        return v - (s / np.maximum(n, 1))[self.root]


def lin_coefs(path):
    """lin's antisymmetric coefficients (58: own minus opponent's side features, cross) of an .npz."""
    with np.load(path) as d:
        w1 = d["w1"].astype(np.float64)[0]
    return np.concatenate([w1[:N_SIDE] - w1[N_SIDE:2 * N_SIDE], 2 * w1[2 * N_SIDE:]])


def metrics(data, coef):
    """How the one-turn lookahead with `coef` values a decision's options against the search: the
    mean per-decision correlation of the centered values, the best option = the search's best q,
    = its most visited option, and the spread (std) of the centered values against the targets'."""
    m = data.centered(data.z @ coef)
    t = data.centered(data.t)
    num = np.bincount(data.root, m * t, data.n_roots)
    den = np.sqrt(np.bincount(data.root, m * m, data.n_roots) * np.bincount(data.root, t * t, data.n_roots))
    ok = den > 1e-9

    def best(v):
        order = np.lexsort((-v, data.root))
        return order[np.r_[True, data.root[order][1:] != data.root[order][:-1]]]
    bm = best(m)
    return {"corr": float((num[ok] / den[ok]).mean()), "top_q": float((bm == best(data.t)).mean()),
            "top_visits": float((bm == best(data.share)).mean()), "spread": float(m.std()), "t_spread": float(t.std())}


def fmt(r):
    return (f"corr {r['corr']:+.3f} | best option = the search's best q {r['top_q']:.1%}, its most visited "
            f"{r['top_visits']:.1%} | spread {r['spread']:.3f} (targets {r['t_spread']:.3f})")


def antisym(x):
    return np.concatenate([x[:, :N_SIDE] - x[:, N_SIDE:2 * N_SIDE], x[:, 2 * N_SIDE:]], 1)


def fit(zs, ys, zq, tq, lam, c0, l2=1e-5, steps=1500, lr=0.01):
    """min BCE(zs . coef, ys) + lam * mean((alpha * zq . coef - tq)^2) / var(tq) + l2 |coef std|^2 over
    coef and a free alpha, from c0 (zq, tq centered per decision); Adam in float64."""
    import torch
    import torch.nn.functional as F
    d = torch.float64
    sd = torch.as_tensor(zs.std(0) + 1e-6, dtype=d)
    Zs, Ys = torch.as_tensor(zs, dtype=d) / sd, torch.as_tensor(ys, dtype=d)
    Zq, Tq = torch.as_tensor(zq, dtype=d) / sd, torch.as_tensor(tq, dtype=d)
    vt = float(Tq.var())
    w = (torch.tensor(c0, dtype=d) * sd).requires_grad_(True)
    m0 = Zq @ w.detach()
    alpha = torch.tensor(float((m0 * Tq).sum() / (m0 * m0).sum()), dtype=d, requires_grad=True)
    opt = torch.optim.Adam([w, alpha], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Zs @ w, Ys) + l2 * (w ** 2).sum()
        if lam > 0:
            loss = loss + lam * ((alpha * (Zq @ w) - Tq) ** 2).mean() / vt
        loss.backward()
        opt.step()
    return (w.detach() / sd).numpy(), float(alpha.detach())


def least_squares(files, cache_dir, ridge=1e-3, chunk=20):
    """The coefficients whose one-turn values best fit the options' centered targets, streamed over
    the shards (normal equations of the standardized centered features, ridge toward 0)."""
    a, b, n = np.zeros((58, 58)), np.zeros(58), 0
    for k in range(0, len(files), chunk):
        d = QData(files[k:k + chunk], cache_dir, f"train {k}")
        cnt = np.maximum(np.bincount(d.root, minlength=d.n_roots), 1)
        zc = d.z.astype(np.float64) - (np.stack([np.bincount(d.root, d.z[:, j], d.n_roots) for j in range(58)], 1)
                                       / cnt[:, None])[d.root]
        tc = d.centered(d.t).astype(np.float64)
        a += zc.T @ zc
        b += zc.T @ tc
        n += len(tc)
    sd = np.sqrt(np.diag(a) / n) + 1e-9
    return np.linalg.solve(a / np.outer(sd, sd) / n + ridge * np.eye(58), b / sd / n) / sd, n


def state_logloss(z, y, coef):
    v = z @ coef
    p = 1 / (1 + np.exp(-v))
    return float(-(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))).mean())


def save(path, coef, lin_init, meta):
    """The engine's evaluator format (hidden 0): w1_s1 = the side coefficients, w1_s2 = 0, w1_cross = half."""
    with np.load(lin_init) as d:
        names, mu, sigma = d["names"], d["mu"], d["sigma"]
    w1 = np.zeros((1, N_X), np.float32)
    w1[0, :N_SIDE] = coef[:N_SIDE]
    w1[0, 2 * N_SIDE:] = coef[N_SIDE:] / 2
    np.savez(path + ".tmp.npz", hidden=np.int64(0), w1=w1, b1=np.zeros(0, np.float32), w2=np.zeros(0, np.float32),
             names=names, mu=mu, sigma=sigma, meta=np.array(json.dumps(meta)))
    os.replace(path + ".tmp.npz", path)


def main():
    ap = argparse.ArgumentParser(description="Train the leaf evaluation on the search's per-option values")
    ap.add_argument("--data", nargs="+", default=["datasets/search_selfplay", "datasets/search_selfplay_r1"])
    ap.add_argument("--cache", default="datasets/q_cache")
    ap.add_argument("--mode", choices=["ls", "combined"], default="ls",
                    help="ls: least squares on the targets, rescaled to --lin-init's one-turn spread (linq_ls); "
                         "combined: the outcome loss plus --lam times the targets' loss with a free scale")
    ap.add_argument("--out", default="models/leaf_eval/linq_ls.npz", help="combined: a prefix, <out>_lam<lam>.npz")
    ap.add_argument("--lin-init", default="models/leaf_eval/lin1_86.npz")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--keep", type=float, default=0.3, help="combined: fraction of the training states")
    ap.add_argument("--keep-q", type=float, default=0.5, help="combined: fraction of the training decisions")
    ap.add_argument("--shard-step", type=int, default=4, help="combined: every k-th training shard")
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--max-shards", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    train_files, val_files = split_shards(args.data, args.val_mod)
    if args.max_shards:
        train_files, val_files = train_files[:args.max_shards], val_files[:max(1, args.max_shards // 10)]
    for f in train_files + val_files:   # build the cache first
        shard_cache(f, args.cache)
    va = QData(val_files, args.cache, "validation")
    c0 = lin_coefs(args.lin_init)
    r0 = metrics(va, c0)
    print(f"[{os.path.basename(args.lin_init)}] options: {fmt(r0)}", flush=True)
    if args.mode == "ls":
        coef, n = least_squares(train_files, args.cache, args.ridge)
        factor = r0["spread"] / metrics(va, coef)["spread"]
        coef = coef * factor
        r = metrics(va, coef)
        print(f"[least squares, rescaled x{factor:.2f}] options: {fmt(r)}", flush=True)
        save(args.out, coef, args.lin_init, {"mode": "ls", "ridge": args.ridge, "train_options": n, "val_options": r,
                                             "val_lin_init": r0, "args": vars(args)})
        print(f"saved {args.out}")
        return
    from training.train_leaf_eval import load_many
    sub = train_files[::args.shard_step]
    tr = QData(sub, args.cache, "train", keep_frac=args.keep_q)
    rng = np.random.default_rng(0)
    xs, ys, _ = load_many(sub, args.keep, rng, "train states")
    xv, yv, _ = load_many(val_files, 1.0, rng, "validation states")
    zs, zv = antisym(xs), antisym(xv)
    del xs, xv
    cnt = np.maximum(np.bincount(tr.root, minlength=tr.n_roots), 1)
    zq = tr.z - (np.stack([np.bincount(tr.root, tr.z[:, k], tr.n_roots) for k in range(tr.z.shape[1])], 1)
                 / cnt[:, None])[tr.root]
    tq = tr.centered(tr.t)
    print(f"[{os.path.basename(args.lin_init)}] state log-loss {state_logloss(zv, yv, c0):.4f}", flush=True)
    for lam in args.lam:
        coef, alpha = fit(zs, ys, zq, tq, lam, c0)
        r = metrics(va, coef)
        ll = state_logloss(zv, yv, coef)
        print(f"[lam {lam:<5g}] state log-loss {ll:.4f} | options: {fmt(r)} | alpha {alpha:.2f}", flush=True)
        save(f"{args.out}_lam{lam:g}.npz", coef, args.lin_init,
             {"mode": "combined", "lam": lam, "alpha": alpha, "val_state_logloss": ll, "val_options": r,
              "train_decisions": tr.n_roots, "args": vars(args)})


if __name__ == "__main__":
    main()
