"""Train the learned leaf evaluation (search/leaf_eval.py, agent_docs/step7_brief.md step 3) on the
search self-play states of training/gen_search_games.py: v(state) = g(x) - g(swap(x)), the logit of
the decider (side one of the stored state) winning, fit to the game results with the logistic loss
(ties count as half a win). The features come from the engine (`pe.features_from_strings`), so
training and search compute the same ones. Whole games are held out: shards whose name hashes to 0
mod --val-mod, as training/distill_v21.py.

Baselines on the same validation states: a constant (log-loss 0.693), and the hand-written
evaluation with one fitted scale (v = a * eval / 100), which says how well Foul Play's evaluation
predicts results, and what its leaf score sigmoid(0.0125 * diff) is in logits (a = 1.25 would make
them the same).

    python3 -m training.train_leaf_eval --hidden 0 --out models/leaf_eval_linear.npz
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
import torch
import torch.nn.functional as F

from search.leaf_eval import has_engine_support

TURN_BUCKETS = ((1, 5), (6, 15), (16, 30), (31, 10_000))


def split_shards(path, val_mod):
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    val = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % val_mod == 0]
    return [f for f in files if f not in set(val)], val


def read_states(path):
    with gzip.open(path[:-4] + ".states.gz", "rt") as f:
        return f.read().splitlines()


def load_shard(path, keep_frac, rng):
    """features (n, F) float32, label (n,) 0 / 0.5 / 1, turn (n,), and the kept rows."""
    with np.load(path) as d:
        outcome, turn = d["outcome"].astype(np.float32), d["turn"].astype(np.int32)
    states = read_states(path)
    assert len(states) == len(outcome), path
    keep = np.arange(len(states)) if keep_frac >= 1 else np.flatnonzero(rng.random(len(states)) < keep_frac)
    x = np.asarray(pe.features_from_strings([states[i] for i in keep]), dtype=np.float32)
    return x.reshape(len(keep), -1), (outcome[keep] + 1) / 2, turn[keep], keep


def load_many(paths, keep_frac, rng, label):
    xs, ys, ts = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        x, y, t, _ = load_shard(p, keep_frac, rng)
        xs.append(x), ys.append(y), ts.append(t)
        if (i + 1) % 100 == 0:
            print(f"  {label}: {i + 1}/{len(paths)} shards, {sum(len(v) for v in ys):,} states, "
                  f"{time.time() - t0:.0f} s", flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)


class Antisym(torch.nn.Module):
    """v(x) = g(x) - g(swap(x)) on standardized features."""

    def __init__(self, n_side, n_cross, hidden, mu, sigma):
        super().__init__()
        self.ns, self.hidden = n_side, hidden
        n = 2 * n_side + n_cross
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sigma", torch.as_tensor(sigma))
        if hidden:
            self.l1 = torch.nn.Linear(n, hidden)
            self.l2 = torch.nn.Linear(hidden, 1, bias=False)
        else:
            self.lin = torch.nn.Linear(n, 1, bias=False)
            torch.nn.init.zeros_(self.lin.weight)

    def swap(self, x):
        ns = self.ns
        return torch.cat([x[:, ns:2 * ns], x[:, :ns], -x[:, 2 * ns:]], 1)

    def g(self, z):
        if self.hidden:
            return self.l2(torch.relu(self.l1(z))).squeeze(-1)
        return self.lin(z).squeeze(-1)

    def forward(self, x):
        return self.g((x - self.mu) / self.sigma) - self.g((self.swap(x) - self.mu) / self.sigma)

    def export(self):
        """Raw-feature weights for pe.load_evaluator (standardization folded in)."""
        inv = (1.0 / self.sigma).cpu().numpy()
        mu = self.mu.cpu().numpy()
        if self.hidden:
            w = self.l1.weight.detach().cpu().numpy() * inv[None, :]
            b = self.l1.bias.detach().cpu().numpy() - w @ mu
            return w.astype(np.float32), b.astype(np.float32), self.l2.weight.detach().cpu().numpy()[0].astype(np.float32)
        w = self.lin.weight.detach().cpu().numpy() * inv[None, :]
        return w.astype(np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)


def side_stats(x, ns):
    """Mean and std pooled over the two sides (so that standardizing commutes with the swap);
    cross features are only scaled (they are antisymmetric, mean 0)."""
    both = np.concatenate([x[:, :ns], x[:, ns:2 * ns]])
    mu_s, sd_s = both.mean(0), both.std(0)
    sd_c = np.sqrt((x[:, 2 * ns:] ** 2).mean(0))
    mu = np.concatenate([mu_s, mu_s, np.zeros(x.shape[1] - 2 * ns)])
    sd = np.concatenate([sd_s, sd_s, sd_c])
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def metrics(v, y, turn):
    v, y = np.asarray(v, np.float64), np.asarray(y, np.float64)
    p = 1 / (1 + np.exp(-v))
    ll = -(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))
    dec = y != 0.5
    r = {"logloss": float(ll.mean()), "acc": float(((v > 0) == (y > 0.5))[dec].mean()),
         "brier": float(((p - y) ** 2).mean())}
    for lo, hi in TURN_BUCKETS:
        m = (turn >= lo) & (turn <= hi)
        if m.any():
            r[f"logloss_t{lo}"] = float(ll[m].mean())
            r[f"acc_t{lo}"] = float(((v > 0) == (y > 0.5))[m & dec].mean())
    return r


def fmt(r):
    return (f"log-loss {r['logloss']:.4f} acc {r['acc']:.1%} brier {r['brier']:.4f} | by turn: "
            + " ".join(f"{lo}-{hi if hi < 10_000 else ''} {r.get(f'logloss_t{lo}', float('nan')):.3f}/"
                       f"{r.get(f'acc_t{lo}', float('nan')):.1%}" for lo, hi in TURN_BUCKETS))


def fit_scale(e, y, iters=50):
    """1-parameter logistic regression v = a * e (Newton's method)."""
    a = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-a * e))
        g = ((p - y) * e).mean()
        h = (p * (1 - p) * e * e).mean() + 1e-12
        a -= g / h
    return a


def main():
    ap = argparse.ArgumentParser(description="Train the learned leaf evaluation (step 7, step 3)")
    ap.add_argument("--data", default="datasets/search_selfplay")
    ap.add_argument("--out", default="models/leaf_eval_linear.npz")
    ap.add_argument("--hidden", type=int, default=0, help="0: linear; else one ReLU layer of this size")
    ap.add_argument("--keep", type=float, default=0.5, help="fraction of the training states used")
    ap.add_argument("--val-mod", type=int, default=20)
    ap.add_argument("--val-keep", type=float, default=0.5)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20, help="MLP (the linear model is fit by L-BFGS)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--l2", type=float, default=1e-5, help="linear: L2 penalty on the standardized weights")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not has_engine_support():
        raise SystemExit("this poke-engine build has no learned evaluation: bash search/engine_patch/build.sh")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names = list(pe.feature_names())
    n_cross = sum(1 for n in names if not n.startswith(("s1_", "s2_")))
    ns = (len(names) - n_cross) // 2

    train_files, val_files = split_shards(args.data, args.val_mod)
    if args.max_shards:
        train_files = train_files[:args.max_shards]
    t0 = time.time()
    xt, yt, tt = load_many(train_files, args.keep, rng, "train")
    xv, yv, tv = load_many(val_files, args.val_keep, rng, "validation")
    print(f"Leaf evaluation, {'linear' if not args.hidden else f'MLP {args.hidden}'} on {len(names)} features | "
          f"train {len(train_files)} shards, {len(yt):,} states | validation {len(val_files)} shards, "
          f"{len(yv):,} states | features in {time.time() - t0:.0f} s", flush=True)

    hand_t, hand_v = xt[:, names.index("hand_eval")], xv[:, names.index("hand_eval")]
    a = fit_scale(hand_t.astype(np.float64), yt.astype(np.float64))
    print(f"[constant 0.5]          {fmt(metrics(np.zeros(len(yv)), yv, tv))}")
    print(f"[hand-written, a={a:.3f}] {fmt(metrics(a * hand_v, yv, tv))}  (a per 100 points; the engine's "
          f"leaf score uses 1.25)", flush=True)

    mu, sd = side_stats(xt, ns)
    model = Antisym(ns, n_cross, args.hidden, mu, sd)
    Xt, Yt = torch.as_tensor(xt), torch.as_tensor(yt)
    Xv = torch.as_tensor(xv)
    if not args.hidden:   # convex: full-batch L-BFGS to the optimum
        opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=500, history_size=20,
                                tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(Xt), Yt) + args.l2 * (model.lin.weight ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            tl = float(F.binary_cross_entropy_with_logits(model(Xt), Yt))
            vv = model(Xv).numpy()
        print(f"L-BFGS: train log-loss {tl:.4f} | validation {fmt(metrics(vv, yv, tv))}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        steps = args.epochs * (len(yt) // args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
        for ep in range(args.epochs):
            perm = torch.randperm(len(yt))
            tot = 0.0
            for k in range(len(yt) // args.batch):
                sel = perm[k * args.batch:(k + 1) * args.batch]
                loss = F.binary_cross_entropy_with_logits(model(Xt[sel]), Yt[sel])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach())
            with torch.no_grad():
                vv = model(Xv).numpy()
            print(f"epoch {ep + 1}: train log-loss {tot / max(1, len(yt) // args.batch):.4f} | validation "
                  f"{fmt(metrics(vv, yv, tv))}", flush=True)

    w1, b1, w2 = model.export()
    meta = {"hidden": args.hidden, "train_shards": len(train_files), "train_states": int(len(yt)),
            "val_shards": len(val_files), "val": metrics(vv, yv, tv), "hand_scale": a,
            "hand_val": metrics(a * hand_v, yv, tv), "args": vars(args)}
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, hidden=np.int64(args.hidden), w1=w1, b1=b1, w2=w2, names=np.array(names),
             mu=mu, sigma=sd, meta=np.array(json.dumps(meta)))
    os.replace(tmp, args.out)
    print(f"saved {args.out}")

    # the engine's evaluator must give the same values as the model
    from search.leaf_eval import evaluator_id
    states = read_states(val_files[0])[:2000]
    eid = evaluator_id(args.out)
    rust = np.asarray(pe.evaluator_values(eid, states))
    x = np.asarray(pe.features_from_strings(states), dtype=np.float32).reshape(len(states), -1)
    with torch.no_grad():
        ours = model(torch.as_tensor(x)).numpy()
    print(f"engine vs model on {len(states)} states: max |diff| {np.abs(rust - ours).max():.2e}")
    if not args.hidden:
        lin = w1[0]
        coef = lin[:ns] - lin[ns:2 * ns]
        print("linear weights (logits per unit; side features: own minus opponent's):")
        for i in np.argsort(-np.abs(coef * sd[:ns])):
            print(f"  {names[i][3:]:<16} {coef[i]:+8.4f}   (x std {coef[i] * sd[i]:+.3f})")
        for j in range(2 * ns, len(names)):
            print(f"  {names[j]:<16} {2 * lin[j]:+8.4f}   (x std {2 * lin[j] * sd[j]:+.3f})")


if __name__ == "__main__":
    main()
