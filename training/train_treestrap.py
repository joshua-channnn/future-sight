"""Train the leaf evaluation toward the values of the searched nodes (TreeStrap; agent_docs/step8.md
section 5) on training/gen_treestrap.py's data. For each node n of a search, but the root and
nodes where the battle is over, the target is the search's value relative to the root,
t = logit(q_n) / scale: mode 2 scores a leaf sigmoid(scale * (v(leaf) - v(root))), so t is the
v(n) - v(root) that a leaf at n would need to score what the search below n found. The model is
lin's (train_q_eval.lin_coefs: 58 antisymmetric coefficients), v(n) - v(root) = coef . (z_n - z_root).

--mode ls: least squares (closed form, ridge toward 0 on the standardized features), saved as
fitted (<out>_raw.npz) and rescaled so that its node values spread like --lin-init's (<out>.npz;
the search's leaf scale is tuned to that spread, and the fit is compressed like the targets).
--mode combined: the roots' outcome loss (train_leaf_eval's) plus lam times the node loss with a
free scale (train_q_eval.fit), saved as <out>_lam<lam>.npz.
--mode mlp: g one ReLU layer (--hidden; train_leaf_eval.Antisym) on the node loss (plus
--outcome-weight times the roots' outcome loss), rescaled like ls, saved as <out>_mlp<hidden>.npz;
--init-lin starts it as --lin-init (two units are its linear model) so it learns corrections.
Validation (held-out games, train_q_eval.split_shards): the nodes' correlation and R^2 with the
best scale, per depth, the spread, and the roots' outcome log-loss, for --lin-init and the fits.

    python3 -m training.train_treestrap --data datasets/treestrap_r0 --out models/leaf_eval/lints_r0
"""
import argparse
import glob
import json
import os
import time
import zlib

import numpy as np

from training.train_q_eval import antisym, fit, lin_coefs, save, state_logloss


class Nodes:
    """The nodes of training/gen_treestrap.py's shards: dz = z_n - z_root (58, float32), t, depth,
    visits; and per search the root's z and the decider's result (0 / 0.5 / 1)."""

    def __init__(self, files, scale, min_visits=0, min_depth=1, max_depth=99, label=""):
        t0 = time.time()
        dz, t, depth, visits, zr, y = [], [], [], [], [], []
        for f in files:
            with np.load(f) as d:
                z = antisym(d["x"].astype(np.float32))
                tree, dep, vis, q, over = d["tree"], d["depth"], d["visits"], d["q"], d["over"]
                root = np.flatnonzero(dep == 0)
                assert np.array_equal(tree[root], np.arange(len(root))), f
                keep = ~over & (dep >= min_depth) & (dep <= max_depth) & (vis >= min_visits)
                dz.append(z[keep] - z[root][tree[keep]])
                qk = np.clip(q[keep].astype(np.float64), 1e-4, 1 - 1e-4)
                t.append((np.log(qk / (1 - qk)) / scale).astype(np.float32))
                depth.append(dep[keep]), visits.append(vis[keep])
                zr.append(z[root]), y.append((d["outcome"].astype(np.float32) + 1) / 2)
        self.dz, self.t = np.concatenate(dz), np.concatenate(t)
        self.depth, self.visits = np.concatenate(depth), np.concatenate(visits)
        self.zr, self.y = np.concatenate(zr), np.concatenate(y)
        print(f"  {label}: {len(files)} shards, {len(self.y):,} searches, {len(self.t):,} nodes, "
              f"{time.time() - t0:.0f} s", flush=True)


def split(files, val_mod=10):
    """Validation: the shards of held-out games (the source shard's name, as train_q_eval.split_shards)."""
    def src(f):
        return os.path.basename(f).split("__", 1)[1]
    val = [f for f in files if zlib.crc32(src(f).encode()) % val_mod == 0]
    return [f for f in files if f not in set(val)], val


def metrics(data, coef):
    p = data.dz.astype(np.float64) @ coef
    t = data.t.astype(np.float64)

    def corr(m):
        return float(np.corrcoef(p[m], t[m])[0, 1]) if m.sum() > 2 else float("nan")
    alpha = float(p @ t / (p @ p))
    r2 = 1 - float(((alpha * p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return {"corr": corr(np.ones(len(t), bool)), "r2": r2, "alpha": alpha, "spread": float(p.std()),
            "t_spread": float(t.std()), "corr_d1": corr(data.depth == 1), "corr_d2": corr(data.depth == 2),
            "corr_d3+": corr(data.depth >= 3)}


def fmt(r):
    return (f"corr {r['corr']:.3f} (depth 1 {r['corr_d1']:.3f}, 2 {r['corr_d2']:.3f}, 3+ {r['corr_d3+']:.3f}) | "
            f"R^2 {r['r2']:.3f} at scale {r['alpha']:.2f} | spread {r['spread']:.3f} (targets {r['t_spread']:.3f})")


class RawNodes:
    """Nodes with the raw features (float16; --mode mlp): x, the index of their search's root in
    xr, t, depth; per search the root's x and the decider's result."""

    def __init__(self, files, scale, min_visits=0, min_depth=1, label=""):
        t0 = time.time()
        x, root, t, depth, xr, y = [], [], [], [], [], []
        n_roots = 0
        for f in files:
            with np.load(f) as d:
                xs, tree, dep, vis, q, over = d["x"], d["tree"], d["depth"], d["visits"], d["q"], d["over"]
                roots = np.flatnonzero(dep == 0)
                keep = ~over & (dep >= min_depth) & (vis >= min_visits)
                x.append(xs[keep]), root.append(tree[keep] + n_roots), depth.append(dep[keep])
                qk = np.clip(q[keep].astype(np.float64), 1e-4, 1 - 1e-4)
                t.append((np.log(qk / (1 - qk)) / scale).astype(np.float32))
                xr.append(xs[roots]), y.append((d["outcome"].astype(np.float32) + 1) / 2)
                n_roots += len(roots)
        self.x, self.root, self.t = np.concatenate(x), np.concatenate(root), np.concatenate(t)
        self.depth, self.xr, self.y = np.concatenate(depth), np.concatenate(xr), np.concatenate(y)
        print(f"  {label}: {len(files)} shards, {n_roots:,} searches, {len(self.t):,} nodes, "
              f"{time.time() - t0:.0f} s", flush=True)


def train_mlp(args, train_files, val_files, c0, meta):
    """g = one ReLU layer (train_leaf_eval.Antisym): the node loss (relative values, as --mode ls)
    plus --outcome-weight times the roots' outcome loss; then w2 rescaled to --lin-init's spread."""
    import torch
    import torch.nn.functional as F
    from training.train_leaf_eval import Antisym, side_stats
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tr = RawNodes(train_files, args.scale, args.min_visits, args.min_depth, "train")
    va = RawNodes(val_files, args.scale, args.min_visits, args.min_depth, "validation")
    rng = np.random.default_rng(0)
    sample = tr.x[rng.choice(len(tr.x), min(len(tr.x), 200_000), replace=False)].astype(np.float32)
    mu, sd = side_stats(sample, 52)
    model = Antisym(52, 6, args.hidden, mu, sd)
    if args.init_lin:   # units 0 and 1: relu(a) - relu(-a) = a = --lin-init's g; the others start small
        with np.load(args.lin_init) as d:
            w = d["w1"].astype(np.float64)[0]
        lin = np.concatenate([w[:52] - w[52:104], w[52:104] - w[:52], 2 * w[104:]])   # learned.rs's `linear`
        row, b = 0.5 * lin * sd, 0.5 * float(lin @ mu)   # g = lin . x / 2 on the standardized features
        with torch.no_grad():
            model.l1.weight[0], model.l1.bias[0] = torch.as_tensor(row, dtype=torch.float32), b
            model.l1.weight[1], model.l1.bias[1] = -torch.as_tensor(row, dtype=torch.float32), -b
            model.l2.weight[0, :2] = torch.tensor([1.0, -1.0])
            model.l2.weight[0, 2:] *= 0.01
    vt = float(tr.t.var())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr.t) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
    Xr, Yr = torch.as_tensor(tr.xr.astype(np.float32)), torch.as_tensor(tr.y)

    def rel(x, root, xr_all):
        xn = torch.as_tensor(x.astype(np.float32))
        return model(xn) - model(xr_all[torch.as_tensor(root)])

    xva = torch.as_tensor(va.xr.astype(np.float32))

    def val_values():
        with torch.no_grad():
            out = []
            for k in range(0, len(va.t), 65536):
                out.append(rel(va.x[k:k + 65536], va.root[k:k + 65536], xva).numpy())
            return np.concatenate(out).astype(np.float64)

    for ep in range(args.epochs):
        perm = rng.permutation(len(tr.t))
        tot = 0.0
        for k in range(len(tr.t) // args.batch):
            sel = np.sort(perm[k * args.batch:(k + 1) * args.batch])
            loss = ((rel(tr.x[sel], tr.root[sel], Xr) - torch.as_tensor(tr.t[sel])) ** 2).mean() / vt
            if args.outcome_weight:
                rs = torch.as_tensor(rng.integers(0, len(tr.y), args.batch // 4))
                loss = loss + args.outcome_weight * F.binary_cross_entropy_with_logits(model(Xr[rs]), Yr[rs])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        r = raw_metrics(val_values(), va)
        print(f"epoch {ep + 1}: train loss {tot / max(1, len(tr.t) // args.batch):.4f} | validation nodes: {fmt(r)}",
              flush=True)
    p = val_values()
    lin_spread = float((np.concatenate([antisym(va.x[k:k + 65536].astype(np.float64)) -
                                        antisym(va.xr[va.root[k:k + 65536]].astype(np.float64))
                                        for k in range(0, len(va.t), 65536)]) @ c0).std())
    factor = lin_spread / float(p.std())
    w1, b1, w2 = model.export()
    w2 = (w2 * factor).astype(np.float32)
    with torch.no_grad():
        vr = model(xva).numpy() * factor
    yv = va.y
    ll = float(-(yv * np.log(np.clip(1 / (1 + np.exp(-vr)), 1e-7, 1)) +
                 (1 - yv) * np.log(np.clip(1 - 1 / (1 + np.exp(-vr)), 1e-7, 1))).mean())
    r = raw_metrics(p * factor, va)
    print(f"[MLP {args.hidden} x{factor:.2f}, {os.path.basename(args.lin_init)}'s spread] nodes: {fmt(r)} | "
          f"roots' log-loss {ll:.4f}", flush=True)
    with np.load(args.lin_init) as d:
        names = d["names"]
    path = f"{args.out}_mlp{args.hidden}.npz"
    np.savez(path + ".tmp.npz", hidden=np.int64(args.hidden), w1=w1, b1=b1, w2=w2, names=names, mu=mu, sigma=sd,
             meta=np.array(json.dumps({**meta, "mode": "mlp", "rescaled": factor, "val": r, "val_logloss": ll})))
    os.replace(path + ".tmp.npz", path)
    # the engine's evaluator must give the model's values: on the first validation shard's roots
    import poke_engine as pe
    from search.leaf_eval import evaluator_id
    from training.gen_treestrap import read_states
    dataset, shard = os.path.basename(val_files[0]).split("__", 1)
    with np.load(val_files[0]) as d:
        rows = d["row"][:1000]
    src = read_states(os.path.join("datasets", dataset, shard))
    states = [src[i] for i in rows]
    rust = np.asarray(pe.evaluator_values(evaluator_id(path, 2, 1.0), states))
    x = np.asarray(pe.features_from_strings(states), np.float32).reshape(len(states), -1)
    with torch.no_grad():
        theirs = model(torch.as_tensor(x)).numpy() * factor
    print(f"saved {path} (engine vs model on {len(states)} roots: max |diff| {np.abs(rust - theirs).max():.2e})")


def raw_metrics(p, data):
    """metrics() for precomputed relative values p."""
    t = data.t.astype(np.float64)

    def corr(m):
        return float(np.corrcoef(p[m], t[m])[0, 1]) if m.sum() > 2 else float("nan")
    alpha = float(p @ t / (p @ p))
    r2 = 1 - float(((alpha * p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return {"corr": corr(np.ones(len(t), bool)), "r2": r2, "alpha": alpha, "spread": float(p.std()),
            "t_spread": float(t.std()), "corr_d1": corr(data.depth == 1), "corr_d2": corr(data.depth == 2),
            "corr_d3+": corr(data.depth >= 3)}


def least_squares(files, scale, ridge, min_visits, min_depth, chunk=20):
    """Normal equations of the standardized dz, streamed over the shards."""
    a, b, n = np.zeros((58, 58)), np.zeros(58), 0
    for k in range(0, len(files), chunk):
        d = Nodes(files[k:k + chunk], scale, min_visits, min_depth, label=f"train {k}")
        dz = d.dz.astype(np.float64)
        a += dz.T @ dz
        b += dz.T @ d.t
        n += len(d.t)
    sd = np.sqrt(np.diag(a) / n) + 1e-9
    return np.linalg.solve(a / np.outer(sd, sd) / n + ridge * np.eye(58), b / sd / n) / sd, n


def main():
    ap = argparse.ArgumentParser(description="Train the leaf evaluation toward the searched nodes' values (TreeStrap)")
    ap.add_argument("--data", nargs="+", default=["datasets/treestrap_r0"])
    ap.add_argument("--out", default="models/leaf_eval/lints_r0")
    ap.add_argument("--mode", choices=["ls", "combined", "mlp"], default="ls")
    ap.add_argument("--lin-init", default="models/leaf_eval/lin1_86.npz")
    ap.add_argument("--scale", type=float, default=1.4, help="the leaf scale of the searches that made the data")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--min-visits", type=int, default=0)
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--keep", type=float, default=0.25, help="combined: the training nodes' fraction")
    ap.add_argument("--hidden", type=int, default=16, help="mlp: hidden units")
    ap.add_argument("--epochs", type=int, default=6, help="mlp")
    ap.add_argument("--batch", type=int, default=4096, help="mlp")
    ap.add_argument("--lr", type=float, default=3e-3, help="mlp")
    ap.add_argument("--outcome-weight", type=float, default=0.0, help="mlp: weight of the roots' outcome loss")
    ap.add_argument("--init-lin", action="store_true", help="mlp: start as --lin-init (2 of the hidden units)")
    args = ap.parse_args()
    files = sorted(f for p in args.data for f in glob.glob(os.path.join(p, "*.npz")) if ".tmp." not in f)
    train_files, val_files = split(files)
    va = Nodes(val_files, args.scale, args.min_visits, args.min_depth, label="validation")
    c0 = lin_coefs(args.lin_init)
    r0 = metrics(va, c0)
    ll0 = state_logloss(va.zr, va.y, c0)
    print(f"[{os.path.basename(args.lin_init)}] nodes: {fmt(r0)} | roots' log-loss {ll0:.4f}", flush=True)
    meta = {"args": vars(args), "val_lin_init": r0, "val_logloss_lin_init": ll0}
    if args.mode == "mlp":
        del va
        return train_mlp(args, train_files, val_files, c0, meta)
    if args.mode == "ls":
        coef, n = least_squares(train_files, args.scale, args.ridge, args.min_visits, args.min_depth)
        r = metrics(va, coef)
        print(f"[least squares] nodes: {fmt(r)} | roots' log-loss {state_logloss(va.zr, va.y, coef):.4f} "
              f"| cosine to {os.path.basename(args.lin_init)} {coef @ c0 / np.linalg.norm(coef) / np.linalg.norm(c0):.3f}",
              flush=True)
        save(args.out + "_raw.npz", coef, args.lin_init, {**meta, "mode": "ls", "train_nodes": n, "val": r})
        factor = r0["spread"] / r["spread"]
        rs = metrics(va, coef * factor)
        lls = state_logloss(va.zr, va.y, coef * factor)
        print(f"[least squares x{factor:.2f}, {os.path.basename(args.lin_init)}'s spread] nodes: {fmt(rs)} | "
              f"roots' log-loss {lls:.4f}", flush=True)
        save(args.out + ".npz", coef * factor, args.lin_init,
             {**meta, "mode": "ls", "rescaled": factor, "train_nodes": n, "val": rs, "val_logloss": lls})
        print(f"saved {args.out}_raw.npz, {args.out}.npz")
        return
    tr = Nodes(train_files, args.scale, args.min_visits, args.min_depth, label="train")
    rng = np.random.default_rng(0)
    sub = rng.random(len(tr.t)) < args.keep
    for lam in args.lam:
        coef, alpha = fit(tr.zr.astype(np.float64), tr.y, tr.dz[sub].astype(np.float64), tr.t[sub].astype(np.float64),
                          lam, c0)
        r = metrics(va, coef)
        ll = state_logloss(va.zr, va.y, coef)
        print(f"[lam {lam:<5g}] nodes: {fmt(r)} | roots' log-loss {ll:.4f} | alpha {alpha:.2f}", flush=True)
        save(f"{args.out}_lam{lam:g}.npz", coef, args.lin_init,
             {**meta, "mode": "combined", "lam": lam, "alpha": alpha, "val": r, "val_logloss": ll})


if __name__ == "__main__":
    main()
