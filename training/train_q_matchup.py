"""Train the matchup weights (search/matchup_eval.py) on the search's per-option values, as
training/train_q_eval.py does for the linear evaluation (agent_docs/step8.md section 4): from each
stored decision, the network sets the root's weights; every option of ours is played one turn
ahead (the cache of train_q_eval.shard_cache(..., matchup=True)); the one-turn value of an option is
lin(x) + the matchup term of its successors with the root's weights, and the loss fits
alpha * (these values, centered per decision) to the search's (t = logit(q) / 1.44 or / 1.4,
centered), alpha free: only how the options rank and differ counts. lin stays fixed (--lin), so
the matchup term is what is learned. The targets differ between the options of one decision, so
memorizing a game's result doesn't help (training/train_matchup_eval.py's problem).

    python3 -m training.train_q_matchup --out models/leaf_eval/qmu1.pt

It only reads the cache (shards missing from it are skipped), so build it first (about 5 minutes):

    import os
    from training.train_q_eval import split_shards, shard_cache
    os.makedirs("datasets/q_cache_mu", exist_ok=True)
    tr, va = split_shards(["datasets/search_selfplay", "datasets/search_selfplay_r1"], 10)
    for f in va + tr:
        shard_cache(f, "datasets/q_cache_mu", matchup=True, keep_frac=1.0 if f in va else 0.25)
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from search.matchup_eval import MatchupNet, N_FLOATS, N_INTS, N_PAIR, N_ROOT_WEIGHTS, N_SIDE, N_X, M_LEAF, \
    N_POKE_INTS, N_POKE_FLOATS, N_FIELD
from training.train_q_eval import KAPPA, lin_coefs, split_shards

N_MU = 12 * M_LEAF + 36
MAX_OPT = 14


class QMuData:
    """Decisions from a matchup cache: the root inputs per decision, the options flat (x, phi, t,
    share) with each decision's first option and count; batch() pads a batch's options."""

    def __init__(self, files, cache_dir, label, max_term=0.02):
        t0 = time.time()
        parts = {k: [] for k in ("ints", "floats", "pairs", "x", "phi", "t", "share", "n_opt", "game")}
        n = 0
        for si, f in enumerate(files):
            path = os.path.join(cache_dir, os.path.basename(os.path.dirname(f)) + "__" + os.path.basename(f))
            if not os.path.exists(path):
                continue
            c = np.load(path)
            n_opt = c["n_opt"].astype(np.int64)
            if not len(n_opt):
                continue
            rid = np.repeat(np.arange(len(n_opt)), n_opt)
            bad = np.zeros(len(n_opt), bool)
            np.logical_or.at(bad, rid, c["term"][:, 0] < 1 - max_term)
            bad |= n_opt > MAX_OPT
            ok = ~bad
            sel = ok[rid]
            kappa = KAPPA[os.path.basename(os.path.dirname(f))]
            qq = np.clip(c["q"][sel], 0.01, 0.99)
            parts["t"].append((kappa * np.log(qq / (1 - qq))).astype(np.float32))
            parts["x"].append(c["x"][sel]), parts["phi"].append(c["opt_phi"][sel]), parts["share"].append(c["share"][sel])
            parts["ints"].append(c["root_ints"][ok]), parts["floats"].append(c["root_floats"][ok])
            parts["pairs"].append(c["root_pairs"][ok]), parts["n_opt"].append(n_opt[ok])
            parts["game"].append(si * 100_000 + c["game"][ok].astype(np.int64))
            n += int(ok.sum())
            if (si + 1) % 25 == 0:
                print(f"  {label}: {si + 1}/{len(files)} shards, {n:,} decisions, {time.time() - t0:.0f} s", flush=True)
        for k, v in parts.items():
            setattr(self, k, np.concatenate(v))
        self.start = np.r_[0, np.cumsum(self.n_opt)[:-1]]
        self.n = n
        print(f"  {label}: {len(files)} shards, {n:,} decisions, {len(self.t):,} options, {time.time() - t0:.0f} s",
              flush=True)

    def batch(self, idx, device):
        k = int(self.n_opt[idx].max())
        pos = np.arange(k)[None, :]
        mask = pos < self.n_opt[idx][:, None]
        rows = np.where(mask, self.start[idx][:, None] + pos, self.start[idx][:, None])
        f = lambda a: torch.as_tensor(a.astype(np.float32), device=device)
        return (torch.as_tensor(self.ints[idx].astype(np.int64), device=device), f(self.floats[idx]),
                f(self.pairs[idx]), f(self.x[rows]), f(self.phi[rows]), f(self.t[rows]),
                torch.as_tensor(mask, device=device), f(self.share[rows]))


def option_values(net, lin_side, lin_cross, ints, floats, pairs, x, phi, matchup=True):
    """One-turn values of the options (B, MAX_OPT): lin of the successors + the matchup term."""
    v = (x[..., :N_SIDE] - x[..., N_SIDE:2 * N_SIDE]) @ lin_side + x[..., 2 * N_SIDE:] @ lin_cross
    if matchup:
        c, u1, u2, w = net(ints, floats, pairs)
        b, k = x.shape[:2]
        ph = phi[..., :12 * M_LEAF].reshape(b, k, 12, M_LEAF)
        pr = phi[..., 12 * M_LEAF:]
        v = v + (u1[:, None] * ph[:, :, :6]).sum((2, 3)) - (u2[:, None] * ph[:, :, 6:]).sum((2, 3)) \
            + (w.reshape(b, 1, 36) * pr).sum(-1)
    return v


def centered(v, mask):
    n = mask.sum(1, keepdim=True).clamp(min=1)
    mean = (v * mask).sum(1, keepdim=True) / n
    return (v - mean) * mask


@torch.no_grad()
def evaluate(net, lin_side, lin_cross, data, device, matchup, batch=1024):
    net.eval()
    corr, top_q, top_v, spread, n = [], [], [], [], 0
    for s in range(0, data.n, batch):
        idx = np.arange(s, min(s + batch, data.n))
        ints, floats, pairs, x, phi, t, mask, share = data.batch(idx, device)
        m = centered(option_values(net, lin_side, lin_cross, ints, floats, pairs, x, phi, matchup), mask)
        tc = centered(t, mask)
        num, den = (m * tc).sum(1), torch.sqrt((m * m).sum(1) * (tc * tc).sum(1))
        ok = den > 1e-9
        corr.append((num[ok] / den[ok]).cpu().numpy())
        neg = torch.tensor(-1e9, device=device)
        bm = torch.where(mask, m, neg).argmax(1)
        top_q.append((bm == torch.where(mask, t, neg).argmax(1)).cpu().numpy())
        top_v.append((bm == torch.where(mask, share, neg).argmax(1)).cpu().numpy())
        spread.append(m[mask].cpu().numpy())
    net.train()
    return {"corr": float(np.concatenate(corr).mean()), "top_q": float(np.concatenate(top_q).mean()),
            "top_visits": float(np.concatenate(top_v).mean()), "spread": float(np.concatenate(spread).std())}


def fmt(r):
    return (f"corr {r['corr']:+.3f} | best option = the search's best q {r['top_q']:.1%}, its most visited "
            f"{r['top_visits']:.1%} | spread {r['spread']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Train the matchup weights on the search's per-option values")
    ap.add_argument("--data", nargs="+", default=["datasets/search_selfplay", "datasets/search_selfplay_r1"])
    ap.add_argument("--cache", default="datasets/q_cache_mu")
    ap.add_argument("--out", default="models/leaf_eval/qmu.pt")
    ap.add_argument("--lin", default="models/leaf_eval/lin1_86.npz", help="the fixed linear part")
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=256, help="decisions")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ids", choices=["all", "items", "none"], default="none")
    ap.add_argument("--val-every", type=float, default=0.5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    train_files, val_files = split_shards(args.data, args.val_mod)
    tr = QMuData(train_files, args.cache, "train")
    va = QMuData(val_files, args.cache, "validation")

    c = lin_coefs(args.lin)
    lin_side = torch.tensor(c[:N_SIDE], dtype=torch.float32, device=dev)
    lin_cross = torch.tensor(c[N_SIDE:], dtype=torch.float32, device=dev)
    with np.load(args.lin) as d:
        mu, sigma = d["mu"], d["sigma"]
    w1 = np.load(args.lin)["w1"].astype(np.float32)[0]
    net = MatchupNet(d=args.d, layers=args.layers, dropout=args.dropout, ids=args.ids, move_props=True, pairs=True,
                     x_mu=mu, x_sigma=sigma, lin_side=w1[:N_SIDE] - w1[N_SIDE:2 * N_SIDE],
                     lin_cross=2 * w1[2 * N_SIDE:]).to(dev)
    net.lin_side.requires_grad_(False)
    net.lin_cross.requires_grad_(False)
    # alpha: the deep values' compression against one-turn values, fit by lin alone to start
    with torch.no_grad():
        idx = rng.choice(tr.n, min(tr.n, 20000), replace=False)
        ints, floats, pairs, x, phi, t, mask, share = tr.batch(idx, dev)
        m = centered(option_values(net, lin_side, lin_cross, ints, floats, pairs, x, phi, False), mask)
        tc = centered(t, mask)
        alpha0 = float((m * tc).sum() / (m * m).sum())
        vt = float((tc[mask] ** 2).mean())
    log_alpha = torch.tensor(math.log(max(alpha0, 1e-3)), device=dev, requires_grad=True)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "weight_decay": args.wd}, {"params": [log_alpha], "weight_decay": 0.0}],
                            lr=args.lr)
    steps_per_epoch = tr.n // args.batch
    total = int(args.epochs * steps_per_epoch)
    warm = max(1, total // 50)
    print(f"Matchup weights on the search's values | train {tr.n:,} decisions | validation {va.n:,} | "
          f"{sum(p.numel() for p in params):,} parameters, {total:,} steps on {dev} | alpha0 {alpha0:.3f}", flush=True)
    r0 = evaluate(net, lin_side, lin_cross, va, dev, False)
    print(f"[lin alone]  {fmt(r0)}", flush=True)
    best = (-1.0, 0.0, None, None)
    val_steps = max(1, int(args.val_every * steps_per_epoch))
    t0 = time.time()
    step, run = 0, 0.0
    while step < total:
        perm = rng.permutation(tr.n)
        for s in range(steps_per_epoch):
            if step >= total:
                break
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = args.lr * f
            ints, floats, pairs, x, phi, t, mask, share = tr.batch(perm[s * args.batch:(s + 1) * args.batch], dev)
            m = centered(option_values(net, lin_side, lin_cross, ints, floats, pairs, x, phi), mask)
            tc = centered(t, mask)
            loss = (((log_alpha.exp() * m - tc) ** 2) * mask).sum() / mask.sum() / vt
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run = 0.98 * run + 0.02 * float(loss.detach()) if step else float(loss.detach())
            step += 1
            if step % 200 == 0:
                print(f"  step {step:,}/{total:,}: loss {run:.4f} (lin alone ~{1 - r0['corr'] ** 2:.3f}), alpha "
                      f"{float(log_alpha.exp()):.3f}, {time.time() - t0:.0f} s", flush=True)
            if step % val_steps == 0 or step == total:
                r = evaluate(net, lin_side, lin_cross, va, dev, True)
                star = r["corr"] > best[0]
                if star:
                    best = (r["corr"], step / steps_per_epoch,
                            {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, r)
                print(f"epoch {step / steps_per_epoch:.2f}: {fmt(r)}{'  *' if star else ''}", flush=True)
    net.cpu()
    net.load_state_dict(best[2])
    print(f"best at epoch {best[1]:.2f}: {fmt(best[3])} (lin alone: {fmt(r0)})")
    meta = {"val": best[3], "val_lin": r0, "best_epoch": best[1], "alpha": float(log_alpha.exp()), "args": vars(args)}
    torch.save({"config": net.config, "state_dict": net.state_dict(), "matchup": True, "lin_npz": args.lin,
                "layout": [N_POKE_INTS, N_POKE_FLOATS, N_FIELD, M_LEAF, N_X, N_ROOT_WEIGHTS], "meta": meta},
               args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
