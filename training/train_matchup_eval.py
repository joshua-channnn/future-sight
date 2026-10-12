"""Train the matchup weights of the leaf evaluation (search/matchup_eval.py) on the search self-play
states of training/gen_search_games.py.

The search scores a leaf by v(leaf) - v(root), so the network is trained on pairs from the stored
games: a root (a decision's state) and a leaf (the same side's state k = 1..--max-k decisions
later), with the logistic loss on the game's result for the root's decider (ties count half):

    logit = [lin(x(root)) + c(root)] + [lin(x(leaf)) - lin(x(root))] + [m(leaf) - m(root)]

lin(x(root)) + c(root) is the root's value (c: the network's correction, used only here: it absorbs
what the root already says; it starts at 0, so training starts at lin's own prediction), lin the
linear evaluation (initialized from --lin-init, lin1_86, and trained too unless --freeze-lin), m the
matchup term with the root's weights (engine::learned). --no-matchup trains the same model with m = 0:
the baseline that says what the matchup weights add. Whole games are held out (shards whose name
hashes to 0 mod --val-mod, as train_leaf_eval.py; with 20 none of them was in lin1_86's training).

The network memorizes games through anything that identifies the teams (ids, the value head c):
the settings that generalize are those of models/leaf_eval/mu5_reg.pt (agent_docs/step8.md), which
predicts better but doesn't make the search stronger:

    python3 -m training.train_matchup_eval --val-mod 10 --no-value --ids none --move-props --pairs \
        --d 32 --layers 1 --dropout 0.3 --wd 0.01 --epochs 1.5 --out models/leaf_eval/mu5_reg.pt
"""
import argparse
import glob
import gzip
import json
import math
import os
import time
import zlib

import numpy as np
import poke_engine as pe
import torch
import torch.nn.functional as F

from search.matchup_eval import (MatchupLinear, MatchupNet, N_FLOATS, N_INTS, N_PAIR, N_ROOT_WEIGHTS, N_SIDE, N_X,
                                 has_engine_support, leaf_phi, split_inputs, N_POKE_INTS, N_POKE_FLOATS, N_FIELD,
                                 M_LEAF)


def split_shards(paths, val_mod):
    files = sorted(f for p in paths for f in glob.glob(os.path.join(p, "*.npz")))
    val = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % val_mod == 0]
    return [f for f in files if f not in set(val)], val


def read_states(path):
    with gzip.open(path[:-4] + ".states.gz", "rt") as f:
        return f.read().splitlines()


class Data:
    """Every state of some shards, sorted by game, side and decision: matchup inputs (ints int16,
    floats float16 without lin's features), lin's features x (float32), the result y, the game key
    and how many later states of the same side follow (`rest`); with pairs, pe.pair_features
    (float16)."""

    def __init__(self, files, label, pairs=False):
        t0 = time.time()
        sizes = []
        for f in files:
            with np.load(f) as d:
                sizes.append(len(d["outcome"]))
        n = sum(sizes)
        self.ints = np.zeros((n, N_INTS), np.int16)
        self.tok = np.zeros((n, N_FLOATS - N_X), np.float16)
        self.x = np.zeros((n, N_X), np.float32)
        self.y = np.zeros(n, np.float32)
        self.game = np.zeros(n, np.int64)
        self.rest = np.zeros(n, np.int32)
        self.turn = np.zeros(n, np.int16)
        self.pairs = np.zeros((n, 36 * N_PAIR), np.float16) if pairs else None
        o = 0
        for si, (f, m) in enumerate(zip(files, sizes)):
            with np.load(f) as d:
                outcome, game, side, step, turn = (d[k] for k in ("outcome", "game", "side", "step", "turn"))
            states = read_states(f)
            assert len(states) == m, f
            order = np.lexsort((step, side, game))
            ints, floats = pe.matchup_inputs_from_strings([states[i] for i in order])
            floats = np.asarray(floats, np.float32).reshape(m, N_FLOATS)
            self.ints[o:o + m] = np.asarray(ints, np.int16).reshape(m, N_INTS)
            if pairs:
                self.pairs[o:o + m] = np.asarray(pe.pair_features_from_strings([states[i] for i in order]),
                                                 np.float32).reshape(m, -1)
            self.tok[o:o + m] = floats[:, :N_FLOATS - N_X]
            self.x[o:o + m] = floats[:, N_FLOATS - N_X:]
            self.y[o:o + m] = (outcome[order].astype(np.float32) + 1) / 2
            self.game[o:o + m] = si * 100_000 + game[order].astype(np.int64)
            self.turn[o:o + m] = turn[order]
            seq = self.game[o:o + m] * 4 + side[order]
            # states of the same side that follow in this game
            ends = np.r_[np.flatnonzero(seq[1:] != seq[:-1]), m - 1]
            end_of = np.repeat(ends, np.diff(np.r_[-1, ends]))
            self.rest[o:o + m] = end_of - np.arange(m)
            o += m
            if (si + 1) % 25 == 0:
                print(f"  {label}: {si + 1}/{len(files)} shards, {o:,} states, {time.time() - t0:.0f} s", flush=True)
        print(f"  {label}: {len(files)} shards, {n:,} states, {time.time() - t0:.0f} s", flush=True)

    def batch(self, idx, device):
        ints = torch.as_tensor(self.ints[idx].astype(np.int64), device=device)
        floats = torch.as_tensor(np.concatenate([self.tok[idx].astype(np.float32), self.x[idx]], 1), device=device)
        pairs = None if self.pairs is None else torch.as_tensor(self.pairs[idx].astype(np.float32), device=device)
        return ints, floats, pairs


def pair_logits(net, data, r, l, device, matchup, value=True):
    """logit of each pair (root r, leaf l) and the parts: c, lin's and the matchup term's change."""
    ints_r, floats_r, pairs_r = data.batch(r, device)
    ints_l, floats_l, _ = data.batch(l, device)
    c, u1, u2, w = net(ints_r, floats_r, pairs_r)
    pk_r, pf_r, act_r, _, x_r = split_inputs(ints_r, floats_r)
    pk_l, pf_l, act_l, _, x_l = split_inputs(ints_l, floats_l)
    d_lin = net.lin(x_l) - net.lin(x_r)
    d_mu = torch.zeros_like(c)
    if matchup:
        phi_r, pair_r = leaf_phi(pk_r, pf_r, act_r)
        phi_l, pair_l = leaf_phi(pk_l, pf_l, act_l)
        d_mu = net.matchup_value(u1, u2, w, phi_l, pair_l) - net.matchup_value(u1, u2, w, phi_r, pair_r)
    if not value:   # no root value: the leaf's lin plus the matchup change
        c = torch.zeros_like(c)
    return net.lin(x_r) + c + d_lin + d_mu, c, d_lin, d_mu


def bce(logit, y):
    return F.binary_cross_entropy_with_logits(logit, y, reduction="none")


@torch.no_grad()
def validate(net, data, device, matchup, max_k, value=True, batch=2048):
    """Per-pair losses for every root and k = 1..max_k, plus the root value's own loss (k = 0)."""
    net.eval()
    out = {"loss": [], "k": [], "game": [], "c_loss": [], "lin_loss": [], "d_mu_abs": []}
    roots = np.arange(len(data.y))
    for s in range(0, len(roots), batch):
        r = roots[s:s + batch]
        ints_r, floats_r, pairs_r = data.batch(r, device)
        c, u1, u2, w = net(ints_r, floats_r, pairs_r)
        pk_r, pf_r, act_r, _, x_r = split_inputs(ints_r, floats_r)
        y_r = torch.as_tensor(data.y[r], device=device)
        lin_r = net.lin(x_r)
        c = lin_r + (c if value else 0)   # the root's value
        out["c_loss"].append(bce(c, y_r).cpu().numpy())
        out["lin_loss"].append(bce(lin_r, y_r).cpu().numpy())
        if matchup:
            phi_r, pair_r = leaf_phi(pk_r, pf_r, act_r)
            m_r = net.matchup_value(u1, u2, w, phi_r, pair_r)
        for k in range(1, max_k + 1):
            ok = data.rest[r] >= k
            if not ok.any():
                continue
            sel = torch.as_tensor(np.flatnonzero(ok), device=device)
            l = r[ok] + k
            ints_l, floats_l, _ = data.batch(l, device)
            pk_l, pf_l, act_l, _, x_l = split_inputs(ints_l, floats_l)
            logit = c[sel] + net.lin(x_l) - lin_r[sel]
            if matchup:
                phi_l, pair_l = leaf_phi(pk_l, pf_l, act_l)
                d = net.matchup_value(u1[sel], u2[sel], w[sel], phi_l, pair_l) - m_r[sel]
                logit = logit + d
                out["d_mu_abs"].append(d.abs().cpu().numpy())
            out["loss"].append(bce(logit, y_r[sel]).cpu().numpy())
            out["k"].append(np.full(len(l), k, np.int8))
            out["game"].append(data.game[r[ok]])
    net.train()
    return {key: np.concatenate(v) if v else np.zeros(0) for key, v in out.items()}


def summary(v, max_k):
    s = f"pairs {v['loss'].mean():.4f} (" + " ".join(
        f"k{k} {v['loss'][v['k'] == k].mean():.4f}" for k in range(1, max_k + 1) if (v['k'] == k).any()) + ")"
    s += f" | root value c {v['c_loss'].mean():.4f}, lin {v['lin_loss'].mean():.4f}"
    if len(v["d_mu_abs"]):
        s += f" | mean |matchup change| {v['d_mu_abs'].mean():.3f}"
    return s


def main():
    ap = argparse.ArgumentParser(description="Train the matchup weights of the leaf evaluation (step 8)")
    ap.add_argument("--data", nargs="+", default=["datasets/search_selfplay", "datasets/search_selfplay_r1"])
    ap.add_argument("--out", default="models/leaf_eval/matchup.pt")
    ap.add_argument("--lin-init", default="models/leaf_eval/lin1_86.npz")
    ap.add_argument("--no-matchup", action="store_true", help="the baseline: m = 0")
    ap.add_argument("--freeze-lin", action="store_true")
    ap.add_argument("--max-k", type=int, default=6)
    ap.add_argument("--val-mod", type=int, default=20)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lin-lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-value", action="store_true",
                    help="no root value c: logit = lin(leaf) + the matchup change (less to memorize games with)")
    ap.add_argument("--val-every", type=float, default=0.25, help="epochs; the best validation is kept")
    ap.add_argument("--no-ids", action="store_true",
                    help="zero (and freeze) the id embeddings of --ids all (models/leaf_eval/mu3_noids.pt)")
    ap.add_argument("--ids", choices=["all", "items", "none"], default="all", help="which ids get embeddings")
    ap.add_argument("--move-props", action="store_true", help="moves by their type, category, power, ...")
    ap.add_argument("--pairs", action="store_true", help="pe.pair_features as inputs")
    ap.add_argument("--arch", choices=["net", "linear"], default="net",
                    help="linear: search.matchup_eval.MatchupLinear (needs --pairs)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not has_engine_support():
        raise SystemExit("this poke-engine build has no matchup inputs: bash search/engine_patch/build.sh")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    matchup = not args.no_matchup

    train_files, val_files = split_shards(args.data, args.val_mod)
    if args.max_shards:
        train_files = train_files[:args.max_shards]
        val_files = val_files[:max(1, args.max_shards // 10)]
    tr = Data(train_files, "train", args.pairs)
    va = Data(val_files, "validation", args.pairs)

    with np.load(args.lin_init) as d:
        names = [str(s) for s in d["names"]]
        if names != list(pe.feature_names()):
            raise SystemExit(f"{args.lin_init}: its features don't match this engine build's")
        w1 = d["w1"].astype(np.float32)[0]
        mu, sigma = d["mu"], d["sigma"]
    arch = MatchupLinear if args.arch == "linear" else MatchupNet
    net = arch(d=args.d, layers=args.layers, dropout=args.dropout, ids=args.ids, move_props=args.move_props,
                     pairs=args.pairs, x_mu=mu, x_sigma=sigma,
                     lin_side=w1[:N_SIDE] - w1[N_SIDE:2 * N_SIDE], lin_cross=2 * w1[2 * N_SIDE:]).to(dev)
    if args.no_ids and args.ids == "all":
        for e in (net.sp, net.ab, net.it, net.mv):
            torch.nn.init.zeros_(e.weight)
            e.weight.requires_grad_(False)
    lin_params = [net.lin_side, net.lin_cross]
    if args.freeze_lin:
        for p in lin_params:
            p.requires_grad_(False)
    other = [p for n, p in net.named_parameters() if n not in ("lin_side", "lin_cross") and p.requires_grad]
    groups = [{"params": other, "lr": args.lr, "weight_decay": args.wd}]
    if not args.freeze_lin:
        groups.append({"params": lin_params, "lr": args.lin_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    roots = np.flatnonzero(tr.rest >= 1)
    steps_per_epoch = len(roots) // args.batch
    total = int(args.epochs * steps_per_epoch)
    warm = max(1, total // 50)
    base = [g["lr"] for g in opt.param_groups]
    print(f"Matchup evaluation ({'matchup' if matchup else 'baseline: no matchup term'}"
          f"{', lin frozen' if args.freeze_lin else ''}) | train {len(train_files)} shards, {len(tr.y):,} states, "
          f"{len(roots):,} roots | validation {len(val_files)} shards, {len(va.y):,} states | "
          f"{sum(p.numel() for p in net.parameters()):,} parameters, {total:,} steps on {dev}", flush=True)
    value = not args.no_value
    v = validate(net, va, dev, matchup, args.max_k, value)
    print(f"start: {summary(v, args.max_k)}", flush=True)
    best = (float(v["loss"].mean()), 0.0, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}, v)
    val_steps = max(1, int(args.val_every * steps_per_epoch))

    t0 = time.time()
    step, run = 0, 0.0
    while step < total:
        perm = rng.permutation(roots)
        for s in range(steps_per_epoch):
            if step >= total:
                break
            frac = step / total
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * frac))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            r = perm[s * args.batch:(s + 1) * args.batch]
            k = 1 + (rng.random(len(r)) * np.minimum(tr.rest[r], args.max_k)).astype(np.int64)
            logit, c, d_lin, d_mu = pair_logits(net, tr, r, r + k, dev, matchup, value)
            loss = bce(logit, torch.as_tensor(tr.y[r], device=dev)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run = 0.98 * run + 0.02 * float(loss.detach()) if step else float(loss.detach())
            step += 1
            if step % (200 if total > 2000 else 10) == 0:
                print(f"  step {step:,}/{total:,}: train loss {run:.4f}, {time.time() - t0:.0f} s", flush=True)
            if step % val_steps == 0 or step == total:
                v = validate(net, va, dev, matchup, args.max_k, value)
                loss_v = float(v["loss"].mean())
                if loss_v < best[0]:
                    best = (loss_v, step / steps_per_epoch,
                            {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}, v)
                print(f"epoch {step / steps_per_epoch:.2f} ({time.time() - t0:.0f} s): {summary(v, args.max_k)}"
                      f"{'  *' if best[1] == step / steps_per_epoch else ''}", flush=True)

    # save the best validation: the network, and the linear part as an evaluator .npz (search/leaf_eval.py)
    net.cpu()
    net.load_state_dict(best[2])
    v = best[3]
    print(f"best validation at epoch {best[1]:.2f}: {summary(v, args.max_k)}", flush=True)
    lin_npz = args.out[:-3] + "_lin.npz"
    meta = {"train_shards": len(train_files), "train_states": int(len(tr.y)), "val_shards": len(val_files),
            "val_pairs": float(v["loss"].mean()), "val_c": float(v["c_loss"].mean()), "matchup": matchup,
            "best_epoch": best[1],
            "args": vars(args)}
    np.savez(lin_npz + ".tmp.npz", hidden=np.int64(0), w1=net.lin_export(), b1=np.zeros(0, np.float32),
             w2=np.zeros(0, np.float32), names=np.array(names), mu=mu, sigma=sigma, meta=np.array(json.dumps(meta)))
    os.replace(lin_npz + ".tmp.npz", lin_npz)
    torch.save({"config": net.config, "state_dict": net.state_dict(), "matchup": matchup, "lin_npz": lin_npz,
                "layout": [N_POKE_INTS, N_POKE_FLOATS, N_FIELD, M_LEAF, N_X, N_ROOT_WEIGHTS], "meta": meta},
               args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)
    np.savez(args.out[:-3] + "_val.npz", loss=v["loss"], k=v["k"], game=v["game"])
    print(f"saved {args.out} and {lin_npz}")

    # the engine must give the same v(leaf) - v(root) as the network (inputs recomputed in float32)
    from search.leaf_eval import evaluator_id
    states = read_states(val_files[0])[:400]
    eid = evaluator_id(lin_npz, 2, 1.0)
    ints, floats = pe.matchup_inputs_from_strings(states)
    ints = torch.as_tensor(np.asarray(ints, np.int64).reshape(len(states), N_INTS))
    floats = torch.as_tensor(np.asarray(floats, np.float32).reshape(len(states), N_FLOATS))
    pairs = torch.as_tensor(np.asarray(pe.pair_features_from_strings(states), np.float32).reshape(len(states), -1)) \
        if args.pairs else None
    with torch.no_grad():
        c, u1, u2, w = net.eval()(ints, floats, pairs)
        rw = net.root_weight_vector(u1, u2, w)
        pk, pf, act, _, x = split_inputs(ints, floats)
        phi, pair = leaf_phi(pk, pf, act)
        ours = net.lin(x)
        if matchup:
            ours = ours + net.matchup_value(u1, u2, w, phi, pair)
    worst = 0.0
    for i in range(0, len(states), 40):   # each state with its own root weights
        rust = pe.evaluator_values(eid, [states[i]], rw[i].tolist() if matchup else None)[0]
        worst = max(worst, abs(rust - float(ours[i])))
    print(f"engine vs network on {len(range(0, len(states), 40))} states: max |diff| {worst:.2e}")


if __name__ == "__main__":
    main()
