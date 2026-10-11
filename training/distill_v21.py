"""Distillation (agent_docs/step7_brief.md, step 2): train v21 from its 20M weights on the search
self-play data of training/gen_search_games.py: policy cross-entropy to the search's visit shares,
value to the game result, the aux heads on the labels (v21's aux losses, PPO's coefficients). Only
searched decisions are used. Whole games are held out: every shard whose name hashes to 0 mod
--val-mod (5% by default) is validation.

The data doesn't fit in memory (~4 KB per sample), so each epoch streams the training shards in
random order, --chunk shards at a time shuffled together (the next chunk loads in a background
thread). Every minibatch has --batch samples (MPS compiles a graph per shape).

--target mix / qadv (policy_target) train toward the visit shares mixed with the starting network's
policy, or toward the starting policy reweighted by the search's per-action values: conservative
variants, for when plain imitation makes the network play worse (agent_docs/step7.md).

Validation metrics: policy cross-entropy and KL to the visit shares, top-1 agreement with the most
visited action, the greedy pick's regret by the search's own per-action values (q of the best
visited action minus q of the pick; an unvisited pick counts 0), value MSE against the outcome and
its sign accuracy, aux losses and accuracies; the
starting weights are scored first, as the baseline. Checkpoints are v21 checkpoints (V20Policy.load,
eval/ppo_test.py, search.net.NetModel load them): <out>_best.pt (lowest validation loss), <out>.pt
(the last), <out>_epoch<k>.pt.

    python3 -m training.distill_v21 --data datasets/search_selfplay --epochs 4 --out models/v21_distill
"""
import argparse
import glob
import math
import os
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from networks.v21_policy import AUX_HEADS, aux_obs, aux_terms, build_from_checkpoint

FIELDS = ("idx", "flt", "mask", "target", "outcome", "lab", "value", "action", "q")
AUX_COEFS = {"next": 0.1, "item": 0.05, "ability": 0.05, "tera": 0.05, "moves": 0.05}   # as PPO's CONFIG


def split_shards(paths, val_mod):
    """Shards of the directories `paths` (a list, or one path): (train, validation)."""
    paths = [paths] if isinstance(paths, str) else paths
    files = sorted(f for p in paths for f in glob.glob(os.path.join(p, "*.npz")))
    val = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % val_mod == 0]
    return [f for f in files if f not in set(val)], val


def load_shard(path):
    with np.load(path) as d:
        keep = d["searched"]
        return {k: d[k][keep] for k in FIELDS}


def load_many(paths):
    parts = [load_shard(p) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


def batches(data, batch, rng=None):
    n = len(data["action"])
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for k in range(n // batch):
        sel = order[k * batch:(k + 1) * batch]
        yield {f: data[f][sel] for f in data}


def stream(files, chunk, batch, rng):
    """One epoch of training minibatches: shards in random order, `chunk` at a time."""
    order = [files[i] for i in rng.permutation(len(files))]
    groups = [order[i:i + chunk] for i in range(0, len(order), chunk)]
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(load_many, groups[0])
        for gi in range(len(groups)):
            data = fut.result()
            if gi + 1 < len(groups):
                fut = ex.submit(load_many, groups[gi + 1])
            yield from batches(data, batch, rng)
            del data


def to_device(b, device):
    return (torch.as_tensor(b["idx"]).to(device).long(), torch.as_tensor(b["flt"]).to(device).float(),
            torch.as_tensor(b["mask"]).to(device), torch.as_tensor(b["target"]).to(device),
            torch.as_tensor(b["outcome"]).to(device).float(), torch.as_tensor(b["lab"]).to(device).long(),
            torch.as_tensor(b["value"]).to(device), torch.as_tensor(b["action"]).to(device).long(),
            torch.as_tensor(b["q"]).to(device))


@torch.no_grad()
def policy_target(b, cfg, ref):
    """The policy target of a minibatch: the visit shares ("shares"); their mix with the starting
    network's policy ("mix": (1 - mix) shares + mix ref); or the starting policy reweighted by the
    search's per-action values ("qadv": ref * exp(beta * (q - max q)), unvisited actions at the
    lowest visited q), a KL-regularized improvement step on the starting network."""
    idx, flt, mask, target, _, _, _, _, q = b
    if cfg["target"] == "shares":
        return target
    pref = F.softmax(ref(idx, flt, mask)[0], -1)
    if cfg["target"] == "mix":
        return (1 - cfg["mix"]) * target + cfg["mix"] * pref
    visited = ~torch.isnan(q)
    qmax = torch.where(visited, q, torch.full_like(q, -1.0)).max(-1, keepdim=True).values
    qmin = torch.where(visited, q, torch.full_like(q, 2.0)).min(-1, keepdim=True).values
    qf = torch.where(visited, q, qmin.expand_as(q))
    w = pref * torch.exp(cfg["beta"] * (qf - qmax)) * mask
    return w / w.sum(-1, keepdim=True).clamp(min=1e-12)


def compute(model, b, cfg, metrics=False, ref=None):
    """Loss terms (and with metrics, sums for the validation report) for one minibatch. The policy
    loss is to policy_target (needs ref, the starting network, unless --target shares); the
    validation metrics compare with the visit shares."""
    idx, flt, mask, target, z, lab, sv, act, q = b
    logits, value, aux = model(idx, flt, mask, aux=True)
    pt = target if ref is None else policy_target(b, cfg, ref)
    logp = F.log_softmax(logits, -1)
    ce_t = -(pt * logp).sum(-1)
    ce = ce_t if pt is target else -(target * logp).sum(-1)
    if cfg["value_target"] == "mix":   # the outcome and the search's root value (0..1 -> -1..1)
        vt = 0.5 * z + 0.5 * (2 * sv - 1)
    else:
        vt = z
    v_loss = (value - vt) ** 2
    obs = aux_obs(idx, flt, model.layout)
    al, mets = aux_terms(aux, lab, obs, metrics=metrics)
    loss = ce_t.mean() + cfg["value_coef"] * v_loss.mean() + sum(cfg["aux_coefs"][h] * al[h] for h in AUX_HEADS)
    out = {"loss": loss, "policy_ce": ce.mean(), "value_mse": v_loss.mean(), **{f"aux_{h}": al[h] for h in AUX_HEADS}}
    if metrics:
        with torch.no_grad():
            ent = -(target * torch.log(target.clamp(min=1e-12))).sum(-1)
            top_t = target.argmax(-1)
            top_p = logits.argmax(-1)
            p = logp.exp()
            dec = z != 0
            s = {"n": torch.tensor(float(len(z)), device=z.device),
                 "policy_ce": ce.sum(), "kl": (ce - ent).sum(),
                 "top1": (top_p == top_t).float().sum(),
                 "p_top": p.gather(-1, top_t[:, None]).squeeze(-1).sum(),
                 "p_played": p.gather(-1, act[:, None]).squeeze(-1).sum(),
                 "value_mse_z": ((value - z) ** 2).sum(),
                 "value_sign": ((value > 0) == (z > 0)).float().mul(dec).sum(), "n_dec": dec.float().sum(),
                 "search_value_sign": ((sv > 0.5) == (z > 0)).float().mul(dec).sum()}
            # regret of the greedy pick by the search's own per-action values
            qa = q.gather(-1, top_p[:, None]).squeeze(-1)
            qbest = torch.where(torch.isnan(q), torch.full_like(q, -1.0), q).max(-1).values
            reg = torch.where(torch.isnan(qa), torch.zeros_like(qa), qbest - qa)
            s["regret"] = reg.sum()
            s["regret_big"] = (reg > 0.05).float().sum()
            for h in AUX_HEADS:
                s[f"aux_{h}"] = al[h] * len(z)
            for name, (c, n) in mets.items():
                s[f"acc_{name}"], s[f"accn_{name}"] = c.float(), n.float()
        out["sums"] = s
    return out


@torch.no_grad()
def validate(model, val, cfg, device, batch):
    model.eval()
    tot = defaultdict(float)
    for b in batches(val, batch):
        s = compute(model, to_device(b, device), cfg, metrics=True)["sums"]
        for k, v in s.items():
            tot[k] += float(v)
    model.train()
    n = max(tot["n"], 1)
    r = {k: tot[k] / n for k in ("policy_ce", "kl", "top1", "p_top", "p_played", "value_mse_z", "regret", "regret_big")}
    r["value_sign"] = tot["value_sign"] / max(tot["n_dec"], 1)
    r["search_value_sign"] = tot["search_value_sign"] / max(tot["n_dec"], 1)
    for h in AUX_HEADS:
        r[f"aux_{h}"] = tot[f"aux_{h}"] / n
    for k in list(tot):
        if k.startswith("acc_"):
            r[k] = tot[k] / max(tot["accn_" + k[4:]], 1)
    r["loss"] = r["policy_ce"] + cfg["value_coef"] * r["value_mse_z"] + sum(cfg["aux_coefs"][h] * r[f"aux_{h}"]
                                                                             for h in AUX_HEADS)
    return r


def fmt(r):
    return (f"loss {r['loss']:.4f} | policy CE {r['policy_ce']:.4f} KL {r['kl']:.4f} top-1 {r['top1']:.1%} "
            f"p(top) {r['p_top']:.3f} p(played) {r['p_played']:.3f} regret {r['regret']:.4f} "
            f"(>0.05 {r['regret_big']:.1%}) | value MSE {r['value_mse_z']:.4f} sign "
            f"{r['value_sign']:.1%} (search's root value {r['search_value_sign']:.1%}) | aux "
            + " ".join(f"{h} {r[f'aux_{h}']:.3f}" for h in AUX_HEADS)
            + f" | acc next {r.get('acc_next', float('nan')):.3f} item {r.get('acc_item_unrev', float('nan')):.3f} "
            f"moves {r.get('acc_moves_unrev', float('nan')):.3f} (unrevealed)")


def save(model, path, extra):
    from training.ppo_v20 import model_state
    state = model_state(model, "v21", extra)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def wait_for_power():
    from training.gen_search_games import on_ac_power
    if on_ac_power():
        return
    print(f"[{datetime.now():%m-%d %H:%M}] on battery: waiting for AC power", flush=True)
    while not on_ac_power():
        time.sleep(60)
    print(f"[{datetime.now():%m-%d %H:%M}] on AC power: continuing", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Distill the search into v21 (agent_docs/step7_brief.md, step 2)")
    ap.add_argument("--data", nargs="+", default=["datasets/search_selfplay"], help="shard directories")
    ap.add_argument("--init", default="models/ppo_pokemon_6v6_v21.pt")
    ap.add_argument("--out", default="models/v21_distill", help="checkpoint path prefix")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-final", type=float, default=0.1, help="cosine decay to this fraction of --lr")
    ap.add_argument("--warmup", type=float, default=0.02, help="fraction of the steps")
    ap.add_argument("--embed-lr-mult", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--value-target", choices=["outcome", "mix"], default="outcome")
    ap.add_argument("--target", choices=["shares", "mix", "qadv"], default="shares", help="policy target (policy_target)")
    ap.add_argument("--mix", type=float, default=0.5, help="--target mix: weight of the starting policy")
    ap.add_argument("--beta", type=float, default=30.0, help="--target qadv: inverse temperature on q")
    ap.add_argument("--aux-scale", type=float, default=1.0, help="multiplies PPO's aux coefficients")
    ap.add_argument("--val-mod", type=int, default=20)
    ap.add_argument("--val-max", type=int, default=150_000, help="validation samples (whole shards)")
    ap.add_argument("--max-train-shards", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=24, help="shards shuffled together")
    ap.add_argument("--val-every", type=int, default=1000, help="steps")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads (leave the cores to the generator)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto"
                          else args.device)
    train, val_files = split_shards(args.data, args.val_mod)
    if args.max_train_shards:
        train = [train[i] for i in sorted(rng.choice(len(train), min(args.max_train_shards, len(train)), replace=False))]
    val_sel, n_val = [], 0
    for f in val_files:
        if n_val >= args.val_max:
            break
        val_sel.append(f)
        with np.load(f) as d:
            n_val += int(d["searched"].sum())
    val = load_many(val_sel)
    n_train = 0
    for f in train:
        with np.load(f) as d:
            n_train += int(d["searched"].sum())
    steps_per_epoch = max(1, n_train // args.batch)   # each chunk drops its last partial batch: epochs are approximate
    total = int(args.epochs * steps_per_epoch)
    cfg = {"value_coef": args.value_coef, "value_target": args.value_target, "target": args.target,
           "mix": args.mix, "beta": args.beta, "aux_coefs": {h: c * args.aux_scale for h, c in AUX_COEFS.items()}}

    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    model = build_from_checkpoint(ckpt).to(device)
    model.train()
    ref = None
    if args.target != "shares":   # the starting network, frozen
        ref = build_from_checkpoint(ckpt).to(device).eval()
        for prm in ref.parameters():
            prm.requires_grad_(False)
    emb = model.embedding_parameters()
    groups = [{"params": emb, "lr_mult": args.embed_lr_mult}, {"params": model.other_parameters(), "lr_mult": 1.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    print(f"Distill v21 from {args.init} | train {len(train)} shards, {n_train:,} searched samples | validation "
          f"{len(val_sel)} shards, {len(val['action']):,} samples | {args.epochs} epochs x {steps_per_epoch:,} steps of "
          f"{args.batch} | lr {args.lr} (embeddings x{args.embed_lr_mult}) warmup {args.warmup} cosine to "
          f"{args.lr_final} | wd {args.wd} | target {args.target}"
          f"{f' {args.mix}' if args.target == 'mix' else (f' beta {args.beta}' if args.target == 'qadv' else '')} | "
          f"value coef {args.value_coef} ({args.value_target}) | aux "
          f"{cfg['aux_coefs']} | {device}", flush=True)

    def lr_at(step):
        w = max(1, int(args.warmup * total))
        if step < w:
            return args.lr * (step + 1) / w
        x = (step - w) / max(1, total - w)
        return args.lr * (args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * min(x, 1.0))))

    t0 = time.time()
    r = validate(model, val, cfg, device, args.batch)
    print(f"[val @ 0 (the starting weights)] {fmt(r)} | {time.time() - t0:.0f} s", flush=True)
    best = r["loss"]
    history = [{"step": 0, **r}]
    step, epoch = 0, 0
    run = defaultdict(float)
    n_run = 0
    t_log = time.time()
    done = False
    while not done:
        for b in stream(train, args.chunk, args.batch, rng):
            if step % 100 == 0:
                wait_for_power()
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]
            out = compute(model, to_device(b, device), cfg, ref=ref)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            for k in ("loss", "policy_ce", "value_mse", *(f"aux_{h}" for h in AUX_HEADS)):
                run[k] += float(out[k].detach()) if n_run % 10 == 0 else 0.0
            run["gn"] += float(gn) if n_run % 10 == 0 else 0.0
            n_run += 1
            step += 1
            if step % args.log_every == 0:
                m = max(1, (n_run + 9) // 10)
                el = time.time() - t_log
                print(f"step {step:,}/{total:,} (epoch {step / steps_per_epoch:.2f}) | lr {lr:.2e} | train loss "
                      f"{run['loss'] / m:.4f} policy CE {run['policy_ce'] / m:.4f} value {run['value_mse'] / m:.4f} "
                      f"aux next {run['aux_next'] / m:.3f} | grad {run['gn'] / m:.2f} | "
                      f"{n_run * args.batch / max(el, 1e-6):,.0f} samples/s", flush=True)
                run.clear()
                n_run = 0
                t_log = time.time()
            if step % args.val_every == 0 or step >= total:
                t1 = time.time()
                r = validate(model, val, cfg, device, args.batch)
                history.append({"step": step, **r})
                tag = ""
                extra = {"step": step, "distill": vars(args), "val": r, "history": history}
                if r["loss"] < best:
                    best = r["loss"]
                    save(model, args.out + "_best.pt", extra)
                    tag = " (best)"
                save(model, args.out + ".pt", extra)
                print(f"[val @ {step:,}, epoch {step / steps_per_epoch:.2f}] {fmt(r)}{tag} | {time.time() - t1:.0f} s",
                      flush=True)
                t_log = time.time()
            if step >= total:
                done = True
                break
        epoch += 1
        if not done:
            save(model, f"{args.out}_epoch{epoch}.pt", {"step": step, "distill": vars(args), "history": history})
    print(f"done: {step:,} steps in {(time.time() - t0) / 60:.1f} min; best validation loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
