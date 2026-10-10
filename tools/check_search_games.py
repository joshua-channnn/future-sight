"""Checks and statistics for search self-play shards (training/gen_search_games.py).

    python3 -m tools.check_search_games datasets/search_selfplay [--states 2000]

Per shard: shapes and dtypes; finite features; targets >= 0, zero on illegal actions, summing to 1;
one-hot on the action played when not searched; the action played is legal; outcomes constant per
(game, side), opposite between the sides and equal to the game table's winner; no labels in Illusion
games, next-action labels in range. States (a sample of --states lines per shard, all with 0): each
line round-trips through poke_engine.State.from_string, and when both sides decided at the same step
the two states are mirror images (side one of one = side two of the other). Then dataset statistics:
decisions per game, searched share, legal actions, target entropy, the action played vs the most
visited, the search's value vs the outcome, p1's win rate, bytes per sample.
"""
import argparse
import glob
import gzip
import os
from collections import Counter

import numpy as np

SHAPES = {"idx": ((188,), np.int16), "flt": ((1755,), np.float16), "mask": ((14,), np.bool_),
          "target": ((14,), np.float32), "searched": ((), np.bool_), "action": ((), np.int8),
          "value": ((), np.float32), "q": ((14,), np.float32), "iterations": ((), np.int32), "samples": ((), np.int8),
          "lab": ((43,), np.int16), "outcome": ((), np.int8), "game": ((), np.int32),
          "side": ((), np.int8), "turn": ((), np.int16), "step": ((), np.int16)}
LAB_NEXT, N_NEXT = 42, 2 * 354 + 6


def load_states(path):
    with gzip.open(path, "rt") as f:
        return f.read().split("\n")[:-1]


def normalize(state):
    """poke-engine writes each volatile status followed by ":" and reads the empty item after the
    last one as a NONE volatile (Side::deserialize), so a round trip adds NONE to every non-empty set:
    compare the sets without NONE."""
    parts = state.split("/")
    for i in (0, 1):
        f = parts[i].split("=")
        f[8] = ":".join(sorted(x for x in f[8].split(":") if x and x != "NONE"))
        parts[i] = "=".join(f)
    return "/".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+", help="shard directories or .npz files")
    ap.add_argument("--states", type=int, default=2000, help="state lines to round-trip per shard (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import poke_engine as pe
    rng = np.random.default_rng(args.seed)
    files = []
    for p in args.path:
        files += sorted(glob.glob(os.path.join(p, "*.npz"))) if os.path.isdir(p) else [p]
    if not files:
        raise SystemExit("no shards found")
    c = Counter()
    bad = Counter()
    examples = {}
    agg = {k: [] for k in ("searched", "n_legal", "entropy", "top", "played_top", "value", "outcome", "side",
                           "iterations", "turn", "target_played")}
    games = Counter()
    p1 = Counter()
    bytes_npz = bytes_states = 0

    def fail(name, detail):
        bad[name] += 1
        examples.setdefault(name, str(detail)[:300])

    for path in files:
        d = np.load(path)
        n = len(d["action"])
        c["shards"] += 1
        c["samples"] += n
        bytes_npz += os.path.getsize(path)
        sp = path[:-4] + ".states.gz"
        bytes_states += os.path.getsize(sp) if os.path.exists(sp) else 0
        for k, (shape, dt) in SHAPES.items():
            if k not in d or d[k].shape != (n,) + shape or d[k].dtype != dt:
                fail(f"shape/dtype {k}", (d[k].shape, d[k].dtype) if k in d else "missing")
        mask, target, act = d["mask"], d["target"], d["action"].astype(int)
        searched = d["searched"]
        if not np.isfinite(d["flt"].astype(np.float32)).all():
            fail("non-finite features", np.argwhere(~np.isfinite(d["flt"].astype(np.float32)))[:3])
        if (target < 0).any():
            fail("negative target", np.flatnonzero((target < 0).any(1))[:5])
        if (target[~mask] != 0).any():
            fail("target on an illegal action", np.flatnonzero((target * ~mask).sum(1) > 0)[:5])
        s = target.sum(1)
        if np.abs(s - 1).max() > 1e-4:
            fail("target doesn't sum to 1", s[np.abs(s - 1) > 1e-4][:5])
        if not mask[np.arange(n), act].all():
            fail("illegal action played", np.flatnonzero(~mask[np.arange(n), act])[:5])
        ns = ~searched
        if ns.any() and not (target[ns, act[ns]] == 1).all():
            fail("non-searched target isn't one-hot on the action", np.flatnonzero(ns)[:5])
        if np.isfinite(d["value"][ns]).any() or not np.isfinite(d["value"][searched]).all():
            fail("value NaN iff not searched", "")
        if ((d["value"][searched] < -1e-6) | (d["value"][searched] > 1 + 1e-6)).any():
            fail("value outside 0..1", d["value"][searched].min())
        q = d["q"]
        vis = np.isfinite(q[searched]) != (target[searched] > 0)
        if vis.any() or np.isfinite(q[~searched]).any():
            fail("q finite iff the action was visited (NaN when not searched)", np.flatnonzero(vis.any(1))[:5])
        qv = np.where(np.isfinite(q), q, 0.0)
        if ((qv < -1e-6) | (qv > 1 + 1e-6)).any():
            fail("q outside 0..1", qv.min())
        # the root value is the visit-weighted mean of the actions' q (up to dropped illegal visits)
        vq = (target * qv).sum(1)
        c["value vs visit-weighted q: max abs diff (searched)"] = max(
            c["value vs visit-weighted q: max abs diff (searched)"],
            float(np.abs(vq - d["value"])[searched].max()) if searched.any() else 0.0)
        # outcomes
        g, side, out = d["game"], d["side"], d["outcome"].astype(int)
        winner = d["g_winner"].astype(int)
        for gi in range(len(winner)):
            for sd in (1, 2):
                o = out[(g == gi) & (side == sd)]
                if len(o) == 0:
                    fail("a side without samples", (path, gi, sd))
                    continue
                if (o != o[0]).any():
                    fail("outcome not constant per (game, side)", (path, gi, sd))
                want = 0 if winner[gi] == 0 else (1 if winner[gi] == sd else -1)
                if o[0] != want:
                    fail("outcome != game table's winner", (path, gi, sd, o[0], winner[gi]))
            n1, n2 = ((g == gi) & (side == 1)).sum(), ((g == gi) & (side == 2)).sum()
            if n1 != d["g_n1"][gi] or n2 != d["g_n2"][gi]:
                fail("sample counts != game table", (path, gi, n1, n2, d["g_n1"][gi], d["g_n2"][gi]))
            games["games"] += 1
            games["illusion"] += bool(d["g_illusion"][gi])
            p1[winner[gi]] += 1
        # labels
        lab = d["lab"].astype(int)
        ill = d["g_illusion"][g]
        if (lab[ill] != -1).any():
            fail("labels in an Illusion game", np.flatnonzero((lab[ill] != -1).any(1))[:5])
        nxt = lab[:, LAB_NEXT]
        if ((nxt < -1) | (nxt > N_NEXT)).any():
            fail("next-action label out of range", nxt.min())
        c["next-action labels"] += int((nxt >= 0).sum())
        c["item labels"] += int((lab[:, 0:6] >= 0).sum())
        # states
        if os.path.exists(sp):
            lines = load_states(sp)
            if len(lines) != n:
                fail("state lines != samples", (path, len(lines), n))
            else:
                have = np.array([bool(x) for x in lines])
                c["states stored"] += int(have.sum())
                idxs = np.flatnonzero(have)
                if args.states and len(idxs) > args.states:
                    idxs = np.sort(rng.choice(idxs, args.states, replace=False))
                for i in idxs:
                    try:
                        back = pe.State.from_string(lines[i]).to_string()
                        exact = back == lines[i]
                        ok = exact or normalize(back) == normalize(lines[i])
                    except BaseException as e:  # noqa: BLE001
                        exact = ok = False
                        examples.setdefault("state round trip", f"{type(e).__name__}: {e}")
                    c["states round-tripped"] += 1
                    c["states round-tripped exactly (else only with the NONE volatile added)"] += exact
                    if not ok:
                        fail("state round trip", (path, i))
                # mirror images: both sides decided at the same step (side 2's sample follows side 1's)
                st_ = d["step"]
                for i in range(n - 1):
                    if g[i] == g[i + 1] and side[i] == 1 and side[i + 1] == 2 and st_[i] == st_[i + 1] \
                            and have[i] and have[i + 1]:
                        a, b = lines[i].split("/"), lines[i + 1].split("/")
                        c["mirror pairs"] += 1
                        if not (a[0] == b[1] and a[1] == b[0] and a[2:] == b[2:]):
                            fail("states of the same step aren't mirror images", (path, i, d["turn"][i]))
        # statistics
        agg["searched"].append(searched)
        agg["n_legal"].append(mask.sum(1))
        t = np.clip(target, 1e-12, 1)
        agg["entropy"].append(-(target * np.log(t)).sum(1))
        agg["top"].append(target.argmax(1))
        agg["played_top"].append(act == target.argmax(1))
        agg["target_played"].append(target[np.arange(n), act])
        agg["value"].append(d["value"])
        agg["outcome"].append(out)
        agg["side"].append(side)
        agg["iterations"].append(d["iterations"])
        agg["turn"].append(d["turn"])

    A = {k: np.concatenate(v) for k, v in agg.items()}
    srch = A["searched"]
    print(f"{c['shards']} shards, {games['games']:,} games ({games['illusion']} with Illusion), {c['samples']:,} samples "
          f"({c['samples'] / max(games['games'], 1):.1f} per game), {srch.mean():.1%} searched")
    print(f"  size: npz {bytes_npz / 1e6:.1f} MB + states {bytes_states / 1e6:.1f} MB = "
          f"{(bytes_npz + bytes_states) / max(c['samples'], 1) / 1024:.2f} KB/sample "
          f"(npz {bytes_npz / max(c['samples'], 1) / 1024:.2f}, states {bytes_states / max(c['states stored'], 1) / 1024:.2f} "
          f"per stored state)")
    decided = p1[1] + p1[2]
    pr = p1[1] / max(decided, 1)
    print(f"  p1 wins {pr:.1%} ± {1.96 * np.sqrt(pr * (1 - pr) / max(decided, 1)):.1%} of {decided} decided games, "
          f"{p1[0]} ties")
    nl = A["n_legal"][srch]
    print(f"  searched decisions: legal actions mean {nl.mean():.2f} (" +
          ", ".join(f"{k}: {v / len(nl):.0%}" for k, v in sorted(Counter(nl.tolist()).items())) + ")")
    print(f"  target entropy mean {A['entropy'][srch].mean():.3f} nats")
    print(f"  played the most visited action {A['played_top'][srch].mean():.1%}; mean share of the action played "
          f"{A['target_played'][srch].mean():.3f}")
    v, o = A["value"][srch], A["outcome"][srch]
    dec = o != 0
    if dec.any():
        print(f"  search value: mean {v[dec & (o == 1)].mean():.3f} for the eventual winner, "
              f"{v[dec & (o == -1)].mean():.3f} for the loser; corr(value, outcome) {np.corrcoef(v[dec], o[dec])[0, 1]:.3f}; "
              f"sign agreement {((v[dec] > 0.5) == (o[dec] > 0)).mean():.1%}")
    print(f"  iterations per searched decision: median {np.median(A['iterations'][srch]):,.0f}; turns per sample median "
          f"{np.median(A['turn']):.0f}")
    print(f"  labels: {c['item labels']:,} item, {c['next-action labels']:,} next-action; root value vs visit-weighted "
          f"q: max abs diff {c['value vs visit-weighted q: max abs diff (searched)']:.4f}")
    print(f"  states: {c['states stored']:,} stored, {c['states round-tripped']:,} round-tripped "
          f"({c['states round-tripped exactly (else only with the NONE volatile added)']:,} exactly, the rest with "
          f"poke-engine's NONE volatile added), {c['mirror pairs']:,} mirror pairs checked")
    if bad:
        print("\nPROBLEMS:")
        for k, v in bad.items():
            print(f"  {k}: {v} (e.g. {examples.get(k)})")
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
