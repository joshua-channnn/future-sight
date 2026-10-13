"""How often a leaf evaluation changes the search's pick (agent_docs/step8.md section 5): stored
decisions (true states, held-out shards of a self-play round) searched with each evaluation, the
share where the most visited option differs from the first evaluation's, against a second run of
the first (the search's own noise). An evaluation that changes few picks beyond the noise can't
be worth much in play (TreeStrap's network: 8.6% against 6.4%; the hand-written evaluation, which
lin1_86 beat by 3.6 points: 21.0%; averages of two runs, one run of 210 moves by ~2 points).

    python3 -m tools.compare_picks models/leaf_eval/lin1_86.npz models/leaf_eval/ts_r0_mlp16.npz hand
"""
import argparse
import glob
import gzip
import os
import zlib

import numpy as np
import poke_engine as pe

from search.leaf_eval import evaluator_id


def main():
    ap = argparse.ArgumentParser(description="How often leaf evaluations change the search's pick")
    ap.add_argument("evals", nargs="+", help=".npz leaf evaluations, or 'hand' (the engine's own); the first is the reference")
    ap.add_argument("--data", default="datasets/search_selfplay_r1")
    ap.add_argument("--shards", type=int, default=3, help="held-out shards to draw from")
    ap.add_argument("--per-shard", type=int, default=70)
    ap.add_argument("--iterations", type=int, default=56500)
    ap.add_argument("--scale", type=float, default=1.4)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    val = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % 10 == 0][:args.shards]
    rng = np.random.default_rng(1)
    states = []
    for f in val:
        with np.load(f) as d:
            rows = np.flatnonzero(d["searched"])
        with gzip.open(f[:-4] + ".states.gz", "rt") as fh:
            src = fh.read().splitlines()
        states += [src[i] for i in rng.choice(rows, min(args.per_shard, len(rows)), replace=False)]
    ids = [0 if e == "hand" else evaluator_id(e, 2, args.scale) for e in args.evals]

    def picks(eid):
        out = []
        for s in states:
            r = pe.MctsResult._from_rust(pe.mcts_with(pe.State.from_string(s), 0, args.iterations, eid))
            visits = [o.visits for o in r.side_one]
            out.append((r.side_one[int(np.argmax(visits))].move_choice, max(visits) / sum(visits)))
        return out

    ref = picks(ids[0])
    print(f"{len(states)} decisions from {len(val)} held-out shards of {args.data}, {args.iterations} iterations")
    for name, eid, label in [(args.evals[0], ids[0], "a second run")] + [(e, i, e) for e, i in zip(args.evals[1:], ids[1:])]:
        p = picks(eid)
        changed = np.mean([a[0] != b[0] for a, b in zip(ref, p)])
        print(f"{label:<44} pick differs from {os.path.basename(args.evals[0])}'s in {changed:5.1%} "
              f"(top option's share {np.mean([x[1] for x in p]):.3f})")


if __name__ == "__main__":
    main()
