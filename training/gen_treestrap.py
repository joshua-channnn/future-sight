"""TreeStrap data (agent_docs/step8.md section 5): the stored self-play decisions searched again
(pe.mcts_tree: mcts_with with the learned evaluation, then every node with at least --min-visits
visits), keeping per search the root and up to --max-nodes of the other nodes: their engine
features, visits, depth and side one's mean score q over the iterations through them (0..1; in
mode 2 a leaf scores sigmoid(scale * (v(leaf) - v(root)))). training/train_treestrap.py fits the
evaluation toward these values, so it learns where the search uses it, a few turns from the root,
instead of from the game results or the root's options only.

The stored state is the true one (both teams known) with the decider as side one; decisions with
one legal action (not searched) are skipped. One output shard per input shard
(<dataset>__<shard>.npz), written once: a rerun continues where it stopped.

    python3 -m training.gen_treestrap --out datasets/treestrap_r0
"""
import argparse
import gzip
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np
import poke_engine as pe

from search.leaf_eval import evaluator_id
from training.train_q_eval import split_shards


def read_states(path):   # as training/train_leaf_eval.py (which imports torch)
    with gzip.open(path[:-4] + ".states.gz", "rt") as f:
        return f.read().splitlines()


def out_name(path):
    return os.path.basename(os.path.dirname(path)) + "__" + os.path.basename(path)


def one_shard(job):
    path, out_dir, a = job
    out_path = os.path.join(out_dir, out_name(path))
    if os.path.exists(out_path):
        return out_path, 0, 0.0
    t0 = time.time()
    eid = evaluator_id(a["leaf_eval"], 2, a["scale"])
    rng = np.random.default_rng([a["seed"], zlib.crc32(out_name(path).encode())])
    with np.load(path) as d:
        searched, outcome, game, turn = d["searched"], d["outcome"], d["game"], d["turn"]
    states = read_states(path)
    rows = np.flatnonzero(searched)[::a["every"]]
    parts = {k: [] for k in ("x", "q", "visits", "depth", "over", "tree")}
    kept_rows, n_nodes = [], []
    for row in rows:
        st = pe.State.from_string(states[row])
        r, feats, depth, _, visits, q, over = pe.mcts_tree(st, 0, a["iterations"], eid, a["min_visits"])
        n = len(q)
        if n < 2:
            continue
        keep = np.arange(n)
        if n - 1 > a["max_nodes"]:   # the root, then a uniform sample of the others
            keep = np.r_[0, 1 + np.sort(rng.choice(n - 1, a["max_nodes"], replace=False))]
        x = np.frombuffer(feats, np.float32).reshape(n, -1)[keep]
        parts["x"].append(x.astype(np.float16))
        parts["q"].append(np.asarray(q, np.float32)[keep])
        parts["visits"].append(np.asarray(visits, np.int32)[keep])
        parts["depth"].append(np.asarray(depth, np.int16)[keep])
        parts["over"].append(np.asarray(over, bool)[keep])
        parts["tree"].append(np.full(len(keep), len(kept_rows), np.int32))
        kept_rows.append(row)
        n_nodes.append(n)
    kept_rows = np.asarray(kept_rows, np.int64)
    np.savez(out_path + ".tmp.npz", **{k: np.concatenate(v) for k, v in parts.items()},
             row=kept_rows, n_nodes=np.asarray(n_nodes, np.int32), outcome=outcome[kept_rows],
             game=game[kept_rows], turn=turn[kept_rows])
    os.replace(out_path + ".tmp.npz", out_path)
    return out_path, len(kept_rows), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="TreeStrap data: stored decisions searched again, their trees' nodes")
    ap.add_argument("--data", nargs="+", default=["datasets/search_selfplay", "datasets/search_selfplay_r1"])
    ap.add_argument("--out", default="datasets/treestrap_r0")
    ap.add_argument("--leaf-eval", default="models/leaf_eval/lin1_86.npz")
    ap.add_argument("--scale", type=float, default=1.4)
    ap.add_argument("--iterations", type=int, default=56500, help="per search (the A/Bs: 565 per ms x 100 ms)")
    ap.add_argument("--every", type=int, default=10, help="every k-th searched decision")
    ap.add_argument("--min-visits", type=int, default=25)
    ap.add_argument("--max-nodes", type=int, default=32, help="nodes kept per search besides the root")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train, val = split_shards(args.data, 10)
    files = val + train   # the validation shards first, so a short run has some
    if args.max_shards:
        files = files[:args.max_shards]
    a = {k: getattr(args, k) for k in ("leaf_eval", "scale", "iterations", "every", "min_visits", "max_nodes", "seed")}
    t0, done, trees = time.time(), 0, 0
    with Pool(args.workers) as pool:
        for out_path, n, dt in pool.imap_unordered(one_shard, [(f, args.out, a) for f in files]):
            done += 1
            trees += n
            print(f"{done}/{len(files)} shards, {trees:,} searches, {time.time() - t0:.0f} s "
                  f"({os.path.basename(out_path)}: {n} in {dt:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
