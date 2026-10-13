"""Checks of pe.mcts_tree (the search's nodes for TreeStrap, training/gen_treestrap.py) on stored
search self-play states:
1. every node's mean score times its visits is its own rollout (one, when it was the chance
   outcome sampled at its creation; none, when it was first reached later) plus its children's
   (all nodes dumped), with the own rollout scored from the dumped features: the features are the
   node's state, the values add up;
2. visits are the children's plus 0 or 1;
3. the root: its mean score is the result's side-one mean, its features features_from_strings';
4. min_visits and max_depth hold.

    python3 -m tools.check_treestrap
"""
import argparse
import glob
import gzip

import numpy as np
import poke_engine as pe

from search.leaf_eval import BEST, evaluator_id
from training.train_q_eval import antisym, lin_coefs


def tree(state_str, eid, iterations, min_visits, max_depth=30):
    st = pe.State.from_string(state_str)
    r, feats, depth, parent, visits, q, over = pe.mcts_tree(st, 0, iterations, eid, min_visits, max_depth)
    x = np.frombuffer(feats, np.float32).reshape(len(q), -1)
    return (pe.MctsResult._from_rust(r), x, np.array(depth), np.array(parent), np.array(visits), np.array(q),
            np.array(over))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/search_selfplay_r1")
    ap.add_argument("--states", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=20000)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.data}/*.npz"))
    states = []
    for f in files[::max(1, len(files) // 4)]:
        with gzip.open(f[:-4] + ".states.gz", "rt") as fh:
            states += fh.read().splitlines()[::97]
    states = states[:args.states]
    scale = 1.4
    eid = evaluator_id(BEST, 2, scale)
    coef = lin_coefs(BEST)
    ok = True

    def report(name, err, tol):
        nonlocal ok
        good = err <= tol
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: max |diff| {err:.2e} (tolerance {tol:g})")

    e_value, e_visits, e_root, e_feat, never = 0.0, 0, 0.0, 0.0, []
    for s in states:
        r, x, depth, parent, visits, q, over = tree(s, eid, args.iterations, 1)
        v = antisym(x.astype(np.float64)) @ coef
        child_v = np.bincount(parent[1:], visits[1:] * q[1:], len(q))
        child_n = np.bincount(parent[1:], visits[1:], len(q))
        own = np.where(over, q, 1 / (1 + np.exp(-scale * (v - v[0]))))
        inner = ~over
        inner[0] = False   # the root never rolls out itself
        selfs = (visits - child_n)[inner]
        e_visits += int(((selfs != 0) & (selfs != 1)).sum())
        never.append(float((selfs == 0).mean()))
        lhs = (q * visits)[inner]
        e_value = max(e_value, float(np.abs(lhs - (selfs * own[inner] + child_v[inner])).max() / max(1, np.abs(lhs).max())))
        mean = sum(o.total_score for o in r.side_one) / sum(o.visits for o in r.side_one)
        e_root = max(e_root, abs(mean - q[0]))
        fr = np.asarray(pe.features_from_strings([s]), np.float32).reshape(-1)
        e_feat = max(e_feat, float(np.abs(fr - x[0]).max()))
    n = len(states)
    report(f"1. node values = own rollout + children ({n} searches of {args.iterations}, relative)", e_value, 1e-4)
    report(f"2. visits = children's + 0 or 1 (never rolled out themselves: {np.mean(never):.1%})", e_visits, 0)
    report("3. root mean score vs the result's side-one mean", e_root, 1e-4)
    report("3. root features vs features_from_strings", e_feat, 0)
    worst = 0
    for s in states[:5]:
        _, _, depth, _, visits, _, _ = tree(s, eid, args.iterations, 50, 2)
        worst = max(worst, int((visits[1:] < 50).sum()) + int((depth > 2).sum()))
    report("4. min_visits 50 and max_depth 2 hold", worst, 0)
    print("all checks passed" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
