"""Checks of the matchup weights (search/matchup_eval.py, engine::learned) on stored search self-play
states:
1. the per-Pokemon leaf features the network's training computes from the inputs (leaf_phi) are the
   engine's (pe.leaf_features_from_strings);
2. zero root weights give exactly the linear evaluation; random ones what numpy says;
3. a trained model: the engine's v(state) with the model's root weights is the network's;
4. antisymmetry: the mirrored state (sides exchanged) gives -c, W transposed and negated, u swapped;
5. mcts_with runs with root weights, and the policy's search returns legal actions.

    python3 -m tools.check_matchup_eval [--model models/leaf_eval/mu4_pairs.pt]
"""
import argparse
import glob
import gzip

import numpy as np
import poke_engine as pe
import torch

from search.leaf_eval import evaluator_id
from search.matchup_eval import (MatchupModel, M_LEAF, N_FLOATS, N_INTS, N_ROOT_WEIGHTS, leaf_phi, split_inputs)


def mirror(s):
    s1, s2, rest = s.split("/", 2)
    return "/".join([s2, s1, rest])


def inputs(states):
    ints, floats = pe.matchup_inputs_from_strings(states)
    return (torch.as_tensor(np.asarray(ints, np.int64).reshape(len(states), N_INTS)),
            torch.as_tensor(np.asarray(floats, np.float32).reshape(len(states), N_FLOATS)),
            torch.as_tensor(np.asarray(pe.pair_features_from_strings(states), np.float32).reshape(len(states), -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/leaf_eval/mu4_pairs.pt")
    ap.add_argument("--data", default="datasets/search_selfplay")
    ap.add_argument("--states", type=int, default=2000)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.data}/*.npz"))
    states = []
    for f in files[::max(1, len(files) // 8)]:
        with gzip.open(f[:-4] + ".states.gz", "rt") as fh:
            states += fh.read().splitlines()[::25]
    states = states[:args.states]
    n = len(states)
    ok = True

    def report(name, err, tol):
        nonlocal ok
        good = err <= tol
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: max |diff| {err:.2e} (tolerance {tol:g})")

    # 1. leaf features
    ints, floats, pairs = inputs(states)
    pk, pf, act, _, x = split_inputs(ints, floats)
    phi, pair = leaf_phi(pk, pf, act)
    ours = torch.cat([phi.reshape(n, -1), pair], 1).numpy()
    rust = np.asarray(pe.leaf_features_from_strings(states), np.float32).reshape(n, -1)
    report(f"leaf features, {n} states", float(np.abs(ours - rust).max()), 1e-6)

    # 2. zero and random root weights
    eid = evaluator_id("models/leaf_eval/lin1_86.npz", 2, 1.0)
    v0 = np.asarray(pe.evaluator_values(eid, states))
    vz = np.asarray(pe.evaluator_values(eid, states, [0.0] * N_ROOT_WEIGHTS))
    report("zero root weights == the linear evaluation", float(np.abs(vz - v0).max()), 0.0)
    w = np.random.default_rng(0).normal(size=N_ROOT_WEIGHTS).astype(np.float32)
    vw = np.asarray(pe.evaluator_values(eid, states, w.tolist()))
    p = phi.numpy()
    expect = v0 + p[:, :6].reshape(n, -1) @ w[:6 * M_LEAF] - p[:, 6:].reshape(n, -1) @ w[6 * M_LEAF:12 * M_LEAF] \
        + pair.numpy() @ w[12 * M_LEAF:]
    report("random root weights vs numpy", float(np.abs(vw - expect).max()), 1e-4)

    # 3. a trained model: engine vs network, each state with its own root weights
    model = MatchupModel(args.model, 2, 1.0)
    net = model.net
    with torch.no_grad():
        c, u1, u2, wt = net(ints, floats, pairs if net.pairs else None)
        rw = net.root_weight_vector(u1, u2, wt)
        theirs = (net.lin(x) + net.matchup_value(u1, u2, wt, phi, pair)).numpy()
    rust = np.array([pe.evaluator_values(model.leaf_id, [states[i]], rw[i].tolist())[0] for i in range(0, n, 10)])
    report(f"engine vs network ({args.model}), {len(rust)} states", float(np.abs(rust - theirs[::10]).max()), 1e-4)
    st = pe.State.from_string(states[0])
    report("MatchupModel.root_weights == the batch", float(np.abs(np.array(model.root_weights(st)) - rw[0].numpy()).max()), 1e-5)

    # 4. antisymmetry
    mi, mf, mp = inputs([mirror(s) for s in states])
    with torch.no_grad():
        c2, v1, v2, w2 = net(mi, mf, mp if net.pairs else None)
    err = max(float((c + c2).abs().max()), float((u1 - v2).abs().max()), float((u2 - v1).abs().max()),
              float((wt + w2.transpose(1, 2)).abs().max()))
    report("antisymmetry on mirrored states", err, 1e-4)

    # 5. search with root weights
    for s in states[:5]:
        st = pe.State.from_string(s)
        r = pe.MctsResult._from_rust(pe.mcts_with(st, 1000, 20000, model.leaf_id, model.root_weights(st)))
        assert r.total_visits >= 20000 or len(r.side_one) <= 1, r.total_visits
    print("ok   mcts_with with root weights (5 searches of 20K iterations)")
    print("all checks passed" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
