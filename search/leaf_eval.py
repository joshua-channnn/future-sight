"""A learned leaf evaluation for the search's MCTS (agent_docs/step7_brief.md, step 3).

The engine patch (search/engine_patch/learned_eval.patch) adds features of a state
(`pe.features_from_strings`, `pe.feature_names`) and a small antisymmetric model on them,
v(state) = g(x) - g(swap(x)) (the logit of side one winning; g linear, or an MLP with one ReLU
layer), registered with `pe.load_evaluator` and used by `pe.mcts_with`. The weights come from
training/train_leaf_eval.py (an .npz: hidden, w1, b1, w2, feature names; standardization already
folded in). Leaf scores: mode 1 sigmoid(v); mode 2 sigmoid(scale * (v - v(root))), like the
hand-written evaluation's sigmoid(0.0125 * (eval - root eval)).
"""
import numpy as np
import poke_engine as pe

_IDS = {}


def has_engine_support():
    return hasattr(pe, "mcts_with")


def load_weights(path):
    with np.load(path, allow_pickle=False) as d:
        w = {k: d[k] for k in d.files}
    names = [str(x) for x in w["names"]]
    if names != list(pe.feature_names()):
        raise ValueError(f"{path}: its features don't match this engine build's")
    return w


def evaluator_id(path, mode=2, scale=1.0):
    """Register the weights in `path` with the engine (once per path, mode and scale)."""
    if not has_engine_support():
        raise RuntimeError("this poke-engine build has no learned evaluation: bash search/engine_patch/build.sh")
    key = (str(path), int(mode), float(scale))
    if key not in _IDS:
        w = load_weights(path)
        _IDS[key] = pe.load_evaluator(int(w["hidden"]), w["w1"].astype(np.float32).ravel().tolist(),
                                      w["b1"].astype(np.float32).tolist(), w["w2"].astype(np.float32).tolist(),
                                      int(mode), float(scale))
    return _IDS[key]


def mcts(state, ms, iterations, evaluator):
    """pe.monte_carlo_tree_search on one thread with evaluator (0: the hand-written evaluation)."""
    return pe.MctsResult._from_rust(pe.mcts_with(state, int(ms), int(iterations), int(evaluator)))
