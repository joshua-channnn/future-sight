"""The v21 network in the search loop (agent_docs/step6_brief.md, Phase 2).

NetModel runs v21 once per decision and gives:
  - policy(): its action probabilities over the 14 actions (legal ones), the prior for (a);
  - set_weight_fn(): candidate opponent sets reweighted by the aux heads (item, ability and moves
    of each revealed opponent; tera type optional), for (b). A candidate's weight is its prior weight times, for
    each attribute the observation doesn't show yet, the ratio of the head's probability of the
    candidate's value to the prior's own probability of it among the candidates: where the head only
    knows the per-species rates, nothing changes; where it knows more (a Choice item from repeated
    moves), the candidates shift. Ratios are clipped and tempered (gamma).
  - opponent_probs(): the next-action head's distribution over the opponent's poke-engine options in
    a State, for (c).
"""
import numpy as np
import torch

from envs.rl_player_v20 import (ABILITY_OTHER, ABILITY_UNKNOWN, ABILITY_VOCAB, ITEM_NONE, ITEM_OTHER,
                                ITEM_UNKNOWN, ITEM_VOCAB, N_PK_IDX, TYPE_IDX, TYPE_UNKNOWN, move_index)
from envs.rl_player_v21 import NEXT_MOVE0, NEXT_SWITCH0, NEXT_TERA0
from players.v20_player import V20Policy
from poke_env.battle import PokemonType

V21_PATH = "models/ppo_pokemon_6v6_v21.pt"
TYPE_BY_NAME = {t.name.lower(): i for t, i in TYPE_IDX.items()}


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def item_index(item):
    return ITEM_NONE if not item or item == "none" else (ITEM_VOCAB.get(item) or ITEM_OTHER)


def ability_index(ability):
    return ABILITY_VOCAB.get(ability) or ABILITY_OTHER


class NetModel:
    def __init__(self, path=V21_PATH):
        self.policy_obj = V20Policy.load(path, deterministic=True)
        if self.policy_obj.arch != "v21":
            raise ValueError(f"{path} is {self.policy_obj.arch}; Phase 2 needs v21's aux heads")
        self.net = self.policy_obj.net
        self.fz = self.policy_obj.featurizer
        self._key = None
        self._out = None

    def outputs(self, battle):
        """Network outputs for battle's current decision (computed once per decision)."""
        key = (id(battle), battle.turn, len(battle.current_observation.events), id(battle.last_request))
        if key == self._key:
            return self._out
        idx, flt, mask = self.fz.featurize(battle)
        flt = flt.astype(np.float16).astype(np.float32)   # as in training
        with torch.inference_mode():
            logits, value, aux = self.net(torch.as_tensor(idx)[None], torch.as_tensor(flt)[None],
                                          torch.as_tensor(mask)[None], aux=True)
        out = {"policy": _softmax(logits[0].numpy().astype(np.float64)), "value": float(value[0]),
               "idx": idx}
        for name in ("item", "ability", "tera", "moves"):
            out[name] = _softmax(aux[name][0].numpy().astype(np.float64))   # (6, classes)
        out["next"] = _softmax(aux["next"][0].numpy().astype(np.float64))
        self._key, self._out = key, out
        return out

    # ---- (a) ----------------------------------------------------------------------------------
    def policy(self, battle):
        return self.outputs(battle)["policy"]

    # ---- (b) ----------------------------------------------------------------------------------
    def set_weight_fn(self, battle, gamma=0.5, clip=20.0, tera=False):
        """weight_fn(j, cands, weights) -> new weights for search.sampling.sample_opponent.
        tera: also reweight by the tera-type head (off by default: tools/check_set_model.py found it
        no better than the prior on the hidden tera type)."""
        out = self.outputs(battle)
        idx = out["idx"]

        def tok(j, field):
            return int(idx[(6 + j) * N_PK_IDX + field])

        def fn(j, cands, weights):
            if j >= 6 or not cands:
                return weights
            w = np.asarray(weights, dtype=np.float64)
            if w.sum() <= 0:
                return weights
            ratio = np.ones(len(cands))
            shown = {"item": tok(j, 1) != ITEM_UNKNOWN, "ability": tok(j, 2) != ABILITY_UNKNOWN,
                     "tera": tok(j, 3) != TYPE_UNKNOWN or not tera}
            for name, value_of, n_cls in (
                    ("item", lambda s: item_index(s.item), out["item"].shape[-1]),
                    ("ability", lambda s: ability_index(s.ability), out["ability"].shape[-1]),
                    ("tera", lambda s: TYPE_BY_NAME.get(s.tera, 0), out["tera"].shape[-1])):
                if shown[name]:
                    continue
                vals = np.array([value_of(s) for s in cands])
                base = np.bincount(vals, weights=w, minlength=n_cls) / w.sum()
                q = out[name][j]
                ratio *= np.clip(q[vals] / np.maximum(base[vals], 1e-9), 1 / clip, clip)
            # moves: unrevealed ones only, both distributions over the moves not shown yet
            shown_moves = {int(x) for x in idx[(6 + j) * N_PK_IDX + 6:(6 + j) * N_PK_IDX + 10] if x}
            mv = [[move_index(m) for m in s.moves if move_index(m) not in shown_moves] for s in cands]
            n_cls = out["moves"].shape[-1]
            base = np.zeros(n_cls)
            for wi, ms in zip(w, mv):
                base[ms] += wi
            base /= w.sum()
            q = out["moves"][j].copy()
            q[list(shown_moves)] = 0.0
            if q.sum() > 0 and base.sum() > 0:
                q *= base.sum() / q.sum()
                for i, ms in enumerate(mv):
                    if ms:
                        ratio[i] *= float(np.prod(np.clip(q[ms] / np.maximum(base[ms], 1e-9), 1 / clip, clip)))
            new = w * np.clip(ratio, 1e-6, 1e6) ** gamma
            return list(new) if new.sum() > 0 else weights

        return fn

    # ---- (c) ----------------------------------------------------------------------------------
    def opponent_probs(self, battle, state, options, n_revealed, eps=0.05):
        """Probabilities of the opponent's poke-engine options (side two of `state`, from
        battle's player's view) under the next-action head; eps of uniform mixed in. n_revealed:
        how many of side two's Pokemon (in order) are revealed ones; the rest were sampled."""
        p = self.outputs(battle)["next"]
        n_moves = NEXT_TERA0 - NEXT_MOVE0
        placeholders = p[NEXT_SWITCH0 + n_revealed:NEXT_SWITCH0 + 6].sum()
        unrevealed_opts = [o for o in options if o.startswith("switch ")
                           and _side_index(state.side_two, o[7:]) >= n_revealed]
        probs = []
        for o in options:
            if o in ("none", "No Move"):
                probs.append(1.0)
            elif o.startswith("switch "):
                k = _side_index(state.side_two, o[7:])
                if 0 <= k < n_revealed:
                    probs.append(p[NEXT_SWITCH0 + k])
                else:
                    probs.append(placeholders / max(len(unrevealed_opts), 1))
            else:
                tera = o.endswith("-tera")
                mi = move_index(o[:-5] if tera else o)
                probs.append(p[(NEXT_TERA0 if tera else NEXT_MOVE0) + mi] if mi < n_moves else 0.0)
        probs = np.array(probs, dtype=np.float64)
        if probs.sum() <= 0:
            probs = np.ones(len(options))
        probs = probs / probs.sum()
        return (1 - eps) * probs + eps / len(options)


def _side_index(side, pokemon_id):
    for k, p in enumerate(side.pokemon):
        if p.id.lower() == pokemon_id:
            return k
    return -1
