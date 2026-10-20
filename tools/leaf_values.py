"""What a leaf evaluation values: real search self-play states (training/gen_search_games.py, the
decider as side one) edited one field at a time, and the change of the evaluation's v (the logit of
side one winning, pe.evaluator_values) averaged over the states (agent_docs/loss_review.md, section 2).

    python3 -m tools.leaf_values [--eval models/leaf_eval/lin1_86.npz] [--shards 20 --per-shard 150]

Edits, for side one's active ("own") and side two's ("foe"): Attack / Special Attack / Speed /
Defense +1, +2, +6 (from 0); HP from full to 1 HP or 50%, and from 1 HP to fainted; a Substitute;
Leech Seed, Yawn, Taunt, Salt Cure, partial trapping, Encore, confusion; Reflect, Light Screen,
Aurora Veil, Tailwind; a layer of Spikes or Toxic Spikes, Stealth Rock, Sticky Web. States with a
fainted active or a forced switch are left out. Engine state strings: sides are split by "/", a
side's fields by "=", a Pokemon's by "," (poke-engine's state.rs serialize).
"""
import argparse
import glob
import gzip
import random
from collections import defaultdict

import numpy as np
import poke_engine as pe

from search.leaf_eval import evaluator_id

# a side's fields (state.rs Side::serialize)
ACTIVE, CONDITIONS, VOLATILES, SUB_HEALTH, FORCE_SWITCH = 6, 7, 8, 10, 21
BOOSTS = {"Attack": 11, "Defense": 12, "Special Attack": 13, "Speed": 15}
# a Pokemon's fields (Pokemon::serialize) and the side conditions' order (SideConditions::serialize)
HP, MAXHP = 6, 7
CONDITION = {"Reflect": (10, 5), "Light Screen": (3, 5), "Aurora Veil": (0, 5), "Tailwind": (15, 4),
             "a layer of Spikes": (12, 1), "a layer of Toxic Spikes": (17, 1), "Stealth Rock": (13, 1),
             "Sticky Web": (14, 1)}
VOLATILE = {"Substitute": "SUBSTITUTE", "Leech Seed": "LEECHSEED", "Yawn": "YAWN", "Taunt": "TAUNT",
            "Salt Cure": "SALTCURE", "partially trapped": "PARTIALLYTRAPPED", "Encore": "ENCORE",
            "confusion": "CONFUSION"}


class Edit:
    """A state string's two sides as lists of fields, edited and joined back."""

    def __init__(self, s):
        self.parts = s.split("/")
        self.sides = [self.parts[0].split("="), self.parts[1].split("=")]

    def active(self, k):
        return self.sides[k][int(self.sides[k][ACTIVE])].split(",")

    def set_active(self, k, fields):
        self.sides[k][int(self.sides[k][ACTIVE])] = ",".join(fields)

    def string(self):
        return "/".join(["=".join(self.sides[0]), "=".join(self.sides[1])] + self.parts[2:])


def edits(s, k):
    """(name, edited state) for side k of state s, skipping edits that change nothing."""
    out = []
    for stat, i in BOOSTS.items():
        for n in (1, 2, 6):
            e = Edit(s)
            if int(e.sides[k][i]) == 0:
                e.sides[k][i] = str(n)
                out.append((f"active's {stat} +{n}", e.string()))
    e = Edit(s)
    p = e.active(k)
    if int(p[HP]) == int(p[MAXHP]):
        for name, hp in (("active from full HP to 1 HP", 1), ("active from full HP to 50%", int(p[MAXHP]) // 2)):
            e = Edit(s)
            q = e.active(k)
            q[HP] = str(hp)
            e.set_active(k, q)
            out.append((name, e.string()))
    for name, (i, n) in CONDITION.items():
        e = Edit(s)
        sc = e.sides[k][CONDITIONS].split(";")
        if int(sc[i]) == 0:
            sc[i] = str(n)
            e.sides[k][CONDITIONS] = ";".join(sc)
            out.append((name, e.string()))
    for name, v in VOLATILE.items():
        e = Edit(s)
        if v not in e.sides[k][VOLATILES].split(":"):
            e.sides[k][VOLATILES] += v + ":"
            if v == "SUBSTITUTE":
                e.sides[k][SUB_HEALTH] = str(int(e.active(k)[MAXHP]) // 4)
            out.append((name, e.string()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="models/leaf_eval/lin1_86.npz")
    ap.add_argument("--data", default="datasets/search_selfplay")
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--per-shard", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    eid = evaluator_id(args.eval, 2, 1.0)
    rng = random.Random(args.seed)
    states = []
    for f in sorted(glob.glob(f"{args.data}/*.states.gz"))[:args.shards]:
        lines = gzip.open(f, "rt").read().split("\n")[:-1]
        states += rng.sample(lines, min(args.per_shard, len(lines)))
    base, edited, names = [], [], []
    for s in states:
        e = Edit(s)
        if any(int(e.active(k)[HP]) <= 0 or e.sides[k][FORCE_SWITCH] == "true" for k in (0, 1)):
            continue
        for k, who in ((0, "own"), (1, "foe")):
            for name, t in edits(s, k):
                base.append(s), edited.append(t), names.append(f"{who} {name}")
            # the KO: the active at 1 HP, then fainted
            e = Edit(s)
            p = e.active(k)
            p[HP] = "1"
            e.set_active(k, p)
            one = e.string()
            p[HP] = "0"
            e.set_active(k, p)
            base.append(one), edited.append(e.string()), names.append(f"{who} active from 1 HP to fainted")
    v0 = np.asarray(pe.evaluator_values(eid, base))
    v1 = np.asarray(pe.evaluator_values(eid, edited))
    rows = defaultdict(list)
    for name, a, b in zip(names, v0, v1):
        rows[name].append(b - a)
    print(f"{args.eval}: {len(states)} states; the mean change of v (logits of side one winning) "
          f"[10th, 90th percentile]:")
    for name in sorted(rows, key=lambda n: (n.split(" ", 1)[1], n)):
        d = np.array(rows[name])
        print(f"  {name:44} {d.mean():+.3f}  [{np.percentile(d, 10):+.3f}, {np.percentile(d, 90):+.3f}]  "
              f"n={len(d)}")


if __name__ == "__main__":
    main()
