"""Checks for v20 self-play data (milestone 4 of agent_docs/step3_handoff.md).

Runs a training Worker in-process on a local Showdown server and checks:
  - vs the live policy both sides are recorded, with opposite results (or a tie for both);
    vs a pool model or SimpleHeuristics only the learner's side is;
  - every recorded action is legal under its mask, log-probs are <= 0;
  - GAE: the last return of each side equals its terminal reward (+1 / -1 / 0);
  - collect() returns consistent arrays (lengths, dtypes).

    python3 -m tools.check_v20_selfplay --port 8123 --games 12
"""
import argparse
import os
from collections import Counter

import numpy as np

from networks.v20_policy import V20Net, DEFAULT_CONFIG
from training.ppo_v20 import CONFIG, Worker, gae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--pool", default="models/ppo_pokemon_6v6_v18.3_best.zip")
    args = ap.parse_args()
    cfg = dict(CONFIG)
    cfg.update(base_port=args.port, n_servers=1, net_config=dict(DEFAULT_CONFIG))
    w = Worker(0, cfg)
    weights = {k: v.numpy() for k, v in V20Net(cfg["net_config"]).state_dict().items()}
    w.net.load_state_dict({k: __import__("torch").as_tensor(v) for k, v in weights.items()})
    w.new_env()
    c = Counter()
    bad = []
    for kind_wanted in ("live", "pool", "heuristic"):
        cfg["heuristic_ratio"] = 1.0 if kind_wanted == "heuristic" else 0.0
        cfg["latest_opponent_ratio"] = 1.0 if kind_wanted == "live" else 0.0
        pool = [args.pool] if kind_wanted != "live" else [args.pool]
        for _ in range(args.games):
            res = w.play_game(pool)
            if res is None:
                c["dropped"] += 1
                continue
            kind, sides, n_learner = res
            c[f"games vs {kind}"] += 1
            expect_sides = 2 if kind == "live" else 1
            if len(sides) != expect_sides:
                bad.append(("sides", kind, len(sides)))
            if kind == "live" and len(sides) == 2:
                w1, w2 = sides[0][1], sides[1][1]
                if not ((w1 is None and w2 is None) or (w1 is True and w2 is False) or (w1 is False and w2 is True)):
                    bad.append(("results not opposite", w1, w2))
                else:
                    c["live: results opposite (or both tie)"] += 1
            if len(sides[0][0]) != n_learner:
                bad.append(("learner samples != learner decisions", len(sides[0][0]), n_learner))
            for steps, won in sides:
                for idx, flt, mask, a, logp, val in steps:
                    if not mask[a]:
                        bad.append(("illegal action recorded", a))
                    if logp > 1e-6:
                        bad.append(("logp > 0", logp))
                reward = 1.0 if won is True else (-1.0 if won is False else 0.0)
                adv, ret = gae([s[5] for s in steps], reward, cfg["gamma"], cfg["gae_lambda"])
                if abs(ret[-1] - reward) > 1e-5:
                    bad.append(("last return != reward", float(ret[-1]), reward))
                c["sides checked"] += 1
    # collect(): array consistency
    cfg["heuristic_ratio"], cfg["latest_opponent_ratio"] = 0.1, 0.5
    data, stats = w.collect(weights, 600, [args.pool])
    n = len(data["act"])
    lens = {k: len(v) for k, v in data.items()}
    if set(lens.values()) != {n} or n != stats["samples"]:
        bad.append(("collect arrays inconsistent", lens, stats["samples"]))
    if not data["mask"][np.arange(n), data["act"]].all():
        bad.append(("collect: illegal action",))
    c[f"collect: {n} samples from {stats['games']} games {dict(stats['by_kind'])}"] += 1
    try:
        w.env.close(force=True, wait=False)
    except Exception:
        pass
    for k in sorted(c):
        print(f"  {k:60s} {c[k]:5d}")
    print("RESULT:", "all checks passed" if not bad else f"{len(bad)} problems: {bad[:5]}")


if __name__ == "__main__":
    main()
