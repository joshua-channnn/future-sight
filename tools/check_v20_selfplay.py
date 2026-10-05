"""Checks for v20/v21 self-play data (milestone 4 of agent_docs/step3_handoff.md, milestone 3 of
step4_brief.md).

Runs a training Worker in-process on a local Showdown server and checks:
  - vs the live policy both sides are recorded, with opposite results (or a tie for both);
    vs a pool model or SimpleHeuristics only the learner's side is;
  - every recorded action is legal under its mask, log-probs are <= 0;
  - GAE: the last return of each side equals its terminal reward (+1 / -1 / 0);
  - collect() returns consistent arrays (lengths, dtypes).
With --arch v21 also the labels (one row per sample):
  - set labels only for revealed opponent tokens, and consistent with what that sample's own
    observation shows (revealed moves within the labelled moves, a known item equal to the label),
    so a row shifted to another step or side would show;
  - in live games, side 1's next-action label at a step where both sides chose is side 2's
    recorded action there (the move behind it, tera, or the Pokemon it switched to), and the
    reverse: pairs the k-th labelled sample of each side;
  - Illusion games have no labels.

    python3 -m tools.check_v20_selfplay --port 8123 --games 12
    python3 -m tools.check_v20_selfplay --port 8123 --games 12 --arch v21
"""
import argparse
from collections import Counter

import numpy as np
import torch

from networks import registry
from training.ppo_v20 import CONFIG, Worker, gae


def check_labels(steps, labels, bad, c):
    from envs import rl_player_v20 as V
    from envs import rl_player_v21 as W
    if len(labels) != len(steps):
        bad.append(("labels per sample", len(labels), len(steps)))
        return
    for s, row in zip(steps, labels):
        idx, flt = s[0], s[1]
        for j in range(6):
            ti = idx[(6 + j) * V.N_PK_IDX:(7 + j) * V.N_PK_IDX]
            placeholder = flt[(6 + j) * V.N_PK_FLT + V.PK_FLT_POS["is_placeholder"]] > 0.5
            item = row[W.LAB_ITEM + j]
            moves = {int(m) for m in row[W.LAB_MOVES + 4 * j:W.LAB_MOVES + 4 * j + 4] if m >= 0}
            if placeholder:
                if item >= 0 or moves:
                    bad.append(("label for an unrevealed token", j))
                continue
            if item < 0:
                c["labels: revealed token without labels (Illusion game / species differs)"] += 1
                continue
            c["labels: revealed tokens labelled"] += 1
            if ti[1] != V.ITEM_UNKNOWN and ti[1] != item:
                bad.append(("item label != item the sample shows", int(ti[1]), int(item)))
            shown = {int(x) for x in ti[6:10] if x != 0}
            if moves and not shown <= moves:
                bad.append(("revealed moves not within the move labels", sorted(shown), sorted(moves)))
            c["labels: tokens checked against the sample's observation"] += 1


def check_next_pairs(sides, labels, bad, c):
    """Live game: side 1's labelled samples and side 2's, in order, are the steps where both chose."""
    from envs import rl_player_v20 as V
    from envs import rl_player_v21 as W
    lab = [[(s, r) for s, r in zip(steps, rows) if r[W.LAB_NEXT] >= 0] for (steps, _), rows in zip(sides, labels)]
    if len(lab[0]) != len(lab[1]):
        bad.append(("live: labelled steps differ between sides", len(lab[0]), len(lab[1])))
        return
    for me, other in ((0, 1), (1, 0)):
        for (s_me, r_me), (s_other, _) in zip(lab[me], lab[other]):
            t = int(r_me[W.LAB_NEXT])
            a = s_other[3]  # the other side's recorded action at this step
            if a < 8:
                want = (W.NEXT_TERA0 if a >= 4 else W.NEXT_MOVE0) + int(s_other[0][V.IDX_PK_END + 2 * (a % 4)])
                ok = t == want
            else:
                species = int(s_other[0][(a - 8) * V.N_PK_IDX])  # own token a - 8 in the other's view
                if t == W.NEXT_UNREVEALED:
                    ok = True
                    c["next: switch to an unrevealed Pokemon"] += 1
                else:
                    j = t - W.NEXT_SWITCH0
                    ok = 0 <= j < 6 and int(s_me[0][(6 + j) * V.N_PK_IDX]) == species
            if not ok:
                bad.append(("next label != the other side's action", t, a))
            c["next: live labels matched to the other side's action"] += ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--arch", default="v20", choices=sorted(registry.ARCHS))
    ap.add_argument("--pool", default="models/ppo_pokemon_6v6_v18.3_best.zip")
    args = ap.parse_args()
    spec = registry.get(args.arch)
    cfg = dict(CONFIG)
    cfg.update(arch=args.arch, base_port=args.port, n_servers=1, net_config=spec.default_net_config())
    w = Worker(0, cfg)
    weights = {k: v.numpy() for k, v in spec.net(cfg["net_config"], spec.layout(cfg)).state_dict().items()}
    w.net.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})
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
            if spec.has_labels:
                from envs.rl_player_v21 import illusion_game
                labels = w.game_labels
                if illusion_game(w.env.battle1, w.env.battle2):
                    c["illusion games"] += 1
                    if any((row != -1).any() for rows in labels for row in rows):
                        bad.append(("labels in an Illusion game",))
                    continue
                for (steps, _), rows in zip(sides, labels):
                    check_labels(steps, rows, bad, c)
                if kind == "live" and len(sides) == 2:
                    check_next_pairs(sides, labels, bad, c)
                else:
                    n_next = sum(int(r[-1] >= 0) for r in labels[0])
                    c[f"next: labelled samples vs {kind}"] += n_next
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
    if spec.has_labels:
        lab = data["lab"]
        c["collect: label rows"] += len(lab)
        for name, start, width in (("item", 0, 6), ("moves", 18, 24), ("next", 42, 1)):
            c[f"collect: samples with {name} labels"] += int((lab[:, start:start + width] >= 0).any(1).sum())
    try:
        w.env.close(force=True, wait=False)
    except Exception:
        pass
    for k in sorted(c):
        print(f"  {k:70s} {c[k]:6d}")
    print("RESULT:", "all checks passed" if not bad else f"{len(bad)} problems: {bad[:5]}")


if __name__ == "__main__":
    main()
