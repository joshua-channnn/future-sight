"""Sanity check for the v20 damage features: predicted damage range (with estimated opponent
stats) vs the HP the opponent actually lost when the move hit.

    python3 -m tools.check_v20_damage --games 40 --port 8123
"""
import argparse
import random
import uuid
from collections import Counter

import numpy as np
from poke_env.data import to_id_str
from poke_env.ps_client import AccountConfiguration

from envs import rl_player_v20 as V
from envs.rl_player_v20 import RLEnvV20, Featurizer, action_to_order
from tools.check_v20_env import server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    uid = uuid.uuid4().hex[:5]
    env = RLEnvV20(server_configuration=server(args.port),
                   account_configuration1=AccountConfiguration.generate(f"DmgA{uid}", rand=True),
                   account_configuration2=AccountConfiguration.generate(f"DmgB{uid}", rand=True))
    fz = Featurizer()
    c = Counter()
    errs = []
    lo_i, hi_i = V.MV_FLT_POS["dmg_lo"], V.MV_FLT_POS["dmg_hi"]
    for g in range(args.games):
        env.reset()
        pending = []  # (battle, turn, move_id, lo, hi, opp_key, opp_hp_before)
        while not env.battle1.finished:
            acts = {}
            for agent, b, to_move in ((env.agent1.username, env.battle1, env.agent1_to_move),
                                      (env.agent2.username, env.battle2, env.agent2_to_move)):
                if not to_move or b.finished:
                    acts[agent] = 0
                    continue
                idx, flt, mask = fz.featurize(b)
                # prefer damaging moves so there is something to measure
                moves = b.available_moves[:4]
                dmg_moves = [i for i, m in enumerate(moves) if mask[i] and m.base_power > 0]
                a = random.choice(dmg_moves) if dmg_moves and random.random() < 0.8 else \
                    int(random.choice(np.flatnonzero(mask)))
                if a < 4 and a in dmg_moves and b.opponent_active_pokemon is not None:
                    f0 = V.FLT_PK_END + a * V.N_MV_FLT
                    opp = b.opponent_active_pokemon
                    pending.append((b, b.turn, moves[a].id, flt[f0 + lo_i], flt[f0 + hi_i],
                                    opp.name, opp.current_hp_fraction, b.player_role))
                acts[agent] = action_to_order(a, b)
            env.step(acts)
        for b, turn, mid, lo, hi, opp_name, hp_before, role in pending:
            obs = b.observations.get(turn)
            if obs is None:
                continue
            evs = obs.events
            opp_role = "p2" if role == "p1" else "p1"
            for i, e in enumerate(evs):
                if len(e) > 3 and e[1] == "move" and e[2][:2] == role and to_id_str(e[3]) == mid:
                    # the next event should be the hit on the opponent's active
                    nxt = evs[i + 1:i + 4]
                    if any(x[1] in ("-crit", "-miss", "-immune", "-activate", "-fail") for x in nxt if len(x) > 1):
                        c["skipped (crit/miss/immune/activate)"] += 1
                        break
                    dmg = [x for x in nxt if len(x) > 3 and x[1] == "-damage" and x[2][:2] == opp_role
                           and x[2][4:].strip() == opp_name and len(x) == 4]
                    if not dmg:
                        c["no plain damage event"] += 1
                        break
                    hp = dmg[0][3]
                    after = 0.0 if hp.startswith("0") else int(hp.split("/")[0]) / int(hp.split("/")[1].split(" ")[0])
                    actual = hp_before - after
                    if after == 0.0:
                        c["KO (actual is a lower bound)"] += 1
                        ok = hi >= actual - 0.02
                        c["KO: predicted max >= HP lost " + ("yes" if ok else "no")] += 1
                        break
                    if lo == 0 and hi == 0:
                        c["predicted 0 (calc failed / immune by ability)"] += 1
                        break
                    inside = lo - 0.03 <= actual <= hi + 0.03
                    c["within predicted range +/-3%" if inside else "outside range"] += 1
                    errs.append(actual - (lo + hi) / 2)
                    break
    env.close()
    for k in sorted(c):
        print(f"  {k:45s} {c[k]:6d}")
    if errs:
        e = np.abs(np.array(errs))
        print(f"  |actual - midpoint|: median {np.median(e):.3f}, p90 {np.quantile(e, 0.9):.3f} (fraction of max HP)")


if __name__ == "__main__":
    main()
