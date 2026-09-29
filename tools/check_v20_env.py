"""Checks for the v20 environment against live battles (milestone 1 of agent_docs/step3_handoff.md).

Plays games on a local Showdown server with both sides in one RLEnvV20, so each side's true team
(the other battle object, and the request) is available as ground truth, and counts mismatches:
  1. features: own tokens vs the request; opponent tokens vs the other side's true team
     (species, level, HP, status, boosts, item, ability, tera, moves, stats, active, fainted);
     field state mirrored between the two sides.
  2. mapping: every legal order (battle.valid_orders) maps to exactly one unmasked action and back,
     and every unmasked action maps to a legal order.
  3. orders: the order sent for an action shows up in the battle log (the chosen move, switch or
     Terastallization), unless the log explains why not (cant, faint, forced out).

    python3 -m tools.check_v20_env --games 50 --port 8123
"""
import argparse
import random
import uuid
from collections import Counter

import numpy as np
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.data import to_id_str

from envs import rl_player_v20 as V
from envs.rl_player_v20 import (RLEnvV20, Featurizer, action_mask, action_to_order, order_to_action,
                                N_PK_IDX, N_PK_FLT, PK_FLT_POS, IDX_PK_END)


def server(port):
    return ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket",
                               "https://play.pokemonshowdown.com/action.php?")


def tok(idx, flt, k):
    return idx[k * N_PK_IDX:(k + 1) * N_PK_IDX], flt[k * N_PK_FLT:(k + 1) * N_PK_FLT]


def item_idx(item):
    if not item:
        return V.ITEM_NONE
    return V.ITEM_VOCAB.get(to_id_str(item)) or V.ITEM_OTHER


class Checker:
    def __init__(self):
        self.c = Counter()
        self.examples = {}
        self.unexplained = []
        self.move_debug = {}

    prefix = ""

    def eq(self, name, ok, detail=""):
        name = self.prefix + name
        self.c[name + (" ok" if ok else " MISMATCH")] += 1
        if not ok and name not in self.examples:
            self.examples[name] = repr(detail)

    # ---- 1. features ------------------------------------------------------------------------
    def own_tokens(self, battle, idx, flt):
        req = battle.last_request
        by_ident = {p["ident"]: p for p in req["side"]["pokemon"]}
        for k, (key, mon) in enumerate(list(battle.team.items())[:6]):
            p = by_ident.get(key)
            if p is None:
                self.eq("own:in_request", False, key)
                continue
            ti, tf = tok(idx, flt, k)
            self.eq("own:species", ti[0] == V.species_index(mon) and ti[0] != V.SPECIES_OTHER,
                    (mon.species, int(ti[0])))
            tera = p.get("teraType")
            self.eq("own:tera_type", tera is not None and
                    ti[3] == V.TYPE_IDX[V.PokemonType.from_name(tera)], (tera, int(ti[3])))
            self.eq("own:item", ti[1] == item_idx(p.get("item")), (p.get("item"), int(ti[1])))
            ab = p.get("ability") or p.get("baseAbility")
            self.eq("own:ability", ti[2] == (V.ABILITY_VOCAB.get(to_id_str(ab)) or V.ABILITY_OTHER),
                    (ab, int(ti[2])))
            want_moves = sorted(V.move_index(m) for m in p["moves"])
            got_moves = sorted(int(x) for x in ti[6:10] if x != 0)
            self.eq("own:moves", want_moves == got_moves, (p["moves"], got_moves))
            cond = p["condition"]
            fnt = cond.endswith("fnt")
            self.eq("own:fainted", bool(tf[PK_FLT_POS["fainted"]]) == fnt, cond)
            if not fnt:
                cur, mx = cond.split(" ")[0].split("/")
                self.eq("own:hp", abs(tf[PK_FLT_POS["hp_frac"]] - int(cur) / int(mx)) < 1e-4,
                        (cond, float(tf[PK_FLT_POS["hp_frac"]])))
                self.eq("own:stat_hp", round(tf[PK_FLT_POS["stat_hp"]] * 500) == int(mx),
                        (mx, tf[PK_FLT_POS["stat_hp"]] * 500))
                status = cond.split(" ")[1] if " " in cond else None
                want = {None: "st_none", "brn": "st_brn", "frz": "st_frz", "par": "st_par",
                        "psn": "st_psn", "slp": "st_slp", "tox": "st_tox"}[status]
                self.eq("own:status", tf[PK_FLT_POS[want]] == 1.0, (cond,))
            for s in ("atk", "def", "spa", "spd", "spe"):
                self.eq("own:stats", round(tf[PK_FLT_POS["stat_" + s]] * 500) == p["stats"][s],
                        (s, p["stats"][s], tf[PK_FLT_POS["stat_" + s]] * 500))
            self.eq("own:active", bool(tf[PK_FLT_POS["active"]]) == p["active"], key)
        if req.get("active") and not battle.force_switch:
            # move tokens follow available_moves; PP comes from the request
            reqmoves = {m["id"]: m for m in req["active"][0]["moves"]}
            mv0 = IDX_PK_END
            for i, m in enumerate(battle.available_moves[:4]):
                mi = idx[mv0 + 2 * i]
                self.eq("move:id", mi == V.move_index(m.id), (m.id, int(mi)))
                rm = reqmoves.get(m.id)
                if rm and "pp" in rm:
                    f0 = V.FLT_PK_END + i * V.N_MV_FLT
                    self.eq("move:pp", abs(flt[f0 + V.MV_FLT_POS["pp_frac"]] - rm["pp"] / rm["maxpp"]) < 1e-4,
                            (m.id, rm["pp"], rm["maxpp"]))

    def opp_tokens(self, battle, truth_battle, idx, flt):
        opp_items = list(battle.opponent_team.items())[:6]
        truth_team = truth_battle.team
        n_revealed_truth = sum(1 for m in truth_team.values() if m.revealed)
        self.eq("opp:n_revealed", len(opp_items) == n_revealed_truth,
                (len(opp_items), n_revealed_truth))
        for j in range(6):
            ti, tf = tok(idx, flt, 6 + j)
            if j >= len(opp_items):
                self.eq("opp:placeholder", ti[0] == V.SPECIES_UNREVEALED and tf[PK_FLT_POS["is_placeholder"]] == 1)
                continue
            key, mon = opp_items[j]
            t = truth_team.get(key)
            if t is None:
                self.eq("opp:truth_found", False, key)
                continue
            if mon.species != t.species:
                # a forme change the two sides report differently (e.g. Eiscue-Noice)
                self.c["opp:skipped (species differs between sides)"] += 1
                continue
            self.eq("opp:species", ti[0] == V.species_index(t), (mon.species, t.species))
            self.eq("opp:level", mon.level == t.level, (mon.level, t.level))
            self.eq("opp:fainted", mon.fainted == t.fainted, key)
            self.eq("opp:active", bool(mon.active) == bool(t.active), key)
            if not t.fainted:
                hp_ok = abs(tf[PK_FLT_POS["hp_frac"]] - t.current_hp_fraction) <= 0.011
                if not hp_ok and not t.active and t.ability == "regenerator":
                    self.c["opp:hp hidden (Regenerator heals off the field)"] += 1
                else:
                    self.eq("opp:hp", hp_ok, (mon.species, t.ability, bool(t.active),
                                              float(tf[PK_FLT_POS["hp_frac"]]), t.current_hp_fraction))
                st_ok = mon.status == t.status
                if not st_ok and not t.active and t.ability == "naturalcure":
                    self.c["opp:status hidden (Natural Cure off the field)"] += 1
                else:
                    self.eq("opp:status", st_ok, (mon.species, t.ability, mon.status, t.status))
                self.eq("opp:boosts", mon.boosts == t.boosts, (mon.boosts, t.boosts))
            if ti[1] != V.ITEM_UNKNOWN:
                self.eq("opp:item(known)", ti[1] == item_idx(t.item), (mon.item, t.item))
            else:
                self.c["opp:item unknown"] += 1
            if ti[2] != V.ABILITY_UNKNOWN and (t.ability in ("trace", "imposter") or (t.ability or "").startswith("asone")
                                               or t.species.startswith("ogerpon")):
                # Trace reports the copied ability; As One announces itself as Unnerve; Ogerpon's
                # ability becomes Embody Aspect when it terastallizes
                self.c["opp:ability changed in battle (Trace, As One, Imposter, Ogerpon)"] += 1
            elif ti[2] != V.ABILITY_UNKNOWN:
                self.eq("opp:ability(known)", ti[2] == (V.ABILITY_VOCAB.get(t.ability) or V.ABILITY_OTHER),
                        (mon.ability, t.ability))
            else:
                self.c["opp:ability unknown"] += 1
            if t.is_terastallized:
                self.eq("opp:tera(after tera)", ti[3] == V.TYPE_IDX.get(t.tera_type), (mon.tera_type, t.tera_type))
            else:
                self.eq("opp:tera hidden", ti[3] == V.TYPE_UNKNOWN, int(ti[3]))
            revealed = {int(x) for x in ti[6:10] if x != 0}
            true_moves = {V.move_index(m) for m in t.moves}
            if "transform" in t.moves or t.ability == "imposter" or t.species == "ditto":
                self.c["opp:moves skipped (Transform copies moves)"] += 1
            else:
                ok = revealed <= true_moves
                if not ok and key not in self.move_debug:
                    extra = set(mon.moves) - set(t.moves)
                    evs = [e for turn in sorted(battle.observations) for e in battle.observations[turn].events
                           if len(e) > 3 and e[1] == "move" and to_id_str(e[3]) in extra]
                    evs += [e for e in battle.current_observation.events
                            if len(e) > 3 and e[1] == "move" and to_id_str(e[3]) in extra]
                    self.move_debug[key] = (t.species, sorted(extra), evs[:3])
                self.eq("opp:moves subset", ok, (list(mon.moves), list(t.moves)))
            ts = t.stats
            dex = V.GEN_DATA.pokedex.get(mon.species, {}).get("baseStats")
            if dex is not None and dex != mon.base_stats:
                self.c["opp:stats skipped (forme changed in battle)"] += 1
                continue
            for s in ("hp", "atk", "def", "spa", "spd", "spe"):
                est = round(tf[PK_FLT_POS["stat_" + s]] * 500)
                true = ts.get(s) if s != "hp" else (ts.get("hp") or t.max_hp)
                if s == "hp":
                    # HP EVs are trimmed below 85 for Stealth Rock / Sitrus numbers: within 3
                    self.eq("opp:stat_hp (within 3)", true is not None and abs(est - true) <= 3,
                            (mon.species, est, true))
                elif s in ("atk", "spe") and len(mon.moves) < 4:
                    # 0 Atk / 0 Spe depend on moves not revealed yet: an estimate
                    self.c[f"opp:stat_{s} estimate (<4 moves known) " + ("right" if est == true else "wrong")] += 1
                else:
                    self.eq(f"opp:stat_{s}", true is not None and est == true, (mon.species, s, est, true))

    def field(self, b1, b2, fz):
        self.eq("field:weather", set(b1.weather) == set(b2.weather), (b1.weather, b2.weather))
        self.eq("field:weather_start", fz.tracker(b1).weather_start == fz.tracker(b2).weather_start,
                (fz.tracker(b1).weather_start, fz.tracker(b2).weather_start))
        self.eq("field:fields", b1.fields == b2.fields, (b1.fields, b2.fields))
        self.eq("field:side mirror", b1.side_conditions == b2.opponent_side_conditions
                and b1.opponent_side_conditions == b2.side_conditions,
                (b1.side_conditions, b2.opponent_side_conditions))
        self.eq("field:opp used tera", b1.opponent_used_tera == b2.used_tera)

    # ---- 2. mapping -------------------------------------------------------------------------
    def mapping(self, battle, mask):
        valid = [o for o in battle.valid_orders if not isinstance(o, V.DefaultBattleOrder)]
        if battle._wait:
            return
        acts = set()
        for o in valid:
            a = order_to_action(o, battle)
            special = (o.terastallize and len(battle.available_moves) == 1
                       and battle.available_moves[0].id in ("struggle", "recharge"))
            if special:
                self.c["map:tera+struggle/recharge (masked on purpose)"] += 1
                continue
            ok = a >= 0 and mask[a] and action_to_order(a, battle).message == o.message
            self.eq("map:order->action->order", ok, (o.message, a))
            acts.add(a)
        unmasked = set(np.flatnonzero(mask).tolist())
        if valid:
            self.eq("map:unmasked == legal", unmasked == acts, (sorted(unmasked), sorted(acts)))
        else:
            self.eq("map:no legal order -> only fallback", unmasked == {0}, sorted(unmasked))


def check_orders(chk, decisions):
    """decisions: (battle, turn, order, role, forced, retried) collected during play."""
    for battle, turn, order, role, forced, retried in decisions:
        if retried:
            # Showdown rejected the choice (e.g. a switch while trapped) and asked again
            chk.c["orders:rejected and re-requested (trapped etc.)"] += 1
            continue
        obs = battle.observations.get(turn)
        if obs is None:
            chk.c["orders: turn log missing"] += 1
            continue
        evs = obs.events
        side = [e for e in evs if len(e) > 2 and e[2][:2] == role]
        o = order.order
        if isinstance(o, V.Move):
            used = any(e[1] == "move" and to_id_str(e[3]) == o.id for e in side if len(e) > 3)
            if o.id == "recharge":
                used = True
            explained = any(e[1] in ("cant", "faint", "drag") for e in side) or \
                any(e[1] in ("faint", "win") for e in evs) or \
                any(e[1] == "-start" and len(e) > 3 and "Encore" in e[3] for e in side) or \
                any(e[1] == "switch" for e in side) or \
                any(e[1] == "-activate" and len(e) > 3 and "confusion" in e[3] for e in side)
            if used:
                chk.c["orders:move used"] += 1
            elif explained:
                chk.c["orders:move not used (cant/faint/forced out)"] += 1
            else:
                chk.eq("orders:move", False, (turn, order.message, [e[1:4] for e in side][:8]))
                chk.unexplained.append((turn, order.message, [e[1:5] for e in evs][:30]))
            if order.terastallize:
                chk.eq("orders:tera", any(e[1] == "-terastallize" for e in side),
                       (turn, order.message))
        elif isinstance(o, V.Pokemon):
            ok = any(e[1] == "switch" and e[2][4:].strip() == o.name for e in side if len(e) > 3)
            if not ok and any(e[1] == "-heal" and e[2] == f"{role}: {o.name}" for e in side):
                chk.c["orders:revival blessing revived the chosen slot"] += 1
                continue
            if ok:
                chk.c["orders:switch happened"] += 1
            else:
                chk.eq("orders:switch", False, (turn, order.message, [e[1:4] for e in side][:8]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    uid = uuid.uuid4().hex[:5]
    env = RLEnvV20(server_configuration=server(args.port),
                   account_configuration1=AccountConfiguration.generate(f"ChkA{uid}", rand=True),
                   account_configuration2=AccountConfiguration.generate(f"ChkB{uid}", rand=True))
    sh = SimpleHeuristicsPlayer(battle_format="gen9randombattle", start_listening=False)
    fz = Featurizer()
    chk = Checker()
    n_decisions = 0
    for g in range(args.games):
        env.reset()
        decisions = []
        use_sh = g % 2 == 1  # side 2 plays SimpleHeuristics in odd games, random legal otherwise
        illusion = any(m.ability == "illusion" or m.base_species in ("zoroark", "zorua")
                       for b in (env.battle1, env.battle2) for m in b.team.values())
        # Illusion disguises a Pokemon as another, so the observer's view is wrong by design
        chk.prefix = "illusion game|" if illusion else ""
        last = {}  # role -> index of that side's last decision in `decisions`
        while not env.battle1.finished:
            b1, b2 = env.battle1, env.battle2
            acts = {}
            for agent, b, other, to_move, is_sh in (
                    (env.agent1.username, b1, b2, env.agent1_to_move, False),
                    (env.agent2.username, b2, b1, env.agent2_to_move, use_sh)):
                if not to_move or b.finished:
                    acts[agent] = 0
                    continue
                idx, flt, mask = fz.featurize(b)
                n_decisions += 1
                chk.own_tokens(b, idx, flt)
                chk.opp_tokens(b, other, idx, flt)
                chk.mapping(b, mask)
                if is_sh:
                    order = sh.choose_move(b)
                    a = order_to_action(order, b)
                else:
                    a = int(random.choice(np.flatnonzero(mask)))
                    order = action_to_order(a, b)
                acts[agent] = order
                if b.active_pokemon is not None and not b._wait:
                    prev = last.get(b.player_role)
                    if prev is not None:
                        pb, pturn, porder, prole, pforced, _ = decisions[prev]
                        if pturn == b.turn and not pforced and not b.force_switch:
                            decisions[prev] = (pb, pturn, porder, prole, pforced, True)
                    last[b.player_role] = len(decisions)
                    decisions.append((b, b.turn, order, b.player_role, bool(b.force_switch), False))
            chk.field(b1, b2, fz)
            env.step(acts)
        check_orders(chk, decisions)
    hold = [getattr(a, "request_hold_stats", {}) for a in (env.agent1, env.agent2)]
    env.close()
    print(f"\n{args.games} games, {n_decisions} decisions | requests held {sum(h.get('held', 0) for h in hold)}, "
          f"released by timeout {sum(h.get('timeouts', 0) for h in hold)}")
    for k in sorted(chk.c):
        print(f"  {k:55s} {chk.c[k]:8d}")
    bad = {k: v for k, v in chk.c.items() if "MISMATCH" in k}
    if bad:
        print("\nFirst example of each mismatch:")
        for name, ex in chk.examples.items():
            print(f"  {name}: {ex}")
    for k, v in chk.move_debug.items():
        print("  move not in true set:", k, v)
    for u in chk.unexplained[:6]:
        print("  unexplained move:", u)
    print(f"  featurizer fallbacks (moves with missing data): {fz.fallbacks}")
    print("\nRESULT:", "all checks passed" if not bad else f"{sum(bad.values())} mismatches")


if __name__ == "__main__":
    main()
