"""Checks for the v21 featurizer and labels against live battles (milestone 1 of
agent_docs/step4_brief.md).

Plays games on a local Showdown server with both sides in one RLEnvV20 (side 1 random legal moves,
side 2 SimpleHeuristics in odd games and random in even ones) and counts mismatches:
  1. v20 part: the first v20.N_IDX / N_FLT entries equal v20's Featurizer on the same battle.
  2. history vs the log: the last H move/switch/drag/cant events of the battle log (kind, side,
     actor species, move, turns ago), in order, most recent first.
  3. incremental vs one pass: the tracker that featurize updates at every decision ends up with the
     same events as a fresh tracker that parses the whole log at once.
  4. both sides: at the end of each game the two sides' histories have the same events, with HP
     amounts within percentage rounding (one side sees exact HP, the other percentages).
  5. set labels: each labelled opponent token is the Pokemon the label was read from (species),
     labels agree with what the observation already reveals, and match the true team.
  6. next action: the label converts back to the order the opponent sent (move id, tera, switch
     target), and a move label names one of the opponent active's true moves.
Illusion games (a Zoroark on either side) must have no labels at all.

    python3 -m tools.check_v21_env --games 50 --port 8123
    python3 -m tools.check_v21_env --games 1 --show 1     # print one side's history next to its log
"""
import argparse
import random
import time
import uuid
from collections import Counter

import numpy as np
from poke_env.data import to_id_str
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from envs import rl_player_v20 as V
from envs import rl_player_v21 as W
from envs.rl_player_v20 import RLEnvV20, action_mask, action_to_order, order_to_action
from envs.rl_player_v21 import (HS_FLT_POS, LAB_ABILITY, LAB_ITEM, LAB_MOVES, LAB_NEXT, LAB_TERA,
                                N_HS_FLT, N_HS_IDX, NEXT_SWITCH0, NEXT_TERA0, NEXT_UNREVEALED,
                                illusion_game, next_action_label, set_labels,
                                species_index_from_details)

HP_FIELDS = ("dmg", "target_hp", "self_dmg", "self_heal", "actor_hp", "hazard_dmg")
HP_TOL = 0.02  # Showdown rounds the opponent's HP to whole percent (1% for 0 < hp and hp < max)


def server(port):
    return ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket",
                               "https://play.pokemonshowdown.com/action.php?")


class Checker:
    def __init__(self):
        self.c = Counter()
        self.examples = {}
        self.maxdiff = Counter()

    def eq(self, name, ok, detail=""):
        self.c[name + (" ok" if ok else " MISMATCH")] += 1
        if not ok and name not in self.examples:
            self.examples[name] = repr(detail)[:400]

    # ---- 1. v20 part ------------------------------------------------------------------------
    def v20_part(self, idx, flt, idx20, flt20):
        n, m = len(idx20), len(flt20)
        self.eq("v20 part: idx identical", np.array_equal(idx[:n], idx20), np.flatnonzero(idx[:n] != idx20)[:5])
        self.eq("v20 part: flt identical", np.array_equal(flt[:m], flt20), np.flatnonzero(flt[:m] != flt20)[:5])

    # ---- 2. history vs the log --------------------------------------------------------------
    @staticmethod
    def log_events(battle):
        """(turn, kind, actor key, move id or None, actor species index) for every action event."""
        out, details = [], {}
        turns = sorted(battle.observations)
        streams = [(t, battle.observations[t].events) for t in turns
                   if battle.observations[t] is not battle.current_observation]
        streams.append((battle.turn, battle.current_observation.events))
        for t, events in streams:
            for ev in events:
                kind = ev[1] if len(ev) > 1 else ""
                if kind in ("switch", "drag", "detailschange", "replace") and len(ev) > 3:
                    details[W._Tracker._key(ev[2].split(",")[0])] = ev[3]
                if kind in W.HS_KINDS and len(ev) > 2:
                    key = W._Tracker._key(ev[2].split(",")[0])
                    move = None
                    if kind == "move":
                        move = to_id_str(ev[3])
                    elif kind == "cant" and len(ev) > 4 and ev[4]:
                        move = to_id_str(ev[4])
                    sp = species_index_from_details(details[key]) if key in details else 0
                    out.append((t, kind, key, move, sp))
        return out

    def history(self, battle, idx, flt, H):
        hidx = idx[V.N_IDX:].reshape(H, N_HS_IDX)
        hflt = flt[V.N_FLT:].reshape(H, N_HS_FLT)
        want = self.log_events(battle)[-H:][::-1]
        n_present = int(hflt[:, HS_FLT_POS["present"]].sum())
        self.eq("history: number of events", n_present == len(want), (n_present, len(want)))
        for h, (t, kind, key, move, sp) in enumerate(want[:n_present]):
            f = hflt[h]
            got_kind = W.HS_KINDS[int(np.argmax(f[HS_FLT_POS["k_move"]:HS_FLT_POS["k_cant"] + 1]))]
            self.eq("history: kind", got_kind == kind, (h, got_kind, kind))
            self.eq("history: side", bool(f[HS_FLT_POS["own"]]) == (key[:2] == battle.player_role),
                    (h, key, battle.player_role))
            ago = battle.turn - t
            self.eq("history: turns ago", abs(f[HS_FLT_POS["turns_ago"]] - min(ago / 10.0, 1.0)) < 1e-6
                    and bool(f[HS_FLT_POS["this_turn"]]) == (ago == 0), (h, f[HS_FLT_POS["turns_ago"]], ago))
            self.eq("history: actor species", hidx[h, 0] == sp, (h, int(hidx[h, 0]), sp, key))
            if move is not None:
                self.eq("history: move", hidx[h, 1] == V.move_index(move), (h, int(hidx[h, 1]), move))
        if n_present < H:
            self.eq("history: padding is zero", not hidx[n_present:].any() and not hflt[n_present:].any())

    # ---- 3. incremental vs one pass ---------------------------------------------------------
    def incremental(self, fz, battle, H):
        inc = fz._trackers.get(battle)
        fresh = W._HistTracker(max_len=10 ** 6)
        fresh.update(battle)
        a, b = inc.events[-H:], fresh.events[-H:]
        same = len(a) == len(b) and all(x.turn == y.turn and x.idx == y.idx and x.flt == y.flt
                                        for x, y in zip(a, b))
        self.eq("incremental == one pass", same, (len(a), len(b)))

    # ---- 4. both sides ----------------------------------------------------------------------
    def both_sides(self, b1, b2):
        t1, t2 = W._HistTracker(max_len=10 ** 6), W._HistTracker(max_len=10 ** 6)
        t1.update(b1)
        t2.update(b2)
        e1, e2 = t1.events, t2.events
        self.eq("both sides: same number of events", len(e1) == len(e2), (len(e1), len(e2)))
        for x, y in zip(e1, e2):
            ok = x.turn == y.turn and x.idx == y.idx
            self.eq("both sides: turn, species, move", ok, (x.turn, x.idx, y.turn, y.idx))
            if not ok:
                return
            for name in W.HS_FLT_NAMES:
                p = HS_FLT_POS[name]
                if name == "own":
                    self.eq("both sides: own flag mirrored", x.flt[p] + y.flt[p] == 1.0, (x.flt[p], y.flt[p]))
                elif name in HP_FIELDS:
                    d = abs(x.flt[p] - y.flt[p])
                    self.maxdiff[name] = max(self.maxdiff[name], d)
                    self.eq(f"both sides: {name} within {HP_TOL}", d <= HP_TOL, (x.idx, x.flt[p], y.flt[p]))
                else:
                    self.eq("both sides: flags and boosts", x.flt[p] == y.flt[p], (name, x.idx, x.flt[p], y.flt[p]))

    # ---- 5. set labels ----------------------------------------------------------------------
    def set_labels(self, battle, truth, idx, lab, illusion):
        if illusion:
            self.eq("illusion game: no labels", (lab == -1).all())
            return
        opp = list(battle.opponent_team.items())[:6]
        req = W._request_side(truth)
        for j in range(6):
            ti = idx[(6 + j) * V.N_PK_IDX:(7 + j) * V.N_PK_IDX]
            item, ab, tera = lab[LAB_ITEM + j], lab[LAB_ABILITY + j], lab[LAB_TERA + j]
            moves = [int(m) for m in lab[LAB_MOVES + 4 * j:LAB_MOVES + 4 * j + 4] if m >= 0]
            if j >= len(opp):
                self.eq("labels: none for unrevealed tokens", item == ab == tera == -1 and not moves)
                continue
            key, mon = opp[j]
            t = truth.team.get(key)
            if t is None or mon.species != t.species:
                self.c["labels: skipped (species differs between sides)"] += 1
                self.eq("labels: none when skipped", item == ab == tera == -1 and not moves)
                continue
            self.c["labels: tokens labelled"] += 1
            self.eq("labels: token species == labelled Pokemon", ti[0] == V.species_index(t), (ti[0], t.species))
            p = req.get(key)
            if p is None:
                self.eq("labels: labelled Pokemon in the true request", False, key)
                continue
            self.eq("labels: item == true item", item == (V.ITEM_NONE if not p.get("item") else
                                                          V.ITEM_VOCAB.get(p["item"]) or V.ITEM_OTHER), (item, p.get("item")))
            self.eq("labels: tera == true tera type", tera == V.TYPE_IDX[V.PokemonType.from_name(p["teraType"])],
                    (tera, p.get("teraType")))
            if "transform" not in p["moves"] and t.species != "ditto":
                self.eq("labels: moves == true moves", sorted(moves) == sorted(V.move_index(m) for m in p["moves"]),
                        (moves, p["moves"]))
            # what the observation already shows must agree with the labels
            if ti[1] != V.ITEM_UNKNOWN:
                self.eq("labels: revealed item agrees", ti[1] == item, (mon.item, t.item))
            if ti[2] != V.ABILITY_UNKNOWN:
                if to_id_str(p.get("baseAbility", "")) != to_id_str(p.get("ability", "")) or \
                        t.species.startswith("ogerpon") or to_id_str(p.get("baseAbility", "")).startswith("asone") or \
                        to_id_str(p.get("baseAbility", "")) in ("trace", "imposter"):
                    self.c["labels: ability changed in battle (Trace, As One, Ogerpon, ...)"] += 1
                else:
                    self.eq("labels: revealed ability agrees", ti[2] == ab, (mon.ability, p.get("baseAbility")))
            if ti[3] != V.TYPE_UNKNOWN:
                self.eq("labels: revealed tera agrees", ti[3] == tera, (mon.tera_type, p.get("teraType")))
            revealed = {int(x) for x in ti[6:10] if x != 0}
            if moves:
                self.eq("labels: revealed moves within label moves", revealed <= set(moves),
                        (list(mon.moves), p["moves"]))
            for name, v in (("item", item), ("ability", ab), ("tera", tera)):
                self.c[f"labels: {name} present"] += v >= 0
            self.c["labels: moves present"] += bool(moves)

    # ---- 6. next action ---------------------------------------------------------------------
    def next_action(self, battle, other, order, label, other_choosing, illusion):
        if illusion or not other_choosing:
            self.eq("next: no label when the opponent isn't choosing (or Illusion)", label == -1, label)
            return
        a = order_to_action(order, other)
        if a < 0:
            self.eq("next: no label for a default order", label == -1, label)
            return
        self.c["next: labelled"] += label >= 0
        if label < 0:
            self.c["next: unlabelled (revive / forced-switch move slot)"] += 1
            return
        o = order.order
        if label < NEXT_SWITCH0:
            tera = label >= NEXT_TERA0
            mi = label - (NEXT_TERA0 if tera else 0)
            self.eq("next: move label == order", isinstance(o, V.Move) and V.move_index(o.id) == mi
                    and bool(order.terastallize) == tera, (label, order.message))
            act = other.active_pokemon
            if act is not None and act.species != "ditto" and "transform" not in act.moves:
                self.eq("next: move is one of the active's moves", isinstance(o, V.Move) and
                        (o.id in act.moves or o.id in ("struggle", "recharge")), (o, list(act.moves)))
            self.c["next: move" + (" + tera" if tera else "")] += 1
        else:
            keys = list(battle.opponent_team.keys())[:6]
            key = next((k for k, m in other.team.items() if m is o), None)
            if label == NEXT_UNREVEALED:
                self.eq("next: unrevealed switch target not yet seen", key is not None and key not in keys, key)
                self.c["next: switch to unrevealed"] += 1
            else:
                j = label - NEXT_SWITCH0
                self.eq("next: switch label == order target", key is not None and j < len(keys) and keys[j] == key,
                        (label, key, keys))
                self.c["next: switch to revealed"] += 1


def show(battle, idx, flt, H):
    """One side's history tokens next to the raw log of the last turns."""
    hidx = idx[V.N_IDX:].reshape(H, N_HS_IDX)
    hflt = flt[V.N_FLT:].reshape(H, N_HS_FLT)
    inv_sp = {v: k for k, v in V.SPECIES_VOCAB.items()}
    inv_mv = {v: k for k, v in V.MOVE_VOCAB.items()}
    print(f"\n=== {battle.player_role} at turn {battle.turn}: history, most recent first")
    for h in range(H):
        f = hflt[h]
        if not f[HS_FLT_POS["present"]]:
            break
        kind = W.HS_KINDS[int(np.argmax(f[HS_FLT_POS["k_move"]:HS_FLT_POS["k_cant"] + 1]))]
        nz = {n: round(float(f[HS_FLT_POS[n]]), 3) for n in W.HS_FLT_NAMES[6:] if f[HS_FLT_POS[n]]}
        print(f"  {h:2d} {'own' if f[HS_FLT_POS['own']] else 'opp'} {kind:6s} "
              f"{inv_sp.get(int(hidx[h, 0]), hidx[h, 0])!s:18s} {inv_mv.get(int(hidx[h, 1]), '')!s:16s} "
              f"-> {inv_sp.get(int(hidx[h, 2]), '')!s:16s} {nz}")
    print("--- log")
    for t in sorted(battle.observations)[-4:]:
        for ev in battle.observations[t].events:
            print(f"  t{t} {'|'.join(ev[1:])}")
    for ev in battle.current_observation.events:
        print(f"  now {'|'.join(ev[1:])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--history", type=int, default=W.DEFAULT_HISTORY)
    ap.add_argument("--show", type=int, default=0, help="print the history of this many decisions")
    args = ap.parse_args()
    H = args.history
    uid = uuid.uuid4().hex[:5]
    env = RLEnvV20(server_configuration=server(args.port),
                   account_configuration1=AccountConfiguration.generate(f"ChkA{uid}", rand=True),
                   account_configuration2=AccountConfiguration.generate(f"ChkB{uid}", rand=True))
    sh = SimpleHeuristicsPlayer(battle_format="gen9randombattle", start_listening=False)
    fz = W.Featurizer(history_len=H)
    fz20 = V.Featurizer()
    chk = Checker()
    n_dec = 0
    shown = 0
    t21 = t20 = 0.0
    n_illusion = 0
    for g in range(args.games):
        env.reset()
        use_sh = g % 2 == 1
        illusion = illusion_game(env.battle1, env.battle2)
        n_illusion += illusion
        while not env.battle1.finished:
            b1, b2 = env.battle1, env.battle2
            to_move = {1: env.agent1_to_move and not b1.finished, 2: env.agent2_to_move and not b2.finished}
            acts, orders, pending = {}, {}, []
            for side, agent, b, other, is_sh in ((1, env.agent1.username, b1, b2, False),
                                                 (2, env.agent2.username, b2, b1, use_sh)):
                if not to_move[side]:
                    acts[agent] = 0
                    continue
                t0 = time.perf_counter()
                idx, flt, mask = fz.featurize(b)
                t1 = time.perf_counter()
                idx20, flt20, _ = fz20.featurize(b)
                t2 = time.perf_counter()
                t21 += t1 - t0
                t20 += t2 - t1
                n_dec += 1
                chk.v20_part(idx, flt, idx20, flt20)
                chk.history(b, idx, flt, H)
                chk.incremental(fz, b, H)
                lab = np.full(W.N_LAB, -1, np.int16) if illusion else set_labels(b, other)
                chk.set_labels(b, other, idx, lab, illusion)
                if shown < args.show and b.turn > 6:
                    show(b, idx, flt, H)
                    shown += 1
                if is_sh:
                    order = sh.choose_move(b)
                else:
                    order = action_to_order(int(random.choice(np.flatnonzero(mask))), b)
                acts[agent] = order
                orders[side] = order
                pending.append((side, b, other))
            for side, b, other in pending:
                o = orders.get(3 - side)
                label = -1
                if not illusion and to_move[3 - side] and o is not None:
                    label = next_action_label(b, other, o)
                chk.next_action(b, other, o, label, to_move[3 - side] and o is not None, illusion)
            env.step(acts)
        chk.both_sides(env.battle1, env.battle2)
    env.close()
    print(f"\n{args.games} games ({n_illusion} with Illusion), {n_dec} decisions | featurize: v21 "
          f"{1000 * t21 / max(n_dec, 1):.3f} ms, v20 {1000 * t20 / max(n_dec, 1):.3f} ms per call")
    for k in sorted(chk.c):
        print(f"  {k:70s} {chk.c[k]:8d}")
    print("  largest HP differences between the two sides:",
          {k: round(v, 4) for k, v in chk.maxdiff.items()})
    bad = {k: v for k, v in chk.c.items() if "MISMATCH" in k}
    if bad:
        print("\nFirst example of each mismatch:")
        for name, ex in chk.examples.items():
            print(f"  {name}: {ex}")
    print("\nRESULT:", "all checks passed" if not bad else f"{sum(bad.values())} mismatches")


if __name__ == "__main__":
    main()
