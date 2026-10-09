"""Milestone 3 of agent_docs/step6_brief.md: the converter (search/state.py) and the sampler
(search/sampling.py) on live two-sided games, where each side's request gives the true state.

Plays games with both sides in one RLEnvV20 (side 1 random legal actions, side 2 SimpleHeuristics in
odd games and random in even ones) and at every decision:
  1. Observer vs truth: the observer's state (opponent from the log + one sampled team) against the
     true state (opponent from its own request), on what the observer can know: species, level,
     HP (within the percentage rounding), status, revealed moves / item / ability / tera type, the
     active's boosts and volatiles, side conditions, field. Plus how often the sampled set's hidden
     parts (item, ability, tera type, moves) are right.
  2. Engine fidelity: from the true state and both sides' actual choices, does one of poke-engine's
     chance branches (generate_instructions) reproduce the next true state? Compared: every
     Pokemon's HP, fainted, status; actives; the actives' boosts; hazards; weather and terrain.
     HP counts as matching within the engine's damage-roll spread (it uses 92.5% of the highest
     roll; Showdown rolls 85-100%): |engine - true| <= 9% of the true HP change + 1% of max HP.
     Also reported: within 1% of max HP regardless.
  3. Options: poke-engine's options for the deciding side (true state) map onto exactly the legal
     actions (action_mask); separately for the observer's state (sampled trapping abilities can
     remove switches).
Illusion games (Zoroark on either side) are counted apart.

    python3 -m tools.check_search_state --games 100 --port 8123
"""
import argparse
import random
import time
import traceback
import uuid
from collections import Counter, defaultdict

import numpy as np
import poke_engine as pe
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

from envs.rl_player_v20 import RLEnvV20, action_mask, action_to_order
from envs.rl_player_v21 import illusion_game
from eval.v20_eval import server_config
from search.sampling import get_prior, observations, sample_opponent
from search.state import build_state, build_true_state, option_to_action, order_to_option, tracker

HAZARDS = ("spikes", "toxic_spikes", "stealth_rock", "sticky_web")
MOVE_IDS = set(__import__("poke_env.data", fromlist=["GenData"]).GenData.from_gen(9).moves)
SCREENS = ("reflect", "light_screen", "aurora_veil", "tailwind")
BOOSTS = ("attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost", "speed_boost")


class Checker:
    def __init__(self):
        self.c = Counter()
        self.examples = defaultdict(list)
        self.hp_err = []        # fidelity: worst HP error per transition (fraction of max HP)
        self.move_all = Counter()   # fidelity: transitions per choice (move id or "switch")
        self.move_bad = Counter()   # ... of which mismatched (categorical or HP)

    def eq(self, name, ok, detail=""):
        self.c[name + (" ok" if ok else " MISMATCH")] += 1
        if not ok and len(self.examples[name]) < 3:
            self.examples[name].append(str(detail)[:900])
        return ok


# ---- 1. observer vs truth -------------------------------------------------------------------------
def compare_observer(chk, obs_state, obs_info, true_state, true_info, battle, sets):
    """Side two of the observer's state against the true state's side two."""
    o2, t2 = obs_state.side_two, true_state.side_two
    tidx = {k: i for i, k in enumerate(true_info["keys2"])}
    opp = battle.opponent_team
    for i, key in enumerate(obs_info["keys2"]):
        po = o2.pokemon[i]
        if key not in opp:
            continue  # sampled unrevealed Pokemon
        j = tidx.get(key)
        if not chk.eq("obs: revealed Pokemon in the true team", j is not None, key):
            continue
        pt = t2.pokemon[j]
        mon = opp[key]
        chk.eq("obs: species / forme", po.id.lower() == pt.id.lower(), (key, po.id, pt.id))
        chk.eq("obs: level", po.level == pt.level, (key, po.level, pt.level))
        fo = po.hp / max(po.maxhp, 1)
        ft = pt.hp / max(pt.maxhp, 1)
        # Showdown shows the opponent's HP as ceil(100 * hp / maxhp)%: the observer can be 1% off
        chk.eq("obs: HP fraction within 1%", abs(fo - ft) <= 0.0101 or (po.hp == 0 and pt.hp == 0),
               (key, round(fo, 4), round(ft, 4)))
        chk.eq("obs: fainted", (po.hp == 0) == (pt.hp == 0), (key, po.hp, pt.hp))
        if pt.hp > 0:
            chk.eq("obs: status", po.status.lower() == pt.status.lower(), (key, po.status, pt.status))
        true_moves = {m.id.lower() for m in pt.moves}
        shown = {m for m in mon.moves if m not in ("struggle", "recharge")}
        if "transform" not in true_moves and pt.id.lower() != "ditto":
            chk.eq("obs: revealed moves are true moves", shown <= true_moves, (key, sorted(shown), sorted(true_moves)))
        if mon.item not in ("unknown_item",):
            chk.eq("obs: known item", (mon.item or "none") == pt.item.lower(), (key, mon.item, pt.item))
        else:
            chk.c["sampled item right"] += po.item.lower() == pt.item.lower()
            chk.c["sampled item (count)"] += 1
        if mon.ability:
            chk.c["revealed ability == current true ability"] += mon.ability == pt.ability.lower()
            chk.c["revealed ability (count)"] += 1
        else:
            chk.c["sampled ability right"] += po.ability.lower() == pt.ability.lower()
            chk.c["sampled ability (count)"] += 1
        if pt.terastallized:
            chk.eq("obs: terastallized", po.terastallized, key)
            chk.eq("obs: tera type once terastallized", po.tera_type.lower() == pt.tera_type.lower(),
                   (key, po.tera_type, pt.tera_type))
        else:
            chk.eq("obs: not terastallized", not po.terastallized, key)
            chk.c["sampled tera type right"] += po.tera_type.lower() == pt.tera_type.lower()
            chk.c["sampled tera type (count)"] += 1
        hidden = true_moves - shown
        if hidden and "transform" not in true_moves:
            got = {m.id.lower() for m in po.moves} - shown
            chk.c["sampled hidden moves right"] += len(got & hidden)
            chk.c["sampled hidden moves (count)"] += len(hidden)
        chk.c["stats right (hp, atk, def, spa, spd, spe all)"] += (po.maxhp, po.attack, po.defense, po.special_attack,
                                                                    po.special_defense, po.speed) == \
            (pt.maxhp, pt.attack, pt.defense, pt.special_attack, pt.special_defense, pt.speed)
        chk.c["stats (count)"] += 1
    # actives
    ao, at = int(o2.active_index), int(t2.active_index)
    ko, kt = obs_info["keys2"][ao], true_info["keys2"][at]
    chk.eq("obs: opponent active", ko == kt, (ko, kt))
    if ko == kt and t2.pokemon[at].hp > 0:
        for b in BOOSTS + ("accuracy_boost", "evasion_boost"):
            chk.eq("obs: active boosts", getattr(o2, b) == getattr(t2, b), (b, getattr(o2, b), getattr(t2, b)))
        vo = {v.lower() for v in o2.volatile_statuses}
        vt = {v.lower() for v in t2.volatile_statuses}
        chk.eq("obs: active volatiles", vo == vt, (sorted(vo), sorted(vt)))
        chk.eq("obs: substitute", (o2.substitute_health > 0) == (t2.substitute_health > 0),
               (o2.substitute_health, t2.substitute_health))
    # side conditions and field (durations can depend on an unknown item: counted apart)
    for side_name in ("side_one", "side_two"):
        so, st = getattr(obs_state, side_name).side_conditions, getattr(true_state, side_name).side_conditions
        for h in HAZARDS:
            chk.eq("obs: hazards", getattr(so, h) == getattr(st, h), (side_name, h, getattr(so, h), getattr(st, h)))
        for s in SCREENS:
            a, b = getattr(so, s), getattr(st, s)
            chk.eq("obs: screens / tailwind present", (a > 0) == (b > 0), (side_name, s, a, b))
            if a > 0 and b > 0:
                chk.c["screen duration right" if a == b else "screen duration differs (Light Clay?)"] += 1
    chk.eq("obs: weather", obs_state.weather.lower() == true_state.weather.lower(), (obs_state.weather, true_state.weather))
    if obs_state.weather.lower() != "none":
        chk.c["weather turns right" if obs_state.weather_turns_remaining == true_state.weather_turns_remaining
              else "weather turns differ (rock?)"] += 1
    chk.eq("obs: terrain", obs_state.terrain.lower() == true_state.terrain.lower(), (obs_state.terrain, true_state.terrain))
    chk.eq("obs: trick room", obs_state.trick_room == true_state.trick_room)


# ---- 2. engine fidelity ---------------------------------------------------------------------------
def summarize(state):
    out = {"weather": state.weather.lower(), "terrain": state.terrain.lower()}
    for name in ("side_one", "side_two"):
        s = getattr(state, name)
        out[name] = {
            "active": int(s.active_index),
            "hp": [p.hp for p in s.pokemon], "maxhp": [p.maxhp for p in s.pokemon],
            "status": [p.status.lower() if p.hp > 0 else "fnt" for p in s.pokemon],
            "boosts": tuple(getattr(s, b) for b in BOOSTS),
            "hazards": tuple(getattr(s.side_conditions, h) for h in HAZARDS),
        }
    return out


def diff(eng, true, before):
    """Mismatches between an engine branch's result and the true next state: (categorical list,
    worst HP error as a fraction of max HP, HP outside the damage-roll tolerance)."""
    cats = []
    worst = 0.0
    hp_bad = []
    for name in ("side_one", "side_two"):
        e, t, b = eng[name], true[name], before[name]
        if e["active"] != t["active"]:
            cats.append(f"{name} active")
        for i in range(6):
            mx = max(t["maxhp"][i], 1)
            err = abs(e["hp"][i] - t["hp"][i]) / mx
            worst = max(worst, err)
            change = abs(b["hp"][i] - t["hp"][i])
            if abs(e["hp"][i] - t["hp"][i]) > 0.09 * change + 0.01 * mx + 1:
                hp_bad.append((name, i))
            if (e["hp"][i] == 0) != (t["hp"][i] == 0):
                cats.append(f"{name} faint")
            elif t["hp"][i] > 0 and e["status"][i] != t["status"][i]:
                cats.append(f"{name} status")
        if e["active"] == t["active"] and t["hp"][t["active"]] > 0 and e["boosts"] != t["boosts"]:
            cats.append(f"{name} boosts")
        if e["hazards"] != t["hazards"]:
            cats.append(f"{name} hazards")
    if eng["weather"] != true["weather"]:
        cats.append("weather")
    if eng["terrain"] != true["terrain"]:
        cats.append("terrain")
    return cats, worst, hp_bad


def all_events(battle):
    obs = battle.observations
    out = [ev for t in sorted(obs) if obs[t] is not battle.current_observation for ev in obs[t].events]
    return out + list(battle.current_observation.events)


SHOW = {"n": 0, "events": None}


def show_mismatch(label, s1, s2, cats, before, true, res, branch, events):
    print(f"\n--- fidelity mismatch {label}: s1={s1} s2={s2}: {cats}")
    for ev in events:
        print("    |" + "|".join(ev[1:]))
    for name in ("side_one", "side_two"):
        b, t, e = before[name], true[name], res[name]
        print(f"  {name}: active {b['active']}->{t['active']} (engine {e['active']}); boosts {b['boosts']}->"
              f"{t['boosts']} (engine {e['boosts']}); hazards {b['hazards']}->{t['hazards']} (engine {e['hazards']})")
        print(f"     hp {b['hp']}\n     -> {t['hp']}\n  eng  {e['hp']}; status {t['status']} (engine {e['status']})")
    print(f"  best branch ({branch.percentage:.3f}): {branch}")


def fidelity(chk, before_state, s1, s2, after_state, label):
    try:
        branches = pe.generate_instructions(before_state, s1, s2)
    except BaseException as e:  # noqa: BLE001 - poke-engine panics surface as PanicException
        chk.eq("fidelity: generate_instructions ran", False, (label, s1, s2, repr(e)[:200]))
        return
    chk.eq("fidelity: generate_instructions ran", True)
    before = summarize(before_state)
    true = summarize(after_state)
    best = None
    for br in branches:
        res = summarize(before_state.apply_instructions(br))
        cats, worst, hp_bad = diff(res, true, before)
        score = (len(cats), len(hp_bad), worst)
        if best is None or score < best[0]:
            best = (score, cats, worst, hp_bad, br, res)
    _, cats, worst, hp_bad, br, res = best
    for m in {s1, s2}:
        name = "switch / none" if (m == "none" or not any(m.replace("-tera", "") == x for x in MOVE_IDS)) else \
            m.replace("-tera", "")
        chk.move_all[name] += 1
        chk.move_bad[name] += bool(cats or hp_bad)
    if (cats or hp_bad) and SHOW["n"] > 0:
        SHOW["n"] -= 1
        show_mismatch(label, s1, s2, cats or ["hp"], before, true, res, br, SHOW["events"] or [])
    chk.c["fidelity: transitions"] += 1
    chk.c["fidelity: branches (total)"] += len(branches)
    chk.eq("fidelity: categorical match (faints, status, actives, boosts, hazards, field)", not cats,
           (label, s1, s2, cats))
    for c in cats:
        chk.c[f"fidelity: mismatch in {c.split(' ', 1)[-1] if ' ' in c else c}"] += 1
    if not cats:
        chk.eq("fidelity: HP within the damage-roll spread", not hp_bad, (label, s1, s2, hp_bad, round(worst, 3)))
        chk.c["fidelity: HP within 1% of max HP (all Pokemon)"] += worst <= 0.0101
        chk.hp_err.append(worst)


# ---- 3. options -----------------------------------------------------------------------------------
def options_match(state, battle, mask):
    """(ok, detail, known) where known names a known difference: Showdown's legal set itself isn't
    what it will accept (switches while "maybeTrapped" by a hidden trapping ability), or a case the
    engine can't represent (Struggle; Terastallization during a locked move: one legal action)."""
    r = pe.monte_carlo_tree_search(state, iterations=100)
    opts = {x.move_choice for x in r.side_one}
    acts = {option_to_action(o, battle, state) for o in opts}
    legal = set(int(a) for a in np.flatnonzero(mask))
    act = ((battle.last_request or {}).get("active") or [{}])[0]
    req_ids = [m.get("id") for m in act.get("moves", [])]
    if opts == {"none"} or opts == {"No Move"}:
        return len(legal) == 1, (sorted(opts), sorted(legal)), None
    if battle.reviving:
        return False, None, "Revival Blessing: choosing a fainted Pokemon (not searched)"
    if acts != legal:
        if act.get("maybeTrapped") and acts == {a for a in legal if a < 8}:
            return False, None, "maybeTrapped: engine traps (hidden trapping ability), mask allows switches"
        if req_ids == ["struggle"] and acts == legal - {0}:
            return False, None, "Struggle: no engine option for it"
        if len(legal) == 1 and acts - {a + 4 for a in legal} == legal:
            return False, None, "one legal action (locked move); engine also offers it with tera"
    s1, s2 = state.side_one, state.side_two
    a1, a2 = s1.pokemon[int(s1.active_index)], s2.pokemon[int(s2.active_index)]
    act = ((battle.last_request or {}).get("active") or [{}])[0]
    ctx = (f"vols={sorted(s1.volatile_statuses)} trapped={s1.force_trapped} last={s1.last_used_move} "
           f"own={a1.id}/{a1.types}/{a1.item} opp={a2.id}/{a2.ability} engine moves="
           f"{[(m.id, m.pp, m.disabled) for m in a1.moves]} request={act}")
    return acts == legal, (sorted(opts), sorted(acts), sorted(legal), ctx), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, help="print this many fidelity mismatches in detail")
    args = ap.parse_args()
    SHOW["n"] = args.show
    rng = random.Random(args.seed)
    random.seed(args.seed)
    prior = get_prior()
    uid = uuid.uuid4().hex[:5]
    env = RLEnvV20(server_configuration=server_config(args.port),
                   account_configuration1=AccountConfiguration.generate(f"SsA{uid}", rand=True),
                   account_configuration2=AccountConfiguration.generate(f"SsB{uid}", rand=True))
    sh = SimpleHeuristicsPlayer(battle_format="gen9randombattle", start_listening=False)
    chk = Checker()
    ill = Checker()
    n_dec = 0
    t_build = 0.0
    relaxed = Counter()
    t_start = time.time()
    for g in range(args.games):
        env.reset()
        use_sh = g % 2 == 1
        illusion = illusion_game(env.battle1, env.battle2)
        C = ill if illusion else chk
        C.c["games"] += 1
        prev = None        # (true state from p1's view, p1 option, p2 option)
        pending = {}       # side -> (turn, option) of a choice not yet carried out
        while not env.battle1.finished:
            b1, b2 = env.battle1, env.battle2
            try:
                true_now, _ = build_true_state(b1, b2)
            except BaseException:  # noqa: BLE001
                C.eq("true state built", False, traceback.format_exc()[-300:])
                true_now = None
            if prev is not None and true_now is not None:
                evs = all_events(b1)
                SHOW["events"] = evs[prev[3]:]
                # no events: the choice was rejected (e.g. a switch while trapped) and asked again
                if any(len(ev) > 1 and ev[1] for ev in evs[prev[3]:]):
                    fidelity(C, prev[0], prev[1], prev[2], true_now, f"g{g} t{b1.turn}")
                else:
                    C.c["fidelity: skipped (choice rejected, no events)"] += 1
            to_move = {1: env.agent1_to_move and not b1.finished, 2: env.agent2_to_move and not b2.finished}
            acts, opts = {}, {}
            for side, agent, b, other, is_sh in ((1, env.agent1.username, b1, b2, False),
                                                 (2, env.agent2.username, b2, b1, use_sh)):
                if not to_move[side]:
                    acts[agent] = 0
                    continue
                mask = action_mask(b)
                n_dec += 1
                try:
                    t0 = time.perf_counter()
                    obs = observations(b, tracker(b))
                    sets, rel = sample_opponent(prior, obs, rng)
                    relaxed.update(rel)
                    st, info = build_state(b, sets)
                    ts, tinfo = build_true_state(b, other)
                    t_build += time.perf_counter() - t0
                    compare_observer(C, st, info, ts, tinfo, b, sets)
                    for name, state_ in (("true", ts), ("observer", st)):
                        ok, detail, known = options_match(state_, b, mask)
                        if known:
                            C.c[f"options ({name} state), known difference: {known}"] += 1
                        else:
                            C.eq(f"options: {name} state == legal actions", ok,
                                 (f"g{g} t{b.turn} {b.player_role}", detail))
                except BaseException:  # noqa: BLE001
                    C.eq("converter ran", False, traceback.format_exc()[-400:])
                    st = ts = None
                order = sh.choose_move(b) if is_sh else action_to_order(int(rng.choice(np.flatnonzero(mask))), b)
                acts[agent] = order
                if true_now is not None:
                    o = order_to_option(order, b, true_now, "one" if side == 1 else "two")
                    if o is not None and (o in ("struggle", "recharge") or
                                          (b.active_pokemon is not None and b.active_pokemon.must_recharge)):
                        o = "none"
                    opts[side] = o
            # what each side does this step in the engine: its choice now, a choice from earlier this
            # turn it hasn't carried out yet (the other side of a U-turn), or nothing
            moves = {}
            for side, b in ((1, b1), (2, b2)):
                role = b.player_role
                if side in opts and opts[side] is not None:
                    moves[side] = opts[side]
                    pending[side] = (b.turn, opts[side])
                else:
                    p = pending.get(side)
                    tr = tracker(b1)
                    if p is not None and p[0] == b1.turn and tr.acted.get(role) != b1.turn:
                        moves[side] = p[1]
                    else:
                        moves[side] = "none"
            for side in (1, 2):
                m = moves[side]
                if m.startswith("switch "):
                    moves[side] = m[7:]
            prev = (true_now, moves[1], moves[2], len(all_events(b1))) if true_now is not None and \
                all(opts.get(s) is not None or not to_move[s] for s in (1, 2)) else None
            env.step(acts)
        if (g + 1) % 10 == 0:
            print(f"  {g + 1}/{args.games} games, {n_dec} decisions, {time.time() - t_start:.0f} s", flush=True)
    env.close()

    for name, C in (("main", chk), ("illusion games", ill)):
        print(f"\n===== {name}: {C.c['games']} games")
        for k in sorted(C.c):
            if k != "games":
                print(f"  {k:75s} {C.c[k]:8d}")
        if C.move_bad:
            top = sorted(C.move_bad, key=lambda m: -C.move_bad[m])[:15]
            print("  choices in mismatched transitions (mismatched / all):",
                  ", ".join(f"{m} {C.move_bad[m]}/{C.move_all[m]}" for m in top))
        if C.hp_err:
            e = np.array(C.hp_err)
            print(f"  fidelity HP error (worst Pokemon, fraction of max HP) on categorical matches: median "
                  f"{np.median(e):.4f}, 90th pct {np.percentile(e, 90):.4f}, 99th pct {np.percentile(e, 99):.4f}")
    print(f"\n{n_dec} decisions; build (sample + observer state + true state) {1000 * t_build / max(n_dec, 1):.2f} ms; "
          f"relaxed constraints per revealed Pokemon: {dict(relaxed)}")
    bad = {k: v for k, v in chk.c.items() if "MISMATCH" in k}
    if bad:
        print("\nFirst examples of each mismatch (main):")
        for name, exs in chk.examples.items():
            for ex in exs:
                print(f"  {name}: {ex}")


if __name__ == "__main__":
    main()
