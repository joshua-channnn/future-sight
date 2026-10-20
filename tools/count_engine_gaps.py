"""How often the mechanics poke-engine gets wrong come up in the search bot's own games.

The fidelity check (agent_docs/step6.md, tools/check_search_state.py) found engine mechanics that
differ from Showdown's. Over the search self-play shards (training/gen_search_games.py: the true
engine state of every decision, the decider as side one), this counts for each one the decisions
whose situation it is (the search's simulation goes wrong there), the games with at least one, and
the decisions where the decider plays the move or switch involved.

    python3 -m tools.count_engine_gaps datasets/search_selfplay datasets/search_selfplay_r1

A move action is the i-th of the active's moves that aren't disabled (search/state.py's
option_to_action uses battle.available_moves); decisions where those don't match the legal move
actions (Struggle, a locked move) count as arising but never as played.
"""
import argparse
import bisect
import glob
import gzip
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

MOVE0, TERA0, SWITCH0 = 0, 4, 8

# choices.rs MultiHitMove::TwoToFiveHits: always 3 hits (Skill Link 5, Loaded Dice 4)
TWO_TO_FIVE = {"ARMTHRUST", "BARRAGE", "BONERUSH", "BULLETSEED", "COMETPUNCH", "DOUBLESLAP",
               "FURYATTACK", "FURYSWIPES", "ICICLESPEAR", "PINMISSILE", "ROCKBLAST", "SCALESHOT",
               "SPIKECANNON", "TAILSLAP", "WATERSHURIKEN"}
APPROX_MULTI = {"POPULATIONBOMB", "TRIPLEAXEL", "TRIPLEKICK"}   # per-hit accuracy approximated
EXPLODE = {"EXPLOSION", "SELFDESTRUCT", "MISTYEXPLOSION"}
UNMODELED_MOVES = {"TAKEHEART", "BEATUP", "ARMORCANNON"}   # a no-op, 1 hit, no self-drop
DRAIN = {"ABSORB", "MEGADRAIN", "GIGADRAIN", "DRAININGKISS", "DRAINPUNCH", "LEECHLIFE", "HORNLEECH",
         "PARABOLICCHARGE", "OBLIVIONWING", "BITTERBLADE", "MATCHAGOTCHA", "STRENGTHSAP"}
WEATHER = {"DROUGHT", "DRIZZLE", "SANDSTREAM", "SNOWWARNING", "ORICHALCUMPULSE"}
TERRAIN = {"ELECTRICSURGE", "GRASSYSURGE", "PSYCHICSURGE", "MISTYSURGE", "HADRONENGINE"}
FOE_READERS = {"INTIMIDATE", "DOWNLOAD", "TRACE"}   # act on the foe that's in when they activate

GAPS = [
    ("multi3", "2-5 hit moves always hit 3 times (no Skill Link / Loaded Dice)"),
    ("dice", "Loaded Dice: always 4 hits (4-5 in Showdown)"),
    ("approx", "Population Bomb / Triple Axel / Triple Kick: hits approximated"),
    ("rest", "Rest at full HP with a status: cures it in the engine, fails in Showdown"),
    ("sflo", "Sheer Force + Life Orb: recoil the engine charges"),
    ("noise", "Heal Block (Psychic Noise) doesn't stop Leftovers / drain: a foe that heals so"),
    ("boom", "Explosion into a Ghost doesn't faint the user"),
    ("unmod_move", "Take Heart / Beat Up / Armor Cannon not modeled"),
    ("toxicchain", "Toxic Chain (modeled as a 30% secondary: edge cases only)"),
    ("intrepid", "Intrepid Sword again on a later switch-in (once per battle in gen9)"),
    ("residual", "burn/poison before Leftovers: a holder at <= 1/8 HP"),
    ("dswitch", "both sides switch at once (side one always first)"),
    ("dswitch_weather", "  ... both bring a weather setter (the later one's stays)"),
    ("dswitch_terrain", "  ... both bring a terrain setter"),
    ("dswitch_reader", "  ... either brings Intimidate / Download / Trace"),
    ("illusion", "(observation) Illusion: a Zoroark on either team"),
    ("transform", "(observation) Transform / Imposter on either active"),
]


def _dex():
    from poke_env.data import GenData
    return GenData.from_gen(9).moves


_MOVES = None


def has_secondary(move_id):
    """Sheer Force removes this move's secondary effects (and with them Life Orb's recoil)."""
    global _MOVES
    if _MOVES is None:
        _MOVES = _dex()
    m = _MOVES.get(move_id.lower(), {})
    return m.get("category") != "Status" and bool(m.get("secondary") or m.get("secondaries")
                                                  or m.get("hasSheerForce"))


def is_ghost(p):
    return p.tera_type == "GHOST" if p.terastallized else "GHOST" in p.types


def move_played(side, mask, action):
    """The move id the decider played, or None (a switch, or the slots don't match)."""
    if action >= SWITCH0:
        return None
    act = side.pokemon[int(side.active_index)]
    avail = [m.id for m in act.moves if not m.disabled]
    if len(avail) != int(mask[MOVE0:MOVE0 + 4].sum()):
        return None
    i = action - (TERA0 if action >= TERA0 else MOVE0)
    return avail[i] if i < len(avail) else None


def shard(path):
    import poke_engine as pe
    d = np.load(path)
    lines = gzip.open(path[:-4] + ".states.gz", "rt").read().split("\n")[:-1]
    n = len(d["action"])
    assert len(lines) == n, (path, len(lines), n)
    game, side_of, step, action, mask = d["game"], d["side"], d["step"], d["action"].astype(int), d["mask"]
    arises, played = Counter(), Counter()
    games = defaultdict(set)
    states = [pe.State.from_string(s) for s in lines]

    # Intrepid Sword: the slots each side has had active before each step (from its own decisions)
    hist = defaultdict(list)   # (game, side) -> [(step, active_index)]
    for r in range(n):
        hist[(int(game[r]), int(side_of[r]))].append((int(step[r]), int(states[r].side_one.active_index)))
    been = {}
    for k, seq in hist.items():
        seq.sort()
        acc, cum = set(), []
        for _, a in seq:
            cum.append(frozenset(acc))
            acc.add(a)
        been[k] = ([s for s, _ in seq], cum)

    def been_active(g, sd, t):
        steps, cum = been.get((g, sd), ([], []))
        i = bisect.bisect_left(steps, t)
        return cum[i] if i < len(cum) else (cum[-1] if cum else frozenset())

    by_step = defaultdict(list)
    for r in range(n):
        g = int(game[r])
        by_step[(g, int(step[r]))].append(r)
        st = states[r]
        s1, s2 = st.side_one, st.side_two
        a1, a2 = s1.pokemon[int(s1.active_index)], s2.pokemon[int(s2.active_index)]
        mv = move_played(s1, mask[r], int(action[r]))
        hit = set()
        play = set()
        for p, foe, own in ((a1, a2, True), (a2, a1, False)):
            moves = {m.id for m in p.moves}
            sl, dice = p.ability == "SKILLLINK", p.item == "LOADEDDICE"
            if moves & TWO_TO_FIVE and not sl:
                hit.add("dice" if dice else "multi3")
                if own and mv in TWO_TO_FIVE:
                    play.add("dice" if dice else "multi3")
            if moves & APPROX_MULTI:
                hit.add("approx")
                if own and mv in APPROX_MULTI:
                    play.add("approx")
            if "REST" in moves and p.hp == p.maxhp and p.status not in ("NONE", "SLEEP"):
                hit.add("rest")
                if own and mv == "REST":
                    play.add("rest")
            if p.ability == "SHEERFORCE" and p.item == "LIFEORB":
                hit.add("sflo")
                if own and mv and has_secondary(mv):
                    play.add("sflo")
            if moves & {"PSYCHICNOISE", "HEALBLOCK"} and (foe.item in ("LEFTOVERS", "BLACKSLUDGE")
                                                           or {m.id for m in foe.moves} & DRAIN):
                hit.add("noise")
                if own and mv in ("PSYCHICNOISE", "HEALBLOCK"):
                    play.add("noise")
            if moves & EXPLODE and is_ghost(foe):
                hit.add("boom")
                if own and mv in EXPLODE:
                    play.add("boom")
            if moves & UNMODELED_MOVES:
                hit.add("unmod_move")
                if own and mv in UNMODELED_MOVES:
                    play.add("unmod_move")
            if p.ability == "TOXICCHAIN":
                hit.add("toxicchain")
                if own and mv:
                    play.add("toxicchain")
            if p.status in ("BURN", "POISON", "TOXIC") and p.item in ("LEFTOVERS", "BLACKSLUDGE") \
                    and 0 < p.hp <= p.maxhp / 8:
                hit.add("residual")
            if "TRANSFORM" in moves or p.ability == "IMPOSTER":
                hit.add("transform")
        # Intrepid Sword: alive on a bench, active before (the engine boosts it again on re-entry)
        for sd, side, own in ((int(side_of[r]), s1, True), (1 - int(side_of[r]), s2, False)):
            for k, p in enumerate(side.pokemon):
                if p.ability == "INTREPIDSWORD" and p.hp > 0 and k != int(side.active_index) \
                        and k in been_active(g, sd, int(step[r])):
                    hit.add("intrepid")
                    if own and int(action[r]) == SWITCH0 + k:
                        play.add("intrepid")
        if any(p.ability == "ILLUSION" for side in (s1, s2) for p in side.pokemon):
            hit.add("illusion")
        for x in hit:
            arises[x] += 1
            games[x].add(g)
        for x in play:
            played[x] += 1

    # both sides switching at the same step
    pairs = Counter()
    for (g, t), rs in by_step.items():
        if len(rs) != 2 or not all(action[r] >= SWITCH0 for r in rs):
            continue
        forced = all(states[r].side_one.force_switch for r in rs)
        incoming = [states[r].side_one.pokemon[int(action[r]) - SWITCH0].ability for r in rs]
        kinds = {"dswitch"}
        if all(a in WEATHER for a in incoming) and incoming[0] != incoming[1]:
            kinds.add("dswitch_weather")
        if all(a in TERRAIN for a in incoming) and incoming[0] != incoming[1]:
            kinds.add("dswitch_terrain")
        if any(a in FOE_READERS for a in incoming):
            kinds.add("dswitch_reader")
        for x in kinds:
            arises[x] += 2   # both sides' decisions
            played[x] += 2
            games[x].add(g)
        pairs["forced" if forced else "chosen"] += 1
    return {"n": n, "games": {int(x) for x in game}, "arises": arises, "played": played,
            "gap_games": {k: v for k, v in games.items()}, "pairs": pairs, "key": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+", help="shard directories or .npz files")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    files = []
    for p in args.path:
        files += sorted(glob.glob(os.path.join(p, "*.npz"))) if os.path.isdir(p) else [p]
    if not files:
        raise SystemExit("no shards found")
    n, games = 0, 0
    arises, played, pairs = Counter(), Counter(), Counter()
    gap_games = Counter()
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(shard, files):
            n += res["n"]
            games += len(res["games"])   # game ids are per shard
            arises.update(res["arises"])
            played.update(res["played"])
            pairs.update(res["pairs"])
            for k, v in res["gap_games"].items():
                gap_games[k] += len(v)
    print(f"{len(files)} shards, {games:,} games, {n:,} decisions; both sides switching at once: "
          f"{pairs['chosen']:,} chosen, {pairs['forced']:,} after both fainted")
    print(f"{'':76} {'decisions':>17} {'games':>13} {'played':>15}")
    for k, name in GAPS:
        a, gg, p = arises[k], gap_games[k], played[k]
        print(f"{name:76} {a:>9,} {100 * a / n:6.2f}% {gg:>6,} {100 * gg / games:5.1f}% "
              f"{p:>8,} {100 * p / n:5.2f}%")


if __name__ == "__main__":
    main()
