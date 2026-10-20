"""Where recorded ladder games were lost (players/ladder_parallel.py --record DIR).

For each lost game (every game with --all): the search's win chance at each of its decisions and
the turns where it fell the most, with what happened in them (the game's log) and what the search
chose and expected before and after. Then, over all games: how often a lead was lost (a win chance
of 70%+ at some point in a lost game), the search's calibration (its win chance against how the
games ended), and how well it predicted the opponent: the share of the search it gave to what the
opponent did (its side-two visit shares; a switch to a Pokemon not yet seen gets the shares of all
such switches).

    python3 -m tools.review_ladder_games logs/ladder_games_20261016 [--top 3] [--all] [--game TAG]
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

NOISE = ("|t:|", "|j|", "|l|", "|n|", "|c|", "|chat|", "|raw|", "|html|", "|uhtml", "|inactive",
         "|-hint|", "|-message|", "|upkeep", "|badge|", "|timer|")


def load(folder):
    """{battle: {"decisions": [...], "result": {...} or None, "log": [lines]}}"""
    games = defaultdict(lambda: {"decisions": [], "result": None, "log": []})
    with open(os.path.join(folder, "decisions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            g = games[r["battle"]]
            if "won" in r:
                g["result"] = r
            else:
                g["decisions"].append(r)
    for path in glob.glob(os.path.join(folder, "*.html")):
        me, tag = os.path.basename(path)[:-5].split(" - ", 1)
        if tag in games:
            games[tag]["me"] = me
            log = [x for x in open(path, encoding="utf-8").read().split("\n")
                   if x.startswith("|") and x != "|" and not x.startswith(NOISE)]
            games[tag]["log"] = log
            # |player|p1|<name>|<avatar>|<rating>: poke-env doesn't keep the public server's ratings
            games[tag]["ratings"] = {f[3]: f[5] for f in (x.split("|") for x in log)
                                     if f[1] == "player" and len(f) > 5 and f[5]}
    return games


def turn_log(log, first, last):
    """The log from the start of turn `first` (after |turn|first) to the start of turn last + 1."""
    out, t = [], 0
    for x in log:
        if x.startswith("|turn|"):
            t = int(x.split("|")[2])
            if t > last:
                break
            continue
        if first <= t <= last:
            out.append(x)
    return out


def to_id(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def opponent_action(lines, opp):
    """What the opponent (role `opp`, e.g. "p2") chose in a turn's log, as the engine's option
    (a move id, "-tera" if it terastallized first, or "switch <species id>"), or None when the log
    doesn't show it (it couldn't move or fainted first, or the move was called or locked)."""
    tera = False
    for x in lines:
        f = x.split("|")
        if len(f) < 3 or not f[2].startswith(opp):
            continue
        if f[1] == "-terastallize":
            tera = True
        elif f[1] == "switch":
            return "switch " + to_id(f[3].split(",")[0])
        elif f[1] == "move":
            if "[from]" in x:
                return None
            return to_id(f[3]) + ("-tera" if tera else "")
        elif f[1] in ("cant", "faint"):
            return None
    return None


def opponent_predictions(games):
    """{kind: [share the search gave to what the opponent did, ...]} over the searched decisions
    (not switch-ins) whose turn's log shows the opponent's choice; kind: move, switch (to a Pokemon
    seen before), new (to one not seen yet)."""
    out = defaultdict(list)
    top = defaultdict(int)
    for g in games.values():
        names = {f[2]: f[3] for f in (x.split("|") for x in g["log"]) if f[1] == "player" and len(f) > 3}
        opp = next((r for r, n in names.items() if n != g.get("me")), None)
        if opp is None:
            continue
        for d in g["decisions"]:
            if d["switch_in"]:
                continue
            seen, t = set(), 0
            for x in g["log"]:
                if x.startswith("|turn|"):
                    t = int(x.split("|")[2])
                    if t >= d["turn"]:
                        break
                f = x.split("|")
                if f[1] in ("switch", "drag", "replace") and f[2].startswith(opp):
                    seen.add(to_id(f[3].split(",")[0]))
            act = opponent_action(turn_log(g["log"], d["turn"], d["turn"]), opp)
            if act is None:
                continue
            opts = d["opp_options"]
            if act.startswith("switch ") and act[7:] not in seen:
                kind = "new"
                p = sum(v for k, v in opts.items() if k.startswith("switch ") and k[7:] not in seen)
            else:
                kind = "switch" if act.startswith("switch ") else "move"
                p = opts.get(act, 0.0)
                top[kind] += bool(opts) and max(opts, key=opts.get) == act
            out[kind].append(p)
    return out, top


def luck(lines, me):
    """Chance events against `me` (e.g. "p2") in a turn's log: a crit on its Pokemon, its move
    missing, full paralysis, a flinch, a confusion self-hit or a freeze."""
    out = []
    for x in lines:
        f = x.split("|")
        if len(f) < 3:
            continue
        if f[1] == "-crit" and f[2].startswith(me):
            out.append("crit taken")
        elif f[1] == "-miss" and f[2].startswith(me):
            out.append("missed")
        elif f[1] == "cant" and f[2].startswith(me) and len(f) > 3 and f[3] in ("par", "flinch", "frz"):
            out.append({"par": "fully paralyzed", "flinch": "flinched", "frz": "frozen"}[f[3]])
        elif f[1] == "-damage" and f[2].startswith(me) and "[from] confusion" in x:
            out.append("hit itself in confusion")
    return out


def my_role(g):
    names = {f[2]: f[3] for f in (x.split("|") for x in g["log"]) if f[1] == "player" and len(f) > 3}
    return next((r for r, n in names.items() if n == g.get("me")), None)


def drops(decisions):
    """(fall, i): the win chance's fall from decision i to i + 1, largest first."""
    w = [d["win"] for d in decisions]
    return sorted(((w[i] - w[i + 1], i) for i in range(len(w) - 1)
                   if w[i] is not None and w[i + 1] is not None), reverse=True)


def show(tag, g, top):
    res, ds = g["result"] or {}, g["decisions"]
    outcome = {True: "WIN", False: "LOSS", None: "?"}[res.get("won")]
    rating = res.get("opponent_rating") or g.get("ratings", {}).get(res.get("opponent"))
    print(f"\n=== {tag} vs {res.get('opponent')} ({rating}): {outcome} "
          f"in {res.get('turns')} turns; {len(ds)} searched decisions")
    print("  win chance: " + " ".join(f"{d['turn']}{'s' if d['switch_in'] else ''}:"
                                      f"{round(100 * d['win']) if d['win'] is not None else '?'}"
                                      for d in ds))
    for k, (fall, i) in enumerate(drops(ds)[:top]):
        a, b = ds[i], ds[i + 1]
        print(f"\n  {k + 1}. turn {a['turn']} -> {b['turn']}: {100 * a['win']:.0f}% -> {100 * b['win']:.0f}%")
        lines = turn_log(g["log"], a["turn"], b["turn"] - (0 if b["switch_in"] else 1))
        bad = luck(lines, my_role(g) or "?")
        print(f"     before: {a['note']}")
        for x in lines:
            print(f"       {x}")
        print(f"     after: {b['note']}" + (f"\n     chance against it: {', '.join(bad)}" if bad else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--top", type=int, default=3, help="largest falls to show per game")
    ap.add_argument("--all", action="store_true", help="wins too")
    ap.add_argument("--game", default=None, help="only this battle tag")
    args = ap.parse_args()
    games = load(args.folder)
    done = {t: g for t, g in games.items() if g["result"] and g["decisions"]}
    for tag, g in sorted(done.items()):
        if (args.game and tag != args.game) or (not args.all and not args.game and g["result"]["won"]):
            continue
        show(tag, g, args.top)
    if args.game:
        return
    lost = [g for g in done.values() if not g["result"]["won"]]
    led = [g for g in lost if max((d["win"] or 0) for d in g["decisions"]) >= 0.7]
    never = [g for g in lost if max((d["win"] or 0) for d in g["decisions"]) < 0.5]
    print(f"\n{len(done)} finished games, {len(done) - len(lost)} won, {len(lost)} lost. Of the losses, "
          f"{len(led)} had a win chance of 70%+ at some point; {len(never)} never reached 50%.")
    unlucky = 0
    for g in lost:
        dd = drops(g["decisions"])
        if dd:
            a, b = g["decisions"][dd[0][1]], g["decisions"][dd[0][1] + 1]
            unlucky += bool(luck(turn_log(g["log"], a["turn"], b["turn"] - (0 if b["switch_in"] else 1)),
                                 my_role(g) or "?"))
    if lost:
        print(f"In {unlucky} of the {len(lost)} losses, the largest fall came with a chance event against "
              f"the bot (a crit, a miss, full paralysis, a flinch, a confusion self-hit or a freeze).")
    bins = defaultdict(lambda: [0, 0])
    for g in done.values():
        for d in g["decisions"]:
            if d["win"] is not None:
                b = min(int(d["win"] * 10), 9)
                bins[b][0] += 1
                bins[b][1] += bool(g["result"]["won"])
    print("calibration (the search's win chance -> the share of those decisions' games won):")
    for b in sorted(bins):
        n, w = bins[b]
        print(f"  {10 * b:3d}-{10 * b + 10:3d}%: {n:5d} decisions, {100 * w / n:5.1f}% won")
    preds, top = opponent_predictions(done)
    allp = [p for v in preds.values() for p in v]
    if allp:
        print(f"opponent predictions over {len(allp)} decisions: mean share given to what the opponent "
              f"did {100 * sum(allp) / len(allp):.1f}%, mean log share {sum(math.log(max(p, 0.005)) for p in allp) / len(allp):.2f}")
        names = {"move": "a move", "switch": "a switch to a Pokemon seen before",
                 "new": "a switch to one not seen yet"}
        for k in ("move", "switch", "new"):
            v = preds.get(k, [])
            if v:
                hit = f"; it was the search's most likely option {100 * top[k] / len(v):.0f}% of the time" \
                    if k != "new" else ""
                print(f"  {names[k]}: {len(v)} ({100 * len(v) / len(allp):.0f}%), mean share "
                      f"{100 * sum(v) / len(v):.1f}%{hit}")


if __name__ == "__main__":
    main()
