"""How well set priors fit the public server: recent gen9randombattle replays from
replay.pokemonshowdown.com, read from each side by poke-env and search/state.py's tracker as the
search bot reads its own games, give what every opponent Pokemon showed by the end of the game
(search.sampling.observations). Per prior:
  - misses: Pokemon none of whose sets fit what it showed: a species the prior doesn't have, a level
    none of its sets has, or sample_opponent having to drop a constraint (relaxed > 0: the item,
    ability, tera type or a move). Some misses are the log's (a move called by another move, a
    traded item), so even the right prior has a few.
  - log p: on the Pokemon every prior fits, the mean log of the prior's share of the species' sets
    that fit (higher: the prior expected what was shown more often).

    python3 -m tools.check_prior_replays --replays 300 \
        --priors data/gen9_randbats_set_counts.json data/gen9_randbats_set_counts_live.json
"""
import argparse
import json
import logging
import math
import ssl
import time
import urllib.request
from collections import Counter

import certifi
from poke_env.battle import Battle

from search.sampling import SetPrior, observations
from search.state import tracker

SEARCH = "https://replay.pokemonshowdown.com/search.json?format=gen9randombattle"
SKIP = {"win", "tie", "request", "t:", "inactive", "raw", "bigerror", "error", "message", "seed",
        "timer", "badge", "unlink", "hidelines"}
_CTX = ssl.create_default_context(cafile=certifi.where())


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pokemon-rl set prior check"})
    with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
        return r.read().decode()


def replay_ids(n, pause):
    ids, before = [], None
    while len(ids) < n:
        page = json.loads(_get(SEARCH + (f"&before={before}" if before else "")))
        new = [p["id"] for p in page if not p.get("password") and p["id"] not in ids]
        if not new:
            break
        ids += new
        before = page[-1]["uploadtime"]
        time.sleep(pause)
    return ids[:n]


def opponent_observations(log, role):
    """observations() of the side that isn't `role` at the end of the game, and the number of log
    lines poke-env couldn't parse."""
    lines = [line for line in log.split("\n") if line.startswith("|")]
    names = {}
    for line in lines:
        s = line.split("|")
        if len(s) >= 4 and s[1] == "player" and s[3]:
            names.setdefault(s[2], s[3])
    if role not in names:
        return [], 0
    battle = Battle(f"battle-gen9randombattle-{role}", names[role], logging.getLogger("replay"), gen=9)
    bad = 0
    for line in lines:
        s = line.split("|")
        if len(s) < 2 or s[1] in SKIP:
            continue
        try:
            battle.parse_message(s)
        except Exception:
            bad += 1
    return observations(battle, tracker(battle)), bad


def fit(prior, o):
    """(why it misses or None, log share of the fitting sets or None)."""
    keys = prior.keys_for(o["species"])
    if not keys:
        return "species", None
    sets = [s for k in keys for s in prior.sets[k]]
    if all(s.level != o["level"] for s in sets):
        return "level", None
    found = [prior.candidates(k, o["level"], o["moves"], o["item"], o["ability"], o["tera"]) for k in keys]
    relaxed = min((r for c, _, r in found if c), default=4)
    if relaxed:
        return ("", "item", "ability", "tera", "moves")[relaxed], None
    fitting = sum(s.count for c, _, r in found if r == 0 for s in c)
    return None, math.log(fitting / sum(s.count for s in sets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--priors", nargs="+", default=["data/gen9_randbats_set_counts.json"])
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--show", type=int, default=15, help="most common misses to list per prior")
    args = ap.parse_args()
    priors = {p: SetPrior(p) for p in args.priors}
    ids = replay_ids(args.replays, args.pause)
    print(f"{len(ids)} replays")
    misses = {p: Counter() for p in priors}
    examples = {p: Counter() for p in priors}
    logp = {p: [] for p in priors}
    n_mons = bad = 0
    for i, rid in enumerate(ids):
        try:
            log = _get(f"https://replay.pokemonshowdown.com/{rid}.log")
        except Exception as e:
            print(f"  {rid}: {e}")
            continue
        for role in ("p1", "p2"):
            obs, b = opponent_observations(log, role)
            bad += b
            for o in obs:
                n_mons += 1
                res = {p: fit(prior, o) for p, prior in priors.items()}
                for p, (why, lp) in res.items():
                    if why:
                        misses[p][why] += 1
                        examples[p][(o["species"], why)] += 1
                if all(lp is not None for _, lp in res.values()):
                    for p, (_, lp) in res.items():
                        logp[p].append(lp)
        time.sleep(args.pause)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} replays, {n_mons} Pokemon")
    print(f"\n{n_mons} opponent Pokemon from {len(ids)} replays ({bad} log lines poke-env couldn't parse)")
    for p in priors:
        m = misses[p]
        total = sum(m.values())
        lp = logp[p]
        print(f"\n{p}: misses {total} ({total / max(n_mons, 1):.1%}): "
              + ", ".join(f"{k} {v}" for k, v in m.most_common())
              + f"; log p {sum(lp) / max(len(lp), 1):.3f} on the {len(lp)} every prior fits")
        print("  most common: " + ", ".join(f"{s} ({w}) {c}" for (s, w), c in examples[p].most_common(args.show)))


if __name__ == "__main__":
    main()
