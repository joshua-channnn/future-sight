"""Builds data/gen9_randbats_set_counts.json, the set prior for search (agent_docs/step6_brief.md,
milestone 2): how often the local Showdown fork's gen9 random battle generator produces each complete
set (level, item, ability, 4 moves, tera type, EVs, IVs) of each species.

Runs tools/gen_set_counts.js in parallel worker processes (different PRNG seeds) and merges them.
1M teams take ~2 minutes with 8 workers.

    python3 -m tools.build_set_prior --teams 1000000 --workers 8

File layout:
  meta: teams, seeds, sha256 of the fork's gen9 sets.json and teams.ts
  species: {sets.json key: {count, lead, formes: {forme id: count},
                            sets: [[count, lead count, level, item, ability, [moves], tera, [evs], [ivs]], ...]}}
           sets sorted by count, most frequent first; "lead": times generated as the team's first Pokemon
  info: per key the base species, types, weaknesses, level (the generator's team constraints), and
        the generator's base-species pool with its weights (search/sampling.py rebuilds teams from it)
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOWDOWN = ROOT.parent / "pokemon-showdown"
OUT = ROOT / "data" / "gen9_randbats_set_counts.json"


def run_workers(showdown, teams, workers, seed0, tmp, with_teams=False):
    """Runs gen_set_counts.js in `workers` processes with seeds seed0, seed0 + 1, ...; returns their
    outputs (and, with_teams, every generated team as [key, ability, level] lists)."""
    procs, paths = [], []
    per = [teams // workers + (1 if i < teams % workers else 0) for i in range(workers)]
    for i, n in enumerate(per):
        p = Path(tmp) / f"counts_{seed0 + i}.json"
        t = Path(tmp) / f"teams_{seed0 + i}.json"
        paths.append((p, t))
        cmd = ["node", str(ROOT / "tools" / "gen_set_counts.js"), str(showdown), str(n), str(seed0 + i), str(p)]
        procs.append(subprocess.Popen(cmd + ([str(t)] if with_teams else [])))
    for pr in procs:
        if pr.wait() != 0:
            raise RuntimeError("a gen_set_counts.js worker failed")
    parts = [json.loads(p.read_text()) for p, _ in paths]
    if not with_teams:
        return parts
    return parts, [team for _, t in paths for team in json.loads(t.read_text())]


def merge(parts):
    sets = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    formes = defaultdict(lambda: defaultdict(int))
    for part in parts:
        for key, by_set in part["sets"].items():
            for rec, (c, lead) in by_set.items():
                s = sets[key][rec]
                s[0] += c
                s[1] += lead
        for key, fs in part["formes"].items():
            for f, c in fs.items():
                formes[key][f] += c
    species = {}
    for key in sorted(sets):
        rows = sorted(((c, lead, *json.loads(rec)) for rec, (c, lead) in sets[key].items()),
                      key=lambda r: -r[0])
        species[key] = {"count": sum(r[0] for r in rows), "lead": sum(r[1] for r in rows),
                        "formes": dict(sorted(formes[key].items(), key=lambda kv: -kv[1])), "sets": rows}
    return species


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1, help="first worker's seed; seed 1 also writes the info")
    ap.add_argument("--showdown", default=str(SHOWDOWN))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        parts = run_workers(args.showdown, args.teams, args.workers, args.seed, tmp)
    info = next((p["info"] for p in parts if "info" in p), None)
    species = merge(parts)
    gen9 = Path(args.showdown) / "data" / "random-battles" / "gen9"
    out = {
        "meta": {"teams": sum(p["teams"] for p in parts), "seeds": [p["seed"] for p in parts],
                 "sets_json_sha256": sha256(gen9 / "sets.json"),
                 "teams_ts_sha256": sha256(gen9 / "teams.ts")},
        "species": species,
        "info": info,
    }
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    n_sets = sum(len(s["sets"]) for s in species.values())
    print(f"{out['meta']['teams']:,} teams, {len(species)} species, {n_sets:,} distinct sets "
          f"-> {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
