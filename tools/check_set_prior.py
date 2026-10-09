"""Milestone 2 of agent_docs/step6_brief.md: checks of the set prior (data/gen9_randbats_set_counts.json)
and the sampler that draws from it (search/sampling.py).

1. Coverage: every species in sets.json has sets; poke-engine knows every species / forme, move,
   ability and item name in the prior (unknown names silently become NONE in the engine).
2. Set frequencies: a fresh holdout run of the generator (different seeds) against the prior, per
   species, as total variation distance (TV) between the holdout's set frequencies and the prior's,
   next to the TV of the same number of sets drawn by the sampler (pure sampling noise). If the
   prior reproduces the generator, the two are about equal. Separately for leads (the team's first
   Pokemon) and the rest.
3. Filtering: for holdout sets, observations drawn from the set itself (level, some of its moves,
   item, ability, tera type) must keep it among the candidates without relaxing anything.
4. Teams: species frequencies of teams built by fill_team against the holdout's, team-level
   statistics, and, for the most common species, which species share a team with it
   (fill_team given that species vs holdout teams containing it vs ignoring it).

    python3 -m tools.check_set_prior --teams 200000
"""
import argparse
import math
import random
import tempfile
import time
from collections import Counter, defaultdict

import poke_engine as pe

from search.sampling import SetPrior, _Weighted
from tools.build_set_prior import SHOWDOWN, run_workers

ENGINE_UNKNOWN_ITEMS_OK = {"leppaberry", "lightclay", "scopelens"}


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def normalize(c):
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()} if tot else {}


def engine_names(prior):
    """Names in the prior that poke-engine doesn't know (they round-trip to NONE)."""
    def back(**kw):
        st = pe.State(side_one=pe.Side(pokemon=[pe.Pokemon(**kw)]), side_two=pe.Side(pokemon=[pe.Pokemon()]))
        return pe.State.from_string(st.to_string()).side_one.pokemon[0]

    species = set(prior.sets) | set(prior._forme_keys)
    sets = [s for ss in prior.sets.values() for s in ss]
    moves = {m for s in sets for m in s.moves}
    items = {s.item for s in sets if s.item}
    abilities = {s.ability for s in sets}
    return {
        "species": [x for x in sorted(species) if back(id=x).id.lower() != x],
        "moves": [x for x in sorted(moves) if back(moves=[pe.Move(id=x)]).moves[0].id.lower() != x],
        "abilities": [x for x in sorted(abilities) if back(ability=x).ability.lower() != x],
        "items": [x for x in sorted(items) if back(item=x).item.lower() != x],
    }


def set_id(s):
    return (s.level, s.item, s.ability, s.moves, s.tera, s.evs, s.ivs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=200_000, help="holdout teams")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--cooc-species", type=int, default=60, help="species for the co-occurrence check")
    args = ap.parse_args()
    rng = random.Random(0)
    prior = SetPrior()
    import json
    sets_json = json.load(open("data/gen9_randbats_sets.json"))
    problems = []

    # ---- 1. coverage ---------------------------------------------------------------------------
    missing = sorted(k for k in sets_json if not prior.sets.get(k))
    print(f"1. coverage: {len(prior.sets)} species in the prior, sets.json has {len(sets_json)}; "
          f"missing: {missing or 'none'}")
    if missing:
        problems.append("species missing from the prior")
    unknown = engine_names(prior)
    print(f"   names poke-engine doesn't know: {unknown}")
    if unknown["species"] or unknown["moves"] or unknown["abilities"] or \
            set(unknown["items"]) - ENGINE_UNKNOWN_ITEMS_OK:
        problems.append("names unknown to poke-engine")

    # ---- holdout run ---------------------------------------------------------------------------
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        parts, teams = run_workers(SHOWDOWN, args.teams, args.workers, args.seed, tmp, with_teams=True)
    hold = defaultdict(Counter)       # key -> set id -> non-lead count
    hold_lead = defaultdict(Counter)  # key -> set id -> lead count
    for part in parts:
        for key, by_set in part["sets"].items():
            for rec, (c, lead) in by_set.items():
                r = json.loads(rec)
                sid = (r[0], r[1], r[2], tuple(r[3]), r[4], tuple(r[5]), tuple(r[6]))
                hold[key][sid] += c - lead
                hold_lead[key][sid] += lead
    print(f"\nholdout: {len(teams):,} teams (seeds {args.seed}..{args.seed + args.workers - 1}) "
          f"in {time.time() - t0:.0f} s")

    # ---- 2. set frequencies --------------------------------------------------------------------
    for label, held, lead in (("non-lead", hold, False), ("lead", hold_lead, True)):
        tot_h = tot_s = n_tot = 0.0
        worst = []
        unseen_mass = 0.0
        for key, counts in held.items():
            n = sum(counts.values())
            if n < 50 or key not in prior.sets:
                continue
            sets = prior.sets[key]
            w = SetPrior._weights(sets, lead)
            p = normalize({set_id(s): wi for s, wi in zip(sets, w)})
            q_hold = normalize(counts)
            unseen_mass += n * sum(v for k, v in q_hold.items() if k not in p)
            d = _Weighted(sets, w)
            q_samp = normalize(Counter(set_id(d.draw(rng)) for _ in range(n)))
            th, ts = tv(q_hold, p), tv(q_samp, p)
            tot_h += n * th
            tot_s += n * ts
            n_tot += n
            worst.append((th - ts, key, n, th, ts))
        worst.sort(reverse=True)
        print(f"\n2. {label} sets: {int(n_tot):,} holdout sets; count-weighted TV holdout vs prior "
              f"{tot_h / n_tot:.4f}, sampler draws vs prior {tot_s / n_tot:.4f} (noise); "
              f"holdout sets the prior never saw: {unseen_mass / n_tot:.5f} of the mass")
        print("   largest excess over noise:", ", ".join(
            f"{k} (n={n}, {th:.3f} vs {ts:.3f})" for _, k, n, th, ts in worst[:5]))
        if tot_h / n_tot > 1.25 * tot_s / n_tot + 0.005:
            problems.append(f"{label} set frequencies differ from the generator's beyond noise")

    # ---- 3. filtering --------------------------------------------------------------------------
    n_checks = kept = relaxed_any = 0
    mass = 0.0
    keys = [k for k in hold if k in prior.sets]
    for _ in range(20000):
        key = rng.choice(keys)
        sid = rng.choice(list(hold[key]))
        level, item, ability, moves, tera = sid[:5]
        shown = rng.sample(moves, rng.randint(0, len(moves)))
        obs = dict(level=level, moves=shown,
                   item=item if rng.random() < 0.5 else None,
                   ability=ability if rng.random() < 0.5 else None,
                   tera=tera if rng.random() < 0.3 else None)
        cands, w, relaxed = prior.candidates(key, **obs)
        n_checks += 1
        ids = [set_id(c) for c in cands]
        relaxed_any += relaxed > 0
        if sid in ids:
            kept += 1
            mass += w[ids.index(sid)] / max(sum(w), 1)
    print(f"\n3. filtering: {n_checks:,} holdout sets with observations drawn from themselves: true set among "
          f"the candidates {kept:,} times, constraints relaxed {relaxed_any} times; mean candidate "
          f"probability of the true set {mass / n_checks:.3f}")
    if kept != n_checks or relaxed_any:
        problems.append("filtering lost true sets")

    # ---- 4. teams ------------------------------------------------------------------------------
    hold_species = normalize(Counter(k for team in teams for k, _, _ in team))
    prior_species = normalize({k: sum(s.count for s in ss) for k, ss in prior.sets.items()})
    t0 = time.time()
    sampled = [prior.fill_team([], rng) for _ in range(len(teams))]
    dt = time.time() - t0
    samp_species = normalize(Counter(s.key for team in sampled for s in team))
    print(f"\n4. teams: {len(sampled):,} teams from fill_team ({1e6 * dt / len(sampled):.0f} us per team); "
          f"species TV vs the prior's frequencies: fill_team {tv(samp_species, prior_species):.4f}, "
          f"holdout {tv(hold_species, prior_species):.4f} (noise)")

    def team_stats(team_keys):
        types = Counter(t for k in team_keys for t in prior.info[k]["types"])
        weak = Counter(t for k in team_keys for t in prior.info[k]["weak"])
        return (max(types.values()), max(weak.values()),
                sum(prior.info[k]["level"] == 100 for k in team_keys))

    for name, idx in (("max Pokemon of one type", 0), ("max weak to one type", 1), ("level-100 Pokemon", 2)):
        h = normalize(Counter(team_stats([k for k, _, _ in t])[idx] for t in teams))
        s = normalize(Counter(team_stats([x.key for x in t])[idx] for t in sampled))
        print(f"   {name:24s} holdout {dict(sorted((k, round(v, 3)) for k, v in h.items()))}  "
              f"fill_team {dict(sorted((k, round(v, 3)) for k, v in s.items()))}")

    # which species share a team with X: fill_team given X vs ignoring X, scored by the log-likelihood
    # of the holdout's teams containing X (both smoothed the same way)
    with_x = defaultdict(Counter)
    for team in teams:
        ks = [k for k, _, _ in team]
        for k in ks:
            with_x[k].update(o for o in ks if o != k)
    common = sorted(with_x, key=lambda k: -sum(with_x[k].values()))[:args.cooc_species]
    ll_fill = ll_marg = n_obs = 0.0
    better = 0
    for x in common:
        n_teams = sum(with_x[x].values()) // 5
        fill = Counter()
        for _ in range(10 * n_teams):
            fill.update(s.key for s in prior.fill_team([prior.draw(x, rng)], rng))
        others = [k for k in prior.sets if prior.info[k]["base"] != prior.info[x]["base"]]
        marg = {k: prior_species[k] for k in others}
        f_tot, m_tot = sum(fill.values()), sum(marg.values())
        lf = lm = 0.0
        for k, c in with_x[x].items():
            lf += c * math.log((fill.get(k, 0) + 0.5) / (f_tot + 0.5 * len(others)))
            lm += c * math.log(marg.get(k, 0) / m_tot)
        ll_fill += lf
        ll_marg += lm
        n_obs += sum(with_x[x].values())
        better += lf > lm
    print(f"   species sharing a team with each of the {len(common)} most common species: holdout "
          f"log-likelihood per teammate, fill_team given it {ll_fill / n_obs:.4f}, prior frequencies "
          f"ignoring it {ll_marg / n_obs:.4f}; fill_team better for {better} of {len(common)}")
    if ll_fill <= ll_marg:
        problems.append("fill_team does no better than ignoring the revealed Pokemon")

    print("\nRESULT:", "all checks passed" if not problems else "PROBLEMS: " + "; ".join(problems))


if __name__ == "__main__":
    main()
