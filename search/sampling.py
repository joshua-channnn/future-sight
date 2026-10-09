"""The opponent set prior and team sampler (agent_docs/step6_brief.md, milestones 2-3).

The prior (data/gen9_randbats_set_counts.json, built by tools/build_set_prior.py) counts the complete
sets the local Showdown fork's gen9 random battle generator produced in 1M teams. A revealed opponent
Pokemon gets a set drawn from the sets of its species consistent with what it has shown (level, moves,
item, ability, tera type), weighted by how often the generator produced each; the opponent's first
Pokemon (its lead) by how often each set was a lead. Unrevealed Pokemon are drawn the way the
generator builds a team: a base species from its pool, rejected when it breaks the generator's team
limits given the Pokemon already on the team, then a set for it.
"""
import bisect
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from poke_env.data import GenData

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRIOR_PATH = DATA_DIR / "gen9_randbats_set_counts.json"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
POKEDEX = GenData.from_gen(9).pokedex
# moves a Pokemon can show without having them in its set
NOT_SET_MOVES = frozenset({"struggle", "recharge"})


@dataclass(frozen=True)
class RandSet:
    key: str          # the sets.json entry it was generated from
    level: int
    item: str         # item id, "" for none
    ability: str
    moves: tuple      # 4 move ids, sorted
    tera: str         # tera type, lower case
    evs: tuple        # hp, atk, def, spa, spd, spe
    ivs: tuple
    count: int        # times the generator produced it in the prior's teams
    lead: int         # of which as the team's first Pokemon

    def stats(self, species):
        """Stats of this set as `species` (a poke-env species id: the current forme, whose base stats
        can differ from the generated forme's), with random battles' neutral nature."""
        base = POKEDEX[species]["baseStats"]
        out = {}
        for i, s in enumerate(STATS):
            core = (2 * base[s] + self.ivs[i] + self.evs[i] // 4) * self.level // 100
            out[s] = core + self.level + 10 if s == "hp" else core + 5
        if species == "shedinja":
            out["hp"] = 1
        return out


class _Weighted:
    """Draws from a fixed list with fixed weights (cumulative sums + bisect)."""
    __slots__ = ("items", "cum", "total")

    def __init__(self, items, weights):
        self.items = items
        self.cum = list(itertools.accumulate(weights))
        self.total = self.cum[-1] if self.cum else 0

    def draw(self, rng):
        return self.items[bisect.bisect_right(self.cum, rng.random() * self.total)]


class _TeamLimits:
    """The generator's team limits (gen9 teams.ts randomTeam): one Pokemon per base species, at most
    2 of a type, 3 weak to a type (Dry Skin / Fluffy count as a Fire weakness), 1 double weak to a
    type, 4 weak to Freeze-Dry, one level 100 Pokemon and one of Ogerpon / Ogerpon-Hearthflame /
    Terapagos (its Tera Blast user slot; Tera Blast user roles aren't in the prior)."""

    def __init__(self, info):
        self.info = info
        self.bases = set()
        self.types = Counter()
        self.weak = Counter()
        self.double_weak = Counter()
        self.level100 = 0
        self.tera_blast = False

    def fits(self, key):
        i = self.info[key]
        if i["base"] in self.bases:
            return False
        if any(self.types[t] >= 2 for t in i["types"]):
            return False
        if any(self.weak[t] >= 3 for t in i["weak"]) or any(self.double_weak[t] >= 1 for t in i["double_weak"]):
            return False
        if i["fire_ability"] and self.weak["fire"] >= 3:
            return False
        if i["weak_freeze_dry"] and self.weak["freezedry"] >= 4:
            return False
        if i["level"] == 100 and self.level100 >= 1:
            return False
        return not (i["tera_blast_species"] and self.tera_blast)

    def add(self, rset):
        i = self.info.get(rset.key)
        if i is None:
            return
        self.bases.add(i["base"])
        self.types.update(i["types"])
        self.weak.update(i["weak"])
        self.double_weak.update(i["double_weak"])
        if i["fire_ability"] and rset.ability in ("dryskin", "fluffy"):
            self.weak["fire"] += 1
        if i["weak_freeze_dry"]:
            self.weak["freezedry"] += 1
        self.level100 += rset.level == 100
        self.tera_blast |= i["tera_blast_species"]


class SetPrior:
    def __init__(self, path=PRIOR_PATH):
        with open(path) as f:
            data = json.load(f)
        self.meta = data["meta"]
        self.info = data["info"]["species"]
        self.pool = data["info"]["pool"]
        self.sets = {}
        forme_keys = {}
        for key, entry in data["species"].items():
            self.sets[key] = [RandSet(key, r[2], r[3], r[4], tuple(r[5]), r[6], tuple(r[7]), tuple(r[8]),
                                      r[0], r[1]) for r in entry["sets"]]
            for forme in entry["formes"]:
                forme_keys.setdefault(forme, []).append(key)
        self._forme_keys = forme_keys
        self._move_sets = {key: [frozenset(s.moves) for s in sets] for key, sets in self.sets.items()}
        self._keys_cache = {}
        bases = sorted(self.pool)
        self._bases = _Weighted(bases, [data["info"]["pool_weights"][b] for b in bases])
        self._draw_cache = {}

    # ---- species -------------------------------------------------------------------------------
    def keys_for(self, species):
        """The sets.json entries a Pokemon shown as `species` (a poke-env species id) can have been
        generated from: the entry itself; for a cosmetic forme its entry (Sawsbuck-Autumn); for a
        battle-only forme the forme it changes from (Palafin-Hero, Ogerpon-Wellspring-Tera)."""
        keys = self._keys_cache.get(species)
        if keys is None:
            keys = self._keys_cache[species] = self._find_keys(species, 0)
        return keys

    def _find_keys(self, species, depth):
        if species in self.sets:
            return [species]
        if species in self._forme_keys:
            return list(self._forme_keys[species])
        entry = POKEDEX.get(species)
        if entry is None or depth > 3:
            return []
        for field in ("battleOnly", "changesFrom"):
            src = entry.get(field)
            if src:
                found = []
                for s in ([src] if isinstance(src, str) else src):
                    found += [k for k in self._find_keys(_to_id(s), depth + 1) if k not in found]
                if found:
                    return found
        base = entry.get("baseSpecies")
        for b, keys in self.pool.items():
            if base and _to_id(b) == _to_id(base):
                return list(keys)
        return []

    # ---- sets ----------------------------------------------------------------------------------
    def candidates(self, key, level=None, moves=(), item=None, ability=None, tera=None, lead=False,
                   no_items=frozenset()):
        """The sets of `key` consistent with what the Pokemon has shown, and their weights.

        moves: move ids it has used; item / ability: its original item ('' = none) and ability, None
        if not known; tera: its tera type if it has terastallized; lead: it was its team's first
        Pokemon (weights from lead counts); no_items: items its original item can't be (from what it
        did, search.state.LogTracker). When no set fits everything, constraints are dropped in
        order (item, ability, tera, then sets with the most of the shown moves), so a wrong
        observation (a move called by another move, a traded item) can't leave nothing to sample.
        Returns (sets, weights, relaxed) where relaxed counts the constraints dropped (0 = none).
        """
        sets = self.sets.get(key)
        if not sets:
            return [], [], 0
        moves = frozenset(moves) - NOT_SET_MOVES
        msets = self._move_sets[key]
        idx = range(len(sets))
        lvl = [i for i in idx if level is None or sets[i].level == level] or list(idx)
        full = [i for i in lvl if moves <= msets[i]]
        relaxed = 0
        chosen = None
        if full:
            for n_drop, checks in enumerate((
                    (("tera", tera), ("ability", ability), ("item", item)),
                    (("tera", tera), ("ability", ability)),
                    (("tera", tera),),
                    ())):
                ok = [i for i in full if all(v is None or getattr(sets[i], f) == v for f, v in checks)
                      and (n_drop > 0 or sets[i].item not in no_items)]
                if ok:
                    chosen, relaxed = ok, n_drop
                    break
        if chosen is None:
            best = max(len(moves & msets[i]) for i in lvl)
            chosen, relaxed = [i for i in lvl if len(moves & msets[i]) == best], 4
        out = [sets[i] for i in chosen]
        return out, self._weights(out, lead), relaxed

    @staticmethod
    def _weights(sets, lead):
        if lead:
            w = [s.lead for s in sets]
        else:
            w = [s.count - s.lead for s in sets]
        if sum(w) <= 0:
            w = [s.count for s in sets]
        return w

    def draw(self, key, rng, lead=False):
        """A set for `key` from the generator's distribution (no observations)."""
        d = self._draw_cache.get((key, lead))
        if d is None:
            sets = self.sets[key]
            d = self._draw_cache[(key, lead)] = _Weighted(sets, self._weights(sets, lead))
        return d.draw(rng)

    # ---- teams ---------------------------------------------------------------------------------
    def fill_team(self, revealed, rng, size=6, max_tries=2000):
        """Sets for the opponent's unrevealed Pokemon, given the RandSets drawn for its revealed ones:
        species the way the generator picks them (a base species from its weighted pool, then one of
        that base species' entries), rejected when they break its limits with the team so far."""
        limits = _TeamLimits(self.info)
        for s in revealed:
            limits.add(s)
        out = []
        tries = 0
        while len(revealed) + len(out) < size:
            tries += 1
            base = self._bases.draw(rng)
            key = rng.choice(self.pool[base])
            if self.info[key]["base"] in limits.bases:
                continue
            # past max_tries (a team the limits can't complete) only the species clause applies
            if tries <= max_tries and not limits.fits(key):
                continue
            s = self.draw(key, rng)
            limits.add(s)
            out.append(s)
        return out


def _to_id(name):
    return "".join(c for c in name.lower() if c.isalnum())


def observations(battle, tr, infer_items=False):
    """What the observer of `battle` knows about each revealed opponent Pokemon, in
    battle.opponent_team order: dicts of candidates() arguments plus its species.
    tr: the battle's search.state.LogTracker (current forme, original item). infer_items: also the
    items ruled out by what it did (two moves in a stint: no Choice item; a status move: no Assault
    Vest; hazard damage: no Heavy-Duty Boots), as Foul Play does."""
    out = []
    for j, (key, mon) in enumerate(list(battle.opponent_team.items())[:6]):
        m = tr.mons.get(key)
        species = m.forme if m is not None and m.forme in POKEDEX else mon.species
        tera = mon.tera_type.name.lower() if mon.is_terastallized and mon.tera_type is not None else None
        item = m.orig_item if m is not None and not m.traded else None
        no_items = frozenset(m.no_items) if infer_items and m is not None else frozenset()
        out.append(dict(species=species, level=mon.level, moves=tuple(mon.moves), item=item,
                        ability=mon.ability, tera=tera, lead=j == 0, no_items=no_items))
    return out


def sample_opponent(prior, obs, rng, weight_fn=None):
    """One sampled opponent team for observations() output: a RandSet per revealed Pokemon (in
    order), then the unrevealed ones. Returns (sets, relaxed) with relaxed[j] the number of
    constraints dropped for revealed Pokemon j (-1: species not in the prior, a stand-in set).
    weight_fn(j, candidates, weights) -> weights reweights revealed Pokemon j's candidates
    (search/net.py: the network's hidden-set model)."""
    sets, relaxed = [], []
    for j, o in enumerate(obs):
        keys = prior.keys_for(o["species"])
        cands, weights, n_relaxed = [], [], 0
        for k in keys:
            c, w, r = prior.candidates(k, o["level"], o["moves"], o["item"], o["ability"], o["tera"],
                                       o["lead"], o.get("no_items", frozenset()))
            if c and (not cands or r < n_relaxed):
                cands, weights, n_relaxed = c, w, r
            elif c and r == n_relaxed:
                cands, weights = cands + c, weights + w
        if cands and weight_fn is not None:
            weights = weight_fn(j, cands, weights)
        if cands:
            sets.append(_Weighted(cands, weights).draw(rng))
            relaxed.append(n_relaxed)
        else:
            sets.append(_stand_in(o))
            relaxed.append(-1)
    sets += prior.fill_team(sets, rng)
    return sets, relaxed


def _stand_in(o):
    """A set for a Pokemon the prior doesn't have: what it has shown, neutral defaults otherwise."""
    entry = POKEDEX.get(o["species"], {})
    abilities = [_to_id(a) for a in entry.get("abilities", {}).values()]
    types = [t.lower() for t in entry.get("types", ["Normal"])]
    moves = tuple(m for m in o["moves"] if m not in NOT_SET_MOVES)[:4] or ("tackle",)
    return RandSet(o["species"], o["level"], o["item"] or "", o["ability"] or (abilities[0] if abilities else ""),
                   moves, o["tera"] or types[0], (85,) * 6, (31,) * 6, 1, 0)


_PRIOR = None


def get_prior():
    global _PRIOR
    if _PRIOR is None:
        _PRIOR = SetPrior()
    return _PRIOR
