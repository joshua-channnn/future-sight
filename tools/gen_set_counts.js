// Counts the complete sets the local Showdown fork's gen9 random battle generator produces, for
// tools/build_set_prior.py (which runs several of these in parallel and merges them).
//
//   node tools/gen_set_counts.js <showdown dir> <teams> <seed> <out.json> [teams.json]
//
// Output: {"teams": n, "sets": {key: {"<set json>": [count, lead count]}}, "formes": ..., "info": ...}
// where key is the species' entry in sets.json the set was generated from, the set json is
// [level, item, ability, sorted moves, tera type, evs[6], ivs[6]] (ids), and "lead" counts the times
// the set was the team's first Pokemon. "formes" counts the forme each key was generated as (cosmetic
// formes; battle-only formes such as Zacian-Crowned generate as their out-of-battle forme).
// "info" (written only for seed 1) has what the sampler needs to rebuild teams the way the
// generator does: each key's base species, types, weaknesses and level, and the base-species pool.
// teams.json (optional): every team as a list of [key, ability, level], for tools/check_set_prior.py.
const fs = require('fs');
const path = require('path');

const [showdownArg, nArg, seedArg, outArg, teamsArg] = process.argv.slice(2);
const showdown = path.resolve(showdownArg);
const {Teams} = require(path.join(showdown, 'dist', 'sim', 'teams'));
const {Dex, toID} = require(path.join(showdown, 'dist', 'sim', 'dex'));

const n = parseInt(nArg, 10);
const seed = parseInt(seedArg, 10);
const gen = Teams.getGenerator('gen9randombattle', [seed, seed * 7 + 1, seed * 13 + 2, seed * 31 + 3]);
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];

// record which sets.json entry each set was generated from (battle-only formes such as Zacian-Crowned
// generate as their out-of-battle forme, so the set's species doesn't say)
const randomSet = gen.randomSet;
gen.randomSet = function (s, ...rest) {
  const set = randomSet.call(this, s, ...rest);
  set.setsKey = this.dex.species.get(s).id;
  return set;
};

const counts = {};
const formes = {};
const teams = [];
for (let t = 0; t < n; t++) {
  const team = gen.getTeam();
  if (teamsArg) teams.push(team.map(set => [set.setsKey, toID(set.ability), set.level]));
  team.forEach((set, slot) => {
    const forme = Dex.species.get(set.species);
    const key = set.setsKey;
    if (!gen.randomSets[key]) throw new Error(`no sets.json entry ${key} for ${forme.id}`);
    const rec = JSON.stringify([
      set.level, toID(set.item), toID(set.ability), set.moves.map(toID).sort(),
      toID(set.teraType), STATS.map(s => set.evs[s]), STATS.map(s => set.ivs[s]),
    ]);
    const bySet = counts[key] || (counts[key] = {});
    const c = bySet[rec] || (bySet[rec] = [0, 0]);
    c[0]++;
    if (slot === 0) c[1]++;
    const f = formes[key] || (formes[key] = {});
    f[forme.id] = (f[forme.id] || 0) + 1;
  });
}

const out = {teams: n, seed, sets: counts, formes};
if (seed === 1) {
  const dex = gen.dex;
  const typeNames = dex.types.names();
  const info = {};
  const pool = {};
  for (const k of Object.keys(gen.randomSets)) {
    const sp = dex.species.get(k);
    const eff = t => dex.getEffectiveness(t, sp);
    info[k] = {
      base: sp.baseSpecies, types: sp.types.map(x => x.toLowerCase()),
      weak: typeNames.filter(t => eff(t) > 0).map(x => x.toLowerCase()),
      double_weak: typeNames.filter(t => eff(t) > 1).map(x => x.toLowerCase()),
      fire_ability: eff('Fire') === 0 &&
        Object.values(sp.abilities).some(a => ['Dry Skin', 'Fluffy'].includes(a)),
      weak_freeze_dry: eff('Ice') > 0 || (eff('Ice') > -2 && sp.types.includes('Water')),
      level: gen.getLevel(sp, false),
      tera_blast_species: ['ogerpon', 'ogerponhearthflame', 'terapagos'].includes(sp.id),
    };
    (pool[sp.baseSpecies] || (pool[sp.baseSpecies] = [])).push(k);
  }
  // the generator's base-species pool: 1 copy for 1-3 formes, 2 for 4-6, 3 for 7+ (getPokemonPool)
  const weights = {};
  for (const [b, ks] of Object.entries(pool)) {
    weights[b] = b === 'Squawkabilly' ? 1 : Math.min(Math.ceil(ks.length / 3), 3);
  }
  out.info = {species: info, pool, pool_weights: weights};
}
fs.writeFileSync(outArg, JSON.stringify(out));
if (teamsArg) fs.writeFileSync(teamsArg, JSON.stringify(teams));
