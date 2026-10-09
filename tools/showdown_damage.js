// Damage from the local Showdown fork's simulator, for tools/check_engine_build.py.
//
// Reads a JSON list of cases on stdin, writes a JSON list of results on stdout. Each case:
//   {att: set, def: set, move?, weather?, terrain?, att_state?, def_state?, def_side?}
// where a set is Showdown's PokemonSet (species, moves, level, ability, item, evs, ivs, nature,
// teraType), *_state is {boosts?: {atk: 2, ...}, status?: 'brn', tera?: true}, and def_side lists
// side conditions on the defender's side (e.g. ['reflect']).
// Each result: both Pokemon as the simulator built them (species id, level, types, stats, max HP,
// ability, item, tera type, weight) and the move's damage with the highest roll, the lowest roll
// and the highest roll on a critical hit (getDamage with the random roll fixed, no crit unless asked).
//
//   node tools/showdown_damage.js ../pokemon-showdown < cases.json
const fs = require('fs');
const path = require('path');

const showdown = path.resolve(process.argv[2] || path.join(__dirname, '..', '..', 'pokemon-showdown'));
const {Battle} = require(path.join(showdown, 'dist', 'sim', 'battle'));
const {Teams} = require(path.join(showdown, 'dist', 'sim', 'teams'));

const STATS = {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85};
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function fullSet(s) {
  return Object.assign({name: s.species, level: 80, ability: '', item: '', nature: 'Serious',
    evs: Object.assign({}, STATS), ivs: Object.assign({}, IVS)}, s);
}

function describe(p) {
  return {
    species: p.species.id, level: p.level, types: p.getTypes(true).map(t => t.toLowerCase()),
    base_types: p.species.types.map(t => t.toLowerCase()), stats: p.storedStats, maxhp: p.maxhp,
    hp: p.hp, ability: p.ability, item: p.item, tera_type: (p.teraType || '').toLowerCase(),
    terastallized: !!p.terastallized, weight_kg: p.species.weightkg,
    status: p.status || '',
  };
}

function prepare(battle, p, st) {
  if (!st) return;
  if (st.tera) battle.actions.terastallize(p);
  if (st.status) p.setStatus(st.status, p, null, true);
  for (const [k, v] of Object.entries(st.boosts || {})) p.boosts[k] = v;
  if (st.hp_frac !== undefined) p.hp = Math.max(1, Math.floor(p.maxhp * st.hp_frac));
}

function damage(battle, src, tgt, moveId, crit, roll) {
  const move = battle.dex.getActiveMove(moveId);
  move.willCrit = crit;
  battle.activeMove = move;  // Choice items lock the move in ModifyMove, which needs these
  battle.activePokemon = src;
  battle.activeTarget = tgt;
  // what useMoveInner runs before the hit: type and move modifiers (Tera Blast, -ate abilities)
  battle.singleEvent('ModifyType', move, null, src, tgt, move, move);
  battle.runEvent('ModifyType', src, tgt, move, move);
  battle.singleEvent('ModifyMove', move, null, src, tgt, move, move);
  battle.runEvent('ModifyMove', src, tgt, move, move);
  const random = battle.random;
  battle.random = (m, n) => (m === 16 && n === undefined) ? roll : random.call(battle, m, n);
  try {
    const d = battle.actions.getDamage(src, tgt, move, true);
    return d === false || d === undefined ? null : d;
  } finally {
    battle.random = random;
  }
}

const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = [];
for (const c of cases) {
  const battle = new Battle({formatid: 'gen9customgame', seed: [1, 2, 3, 4]});
  battle.setPlayer('p1', {name: 'A', team: Teams.pack([fullSet(c.att)])});
  battle.setPlayer('p2', {name: 'B', team: Teams.pack([fullSet(c.def)])});
  if (battle.turn === 0) battle.makeChoices('default', 'default');  // team preview
  const src = battle.p1.active[0], tgt = battle.p2.active[0];
  if (c.weather) {
    battle.field.weather = c.weather;
    battle.field.weatherState = {id: c.weather, duration: 5};
  }
  if (c.terrain) {
    battle.field.terrain = c.terrain;
    battle.field.terrainState = {id: c.terrain, duration: 5};
  }
  for (const sc of c.def_side || []) battle.p2.addSideCondition(sc, 'debug');
  prepare(battle, src, c.att_state);
  prepare(battle, tgt, c.def_state);
  out.push({
    att: describe(src), def: describe(tgt), weather: battle.field.effectiveWeather() || '',
    terrain: battle.field.terrain || '',
    max: damage(battle, src, tgt, c.move || c.att.moves[0], false, 0),
    min: damage(battle, src, tgt, c.move || c.att.moves[0], false, 15),
    crit: damage(battle, src, tgt, c.move || c.att.moves[0], true, 0),
  });
}
process.stdout.write(JSON.stringify(out));
