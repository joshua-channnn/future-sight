'use strict';

const http = require('http');
const {Battle} = require('./dist/sim');
const {Teams} = require('./dist/sim');
const {Dex} = require('./dist/sim');

const PORT = parseInt(process.argv.find((a, i) => process.argv[i - 1] === '--port') || '9002');

let RANDBATS_SETS = {};
try {
	RANDBATS_SETS = require('./data/random-battles/gen9/sets.json');
	console.log(`Loaded randbats sets for ${Object.keys(RANDBATS_SETS).length} species`);
} catch (e) {
	console.warn('Warning: Could not load randbats sets.json, move inference disabled');
}

/**
 * Given a Pokemon's species and its revealed moves, infer the full moveset
 * by matching to the most likely role in sets.json.
 *
 * @param {string} species - Pokemon species name (lowercase, e.g. "grimmsnarl")
 * @param {string[]} revealedMoves - Moves that have been seen
 * @returns {string[]} Full moveset (4 moves), with revealed moves + inferred ones
 */
function inferMoves(species, revealedMoves) {
	const speciesKey = species.toLowerCase().replace(/[^a-z0-9]/g, '');
	const data = RANDBATS_SETS[speciesKey];
	if (!data || !data.sets || data.sets.length === 0) {
		return revealedMoves.length > 0 ? revealedMoves : ['struggle'];
	}

	// Normalize revealed move names for comparison
	const revealedNorm = revealedMoves.map(m =>
		m.toLowerCase().replace(/[^a-z0-9]/g, ''));

	// Score each role by how many revealed moves match its movepool
	let bestRole = data.sets[0];
	let bestScore = -1;

	for (const set of data.sets) {
		const poolNorm = set.movepool.map(m =>
			m.toLowerCase().replace(/[^a-z0-9]/g, ''));
		const matches = revealedNorm.filter(m => poolNorm.includes(m)).length;
		if (matches > bestScore) {
			bestScore = matches;
			bestRole = set;
		}
	}

	// Start with revealed moves, fill remaining from best role's movepool
	const result = [...revealedMoves];
	const resultNorm = new Set(revealedNorm);

	for (const move of bestRole.movepool) {
		if (result.length >= 4) break;
		const moveNorm = move.toLowerCase().replace(/[^a-z0-9]/g, '');
		if (!resultNorm.has(moveNorm)) {
			result.push(move);
			resultNorm.add(moveNorm);
		}
	}

	// If still < 4 moves (shouldn't happen), pad with struggle
	while (result.length < 4) {
		result.push('struggle');
	}

	return result.slice(0, 4);
}

/**
 * Infer ability for an opponent Pokemon from sets.json.
 */
function inferAbility(species, knownAbility) {
	if (knownAbility) return knownAbility;
	const speciesKey = species.toLowerCase().replace(/[^a-z0-9]/g, '');
	const data = RANDBATS_SETS[speciesKey];
	if (data && data.sets && data.sets[0] && data.sets[0].abilities) {
		return data.sets[0].abilities[0];
	}
	// Fallback: use dex default
	const dexSpecies = Dex.species.get(species);
	if (dexSpecies && dexSpecies.abilities) {
		return dexSpecies.abilities[0] || 'Pressure';
	}
	return 'Pressure';
}

/**
 * Infer level for a Pokemon from sets.json.
 */
function inferLevel(species) {
	const speciesKey = species.toLowerCase().replace(/[^a-z0-9]/g, '');
	const data = RANDBATS_SETS[speciesKey];
	return (data && data.level) ? data.level : 80;
}

/**
 * Convert a simplified Pokemon description into Showdown's packed team format.
 *
 * @param {Object[]} team - Array of pokemon objects from Python
 * @param {boolean} isOpponent - If true, infer missing data from sets.json
 * @returns {string} Packed team string for Teams.pack()
 */
function buildPackedTeam(team, isOpponent = false) {
	const teamData = team.map(p => {
		let moves = p.moves || [];

		// Infer moves for opponent if we don't have all 4
		if (isOpponent && moves.length < 4) {
			moves = inferMoves(p.species, moves);
		}

		const ability = p.ability || (isOpponent ? inferAbility(p.species, null) : 'Pressure');
		const level = p.level || (isOpponent ? inferLevel(p.species) : 80);

		return {
			name: p.species,
			species: p.species,
			gender: p.gender || '',
			moves: moves,
			ability: ability,
			item: p.item || '',
			level: level,
			// Random battles use 85 EVs across the board
			evs: {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85},
			ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
			teraType: p.teraType || 'Normal',
		};
	});

	return Teams.pack(teamData);
}

/**
 * STATUS_MAP: Convert status string identifiers to Showdown's internal format.
 */
const STATUS_MAP = {
	'brn': 'brn',
	'burn': 'brn',
	'frz': 'frz',
	'freeze': 'frz',
	'par': 'par',
	'paralysis': 'par',
	'psn': 'psn',
	'poison': 'psn',
	'tox': 'tox',
	'toxic': 'tox',
	'slp': 'slp',
	'sleep': 'slp',
	'': '',
	'none': '',
	'null': '',
};

/**
 * Apply the simplified state onto a constructed battle.
 * Sets HP, status, boosts, side conditions, weather, terrain.
 *
 * @param {Battle} battle - The Showdown Battle object
 * @param {Object} state - The state description from Python
 */
function applyState(battle, state) {
	// Apply Pokemon state for both sides
	for (const [sideKey, teamKey] of [['p1', 'p1Team'], ['p2', 'p2Team']]) {
		const side = battle[sideKey];
		const teamState = state[teamKey];
		if (!side || !teamState) continue;

		for (let i = 0; i < Math.min(side.pokemon.length, teamState.length); i++) {
			const pokemon = side.pokemon[i];
			const stateP = teamState[i];

			// Set HP
			if (stateP.hp !== undefined && stateP.maxhp !== undefined) {
				pokemon.hp = stateP.hp;
				pokemon.maxhp = stateP.maxhp;
			} else if (stateP.hpFraction !== undefined) {
				pokemon.hp = Math.round(pokemon.maxhp * stateP.hpFraction);
			}

			// Handle fainted
			if (pokemon.hp <= 0) {
				pokemon.hp = 0;
				pokemon.fainted = true;
				pokemon.status = '';
			}

			// Set status
			if (stateP.status !== undefined && stateP.status !== '' && stateP.status !== null) {
				const mapped = STATUS_MAP[stateP.status.toLowerCase()] || '';
				if (mapped) {
					pokemon.status = mapped;
					pokemon.statusState = {id: mapped, time: 1, stage: 0};
					if (mapped === 'tox') {
						pokemon.statusState.stage = stateP.toxicCounter || 1;
					}
					if (mapped === 'slp') {
						pokemon.statusState.time = stateP.sleepTurns || 1;
					}
				}
			}

			// Set boosts
			if (stateP.boosts) {
				for (const [stat, val] of Object.entries(stateP.boosts)) {
					if (val !== 0) {
						pokemon.boosts[stat] = val;
					}
				}
			}

			// Set terastallized
			if (stateP.terastallized) {
				pokemon.terastallized = stateP.terastallized;
			}
		}
	}

	// Apply side conditions (hazards, screens)
	if (state.sideConditions) {
		for (const [sideKey, conditions] of Object.entries(state.sideConditions)) {
			const side = battle[sideKey];
			if (!side || !conditions) continue;

			for (const [condition, layers] of Object.entries(conditions)) {
				// Showdown uses camelCase internally
				const condId = condition.toLowerCase().replace(/[^a-z]/g, '');
				side.sideConditions[condId] = {
					id: condId,
					target: side,
					layers: typeof layers === 'number' ? layers : 1,
				};
			}
		}
	}

	// Apply weather
	if (state.weather && state.weather !== '' && state.weather !== 'none') {
		const weatherId = state.weather.toLowerCase().replace(/[^a-z]/g, '');
		battle.field.weatherState = {
			id: weatherId,
			source: null,
			sourceSlot: '',
			duration: state.weatherDuration || 5,
		};
	}

	// Apply terrain
	if (state.terrain && state.terrain !== '' && state.terrain !== 'none') {
		const terrainId = state.terrain.toLowerCase().replace(/[^a-z]/g, '');
		battle.field.terrainState = {
			id: terrainId,
			source: null,
			sourceSlot: '',
			duration: state.terrainDuration || 5,
		};
	}
}

/**
 * Get available choices for a side, handling the activeRequest properly.
 */
function getAvailableChoices(battle, playerid) {
	const side = battle[playerid];
	if (!side) return ['move 1'];

	try {
		// Try using side's getChoices if available
		if (side.activeRequest && side.activeRequest.active) {
			const choices = [];
			const active = side.activeRequest.active[0];

			// Moves
			if (active && active.moves) {
				for (let i = 0; i < active.moves.length; i++) {
					if (!active.moves[i].disabled) {
						choices.push(`move ${i + 1}`);
					}
				}
			}

			// Switches
			if (side.activeRequest.side && side.activeRequest.side.pokemon) {
				for (let i = 1; i < side.activeRequest.side.pokemon.length; i++) {
					const p = side.activeRequest.side.pokemon[i];
					if (p.condition !== '0 fnt' && !p.active) {
						choices.push(`switch ${i + 1}`);
					}
				}
			}

			if (choices.length > 0) return choices;
		}
	} catch (e) {
		// Fall through to basic approach
	}

	// Fallback: basic choices based on pokemon state
	const choices = [];
	const active = side.active[0];
	if (active && !active.fainted) {
		const nMoves = active.moveSlots ? active.moveSlots.length : 4;
		for (let i = 0; i < nMoves; i++) {
			choices.push(`move ${i + 1}`);
		}
	}
	for (let i = 1; i < side.pokemon.length; i++) {
		if (!side.pokemon[i].fainted) {
			choices.push(`switch ${i + 1}`);
		}
	}
	return choices.length > 0 ? choices : ['move 1'];
}

/**
 * Extract rich state information from a battle for NN value evaluation.
 * Returns enough data for Python to construct a 677-dim observation.
 *
 * @param {Battle} battle - The Showdown Battle after simulation
 * @param {string} perspective - 'p1' or 'p2' (whose perspective for the observation)
 * @returns {Object} Rich state description
 */
function extractRichState(battle, perspective = 'p1') {
	const result = {
		ended: battle.ended,
		winner: battle.winner || null,
		turn: battle.turn,
	};

	// Extract all pokemon for both sides
	for (const sideKey of ['p1', 'p2']) {
		const side = battle[sideKey];
		if (!side) continue;

		result[sideKey] = {
			pokemon: side.pokemon.map((p, idx) => ({
				species: p.species.name || p.species.id || 'unknown',
				hp: p.hp,
				maxhp: p.maxhp,
				hpFraction: p.maxhp > 0 ? p.hp / p.maxhp : 0,
				fainted: p.fainted,
				status: p.status || '',
				statusTurns: p.statusState ? p.statusState.time || 0 : 0,
				toxicCounter: (p.status === 'tox' && p.statusState) ? p.statusState.stage || 0 : 0,
				boosts: {...p.boosts},
				isActive: p.isActive || (side.active[0] === p),
				types: p.types ? p.types.slice() : [],
				terastallized: p.terastallized || '',
				ability: (p.ability || p.baseAbility || ''),
				item: p.item || '',
				level: p.level || 80,
				baseStats: p.baseStats ? {...p.baseStats} : {},
				// Move info for active pokemon
				moves: (p.moveSlots || []).map(ms => ({
					id: ms.id,
					pp: ms.pp,
					maxpp: ms.maxpp,
				})),
				// Volatile status effects
				volatiles: Object.keys(p.volatiles || {}),
			})),
			sideConditions: {},
			alive: side.pokemon.filter(p => !p.fainted).length,
		};

		// Side conditions
		for (const [condId, condData] of Object.entries(side.sideConditions || {})) {
			result[sideKey].sideConditions[condId] = condData.layers || 1;
		}
	}

	// Field conditions
	result.field = {
		weather: battle.field.weatherState ? battle.field.weatherState.id : '',
		weatherDuration: battle.field.weatherState ? battle.field.weatherState.duration : 0,
		terrain: battle.field.terrainState ? battle.field.terrainState.id : '',
		terrainDuration: battle.field.terrainState ? battle.field.terrainState.duration : 0,
	};

	// Pseudo-weather (trick room, etc.)
	result.field.pseudoWeather = {};
	if (battle.field.pseudoWeather) {
		for (const [id, data] of Object.entries(battle.field.pseudoWeather)) {
			result.field.pseudoWeather[id] = data.duration || 0;
		}
	}

	return result;
}

/**
 * Main handler: simulate all move pairs in batch.
 *
 * Input:
 * {
 *   p1Team: [{species, moves, ability, item, level, hp, maxhp, status, boosts, teraType, ...}],
 *   p2Team: [same format],
 *   sideConditions: {p1: {stealthrock: 1}, p2: {}},
 *   weather: "",
 *   terrain: "",
 *   turn: 5,
 *   movePairs: [{p1: "move 1", p2: "move 2"}, ...],
 *   // OR: auto-generate pairs from legal moves
 *   p1Actions: ["move 1", "move 2", ...],  // if movePairs not provided
 *   p2Actions: ["move 1", "switch 2", ...], // opponent actions to sample
 * }
 *
 * Output:
 * {
 *   results: [{p1Move, p2Move, state: <rich state>}, ...],
 *   templateChoices: {p1: [...], p2: [...]},
 *   timing: {constructMs, simulateMs, totalMs}
 * }
 */
function simulateBatch(state) {
	const t0 = Date.now();

	// === Step 1: Construct battle with both teams ===
	const p1Packed = buildPackedTeam(state.p1Team, false);
	const p2Packed = buildPackedTeam(state.p2Team, true);

	const battle = new Battle({formatid: 'gen9randombattle'});
	battle.setPlayer('p1', {name: 'SearchP1', team: p1Packed});
	battle.setPlayer('p2', {name: 'SearchP2', team: p2Packed});

	// === Step 2: Apply current state (HP, status, boosts, conditions) ===
	applyState(battle, state);

	// === Step 3: Get template JSON for cloning ===
	const templateJSON = battle.toJSON();
	const tConstruct = Date.now();

	// === Step 4: Determine move pairs ===
	let movePairs = state.movePairs;
	if (!movePairs) {
		// Auto-generate from p1Actions × p2Actions
		const p1Actions = state.p1Actions || getAvailableChoices(battle, 'p1');
		const p2Actions = state.p2Actions || getAvailableChoices(battle, 'p2');
		movePairs = [];
		for (const p1Move of p1Actions) {
			for (const p2Move of p2Actions) {
				movePairs.push({p1: p1Move, p2: p2Move});
			}
		}
	}

	// === Step 5: Simulate each pair ===
	const results = [];
	for (const pair of movePairs) {
		try {
			const clone = Battle.fromJSON(templateJSON);
			clone.choose('p1', pair.p1);
			clone.choose('p2', pair.p2);
			const richState = extractRichState(clone, 'p1');
			results.push({
				p1Move: pair.p1,
				p2Move: pair.p2,
				state: richState,
				success: true,
			});
		} catch (e) {
			results.push({
				p1Move: pair.p1,
				p2Move: pair.p2,
				error: e.message,
				success: false,
			});
		}
	}

	const tSimulate = Date.now();

	// Get available choices for reference
	const templateChoices = {
		p1: getAvailableChoices(battle, 'p1'),
		p2: getAvailableChoices(battle, 'p2'),
	};

	return {
		results,
		templateChoices,
		nPairs: movePairs.length,
		nSuccess: results.filter(r => r.success).length,
		timing: {
			constructMs: tConstruct - t0,
			simulateMs: tSimulate - tConstruct,
			totalMs: tSimulate - t0,
		},
	};
}

function readBody(req) {
	return new Promise((resolve, reject) => {
		let body = '';
		req.on('data', chunk => {
			body += chunk;
		});
		req.on('end', () => resolve(body));
		req.on('error', reject);
	});
}

const server = http.createServer(async (req, res) => {
	// CORS headers
	res.setHeader('Access-Control-Allow-Origin', '*');
	res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
	res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

	if (req.method === 'OPTIONS') {
		res.writeHead(200);
		res.end();
		return;
	}

	try {
		// Health check
		if (req.url === '/health' && req.method === 'GET') {
			res.writeHead(200, {'Content-Type': 'application/json'});
			res.end(JSON.stringify({
				status: 'ok',
				version: 'v2',
				randbatsSetsLoaded: Object.keys(RANDBATS_SETS).length > 0,
			}));
			return;
		}

		// Self-test
		if (req.url === '/test' && req.method === 'GET') {
			const testState = {
				p1Team: [
					{
						species: 'Grimmsnarl', level: 83,
						moves: ['thunderwave', 'spiritbreak', 'suckerpunch', 'taunt'],
						ability: 'Prankster', item: 'Leftovers',
						hp: 293, maxhp: 293, status: '', boosts: {},
						teraType: 'Dark', isActive: true,
					},
					{
						species: 'Donphan', level: 84,
						moves: ['earthquake', 'iceshard', 'stealthrock', 'rapidspin'],
						ability: 'Sturdy', item: 'Leftovers',
						hp: 300, maxhp: 300, status: '', boosts: {},
						teraType: 'Ground',
					},
				],
				p2Team: [
					{
						species: 'Garchomp', level: 77,
						moves: ['earthquake', 'outrage'], // Only 2 revealed — will be inferred
						ability: 'Rough Skin', item: '',
						hpFraction: 0.85, status: '', boosts: {},
						teraType: 'Dragon', isActive: true,
					},
					{
						species: 'Slowbro', level: 82,
						moves: ['scald'], // Only 1 revealed
						ability: '', item: '',
						hpFraction: 1.0, status: '', boosts: {},
						teraType: 'Water',
					},
				],
				sideConditions: {p1: {}, p2: {}},
				weather: '',
				terrain: '',
				turn: 1,
				p1Actions: ['move 1', 'move 2', 'switch 2'],
				p2Actions: ['move 1', 'move 2'],
			};

			const result = simulateBatch(testState);
			res.writeHead(200, {'Content-Type': 'application/json'});
			res.end(JSON.stringify(result, null, 2));
			return;
		}

		// Main endpoint: simulate batch
		if (req.url === '/simulate-batch' && req.method === 'POST') {
			const body = await readBody(req);
			const state = JSON.parse(body);
			const result = simulateBatch(state);
			res.writeHead(200, {'Content-Type': 'application/json'});
			res.end(JSON.stringify(result));
			return;
		}

		// 404
		res.writeHead(404, {'Content-Type': 'application/json'});
		res.end(JSON.stringify({error: 'Not found'}));
	} catch (e) {
		console.error('Server error:', e);
		res.writeHead(500, {'Content-Type': 'application/json'});
		res.end(JSON.stringify({error: e.message, stack: e.stack}));
	}
});

server.listen(PORT, () => {
	console.log(`search_server_v2 running on http://localhost:${PORT}`);
	console.log(`  POST /simulate-batch  — Simulate move pairs`);
	console.log(`  GET  /health          — Health check`);
	console.log(`  GET  /test            — Self-test`);
});
