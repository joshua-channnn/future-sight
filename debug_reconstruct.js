'use strict';

const {Battle, Teams, Dex} = require('./dist/sim');

// Test 1: Create a simple battle and verify moves work
console.log('=== Test 1: Simple packed team battle ===\n');

// Pack format: name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|rest
const p1Packed = [
	'|Grimmsnarl||Prankster|thunderwave,spiritbreak,suckerpunch,taunt|||85,,85,85,85,85||31,31,31,31,31,31||83|,,,,,Dark',
].join(']');

const p2Packed = [
	'|Dragapult||Infiltrator|shadowball,dracometeor,uturn,thunderbolt|||85,,85,85,85,85||31,31,31,31,31,31||78|,,,,,Ghost',
].join(']');

console.log('P1 packed:', p1Packed);
console.log('P2 packed:', p2Packed);
console.log();

try {
	const battle = new Battle({formatid: 'gen9randombattle'});
	battle.setPlayer('p1', {name: 'P1', team: p1Packed});
	battle.setPlayer('p2', {name: 'P2', team: p2Packed});

	console.log('Battle created successfully');
	console.log('Turn:', battle.turn);
	console.log('P1 pokemon:', battle.p1.pokemon.map(p => `${p.species.name} HP:${p.hp}/${p.maxhp}`));
	console.log('P2 pokemon:', battle.p2.pokemon.map(p => `${p.species.name} HP:${p.hp}/${p.maxhp}`));
	console.log();

	// Check active request
	const p1req = battle.p1.activeRequest;
	const p2req = battle.p2.activeRequest;
	console.log('P1 activeRequest type:', p1req ? Object.keys(p1req) : 'null');
	console.log('P2 activeRequest type:', p2req ? Object.keys(p2req) : 'null');

	if (p1req && p1req.active) {
		console.log('P1 active moves:', p1req.active[0].moves.map(m => `${m.move}(${m.id})`));
	} else {
		console.log('P1 has NO active moves! Request:', JSON.stringify(p1req, null, 2).slice(0, 500));
	}
	if (p2req && p2req.active) {
		console.log('P2 active moves:', p2req.active[0].moves.map(m => `${m.move}(${m.id})`));
	}
	console.log();

	// Try simulating
	const json = battle.toJSON();
	const clone = Battle.fromJSON(json);
	console.log('Cloned battle. Trying move 1 vs move 1...');
	clone.choose('p1', 'move 1');
	clone.choose('p2', 'move 1');

	console.log('After simulation:');
	console.log('P1 active:', clone.p1.active[0] ? `${clone.p1.active[0].species.name} HP:${clone.p1.active[0].hp}/${clone.p1.active[0].maxhp}` : 'none');
	console.log('P2 active:', clone.p2.active[0] ? `${clone.p2.active[0].species.name} HP:${clone.p2.active[0].hp}/${clone.p2.active[0].maxhp}` : 'none');
	console.log('Log excerpt:', clone.log.slice(-10).join('\n'));

} catch (e) {
	console.error('ERROR:', e.message);
	console.error(e.stack);
}

// Test 2: Check if Teams.unpack works with our format
console.log('\n=== Test 2: Teams.unpack validation ===\n');

try {
	const unpacked = Teams.unpack(p1Packed);
	if (unpacked) {
		console.log('Unpacked team:');
		for (const mon of unpacked) {
			console.log(`  ${mon.species}: moves=[${mon.moves}] ability=${mon.ability} item=${mon.item} level=${mon.level}`);
		}
	} else {
		console.log('Teams.unpack returned null! Format is wrong.');
	}
} catch (e) {
	console.error('Unpack error:', e.message);
}

// Test 3: Check how Showdown formats a packed team
console.log('\n=== Test 3: Showdown native pack/unpack ===\n');

try {
	const testTeam = [{
		name: '',
		species: 'Grimmsnarl',
		item: 'Leftovers',
		ability: 'Prankster',
		moves: ['thunderwave', 'spiritbreak', 'suckerpunch', 'taunt'],
		nature: '',
		evs: {hp: 85, atk: 0, def: 85, spa: 85, spd: 85, spe: 85},
		gender: '',
		ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
		shiny: false,
		level: 83,
		teraType: 'Dark',
	}];

	const nativePacked = Teams.pack(testTeam);
	console.log('Native packed:', nativePacked);

	// Now unpack it back
	const nativeUnpacked = Teams.unpack(nativePacked);
	if (nativeUnpacked) {
		for (const mon of nativeUnpacked) {
			console.log(`  ${mon.species}: moves=[${mon.moves}] ability=${mon.ability} level=${mon.level} tera=${mon.teraType}`);
		}
	}

	// Compare with our format
	console.log('\nOur packed:   ', p1Packed);
	console.log('Native packed:', nativePacked);
	console.log('Match:', p1Packed === nativePacked);

} catch (e) {
	console.error('Native pack error:', e.message);
}

// Test 4: Test what happens with multiple moves producing different results
console.log('\n=== Test 4: Different moves → different outcomes ===\n');

try {
	const battle = new Battle({formatid: 'gen9randombattle'});
	battle.setPlayer('p1', {name: 'P1', team: p1Packed});
	battle.setPlayer('p2', {name: 'P2', team: p2Packed});

	const baseJSON = battle.toJSON();

	const moves = ['move 1', 'move 2', 'move 3', 'move 4'];
	for (const move of moves) {
		try {
			const clone = Battle.fromJSON(baseJSON);
			clone.choose('p1', move);
			clone.choose('p2', 'move 1');

			const p2hp = clone.p2.active[0] ? clone.p2.active[0].hp : 'fainted';
			const p1hp = clone.p1.active[0] ? clone.p1.active[0].hp : 'fainted';
			console.log(`  P1 uses ${move}: P1=${p1hp} P2=${p2hp}`);
		} catch (e) {
			console.log(`  P1 uses ${move}: ERROR - ${e.message}`);
		}
	}
} catch (e) {
	console.error('Test 4 error:', e.message);
}
