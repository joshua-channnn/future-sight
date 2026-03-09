import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.battle import SideCondition

_DATA_DIR = Path(__file__).parent / "data"


def _load_json(filename: str) -> dict:
    path = _DATA_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


POKEMON_TO_IDX = _load_json("pokemon_to_idx.json")
MOVE_TO_IDX = _load_json("move_to_idx.json")
ABILITY_TO_IDX = _load_json("ability_to_idx.json")
ITEM_TO_IDX = _load_json("item_to_idx.json")

try:
    from poke_env.data import GenData
    GEN_DATA = GenData.from_gen(9)
except Exception:
    GEN_DATA = None

TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2,
             "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2,
              "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2,
            "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5,
                 "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5,
               "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0,
               "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5,
            "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2,
             "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2,
              "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5},
}


def get_type_effectiveness(atk_type: str, def_types: list) -> float:
    atk = atk_type.lower()
    mult = 1.0
    for dt in def_types:
        if atk in TYPE_CHART:
            mult *= TYPE_CHART[atk].get(dt.lower(), 1.0)
    return mult


# MUST match RLPlayer6v6_v18.py
N_POKEMON = 12
PER_POKEMON_INDICES = 13
PER_POKEMON_FLOATS = 31   # includes move_knowledge as the 31st float
PER_POKEMON = PER_POKEMON_INDICES + PER_POKEMON_FLOATS  # 44

N_ACTIVE_MOVE_FLOATS = 60
N_ACTIVE_MOVE_INDICES = 4
N_GLOBAL = 80
N_SELF_HISTORY = 5

POKEMON_END = N_POKEMON * PER_POKEMON               # 528
MOVE_FLOATS_START = POKEMON_END                      # 528
MOVE_IDX_START = MOVE_FLOATS_START + N_ACTIVE_MOVE_FLOATS  # 588
GLOBAL_START = MOVE_IDX_START + N_ACTIVE_MOVE_INDICES       # 592
HISTORY_START = GLOBAL_START + N_GLOBAL                      # 672
OBSERVATION_SIZE = HISTORY_START + N_SELF_HISTORY             # 677

TYPE_TO_IDX = {
    '': 0, 'normal': 1, 'fire': 2, 'water': 3, 'electric': 4, 'grass': 5,
    'ice': 6, 'fighting': 7, 'poison': 8, 'ground': 9, 'flying': 10,
    'psychic': 11, 'bug': 12, 'rock': 13, 'ghost': 14, 'dragon': 15,
    'dark': 16, 'steel': 17, 'fairy': 18,
}

STATUS_ONEHOT_IDX = {'': 0, 'brn': 1, 'frz': 2, 'par': 3, 'psn': 4, 'slp': 5, 'tox': 6}

STATUS_TO_STR = {
    Status.BRN: 'brn', Status.FRZ: 'frz', Status.PAR: 'par',
    Status.PSN: 'psn', Status.TOX: 'tox', Status.SLP: 'slp',
}

SETUP_MOVES = {
    "swordsdance", "nastyplot", "calmmind", "dragondance",
    "quiverdance", "shellsmash", "bulkup", "irondefense",
    "agility", "rockpolish", "autotomize", "coil", "shiftgear",
    "victorydance", "tidyup", "tailglow", "growth",
}

RECOVERY_MOVES = {
    "recover", "softboiled", "roost", "slackoff", "moonlight",
    "morningsun", "synthesis", "wish", "rest", "shoreup",
    "milkdrink", "healorder", "strengthsap",
}

BOOSTING_MOVES = SETUP_MOVES | {
    "workup", "howl", "meditate", "sharpen", "minimize",
    "defensecurl", "harden", "withdraw", "amnesia", "barrier",
    "cosmicpower", "stockpile", "acidarmor", "cottonguard",
}


def _has_ground_immunity(p_data: dict) -> bool:
    """Check if pokemon is immune to ground (for hazard calc)."""
    types = [t.lower() for t in p_data.get('types', []) if t]
    if 'flying' in types:
        return True
    ability = _normalize_id(p_data.get('ability', ''))
    if ability == 'levitate':
        return True
    item = _normalize_id(p_data.get('item', ''))
    if item == 'airballoon':
        return True
    return False


def _calc_hazard_damage(p_data: dict, side_conditions: dict) -> float:
    """Estimate hazard damage for a pokemon switching in."""
    d = 0.0
    types = [t.lower() for t in p_data.get('types', []) if t]
    # Stealth Rock
    if side_conditions.get('stealthrock', 0) > 0:
        rock_eff = 1.0
        for t in types:
            rock_eff *= TYPE_CHART.get('rock', {}).get(t, 1.0)
        d += 0.125 * rock_eff
    # Spikes
    sp = side_conditions.get('spikes', 0)
    if sp > 0 and not _has_ground_immunity(p_data):
        d += [0, 0.125, 0.167, 0.25][min(sp, 3)]
    # Toxic Spikes
    ts = side_conditions.get('toxicspikes', 0)
    if ts > 0 and not _has_ground_immunity(p_data):
        if 'poison' not in types:
            d += 0.06 * ts
    # Sticky Web
    if side_conditions.get('stickyweb', 0) > 0:
        if not _has_ground_immunity(p_data):
            d += 0.02
    return min(d, 1.0)


def _pokemon_has_moves_of_type(p_data: dict, move_set: set) -> bool:
    """Check if pokemon has any moves in the given set."""
    moves, _ = _parse_moves(p_data)
    return any(_normalize_id(m) in move_set for m in moves if m)


def _normalize_id(name: str) -> str:
    if not name: return ''
    return name.lower().replace(' ', '').replace('-', '').replace("'", '').replace('.', '')


def _get_pokemon_idx(s): return POKEMON_TO_IDX.get(_normalize_id(s), 0)
def _get_move_idx(s): return MOVE_TO_IDX.get(_normalize_id(s), 0)
def _get_ability_idx(s): return ABILITY_TO_IDX.get(_normalize_id(s), 0)
def _get_item_idx(s): return ITEM_TO_IDX.get(_normalize_id(s), 0)
def _get_type_idx(s): return TYPE_TO_IDX.get(s.lower() if s else '', 0)


def _get_move_data(move_id: str) -> dict:
    if GEN_DATA is None: return {}
    return GEN_DATA.moves.get(_normalize_id(move_id), {})


def _get_move_type(move_id: str) -> str:
    return _get_move_data(move_id).get('type', '').lower()


def _pokemon_to_dict(pokemon: Pokemon, is_opponent: bool = False) -> dict:
    result = {
        'species': pokemon.species,
        'level': pokemon.level,
        'ability': pokemon.ability or '',
        'item': pokemon.item or '',
        'status': STATUS_TO_STR.get(pokemon.status, '') if pokemon.status else '',
        'boosts': dict(pokemon.boosts) if pokemon.boosts else {},
        'teraType': (pokemon.tera_type.name if pokemon.tera_type else 'Normal'),
        'terastallized': (pokemon.terastallized if hasattr(pokemon, 'terastallized')
                          and pokemon.terastallized else ''),
        'isActive': pokemon.active,
        'types': [t.name.lower() for t in pokemon.types if t is not None] if pokemon.types else [],
        'baseStats': dict(pokemon.base_stats) if pokemon.base_stats else {},
        'moves': [m.id for m in pokemon.moves.values()] if pokemon.moves else [],
    }

    if is_opponent:
        result['hpFraction'] = pokemon.current_hp_fraction
        if pokemon.base_stats and 'hp' in pokemon.base_stats:
            base_hp = pokemon.base_stats['hp']
            estimated_max = int(((2 * base_hp + 31 + 21) * pokemon.level / 100) + pokemon.level + 10)
            result['maxhp'] = estimated_max
            result['hp'] = max(0, round(estimated_max * pokemon.current_hp_fraction))
    else:
        result['hp'] = pokemon.current_hp if pokemon.current_hp is not None else 0
        result['maxhp'] = pokemon.max_hp if pokemon.max_hp is not None else 1

    return result


def _side_conditions_to_dict(conditions: dict) -> dict:
    result = {}
    for cond, value in conditions.items():
        name = cond.name.lower() if hasattr(cond, 'name') else str(cond).lower()
        name = name.replace('_', '').replace(' ', '')
        result[name] = value if isinstance(value, int) else 1
    return result


def battle_to_search_request(battle: AbstractBattle) -> dict:
    """Convert a poke-env Battle to the dict format for search_server_v2.js."""
    p1_team = []
    active_added = False
    if battle.active_pokemon and not battle.active_pokemon.fainted:
        p1_team.append(_pokemon_to_dict(battle.active_pokemon, is_opponent=False))
        active_added = True
    for pokemon in battle.team.values():
        if active_added and pokemon == battle.active_pokemon:
            continue
        p1_team.append(_pokemon_to_dict(pokemon, is_opponent=False))
    if p1_team:
        p1_team[0]['isActive'] = True

    p2_team = []
    if battle.opponent_active_pokemon and not battle.opponent_active_pokemon.fainted:
        p2_team.append(_pokemon_to_dict(battle.opponent_active_pokemon, is_opponent=True))
    for pokemon in battle.opponent_team.values():
        if battle.opponent_active_pokemon and pokemon == battle.opponent_active_pokemon:
            continue
        p2_team.append(_pokemon_to_dict(pokemon, is_opponent=True))
    if p2_team:
        p2_team[0]['isActive'] = True

    weather = ''
    if battle.weather:
        for w, turns in battle.weather.items():
            weather = w.name.lower() if hasattr(w, 'name') else str(w).lower()
            break

    terrain = ''
    if battle.fields:
        for f, turns in battle.fields.items():
            fn = f.name.lower() if hasattr(f, 'name') else str(f).lower()
            if 'terrain' in fn:
                terrain = fn.replace('_terrain', '').replace('terrain', '')
                break

    return {
        'p1Team': p1_team,
        'p2Team': p2_team,
        'sideConditions': {
            'p1': _side_conditions_to_dict(battle.side_conditions),
            'p2': _side_conditions_to_dict(battle.opponent_side_conditions),
        },
        'weather': weather,
        'terrain': terrain,
        'turn': battle.turn,
    }


def _parse_moves(p_data: dict):
    """Extract move ID list and detail list from pokemon data."""
    moves_raw = p_data.get('moves', [])
    ids, details = [], []
    for m in moves_raw:
        if isinstance(m, dict):
            ids.append(m.get('id', ''))
            details.append(m)
        else:
            ids.append(m)
    if not details:
        details = p_data.get('moveDetails', [])
    return ids, details


def _embed_pokemon(p_data: dict, is_opponent: bool = False) -> np.ndarray:
    """
    Encode one pokemon as 44 dims (13 indices + 31 floats).
    Matches _encode_pokemon_block() in RLPlayer6v6_v18.py.
    """
    vec = np.zeros(PER_POKEMON, dtype=np.float32)
    moves, move_details = _parse_moves(p_data)

    # INDICES (13)
    vec[0] = _get_pokemon_idx(p_data.get('species', ''))
    for i in range(4):
        vec[1 + i] = _get_move_idx(moves[i]) if i < len(moves) else 0
    vec[5] = _get_ability_idx(p_data.get('ability', ''))
    vec[6] = _get_item_idx(p_data.get('item', ''))
    types = p_data.get('types', [])
    vec[7] = _get_type_idx(types[0]) if len(types) > 0 else 0
    vec[8] = _get_type_idx(types[1]) if len(types) > 1 else 0
    for i in range(4):
        vec[9 + i] = _get_type_idx(_get_move_type(moves[i])) if i < len(moves) else 0

    # FLOATS (31), starting at index 13
    fo = PER_POKEMON_INDICES  # 13
    bs = p_data.get('baseStats', p_data.get('basestats', {}))
    for j, stat in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
        vec[fo + j] = bs.get(stat, 80) / 255.0

    # HP fraction [fo+6]
    if 'hpFraction' in p_data:
        vec[fo + 6] = p_data['hpFraction']
    elif 'hp' in p_data and 'maxhp' in p_data and p_data['maxhp'] > 0:
        vec[fo + 6] = p_data['hp'] / p_data['maxhp']
    else:
        vec[fo + 6] = 1.0

    # Status one-hot (7) [fo+7 : fo+14]
    status = (p_data.get('status', '') or '').lower()
    svec = np.zeros(7, dtype=np.float32)
    svec[STATUS_ONEHOT_IDX.get(status, 0)] = 1.0
    vec[fo + 7: fo + 14] = svec

    # Boosts /6 (7) [fo+14 : fo+21]
    boosts = p_data.get('boosts', {})
    for j, stat in enumerate(['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']):
        vec[fo + 14 + j] = boosts.get(stat, 0) / 6.0

    # Is alive [fo+21]
    fainted = p_data.get('fainted', False)
    hp_val = p_data.get('hp', 1)
    vec[fo + 21] = 0.0 if (fainted or hp_val <= 0) else 1.0

    # Volatiles (6) [fo+22 : fo+28]
    vols = set(v.lower() for v in p_data.get('volatiles', []))
    vec[fo + 22] = 1.0 if 'leechseed' in vols else 0.0
    vec[fo + 23] = 1.0 if 'confusion' in vols else 0.0
    vec[fo + 24] = 1.0 if 'substitute' in vols else 0.0
    vec[fo + 25] = 1.0 if 'taunt' in vols else 0.0
    vec[fo + 26] = 1.0 if 'encore' in vols else 0.0
    vec[fo + 27] = 1.0 if ('trapped' in vols or 'partiallytrapped' in vols) else 0.0

    # Toxic counter [fo+28]
    vec[fo + 28] = p_data.get('toxicCounter', 0) / 16.0

    # Avg move PP [fo+29]
    if move_details:
        pp_fracs = [md.get('pp', md.get('maxpp', 1)) / max(md.get('maxpp', 1), 1)
                    for md in move_details]
        vec[fo + 29] = np.mean(pp_fracs)
    else:
        vec[fo + 29] = 1.0

    # Move knowledge [fo+30] — the 31st float
    if is_opponent:
        vec[fo + 30] = len([m for m in moves if m]) / 4.0
    else:
        vec[fo + 30] = 1.0

    return vec


def _embed_active_moves(our_active: dict, opp_active: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Active move details: 60 floats (4 x 15) + 4 index floats.
    Matches _encode_active_move_details() in RLPlayer6v6_v18.py.
    """
    mf = np.zeros(N_ACTIVE_MOVE_FLOATS, dtype=np.float32)
    mi = np.zeros(N_ACTIVE_MOVE_INDICES, dtype=np.float32)

    our_types = [t.lower() for t in our_active.get('types', []) if t]
    opp_types = [t.lower() for t in opp_active.get('types', []) if t]
    our_bs = our_active.get('baseStats', our_active.get('basestats', {}))
    opp_bs = opp_active.get('baseStats', opp_active.get('basestats', {}))

    moves, move_details = _parse_moves(our_active)

    for i in range(4):
        o = i * 15
        if i >= len(moves) or not moves[i]:
            # Empty slot defaults (must match RLPlayer6v6_v18.py)
            mf[o:o+15] = [0.0, 1.0, 0.0, 0.0, 1.0,
                           0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
            continue

        move_id = moves[i]
        md = _get_move_data(move_id)
        if not md:
            mf[o:o+15] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
            continue

        mi[i] = _get_move_idx(move_id)
        bp = md.get('basePower', 0)
        acc = md.get('accuracy', True)
        cat = md.get('category', 'Status')
        mtype = md.get('type', '').lower()
        prio = md.get('priority', 0)

        mf[o + 0] = bp / 150.0
        mf[o + 1] = (acc / 100.0) if isinstance(acc, (int, float)) else 1.0
        mf[o + 2] = 1.0 if cat == 'Physical' else 0.0
        mf[o + 3] = 1.0 if cat == 'Special' else 0.0
        mf[o + 4] = 1.0 if cat == 'Status' else 0.0

        eff = get_type_effectiveness(mtype, opp_types) if opp_types else 1.0
        mf[o + 5] = 1.0 if eff == 0 else 0.0
        mf[o + 6] = 1.0 if 0 < eff < 1 else 0.0
        mf[o + 7] = 1.0 if eff == 1 else 0.0
        mf[o + 8] = 1.0 if eff > 1 else 0.0

        mf[o + 9] = 1.0 if mtype in our_types else 0.0
        mf[o + 10] = 1.0 if prio > 0 else 0.0
        mf[o + 11] = 1.0 if prio < 0 else 0.0

        if cat == 'Physical':
            a, d = our_bs.get('atk', 100), opp_bs.get('def', 100)
        elif cat == 'Special':
            a, d = our_bs.get('spa', 100), opp_bs.get('spd', 100)
        else:
            a, d = 100, 100
        mf[o + 12] = a / max(a + d, 1)

        mf[o + 13] = 1.0 if _normalize_id(move_id) in SETUP_MOVES else 0.0

        if i < len(move_details):
            det = move_details[i]
            maxpp = det.get('maxpp', 1)
            mf[o + 14] = det.get('pp', maxpp) / max(maxpp, 1)
        else:
            mf[o + 14] = 1.0

    return mf, mi


def _embed_global(state, p1_data, p2_data, sim_context=None):
    """
    Global features (80 dims). Matches _encode_global() in v18.
    
    sim_context (optional): dict with simulated action info and live state.
    """
    g = np.zeros(N_GLOBAL, dtype=np.float32)
    ctx = sim_context or {}

    field = state.get('field', {})
    weather_str = (field.get('weather', '') if isinstance(field, dict) else '') or state.get('weather', '')
    terrain_str = (field.get('terrain', '') if isinstance(field, dict) else '') or state.get('terrain', '')

    p1_pokemon = p1_data.get('pokemon', [])
    p2_pokemon = p2_data.get('pokemon', [])
    our_active = next((p for p in p1_pokemon if p.get('isActive')), None)
    opp_active = next((p for p in p2_pokemon if p.get('isActive')), None)

    wmap = {'sunnyday': 1, 'desolateland': 1, 'raindance': 2, 'primordialsea': 2,
            'sandstorm': 3, 'snow': 4, 'hail': 4, 'deltastream': 5}
    wn = (weather_str or '').lower().replace(' ', '').replace('_', '')
    if not weather_str: g[0] = 1.0
    else: g[min(wmap.get(wn, 0), 5)] = 1.0

    tmap = {'electricterrain': 1, 'electric': 1, 'grassyterrain': 2, 'grassy': 2,
            'mistyterrain': 3, 'misty': 3, 'psychicterrain': 4, 'psychic': 4}
    tn = (terrain_str or '').lower().replace(' ', '').replace('_', '')
    if not terrain_str: g[6] = 1.0
    else: g[6 + min(tmap.get(tn, 0), 4)] = 1.0

    sc = state.get('sideConditions', {})
    p1_sc = sc.get('p1', p1_data.get('sideConditions', {}))
    p2_sc = sc.get('p2', p2_data.get('sideConditions', {}))

    g[11] = min(p1_sc.get('spikes', 0), 3) / 3.0
    g[12] = 1.0 if p1_sc.get('stealthrock', 0) > 0 else 0.0
    g[13] = min(p1_sc.get('toxicspikes', 0), 2) / 2.0
    g[14] = 1.0 if p1_sc.get('stickyweb', 0) > 0 else 0.0
    g[15] = min(p2_sc.get('spikes', 0), 3) / 3.0
    g[16] = 1.0 if p2_sc.get('stealthrock', 0) > 0 else 0.0
    g[17] = min(p2_sc.get('toxicspikes', 0), 2) / 2.0
    g[18] = 1.0 if p2_sc.get('stickyweb', 0) > 0 else 0.0

    g[19] = 1.0 if p1_sc.get('reflect', 0) > 0 else 0.0
    g[20] = 1.0 if p1_sc.get('lightscreen', 0) > 0 else 0.0
    g[21] = 1.0 if p2_sc.get('reflect', 0) > 0 else 0.0
    g[22] = 1.0 if p2_sc.get('lightscreen', 0) > 0 else 0.0

    turn = state.get('turn', 1)
    g[23] = min(turn / 100.0, 1.0)

    any_p1_tera = any(p.get('terastallized', '') for p in p1_pokemon)
    any_p2_tera = any(p.get('terastallized', '') for p in p2_pokemon)
    g[24] = 0.0 if any_p1_tera else 1.0
    g[25] = 0.0 if any_p2_tera else 1.0
    g[26] = 1.0 if (our_active and our_active.get('terastallized', '')) else 0.0
    g[27] = len(p2_pokemon) / 6.0

    # Switch context (5) [28:33] — computed from sim_context
    p1_action = ctx.get('p1_action', '')
    was_switch = 'switch' in p1_action
    prev_turns = ctx.get('prev_turns_since_switch', 1)
    g[28] = 1.0 if was_switch else 0.0
    g[29] = 0.0 if was_switch else min((prev_turns + 1) / 10.0, 1.0)
    g[30] = 0.0  # oscillation

    bench = [p for p in p1_pokemon if not p.get('isActive', False) and not p.get('fainted', False)]
    g[31] = sum(_calc_hazard_damage(m, p1_sc) for m in bench) / max(len(bench), 1) if bench else 0.0

    best_bm = 0.0
    if opp_active:
        opp_types = [t.lower() for t in opp_active.get('types', []) if t]
        for mon in bench:
            mon_types = [t.lower() for t in mon.get('types', []) if t]
            if mon_types and opp_types:
                o_eff = max((get_type_effectiveness(mt, opp_types) for mt in mon_types), default=1.0)
                d_eff = max((get_type_effectiveness(ot, mon_types) for ot in opp_types), default=1.0)
                best_bm = max(best_bm, min((o_eff/4)-(d_eff/4)+0.5, 1.0))
    g[32] = best_bm

    g[33] = 1.0 if turn <= 5 else 0.0
    g[34] = 1.0 if 5 < turn <= 20 else 0.0
    g[35] = 1.0 if turn > 20 else 0.0

    p1a = p1_data.get('alive', sum(1 for p in p1_pokemon if not p.get('fainted', False)))
    p2a = p2_data.get('alive', sum(1 for p in p2_pokemon if not p.get('fainted', False)))
    g[36] = p1a / 6.0; g[37] = p2a / 6.0; g[38] = (p1a - p2a + 6) / 12.0

    if our_active and opp_active:
        our_spe = our_active.get('baseStats', our_active.get('basestats', {})).get('spe', 80)
        opp_spe = opp_active.get('baseStats', opp_active.get('basestats', {})).get('spe', 80)
        ob = our_active.get('boosts', {}).get('spe', 0)
        pb = opp_active.get('boosts', {}).get('spe', 0)
        # Match v18: (2+boost)/2, NO max — negative boosts reduce speed
        oe = our_spe * (2 + ob) / 2
        pe = opp_spe * (2 + pb) / 2
        if (our_active.get('status', '') or '').lower() in ('par',): oe *= 0.5
        if (opp_active.get('status', '') or '').lower() in ('par',): pe *= 0.5
        g[39] = oe / max(oe + pe, 1)
    else:
        g[39] = 0.5

    g[40] = 1.0 if bench else 0.0

    # Setup opportunity (3) [41:44]
    opp_passive = 0.0
    if our_active and opp_active:
        our_types = [t.lower() for t in our_active.get('types', []) if t]
        opp_types = [t.lower() for t in opp_active.get('types', []) if t]
        if our_types and opp_types:
            if max((get_type_effectiveness(ot, our_types) for ot in opp_types), default=1.0) < 1.0: opp_passive += 0.2
            if max((get_type_effectiveness(mt, opp_types) for mt in our_types), default=1.0) > 1.5: opp_passive += 0.3
        opp_hp = opp_active.get('hpFraction', opp_active.get('hp', 1) / max(opp_active.get('maxhp', 1), 1))
        if opp_hp < 0.3: opp_passive += 0.2
        if opp_active.get('status', ''): opp_passive += 0.15
    g[41] = min(opp_passive, 1.0)
    g[42] = 1.0 if (our_active and _pokemon_has_moves_of_type(our_active, SETUP_MOVES)) else 0.0
    g[43] = min(sum(max(0, v) for v in (our_active or {}).get('boosts', {}).values()) / 6.0, 1.0)

    # Healing (3) [44:47]
    hf = our_active.get('hpFraction', our_active.get('hp', 1) / max(our_active.get('maxhp', 1), 1)) if our_active else 1.0
    hm = 1.0 - hf
    g[44] = hm
    g[45] = 1.0 if (our_active and _pokemon_has_moves_of_type(our_active, RECOVERY_MOVES)) else 0.0
    g[46] = hm if 0.2 < hf < 0.7 else 0.0

    if our_active:
        it = _normalize_id(our_active.get('item', ''))
        g[47] = 1.0 if 'choice' in it else 0.0
        g[48] = 1.0 if it == 'lifeorb' else 0.0
        g[49] = 1.0 if it == 'heavydutyboots' else 0.0
        g[50] = 1.0 if it == 'leftovers' else 0.0
    if opp_active:
        oi = _normalize_id(opp_active.get('item', ''))
        g[51] = 1.0 if ('choice' in oi and oi not in ('', 'unknownitem')) else 0.0

    if our_active:
        ab = _normalize_id(our_active.get('ability', ''))
        g[52] = 1.0 if ab == 'levitate' else 0.0
        g[53] = 1.0 if ab == 'intimidate' else 0.0
    if opp_active:
        g[54] = 1.0 if _normalize_id(opp_active.get('ability', '')) == 'sturdy' else 0.0

    # Bench offensive matchup (5) [55:60]
    for i in range(5):
        if i < len(bench) and opp_active:
            opp_types = [t.lower() for t in opp_active.get('types', []) if t]
            best_eff = 0.0
            moves, _ = _parse_moves(bench[i])
            for mid in moves:
                if not mid: continue
                mt = _get_move_type(mid)
                if mt and opp_types: best_eff = max(best_eff, get_type_effectiveness(mt, opp_types))
            g[55 + i] = min(best_eff / 4.0, 1.0)
        else:
            g[55 + i] = 0.0

    # Bench defensive matchup (5) [60:65]
    for i in range(5):
        if i < len(bench) and opp_active:
            ot = [t.lower() for t in opp_active.get('types', []) if t]
            bt = [t.lower() for t in bench[i].get('types', []) if t]
            if ot and bt: g[60 + i] = min(max(get_type_effectiveness(at, bt) for at in ot) / 4.0, 1.0)
            else: g[60 + i] = 0.5
        else:
            g[60 + i] = 0.5

    # Opponent action history (15) [65:80]
    prev_history = ctx.get('prev_opp_history', [])
    p2_action = ctx.get('p2_action', '')
    if p2_action:
        sim_entry = {'type': 'move', 'move_type_idx': 0, 'was_boosting': 0.0}
        if 'switch' in p2_action: sim_entry['type'] = 'switch'
        elif opp_active:
            try:
                mn = int(p2_action.split()[-1]) - 1
                om, _ = _parse_moves(opp_active)
                if 0 <= mn < len(om) and om[mn]:
                    sim_entry['move_type_idx'] = TYPE_TO_IDX.get(_get_move_type(om[mn]), 0)
                    if _normalize_id(om[mn]) in BOOSTING_MOVES:
                        sim_entry['type'] = 'status_move'; sim_entry['was_boosting'] = 1.0
            except (ValueError, IndexError): pass
        new_history = [sim_entry] + prev_history[:2]
    else:
        new_history = prev_history[:3]

    for hi in range(3):
        base = 65 + hi * 5
        if hi < len(new_history):
            e = new_history[hi]
            at = e.get('type', 'move')
            g[base] = 1.0 if at == 'move' else 0.0
            g[base+1] = 1.0 if at == 'switch' else 0.0
            g[base+2] = 1.0 if at == 'status_move' else 0.0
            g[base+3] = e.get('move_type_idx', 0) / 19.0
            g[base+4] = e.get('was_boosting', 0.0)

    return g



def state_to_observation(rich_state: dict, original_battle: Optional[AbstractBattle] = None,
                         sim_context: Optional[dict] = None) -> np.ndarray:
    """
    Convert search server result dict into a 677-dim observation.
    
    sim_context (optional): dict with simulated action info for filling in
    tracking features that depend on knowing what action was taken.
    Keys: p1_action, p2_action, prev_turns_since_switch, prev_opp_history,
          hazards_set_by_us, total_boosts_used, active_moves_used, turns_on_field
    """
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    ctx = sim_context or {}

    p1_data = rich_state.get('p1', {})
    p2_data = rich_state.get('p2', {})
    p1_pokemon = p1_data.get('pokemon', [])
    p2_pokemon = p2_data.get('pokemon', [])

    # Normalize move data
    for plist in [p1_pokemon, p2_pokemon]:
        for p in plist:
            moves_raw = p.get('moves', [])
            if moves_raw and isinstance(moves_raw[0], dict):
                p['moveDetails'] = moves_raw
                p['moves'] = [m.get('id', '') for m in moves_raw]

    # Pokemon blocks (12 x 44 = 528)
    our = sorted(p1_pokemon, key=lambda p: 0 if p.get('isActive') else 1)
    for i in range(6):
        off = i * PER_POKEMON
        if i < len(our):
            obs[off: off + PER_POKEMON] = _embed_pokemon(our[i], is_opponent=False)

    opp = sorted(p2_pokemon, key=lambda p: 0 if p.get('isActive') else 1)
    for i in range(6):
        off = (6 + i) * PER_POKEMON
        if i < len(opp):
            obs[off: off + PER_POKEMON] = _embed_pokemon(opp[i], is_opponent=True)

    # Active moves (60 + 4)
    our_act = our[0] if our else {}
    opp_act = opp[0] if opp else {}
    mf, mi = _embed_active_moves(our_act, opp_act)
    obs[MOVE_FLOATS_START: MOVE_FLOATS_START + N_ACTIVE_MOVE_FLOATS] = mf
    obs[MOVE_IDX_START: MOVE_IDX_START + N_ACTIVE_MOVE_INDICES] = mi

    # Global (80) — pass sim_context for tracking features
    obs[GLOBAL_START: GLOBAL_START + N_GLOBAL] = _embed_global(rich_state, p1_data, p2_data, ctx)

    # Self-history (5) — computed from sim_context + active mon state
    p1_action = ctx.get('p1_action', '')
    was_switch = 'switch' in p1_action
    prev_turns = ctx.get('prev_turns_since_switch', 1)

    # turns_on_field / 10
    obs[HISTORY_START + 0] = 0.0 if was_switch else min((prev_turns + 1) / 10.0, 1.0)

    # low_pp_moves_frac — from active mon's move details
    if our_act:
        _, move_details = _parse_moves(our_act)
        low_pp = 0
        for md in move_details:
            maxpp = md.get('maxpp', 1)
            if maxpp > 0 and md.get('pp', maxpp) / maxpp < 0.25:
                low_pp += 1
        n_moves = max(len(move_details), 1)
        obs[HISTORY_START + 1] = low_pp / n_moves
    
    # boosts_used / 10
    obs[HISTORY_START + 2] = min(ctx.get('total_boosts_used', 0) / 10.0, 1.0)

    # active_moves_used_frac
    # v18 resets _active_moves_used on switch
    p1_act = ctx.get('p1_action', '')
    if 'switch' in p1_act:
        n_active_used = 0  # Reset on switch
    else:
        n_active_used = len(ctx.get('active_moves_used', set()))
    if our_act:
        moves, _ = _parse_moves(our_act)
        obs[HISTORY_START + 3] = n_active_used / max(len([m for m in moves if m]), 1)

    # hazards_set_by_us
    obs[HISTORY_START + 4] = 1.0 if ctx.get('hazards_set_by_us', False) else 0.0

    return obs