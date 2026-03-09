from typing import Dict, List, Optional, Any
from poke_env.battle.abstract_battle import AbstractBattle

# poke-env has different import paths across versions.
# Try the newer poke_env.battle.* paths first, fall back to poke_env.environment.*
try:
    from poke_env.battle.pokemon import Pokemon
    from poke_env.battle.move import Move
    from poke_env.battle.side_condition import SideCondition
    from poke_env.battle.weather import Weather
    from poke_env.battle.field import Field
    from poke_env.battle.status import Status
except ImportError:
    from poke_env.environment.pokemon import Pokemon
    from poke_env.environment.move import Move
    from poke_env.environment.side_condition import SideCondition
    from poke_env.environment.weather import Weather
    from poke_env.environment.field import Field
    from poke_env.environment.status import Status


SIDE_CONDITION_MAP = {
    SideCondition.AURORA_VEIL: "auroraveil",
    SideCondition.LIGHT_SCREEN: "lightscreen",
    SideCondition.REFLECT: "reflect",
    SideCondition.SPIKES: "spikes",
    SideCondition.STEALTH_ROCK: "stealthrock",
    SideCondition.STICKY_WEB: "stickyweb",
    SideCondition.TAILWIND: "tailwind",
    SideCondition.TOXIC_SPIKES: "toxicspikes",
}

WEATHER_MAP = {
    Weather.SUNNYDAY: "sunnyday",
    Weather.DESOLATELAND: "desolateland",
    Weather.RAINDANCE: "raindance",
    Weather.PRIMORDIALSEA: "primordialsea",
    Weather.SANDSTORM: "sandstorm",
    Weather.HAIL: "hail",
    Weather.SNOW: "snow",
}

FIELD_MAP = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
    Field.TRICK_ROOM: "trickroom",
    Field.GRAVITY: "gravity",
}

STATUS_MAP = {
    Status.BRN: "brn",
    Status.FRZ: "frz",
    Status.PAR: "par",
    Status.PSN: "psn",
    Status.SLP: "slp",
    Status.TOX: "tox",
}


def _estimate_max_hp(pokemon: Pokemon) -> int:
    """
    Estimate a Pokemon's max HP from base stats and level.
    
    For random battles, the formula is:
      HP = floor((2 * base + IV + floor(EV/4)) * level / 100) + level + 10
    
    Random battles typically use 31 IVs and 85 EVs per stat.
    Shedinja always has 1 HP.
    """
    if pokemon.species and "shedinja" in pokemon.species.lower():
        return 1
    
    base_hp = pokemon.base_stats.get("hp", 80) if pokemon.base_stats else 80
    level = pokemon.level or 100
    
    # Random battle defaults: 31 IVs, 85 EVs
    iv = 31
    ev = 85
    
    max_hp = int((2 * base_hp + iv + ev // 4) * level / 100) + level + 10
    return max(1, max_hp)


def _serialize_pokemon(pokemon: Pokemon, is_opponent: bool = False) -> Dict[str, Any]:
    """
    Serialize a single Pokemon's state for the search server.
    
    For our Pokemon: full info available.
    For opponent Pokemon: partial info — fill what we know, mark unknowns.
    """
    species = pokemon.species or "unknown"
    level = pokemon.level or 100

    moves = []
    if pokemon.moves:
        for move_id, move_obj in pokemon.moves.items():
            moves.append(move_id)
    
    ability = pokemon.ability or ""
    item = pokemon.item or ""
    # poke-env uses "unknown_item" or similar when not revealed
    if item and "unknown" in item.lower():
        item = ""
    
    if is_opponent:
        # Opponent HP is a fraction — estimate actual values
        estimated_max = _estimate_max_hp(pokemon)
        hp_fraction = pokemon.current_hp_fraction if pokemon.current_hp_fraction is not None else 1.0
        hp = max(0, int(hp_fraction * estimated_max))
        maxhp = estimated_max
    else:
        # Our Pokemon: exact HP known
        hp = pokemon.current_hp if pokemon.current_hp is not None else 0
        maxhp = pokemon.max_hp if pokemon.max_hp is not None else 1
    
    status = ""
    if pokemon.status is not None:
        status = STATUS_MAP.get(pokemon.status, "")
    
    boosts = pokemon.boosts if pokemon.boosts else {}
    boosts_normalized = {
        "atk": boosts.get("atk", 0),
        "def": boosts.get("def", 0),
        "spa": boosts.get("spa", 0),
        "spd": boosts.get("spd", 0),
        "spe": boosts.get("spe", 0),
        "accuracy": boosts.get("accuracy", 0),
        "evasion": boosts.get("evasion", 0),
    }
    
    tera_type = ""
    if pokemon.tera_type is not None:
        tera_type = pokemon.tera_type.name.capitalize() if hasattr(pokemon.tera_type, 'name') else str(pokemon.tera_type)
    
    is_active = pokemon.active
    fainted = pokemon.fainted
    
    result = {
        "species": species,
        "level": level,
        "moves": moves,
        "ability": ability,
        "item": item,
        "hp": hp,
        "maxhp": maxhp,
        "status": status,
        "boosts": boosts_normalized,
        "teraType": tera_type,
        "isActive": is_active,
        "fainted": fainted,
        "isOpponent": is_opponent,
    }
    
    return result


def _serialize_side_conditions(side_conditions: Dict) -> Dict[str, int]:
    """Convert poke-env side conditions to {name: layer_count} dict."""
    result = {}
    for condition, value in side_conditions.items():
        name = SIDE_CONDITION_MAP.get(condition)
        if name:
            # value is usually an int (layer count) or a tuple
            if isinstance(value, (list, tuple)):
                result[name] = value[0] if value else 1
            elif isinstance(value, int):
                result[name] = value
            else:
                result[name] = 1
    return result


def _get_weather(battle: AbstractBattle) -> str:
    """Extract current weather as a string."""
    if battle.weather:
        for w, _ in battle.weather.items():
            return WEATHER_MAP.get(w, "")
    return ""


def _get_terrain(battle: AbstractBattle) -> str:
    """Extract current terrain/field effects as a string."""
    if battle.fields:
        for f, _ in battle.fields.items():
            mapped = FIELD_MAP.get(f, "")
            if mapped and mapped != "trickroom" and mapped != "gravity":
                return mapped
    return ""


def _get_pseudo_fields(battle: AbstractBattle) -> Dict[str, bool]:
    """Extract pseudo-weather effects like Trick Room and Gravity."""
    result = {}
    if battle.fields:
        for f, _ in battle.fields.items():
            if f == Field.TRICK_ROOM:
                result["trickroom"] = True
            elif f == Field.GRAVITY:
                result["gravity"] = True
    return result


def battle_to_search_request(
    battle: AbstractBattle,
    p1_actions: List[str],
    n_opponent_samples: int = 3,
) -> Dict[str, Any]:
    """
    Convert a poke-env AbstractBattle into a search server request.
    
    This is the main entry point for the state bridge.
    
    Args:
        battle: The current battle state from poke-env
        p1_actions: List of Showdown choice strings to evaluate
                    (e.g. ["move 1", "move 2", "switch 2"])
        n_opponent_samples: How many opponent move samples to average over
    
    Returns:
        Dict ready to POST to the search server's /evaluate endpoint
    """
    p1_team = []
    
    # Active Pokemon first
    if battle.active_pokemon:
        p1_team.append(_serialize_pokemon(battle.active_pokemon, is_opponent=False))
    
    # Bench Pokemon
    if battle.team:
        for mon_id, mon in battle.team.items():
            if not mon.active:
                p1_team.append(_serialize_pokemon(mon, is_opponent=False))
    
    p2_team = []
    
    # Active Pokemon first
    if battle.opponent_active_pokemon:
        p2_team.append(_serialize_pokemon(battle.opponent_active_pokemon, is_opponent=True))
    
    # Revealed bench Pokemon
    if battle.opponent_team:
        for mon_id, mon in battle.opponent_team.items():
            if not mon.active:
                p2_team.append(_serialize_pokemon(mon, is_opponent=True))
    
    # If no opponent info at all, we can't search — return None-safe marker
    if not p2_team:
        return None
    
    p1_side = _serialize_side_conditions(battle.side_conditions)
    p2_side = _serialize_side_conditions(battle.opponent_side_conditions)
    
    weather = _get_weather(battle)
    terrain = _get_terrain(battle)
    pseudo_fields = _get_pseudo_fields(battle)
    
    request = {
        "p1Team": p1_team,
        "p2Team": p2_team,
        "sideConditions": {
            "p1": p1_side,
            "p2": p2_side,
        },
        "weather": weather,
        "terrain": terrain,
        "pseudoFields": pseudo_fields,
        "turn": battle.turn,
        "p1Actions": p1_actions,
        "nOpponentSamples": n_opponent_samples,
    }
    
    return request


def get_showdown_actions(battle: AbstractBattle, action_mask=None) -> List[tuple]:
    """
    Get all legal actions as (action_index, showdown_choice_string) pairs.
    
    Maps our 13-action space to Showdown choice strings:
      0-3:  "move 1" through "move 4"
      4-7:  "move 1 terastallize" through "move 4 terastallize"
      8-12: "switch <species>"
    
    Args:
        battle: Current battle state
        action_mask: Optional 13-dim mask (from get_action_mask). 
                     If None, derives from battle directly.
    
    Returns:
        List of (action_index, choice_string) for all legal actions
    """
    moves = battle.available_moves or []
    switches = battle.available_switches or []
    can_tera = battle.can_tera
    
    actions = []
    
    for i, move in enumerate(moves):
        if i >= 4:
            break
        if action_mask is not None and action_mask[i] <= 0:
            continue
        actions.append((i, f"move {i + 1}"))
    
    if can_tera:
        for i, move in enumerate(moves):
            if i >= 4:
                break
            if action_mask is not None and action_mask[4 + i] <= 0:
                continue
            actions.append((4 + i, f"move {i + 1} terastallize"))
    
    for i, switch_mon in enumerate(switches):
        if i >= 5:
            break
        if action_mask is not None and action_mask[8 + i] <= 0:
            continue
        # Use species name for switch (Showdown accepts both slot number and species)
        actions.append((8 + i, f"switch {switch_mon.species}"))
    
    return actions