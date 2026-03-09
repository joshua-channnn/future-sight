import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from poke_env.player import Player, BattleOrder, DefaultBattleOrder, SingleBattleOrder
from poke_env.battle.abstract_battle import AbstractBattle

from players.ppo_player import PPOPlayer

from poke_engine import (
    State, Side, Pokemon, Move, SideConditions,
    PokemonIndex,
    mcts,
)

# poke-env enums (try both import paths)
try:
    from poke_env.battle.side_condition import SideCondition
    from poke_env.battle.weather import Weather
    from poke_env.battle.field import Field
    from poke_env.battle.status import Status
except ImportError:
    from poke_env.environment.side_condition import SideCondition
    from poke_env.environment.weather import Weather
    from poke_env.environment.field import Field
    from poke_env.environment.status import Status


def _norm(name: str) -> str:
    if not name:
        return ""
    return name.lower().replace(" ", "").replace("-", "").replace("'", "").replace(".", "").replace(":", "")


STATUS_MAP = {
    Status.BRN: "Burn",
    Status.FRZ: "Freeze",
    Status.PAR: "Paralyze",
    Status.PSN: "Poison",
    Status.SLP: "Sleep",
    Status.TOX: "Toxic",
}

WEATHER_MAP = {
    Weather.SUNNYDAY: "sun",
    Weather.DESOLATELAND: "harshsun",
    Weather.RAINDANCE: "rain",
    Weather.PRIMORDIALSEA: "heavyrain",
    Weather.SANDSTORM: "sand",
    Weather.HAIL: "hail",
    Weather.SNOW: "snow",
}

FIELD_TO_TERRAIN = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
}

TYPE_NAMES = {
    "NORMAL": "normal", "FIRE": "fire", "WATER": "water",
    "ELECTRIC": "electric", "GRASS": "grass", "ICE": "ice",
    "FIGHTING": "fighting", "POISON": "poison", "GROUND": "ground",
    "FLYING": "flying", "PSYCHIC": "psychic", "BUG": "bug",
    "ROCK": "rock", "GHOST": "ghost", "DRAGON": "dragon",
    "DARK": "dark", "STEEL": "steel", "FAIRY": "fairy",
}


def _type_str(poke_type) -> str:
    if poke_type is None:
        return "typeless"
    name = poke_type.name if hasattr(poke_type, 'name') else str(poke_type)
    return TYPE_NAMES.get(name.upper(), name.lower())


def _estimate_stat(base: int, level: int, iv: int = 31, ev: int = 85, is_hp: bool = False) -> int:
    """Estimate a stat value for random battles (neutral nature)."""
    if is_hp:
        return int((2 * base + iv + ev // 4) * level / 100) + level + 10
    else:
        return int(((2 * base + iv + ev // 4) * level / 100) + 5)


def _convert_pokemon(pe_mon, is_opponent: bool = False) -> Pokemon:
    """Convert a poke-env Pokemon to a poke_engine Pokemon."""
    species = _norm(pe_mon.species) if pe_mon.species else "pikachu"
    level = pe_mon.level or 100

    types_list = []
    if pe_mon.types:
        for t in pe_mon.types:
            if t is not None:
                types_list.append(_type_str(t))
    if not types_list:
        types_list = ["normal"]
    types_tuple = tuple(types_list[:2]) if len(types_list) >= 2 else (types_list[0], "typeless")

    base = pe_mon.base_stats if pe_mon.base_stats else {}
    if is_opponent:
        maxhp = _estimate_stat(base.get("hp", 80), level, is_hp=True)
        hp_frac = pe_mon.current_hp_fraction if pe_mon.current_hp_fraction is not None else 1.0
        hp = max(0, int(hp_frac * maxhp))
        if pe_mon.fainted:
            hp = 0
        attack = _estimate_stat(base.get("atk", 80), level)
        defense = _estimate_stat(base.get("def", 80), level)
        special_attack = _estimate_stat(base.get("spa", 80), level)
        special_defense = _estimate_stat(base.get("spd", 80), level)
        speed = _estimate_stat(base.get("spe", 80), level)
    else:
        maxhp = pe_mon.max_hp if pe_mon.max_hp else 100
        hp = pe_mon.current_hp if pe_mon.current_hp is not None else 0
        if pe_mon.fainted:
            hp = 0
        # poke-env provides computed stats for our team
        stats = pe_mon.stats if hasattr(pe_mon, 'stats') and pe_mon.stats else {}
        attack = stats.get("atk", _estimate_stat(base.get("atk", 80), level))
        defense = stats.get("def", _estimate_stat(base.get("def", 80), level))
        special_attack = stats.get("spa", _estimate_stat(base.get("spa", 80), level))
        special_defense = stats.get("spd", _estimate_stat(base.get("spd", 80), level))
        speed = stats.get("spe", _estimate_stat(base.get("spe", 80), level))

    ability = _norm(pe_mon.ability) if pe_mon.ability else "None"
    if "unknown" in ability:
        ability = "None"

    item = _norm(pe_mon.item) if pe_mon.item else "None"
    if "unknown" in item:
        item = "None"

    # Status - poke-engine expects: "None", "Burn", "Freeze", "Paralyze", "Poison", "Sleep", "Toxic"
    status = "None"
    if pe_mon.status is not None:
        status = STATUS_MAP.get(pe_mon.status, "None")
        if status == "None" and pe_mon.status is not None:
            status_str = str(pe_mon.status).lower().replace("status.", "").strip()
            status_lookup = {
                "brn": "Burn", "frz": "Freeze", "par": "Paralyze",
                "psn": "Poison", "slp": "Sleep", "tox": "Toxic",
                "burn": "Burn", "freeze": "Freeze", "paralysis": "Paralyze",
                "poison": "Poison", "sleep": "Sleep", "toxic": "Toxic",
            }
            status = status_lookup.get(status_str, "None")

    moves = []
    if pe_mon.moves:
        for move_id, move_obj in pe_mon.moves.items():
            pp = move_obj.current_pp if hasattr(move_obj, 'current_pp') and move_obj.current_pp is not None else 16
            moves.append(Move(id=_norm(move_id), pp=pp))

    # Pad to 4 moves if needed (opponent may have fewer revealed)
    # Use 'splash' instead of 'none' — poke-engine needs real move IDs
    while len(moves) < 4:
        moves.append(Move(id="splash", pp=16, disabled=False))
    moves = moves[:4]

    tera_type = "normal"
    if pe_mon.tera_type is not None:
        tera_type = _type_str(pe_mon.tera_type)

    terastallized = False
    if hasattr(pe_mon, 'terastallized') and pe_mon.terastallized:
        terastallized = True

    weight = pe_mon.weight if hasattr(pe_mon, 'weight') and pe_mon.weight else 0.0

    return Pokemon(
        id=species,
        level=level,
        types=types_tuple,
        base_types=types_tuple,
        hp=hp,
        maxhp=maxhp,
        ability=ability,
        base_ability=ability,
        item=item,
        nature="serious",
        evs=(85, 85, 85, 85, 85, 85),
        attack=attack,
        defense=defense,
        special_attack=special_attack,
        special_defense=special_defense,
        speed=speed,
        status=status,
        weight_kg=weight,
        moves=moves,
        terastallized=terastallized,
        tera_type=tera_type,
    )


def _make_fainted_pokemon() -> Pokemon:
    """Create a fainted placeholder pokemon (matches Foul Play's approach)."""
    return Pokemon(id="pikachu", level=1, types=("electric", "typeless"),
                   hp=0, maxhp=1, ability="static", item="None",
                   attack=1, defense=1, special_attack=1, special_defense=1, speed=1,
                   moves=[Move(id="splash", pp=16, disabled=True)] * 4,
                   tera_type="typeless", status="None")


def _convert_side_conditions(battle_conditions: dict) -> SideConditions:
    """Convert poke-env side conditions to poke_engine SideConditions."""
    kwargs = {}
    mapping = {
        SideCondition.SPIKES: "spikes",
        SideCondition.TOXIC_SPIKES: "toxic_spikes",
        SideCondition.STEALTH_ROCK: "stealth_rock",
        SideCondition.STICKY_WEB: "sticky_web",
        SideCondition.TAILWIND: "tailwind",
        SideCondition.REFLECT: "reflect",
        SideCondition.LIGHT_SCREEN: "light_screen",
        SideCondition.AURORA_VEIL: "aurora_veil",
    }
    for pe_cond, pe_name in mapping.items():
        if pe_cond in battle_conditions:
            val = battle_conditions[pe_cond]
            if isinstance(val, (list, tuple)):
                kwargs[pe_name] = val[0] if val else 1
            elif isinstance(val, int):
                kwargs[pe_name] = val
            else:
                kwargs[pe_name] = 1
    return SideConditions(**kwargs)


def _convert_side(
    battle: AbstractBattle,
    is_opponent: bool = False,
) -> Side:
    """
    Convert one side of the battle to poke_engine Side.

    Returns (Side, action_map) where action_map maps
    poke_engine move strings to poke-env action indices.
    """
    if is_opponent:
        team = battle.opponent_team
        active = battle.opponent_active_pokemon
        conditions = battle.opponent_side_conditions
    else:
        team = battle.team
        active = battle.active_pokemon
        conditions = battle.side_conditions

    pokemon_list = []

    if active:
        pokemon_list.append(_convert_pokemon(active, is_opponent))

    if team:
        for mon_id, mon in team.items():
            if not mon.active:
                if mon.fainted:
                    pokemon_list.append(_make_fainted_pokemon())
                else:
                    pokemon_list.append(_convert_pokemon(mon, is_opponent))

    while len(pokemon_list) < 6:
        pokemon_list.append(_make_fainted_pokemon())
    pokemon_list = pokemon_list[:6]

    side_conditions = _convert_side_conditions(conditions)

    boosts = active.boosts if active and active.boosts else {}

    # Determine last used move (critical for Choice items)
    # poke-engine format: "move:INDEX" or "switch:0" or "move:none"
    last_used_move = "move:none"
    if active and not is_opponent:
        # For our side, check if we have a choice item and can figure out the locked move
        # Default to move:0 (first move) as a safe fallback
        if active.item and any(x in _norm(active.item) for x in ['choice', 'choiceband', 'choicescarf', 'choicespecs']):
            # Try to find which move slot we're locked into
            # poke-env tracks this in active.must_recharge or through available_moves
            if battle.available_moves and len(battle.available_moves) == 1:
                # Choice-locked: only one move available
                locked_move = _norm(battle.available_moves[0].id)
                if active.moves:
                    for idx, (mid, _) in enumerate(active.moves.items()):
                        if _norm(mid) == locked_move:
                            last_used_move = f"move:{idx}"
                            break

    side = Side(
        pokemon=pokemon_list,
        side_conditions=side_conditions,
        active_index=PokemonIndex.P0,  # Active is always first in our list
        attack_boost=boosts.get("atk", 0),
        defense_boost=boosts.get("def", 0),
        special_attack_boost=boosts.get("spa", 0),
        special_defense_boost=boosts.get("spd", 0),
        speed_boost=boosts.get("spe", 0),
        accuracy_boost=boosts.get("accuracy", 0),
        evasion_boost=boosts.get("evasion", 0),
        last_used_move=last_used_move,
    )

    return side


def _get_weather(battle: AbstractBattle) -> Tuple[str, int]:
    """Get weather string and remaining turns."""
    if battle.weather:
        for w, turn_count in battle.weather.items():
            weather_str = WEATHER_MAP.get(w, "none")
            turns = turn_count if isinstance(turn_count, int) else 0
            return weather_str, turns
    return "none", 0


def _get_terrain(battle: AbstractBattle) -> Tuple[str, int]:
    """Get terrain string and remaining turns."""
    if battle.fields:
        for f, turn_count in battle.fields.items():
            terrain_str = FIELD_TO_TERRAIN.get(f)
            if terrain_str:
                turns = turn_count if isinstance(turn_count, int) else 0
                return terrain_str, turns
    return "none", 0


def _is_trick_room(battle: AbstractBattle) -> Tuple[bool, int]:
    """Check if Trick Room is active."""
    if battle.fields:
        for f, turn_count in battle.fields.items():
            if f == Field.TRICK_ROOM:
                turns = turn_count if isinstance(turn_count, int) else 0
                return True, turns
    return False, 0


def battle_to_engine_state(battle: AbstractBattle) -> Tuple[Optional[State], str]:
    """
    Convert a poke-env AbstractBattle to a poke_engine State.
    Returns (state, reason) where reason explains failure if state is None.
    """
    if not battle.active_pokemon or not battle.opponent_active_pokemon:
        return None, "no_active"

    # Don't skip when opponent fainted — we still need to choose our move
    # But track it so we can set force_switch on the opponent side
    opponent_fainted = (
        battle.opponent_active_pokemon.fainted 
        or battle.opponent_active_pokemon.current_hp == 0
    )

    try:
        side_one = _convert_side(battle, is_opponent=False)
        side_two = _convert_side(battle, is_opponent=True)

        # If opponent's active is fainted, they must force switch
        if opponent_fainted:
            side_two.force_switch = True

        weather, weather_turns = _get_weather(battle)
        terrain, terrain_turns = _get_terrain(battle)
        trick_room, tr_turns = _is_trick_room(battle)

        state = State(
            side_one=side_one,
            side_two=side_two,
            weather=weather,
            weather_turns_remaining=weather_turns,
            terrain=terrain,
            terrain_turns_remaining=terrain_turns,
            trick_room=trick_room,
            trick_room_turns_remaining=tr_turns,
        )
        return state, "ok"
    except Exception as e:
        return None, f"state_build_error: {e}"


def _map_mcts_to_actions(
    mcts_result,
    battle: AbstractBattle,
) -> Dict[int, float]:
    """
    Map MCTS move choices back to our 13-action-space indices.

    poke-engine returns move choices as:
      - Move IDs: "thunderbolt", "closecombat", etc.
      - Switch strings: "switch 1", "switch 2", etc.

    Our action space:
      0-3: moves (indexed by position in available_moves)
      4-7: tera moves
      8-12: switches (indexed by position in available_switches)
    """
    scores = {}
    moves = battle.available_moves or []
    switches = battle.available_switches or []

    move_id_to_idx = {}
    for i, move in enumerate(moves):
        move_id_to_idx[_norm(move.id)] = i

    switch_species_to_idx = {}
    for i, mon in enumerate(switches):
        switch_species_to_idx[_norm(mon.species)] = i

    for result in mcts_result.s1:
        choice = result.move_choice
        avg_score = result.total_score / max(result.visits, 1)

        if choice.startswith("switch "):
            # "switch garchomp" — match by species name
            species = _norm(choice[7:])  # strip "switch "
            if species in switch_species_to_idx:
                idx = switch_species_to_idx[species]
                scores[8 + idx] = avg_score
        else:
            norm_choice = _norm(choice)
            if norm_choice in move_id_to_idx:
                idx = move_id_to_idx[norm_choice]
                scores[idx] = avg_score
                if battle.can_tera:
                    scores[4 + idx] = avg_score * 0.95

    return scores


class EngineSearchPlayer(PPOPlayer):
    """
    PPO player enhanced with poke-engine MCTS search.

    For each decision:
    1. Converts battle state to poke_engine State
    2. Runs MCTS for search_time_ms milliseconds
    3. Gets policy probabilities from trained PPO model
    4. Mixes: final = (1-w) * policy_probs + w * mcts_probs
    5. Picks argmax
    """

    def __init__(
        self,
        model_path: str,
        search_weight: float = 0.5,
        search_time_ms: int = 200,
        min_tera_turn: int = 2,
        verbose: bool = False,
        **kwargs,
    ):
        self.search_weight = search_weight
        self.search_time_ms = search_time_ms
        self.min_tera_turn = min_tera_turn
        self.verbose = verbose
        self._search_stats = {
            'total_calls': 0,
            'successful_searches': 0,
            'fallback_to_policy': 0,
            'avg_search_time_ms': 0,
            'search_overrides': 0,
            'avg_iterations': 0,
            # Detailed fallback reasons
            'fb_no_active': 0,
            'fb_opponent_fainted': 0,
            'fb_state_build_error': 0,
            'fb_mcts_crash': 0,
            'fb_no_mapped_actions': 0,
            'fb_empty_scores': 0,
        }
        super().__init__(model_path=model_path, **kwargs)

    def _get_policy_probs(self, obs, action_mask):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits.clone()
            logits[~mask_tensor] = float('-inf')
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        return probs

    def _run_search(self, battle: AbstractBattle) -> Optional[Dict[int, float]]:
        """Run poke-engine MCTS and return {action_index: score}."""
        state, reason = battle_to_engine_state(battle)
        if state is None:
            if "no_active" in reason:
                self._search_stats['fb_no_active'] += 1
            elif "opponent_fainted" in reason:
                self._search_stats['fb_opponent_fainted'] += 1
            else:
                self._search_stats['fb_state_build_error'] += 1
            if self.verbose:
                print(f"  MCTS skipped: {reason}")
            return None

        try:
            t0 = time.time()
            # Round-trip through string serialization to ensure internal consistency
            # This fixes edge cases where Python-constructed states differ from
            # what poke-engine's Rust code expects internally
            state_str = state.to_string()
            state = State.from_string(state_str)
            result = mcts(state, self.search_time_ms)
            elapsed = (time.time() - t0) * 1000

            scores = _map_mcts_to_actions(result, battle)

            if not scores:
                # MCTS ran but nothing mapped — log what it returned
                self._search_stats['fb_no_mapped_actions'] += 1
                if self.verbose:
                    choices = [r.move_choice for r in result.s1]
                    moves = [_norm(m.id) for m in (battle.available_moves or [])]
                    switches = [_norm(m.species) for m in (battle.available_switches or [])]
                    print(f"  MCTS 0 mapped: returned {choices}, "
                          f"avail_moves={moves}, avail_switches={switches}")
                    try:
                        ss = state.to_string()
                        print(f"    FULL STATE: {ss}")
                    except:
                        pass
                return None

            self._search_stats['total_calls'] += 1
            self._search_stats['successful_searches'] += 1
            alpha = 0.1
            self._search_stats['avg_search_time_ms'] = (
                (1 - alpha) * self._search_stats['avg_search_time_ms'] + alpha * elapsed
            )
            self._search_stats['avg_iterations'] = (
                (1 - alpha) * self._search_stats['avg_iterations'] + alpha * result.iteration_count
            )

            if self.verbose:
                print(f"  MCTS: {len(scores)} actions, {result.iteration_count} iterations "
                      f"in {elapsed:.0f}ms")
                moves = battle.available_moves or []
                switches = battle.available_switches or []
                for idx, score in sorted(scores.items(), key=lambda x: -x[1]):
                    if idx < 4 and idx < len(moves):
                        name = moves[idx].id
                    elif idx < 8 and (idx - 4) < len(moves):
                        name = f"{moves[idx-4].id} tera"
                    elif idx >= 8 and (idx - 8) < len(switches):
                        name = f"switch {switches[idx-8].species}"
                    else:
                        name = f"action_{idx}"
                    print(f"    {name}: {score:.3f}")

            return scores

        except BaseException as e:
            self._search_stats['fb_mcts_crash'] += 1
            if self.verbose:
                print(f"  MCTS crash: {e}")
            self._search_stats['total_calls'] += 1
            self._search_stats['fallback_to_policy'] += 1
            return None

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        obs = self._rl_env.embed_battle(battle)
        action_mask = self._rl_env.get_action_mask(battle)

        if self.min_tera_turn > 0 and battle.turn < self.min_tera_turn:
            tera_backup = action_mask[4:8].copy()
            action_mask[4:8] = 0.0
            if action_mask.sum() == 0:
                action_mask[4:8] = tera_backup

        policy_probs = self._get_policy_probs(obs, action_mask)

        mcts_scores = self._run_search(battle)

        if mcts_scores and len(mcts_scores) > 0:
            # MCTS scores are win rates [0, 1]. Normalize directly to probabilities.
            # Use temperature to control peakedness:
            # temp < 1 = more peaked (trust MCTS best move more)
            # temp > 1 = flatter
            temperature = 0.3
            
            mcts_vals = np.zeros(13, dtype=np.float32)
            for idx, score in mcts_scores.items():
                if action_mask[idx] > 0:
                    mcts_vals[idx] = max(score, 0.001)  # Avoid zero

            # Power-scale then normalize: score^(1/temp) concentrates on best
            legal_mask = mcts_vals > 0
            if legal_mask.any():
                powered = np.where(legal_mask, np.power(mcts_vals, 1.0 / temperature), 0.0)
                mcts_sum = powered.sum()
                if mcts_sum > 0:
                    mcts_probs = powered / mcts_sum
                else:
                    mcts_probs = policy_probs.copy()
            else:
                mcts_probs = policy_probs.copy()

            w = self.search_weight
            combined = (1 - w) * policy_probs + w * mcts_probs
            combined = combined * (action_mask > 0)
            total = combined.sum()
            if total > 0:
                combined /= total

            action = int(combined.argmax())

            if self.verbose:
                policy_choice = int(policy_probs.argmax())
                if action != policy_choice:
                    moves = battle.available_moves or []
                    switches = battle.available_switches or []
                    def _name(idx):
                        if idx < 4 and idx < len(moves): return moves[idx].id
                        if idx < 8 and (idx-4) < len(moves): return f"{moves[idx-4].id} tera"
                        if idx >= 8 and (idx-8) < len(switches): return f"switch {switches[idx-8].species}"
                        return f"action_{idx}"
                    print(f"  Turn {battle.turn}: MCTS overrode! "
                          f"{_name(policy_choice)} -> {_name(action)}")
                    self._search_stats['search_overrides'] += 1
        else:
            self._search_stats['fallback_to_policy'] += 1
            self._search_stats['fb_empty_scores'] += 1
            masked = np.where(action_mask > 0, policy_probs, -np.inf)
            action = int(masked.argmax())

        return self._rl_env.action_to_order(action, battle)

    def get_search_stats(self) -> dict:
        stats = self._search_stats.copy()
        if stats['total_calls'] > 0:
            stats['search_rate'] = stats['successful_searches'] / stats['total_calls']
            stats['override_rate'] = (
                stats['search_overrides'] / stats['successful_searches']
                if stats['successful_searches'] > 0 else 0
            )
        return stats


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_pokemon_6v6_v18.2")
    parser.add_argument("--search-weight", type=float, default=0.5)
    parser.add_argument("--search-time", type=int, default=200,
                        help="MCTS search time in ms per decision")
    parser.add_argument("--n-battles", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from poke_env.player import SimpleHeuristicsPlayer

    player = EngineSearchPlayer(
        model_path=args.model,
        search_weight=args.search_weight,
        search_time_ms=args.search_time,
        deterministic=True,
        verbose=args.verbose,
        battle_format="gen9randombattle",
    )

    opponent = SimpleHeuristicsPlayer(battle_format="gen9randombattle")

    async def main():
        print(f"Testing EngineSearchPlayer ({args.n_battles} battles vs SimpleHeuristics)...")
        print(f"  search_weight={args.search_weight}, search_time={args.search_time}ms")
        await player.battle_against(opponent, n_battles=args.n_battles)

        wins = sum(1 for b in player.battles.values() if b.won)
        total = len(player.battles)
        print(f"\nResult: {wins}/{total} ({wins/total*100:.1f}%)")

        stats = player.get_search_stats()
        print(f"\nSearch stats:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    asyncio.run(main())