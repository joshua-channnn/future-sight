from poke_env.environment.env import PokeEnv
from poke_env.player import SingleBattleOrder, DefaultBattleOrder, BattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from gymnasium.spaces import Box, Discrete
from poke_env.data import GenData
from poke_env.battle import SideCondition, Weather, Effect

import numpy as np

GEN_DATA = GenData.from_gen(9)


def get_type_multiplier(atk_types, def_types):
    """Calculate max type effectiveness multiplier"""
    if not atk_types or not def_types:
        return 1.0
    max_mult = 1.0
    for atk_type in atk_types:
        if atk_type is None:
            continue
        mult = 1.0
        for def_type in def_types:
            if def_type is None:
                continue
            try:
                mult *= GEN_DATA.type_chart[atk_type.name][def_type.name]
            except KeyError:
                mult *= 1.0
        max_mult = max(max_mult, mult)
    return max_mult


def check_effect(pokemon, effect_name: str) -> bool:
    if pokemon is None or not hasattr(pokemon, 'effects') or not pokemon.effects:
        return False
    
    EFFECT_MAP = {
        'leechseed': Effect.LEECH_SEED,
        'confusion': Effect.CONFUSION,
        'substitute': Effect.SUBSTITUTE,
        'taunt': Effect.TAUNT,
        'encore': Effect.ENCORE,
        'trapped': Effect.TRAPPED,
        'partiallytrapped': Effect.PARTIALLY_TRAPPED,
    }
    
    effect_enum = EFFECT_MAP.get(effect_name.lower())
    if effect_enum is None:
        return False
    
    return effect_enum in pokemon.effects


class RL6v6Player(PokeEnv):
    OBSERVATION_SIZE = 157  # Fixed: actually count the features
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_battle = None
        # PBRS state
        self._last_potential = 0.0
        self._last_battle_tag = None
        
        # Action history tracking
        self._last_action_was_switch = False
        self._last_action = None  # Track last action for reward calculation
        self._turns_since_switch = 0
        self._switch_history = []
    
    def _reset_battle_state(self):
        """Reset per-battle state. Call this when a new battle starts."""
        self._last_action_was_switch = False
        self._last_action = None
        self._turns_since_switch = 0
        self._switch_history = []
        self._last_potential = 0.0
    
    def _calc_hazard_damage(self, battle: AbstractBattle, pokemon) -> float:
        """Calculate entry hazard damage for a Pokemon switching in."""
        if pokemon is None:
            return 0.0
        
        damage = 0.0
        
        # Stealth Rock
        if SideCondition.STEALTH_ROCK in battle.side_conditions:
            rock_effectiveness = 1.0
            for ptype in pokemon.types:
                if ptype:
                    try:
                        rock_effectiveness *= GEN_DATA.type_chart["ROCK"][ptype.name]
                    except KeyError:
                        pass
            damage += 0.125 * rock_effectiveness
        
        # Spikes
        spikes = battle.side_conditions.get(SideCondition.SPIKES, 0)
        if spikes > 0 and not self._has_ground_immunity(pokemon):
            spikes_damage = [0, 0.125, 0.167, 0.25][min(spikes, 3)]
            damage += spikes_damage
        
        # Toxic Spikes
        tspikes = battle.side_conditions.get(SideCondition.TOXIC_SPIKES, 0)
        if tspikes > 0 and not self._has_ground_immunity(pokemon):
            if "POISON" not in [t.name for t in pokemon.types if t]:
                damage += 0.06 * tspikes
        
        # Sticky Web
        if SideCondition.STICKY_WEB in battle.side_conditions:
            if not self._has_ground_immunity(pokemon):
                damage += 0.02
        
        return min(damage, 1.0)
    
    def _has_ground_immunity(self, pokemon) -> bool:
        if pokemon is None:
            return False
        if "FLYING" in [t.name for t in pokemon.types if t]:
            return True
        if pokemon.ability and pokemon.ability.lower() == "levitate":
            return True
        if pokemon.item and pokemon.item.lower() == "airballoon":
            return True
        return False
    
    def _opponent_seems_passive(self, battle: AbstractBattle) -> float:
        opp = battle.opponent_active_pokemon
        my_pokemon = battle.active_pokemon
        
        if not opp or not my_pokemon:
            return 0.5
        
        score = 0.0
        
        offensive = get_type_multiplier(my_pokemon.types, opp.types)
        defensive = get_type_multiplier(opp.types, my_pokemon.types)
        
        if defensive < 1.0:  # We resist them
            score += 0.2
        if offensive > 1.5:  # We hit them super effective
            score += 0.3
        
        if opp.current_hp_fraction and opp.current_hp_fraction < 0.3:
            score += 0.2
        
        if opp.status:
            score += 0.15
        
        if check_effect(my_pokemon, "substitute"):
            score += 0.3
        
        return min(score, 1.0)
    
    def _is_recovery_move(self, move) -> bool:
        recovery_moves = {
            "recover", "softboiled", "roost", "slackoff", "moonlight",
            "morningsun", "synthesis", "wish", "rest", "shoreup",
            "milkdrink", "healorder", "strengthsap"
        }
        if hasattr(move, 'id'):
            return move.id.lower() in recovery_moves
        return False
    
    def _is_setup_move(self, move) -> bool:
        if hasattr(move, 'boosts') and move.boosts:
            if hasattr(move, 'target') and move.target in ["self", "allySide", "allies"]:
                return any(v > 0 for v in move.boosts.values())
        setup_moves = {
            "swordsdance", "nastyplot", "calmmind", "dragondance", 
            "quiverdance", "shellsmash", "bulkup", "irondefense",
            "agility", "rockpolish", "autotomize", "coil", "shiftgear",
            "victorydance", "tidyup"
        }
        if hasattr(move, 'id'):
            return move.id.lower() in setup_moves
        return False
    
    def _detect_oscillation(self) -> float:
        if len(self._switch_history) < 3:
            return 0.0
        
        # Check for A-B-A-B pattern
        if len(self._switch_history) >= 4:
            recent = self._switch_history[-4:]
            if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
                return 1.0
        
        # Check for A-B-A pattern
        recent3 = self._switch_history[-3:]
        if recent3[0] == recent3[2] and recent3[0] != recent3[1]:
            return 0.7
        
        return 0.0
    
    def update_action_history(self, action: int, battle: AbstractBattle):
        """Call this after taking an action to update history tracking."""
        self._last_action = action
        was_switch = action >= 8
        self._last_action_was_switch = was_switch
        
        if was_switch:
            self._turns_since_switch = 0
            switch_idx = action - 8
            if switch_idx < len(battle.available_switches):
                switched_to = battle.available_switches[switch_idx].species
                self._switch_history.append(switched_to)
                if len(self._switch_history) > 6:
                    self._switch_history.pop(0)
        else:
            self._turns_since_switch += 1

    @property
    def action_spaces(self):
        return {agent: Discrete(13) for agent in self.possible_agents}
    
    @property
    def observation_spaces(self):
        return {agent: Box(low=0, high=1, shape=(self.OBSERVATION_SIZE,), dtype=np.float32)
                for agent in self.possible_agents}

    def _compute_potential(self, battle: AbstractBattle) -> float:
        potential = 0.0
        
        our_hp = sum(mon.current_hp_fraction for mon in battle.team.values())
        opp_hp = sum(
            mon.current_hp_fraction 
            for mon in battle.opponent_team.values() 
            if mon.current_hp_fraction is not None
        )
        potential += (our_hp - opp_hp) * 0.1
        
        our_alive = sum(1 for mon in battle.team.values() if mon.current_hp_fraction > 0)
        opp_alive = sum(
            1 for mon in battle.opponent_team.values() 
            if mon.current_hp_fraction is not None and mon.current_hp_fraction > 0
        )
        potential += (our_alive - opp_alive) * 0.05
        
        my_pokemon = battle.active_pokemon
        opp_pokemon = battle.opponent_active_pokemon
        
        if my_pokemon and opp_pokemon:
            offensive = get_type_multiplier(my_pokemon.types, opp_pokemon.types)
            defensive = get_type_multiplier(opp_pokemon.types, my_pokemon.types)
            matchup_score = (np.log2(max(offensive, 0.25)) - np.log2(max(defensive, 0.25))) * 0.025
            potential += matchup_score
        
        if my_pokemon and my_pokemon.status:
            potential -= 0.03
        if opp_pokemon and opp_pokemon.status:
            potential += 0.03
        
        # Hazards
        our_hazard_damage = 0
        if SideCondition.STEALTH_ROCK in battle.side_conditions:
            our_hazard_damage += 0.125
        our_hazard_damage += battle.side_conditions.get(SideCondition.SPIKES, 0) * 0.08
        our_hazard_damage += battle.side_conditions.get(SideCondition.TOXIC_SPIKES, 0) * 0.1
        
        opp_hazard_damage = 0
        if SideCondition.STEALTH_ROCK in battle.opponent_side_conditions:
            opp_hazard_damage += 0.125
        opp_hazard_damage += battle.opponent_side_conditions.get(SideCondition.SPIKES, 0) * 0.08
        opp_hazard_damage += battle.opponent_side_conditions.get(SideCondition.TOXIC_SPIKES, 0) * 0.1
        
        potential += (opp_hazard_damage - our_hazard_damage) * 0.2
        
        # Stat boosts
        if my_pokemon:
            our_boosts = sum(my_pokemon.boosts.values())
            potential += our_boosts * 0.01
        if opp_pokemon:
            opp_boosts = sum(opp_pokemon.boosts.values())
            potential -= opp_boosts * 0.01
        
        return potential

    def calc_reward(self, battle: AbstractBattle, auditor=None) -> float:
        base_reward = self.reward_computing_helper(
            battle, 
            fainted_value=0.05,
            hp_value=0.025,
            victory_value=1.0
        )
        
        # Potential-based reward shaping
        battle_tag = battle.battle_tag
        if battle_tag != self._last_battle_tag:
            self._last_potential = 0.0
            self._last_battle_tag = battle_tag
        
        current_potential = self._compute_potential(battle)
        gamma = 0.99
        shaping_reward = gamma * current_potential - self._last_potential
        shaping_scale = 0.5
        pbrs_reward = shaping_reward * shaping_scale
        
        self._last_potential = current_potential

        # Oscillation penalty - uses _last_action set by wrapper
        oscillation_penalty = 0.0
        if self._last_action is not None and self._last_action >= 8:
            oscillation = self._detect_oscillation()
            if oscillation > 0.5:
                oscillation_penalty = -0.03 * oscillation
        
        total_reward = base_reward + pbrs_reward + oscillation_penalty

        if auditor is not None:
            auditor.on_step(base_reward, pbrs_reward, total_reward, current_potential)
        
        return total_reward
    
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        state = []
        
        my_pokemon = battle.active_pokemon
        opp_pokemon = battle.opponent_active_pokemon
        
        # === ACTIVE POKEMON HP (2) ===
        state.append(my_pokemon.current_hp_fraction if my_pokemon else 0.0)
        state.append(opp_pokemon.current_hp_fraction if opp_pokemon else 1.0)

        # === STATUS (2) ===
        state.append(1.0 if my_pokemon and my_pokemon.status else 0.0)
        state.append(1.0 if opp_pokemon and opp_pokemon.status else 0.0)

        # === SPEED COMPARISON (1) ===
        if my_pokemon and opp_pokemon:
            my_speed = my_pokemon.base_stats.get("spe", 100)
            opp_speed = opp_pokemon.base_stats.get("spe", 100)
            
            if my_pokemon.status and my_pokemon.status.name == "PAR":
                my_speed *= 0.5
            if opp_pokemon.status and opp_pokemon.status.name == "PAR":
                opp_speed *= 0.5
            
            my_speed *= (2 + my_pokemon.boosts.get("spe", 0)) / 2
            opp_speed *= (2 + opp_pokemon.boosts.get("spe", 0)) / 2
            
            state.append(1.0 if my_speed > opp_speed else 0.0)
        else:
            state.append(0.5)
        
        # === TYPE MATCHUPS (2) ===
        if my_pokemon and opp_pokemon:
            offensive = get_type_multiplier(my_pokemon.types, opp_pokemon.types)
            defensive = get_type_multiplier(opp_pokemon.types, my_pokemon.types)
            state.append(min(offensive / 4, 1.0))
            state.append(1 - min(defensive / 4, 1.0))
        else:
            state.append(0.5)
            state.append(0.5)

        # === WEATHER (4) ===
        weather = battle.weather
        state.append(1.0 if Weather.SUNNYDAY in weather or Weather.DESOLATELAND in weather else 0.0)
        state.append(1.0 if Weather.RAINDANCE in weather or Weather.PRIMORDIALSEA in weather else 0.0)
        state.append(1.0 if Weather.SANDSTORM in weather else 0.0)
        state.append(1.0 if Weather.SNOW in weather or Weather.HAIL in weather else 0.0)

        # === SCREENS (2) ===
        state.append(1.0 if SideCondition.REFLECT in battle.side_conditions else 0.0)
        state.append(1.0 if SideCondition.LIGHT_SCREEN in battle.side_conditions else 0.0)

        # === TERA (4) ===
        state.append(1.0 if battle.can_tera else 0.0)
        state.append(1.0 if my_pokemon and my_pokemon.is_terastallized else 0.0)
        state.append(1.0 if opp_pokemon and opp_pokemon.is_terastallized else 0.0)

        if battle.can_tera and my_pokemon and opp_pokemon and my_pokemon.tera_type:
            tera_offensive = get_type_multiplier([my_pokemon.tera_type], opp_pokemon.types)
            state.append(min(tera_offensive / 4, 1.0))
        else:
            state.append(0.5)

        # === STAT BOOSTS (10) ===
        boost_stats = ["atk", "def", "spa", "spd", "spe"]
        if my_pokemon:
            for stat in boost_stats:
                state.append((my_pokemon.boosts.get(stat, 0) + 6) / 12)
        else:
            state.extend([0.5] * 5)

        if opp_pokemon:
            for stat in boost_stats:
                state.append((opp_pokemon.boosts.get(stat, 0) + 6) / 12)
        else:
            state.extend([0.5] * 5)

        # === HAZARDS (8) ===
        state.append(battle.side_conditions.get(SideCondition.SPIKES, 0) / 3)
        state.append(1.0 if SideCondition.STEALTH_ROCK in battle.side_conditions else 0.0)
        state.append(battle.side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2)
        state.append(1.0 if SideCondition.STICKY_WEB in battle.side_conditions else 0.0)

        state.append(battle.opponent_side_conditions.get(SideCondition.SPIKES, 0) / 3)
        state.append(1.0 if SideCondition.STEALTH_ROCK in battle.opponent_side_conditions else 0.0)
        state.append(battle.opponent_side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2)
        state.append(1.0 if SideCondition.STICKY_WEB in battle.opponent_side_conditions else 0.0)
        
        # === MOVE PP (4) ===
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                if move.max_pp:
                    state.append(move.current_pp / move.max_pp)
                else:
                    state.append(1.0)
            else:
                state.append(0.0)

        # === VOLATILE EFFECTS (12) ===
        state.append(1.0 if check_effect(my_pokemon, "leechseed") else 0.0)
        state.append(1.0 if check_effect(my_pokemon, "confusion") else 0.0)
        state.append(1.0 if check_effect(my_pokemon, "substitute") else 0.0)
        state.append(1.0 if check_effect(my_pokemon, "taunt") else 0.0)
        state.append(1.0 if check_effect(my_pokemon, "encore") else 0.0)
        state.append(1.0 if check_effect(my_pokemon, "trapped") or check_effect(my_pokemon, "partiallytrapped") else 0.0)

        state.append(1.0 if check_effect(opp_pokemon, "leechseed") else 0.0)
        state.append(1.0 if check_effect(opp_pokemon, "confusion") else 0.0)
        state.append(1.0 if check_effect(opp_pokemon, "substitute") else 0.0)
        state.append(1.0 if check_effect(opp_pokemon, "taunt") else 0.0)
        state.append(1.0 if check_effect(opp_pokemon, "encore") else 0.0)
        state.append(1.0 if check_effect(opp_pokemon, "trapped") or check_effect(opp_pokemon, "partiallytrapped") else 0.0)
        
        # === BENCH HP (10) ===
        my_bench = [mon for mon in battle.team.values() if not mon.active]
        for i in range(5):
            if i < len(my_bench):
                state.append(my_bench[i].current_hp_fraction)
            else:
                state.append(0.0)

        opp_bench = [mon for mon in battle.opponent_team.values() if not mon.active]
        for i in range(5):
            if i < len(opp_bench):
                hp = opp_bench[i].current_hp_fraction
                state.append(hp if hp is not None else 1.0)
            else:
                state.append(1.0)
        
        # === BENCH TYPE MATCHUPS (10) ===
        for i in range(5):
            if i < len(my_bench) and opp_pokemon:
                offensive = get_type_multiplier(my_bench[i].types, opp_pokemon.types)
                defensive = get_type_multiplier(opp_pokemon.types, my_bench[i].types)
                state.append(min(offensive / 4, 1.0))
                state.append(1 - min(defensive / 4, 1.0))
            else:
                state.append(0.5)
                state.append(0.5)

        # === MOVE INFO (24) ===
        max_power = max([m.base_power or 0 for m in battle.available_moves], default=0)
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                state.append(move.base_power / 150 if move.base_power else 0.0)
                is_strongest = 1.0 if (move.base_power or 0) == max_power and max_power > 0 else 0.0
                state.append(is_strongest)
                has_stab = 1.0 if my_pokemon and move.type in my_pokemon.types else 0.0
                state.append(has_stab)
                if opp_pokemon:
                    multiplier = opp_pokemon.damage_multiplier(move)
                    state.append(min(multiplier / 4, 1.0))
                else:
                    state.append(0.5)
                state.append(move.accuracy / 100 if move.accuracy else 1.0)
                priority = 0
                if hasattr(move, "entry") and isinstance(move.entry, dict):
                    priority = move.entry.get("priority", 0)
                state.append(1.0 if priority > 0 else 0.0)
            else:
                state.extend([0.0, 0.0, 0.0, 0.5, 1.0, 0.0])

        # === CAN SWITCH (1) ===
        state.append(1.0 if battle.available_switches else 0.0)

        # === STAT RATIOS (4) ===
        for i in range(4):
            if i < len(battle.available_moves) and my_pokemon and opp_pokemon:
                move = battle.available_moves[i]
                if move.category.name == "PHYSICAL":
                    our_stat = my_pokemon.base_stats.get("atk", 100)
                    their_stat = opp_pokemon.base_stats.get("def", 100)
                elif move.category.name == "SPECIAL":
                    our_stat = my_pokemon.base_stats.get("spa", 100)
                    their_stat = opp_pokemon.base_stats.get("spd", 100)
                else:
                    state.append(0.5)
                    continue
                ratio = our_stat / (our_stat + their_stat)
                state.append(ratio)
            else:
                state.append(0.5)

        # === IS STATUS MOVE (4) ===
        for i in range(4):
            if i < len(battle.available_moves):
                is_status = 1.0 if battle.available_moves[i].category.name == "STATUS" else 0.0
                state.append(is_status)
            else:
                state.append(0.0)

        # === TEAM COUNTS (2) ===
        our_remaining = sum(1 for mon in battle.team.values() if mon.current_hp_fraction > 0)
        opp_remaining = sum(1 for mon in battle.opponent_team.values() 
                          if mon.current_hp_fraction is not None and mon.current_hp_fraction > 0)
        state.append(our_remaining / 6)
        state.append(opp_remaining / 6)

        # === TURN NUMBER (1) ===
        state.append(min(battle.turn / 50, 1.0))
        
        # === GAME PHASE (3) ===
        state.append(1.0 if battle.turn <= 5 else 0.0)
        state.append(1.0 if 5 < battle.turn <= 20 else 0.0)
        state.append(1.0 if battle.turn > 20 else 0.0)
        
        # === TEAM ADVANTAGE (3) ===
        my_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() 
                    if p.current_hp_fraction is not None and p.current_hp_fraction > 0)
        state.append(my_alive / 6)
        state.append(opp_alive / 6)
        state.append((my_alive - opp_alive + 6) / 12)
        
        # === SWITCH CONTEXT (5) ===
        state.append(1.0 if self._last_action_was_switch else 0.0)
        state.append(min(self._turns_since_switch / 10, 1.0))
        state.append(self._detect_oscillation())
        
        bench = [m for m in battle.team.values() if not m.active and not m.fainted]
        avg_hazard_cost = 0.0
        if bench:
            avg_hazard_cost = sum(self._calc_hazard_damage(battle, m) for m in bench) / len(bench)
        state.append(avg_hazard_cost)
        
        best_switch_matchup = 0.0
        for mon in bench:
            if opp_pokemon:
                off = get_type_multiplier(mon.types, opp_pokemon.types)
                defe = get_type_multiplier(opp_pokemon.types, mon.types)
                matchup = (off / 4) - (defe / 4) + 0.5
                best_switch_matchup = max(best_switch_matchup, min(matchup, 1.0))
        state.append(best_switch_matchup)
        
        # === SETUP OPPORTUNITY (3) ===
        state.append(self._opponent_seems_passive(battle))
        has_setup = any(self._is_setup_move(m) for m in battle.available_moves)
        state.append(1.0 if has_setup else 0.0)
        current_boosts = sum(max(0, v) for v in my_pokemon.boosts.values()) if my_pokemon else 0
        state.append(min(current_boosts / 6, 1.0))
        
        # === HEALING CONTEXT (3) ===
        hp_frac = my_pokemon.current_hp_fraction if my_pokemon else 1.0
        hp_missing = 1.0 - hp_frac
        state.append(hp_missing)
        has_recovery = any(self._is_recovery_move(m) for m in battle.available_moves)
        state.append(1.0 if has_recovery else 0.0)
        healing_value = hp_missing if 0.2 < hp_frac < 0.7 else 0.0
        state.append(healing_value)

        # === BASE STATS (10) ===
        if my_pokemon:
            state.append(my_pokemon.base_stats.get("hp", 100) / 180)
            state.append(my_pokemon.base_stats.get("atk", 100) / 180)
            state.append(my_pokemon.base_stats.get("def", 100) / 180)
            state.append(my_pokemon.base_stats.get("spa", 100) / 180)
            state.append(my_pokemon.base_stats.get("spd", 100) / 180)
        else:
            state.extend([0.5] * 5)

        if opp_pokemon:
            state.append(opp_pokemon.base_stats.get("hp", 100) / 180)
            state.append(opp_pokemon.base_stats.get("atk", 100) / 180)
            state.append(opp_pokemon.base_stats.get("def", 100) / 180)
            state.append(opp_pokemon.base_stats.get("spa", 100) / 180)
            state.append(opp_pokemon.base_stats.get("spd", 100) / 180)
        else:
            state.extend([0.5] * 5)

        # === ITEMS (10) ===
        if my_pokemon and my_pokemon.item:
            item = my_pokemon.item.lower()
            state.append(1.0 if "choice" in item else 0.0)
            state.append(1.0 if item == "lifeorb" else 0.0)
            state.append(1.0 if item == "focussash" else 0.0)
            state.append(1.0 if item == "heavydutyboots" else 0.0)
            state.append(1.0 if item == "leftovers" else 0.0)
            state.append(1.0 if my_pokemon.item is None else 0.0)
        else:
            state.extend([0.0] * 6)

        if opp_pokemon and opp_pokemon.item and opp_pokemon.item not in [None, "unknown_item"]:
            item = opp_pokemon.item.lower()
            state.append(1.0 if "choice" in item else 0.0)
            state.append(1.0 if item == "focussash" else 0.0)
            state.append(1.0 if item == "leftovers" else 0.0)
            state.append(1.0 if item == "lifeorb" else 0.0)
        else:
            state.extend([0.0] * 4)

        # === ABILITIES (9) ===
        if my_pokemon and my_pokemon.ability:
            ability = my_pokemon.ability.lower()
            state.append(1.0 if ability == "levitate" else 0.0)
            state.append(1.0 if ability == "intimidate" else 0.0)
            state.append(1.0 if ability in ["multiscale", "shadowshield"] else 0.0)
            state.append(1.0 if ability == "sturdy" else 0.0)
            state.append(1.0 if ability in ["swiftswim", "chlorophyll", "sandrush", "slushrush"] else 0.0)
            state.append(1.0 if ability == "magicguard" else 0.0)
        else:
            state.extend([0.0] * 6)

        if opp_pokemon and opp_pokemon.ability:
            ability = opp_pokemon.ability.lower()
            state.append(1.0 if ability == "levitate" else 0.0)
            state.append(1.0 if ability == "sturdy" else 0.0)
            state.append(1.0 if ability == "wonderguard" else 0.0)
        else:
            state.extend([0.0] * 3)

        # === IS SETUP MOVE (4) ===
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]
                is_setup = 1.0 if self._is_setup_move(move) else 0.0
                state.append(is_setup)
            else:
                state.append(0.0)

        # Total: 157 features
        result = np.array(state, dtype=np.float32)
        
        if len(result) != self.OBSERVATION_SIZE:
            raise ValueError(f"Feature count mismatch: got {len(result)}, expected {self.OBSERVATION_SIZE}")
        
        return result

    def action_to_order(self, action: int, battle: AbstractBattle, fake: bool = False, strict: bool = True) -> BattleOrder:
        action = int(action)

        if action < 0:
            if battle.available_moves:
                return SingleBattleOrder(battle.available_moves[0])
            if battle.available_switches:
                return SingleBattleOrder(battle.available_switches[0])
            return DefaultBattleOrder()
        
        if action < 4 and action < len(battle.available_moves):
            return SingleBattleOrder(battle.available_moves[action], terastallize=False)
        
        elif action < 8:
            move_idx = action - 4
            if 0 <= move_idx < len(battle.available_moves): 
                if battle.can_tera:
                    return SingleBattleOrder(battle.available_moves[move_idx], terastallize=True)
                else:
                    return SingleBattleOrder(battle.available_moves[move_idx], terastallize=False)

        else:
            switch_idx = action - 8
            if 0 <= switch_idx < len(battle.available_switches):
                return SingleBattleOrder(battle.available_switches[switch_idx])
        
        if battle.available_moves:
            return SingleBattleOrder(battle.available_moves[0])
        if battle.available_switches:
            return SingleBattleOrder(battle.available_switches[0])
        
        return DefaultBattleOrder()
    
    def order_to_action(self, order: BattleOrder, battle: AbstractBattle, fake: bool = False, strict: bool = True) -> int:
        if order.order in battle.available_moves:
            move_idx = battle.available_moves.index(order.order)
            if hasattr(order, 'terastallize') and order.terastallize:
                return 4 + move_idx
            return move_idx

        if order.order in battle.available_switches:
            return 8 + battle.available_switches.index(order.order)

        return -1
    
    def get_action_mask(self, battle: AbstractBattle) -> np.ndarray:
        mask = np.zeros(13, dtype=np.float32)
        
        for i in range(min(4, len(battle.available_moves))):
            mask[i] = 1.0
        
        if battle.can_tera:
            for i in range(min(4, len(battle.available_moves))):
                mask[4 + i] = 1.0
        
        for i in range(min(5, len(battle.available_switches))):
            mask[8 + i] = 1.0
        
        if mask.sum() == 0:
            mask[0] = 1.0
        
        return mask