import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from gymnasium.spaces import Box, Discrete

from poke_env.environment.env import PokeEnv
from poke_env.player import SingleBattleOrder, DefaultBattleOrder, BattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle import SideCondition, Weather, Effect
from poke_env.data import GenData


TYPE_TO_IDX = {
    None: 0,
    PokemonType.NORMAL: 1, PokemonType.FIRE: 2, PokemonType.WATER: 3,
    PokemonType.ELECTRIC: 4, PokemonType.GRASS: 5, PokemonType.ICE: 6,
    PokemonType.FIGHTING: 7, PokemonType.POISON: 8, PokemonType.GROUND: 9,
    PokemonType.FLYING: 10, PokemonType.PSYCHIC: 11, PokemonType.BUG: 12,
    PokemonType.ROCK: 13, PokemonType.GHOST: 14, PokemonType.DRAGON: 15,
    PokemonType.DARK: 16, PokemonType.STEEL: 17, PokemonType.FAIRY: 18,
}
NUM_TYPES = 19
GEN_DATA = GenData.from_gen(9)

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


def get_type_multiplier(atk_types, def_types):
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
                pass
        max_mult = max(max_mult, mult)
    return max_mult


def check_effect(pokemon, effect_name):
    if pokemon is None or not hasattr(pokemon, 'effects') or not pokemon.effects:
        return False
    EFFECT_MAP = {
        'leechseed': Effect.LEECH_SEED, 'confusion': Effect.CONFUSION,
        'substitute': Effect.SUBSTITUTE, 'taunt': Effect.TAUNT,
        'encore': Effect.ENCORE, 'trapped': Effect.TRAPPED,
        'partiallytrapped': Effect.PARTIALLY_TRAPPED,
    }
    effect_enum = EFFECT_MAP.get(effect_name.lower())
    return effect_enum is not None and effect_enum in pokemon.effects


class RL6v6PlayerV17(PokeEnv):

    PER_POKEMON_INDICES = 13
    PER_POKEMON_FLOATS = 31   # CHANGED: 30 -> 31 (+move_knowledge)
    PER_POKEMON = PER_POKEMON_INDICES + PER_POKEMON_FLOATS  # 44 (was 43)

    N_POKEMON = 12

    N_ACTIVE_MOVE_FLOATS = 48  # 4 moves x 12 floats
    N_ACTIVE_MOVE_INDICES = 4

    N_GLOBAL = 60  # CHANGED: 55 -> 60 (+5 bench move effectiveness)

    OBSERVATION_SIZE = (N_POKEMON * PER_POKEMON + N_ACTIVE_MOVE_FLOATS
                        + N_ACTIVE_MOVE_INDICES + N_GLOBAL)  # 640 (was 623)

    ACTION_SPACE_SIZE = 13

    def __init__(self, data_dir="data", **kwargs):
        super().__init__(**kwargs)
        data_path = Path(data_dir)
        with open(data_path / "pokemon_to_idx.json") as f:
            self.pokemon_to_idx = json.load(f)
        with open(data_path / "move_to_idx.json") as f:
            self.move_to_idx = json.load(f)
        with open(data_path / "ability_to_idx.json") as f:
            self.ability_to_idx = json.load(f)
        with open(data_path / "item_to_idx.json") as f:
            self.item_to_idx = json.load(f)
        with open(data_path / "embedding_config.json") as f:
            self.embed_config = json.load(f)
        self._reset_battle_state()
        self._last_battle_tag = None

    def _reset_battle_state(self):
        self._last_action_was_switch = False
        self._last_action = None
        self._turns_since_switch = 0
        self._switch_history = []
        self._last_potential = 0.0

    def update_action_history(self, action, battle):
        self._last_action = action
        was_switch = action >= 8
        self._last_action_was_switch = was_switch
        if was_switch:
            self._turns_since_switch = 0
            si = action - 8
            if si < len(battle.available_switches):
                self._switch_history.append(battle.available_switches[si].species)
                if len(self._switch_history) > 6:
                    self._switch_history.pop(0)
        else:
            self._turns_since_switch += 1

    @property
    def action_spaces(self):
        return {a: Discrete(self.ACTION_SPACE_SIZE) for a in self.possible_agents}

    @property
    def observation_spaces(self):
        low = np.full(self.OBSERVATION_SIZE, -1.0, dtype=np.float32)
        high = np.full(self.OBSERVATION_SIZE, 510.0, dtype=np.float32)
        return {a: Box(low=low, high=high, shape=(self.OBSERVATION_SIZE,), dtype=np.float32)
                for a in self.possible_agents}

    def _normalize(self, name):
        if name is None: return "<UNK>"
        return name.lower().replace(" ", "").replace("-", "").replace("'", "")

    def _get_pokemon_idx(self, s): return self.pokemon_to_idx.get(self._normalize(s), 0)
    def _get_move_idx(self, s): return self.move_to_idx.get(self._normalize(s), 0)
    def _get_ability_idx(self, s): return self.ability_to_idx.get(self._normalize(s), 0)
    def _get_item_idx(self, s): return self.item_to_idx.get(self._normalize(s), 0)

    def _encode_pokemon_block(self, pokemon, is_opponent=False):
        """
        Encode a single Pokemon into 13 indices + 31 floats = 44 values.
        
        Args:
            pokemon: The Pokemon object (or None for empty slot)
            is_opponent: If True, move_knowledge reflects revealed moves.
                         If False (our team), move_knowledge = 1.0 (all known).
        """
        if pokemon is None:
            return ([0] * 13 +
                    [0.0]*6 + [0.0] + [1.0]+[0.0]*6 + [0.0]*7 + [0.0] +
                    [0.0]*6 + [0.0] + [0.0] +
                    [0.0])  # +1 for move_knowledge

        species_idx = self._get_pokemon_idx(pokemon.species)
        moves = list(pokemon.moves.values())
        move_indices, move_type_indices = [], []
        for i in range(4):
            if i < len(moves):
                move_indices.append(self._get_move_idx(moves[i].id))
                move_type_indices.append(TYPE_TO_IDX.get(moves[i].type, 0))
            else:
                move_indices.append(0)
                move_type_indices.append(0)
        ability_idx = self._get_ability_idx(pokemon.ability) if pokemon.ability else 0
        item_idx = self._get_item_idx(pokemon.item) if pokemon.item else 0
        types = pokemon.types if pokemon.types else []
        t1 = TYPE_TO_IDX.get(types[0] if len(types) > 0 else None, 0)
        t2 = TYPE_TO_IDX.get(types[1] if len(types) > 1 else None, 0)
        indices = [species_idx] + move_indices + [ability_idx, item_idx, t1, t2] + move_type_indices

        stats = pokemon.base_stats
        base = ([stats.get("hp",80)/255, stats.get("atk",80)/255, stats.get("def",80)/255,
                 stats.get("spa",80)/255, stats.get("spd",80)/255, stats.get("spe",80)/255]
                if stats else [80/255]*6)

        hp = pokemon.current_hp_fraction if pokemon.current_hp_fraction is not None else 0.0

        status_vec = [0.0]*7
        if pokemon.status is None:
            status_vec[0] = 1.0
        else:
            sm = {"BRN":1,"FRZ":2,"PAR":3,"PSN":4,"SLP":5,"TOX":6}
            status_vec[sm.get(pokemon.status.name, 0)] = 1.0

        boosts = [pokemon.boosts.get(s,0)/6.0 for s in ["atk","def","spa","spd","spe","accuracy","evasion"]]
        alive = 0.0 if pokemon.fainted else 1.0

        vols = [
            1.0 if check_effect(pokemon, "leechseed") else 0.0,
            1.0 if check_effect(pokemon, "confusion") else 0.0,
            1.0 if check_effect(pokemon, "substitute") else 0.0,
            1.0 if check_effect(pokemon, "taunt") else 0.0,
            1.0 if check_effect(pokemon, "encore") else 0.0,
            1.0 if (check_effect(pokemon, "trapped") or check_effect(pokemon, "partiallytrapped")) else 0.0,
        ]

        toxic_c = 0.0
        if pokemon.status and pokemon.status.name == "TOX":
            toxic_c = min(getattr(pokemon, 'status_counter', 1) / 16.0, 1.0)

        pp_s, pp_n = 0.0, 0
        for m in moves:
            if m.max_pp and m.max_pp > 0:
                pp_s += m.current_pp / m.max_pp
                pp_n += 1
        avg_pp = pp_s / max(pp_n, 1)

        if is_opponent:
            move_knowledge = len(moves) / 4.0
        else:
            move_knowledge = 1.0

        floats = base + [hp] + status_vec + boosts + [alive] + vols + [toxic_c, avg_pp] + [move_knowledge]
        assert len(indices) == self.PER_POKEMON_INDICES
        assert len(floats) == self.PER_POKEMON_FLOATS, f"{len(floats)} != {self.PER_POKEMON_FLOATS}"
        return indices + floats

    def _encode_active_move_details(self, battle):
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        avail = battle.available_moves if battle.available_moves else []
        max_pow = max([m.base_power or 0 for m in avail], default=0)

        mf, mi = [], []
        for i in range(4):
            if i < len(avail):
                m = avail[i]
                power = m.base_power/150.0 if m.base_power else 0.0
                acc = m.accuracy if m.accuracy is not None else 1.0
                ip = 1.0 if m.category == MoveCategory.PHYSICAL else 0.0
                isp = 1.0 if m.category == MoveCategory.SPECIAL else 0.0
                ist = 1.0 if m.category == MoveCategory.STATUS else 0.0
                eff = opp.damage_multiplier(m) if opp else 1.0
                stab = 1.0 if my and m.type in my.types else 0.0
                try:
                    prio = m.priority
                except (KeyError, AttributeError):
                    prio = 0
                pp = 1.0 if prio > 0 else 0.0
                pn = 1.0 if prio < 0 else 0.0
                sr = 0.5
                if my and opp:
                    if m.category == MoveCategory.PHYSICAL:
                        a, d = my.base_stats.get("atk",100), opp.base_stats.get("def",100)
                        sr = a/max(a+d,1)
                    elif m.category == MoveCategory.SPECIAL:
                        a, d = my.base_stats.get("spa",100), opp.base_stats.get("spd",100)
                        sr = a/max(a+d,1)
                isu = 1.0 if hasattr(m,'id') and m.id.lower() in SETUP_MOVES else 0.0
                ppf = (m.current_pp/m.max_pp) if m.max_pp and m.max_pp > 0 else 1.0
                mf.extend([power,acc,ip,isp,ist,eff,stab,pp,pn,sr,isu,ppf])
                mi.append(self._get_move_idx(m.id))
            else:
                mf.extend([0.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.5,0.0,0.0])
                mi.append(0)
        return mf, mi

    def _encode_global(self, battle):
        g = []
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        wv = [0.0]*6
        if battle.weather:
            wm = {"SUNNYDAY":1,"DESOLATELAND":1,"RAINDANCE":2,"PRIMORDIALSEA":2,
                  "SANDSTORM":3,"SNOW":4,"HAIL":4,"DELTASTREAM":5}
            for w in battle.weather: wv[wm.get(w.name,0)]=1.0; break
        else: wv[0]=1.0
        g.extend(wv)

        tv = [0.0]*5
        if battle.fields:
            tm = {"ELECTRIC_TERRAIN":1,"GRASSY_TERRAIN":2,"MISTY_TERRAIN":3,"PSYCHIC_TERRAIN":4}
            for f in battle.fields: tv[tm.get(f.name,0)]=1.0; break
        else: tv[0]=1.0
        g.extend(tv)

        g.append(battle.side_conditions.get(SideCondition.SPIKES,0)/3.0)
        g.append(1.0 if SideCondition.STEALTH_ROCK in battle.side_conditions else 0.0)
        g.append(battle.side_conditions.get(SideCondition.TOXIC_SPIKES,0)/2.0)
        g.append(1.0 if SideCondition.STICKY_WEB in battle.side_conditions else 0.0)
        g.append(battle.opponent_side_conditions.get(SideCondition.SPIKES,0)/3.0)
        g.append(1.0 if SideCondition.STEALTH_ROCK in battle.opponent_side_conditions else 0.0)
        g.append(battle.opponent_side_conditions.get(SideCondition.TOXIC_SPIKES,0)/2.0)
        g.append(1.0 if SideCondition.STICKY_WEB in battle.opponent_side_conditions else 0.0)

        g.append(1.0 if SideCondition.REFLECT in battle.side_conditions else 0.0)
        g.append(1.0 if SideCondition.LIGHT_SCREEN in battle.side_conditions else 0.0)
        g.append(1.0 if SideCondition.REFLECT in battle.opponent_side_conditions else 0.0)
        g.append(1.0 if SideCondition.LIGHT_SCREEN in battle.opponent_side_conditions else 0.0)

        g.append(min(battle.turn/100.0, 1.0))

        g.append(1.0 if getattr(battle,"can_tera",False) else 0.0)
        oct = getattr(battle,"opponent_can_tera",None)
        if oct is None: oct = not getattr(battle,"opponent_used_tera",False)
        g.append(1.0 if oct else 0.0)
        g.append(1.0 if my and my.is_terastallized else 0.0)

        g.append(len(battle.opponent_team)/6.0)

        g.append(1.0 if self._last_action_was_switch else 0.0)
        g.append(min(self._turns_since_switch/10.0, 1.0))
        g.append(self._detect_oscillation())
        bench = [m for m in battle.team.values() if not m.active and not m.fainted]
        ah = sum(self._calc_hazard_damage(battle,m) for m in bench)/max(len(bench),1) if bench else 0.0
        g.append(ah)
        bm = 0.0
        for mon in bench:
            if opp:
                o = get_type_multiplier(mon.types, opp.types)
                d = get_type_multiplier(opp.types, mon.types)
                bm = max(bm, min((o/4)-(d/4)+0.5, 1.0))
        g.append(bm)

        g.append(1.0 if battle.turn <= 5 else 0.0)
        g.append(1.0 if 5 < battle.turn <= 20 else 0.0)
        g.append(1.0 if battle.turn > 20 else 0.0)

        oa = sum(1 for m in battle.team.values() if not m.fainted)
        ea = sum(1 for m in battle.opponent_team.values()
                 if m.current_hp_fraction is not None and m.current_hp_fraction > 0)
        g.extend([oa/6.0, ea/6.0, (oa-ea+6)/12.0])

        if my and opp:
            ms = my.base_stats.get("spe",100)
            os_ = opp.base_stats.get("spe",100)
            if my.status and my.status.name=="PAR": ms*=0.5
            if opp.status and opp.status.name=="PAR": os_*=0.5
            ms *= (2+my.boosts.get("spe",0))/2
            os_ *= (2+opp.boosts.get("spe",0))/2
            speed_ratio = ms / max(ms + os_, 1)
            g.append(speed_ratio)
        else:
            g.append(0.5)

        g.append(1.0 if battle.available_switches else 0.0)

        g.append(self._opponent_seems_passive(battle))
        hs = any(hasattr(m,'id') and m.id.lower() in SETUP_MOVES for m in (battle.available_moves or []))
        g.append(1.0 if hs else 0.0)
        cb = sum(max(0,v) for v in my.boosts.values()) if my else 0
        g.append(min(cb/6.0, 1.0))

        hf = my.current_hp_fraction if my else 1.0
        hm = 1.0 - hf
        hr = any(hasattr(m,'id') and m.id.lower() in RECOVERY_MOVES for m in (battle.available_moves or []))
        hv = hm if 0.2 < hf < 0.7 else 0.0
        g.extend([hm, 1.0 if hr else 0.0, hv])

        mi_ = (my.item or "").lower() if my else ""
        g.append(1.0 if "choice" in mi_ else 0.0)
        g.append(1.0 if mi_=="lifeorb" else 0.0)
        g.append(1.0 if mi_=="heavydutyboots" else 0.0)
        g.append(1.0 if mi_=="leftovers" else 0.0)
        oi = (opp.item or "").lower() if opp else ""
        g.append(1.0 if "choice" in oi and oi not in ["","unknown_item"] else 0.0)

        ma = (my.ability or "").lower() if my else ""
        g.append(1.0 if ma=="levitate" else 0.0)
        g.append(1.0 if ma=="intimidate" else 0.0)
        oa_ = (opp.ability or "").lower() if opp else ""
        g.append(1.0 if oa_=="sturdy" else 0.0)

        for i in range(5):
            if i < len(bench) and opp:
                best_eff = 0.0
                bench_mon = bench[i]
                for move in bench_mon.moves.values():
                    try:
                        eff = opp.damage_multiplier(move)
                        best_eff = max(best_eff, eff)
                    except Exception:
                        pass
                g.append(min(best_eff / 4.0, 1.0))
            else:
                g.append(0.0)

        assert len(g) == self.N_GLOBAL, f"Global: {len(g)} != {self.N_GLOBAL}"
        return g

    def _opponent_seems_passive(self, battle):
        opp = battle.opponent_active_pokemon
        my = battle.active_pokemon
        if not opp or not my: return 0.5
        sc = 0.0
        if get_type_multiplier(opp.types, my.types) < 1.0: sc += 0.2
        if get_type_multiplier(my.types, opp.types) > 1.5: sc += 0.3
        if opp.current_hp_fraction and opp.current_hp_fraction < 0.3: sc += 0.2
        if opp.status: sc += 0.15
        if check_effect(my, "substitute"): sc += 0.3
        return min(sc, 1.0)

    def _calc_hazard_damage(self, battle, pokemon):
        if pokemon is None: return 0.0
        d = 0.0
        if SideCondition.STEALTH_ROCK in battle.side_conditions:
            re = 1.0
            for t in pokemon.types:
                if t:
                    try: re *= GEN_DATA.type_chart["ROCK"][t.name]
                    except KeyError: pass
            d += 0.125 * re
        sp = battle.side_conditions.get(SideCondition.SPIKES, 0)
        if sp > 0 and not self._has_ground_immunity(pokemon):
            d += [0,0.125,0.167,0.25][min(sp,3)]
        ts = battle.side_conditions.get(SideCondition.TOXIC_SPIKES, 0)
        if ts > 0 and not self._has_ground_immunity(pokemon):
            if "POISON" not in [t.name for t in pokemon.types if t]:
                d += 0.06 * ts
        if SideCondition.STICKY_WEB in battle.side_conditions:
            if not self._has_ground_immunity(pokemon): d += 0.02
        return min(d, 1.0)

    def _has_ground_immunity(self, p):
        if p is None: return False
        if "FLYING" in [t.name for t in p.types if t]: return True
        if p.ability and p.ability.lower()=="levitate": return True
        if p.item and p.item.lower()=="airballoon": return True
        return False

    def _detect_oscillation(self):
        if len(self._switch_history) < 3: return 0.0
        if len(self._switch_history) >= 4:
            r = self._switch_history[-4:]
            if r[0]==r[2] and r[1]==r[3] and r[0]!=r[1]: return 1.0
        r = self._switch_history[-3:]
        if r[0]==r[2] and r[0]!=r[1]: return 0.7
        return 0.0

    def embed_battle(self, battle):
        if battle.battle_tag != self._last_battle_tag:
            self._reset_battle_state()
            self._last_battle_tag = battle.battle_tag
        obs = []

        our_team = sorted(battle.team.values(), key=lambda p: not p.active)[:6]
        for mon in our_team:
            obs.extend(self._encode_pokemon_block(mon, is_opponent=False))
        for _ in range(6 - min(6, len(battle.team))):
            obs.extend(self._encode_pokemon_block(None, is_opponent=False))

        opp_list = sorted(battle.opponent_team.values(), key=lambda p: not p.active)
        for i in range(6):
            obs.extend(self._encode_pokemon_block(
                opp_list[i] if i < len(opp_list) else None,
                is_opponent=True
            ))

        mf, mi = self._encode_active_move_details(battle)
        obs.extend(mf)
        obs.extend(mi)
        obs.extend(self._encode_global(battle))
        assert len(obs) == self.OBSERVATION_SIZE, f"{len(obs)} != {self.OBSERVATION_SIZE}"
        return np.array(obs, dtype=np.float32)

    def _compute_potential(self, battle):
        p = 0.0
        oh = sum(m.current_hp_fraction for m in battle.team.values())
        eh = sum(m.current_hp_fraction for m in battle.opponent_team.values() if m.current_hp_fraction is not None)
        p += (oh-eh)*0.1
        oa = sum(1 for m in battle.team.values() if m.current_hp_fraction > 0)
        ea = sum(1 for m in battle.opponent_team.values() if m.current_hp_fraction is not None and m.current_hp_fraction > 0)
        p += (oa-ea)*0.05
        my,opp = battle.active_pokemon, battle.opponent_active_pokemon
        if my and opp:
            o = max((opp.damage_multiplier(m) for m in my.moves.values()), default=1.0)
            d = max((my.damage_multiplier(m) for m in opp.moves.values()), default=1.0)
            p += (np.log2(max(o,0.25))-np.log2(max(d,0.25)))*0.025
        if my and my.status: p -= 0.03
        if opp and opp.status: p += 0.03
        uh=0
        if SideCondition.STEALTH_ROCK in battle.side_conditions: uh+=0.125
        uh += battle.side_conditions.get(SideCondition.SPIKES,0)*0.08
        eh_=0
        if SideCondition.STEALTH_ROCK in battle.opponent_side_conditions: eh_+=0.125
        eh_ += battle.opponent_side_conditions.get(SideCondition.SPIKES,0)*0.08
        p += (eh_-uh)*0.2
        if my: p += sum(my.boosts.values())*0.01
        if opp: p -= sum(opp.boosts.values())*0.01
        return p

    def calc_reward(self, battle, auditor=None):
        base = self.reward_computing_helper(battle, fainted_value=0.15, hp_value=0.0, victory_value=1.0)
        if battle.battle_tag != self._last_battle_tag:
            self._last_potential = 0.0
            self._last_battle_tag = battle.battle_tag
        c = self._compute_potential(battle)
        pbrs = (0.99*c - self._last_potential)*0.3
        self._last_potential = c
        osc = 0.0
        if self._last_action is not None and self._last_action >= 8:
            o = self._detect_oscillation()
            if o > 0.5: osc = -0.03*o
        t = base + pbrs + osc
        if auditor: auditor.on_step(base, pbrs, t, c)
        return t

    def get_action_mask(self, battle):
        mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
        if battle.force_switch:
            for i in range(min(5, len(battle.available_switches))): mask[8+i]=1.0
            if mask.sum()==0: mask[8]=1.0
            return mask
        for i in range(min(4, len(battle.available_moves))): mask[i]=1.0
        if battle.can_tera:
            for i in range(min(4, len(battle.available_moves))): mask[4+i]=1.0
        for i in range(min(5, len(battle.available_switches))): mask[8+i]=1.0
        if mask.sum()==0: mask[0]=1.0
        return mask

    def action_to_order(self, action, battle, *a, **kw):
        action = int(action)
        nm = len(battle.available_moves) if battle.available_moves else 0
        ns = len(battle.available_switches) if battle.available_switches else 0
        if nm==0 and ns==0: return DefaultBattleOrder()
        if battle.force_switch:
            if action>=8:
                si=action-8
                if 0<=si<ns: return SingleBattleOrder(battle.available_switches[si])
            return SingleBattleOrder(battle.available_switches[0]) if ns>0 else DefaultBattleOrder()
        if action<0:
            return SingleBattleOrder(battle.available_moves[0]) if nm>0 else (
                SingleBattleOrder(battle.available_switches[0]) if ns>0 else DefaultBattleOrder())
        if action<4 and action<nm: return SingleBattleOrder(battle.available_moves[action], terastallize=False)
        elif action<8:
            mi=action-4
            if 0<=mi<nm: return SingleBattleOrder(battle.available_moves[mi], terastallize=battle.can_tera)
        else:
            si=action-8
            if 0<=si<ns: return SingleBattleOrder(battle.available_switches[si])
        if nm>0: return SingleBattleOrder(battle.available_moves[0])
        if ns>0: return SingleBattleOrder(battle.available_switches[0])
        return DefaultBattleOrder()

    def order_to_action(self, order, battle, *a, **kw):
        if order.order in battle.available_moves:
            mi = battle.available_moves.index(order.order)
            return (4+mi) if hasattr(order,'terastallize') and order.terastallize else mi
        if order.order in battle.available_switches:
            return 8 + battle.available_switches.index(order.order)
        return -1

    def choose_move(self, battle):
        return self.choose_random_move(battle)