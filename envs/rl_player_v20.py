"""v20 environment: entity-token observations and a 14-action space.

Observation, from `Featurizer.featurize(battle)` -> (idx int64[N_IDX], flt float32[N_FLT], mask bool[14]):
  - 12 Pokemon tokens: 6 own in `battle.team` order, then 6 opponent (revealed ones in first-seen
    order, then "unrevealed" placeholders). Index fields feed embeddings, floats are the rest.
  - 4 move tokens: token i is `battle.available_moves[i]`, the move behind actions i and 4 + i.
  - 1 field token (weather, terrain, hazards, screens, counters, matchup summary).
Actions: 0-3 use available move i, 4-7 the same move with Terastallization, 8-13 switch to
`battle.team` slot k (with Revival Blessing: revive slot k). Masks are legality only.

The featurizer is a function of the battle object alone (per-battle bookkeeping lives in a
WeakKeyDictionary keyed on the battle), so training, self-play opponents, live play and evaluation
all build observations the same way, for either side of a battle.
"""
import asyncio
import functools
import json
import math
import weakref
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box, Discrete

from poke_env.battle import Effect, Field, MoveCategory, PokemonType, SideCondition, Status, Weather
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.concurrency import POKE_LOOP
from poke_env.data import GenData, to_id_str
from poke_env.environment.env import PokeEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, SingleBattleOrder

GEN_DATA = GenData.from_gen(9)
TYPE_CHART = GEN_DATA.type_chart  # TYPE_CHART[defender][attacker]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Copy of pokemon-showdown/data/random-battles/gen9/sets.json (Feb 2026 fork), pinned so the
# observation doesn't change with the Showdown checkout
SETS_PATH = DATA_DIR / "gen9_randbats_sets.json"

N_ACTIONS = 14
MOVE0, TERA0, SWITCH0 = 0, 4, 8

# ---------------------------------------------------------------------------------------------
# Vocabularies. 0 is padding everywhere; ids missing from a vocab map to its OTHER index.
TYPES = [PokemonType.NORMAL, PokemonType.FIRE, PokemonType.WATER, PokemonType.ELECTRIC,
         PokemonType.GRASS, PokemonType.ICE, PokemonType.FIGHTING, PokemonType.POISON,
         PokemonType.GROUND, PokemonType.FLYING, PokemonType.PSYCHIC, PokemonType.BUG,
         PokemonType.ROCK, PokemonType.GHOST, PokemonType.DRAGON, PokemonType.DARK,
         PokemonType.STEEL, PokemonType.FAIRY]
TYPE_IDX = {t: i + 1 for i, t in enumerate(TYPES)}
TYPE_IDX[PokemonType.STELLAR] = 19
TYPE_IDX[PokemonType.THREE_QUESTION_MARKS] = 0
TYPE_UNKNOWN = 20
N_TYPE_IDX = 21


def _load_vocab(name):
    with open(DATA_DIR / f"{name}_to_idx.json") as f:
        return json.load(f)


SPECIES_VOCAB = _load_vocab("pokemon")   # <UNK>=0, 507 species
MOVE_VOCAB = _load_vocab("move")         # <UNK>=0, 350 moves
ABILITY_VOCAB = _load_vocab("ability")   # <UNK>=0, 203 abilities
ITEM_VOCAB = _load_vocab("item")         # <UNK>=0, 62 items including "none"

SPECIES_OTHER = len(SPECIES_VOCAB)       # 508
SPECIES_UNREVEALED = SPECIES_OTHER + 1   # 509
N_SPECIES_IDX = SPECIES_OTHER + 2
MOVE_OTHER = len(MOVE_VOCAB)             # 351
MOVE_STRUGGLE = MOVE_OTHER + 1
MOVE_RECHARGE = MOVE_OTHER + 2
N_MOVE_IDX = MOVE_OTHER + 3
ABILITY_UNKNOWN = len(ABILITY_VOCAB)     # 204: not revealed yet
ABILITY_OTHER = ABILITY_UNKNOWN + 1
N_ABILITY_IDX = ABILITY_UNKNOWN + 2
ITEM_UNKNOWN = len(ITEM_VOCAB)           # 63: not revealed yet
ITEM_OTHER = ITEM_UNKNOWN + 1
ITEM_NONE = ITEM_VOCAB["none"]           # no item, or it was consumed or removed
N_ITEM_IDX = ITEM_UNKNOWN + 2

VOCAB_SIZES = {"species": N_SPECIES_IDX, "move": N_MOVE_IDX, "ability": N_ABILITY_IDX,
               "item": N_ITEM_IDX, "type": N_TYPE_IDX}


def species_index(mon):
    idx = SPECIES_VOCAB.get(mon.species)
    if idx is None:
        # battle formes (e.g. palafinhero) fall back to the species the set was built for
        try:
            idx = SPECIES_VOCAB.get(mon.base_species)
        except KeyError:
            idx = None
    return idx if idx else SPECIES_OTHER


def move_index(move_id):
    if move_id == "struggle":
        return MOVE_STRUGGLE
    if move_id == "recharge":
        return MOVE_RECHARGE
    return MOVE_VOCAB.get(move_id) or MOVE_OTHER


# ---------------------------------------------------------------------------------------------
# Token layouts. Feature order is defined by these name lists; sizes follow from them.
PK_IDX_NAMES = ["species", "item", "ability", "tera_type", "type1", "type2",
                "move0", "move1", "move2", "move3", "last_move"]
PK_EFFECTS = [
    ("v_substitute", (Effect.SUBSTITUTE,)), ("v_leechseed", (Effect.LEECH_SEED,)),
    ("v_confusion", (Effect.CONFUSION,)), ("v_taunt", (Effect.TAUNT,)),
    ("v_encore", (Effect.ENCORE,)), ("v_trapped", (Effect.TRAPPED,)),
    ("v_partialtrap", (Effect.PARTIALLY_TRAPPED, Effect.BIND, Effect.WRAP, Effect.FIRE_SPIN,
                       Effect.WHIRLPOOL, Effect.SAND_TOMB, Effect.MAGMA_STORM, Effect.INFESTATION,
                       Effect.CLAMP, Effect.SNAP_TRAP, Effect.THUNDER_CAGE)),
    ("v_yawn", (Effect.YAWN,)), ("v_destinybond", (Effect.DESTINY_BOND,)),
    ("v_focusenergy", (Effect.FOCUS_ENERGY,)), ("v_magnetrise", (Effect.MAGNET_RISE,)),
    ("v_saltcure", (Effect.SALT_CURE,)), ("v_curse", (Effect.CURSE,)),
    ("v_disable", (Effect.DISABLE,)), ("v_healblock", (Effect.HEAL_BLOCK,)),
    ("v_throatchop", (Effect.THROAT_CHOP,)), ("v_charge", (Effect.CHARGE,)),
    ("v_flashfire", (Effect.FLASH_FIRE,)), ("v_glaiverush", (Effect.GLAIVE_RUSH,)),
    ("v_noretreat", (Effect.NO_RETREAT,)), ("v_tarshot", (Effect.TAR_SHOT,)),
    ("v_torment", (Effect.TORMENT,)), ("v_syrupbomb", (Effect.SYRUP_BOMB,)),
    ("v_smackdown", (Effect.SMACK_DOWN,)), ("v_aquaring", (Effect.AQUA_RING, Effect.INGRAIN)),
    ("v_attract", (Effect.ATTRACT,)), ("v_lockedmove", (Effect.LOCKED_MOVE,)),
    ("v_slowstart", (Effect.SLOW_START,)), ("v_dragoncheer", (Effect.DRAGON_CHEER,)),
    ("v_roost", (Effect.ROOST,)), ("v_embargo", (Effect.EMBARGO,)),
]
BOOSTED_STAT_EFFECTS = {
    Effect.PROTOSYNTHESISATK: 0, Effect.QUARKDRIVEATK: 0,
    Effect.PROTOSYNTHESISDEF: 1, Effect.QUARKDRIVEDEF: 1,
    Effect.PROTOSYNTHESISSPA: 2, Effect.QUARKDRIVESPA: 2,
    Effect.PROTOSYNTHESISSPD: 3, Effect.QUARKDRIVESPD: 3,
    Effect.PROTOSYNTHESISSPE: 4, Effect.QUARKDRIVESPE: 4,
}
PK_FLT_NAMES = (
    ["is_own", "is_placeholder", "revealed", "active", "fainted", "hp_frac", "level",
     "stat_hp", "stat_atk", "stat_def", "stat_spa", "stat_spd", "stat_spe",
     "st_none", "st_brn", "st_frz", "st_par", "st_psn", "st_slp", "st_tox",
     "sleep_turns", "toxic_turns",
     "b_atk", "b_def", "b_spa", "b_spd", "b_spe", "b_acc", "b_eva",
     "terastallized", "tera_known", "item_known", "ability_known", "moves_known",
     "first_turn", "protect_counter", "must_recharge", "preparing", "turns_in",
     "hazard_dmg", "grounded", "off_eff", "def_stab_eff", "def_move_eff", "speed_share",
     "pp0", "pp1", "pp2", "pp3", "stockpile"]
    + [name for name, _ in PK_EFFECTS]
    + ["boosted_atk", "boosted_def", "boosted_spa", "boosted_spd", "boosted_spe"]
)
MV_IDX_NAMES = ["move", "move_type"]
MV_FLT_NAMES = [
    "exists", "bp", "acc", "physical", "special", "status", "priority", "pp_frac",
    "eff", "type_immune", "stab", "stab_tera", "eff_tera",
    "dmg_lo", "dmg_hi", "ko_hi", "ko_lo", "calc_ok",
    "hits", "drain", "recoil", "heal", "self_switch", "force_switch", "hazard", "hazard_removal",
    "inflict_status", "self_boost", "target_boost", "protect", "weather_terrain", "secondary",
    "crit", "contact", "sound", "breaks_protect", "special_move", "volatile", "self_destruct",
    "trick_room",
]
SIDE_NAMES = ["spikes", "tspikes", "sr", "web", "reflect", "reflect_t", "lscreen", "lscreen_t",
              "veil", "veil_t", "tailwind", "tailwind_t", "safeguard", "mist"]
FD_FLT_NAMES = (
    ["w_none", "w_sun", "w_rain", "w_sand", "w_snow", "w_harshsun", "w_heavyrain", "w_winds",
     "weather_t", "t_none", "t_electric", "t_grassy", "t_misty", "t_psychic", "terrain_t",
     "trickroom", "trickroom_t", "gravity", "magicroom", "wonderroom"]
    + ["own_" + n for n in SIDE_NAMES] + ["opp_" + n for n in SIDE_NAMES]
    + ["turn", "log_turn", "can_tera", "used_tera", "opp_used_tera",
       "own_alive", "opp_alive", "opp_revealed",
       "force_switch", "trapped", "maybe_trapped", "reviving", "n_moves", "n_switches",
       "speed_share", "opp_best_dmg", "opp_ko_threat", "our_best_dmg", "our_ko"]
)

N_PK, N_MV = 12, 4
N_PK_IDX, N_PK_FLT = len(PK_IDX_NAMES), len(PK_FLT_NAMES)
N_MV_IDX, N_MV_FLT = len(MV_IDX_NAMES), len(MV_FLT_NAMES)
N_FD_FLT = len(FD_FLT_NAMES)
N_IDX = N_PK * N_PK_IDX + N_MV * N_MV_IDX
N_FLT = N_PK * N_PK_FLT + N_MV * N_MV_FLT + N_FD_FLT
# Offsets into the flat arrays (the network slices with these)
IDX_PK_END = N_PK * N_PK_IDX
FLT_PK_END = N_PK * N_PK_FLT
FLT_MV_END = FLT_PK_END + N_MV * N_MV_FLT
PK_FLT_POS = {n: i for i, n in enumerate(PK_FLT_NAMES)}
MV_FLT_POS = {n: i for i, n in enumerate(MV_FLT_NAMES)}
FD_FLT_POS = {n: i for i, n in enumerate(FD_FLT_NAMES)}

LAYOUT = {
    "n_pk": N_PK, "n_mv": N_MV, "n_pk_idx": N_PK_IDX, "n_pk_flt": N_PK_FLT,
    "n_mv_idx": N_MV_IDX, "n_mv_flt": N_MV_FLT, "n_fd_flt": N_FD_FLT,
    "n_idx": N_IDX, "n_flt": N_FLT, "n_actions": N_ACTIONS,
    "vocab": VOCAB_SIZES,
    "pk_active": PK_FLT_POS["active"],
}

# ---------------------------------------------------------------------------------------------
STATUS_POS = {None: 0, Status.BRN: 1, Status.FRZ: 2, Status.PAR: 3, Status.PSN: 4,
              Status.SLP: 5, Status.TOX: 6}
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
BOOST_MULT = {b: (2 + b) / 2 if b >= 0 else 2 / (2 - b) for b in range(-6, 7)}
WEATHER_POS = {Weather.SUNNYDAY: 1, Weather.RAINDANCE: 2, Weather.SANDSTORM: 3,
               Weather.SNOWSCAPE: 4, Weather.HAIL: 4, Weather.DESOLATELAND: 5,
               Weather.PRIMORDIALSEA: 6, Weather.DELTASTREAM: 7}
TERRAIN_POS = {Field.ELECTRIC_TERRAIN: 1, Field.GRASSY_TERRAIN: 2, Field.MISTY_TERRAIN: 3,
               Field.PSYCHIC_TERRAIN: 4}
HAZARD_CONDITIONS = {SideCondition.STEALTH_ROCK, SideCondition.SPIKES,
                     SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB}
HAZARD_REMOVAL = {"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}
ATE_ABILITIES = {"aerilate": PokemonType.FLYING, "pixilate": PokemonType.FAIRY,
                 "refrigerate": PokemonType.ICE, "galvanize": PokemonType.ELECTRIC}
WEATHER_BALL = {Weather.SUNNYDAY: PokemonType.FIRE, Weather.DESOLATELAND: PokemonType.FIRE,
                Weather.RAINDANCE: PokemonType.WATER, Weather.PRIMORDIALSEA: PokemonType.WATER,
                Weather.SANDSTORM: PokemonType.ROCK, Weather.SNOWSCAPE: PokemonType.ICE,
                Weather.HAIL: PokemonType.ICE}
TERRAIN_PULSE = {Field.ELECTRIC_TERRAIN: PokemonType.ELECTRIC, Field.GRASSY_TERRAIN: PokemonType.GRASS,
                 Field.MISTY_TERRAIN: PokemonType.FAIRY, Field.PSYCHIC_TERRAIN: PokemonType.PSYCHIC}
IVY_CUDGEL = {"ogerponwellspring": PokemonType.WATER, "ogerponhearthflame": PokemonType.FIRE,
              "ogerponcornerstone": PokemonType.ROCK}
RAGING_BULL = {"taurospaldeacombat": PokemonType.FIGHTING, "taurospaldeablaze": PokemonType.FIRE,
               "taurospaldeaaqua": PokemonType.WATER}


def _eff_code(mult):
    """Type effectiveness 0..4 as log2 / 3: immune -1, 1/4 -0.67, 1/2 -0.33, 1 0, 2 0.33, 4 0.67."""
    return math.log2(max(mult, 0.125)) / 3.0


def type_mult(atk_type, def_types):
    """Damage multiplier of an attacking type against a list of defending types."""
    if atk_type is None or atk_type in (PokemonType.STELLAR, PokemonType.THREE_QUESTION_MARKS):
        return 1.0
    m = 1.0
    for t in def_types:
        if t is None or t in (PokemonType.STELLAR, PokemonType.THREE_QUESTION_MARKS):
            continue
        m *= TYPE_CHART[t.name][atk_type.name]
    return m


def move_priority(move):
    try:
        return move.priority
    except KeyError:  # "recharge" has no dex entry
        return 0


def effective_move_type(move, mon, battle, tera=None):
    """The type the move hits with, for the common type-changing cases.

    tera: None uses mon's current state; True/False evaluates it as if (not) terastallized.
    """
    mid = move.id
    terastallized = mon.is_terastallized if tera is None else tera
    if mid == "terablast" and terastallized and mon.tera_type is not None:
        return mon.tera_type
    if mid == "weatherball" and battle.weather:
        return WEATHER_BALL.get(next(iter(battle.weather)), move.type)
    if mid == "terrainpulse" and battle.fields:
        for f in battle.fields:
            if f in TERRAIN_PULSE:
                return TERRAIN_PULSE[f]
    if mid == "ivycudgel":
        return IVY_CUDGEL.get(mon.species, move.type)
    if mid == "ragingbull":
        return RAGING_BULL.get(mon.species, move.type)
    if mid == "revelationdance":
        return mon.types[0] if mon.types else move.type
    if mid in ("judgment", "multiattack") and mon.species != mon.base_species:
        return mon.types[0] if mon.types else move.type
    if move.type == PokemonType.NORMAL and mon.ability in ATE_ABILITIES:
        return ATE_ABILITIES[mon.ability]
    return move.type


# ---------------------------------------------------------------------------------------------
class _Tracker:
    """Per-battle state read from the event log: weather start (poke-env resets it on every
    [upkeep] message), each Pokemon's last move this stint, switch-in turns, own move PP."""
    __slots__ = ("turn_cursor", "event_cursor", "weather_start", "last_move", "switch_turn", "pp")

    def __init__(self):
        self.turn_cursor = 0
        self.event_cursor = 0
        self.weather_start = None
        self.last_move = {}
        self.switch_turn = {}
        self.pp = {}

    @staticmethod
    def _key(ident):
        # "p1a: Name" -> "p1: Name", the form battle.team / opponent_team use as keys
        return ident[:2] + ident[3:] if len(ident) > 3 and ident[3] != " " else ident

    def _on_event(self, ev, turn):
        kind = ev[1] if len(ev) > 1 else ""
        if kind == "move" and len(ev) > 3:
            self.last_move[self._key(ev[2])] = to_id_str(ev[3])
        elif kind in ("switch", "drag") and len(ev) > 2:
            key = self._key(ev[2].split(",")[0])
            self.switch_turn[key] = turn
            self.last_move.pop(key, None)
        elif kind == "-weather" and len(ev) > 2:
            if ev[2] == "none":
                self.weather_start = None
            elif not any("[upkeep]" in x for x in ev[3:]):
                self.weather_start = turn

    def update(self, battle):
        obs = battle.observations
        current = battle.current_observation
        # finish turns that closed since the last call
        while self.turn_cursor in obs and obs[self.turn_cursor] is not current:
            events = obs[self.turn_cursor].events
            for ev in events[self.event_cursor:]:
                self._on_event(ev, self.turn_cursor)
            self.turn_cursor += 1
            self.event_cursor = 0
        events = current.events
        for ev in events[self.event_cursor:]:
            self._on_event(ev, battle.turn)
        self.event_cursor = len(events)
        # own active's PP from the request (poke-env never decrements Move.current_pp)
        req = battle.last_request
        active = req.get("active") if req else None
        if active and req.get("side"):
            for p in req["side"]["pokemon"]:
                if p.get("active"):
                    self.pp[p["ident"]] = {m["id"]: (m.get("pp", 1) / max(m.get("maxpp", 1), 1))
                                           for m in active[0].get("moves", []) if "id" in m}
                    break


# ---------------------------------------------------------------------------------------------
def team_list(battle):
    """Own team in battle.team order: switch action 8 + k targets team_list(battle)[k]."""
    return list(battle.team.values())[:6]


def action_mask(battle):
    """Legality only. Always has at least one legal action (0 stands in when nothing is)."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if not battle._wait and not battle.finished:
        moves = battle.available_moves
        if moves and not battle.force_switch:
            n = min(4, len(moves))
            mask[:n] = True
            special = n == 1 and moves[0].id in ("struggle", "recharge")
            if battle.can_tera and not special:
                mask[TERA0:TERA0 + n] = True
        avail = battle.available_switches
        if avail:
            for k, mon in enumerate(team_list(battle)):
                if any(mon is s for s in avail):
                    mask[SWITCH0 + k] = True
    if not mask.any():
        mask[0] = True
    return mask


def action_to_order(action, battle):
    """Integer action -> BattleOrder. Illegal actions fall back to Showdown's default choice."""
    a = int(action)
    moves = battle.available_moves
    if 0 <= a < 4 and a < len(moves) and not battle.force_switch:
        return SingleBattleOrder(moves[a])
    if 4 <= a < 8 and a - 4 < len(moves) and battle.can_tera and not battle.force_switch:
        return SingleBattleOrder(moves[a - 4], terastallize=True)
    if 8 <= a < 14:
        team = team_list(battle)
        k = a - 8
        if k < len(team) and any(team[k] is s for s in battle.available_switches):
            return SingleBattleOrder(team[k])
    return DefaultBattleOrder()


def order_to_action(order, battle):
    """BattleOrder -> integer action (-1 when it has none, e.g. the default order)."""
    if not isinstance(order, SingleBattleOrder) or isinstance(order, DefaultBattleOrder):
        return -1
    o = order.order
    if isinstance(o, Move):
        for i, m in enumerate(battle.available_moves[:4]):
            if m.id == o.id:
                return (TERA0 if order.terastallize else MOVE0) + i
        return -1
    if isinstance(o, Pokemon):
        for k, mon in enumerate(team_list(battle)):
            if mon is o:
                return SWITCH0 + k
        return -1
    return -1


# ---------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _uses_attack_stat(move_id, physical_tera_blast):
    try:
        mv = Move(move_id, gen=9)
        entry = mv.entry
    except Exception:
        return False
    if entry.get("damageCallback") or entry.get("damage"):
        return False
    if move_id == "shellsidearm":
        return True
    if move_id == "terablast":
        return physical_tera_blast
    return mv.category == MoveCategory.PHYSICAL and move_id not in ("bodypress", "foulplay")


def uses_attack_stat(move_id, species=None, base_stats=None, ability=None, moves=()):
    """Whether Showdown's random battle generator counts the move as needing Attack
    (data/random-battles/gen9/teams.ts, "Minimize confusion damage"): sets with none of these get
    0 Atk EVs and IVs. Tera Blast counts when it would be physical there."""
    physical_tb = (species == "porygon2" or ability in ("contrary", "defiant") or "shiftgear" in moves
                   or (bool(base_stats) and base_stats["atk"] > base_stats["spa"]))
    return _uses_attack_stat(move_id, physical_tb)


class Featurizer:
    """Builds v20 observations from a battle, from that battle's player's point of view."""

    def __init__(self, sets_path=SETS_PATH):
        self._trackers = weakref.WeakKeyDictionary()
        self._stat_cache = {}
        self.fallbacks = 0
        self.species_sets = self._load_sets(sets_path)

    @staticmethod
    def _load_sets(sets_path):
        """species -> [(movepool ids, whether the pool has an Attack-using move)]."""
        with open(sets_path) as f:
            sets = json.load(f)
        out = {}
        for species, entry in sets.items():
            base = GEN_DATA.pokedex[species]["baseStats"] if species in GEN_DATA.pokedex else None
            rows = []
            for s in entry["sets"]:
                pool = frozenset(to_id_str(n) for n in s["movepool"])
                abilities = [to_id_str(a) for a in s.get("abilities", [])]
                has_atk = any(uses_attack_stat(m, species, base, ab, pool)
                              for m in pool for ab in (abilities or [None]))
                rows.append((pool, has_atk))
            out[species] = rows
        return out

    def _atk_zero(self, mon):
        """Best guess at whether the opponent's set has 0 Atk: exact once all 4 moves are known,
        otherwise 0 only if no random-battle set consistent with the revealed moves could use Atk."""
        known = list(mon.moves)
        base = mon.base_stats
        if any(uses_attack_stat(m, mon.species, base, mon.ability, known) for m in known):
            return False
        if len(known) >= 4:
            return True
        rows = self.species_sets.get(mon.species)
        if not rows:
            return False
        consistent = [has_atk for pool, has_atk in rows if all(m in pool for m in known)] or \
            [has_atk for _, has_atk in rows]
        return not any(consistent)

    def tracker(self, battle):
        tr = self._trackers.get(battle)
        if tr is None:
            tr = _Tracker()
            self._trackers[battle] = tr
        tr.update(battle)
        return tr

    # ---- stats ------------------------------------------------------------------------------
    def estimated_stats(self, mon):
        """Opponent stats: gen9 random battles use 85 EVs, 31 IVs and a neutral nature, with
        0 Atk for sets without physical attacks and 0 Spe for Trick Room / Gyro Ball sets."""
        known = mon.moves
        spe_zero = "trickroom" in known or "gyroball" in known
        atk_zero = self._atk_zero(mon)
        base = mon.base_stats
        key = (mon.species, mon.level, atk_zero, spe_zero, base["hp"], base["atk"], base["spe"])
        stats = self._stat_cache.get(key)
        if stats is None:
            lvl = mon.level
            stats = {}
            for s in ("hp", "atk", "def", "spa", "spd", "spe"):
                ev, iv = 85, 31
                if (s == "atk" and atk_zero) or (s == "spe" and spe_zero):
                    ev, iv = 0, 0
                core = (2 * base[s] + iv + ev // 4) * lvl // 100
                stats[s] = core + lvl + 10 if s == "hp" else core + 5
            if mon.species == "shedinja":
                stats["hp"] = 1
            self._stat_cache[key] = stats
        return stats

    @staticmethod
    def own_stats(mon):
        st = mon.stats
        out = {s: (st.get(s) or 0) for s in ("atk", "def", "spa", "spd", "spe")}
        out["hp"] = st.get("hp") or mon.max_hp or 0
        return out

    @staticmethod
    def effective_speed(mon, stats, side_conditions):
        spe = stats["spe"] * BOOST_MULT[mon.boosts.get("spe", 0)]
        if mon.status == Status.PAR and mon.ability != "quickfeet":
            spe *= 0.5
        if mon.item == "choicescarf":
            spe *= 1.5
        if SideCondition.TAILWIND in side_conditions:
            spe *= 2
        return spe

    # ---- damage -----------------------------------------------------------------------------
    @staticmethod
    def _key_of(team, mon):
        for k, v in team.items():
            if v is mon:
                return k
        return None

    def _damage(self, battle, att_key, def_key, move, def_max_hp):
        try:
            lo, hi = calculate_damage(att_key, def_key, move, battle)
        except Exception:
            return None
        if def_max_hp <= 0:
            return None
        return lo / def_max_hp, hi / def_max_hp

    # ---- tokens -----------------------------------------------------------------------------
    def _pokemon_token(self, battle, mon, own, stats, other, other_stats, other_side_cond,
                       side_cond, own_tera, tr, key, req_ability=None):
        """(idx, flt) for one Pokemon. other = the other side's active (or None)."""
        types = mon.types
        t1 = TYPE_IDX.get(types[0], 0) if types else 0
        t2 = TYPE_IDX.get(types[1], 0) if len(types) > 1 else 0
        if own:
            tera_t = own_tera if own_tera is not None else (mon.tera_type or None)
            tera_idx = TYPE_IDX.get(tera_t, 0) if tera_t is not None else TYPE_UNKNOWN
        else:
            tera_idx = TYPE_IDX.get(mon.tera_type, 0) if mon.is_terastallized else TYPE_UNKNOWN
        item = mon.item
        if item == GEN_DATA.UNKNOWN_ITEM:
            item_idx = ITEM_UNKNOWN
        elif not item:
            item_idx = ITEM_NONE
        else:
            item_idx = ITEM_VOCAB.get(item) or ITEM_OTHER
        ability = req_ability if (own and req_ability) else mon.ability
        ability_idx = (ABILITY_VOCAB.get(ability) or ABILITY_OTHER) if ability else ABILITY_UNKNOWN
        moves = list(mon.moves.values())[:4]
        move_idx = [move_index(m.id) for m in moves] + [0] * (4 - len(moves))
        last = tr.last_move.get(key) if key else None
        idx = [species_index(mon), item_idx, ability_idx, tera_idx, t1, t2] + move_idx + [
            move_index(last) if last else 0]

        status = mon.status
        fainted = status == Status.FNT
        st = [0.0] * 7
        if not fainted:
            st[STATUS_POS.get(status, 0)] = 1.0
        boosts = mon.boosts
        effects = mon.effects
        active = bool(mon.active)
        turns_in = 0.0
        first_turn = False
        if active and key in tr.switch_turn:
            # poke-env's Pokemon.first_turn is never cleared by a normal move, so use the log:
            # a Pokemon that came in during turn t acts for the first time on turn t + 1
            since = battle.turn - tr.switch_turn[key]
            turns_in = min(since / 10.0, 1.0)
            first_turn = since == 1
        # entry hazard damage on switch-in, for this Pokemon's own side
        haz = 0.0
        if item != "heavydutyboots" and ability != "magicguard":
            if SideCondition.STEALTH_ROCK in side_cond:
                haz += 0.125 * type_mult(PokemonType.ROCK, types)
            layers = side_cond.get(SideCondition.SPIKES, 0)
            grounded = battle.is_grounded(mon)
            if layers and grounded:
                haz += (0.0, 1 / 8, 1 / 6, 1 / 4)[min(layers, 3)]
        else:
            grounded = battle.is_grounded(mon)
        # matchups against the other side's active
        off = d_stab = d_move = 0.0
        speed_share = 0.5
        if other is not None and not fainted:
            other_types = other.types
            att_types = [effective_move_type(m, mon, battle) for m in moves
                         if m.category != MoveCategory.STATUS]
            if not own:
                att_types += [t for t in mon.original_types if t is not None]
            if att_types:
                off = _eff_code(max(type_mult(t, other_types) for t in att_types))
            d_stab = _eff_code(max((type_mult(t, types) for t in other_types if t is not None),
                                   default=1.0))
            other_moves = [m for m in other.moves.values() if m.category != MoveCategory.STATUS]
            if other_moves:
                d_move = _eff_code(max(type_mult(effective_move_type(m, other, battle), types)
                                       for m in other_moves))
            my_spe = self.effective_speed(mon, stats, side_cond)
            their_spe = self.effective_speed(other, other_stats, other_side_cond)
            speed_share = my_spe / max(my_spe + their_spe, 1e-6)
        pp = [1.0, 1.0, 1.0, 1.0]
        if own:
            known_pp = tr.pp.get(key)
            if known_pp:
                for i, m in enumerate(moves):
                    pp[i] = known_pp.get(m.id, 1.0)
        boosted = [0.0] * 5
        for e in effects:
            pos = BOOSTED_STAT_EFFECTS.get(e)
            if pos is not None:
                boosted[pos] = 1.0
        counter = mon.status_counter
        flt = [
            1.0 if own else 0.0, 0.0, 1.0 if mon.revealed or not own else 0.0,
            1.0 if active else 0.0, 1.0 if fainted else 0.0, mon.current_hp_fraction,
            mon.level / 100.0,
            stats["hp"] / 500.0, stats["atk"] / 500.0, stats["def"] / 500.0,
            stats["spa"] / 500.0, stats["spd"] / 500.0, stats["spe"] / 500.0,
        ] + st + [
            min(counter / 3.0, 1.0) if status == Status.SLP else 0.0,
            min(counter / 15.0, 1.0) if status == Status.TOX else 0.0,
        ] + [boosts.get(b, 0) / 6.0 for b in BOOST_KEYS] + [
            1.0 if mon.is_terastallized else 0.0,
            1.0 if tera_idx != TYPE_UNKNOWN else 0.0,
            1.0 if item_idx != ITEM_UNKNOWN else 0.0,
            1.0 if ability_idx != ABILITY_UNKNOWN else 0.0,
            len(moves) / 4.0,
            1.0 if first_turn else 0.0,
            min(mon.protect_counter / 3.0, 1.0),
            1.0 if mon.must_recharge else 0.0,
            1.0 if mon.preparing else 0.0,
            turns_in, min(haz, 1.0), 1.0 if grounded else 0.0,
            off, d_stab, d_move, speed_share,
        ] + pp + [min(effects.get(Effect.STOCKPILE, 0) / 3.0, 1.0) if Effect.STOCKPILE in effects else 0.0]
        flt += [1.0 if any(e in effects for e in group) else 0.0 for _, group in PK_EFFECTS]
        flt += boosted
        return idx, flt

    @staticmethod
    def _placeholder_token():
        idx = [SPECIES_UNREVEALED, ITEM_UNKNOWN, ABILITY_UNKNOWN, TYPE_UNKNOWN, 0, 0, 0, 0, 0, 0, 0]
        flt = [0.0] * N_PK_FLT
        flt[PK_FLT_POS["is_placeholder"]] = 1.0
        flt[PK_FLT_POS["hp_frac"]] = 1.0  # unrevealed Pokemon are at full HP
        flt[PK_FLT_POS["st_none"]] = 1.0
        flt[PK_FLT_POS["speed_share"]] = 0.5
        flt[PK_FLT_POS["pp0"]] = flt[PK_FLT_POS["pp1"]] = 1.0
        flt[PK_FLT_POS["pp2"]] = flt[PK_FLT_POS["pp3"]] = 1.0
        return idx, flt

    def _move_token(self, battle, move, me, opp, opp_stats, own_tera, tr, me_key, dmg):
        mid = move.id
        cat = move.category
        mtype = effective_move_type(move, me, battle)
        eff_mult = type_mult(mtype, opp.types) if opp is not None else 1.0
        orig = [t for t in me.original_types if t is not None]
        stab = mtype in orig or (me.is_terastallized and mtype == me.tera_type)
        if own_tera is not None and not me.is_terastallized:
            tera_type = effective_move_type(move, me, battle, tera=False)
            if mid == "terablast":
                tera_type = own_tera if own_tera != PokemonType.STELLAR else move.type
            stab_tera = tera_type in orig or tera_type == own_tera or own_tera == PokemonType.STELLAR
            eff_tera = type_mult(tera_type, opp.types) if opp is not None else 1.0
        else:
            stab_tera, eff_tera = stab, eff_mult
        pp = 1.0
        known_pp = tr.pp.get(me_key) if me_key else None
        if known_pp:
            pp = known_pp.get(mid, 1.0)
        dlo = dhi = 0.0
        ko_hi = ko_lo = 0.0
        calc_ok = 0.0
        if dmg is not None:
            dlo, dhi = dmg
            calc_ok = 1.0
            hp = opp.current_hp_fraction if opp is not None else 1.0
            ko_hi = 1.0 if (dhi >= hp and dhi > 0) else 0.0
            ko_lo = 1.0 if (dlo >= hp and dlo > 0) else 0.0
        boosts = move.boosts
        self_boost = move.self_boost
        target = move.target
        sb = 0.0
        tb = 0.0
        if boosts:
            if target is not None and target.name in ("SELF", "ADJACENT_ALLY_OR_SELF", "ALLIES"):
                sb += sum(boosts.values())
            else:
                tb += sum(boosts.values())
        if self_boost:
            sb += sum(self_boost.values())
        secondary = move.secondary
        sec = max((s.get("chance", 0) or 0 for s in secondary), default=0) / 100.0 if secondary else 0.0
        flags = move.flags
        idx = [move_index(mid), TYPE_IDX.get(mtype, 0)]
        flt = [
            1.0, move.base_power / 150.0, float(move.accuracy),
            1.0 if cat == MoveCategory.PHYSICAL else 0.0,
            1.0 if cat == MoveCategory.SPECIAL else 0.0,
            1.0 if cat == MoveCategory.STATUS else 0.0,
            move_priority(move) / 5.0, pp,
            _eff_code(eff_mult) if cat != MoveCategory.STATUS else 0.0,
            1.0 if (eff_mult == 0 and cat != MoveCategory.STATUS) else 0.0,
            1.0 if stab else 0.0, 1.0 if stab_tera else 0.0,
            _eff_code(eff_tera) if cat != MoveCategory.STATUS else 0.0,
            min(dlo, 2.0), min(dhi, 2.0), ko_hi, ko_lo, calc_ok,
            min(move.expected_hits / 5.0, 2.0), move.drain, move.recoil, move.heal,
            1.0 if move.self_switch else 0.0, 1.0 if move.force_switch else 0.0,
            1.0 if move.side_condition in HAZARD_CONDITIONS else 0.0,
            1.0 if mid in HAZARD_REMOVAL else 0.0,
            1.0 if move.status is not None else 0.0,
            max(min(sb / 4.0, 1.0), -1.0), max(min(tb / 4.0, 1.0), -1.0),
            1.0 if move.is_protect_move else 0.0,
            1.0 if (move.weather is not None or move.terrain is not None) else 0.0,
            sec, min(move.crit_ratio / 3.0, 1.0),
            1.0 if "contact" in flags else 0.0, 1.0 if "sound" in flags else 0.0,
            1.0 if move.breaks_protect else 0.0,
            1.0 if mid in ("struggle", "recharge") else 0.0,
            1.0 if move.volatile_status is not None else 0.0,
            1.0 if move.self_destruct else 0.0,
            1.0 if move.pseudo_weather == "trickroom" else 0.0,
        ]
        return idx, flt

    @staticmethod
    def _special_move_token(move_idx):
        """Recharge (no dex entry) and moves whose data is missing: known to exist, nothing else."""
        flt = [0.0] * N_MV_FLT
        flt[MV_FLT_POS["exists"]] = 1.0
        flt[MV_FLT_POS["acc"]] = 1.0
        flt[MV_FLT_POS["pp_frac"]] = 1.0
        flt[MV_FLT_POS["special_move"]] = 1.0
        return [move_idx, 0], flt

    @staticmethod
    def _side_features(conds, turn):
        def t(c, n):
            return min((turn - conds[c]) / n, 1.0) if c in conds else 0.0
        return [
            conds.get(SideCondition.SPIKES, 0) / 3.0, conds.get(SideCondition.TOXIC_SPIKES, 0) / 2.0,
            1.0 if SideCondition.STEALTH_ROCK in conds else 0.0,
            1.0 if SideCondition.STICKY_WEB in conds else 0.0,
            1.0 if SideCondition.REFLECT in conds else 0.0, t(SideCondition.REFLECT, 8.0),
            1.0 if SideCondition.LIGHT_SCREEN in conds else 0.0, t(SideCondition.LIGHT_SCREEN, 8.0),
            1.0 if SideCondition.AURORA_VEIL in conds else 0.0, t(SideCondition.AURORA_VEIL, 8.0),
            1.0 if SideCondition.TAILWIND in conds else 0.0, t(SideCondition.TAILWIND, 4.0),
            1.0 if SideCondition.SAFEGUARD in conds else 0.0,
            1.0 if SideCondition.MIST in conds else 0.0,
        ]

    @staticmethod
    def own_request_info(battle):
        """ident -> (tera type, ability) from the request. Showdown sends every own Pokemon's
        teraType, which poke-env drops, and its current ability, which poke-env can overwrite from
        log announcements (As One shows up as Unnerve)."""
        out = {}
        req = battle.last_request
        if req and req.get("side"):
            for p in req["side"]["pokemon"]:
                tt = p.get("teraType")
                tera = None
                if tt:
                    try:
                        tera = PokemonType.from_name(tt)
                    except KeyError:
                        pass
                ab = p.get("ability") or p.get("baseAbility")
                out[p["ident"]] = (tera, to_id_str(ab) if ab else None)
        return out

    def featurize(self, battle):
        tr = self.tracker(battle)
        turn = battle.turn
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        side_cond = battle.side_conditions
        opp_side_cond = battle.opponent_side_conditions
        req_info = self.own_request_info(battle)
        tera_types = {k: v[0] for k, v in req_info.items()}
        team = battle.team
        opp_team = battle.opponent_team

        own_items = list(team.items())[:6]
        own_stats = {k: self.own_stats(m) for k, m in own_items}
        opp_items = list(opp_team.items())[:6]
        opp_stats = {k: self.estimated_stats(m) for k, m in opp_items}
        me_key = self._key_of(team, me) if me is not None else None
        opp_key = self._key_of(opp_team, opp) if opp is not None else None
        me_stats = own_stats.get(me_key) if me_key else None
        o_stats = opp_stats.get(opp_key) if opp_key else None

        idx, flt = [], []
        for k, mon in own_items:
            i, f = self._pokemon_token(battle, mon, True, own_stats[k], opp, o_stats, opp_side_cond,
                                       side_cond, tera_types.get(k), tr, k,
                                       req_info.get(k, (None, None))[1])
            idx += i
            flt += f
        for _ in range(6 - len(own_items)):
            i, f = self._placeholder_token()
            f[PK_FLT_POS["is_own"]] = 1.0
            idx += i
            flt += f
        for k, mon in opp_items:
            i, f = self._pokemon_token(battle, mon, False, opp_stats[k], me, me_stats, side_cond,
                                       opp_side_cond, None, tr, k)
            idx += i
            flt += f
        for _ in range(6 - len(opp_items)):
            i, f = self._placeholder_token()
            idx += i
            flt += f

        # damage estimates need the opponent's stats filled in; restore them afterwards
        moves = battle.available_moves[:4] if not battle.force_switch else []
        our_dmg = [None] * len(moves)
        opp_dmg = []
        can_calc = me is not None and opp is not None and me_key and opp_key and me_stats and \
            all(me.stats.get(s) is not None for s in ("atk", "def", "spa", "spd", "spe"))
        if can_calc:
            saved = opp._stats
            opp._stats = dict(o_stats)
            try:
                for i, m in enumerate(moves):
                    if m.id != "recharge":
                        our_dmg[i] = self._damage(battle, me_key, opp_key, m, o_stats["hp"])
                for m in list(opp.moves.values())[:4]:
                    d = self._damage(battle, opp_key, me_key, m, me_stats["hp"])
                    if d is not None:
                        opp_dmg.append(d)
            finally:
                opp._stats = saved

        own_tera = tera_types.get(me_key) if me_key else None
        for i in range(4):
            if i < len(moves) and me is not None:
                if moves[i].id == "recharge":
                    mi, mf = self._special_move_token(MOVE_RECHARGE)
                else:
                    try:
                        mi, mf = self._move_token(battle, moves[i], me, opp, o_stats, own_tera, tr,
                                                  me_key, our_dmg[i])
                    except (KeyError, AttributeError, TypeError):
                        # a move with missing dex data; counted so the checks can report it
                        self.fallbacks += 1
                        mi, mf = self._special_move_token(move_index(moves[i].id))
            else:
                mi, mf = [0, 0], [0.0] * N_MV_FLT
            idx += mi
            flt += mf

        # field token
        wv = [0.0] * 8
        weather_t = 0.0
        if battle.weather:
            w = next(iter(battle.weather))
            wv[WEATHER_POS.get(w, 0)] = 1.0
            if tr.weather_start is not None:
                weather_t = min((turn - tr.weather_start) / 8.0, 1.0)
        else:
            wv[0] = 1.0
        tv = [0.0] * 5
        terrain_t = 0.0
        fields = battle.fields
        terrain = [f for f in fields if f in TERRAIN_POS]
        if terrain:
            tv[TERRAIN_POS[terrain[0]]] = 1.0
            terrain_t = min((turn - fields[terrain[0]]) / 8.0, 1.0)
        else:
            tv[0] = 1.0
        tr_on = Field.TRICK_ROOM in fields
        own_alive = sum(1 for m in team.values() if not m.fainted)
        opp_alive = 6 - sum(1 for m in opp_team.values() if m.fainted)
        speed_share = 0.5
        if me is not None and opp is not None and me_stats and o_stats:
            a = self.effective_speed(me, me_stats, side_cond)
            b = self.effective_speed(opp, o_stats, opp_side_cond)
            speed_share = a / max(a + b, 1e-6)
        my_hp = me.current_hp_fraction if me is not None else 0.0
        opp_best = max((d[1] for d in opp_dmg), default=0.0)
        our_best = max((d[1] for d in our_dmg if d is not None), default=0.0)
        opp_hp = opp.current_hp_fraction if opp is not None else 1.0
        fd = wv + [weather_t] + tv + [terrain_t] + [
            1.0 if tr_on else 0.0,
            min((turn - fields[Field.TRICK_ROOM]) / 5.0, 1.0) if tr_on else 0.0,
            1.0 if Field.GRAVITY in fields else 0.0,
            1.0 if Field.MAGIC_ROOM in fields else 0.0,
            1.0 if Field.WONDER_ROOM in fields else 0.0,
        ] + self._side_features(side_cond, turn) + self._side_features(opp_side_cond, turn) + [
            min(turn / 50.0, 2.0), math.log1p(turn) / 4.0,
            1.0 if battle.can_tera else 0.0,
            1.0 if battle.used_tera else 0.0,
            1.0 if battle.opponent_used_tera else 0.0,
            own_alive / 6.0, opp_alive / 6.0, len(opp_team) / 6.0,
            1.0 if battle.force_switch else 0.0,
            1.0 if battle.trapped else 0.0,
            1.0 if battle.maybe_trapped else 0.0,
            1.0 if battle.reviving else 0.0,
            len(moves) / 4.0, len(battle.available_switches) / 5.0,
            speed_share,
            min(opp_best, 2.0), 1.0 if (opp_best >= my_hp and opp_best > 0) else 0.0,
            min(our_best, 2.0), 1.0 if (our_best >= opp_hp and our_best > 0) else 0.0,
        ]
        flt += fd
        assert len(idx) == N_IDX, (len(idx), N_IDX)
        assert len(flt) == N_FLT, (len(flt), N_FLT)
        return (np.asarray(idx, dtype=np.int64), np.asarray(flt, dtype=np.float32),
                action_mask(battle))


# ---------------------------------------------------------------------------------------------
def hold_requests_until_log(player, timeout=0.05):
    """Handle each |request| only after that battle's next log message.

    This Showdown fork (Feb 2026) sends a turn's |request| a fraction of a millisecond before the
    turn's battle log, and poke-env acts on a request as soon as it is parsed. The decision (and
    the env's observation) could then see the state before the turn's events, or halfway through
    them. Current Showdown sends the log first, which is what this restores. Requests that no log
    follows (e.g. a re-request after an invalid choice) are handled after `timeout` seconds.
    Only use this against a server that sends requests first: otherwise every request waits.
    """
    orig = player._handle_battle_message
    held = {}
    stats = {"held": 0, "timeouts": 0}

    async def release(tag):
        item = held.pop(tag, None)
        if item is not None:
            item[1].cancel()
            await orig(item[0])

    def on_timeout(tag):
        if tag in held:
            stats["timeouts"] += 1
            asyncio.ensure_future(release(tag))

    async def handler(split_messages):
        tag = split_messages[0][0]
        if any(len(m) > 2 and m[1] == "request" and m[2] for m in split_messages[1:]):
            await release(tag)
            handle = asyncio.get_running_loop().call_later(timeout, on_timeout, tag)
            held[tag] = (split_messages, handle)
            stats["held"] += 1
            return
        await orig(split_messages)
        await release(tag)

    player.ps_client._handle_battle_message = handler
    player.request_hold_stats = stats


class EnvTimeout(Exception):
    """A battle made no progress within the step timeout."""


def _race_get(queue, events, timeout):
    """Like poke-env's _AsyncQueue.race_get, with a timeout (raises EnvTimeout)."""
    async def _inner():
        get_task = asyncio.create_task(queue.async_get())
        wait_tasks = [asyncio.create_task(e.wait()) for e in events]
        done, pending = await asyncio.wait({get_task, *wait_tasks}, timeout=timeout,
                                           return_when=asyncio.FIRST_COMPLETED)
        for p in pending:
            p.cancel()
        if not done:
            raise EnvTimeout(f"no battle update within {timeout} s")
        if get_task in done:
            return get_task.result()
        return None
    return asyncio.run_coroutine_threadsafe(_inner(), POKE_LOOP).result()


def _get(queue, timeout):
    async def _inner():
        try:
            return await asyncio.wait_for(queue.async_get(), timeout)
        except asyncio.TimeoutError:
            raise EnvTimeout(f"no battle within {timeout} s")
    return asyncio.run_coroutine_threadsafe(_inner(), POKE_LOOP).result()


class RLEnvV20(PokeEnv):
    """Two-agent PokeEnv for v20. `embed_battle` returns the battle itself: the training loop
    featurizes only the sides it needs. Actions are ints (the v20 action space) or BattleOrders
    (for opponents that pick orders themselves, e.g. SimpleHeuristics or v18/v19 models).
    Every blocking wait has a timeout, so a stuck battle raises EnvTimeout instead of hanging."""

    def __init__(self, step_timeout=60.0, hold_requests=True, **kwargs):
        kwargs.setdefault("battle_format", "gen9randombattle")
        kwargs.setdefault("strict", False)
        super().__init__(**kwargs)
        self.step_timeout = step_timeout
        if hold_requests:
            hold_requests_until_log(self.agent1)
            hold_requests_until_log(self.agent2)

    @property
    def action_spaces(self):
        return {a: Discrete(N_ACTIONS) for a in self.possible_agents}

    @property
    def observation_spaces(self):
        return {a: Box(-1e3, 1e3, shape=(N_FLT,), dtype=np.float32) for a in self.possible_agents}

    def embed_battle(self, battle):
        return battle

    def calc_reward(self, battle):
        return 0.0

    @staticmethod
    def action_to_order(action, battle, fake=False, strict=True):
        if isinstance(action, BattleOrder):
            return action
        return action_to_order(action, battle)

    @staticmethod
    def order_to_action(order, battle, fake=False, strict=True):
        return order_to_action(order, battle)

    def reset(self, seed=None, options=None):
        """Start a new battle. Only for a fresh env or after the last battle finished."""
        self.agents = [self.agent1.username, self.agent2.username]
        if self.battle1 is not None and not self.battle1.finished:
            raise RuntimeError("reset() with a battle in progress; close the env instead")
        self.reset_battles()
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self.agent1.battle_against(self.agent2, n_battles=1), POKE_LOOP)
        self.battle1 = _get(self.agent1.battle_queue, self.step_timeout)
        self.battle2 = _get(self.agent2.battle_queue, self.step_timeout)
        self.agent1_to_move = True
        self.agent2_to_move = True
        return {self.agents[0]: self.battle1, self.agents[1]: self.battle2}, self.get_additional_info()

    def step(self, actions):
        """Send the orders of the agents that had to move and wait for the next decision.
        Afterwards agent1_to_move / agent2_to_move say who must decide."""
        a1, a2 = self.agent1.username, self.agent2.username
        if self.agent1_to_move:
            self.agent1_to_move = False
            self.agent1.order_queue.put(self.action_to_order(actions[a1], self.battle1))
        if self.agent2_to_move:
            self.agent2_to_move = False
            self.agent2.order_queue.put(self.action_to_order(actions[a2], self.battle2))
        battle1 = _race_get(self.agent1.battle_queue, (self.agent1._waiting, self.agent2._trying_again),
                            self.step_timeout)
        battle2 = _race_get(self.agent2.battle_queue, (self.agent2._waiting, self.agent1._trying_again),
                            self.step_timeout)
        self.agent1_to_move = battle1 is not None
        self.agent2_to_move = battle2 is not None
        if battle1 is None:
            self.agent1._waiting.clear()
            self.agent2._trying_again.clear()
            battle1 = self.battle1
        if battle2 is None:
            self.agent2._waiting.clear()
            self.agent1._trying_again.clear()
            battle2 = self.battle2
        self.battle1, self.battle2 = battle1, battle2
        done = battle1.finished
        if done:
            self.agents = []
        obs = {a1: battle1, a2: battle2}
        return (obs, {a1: 0.0, a2: 0.0}, {a1: done, a2: done}, {a1: False, a2: False},
                {a1: {}, a2: {}})

    def close(self, force=True, wait=False):
        try:
            super().close(force=force, wait=wait)
        except Exception:
            pass
