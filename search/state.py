"""poke-env battle -> poke-engine State (agent_docs/step6_brief.md, milestone 3).

`build_state(battle, opp_sets)` is the State from the battle's player's point of view: side one is the
player, exact from its request and the log; side two the opponent, what the log shows, the rest from
`opp_sets` (a RandSet per opponent Pokemon: battle.opponent_team order, then the unrevealed ones).
`build_true_state(battle, other)` builds side two from the other player's battle object instead (its
request has the opponent's exact team): the true state, for the checks.

Side one's Pokemon are in battle.team order, so engine index k is switch action 8 + k
(envs/rl_player_v20.team_list). `option_to_action` maps poke-engine's option strings back to actions.

poke-env doesn't keep everything the engine needs, so `LogTracker` reads it from the event log, like
envs.rl_player_v20._Tracker: each Pokemon's forme, switch-in turn, last move this stint, PP used,
substitute hits, toxic ticks, sleep and Rest turns, volatile start turns, original items (for set
filtering), Wish / Future Sight, and when and by whom weather, terrain and screens were set (their
remaining turns, with the setter's duration-extending item).

poke-engine panics on some values (a taunt duration outside 0-2, a yawn duration outside 0-1, Encore
without a last move, rest turns outside 0-3); everything here is clamped to what it accepts.
"""
import weakref
from collections import Counter

import poke_engine as pe
from poke_env.battle import Effect, Field, PokemonType, SideCondition, Status, Weather
from poke_env.data import GenData, to_id_str

from envs.rl_player_v20 import MOVE0, SWITCH0, TERA0, team_list

GEN = GenData.from_gen(9)
POKEDEX = GEN.pokedex
MOVES = GEN.moves

STATUS = {None: "none", Status.BRN: "burn", Status.PAR: "paralyze", Status.PSN: "poison",
          Status.TOX: "toxic", Status.SLP: "sleep", Status.FRZ: "freeze", Status.FNT: "none"}
WEATHER = {Weather.SUNNYDAY: "sun", Weather.RAINDANCE: "rain", Weather.SANDSTORM: "sand",
           Weather.SNOWSCAPE: "snow", Weather.HAIL: "hail", Weather.DESOLATELAND: "harshsun",
           Weather.PRIMORDIALSEA: "heavyrain"}
WEATHER_ROCK = {"sun": "heatrock", "rain": "damprock", "sand": "smoothrock", "snow": "icyrock",
                "hail": "icyrock"}
PRIMAL_WEATHER = {"harshsun", "heavyrain"}
TERRAIN = {Field.ELECTRIC_TERRAIN: "electricterrain", Field.GRASSY_TERRAIN: "grassyterrain",
           Field.MISTY_TERRAIN: "mistyterrain", Field.PSYCHIC_TERRAIN: "psychicterrain"}
# poke-env Effect -> poke-engine volatile status (durations and special cases are handled apart)
VOLATILES = {
    Effect.AQUA_RING: "aquaring", Effect.ATTRACT: "attract", Effect.CHARGE: "charge",
    Effect.CONFUSION: "confusion", Effect.CURSE: "curse", Effect.DEFENSE_CURL: "defensecurl",
    Effect.DESTINY_BOND: "destinybond", Effect.EMBARGO: "embargo", Effect.FLASH_FIRE: "flashfire",
    Effect.FOCUS_ENERGY: "focusenergy", Effect.FORESIGHT: "foresight", Effect.GASTRO_ACID: "gastroacid",
    Effect.GLAIVE_RUSH: "glaiverush", Effect.HEAL_BLOCK: "healblock", Effect.INGRAIN: "ingrain",
    Effect.LASER_FOCUS: "laserfocus", Effect.LEECH_SEED: "leechseed", Effect.MAGNET_RISE: "magnetrise",
    Effect.MINIMIZE: "minimize", Effect.MIRACLE_EYE: "miracleeye", Effect.NIGHTMARE: "nightmare",
    Effect.NO_RETREAT: "noretreat", Effect.OCTOLOCK: "octolock", Effect.POWER_TRICK: "powertrick",
    Effect.POWER_SHIFT: "powershift", Effect.SALT_CURE: "saltcure", Effect.SMACK_DOWN: "smackdown",
    Effect.STOCKPILE: "stockpile", Effect.SYRUP_BOMB: "syrupbomb", Effect.TAR_SHOT: "tarshot",
    Effect.TELEKINESIS: "telekinesis", Effect.THROAT_CHOP: "throatchop", Effect.TORMENT: "torment",
    Effect.UPROAR: "uproar", Effect.IMPRISON: "imprison", Effect.GRUDGE: "grudge", Effect.RAGE: "rage",
    Effect.PERISH1: "perish1", Effect.PERISH2: "perish2", Effect.PERISH3: "perish3",
    Effect.PROTOSYNTHESISATK: "protosynthesisatk", Effect.PROTOSYNTHESISDEF: "protosynthesisdef",
    Effect.PROTOSYNTHESISSPA: "protosynthesisspa", Effect.PROTOSYNTHESISSPD: "protosynthesisspd",
    Effect.PROTOSYNTHESISSPE: "protosynthesisspe", Effect.QUARKDRIVEATK: "quarkdriveatk",
    Effect.QUARKDRIVEDEF: "quarkdrivedef", Effect.QUARKDRIVESPA: "quarkdrivespa",
    Effect.QUARKDRIVESPD: "quarkdrivespd", Effect.QUARKDRIVESPE: "quarkdrivespe",
    Effect.PARTIALLY_TRAPPED: "partiallytrapped", Effect.BIND: "partiallytrapped",
    Effect.WRAP: "partiallytrapped", Effect.FIRE_SPIN: "partiallytrapped",
    Effect.WHIRLPOOL: "partiallytrapped", Effect.SAND_TOMB: "partiallytrapped",
    Effect.MAGMA_STORM: "partiallytrapped", Effect.INFESTATION: "partiallytrapped",
    Effect.CLAMP: "partiallytrapped", Effect.SNAP_TRAP: "partiallytrapped",
    Effect.THUNDER_CAGE: "partiallytrapped",
}
# moves that charge on the first turn: the engine's volatile while charging
CHARGE_MOVES = {"bounce", "dig", "dive", "fly", "freezeshock", "geomancy", "iceburn", "meteorbeam",
                "electroshot", "phantomforce", "razorwind", "shadowforce", "skullbash", "skyattack",
                "skydrop", "solarbeam", "solarblade"}
LOCKING_MOVES = {"outrage", "petaldance", "thrash", "ragingfury"}
CHOICE_ITEMS = ("choiceband", "choicespecs", "choicescarf")
SCREENS = {SideCondition.REFLECT: "reflect", SideCondition.LIGHT_SCREEN: "light_screen",
           SideCondition.AURORA_VEIL: "aurora_veil"}
TIMED_SIDE = {SideCondition.TAILWIND: ("tailwind", 4), SideCondition.SAFEGUARD: ("safeguard", 5),
              SideCondition.MIST: ("mist", 5)}


def _key(ident):
    """'p1a: Name' -> 'p1: Name' (the keys of battle.team / opponent_team)."""
    ident = ident.split(",")[0]
    return ident[:2] + ident[3:] if len(ident) > 3 and ident[3] != " " else ident


def _species_of(details):
    return to_id_str(details.split(",")[0])


def _tags(ev):
    """{'from': 'item: Life Orb', 'of': 'p1a: X', ...} for the [tag] fields of an event."""
    out = {}
    for x in ev[2:]:
        if x.startswith("["):
            name, _, val = x[1:].partition("]")
            out[name] = val.strip()
    return out


class _Mon:
    __slots__ = ("forme", "switch_turn", "after_upkeep", "last_move", "last_move_turn", "moves_since_in",
                 "pp_used", "sub_hit", "tox_ticks", "rest", "sleep_attempts", "taunt_start",
                 "encore_start", "yawn_turn", "lock_move", "lock_turn", "disabled_move", "orig_item",
                 "traded", "item_consumed", "prepare", "no_items", "stint_moves")

    def __init__(self):
        self.forme = None
        self.switch_turn = None
        self.after_upkeep = False
        self.last_move = None
        self.last_move_turn = None
        self.moves_since_in = []      # (move id, event index) this stint
        self.pp_used = Counter()
        self.sub_hit = False
        self.tox_ticks = 0
        self.rest = False
        self.sleep_attempts = 0
        self.taunt_start = None       # event index
        self.encore_start = None
        self.yawn_turn = None
        self.lock_move = None
        self.lock_turn = None
        self.disabled_move = None
        self.orig_item = None         # the first item seen on it that it didn't get from a trade
        self.traded = False
        self.item_consumed = False
        self.prepare = None           # (move id, turn) of a charge turn (-prepare) not yet released
        self.no_items = set()         # items its original item can't be, from what it did with it
        self.stint_moves = set()      # moves it chose itself this stint (not called by another move)


class LogTracker:
    """Facts from the battle's event log that poke-env doesn't keep. One per battle object (see
    tracker()); update() reads the events since the last call."""

    def __init__(self):
        self.turn_cursor = 0
        self.event_cursor = 0
        self.n_events = 0
        self.mons = {}
        self.weather = None           # (turn, after upkeep, setter key or None, by ability)
        self.terrain = None
        self.trick_room = None        # (turn, after upkeep)
        self.side_setters = {}        # (side 'p1', condition id) -> setter key
        self.wish = {}                # side -> (turn, user key)
        self.future_sight = {}        # side -> (turn, user key)
        self.acted = {}               # side -> turn of its last move / switch / cant
        self.turn_after_upkeep = False
        self.last_mover = None        # key of the Pokemon in the latest |move| event
        self.trick_pair = None
        self.pressure = {}            # side -> its active announced Pressure
        self.tera_user = {}           # side -> (key, tera type) of the Pokemon that terastallized

    def mon(self, key):
        m = self.mons.get(key)
        if m is None:
            m = self.mons[key] = _Mon()
        return m

    def update(self, battle):
        obs = battle.observations
        current = battle.current_observation
        while self.turn_cursor in obs and obs[self.turn_cursor] is not current:
            for ev in obs[self.turn_cursor].events[self.event_cursor:]:
                self._on_event(ev, self.turn_cursor)
            self.turn_cursor += 1
            self.event_cursor = 0
            self.turn_after_upkeep = False
        events = current.events
        for ev in events[self.event_cursor:]:
            self._on_event(ev, battle.turn)
        self.event_cursor = len(events)

    # ---- events --------------------------------------------------------------------------------
    def _on_event(self, ev, turn):
        self.n_events += 1
        if len(ev) < 2:
            return
        kind = ev[1]
        if kind == "upkeep":
            self.turn_after_upkeep = True
            return
        if kind == "turn":
            self.turn_after_upkeep = False
            return
        tags = _tags(ev)
        if kind in ("switch", "drag", "replace") and len(ev) > 3:
            key = _key(ev[2])
            m = self.mon(key)
            m.forme = _species_of(ev[3])
            if kind != "replace":
                m.switch_turn = turn
                m.after_upkeep = self.turn_after_upkeep
                m.last_move = m.last_move_turn = None
                m.moves_since_in = []
                m.sub_hit = False
                m.tox_ticks = 0
                m.taunt_start = m.encore_start = m.yawn_turn = None
                m.lock_move = m.lock_turn = None
                m.disabled_move = None
                m.prepare = None
                m.stint_moves = set()
                self.pressure[key[:2]] = False
                if kind == "switch" and not self.turn_after_upkeep:
                    self.acted[key[:2]] = turn
        elif kind == "detailschange" and len(ev) > 3:
            self.mon(_key(ev[2])).forme = _species_of(ev[3])
        elif kind == "-formechange" and len(ev) > 3:
            self.mon(_key(ev[2])).forme = to_id_str(ev[3])
        elif kind == "move" and len(ev) > 3:
            key = _key(ev[2])
            m = self.mon(key)
            mid = to_id_str(ev[3])
            m.prepare = None
            self.last_mover = key
            self.acted[key[:2]] = turn
            called = "from" in tags   # called by another move or a locked-in continuation
            if not called:
                m.pp_used[mid] += 1
                if self.pressure.get("p2" if key[:2] == "p1" else "p1"):
                    m.pp_used[mid] += 1
            if tags.get("from", "").startswith("move:"):
                pass  # Sleep Talk etc.: the caller is the Pokemon's last move
            else:
                m.last_move, m.last_move_turn = mid, turn
                m.moves_since_in.append((mid, self.n_events))
            if not called and mid not in ("struggle", "recharge") and not (m.traded or m.item_consumed):
                # while it still holds its original item: two different moves in one stint rule out
                # a Choice item, a status move an Assault Vest
                m.stint_moves.add(mid)
                if len(m.stint_moves) >= 2:
                    m.no_items.update(CHOICE_ITEMS)
                if MOVES.get(mid, {}).get("category") == "Status":
                    m.no_items.add("assaultvest")
            if mid in LOCKING_MOVES and not called:
                m.lock_move, m.lock_turn = mid, turn
            elif mid not in LOCKING_MOVES:
                m.lock_move = m.lock_turn = None
            if m.rest or m.sleep_attempts:
                pass
            if mid == "wish" and "still" not in tags:
                self.wish[key[:2]] = (turn, key)
        elif kind == "-prepare" and len(ev) > 3:
            self.mon(_key(ev[2])).prepare = (to_id_str(ev[3]), turn)
        elif kind == "-anim" and len(ev) > 3:
            # a charge move released in the same turn (Solar Beam in sun, Power Herb)
            m = self.mon(_key(ev[2]))
            if m.prepare is not None and m.prepare[0] == to_id_str(ev[3]):
                m.prepare = None
        elif kind == "cant" and len(ev) > 3:
            key = _key(ev[2])
            m = self.mon(key)
            self.acted[key[:2]] = turn
            if ev[3] == "slp":
                m.sleep_attempts += 1
            m.lock_move = m.lock_turn = None
            m.prepare = None
        elif kind == "-status" and len(ev) > 3:
            m = self.mon(_key(ev[2]))
            if ev[3] == "slp":
                m.rest = tags.get("from", "") == "move: Rest"
                m.sleep_attempts = 0
            elif ev[3] == "tox":
                m.tox_ticks = 0
        elif kind == "-curestatus" and len(ev) > 3:
            m = self.mon(_key(ev[2]))
            if ev[3] == "slp":
                m.rest = False
                m.sleep_attempts = 0
        elif kind == "-damage" and len(ev) > 3:
            key = _key(ev[2])
            frm = tags.get("from", "")
            if frm in ("Stealth Rock", "Spikes"):
                m = self.mon(key)
                if not (m.traded or m.item_consumed):
                    m.no_items.add("heavydutyboots")
            if frm == "psn":
                self.mon(key).tox_ticks += 1
            elif frm == "confusion":
                m = self.mon(key)
                m.lock_move = m.lock_turn = None  # hurting itself ends a locked move (Outrage)
            self._item_tag(key, tags)
        elif kind == "-heal" and len(ev) > 3:
            key = _key(ev[2])
            if tags.get("from") == "move: Wish":
                self.wish.pop(key[:2], None)
            self._item_tag(key, tags)
        elif kind == "-start" and len(ev) > 3:
            key = _key(ev[2])
            m = self.mon(key)
            eff = ev[3]
            if eff == "Substitute":
                m.sub_hit = False
            elif eff in ("move: Taunt", "Taunt"):
                m.taunt_start = self.n_events
            elif eff in ("Encore", "move: Encore"):
                m.encore_start = self.n_events
            elif eff in ("move: Yawn", "Yawn"):
                m.yawn_turn = turn
            elif eff in ("Disable", "move: Disable") and len(ev) > 4:
                m.disabled_move = to_id_str(ev[4])
            elif eff in ("move: Future Sight", "Future Sight", "move: Doom Desire", "Doom Desire"):
                self.future_sight[key[:2]] = (turn, key)
            elif eff == "confusion" and "fatigue" in tags:
                m.lock_move = m.lock_turn = None
            self._item_tag(key, tags)
        elif kind == "-end" and len(ev) > 3:
            key = _key(ev[2])
            eff = ev[3]
            m = self.mon(key)
            if eff in ("Disable", "move: Disable"):
                m.disabled_move = None
            elif eff in ("move: Future Sight", "Future Sight", "move: Doom Desire", "Doom Desire"):
                self.future_sight.pop(key[:2], None)
        elif kind == "-activate" and len(ev) > 3:
            key = _key(ev[2])
            eff = ev[3]
            if eff in ("move: Substitute", "Substitute") and "damage" in tags:
                self.mon(key).sub_hit = True
            elif eff in ("move: Trick", "move: Switcheroo") and "of" in tags:
                self.trick_pair = (key, _key(tags["of"]))
            self._item_tag(key, tags)
        elif kind == "-item" and len(ev) > 3:
            key = _key(ev[2])
            m = self.mon(key)
            item = to_id_str(ev[3])
            frm = tags.get("from", "")
            if frm in ("move: Trick", "move: Switcheroo"):
                # key received `item` from its trade partner: that was the partner's own item
                m.traded = True
                partner = None
                if self.trick_pair and key in self.trick_pair:
                    partner = self.trick_pair[1] if self.trick_pair[0] == key else self.trick_pair[0]
                if partner is not None:
                    p = self.mon(partner)
                    if p.orig_item is None and not p.traded:
                        p.orig_item = item
            elif not m.traded and m.orig_item is None:
                m.orig_item = item
        elif kind == "-enditem" and len(ev) > 3:
            key = _key(ev[2])
            m = self.mon(key)
            if not m.traded and m.orig_item is None:
                m.orig_item = to_id_str(ev[3])
            m.item_consumed = True
        elif kind == "-ability" and len(ev) > 3:
            key = _key(ev[2])
            if to_id_str(ev[3]) == "pressure":
                self.pressure[key[:2]] = True
            self._item_tag(key, tags)
        elif kind == "-weather" and len(ev) > 2:
            if ev[2] == "none":
                self.weather = None
            elif "upkeep" not in tags:
                frm = tags.get("from", "")
                if frm.startswith("ability:"):
                    setter, by_ability = _key(tags["of"]) if "of" in tags else None, True
                else:
                    setter, by_ability = self.last_mover, False
                self.weather = (turn, self.turn_after_upkeep, setter, by_ability)
        elif kind == "-fieldstart" and len(ev) > 2:
            field = Field.from_showdown_message(ev[2])
            if field in TERRAIN:
                setter = _key(tags["of"]) if "of" in tags else self.last_mover
                self.terrain = (turn, self.turn_after_upkeep, setter, field)
            elif field is Field.TRICK_ROOM:
                self.trick_room = (turn, self.turn_after_upkeep)
        elif kind == "-fieldend" and len(ev) > 2:
            field = Field.from_showdown_message(ev[2])
            if field in TERRAIN:
                self.terrain = None
            elif field is Field.TRICK_ROOM:
                self.trick_room = None
        elif kind == "-sidestart" and len(ev) > 3:
            cond = SideCondition.from_showdown_message(ev[3])
            self.side_setters[(ev[2][:2], cond)] = self.last_mover
        elif kind == "-terastallize" and len(ev) > 3:
            key = _key(ev[2])
            self.tera_user[key[:2]] = (key, to_id_str(ev[3]))
        elif kind == "faint" and len(ev) > 2:
            m = self.mon(_key(ev[2]))
            m.lock_move = m.lock_turn = None

    def _item_tag(self, key, tags):
        """`[from] item: X` on an event: the [of] Pokemon holds X if there is one (Rocky Helmet),
        otherwise the event's Pokemon (Life Orb, Leftovers)."""
        frm = tags.get("from", "")
        if frm.startswith("item:"):
            holder = _key(tags["of"]) if "of" in tags else key
            m = self.mon(holder)
            if not m.traded and m.orig_item is None:
                m.orig_item = to_id_str(frm[5:])


_TRACKERS = weakref.WeakKeyDictionary()


def tracker(battle):
    tr = _TRACKERS.get(battle)
    if tr is None:
        tr = _TRACKERS[battle] = LogTracker()
    tr.update(battle)
    return tr


# ---------------------------------------------------------------------------------------------------
def _types(species):
    t = [x.lower() for x in POKEDEX[species]["types"]]
    return (t[0], t[1] if len(t) > 1 else "typeless")


def _ptype(t):
    return "typeless" if t is None else t.name.lower()


def _max_pp(move_id):
    entry = MOVES.get(move_id)
    if not entry:
        return 16
    pp = entry.get("pp", 16)
    return pp if entry.get("noPPBoosts") else pp * 8 // 5


def _clamp_boost(v):
    return max(-6, min(6, int(v)))


def _species_for_engine(species):
    return species if species in POKEDEX else None


class _SideBuilder:
    """Engine Side for one player, built from that player's own battle object (exact) or from the
    other player's observations plus sampled sets."""

    def __init__(self, battle, tr, role):
        self.battle = battle
        self.tr = tr
        self.role = role          # 'p1' / 'p2': the side being built
        self.turn = battle.turn

    # ---- volatiles of the active -----------------------------------------------------------
    def active_volatiles(self, mon, key, moves, request_moves=None):
        """(volatile set, durations kwargs, substitute hp fraction of max hp, last_used_move index or
        'switch', locked: the move it must use, disabled move ids)."""
        tr = self.tr
        m = tr.mon(key)
        vols = set()
        dur = {}
        effects = mon.effects
        for e in effects:
            v = VOLATILES.get(e)
            if v:
                vols.add(v)
        if Effect.TAUNT in effects:
            # the engine counts the taunted Pokemon's moves since the taunt (0-2) and removes it
            # before the third
            n = sum(1 for _, i in m.moves_since_in if m.taunt_start is not None and i > m.taunt_start)
            vols.add("taunt")
            dur["taunt"] = min(n, 2)
        if Effect.YAWN in effects:
            vols.add("yawn")
            dur["yawn"] = 1 if m.yawn_turn is not None and m.yawn_turn < self.turn else 0
        if Effect.SLOW_START in effects and m.switch_turn is not None:
            passed = self.turn - m.switch_turn - (1 if m.after_upkeep or m.switch_turn == 0 else 0)
            vols.add("slowstart")
            dur["slowstart"] = max(1, 5 - passed)
        if mon.must_recharge:
            vols.add("mustrecharge")
        if m.prepare is not None and m.prepare[0] in CHARGE_MOVES and m.prepare[0] in moves and \
                (request_moves is None or request_moves == [m.prepare[0]]):
            vols.add(m.prepare[0])
        sub = 0.0
        if Effect.SUBSTITUTE in effects:
            vols.add("substitute")
            sub = 0.1 if m.sub_hit else 0.25
        locked = None
        if request_moves is not None:
            lock_ok = len(request_moves) == 1 and request_moves[0] == m.lock_move  # own side: the request says
        else:
            lock_ok = True
        if lock_ok and m.lock_move and m.lock_turn is not None and m.lock_move in moves and \
                0 < self.turn - m.lock_turn <= 2:
            locked = m.lock_move
            vols.add("lockedmove")
            dur["lockedmove"] = min(2, self.turn - m.lock_turn)
        last = "switch"
        if m.last_move is not None and m.last_move in moves:
            last = moves.index(m.last_move)
        if Effect.ENCORE in effects:
            if last != "switch":
                vols.add("encore")
                dur["encore"] = min(2, sum(1 for _, i in m.moves_since_in
                                           if m.encore_start is not None and i > m.encore_start))
        disabled = set()
        if m.disabled_move and Effect.DISABLE in effects:
            disabled.add(m.disabled_move)
        return vols, dur, sub, last, locked, disabled

    def boosts(self, mon):
        b = mon.boosts
        return dict(attack_boost=_clamp_boost(b.get("atk", 0)), defense_boost=_clamp_boost(b.get("def", 0)),
                    special_attack_boost=_clamp_boost(b.get("spa", 0)),
                    special_defense_boost=_clamp_boost(b.get("spd", 0)),
                    speed_boost=_clamp_boost(b.get("spe", 0)),
                    accuracy_boost=_clamp_boost(b.get("accuracy", 0)),
                    evasion_boost=_clamp_boost(b.get("evasion", 0)))

    def sleep(self, mon, key):
        """(rest_turns, sleep_turns) for a sleeping Pokemon."""
        if mon.status is not Status.SLP:
            return 0, 0
        m = self.tr.mon(key)
        if m.rest:
            return max(1, 3 - m.sleep_attempts), 0
        return 0, min(3, m.sleep_attempts)

    def pp(self, key, move_id):
        return max(0, _max_pp(move_id) - self.tr.mon(key).pp_used.get(move_id, 0))


def _side_conditions(battle, tr, conds, role, item_of):
    """SideConditions kwargs for one side. item_of(key) -> the item the Pokemon holds (or held when
    it set the condition), for Light Clay."""
    turn = battle.turn
    out = {}
    for cond, value in conds.items():
        if cond is SideCondition.SPIKES:
            out["spikes"] = min(3, value)
        elif cond is SideCondition.TOXIC_SPIKES:
            out["toxic_spikes"] = min(2, value)
        elif cond is SideCondition.STEALTH_ROCK:
            out["stealth_rock"] = 1
        elif cond is SideCondition.STICKY_WEB:
            out["sticky_web"] = 1
        elif cond in SCREENS:
            setter = tr.side_setters.get((role, cond))
            dur = 8 if setter and item_of(setter) == "lightclay" else 5
            out[SCREENS[cond]] = max(1, dur - (turn - value))
        elif cond in TIMED_SIDE:
            name, dur = TIMED_SIDE[cond]
            out[name] = max(1, dur - (turn - value))
    return out


def _move_list(ids):
    return [m for m in ids if m][:4]


def _remaining(start, duration, turn):
    """Turns left of an effect with `duration` set at (turn, after upkeep): it counts down at the end
    of every turn from the one it was set in, unless it was set after that turn's residual phase or
    before turn 1 (the leads' switch-in abilities)."""
    t0, after_upkeep = start[0], start[1]
    passed = turn - t0 - (1 if after_upkeep or t0 == 0 else 0)
    return max(1, duration - passed)


class StateBuilder:
    """Builds poke-engine States for one player's battle object."""

    def __init__(self, battle):
        self.battle = battle
        self.tr = tracker(battle)
        self.role = battle.player_role
        self.opp_role = "p2" if self.role == "p1" else "p1"

    # ---- own side -----------------------------------------------------------------------------
    def own_side(self, battle=None, tr=None):
        """(Side kwargs, keys in engine order, item_of) for the player of `battle`, from its request."""
        battle = battle or self.battle
        tr = tr or tracker(battle)
        role = battle.player_role
        sb = _SideBuilder(battle, tr, role)
        req = battle.last_request or {}
        side = req.get("side") or {}
        req_mons = {p["ident"]: p for p in side.get("pokemon", [])}
        active_req = (req.get("active") or [None])[0]
        team = team_list(battle)
        keys = []
        pokemon = []
        items = {}
        active_index = None
        active_moves = None
        for i, mon in enumerate(team):
            key = next((k for k, v in battle.team.items() if v is mon), None)
            keys.append(key)
            p = req_mons.get(key)
            if p is None:
                pokemon.append(pe.Pokemon.create_fainted())
                continue
            details = p.get("details", "")
            species = _species_of(details)
            forme = tr.mon(key).forme
            if forme in POKEDEX and to_id_str(POKEDEX[forme].get("baseSpecies", "")) == \
                    to_id_str(POKEDEX.get(species, {}).get("baseSpecies", "")):
                species = forme  # the log has forme changes the request's details lag (Terapagos-Stellar)
            if species not in POKEDEX:
                species = mon.species
            level = mon.level
            cond = p.get("condition", "0 fnt").split(" ")
            hp_part = cond[0]
            if "/" in hp_part:
                hp, maxhp = (int(x) for x in hp_part.split("/"))
            else:
                hp, maxhp = 0, max(1, mon.max_hp or 1)
            status = None if len(cond) < 2 or cond[1] == "fnt" else Status[cond[1].upper()]
            st = p.get("stats", {})
            item = to_id_str(p.get("item") or "") or "none"
            items[key] = item
            ability = to_id_str(p.get("ability") or p.get("baseAbility") or "") or "none"
            base_ability = to_id_str(p.get("baseAbility") or p.get("ability") or "") or ability
            move_ids = _move_list([to_id_str(x) for x in p.get("moves", [])])
            # the active keeps its flag when fainted (forceSwitch requests have no "active" part)
            is_active = bool(p.get("active")) and hp > 0
            pp = {m: sb.pp(key, m) for m in move_ids}
            disabled = set()
            if p.get("active"):
                active_index = i
                req_moves = active_req.get("moves", []) if active_req else []
                req_ids = [m.get("id") for m in req_moves]
                if active_req is None:
                    pass  # a forced switch: no moves to choose (the engine re-enables them on switch-out)
                elif req_ids == ["recharge"]:
                    pass  # recharge turn: the mustrecharge volatile handles it
                elif req_ids == ["struggle"]:
                    disabled |= set(move_ids)  # nothing usable: the engine has no Struggle (option "none")
                else:
                    shown = set()
                    for rm in req_moves:
                        mid = rm.get("id")
                        if mid in pp:
                            shown.add(mid)
                            if "pp" in rm:
                                pp[mid] = rm["pp"]
                            if rm.get("disabled"):
                                disabled.add(mid)
                    # a locked move (Outrage) is the only one listed; the rest can't be chosen
                    disabled |= {m for m in move_ids if m not in shown}
                active_moves = move_ids
            tera = p.get("teraType")
            terastallized = bool(p.get("terastallized"))
            types = _types(species)
            if is_active and mon._temporary_types and not terastallized:
                tt = [t.name.lower() for t in mon._temporary_types if t is not None]
                types = (tt[0], tt[1] if len(tt) > 1 else "typeless") if tt else types
            rest_turns, sleep_turns = sb.sleep(mon, key) if status is Status.SLP else (0, 0)
            pokemon.append(pe.Pokemon(
                id=species, level=level, types=types, base_types=_types(species), hp=hp, maxhp=maxhp,
                ability=ability, base_ability=base_ability, item=item,
                evs=(85, 85, 85, 85, 85, 85),
                attack=st.get("atk", 100), defense=st.get("def", 100), special_attack=st.get("spa", 100),
                special_defense=st.get("spd", 100), speed=st.get("spe", 100),
                status=STATUS.get(status, "none"), rest_turns=rest_turns, sleep_turns=sleep_turns,
                weight_kg=float(POKEDEX[species].get("weightkg", 1.0)),
                moves=[pe.Move(id=m, pp=max(0, min(pp[m], 127)), disabled=m in disabled) for m in move_ids],
                terastallized=terastallized,
                tera_type=(to_id_str(tera) if tera else "typeless")))
        kw = self._side_common(battle, tr, sb, keys, pokemon, active_index, active_moves, items.get,
                               own=True)
        return kw, keys, items.get

    # ---- opponent side from observations ------------------------------------------------------
    def observed_side(self, sets):
        """(Side kwargs, keys, item_of) for the opponent of self.battle: revealed Pokemon from the log
        plus `sets` (RandSet per opponent_team entry in order, then the unrevealed ones)."""
        battle = self.battle
        tr = self.tr
        sb = _SideBuilder(battle, tr, self.opp_role)
        opp = list(battle.opponent_team.items())[:6]
        keys, pokemon, items = [], [], {}
        active = battle.opponent_active_pokemon
        active_index = None
        active_moves = None
        for i, s in enumerate(sets[:6]):
            if i < len(opp):
                key, mon = opp[i]
                m = tr.mon(key)
                species = m.forme if m.forme in POKEDEX else mon.species
                if species not in POKEDEX:
                    species = s.key if s.key in POKEDEX else mon.species
                stats = s.stats(species)
                maxhp = stats["hp"]
                if mon.fainted:
                    hp = 0
                else:
                    # the opponent's HP is shown as ceil(100 * hp / maxhp)% (100 only when full): use
                    # the middle of the range it stands for
                    frac = mon.current_hp_fraction
                    if 0 < frac < 1:
                        frac = (round(frac * 100) - 0.5) / 100
                    hp = min(maxhp, max(1, round(frac * maxhp)))
                    # a benched Pokemon's HP was last shown before it switched out: Regenerator has
                    # healed a third since
                    if mon is not active and (mon.ability or s.ability) == "regenerator":
                        hp = min(maxhp, hp + maxhp // 3)
                revealed = [mv for mv in mon.moves if mv not in ("struggle", "recharge")]
                move_ids = list(dict.fromkeys(revealed))[:4]
                for mv in s.moves:
                    if len(move_ids) >= 4:
                        break
                    if mv not in move_ids:
                        move_ids.append(mv)
                if mon.item == GEN.UNKNOWN_ITEM:
                    item = s.item or "none"
                else:
                    item = mon.item or "none"
                items[key] = item if mon.item != GEN.UNKNOWN_ITEM else (s.item or "none")
                ability = mon.ability or s.ability
                tera = _ptype(mon.tera_type) if mon.is_terastallized and mon.tera_type else s.tera
                is_active = mon is active and not mon.fainted
                if mon is active:
                    active_index = i
                    active_moves = move_ids
                types = _types(species)
                if is_active and mon._temporary_types and not mon.is_terastallized:
                    tt = [t.name.lower() for t in mon._temporary_types if t is not None]
                    types = (tt[0], tt[1] if len(tt) > 1 else "typeless") if tt else types
                status = None if mon.fainted else mon.status
                rest_turns, sleep_turns = sb.sleep(mon, key) if status is Status.SLP else (0, 0)
                pokemon.append(pe.Pokemon(
                    id=species, level=mon.level, types=types, base_types=_types(species), hp=hp,
                    maxhp=maxhp, ability=ability, base_ability=s.ability or ability, item=item,
                    evs=tuple(s.evs), attack=stats["atk"], defense=stats["def"],
                    special_attack=stats["spa"], special_defense=stats["spd"], speed=stats["spe"],
                    status=STATUS.get(status, "none"), rest_turns=rest_turns, sleep_turns=sleep_turns,
                    weight_kg=float(POKEDEX[species].get("weightkg", 1.0)),
                    moves=[pe.Move(id=mv, pp=min(sb.pp(key, mv), 127)) for mv in move_ids],
                    terastallized=bool(mon.is_terastallized), tera_type=tera or "typeless"))
                keys.append(key)
            else:
                species = s.key
                stats = s.stats(species)
                key = f"{self.opp_role}: ?{i}"
                items[key] = s.item or "none"
                pokemon.append(pe.Pokemon(
                    id=species, level=s.level, types=_types(species), base_types=_types(species),
                    hp=stats["hp"], maxhp=stats["hp"], ability=s.ability, base_ability=s.ability,
                    item=s.item or "none", evs=tuple(s.evs), attack=stats["atk"], defense=stats["def"],
                    special_attack=stats["spa"], special_defense=stats["spd"], speed=stats["spe"],
                    weight_kg=float(POKEDEX[species].get("weightkg", 1.0)),
                    moves=[pe.Move(id=mv, pp=_max_pp(mv)) for mv in s.moves], tera_type=s.tera))
                keys.append(key)
        kw = self._side_common(battle, tr, sb, keys, pokemon, active_index, active_moves, items.get,
                               own=False)
        return kw, keys, items.get

    # ---- shared -------------------------------------------------------------------------------
    def _side_common(self, battle, tr, sb, keys, pokemon, active_index, active_moves, item_of, own):
        """Side kwargs given the Pokemon list: active, volatiles, boosts, side conditions, wish,
        future sight, flags. `battle` is the battle object the side is read from: its own battle for
        own=True, the observer's for own=False."""
        role = sb.role
        tera_user = tr.tera_user.get(role)
        if tera_user is not None and tera_user[0] in keys and not any(p.terastallized for p in pokemon):
            # requests clear "terastallized" when the Pokemon faints; the side still can't tera
            i = keys.index(tera_user[0])
            p = pokemon[i]
            pokemon[i] = _with_moves(p, [pe.Move(id=m.id.lower(), pp=m.pp, disabled=m.disabled)
                                         for m in p.moves if m.id.lower() != "none"],
                                     terastallized=True, tera_type=tera_user[1])
        if active_index is None:
            active_index = 0
            for i, p in enumerate(pokemon):
                if p.hp > 0:
                    active_index = i
                    break
        if own:
            mon = battle.active_pokemon
            conds = battle.side_conditions
        else:
            mon = battle.opponent_active_pokemon
            conds = battle.opponent_side_conditions
        kw = dict(pokemon=pokemon, active_index=str(active_index))
        key = keys[active_index] if active_index < len(keys) else None
        if mon is not None and key is not None and not mon.fainted and active_moves is not None:
            request_moves = None
            if own:
                act = ((battle.last_request or {}).get("active") or [None])[0]
                request_moves = [x.get("id") for x in act.get("moves", [])] if act else []
            vols, dur, sub, last, locked, disabled = sb.active_volatiles(mon, key, active_moves, request_moves)
            if disabled:
                p = pokemon[active_index]
                pokemon[active_index] = _with_moves(p, [pe.Move(id=m.id.lower(), pp=m.pp,
                                                                disabled=m.disabled or m.id.lower() in disabled)
                                                        for m in p.moves if m.id.lower() != "none"])
            kw["volatile_statuses"] = vols
            kw["volatile_status_durations"] = pe.VolatileStatusDurations(**dur)
            kw["substitute_health"] = int(sub * pokemon[active_index].maxhp) if sub else 0
            kw.update(sb.boosts(mon))
            kw["last_used_move"] = f"switch:{active_index}" if last == "switch" else f"move:{last}"
            m = tr.mon(key)
            sc = _side_conditions(battle, tr, conds, role, item_of)
            if mon.status is Status.TOX:
                sc["toxic_count"] = min(m.tox_ticks, 15)
            if mon.protect_counter:
                sc["protect"] = min(mon.protect_counter, 6)
        else:
            sc = _side_conditions(battle, tr, conds, role, item_of)
        kw["side_conditions"] = pe.SideConditions(**sc)
        # wish and future sight on this side
        turn = battle.turn
        w = tr.wish.get(role)
        if w is not None:
            left = 2 - (turn - w[0])
            if left > 0 and w[1] in keys:
                kw["wish"] = (left, pokemon[keys.index(w[1])].maxhp // 2)
        fs = tr.future_sight.get(role)
        if fs is not None:
            left = 3 - (turn - fs[0])
            if left > 0:
                kw["future_sight"] = (left, str(keys.index(fs[1]) if fs[1] in keys else active_index))
        return kw

    # ---- field ----------------------------------------------------------------------------------
    def field(self, item_of):
        """State kwargs for weather, terrain and Trick Room. item_of(key) -> item of a Pokemon on
        either side (the setter's rock / Terrain Extender)."""
        battle, tr, turn = self.battle, self.tr, self.battle.turn
        kw = {}
        if battle.weather:
            w = WEATHER.get(next(iter(battle.weather)))
            if w:
                kw["weather"] = w
                if w in PRIMAL_WEATHER:
                    kw["weather_turns_remaining"] = 0  # lasts while its user is in
                elif tr.weather is not None:
                    setter = tr.weather[2]
                    dur = 8 if setter and item_of(setter) == WEATHER_ROCK.get(w) else 5
                    kw["weather_turns_remaining"] = _remaining(tr.weather, dur, turn)
                else:
                    kw["weather_turns_remaining"] = 5
        terrain = [f for f in battle.fields if f in TERRAIN]
        if terrain:
            t = terrain[0]
            kw["terrain"] = TERRAIN[t]
            if tr.terrain is not None and tr.terrain[3] is t:
                dur = 8 if tr.terrain[2] and item_of(tr.terrain[2]) == "terrainextender" else 5
                kw["terrain_turns_remaining"] = _remaining(tr.terrain, dur, turn)
            else:
                kw["terrain_turns_remaining"] = max(1, 5 - (turn - battle.fields[t]))
        if Field.TRICK_ROOM in battle.fields:
            kw["trick_room"] = True
            start = tr.trick_room or (battle.fields[Field.TRICK_ROOM], False)
            kw["trick_room_turns_remaining"] = _remaining(start, 5, turn)
        return kw

    # ---- request flags --------------------------------------------------------------------------
    @staticmethod
    def request_flags(battle, tr, side_kw, other_kw, other_role):
        """Flags from the player's request: forced switch (a pivot or Eject Button, not a faint),
        trapped, Baton Pass / Shed Tail; slow_uturn_move on the other side when its move is still
        to come this turn."""
        req = battle.last_request or {}
        active = battle.active_pokemon
        fs = req.get("forceSwitch")
        if fs and fs[0] and active is not None and not active.fainted:
            side_kw["force_switch"] = True
            last = tr.mon(_key_of(battle.team, active)).last_move if _key_of(battle.team, active) else None
            if last == "batonpass":
                side_kw["baton_passing"] = True
            elif last == "shedtail":
                side_kw["shed_tailing"] = True
            if tr.acted.get(other_role) != battle.turn:
                other_kw["slow_uturn_move"] = True
        act = (req.get("active") or [None])[0]
        if act and act.get("trapped"):
            side_kw["force_trapped"] = True


def _key_of(team, mon):
    for k, v in team.items():
        if v is mon:
            return k
    return None


def _with_moves(p, moves, **changes):
    """A copy of engine Pokemon p with other moves (poke-engine's Python objects are read-only)."""
    kw = dict(id=p.id.lower(), level=p.level, types=tuple(t.lower() for t in p.types),
              base_types=tuple(t.lower() for t in p.base_types), hp=p.hp, maxhp=p.maxhp,
              ability=p.ability.lower(), base_ability=p.base_ability.lower(), item=p.item.lower(),
              nature=p.nature.lower(), evs=p.evs, attack=p.attack, defense=p.defense,
              special_attack=p.special_attack, special_defense=p.special_defense, speed=p.speed,
              status=p.status.lower(), rest_turns=p.rest_turns, sleep_turns=p.sleep_turns,
              weight_kg=p.weight_kg, moves=moves, terastallized=p.terastallized,
              tera_type=p.tera_type.lower())
    kw.update(changes)
    return pe.Pokemon(**kw)


def build_state(battle, opp_sets):
    """(State, info) from battle's player's point of view, the opponent's unknowns from opp_sets.
    info: own and opponent keys in engine order."""
    b = StateBuilder(battle)
    s1, keys1, item1 = b.own_side()
    s2, keys2, item2 = b.observed_side(opp_sets)

    def item_of(key):
        return item1(key) or item2(key)

    StateBuilder.request_flags(battle, b.tr, s1, s2, b.opp_role)
    state = pe.State(side_one=pe.Side(**s1), side_two=pe.Side(**s2), **b.field(item_of))
    return state, {"keys1": keys1, "keys2": keys2}


def build_true_state(battle, other):
    """The true state: side one from battle's request, side two from the other player's (other is
    the other side's battle object of the same game). Field from battle's log, with both sides'
    true items."""
    b = StateBuilder(battle)
    s1, keys1, item1 = b.own_side()
    s2, keys2, item2 = b.own_side(other, tracker(other))

    def item_of(key):
        return item1(key) or item2(key)

    StateBuilder.request_flags(battle, b.tr, s1, s2, b.opp_role)
    StateBuilder.request_flags(other, tracker(other), s2, s1, b.role)
    state = pe.State(side_one=pe.Side(**s1), side_two=pe.Side(**s2), **b.field(item_of))
    return state, {"keys1": keys1, "keys2": keys2}


# ---------------------------------------------------------------------------------------------------
def option_to_action(option, battle, state):
    """poke-engine option string (side one) -> v20 action, or -1."""
    if option.startswith("switch "):
        target = option[7:]
        for k, p in enumerate(state.side_one.pokemon):
            if p.id.lower() == target and k < len(team_list(battle)):
                return SWITCH0 + k
        return -1
    tera = option.endswith("-tera")
    mid = option[:-5] if tera else option
    for i, mv in enumerate(battle.available_moves[:4]):
        if mv.id == mid:
            return (TERA0 if tera else MOVE0) + i
    return -1


def order_to_option(order, battle, state, side="one"):
    """BattleOrder (as sent for `battle`) -> poke-engine option string for that player's side in
    `state`, or None."""
    from poke_env.battle.move import Move
    from poke_env.battle.pokemon import Pokemon
    o = getattr(order, "order", None)
    if isinstance(o, Move):
        return o.id + ("-tera" if getattr(order, "terastallize", False) else "")
    if isinstance(o, Pokemon):
        sd = state.side_one if side == "one" else state.side_two
        team = team_list(battle)
        for k, mon in enumerate(team):
            if mon is o and k < 6:
                return "switch " + sd.pokemon[k].id.lower()
    return None
