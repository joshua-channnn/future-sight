"""v21 environment: v20's observation plus battle-history tokens, and labels for auxiliary heads.

Observation, from `Featurizer.featurize(battle)` -> (idx int64[N_IDX], flt float32[N_FLT], mask bool[14]):
  - the first v20.N_IDX / v20.N_FLT entries are exactly v20's (12 Pokemon, 4 move, 1 field token),
  - then H history tokens, most recent first: the last H moves, switches, drags and "can't move"
    events of either side, each with what followed it (damage dealt as a fraction of HP,
    effectiveness, crit, miss, protection, status, KOs, recoil, drain, stat changes, hazard damage).
    Empty slots are all zero (`present` 0).
Actions and masks are v20's.

Labels (training only; `set_labels`, `next_action_label`) come from the other side's battle object,
which both sides of an RLEnvV20 game have: the opponent's item, base ability, tera type and moves
for each revealed opponent token, and the opponent's action at the same decision. -1 means no label.
"""
import functools

import numpy as np

from poke_env.battle import PokemonType
from poke_env.data import to_id_str
from poke_env.player import BattleOrder

from envs import rl_player_v20 as v20
from envs.rl_player_v20 import (ABILITY_OTHER, ABILITY_VOCAB, GEN_DATA, ITEM_NONE, ITEM_OTHER,
                                ITEM_VOCAB, LAYOUT as V20_LAYOUT, N_MOVE_IDX, SETS_PATH,
                                SPECIES_OTHER, SPECIES_VOCAB, TYPE_IDX, _Tracker, move_index,
                                order_to_action, team_list)

DEFAULT_HISTORY = 16

# ---------------------------------------------------------------------------------------------
# History tokens
HS_IDX_NAMES = ["actor", "move", "other"]  # species; move used (or not, for cant); target or the
#                                            Pokemon switched out
HS_KINDS = ("move", "switch", "drag", "cant")
HS_FLT_NAMES = [
    "present", "own", "k_move", "k_switch", "k_drag", "k_cant", "turns_ago", "this_turn",
    "tera", "called", "replacement",
    "dmg", "target_hp", "super", "resisted", "immune", "crit", "miss", "fail", "protected",
    "hit_sub", "status", "target_ko",
    "self_dmg", "self_heal", "actor_hp", "actor_ko", "boost_self", "boost_target", "hazard_dmg",
    "n_hits",
]
N_HS_IDX, N_HS_FLT = len(HS_IDX_NAMES), len(HS_FLT_NAMES)
HS_FLT_POS = {n: i for i, n in enumerate(HS_FLT_NAMES)}
_P = HS_FLT_POS
PROTECT_MOVES = {"Protect", "Detect", "King's Shield", "Spiky Shield", "Baneful Bunker",
                 "Silk Trap", "Burning Bulwark", "Obstruct", "Max Guard"}

# ---------------------------------------------------------------------------------------------
# Labels: one int16 row per sample. Opponent tokens j = 0..5 (v20's first-seen order).
LAB_ITEM, LAB_ABILITY, LAB_TERA, LAB_MOVES = 0, 6, 12, 18   # moves: 6 x 4 move indices
LAB_NEXT = 42
N_LAB = 43
# Next-action classes, in the observer's frame: a move by vocab index, the same with tera, a switch
# to opponent token j; NEXT_UNREVEALED (a switch to a Pokemon not yet seen) is a label code, not a
# logit: it stands for every unrevealed placeholder token
NEXT_MOVE0, NEXT_TERA0, NEXT_SWITCH0 = 0, N_MOVE_IDX, 2 * N_MOVE_IDX
N_NEXT = NEXT_SWITCH0 + 6
NEXT_UNREVEALED = N_NEXT


def make_layout(history_len=DEFAULT_HISTORY):
    """v21 observation layout: v20's plus the history block. Checkpoints store it."""
    layout = dict(V20_LAYOUT)
    layout.update({
        "arch": "v21",
        "v20_n_idx": V20_LAYOUT["n_idx"], "v20_n_flt": V20_LAYOUT["n_flt"],
        "hist_len": history_len, "n_hs_idx": N_HS_IDX, "n_hs_flt": N_HS_FLT,
        "hs_idx_names": list(HS_IDX_NAMES), "hs_flt_names": list(HS_FLT_NAMES),
        "n_idx": V20_LAYOUT["n_idx"] + history_len * N_HS_IDX,
        "n_flt": V20_LAYOUT["n_flt"] + history_len * N_HS_FLT,
        "pk_placeholder": v20.PK_FLT_POS["is_placeholder"], "pk_is_own": v20.PK_FLT_POS["is_own"],
        "mv_exists": v20.MV_FLT_POS["exists"], "hs_present": HS_FLT_POS["present"],
        "n_lab": N_LAB, "n_next": N_NEXT,
    })
    return layout


@functools.lru_cache(maxsize=None)
def species_index_from_details(details):
    """Species index from a protocol details string ("Landorus-Therian, L76, M"), with v20's
    fallback to the base species for formes outside the vocab."""
    sid = to_id_str(details.split(",")[0])
    idx = SPECIES_VOCAB.get(sid)
    if not idx:
        base = (GEN_DATA.pokedex.get(sid) or {}).get("baseSpecies")
        idx = SPECIES_VOCAB.get(to_id_str(base)) if base else None
    return idx or SPECIES_OTHER


def _hp(status):
    """HP fraction from a protocol HP string: "127/307", "42/100 par", "0 fnt"."""
    s = status.split(" ", 1)[0]
    if "/" in s:
        a, _, b = s.partition("/")
        try:
            return min(max(float(a) / max(float(b), 1.0), 0.0), 1.0)
        except ValueError:
            return None
    return 0.0 if s == "0" else None


class _Event:
    __slots__ = ("turn", "idx", "flt", "actor", "target")

    def __init__(self, turn, actor):
        self.turn = turn
        self.idx = [0] * N_HS_IDX
        self.flt = [0.0] * N_HS_FLT
        self.actor = actor
        self.target = None


class _HistTracker(_Tracker):
    """v20's tracker plus the history: one _Event per move / switch / drag / cant, with the outcome
    folded in from the messages that follow it until the next action, the residual phase or the
    next turn. Keeps each Pokemon's last HP fraction itself (battle objects only hold the latest)."""
    __slots__ = ("role", "events", "open", "hp", "species", "active", "fainted", "pending_tera",
                 "max_len")

    def __init__(self, max_len=64):
        super().__init__()
        self.role = None
        self.events = []
        self.open = None
        self.hp = {}
        self.species = {}
        self.active = {}
        self.fainted = set()
        self.pending_tera = set()
        self.max_len = max_len

    def update(self, battle):
        if self.role is None:
            self.role = battle.player_role
        super().update(battle)

    def _new(self, kind, actor, turn):
        ev = _Event(turn, actor)
        f = ev.flt
        f[_P["present"]] = 1.0
        f[_P["own"]] = 1.0 if actor[:2] == self.role else 0.0
        f[_P["k_" + kind]] = 1.0
        ev.idx[0] = self.species.get(actor, 0)
        self.events.append(ev)
        if len(self.events) > self.max_len:
            del self.events[:len(self.events) - self.max_len // 2]
        self.open = ev
        return ev

    def _on_event(self, ev, turn):
        super()._on_event(ev, turn)
        n = len(ev)
        kind = ev[1] if n > 1 else ""
        if kind == "move" and n > 3:
            actor = self._key(ev[2])
            e = self._new("move", actor, turn)
            e.idx[1] = move_index(to_id_str(ev[3]))
            if n > 4 and ev[4]:
                tgt = self._key(ev[4])
                if tgt != actor:
                    e.target = tgt
                    e.idx[2] = self.species.get(tgt, 0)
                    e.flt[_P["target_hp"]] = self.hp.get(tgt, 1.0)
            for tag in ev[5:]:
                if tag.startswith("[from]"):
                    e.flt[_P["called"]] = 1.0  # Sleep Talk, Magic Bounce, a locked-in Outrage, ...
                elif tag == "[miss]":
                    e.flt[_P["miss"]] = 1.0
                elif tag == "[notarget]":
                    e.flt[_P["fail"]] = 1.0
            if actor in self.pending_tera:
                self.pending_tera.discard(actor)
                e.flt[_P["tera"]] = 1.0
            e.flt[_P["actor_hp"]] = self.hp.get(actor, 1.0)
        elif kind in ("switch", "drag") and n > 3:
            key = self._key(ev[2].split(",")[0])
            side = key[:2]
            self.species[key] = species_index_from_details(ev[3])
            hp = _hp(ev[4]) if n > 4 else None
            self.hp[key] = 1.0 if hp is None else hp
            self.fainted.discard(key)
            prev = self.active.get(side)
            e = self._new(kind, key, turn)
            if prev is not None and prev != key:
                e.idx[2] = self.species.get(prev, 0)
                if prev in self.fainted:
                    e.flt[_P["replacement"]] = 1.0
            e.flt[_P["actor_hp"]] = self.hp[key]
            self.active[side] = key
        elif kind == "cant" and n > 2:
            e = self._new("cant", self._key(ev[2]), turn)
            if n > 4 and ev[4]:
                e.idx[1] = move_index(to_id_str(ev[4]))
            e.flt[_P["actor_hp"]] = self.hp.get(e.actor, 1.0)
        elif kind in ("-damage", "-heal", "-sethp") and n > 3:
            self._hp_change(kind, ev)
        elif kind == "faint" and n > 2:
            key = self._key(ev[2])
            self.hp[key] = 0.0
            self.fainted.add(key)
            e = self.open
            if e is not None:
                e.flt[_P["actor_ko" if key == e.actor else "target_ko"]] = 1.0
                e.flt[_P["actor_hp" if key == e.actor else "target_hp"]] = 0.0
        elif kind in ("", "upkeep", "turn"):
            # a new action group, the residual phase or the next turn: what follows is not the
            # last action's outcome
            self.open = None
            if kind == "turn":
                self.pending_tera.clear()
        elif kind == "-terastallize" and n > 2:
            self.pending_tera.add(self._key(ev[2]))
        elif kind == "detailschange" and n > 3:
            self.species[self._key(ev[2])] = species_index_from_details(ev[3])
        elif kind == "replace" and n > 3:
            # Illusion ends: the active Pokemon was someone else
            key = self._key(ev[2])
            side = key[:2]
            prev = self.active.get(side)
            self.species[key] = species_index_from_details(ev[3])
            self.hp[key] = self.hp.get(prev, 1.0)
            self.active[side] = key
        elif self.open is not None:
            self._outcome(kind, ev, n)

    def _hp_change(self, kind, ev):
        key = self._key(ev[2])
        new = _hp(ev[3])
        if new is None:
            return
        old = self.hp.get(key, 1.0)
        self.hp[key] = new
        if new > 0:
            self.fainted.discard(key)  # Revival Blessing
        e = self.open
        if e is None:
            return
        f = e.flt
        from_tag = next((x for x in ev[4:] if x.startswith("[from]")), None)
        if key == e.actor:
            if kind == "-damage":
                if f[_P["k_switch"]] or f[_P["k_drag"]]:
                    f[_P["hazard_dmg"]] = min(f[_P["hazard_dmg"]] + max(old - new, 0.0), 1.0)
                else:
                    # recoil, Life Orb, Rocky Helmet, Rough Skin, crash damage, Substitute's cost
                    f[_P["self_dmg"]] = min(f[_P["self_dmg"]] + max(old - new, 0.0), 1.0)
            elif kind == "-heal":
                f[_P["self_heal"]] = min(f[_P["self_heal"]] + max(new - old, 0.0), 1.0)
            f[_P["actor_hp"]] = new
            return
        if kind == "-damage":
            if from_tag is not None:
                return  # Leech Seed, partial trapping, ...: not this action's damage
            f[_P["dmg"]] = min(f[_P["dmg"]] + max(old - new, 0.0), 1.0)
            if e.target is None:
                e.target = key
                e.idx[2] = self.species.get(key, 0)
        elif kind == "-heal" and from_tag is not None and "ability:" in from_tag:
            f[_P["immune"]] = 1.0  # absorbed: Water Absorb, Volt Absorb, ...
        if key == e.target:
            f[_P["target_hp"]] = new

    def _outcome(self, kind, ev, n):
        e = self.open
        f = e.flt
        if kind == "-supereffective":
            f[_P["super"]] = 1.0
        elif kind == "-resisted":
            f[_P["resisted"]] = 1.0
        elif kind == "-immune":
            f[_P["immune"]] = 1.0
        elif kind == "-crit":
            f[_P["crit"]] = 1.0
        elif kind == "-miss":
            f[_P["miss"]] = 1.0
        elif kind in ("-fail", "-notarget"):
            f[_P["fail"]] = 1.0
        elif kind == "-activate" and n > 3:
            what = ev[3]
            if what.startswith("move: ") and what[6:] in PROTECT_MOVES:
                f[_P["protected"]] = 1.0
            elif "Substitute" in what and "[damage]" in ev[4:]:
                f[_P["hit_sub"]] = 1.0
        elif kind == "-end" and n > 3 and "Substitute" in ev[3]:
            f[_P["hit_sub"]] = 1.0
        elif kind == "-status" and n > 2:
            if self._key(ev[2]) != e.actor or not f[_P["k_move"]]:
                f[_P["status"]] = 1.0  # inflicted on the target; for a switch-in: Toxic Spikes
        elif kind in ("-boost", "-unboost") and n > 3:
            try:
                amount = int(ev[4]) if n > 4 else 1
            except ValueError:
                amount = 1
            if kind == "-unboost":
                amount = -amount
            pos = _P["boost_self"] if self._key(ev[2]) == e.actor else _P["boost_target"]
            f[pos] = max(min(f[pos] + amount / 6.0, 1.0), -1.0)
        elif kind == "-hitcount" and n > 3:
            try:
                f[_P["n_hits"]] = min(int(ev[3]) / 5.0, 1.0)
            except ValueError:
                pass


class Featurizer(v20.Featurizer):
    """v20's observation plus the last `history_len` battle events."""

    def __init__(self, history_len=DEFAULT_HISTORY, sets_path=SETS_PATH):
        super().__init__(sets_path)
        self.history_len = history_len
        self.layout = make_layout(history_len)

    def tracker(self, battle):
        tr = self._trackers.get(battle)
        if tr is None:
            tr = _HistTracker(max_len=4 * self.history_len)
            self._trackers[battle] = tr
        tr.update(battle)
        return tr

    def history(self, battle):
        """(idx int64[H * N_HS_IDX], flt float32[H * N_HS_FLT]) for the last H events, most
        recent first. Reads the tracker as featurize left it (call after featurize)."""
        tr = self._trackers.get(battle) or self.tracker(battle)
        H = self.history_len
        hidx = np.zeros((H, N_HS_IDX), dtype=np.int64)
        hflt = np.zeros((H, N_HS_FLT), dtype=np.float32)
        turn = battle.turn
        events = tr.events[-H:]
        n = len(events)
        if n:
            hidx[:n] = [e.idx for e in reversed(events)]
            hflt[:n] = [e.flt for e in reversed(events)]
            ago = np.asarray([turn - e.turn for e in reversed(events)], dtype=np.float32)
            hflt[:n, _P["turns_ago"]] = np.minimum(ago / 10.0, 1.0)
            hflt[:n, _P["this_turn"]] = ago <= 0
        return hidx.ravel(), hflt.ravel()

    def featurize(self, battle):
        idx, flt, mask = super().featurize(battle)
        hidx, hflt = self.history(battle)
        return np.concatenate([idx, hidx]), np.concatenate([flt, hflt]), mask


# ---------------------------------------------------------------------------------------------
# Labels
def illusion_game(battle1, battle2):
    """Zoroark disguises itself, which breaks the mapping between the two sides' views (gotcha
    15): such games get no labels."""
    return any(m.ability == "illusion" or m.base_species in ("zoroark", "zorua")
               for b in (battle1, battle2) for m in b.team.values())


def _request_side(battle):
    req = battle.last_request
    if req and req.get("side"):
        return {p["ident"]: p for p in req["side"]["pokemon"]}
    return {}


def set_labels(battle, truth):
    """Labels for `battle`'s opponent tokens (int16[N_LAB], next action left at -1), from `truth`,
    the opponent's own battle object and request: current item, base ability (what Trace or As One
    started as), tera type, the 4 moves. Unrevealed (placeholder) tokens get none, and neither do
    tokens whose species the two sides report differently (forme changes); Transform users get no
    move labels."""
    out = np.full(N_LAB, -1, dtype=np.int16)
    req = _request_side(truth)
    for j, (key, mon) in enumerate(list(battle.opponent_team.items())[:6]):
        t = truth.team.get(key)
        p = req.get(key)
        if t is None or p is None or mon.species != t.species:
            continue
        item = t.item
        out[LAB_ITEM + j] = ITEM_NONE if not item else (ITEM_VOCAB.get(item) or ITEM_OTHER)
        ability = to_id_str(p.get("baseAbility") or "")
        if ability:
            out[LAB_ABILITY + j] = ABILITY_VOCAB.get(ability) or ABILITY_OTHER
        tera = p.get("teraType")
        if tera:
            try:
                out[LAB_TERA + j] = TYPE_IDX[PokemonType.from_name(tera)]
            except KeyError:
                pass
        moves = p.get("moves") or []
        if not ("transform" in moves or ability == "imposter" or t.species == "ditto"):
            for i, m in enumerate(moves[:4]):
                out[LAB_MOVES + 4 * j + i] = move_index(m)
    return out


def next_action_label(battle, other, action):
    """The opponent's choice at this decision in `battle`'s frame (see NEXT_*), or -1. `other` is
    the opponent's battle object before the step; `action` its choice there (an int action or a
    BattleOrder)."""
    if isinstance(action, BattleOrder):
        action = order_to_action(action, other)
    a = int(action)
    if a < 0 or other.reviving:
        return -1
    if a < 8:
        moves = other.available_moves
        i = a % 4
        if i >= len(moves) or other.force_switch:
            return -1
        return (NEXT_TERA0 if a >= 4 else NEXT_MOVE0) + move_index(moves[i].id)
    team = team_list(other)
    k = a - 8
    if k >= len(team):
        return -1
    mon = team[k]
    key = next((kk for kk, m in other.team.items() if m is mon), None)
    keys = list(battle.opponent_team.keys())[:6]
    return NEXT_SWITCH0 + keys.index(key) if key in keys else NEXT_UNREVEALED
