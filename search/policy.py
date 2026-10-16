"""SearchPolicy: poke-engine MCTS over sampled opponent teams (agent_docs/step6_brief.md, Phase 1).

Foul Play's recipe in our own code: sample N complete opponent teams (search/sampling.py), build a
poke-engine State for each (search/state.py), run MCTS on each for a fixed time, add up each option's
share of the root visits over the samples, and pick at random among the options within 75% of the
best share, weighted by share. Default budget as Foul Play's defaults: 2 samples x 100 ms, or 4 x 50 ms
while at most 3 opponents are revealed and its active hasn't shown a move.

Same interface as players.v20_player.V20Policy (action(battle) -> int in the 14-action space,
order(battle) -> BattleOrder), so eval/v20_eval.play_games and the other eval scripts take it as is.
SearchPlayer wraps it as a poke-env Player for live play (e.g. against Foul Play).
"""
import asyncio
import functools
import math
import os
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np
import poke_engine as pe
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.data import GenData
from poke_env.player import DefaultBattleOrder, Player

from envs.rl_player_v20 import action_mask, action_to_order, hold_requests_until_log
from search.leaf_eval import evaluator_id, mcts as leaf_mcts
from search.sampling import get_prior, observations, sample_opponent
from search.state import build_state, option_to_action, tracker


def _skip_unknown_messages():
    """poke-env raises NotImplementedError on a message type it doesn't know, which abandons the rest
    of the server's message (it can hold the turn's log). The public server sends types the fork
    doesn't: |badge| since its ladder seasons (October 2026). This ignores |badge| and skips any
    other unknown type with a warning, once per type (agent_docs/poke_env_gotchas.md, item 16)."""
    AbstractBattle.MESSAGES_TO_IGNORE.add("badge")
    parse = AbstractBattle.parse_message
    if getattr(parse, "skips_unknown", False):
        return
    warned = set()

    @functools.wraps(parse)
    def parse_message(self, split_message):
        try:
            return parse(self, split_message)
        except NotImplementedError as e:
            if not e.args or e.args[0] != split_message:
                raise
            if split_message[1] not in warned:
                warned.add(split_message[1])
                self.logger.warning("skipping |%s| messages, which poke-env doesn't know", split_message[1])

    parse_message.skips_unknown = True
    AbstractBattle.parse_message = parse_message


_skip_unknown_messages()


class SearchPolicy:
    def __init__(self, samples=2, ms=100, early_samples=4, early_ms=50, threads=1, near_best=0.75,
                 deterministic=False, seed=None, iterations=0, iters_per_ms=0.0, net=None, prior_weight=0.0,
                 set_gamma=0.0, infer_items=False, algo="mcts", emm_depth=2, emm_samples=8, worst_weight=1.0,
                 lock_choice=True, leaf_eval=None, leaf_mode=2, leaf_scale=1.0, matchup=None, matchup_weight=1.0):
        """samples x ms: the budget per decision (early_samples x early_ms early in the game, as
        Foul Play). near_best: pick among options with at least this share of the best one's
        visits (Foul Play: 0.75); deterministic: always the best. iterations > 0 replaces the time
        budget with a fixed number of MCTS iterations per sample (reproducible tests); iters_per_ms > 0
        with iters_per_ms * ms iterations (a budget that doesn't depend on the machine's load, for
        head-to-heads between search bots). lock_choice: the opponent's active is choice-locked
        into the one move it used this stint when its item (shown or sampled) is a Choice item
        (search.state.StateBuilder.observed_side; off reproduces the first Phase 1 bot).

        Phase 2 (the v21 network, `net`: a search.net.NetModel):
          prior_weight (a): choose by visit share + prior_weight * the network's action probability.
          set_gamma (b): reweight each revealed opponent's candidate sets by the aux heads (tempered).
          infer_items: rule out items by what the opponent did (search.sampling.observations).
          algo "emm" (c): instead of MCTS, the root payoff matrix of a depth-emm_depth expectiminimax
            (search.net / engine patch emm_matrix) on emm_samples sampled teams; each of our options
            scored by worst_weight * its worst case + (1 - worst_weight) * its expected value under
            the network's prediction of the opponent's action. worst_weight 1 is plain
            expectiminimax (no network). Depth 2 costs 1-4 ms per sample, depth 3 a median 70 ms but
            seconds in complex positions, so depth 2 on 8 samples is the default.

        Step 7: leaf_eval, the weights of a learned leaf evaluation (search/leaf_eval.py, from
        training/train_leaf_eval.py) in place of the engine's hand-written one; leaf_mode 1 scores a
        leaf sigmoid(v), 2 sigmoid(leaf_scale * (v - v(root))). Needs the engine patch
        learned_eval.patch and threads 1.

        Step 8: matchup, a .pt of training/train_matchup_eval.py (search/matchup_eval.py): its linear
        part replaces leaf_eval, and a network sets each search's per-Pokemon weights from its root;
        matchup_weight scales that term."""
        if (prior_weight or set_gamma or (algo == "emm" and worst_weight < 1)) and net is None:
            raise ValueError("prior_weight, set_gamma and the opponent model need net (search.net.NetModel)")
        if algo not in ("mcts", "emm"):
            raise ValueError(f"unknown algo {algo!r}")
        self.net = net
        self.prior_weight, self.set_gamma, self.infer_items = prior_weight, set_gamma, infer_items
        self.algo, self.emm_depth, self.emm_samples, self.worst_weight = algo, emm_depth, emm_samples, worst_weight
        self.samples, self.ms = samples, ms
        self.early_samples, self.early_ms = early_samples, early_ms
        self.threads = threads
        self.near_best = near_best
        self.deterministic = deterministic
        self.iterations = iterations
        self.iters_per_ms = iters_per_ms
        self.lock_choice = lock_choice
        self.leaf_scale = leaf_scale
        self.leaf_id = 0   # 0: the engine's hand-written evaluation
        self.matchup = None
        if leaf_eval or matchup:
            if threads != 1:
                raise ValueError("leaf_eval and matchup need threads 1")
        if matchup:
            from search.matchup_eval import MatchupModel
            self.matchup = MatchupModel(matchup, leaf_mode, leaf_scale, matchup_weight)
            self.leaf_id = self.matchup.leaf_id
        elif leaf_eval:
            self.leaf_id = evaluator_id(leaf_eval, leaf_mode, leaf_scale)
        self.rng = random.Random(seed)
        self.debug_path = None   # a file: log every search whose engine options weren't all legal
        self.prior = get_prior()
        self.stats = Counter()
        self.latency = []
        self.sample_iters = []   # MCTS iterations of each sample
        self.last = None   # details of the last decision: {action: share}, iterations, samples
        self.explain = False   # last also gets what describe_decision needs (SearchPlayer(explain=True))

    # ---- budget ---------------------------------------------------------------------------------
    def budget(self, battle):
        opp = battle.opponent_active_pokemon
        early = (len(battle.opponent_team) <= 3 and opp is not None and not opp.fainted
                 and not any(m not in ("struggle", "recharge") for m in opp.moves))
        return (self.early_samples, self.early_ms) if early else (self.samples, self.ms)

    # ---- search ---------------------------------------------------------------------------------
    def search(self, battle, mask=None):
        """{action: score} over the samples, legal actions only (mask): summed visit shares (MCTS)
        or mean option values (expectiminimax)."""
        if mask is None:
            mask = action_mask(battle)
        n, ms = self.budget(battle)
        obs = observations(battle, tracker(battle), infer_items=self.infer_items)
        weight_fn = self.net.set_weight_fn(battle, gamma=self.set_gamma) if self.set_gamma else None
        if self.algo == "emm":
            return self._search_emm(battle, mask, self.emm_samples, obs, weight_fn)
        shares = defaultdict(float)
        q_score, q_visits = defaultdict(float), defaultdict(int)   # per action, over the samples
        # explain: the opponent's options' shares, its active's sampled items, the root's v
        opp, items, v_root = defaultdict(float), Counter(), 0.0
        iters = 0
        value = 0.0
        for _ in range(n):
            sets, _ = sample_opponent(self.prior, obs, self.rng, weight_fn)
            state, _ = build_state(battle, sets, self.lock_choice)
            its = 0   # 0: the time budget
            if self.iterations or self.iters_per_ms:
                its = self.iterations or max(1, int(self.iters_per_ms * ms))
            if self.leaf_id:
                rw = self.matchup.root_weights(state) if self.matchup else None
                r = leaf_mcts(state, ms, its, self.leaf_id, rw)
            elif its:
                r = pe.monte_carlo_tree_search(state, iterations=its, threads=self.threads)
            else:
                r = pe.monte_carlo_tree_search(state, ms, threads=self.threads)
            if self.explain:
                visits2 = max(sum(x.visits for x in r.side_two), 1)
                for x in r.side_two:
                    opp[x.move_choice] += x.visits / visits2 / n
                items[state.side_two.pokemon[int(state.side_two.active_index)].item.lower()] += 1
                if self.leaf_id:
                    v_root += pe.evaluator_values(self.leaf_id, [state.to_string()], rw)[0] / n
            total = max(r.total_visits, 1)
            iters += r.total_visits
            self.sample_iters.append(r.total_visits)
            # the root's mean score for side one (the engine's sigmoid-of-evaluation units, 0..1)
            visits = sum(x.visits for x in r.side_one)
            value += (sum(x.total_score for x in r.side_one) / visits if visits else 0.5) / n
            bad = []
            for x in r.side_one:
                a = option_to_action(x.move_choice, battle, state)
                if a >= 0 and mask[a]:
                    shares[a] += x.visits / total / n
                    q_score[a] += x.total_score
                    q_visits[a] += x.visits
                else:
                    kind = "none" if x.move_choice in ("none", "No Move") else \
                        ("tera" if x.move_choice.endswith("-tera") else
                         ("switch" if x.move_choice.startswith("switch ") else "move"))
                    self.stats[f"illegal option visits dropped ({kind})"] += x.visits
                    bad.append(x.move_choice)
            if bad and self.debug_path:
                self._debug(battle, mask, state, r, bad)
        # q: each action's mean MCTS score over its visits in all samples (0..1, for side one)
        self.last = {"shares": dict(shares), "iterations": iters, "samples": n, "value": value,
                     "q": {a: q_score[a] / q_visits[a] for a in q_visits if q_visits[a]}}
        if self.explain:
            self.last.update(opp=dict(opp), items=items, v_root=v_root if self.leaf_id else None)
        if self.prior_weight:
            pi = self.net.policy(battle)
            before = max(shares, key=shares.get) if shares else None
            for a in np.flatnonzero(mask):
                shares[int(a)] += self.prior_weight * float(pi[a])
            self.stats["prior: decisions"] += 1
            self.stats["prior: changed the top option"] += max(shares, key=shares.get) != before
            self.stats["prior: network's top option == search's"] += int(np.argmax(pi)) == before
        return shares

    def _search_emm(self, battle, mask, n, obs, weight_fn):
        """Mean over the samples of each of our options' value: worst case and/or expected value
        under the network's opponent model, from the root payoff matrix."""
        totals, counts = defaultdict(float), defaultdict(int)
        n_revealed = min(len(battle.opponent_team), 6)
        for _ in range(n):
            sets, _ = sample_opponent(self.prior, obs, self.rng, weight_fn)
            state, _ = build_state(battle, sets, self.lock_choice)
            t1 = time.perf_counter()
            r = pe.emm_matrix(state, self.emm_depth, False)
            self.sample_iters.append(int(1e6 * (time.perf_counter() - t1)))   # microseconds, for report()
            n1, n2 = len(r.s1), len(r.s2)
            m = np.array(r.matrix, dtype=np.float64).reshape(n1, n2)
            worst = m.min(1)
            if self.worst_weight < 1:
                p = self.net.opponent_probs(battle, state, list(r.s2), n_revealed)
                value = self.worst_weight * worst + (1 - self.worst_weight) * (m @ p)
            else:
                value = worst
            for i, o in enumerate(r.s1):
                a = option_to_action(o, battle, state)
                if a >= 0 and mask[a]:
                    totals[a] += value[i]
                    counts[a] += 1
        scores = {a: totals[a] / counts[a] for a in totals}
        self.last = {"values": dict(scores), "samples": n}
        return scores

    def _debug(self, battle, mask, state, r, bad):
        import json
        req = battle.last_request or {}
        s1 = state.side_one
        rec = {"turn": battle.turn, "role": battle.player_role, "bad": bad,
               "options": [x.move_choice for x in r.side_one], "legal": [int(a) for a in np.flatnonzero(mask)],
               "forceSwitch": req.get("forceSwitch"), "active": (req.get("active") or [None])[0],
               "reviving": battle.reviving, "s1_force_switch": s1.force_switch, "s1_active": s1.active_index,
               "s1": [(p.id, p.hp, [(m.id, m.pp, m.disabled) for m in p.moves]) for p in s1.pokemon],
               "s1_vols": sorted(s1.volatile_statuses), "s2_active": state.side_two.pokemon[int(state.side_two.active_index)].id,
               "s2_ability": state.side_two.pokemon[int(state.side_two.active_index)].ability,
               "team": [m.species for m in battle.team.values()],
               "avail_switches": [m.species for m in battle.available_switches],
               "avail_moves": [m.id for m in battle.available_moves]}
        with open(self.debug_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def choose(self, shares):
        best = max(shares.values())
        if self.deterministic or self.algo == "emm":
            top = [a for a, v in shares.items() if v == best]
            return self.rng.choice(top)
        near = [(a, s) for a, s in shares.items() if s >= self.near_best * best]
        return self.rng.choices([a for a, _ in near], weights=[s for _, s in near])[0]

    def action(self, battle):
        t0 = time.perf_counter()
        mask = action_mask(battle)
        legal = np.flatnonzero(mask)
        if len(legal) == 1 or battle.reviving or getattr(battle, "_wait", False):
            # one choice (a locked move, a recharge, a single switch), or Revival Blessing's
            # choice of a fainted Pokemon, which the engine doesn't model
            self.stats["no search (one legal action / revive)"] += 1
            return int(legal[0])
        try:
            shares = self.search(battle, mask)
        except BaseException as e:  # noqa: BLE001 - poke-engine panics surface as PanicException
            if isinstance(e, KeyboardInterrupt):
                raise
            self.stats["search failed: fell back to the first legal action"] += 1
            self.stats[f"error: {type(e).__name__}: {str(e)[:80]}"] += 1
            return int(legal[0])
        if not shares:
            self.stats["no legal option visited: fell back to the first legal action"] += 1
            return int(legal[0])
        self.stats["searched"] += 1
        a = self.choose(shares)
        self.latency.append(time.perf_counter() - t0)
        return int(a)

    def order(self, battle):
        return action_to_order(self.action(battle), battle)

    # as an env-stepped opponent (players.v20_player.opponent_decision calls choose_move)
    choose_move = order

    def report(self):
        lat = np.array(self.latency) if self.latency else np.zeros(1)
        it = np.array(self.sample_iters) if self.sample_iters else np.zeros(1)
        per_sample = (f"MCTS iterations per sample median {np.median(it):.0f} (p10 {np.percentile(it, 10):.0f})"
                      if self.algo == "mcts" else
                      f"expectiminimax depth {self.emm_depth} per sample median {np.median(it) / 1000:.0f} ms "
                      f"(p90 {np.percentile(it, 90) / 1000:.0f} ms)")
        return (f"{self.stats['searched']} searches, latency mean {1000 * lat.mean():.0f} ms, p95 "
                f"{1000 * np.percentile(lat, 95):.0f} ms, max {1000 * lat.max():.0f} ms; {per_sample}; " +
                ", ".join(f"{k}: {v}" for k, v in sorted(self.stats.items()) if k != "searched"))


@functools.lru_cache(maxsize=1)
def _item_names():
    """Item id -> name from the Showdown fork's data, for describe_decision (ids without it)."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "pokemon-showdown", "data", "items.ts")
    try:
        with open(path) as f:
            return dict(re.findall(r'^\t(\w+): \{\n\t\tname: "([^"]+)"', f.read(), re.M))
    except OSError:
        return {}


def describe_decision(battle, last, order, scale):
    """One chat line about a search's decision (SearchPolicy.last with explain on): the choice and,
    with a learned evaluation, the win chance the search's root value implies (mode 2's leaf score
    inverted: sigmoid(v(root) + logit(q) / scale)); our options' shares of the search with the same
    for each; the opponent's options by their share of the search (unrevealed moves, which come from
    the sampled sets, marked "?"; switches to unrevealed Pokemon added up); the sampled item of its
    active while unrevealed."""
    dex = GenData.from_gen(battle.gen)

    def move(m):
        return dex.moves.get(m, {}).get("name", m)

    def mon(p):
        return dex.pokedex.get(p, {}).get("name", p)

    def ours(o):   # a BattleOrder
        x = getattr(o, "order", None)
        if isinstance(x, Move):
            return move(x.id) + (" (Tera)" if o.terastallize else "")
        return "switch to " + mon(x.species) if x is not None else "the default choice"

    act = battle.opponent_active_pokemon
    seen = {p.species for p in battle.opponent_team.values()}
    seen_moves = set(act.moves) if act is not None else set()
    opp, unseen = defaultdict(float), 0.0
    for o, s in last["opp"].items():
        if o in ("none", "No Move"):
            continue
        if o.startswith("switch "):
            p = o[7:]
            if any(p.startswith(x) or x.startswith(p) for x in seen):   # formes: ids differ in the suffix
                opp["switch to " + mon(p)] += s
            else:
                unseen += s
        else:
            m = o[:-5] if o.endswith("-tera") else o
            opp[move(m) + ("" if m in seen_moves else "?") + (" (Tera)" if o != m else "")] += s
    if unseen:
        opp["a Pokemon I haven't seen"] = unseen

    v = last["v_root"]

    def win(q):
        q = min(max(q, 1e-3), 1 - 1e-3)
        return 100 / (1 + math.exp(-(v + math.log(q / (1 - q)) / scale)))

    top = [(a, s) for a, s in sorted(last["shares"].items(), key=lambda kv: -kv[1])[:4] if s >= 0.02]
    mine = ", ".join(f"{ours(action_to_order(a, battle))} {100 * s:.0f}%" +
                     (f" [~{win(last['q'][a]):.0f}%]" if v is not None and a in last["q"] else "") for a, s in top)
    expected = [(o, s) for o, s in sorted(opp.items(), key=lambda kv: -kv[1]) if s >= 0.02][:4]
    parts = [f"Turn {battle.turn}{' (switch-in)' if battle.force_switch else ''}: I chose {ours(order)}."]
    if v is not None:
        parts.append(f"My win chance ~{win(last['value']):.0f}%.")
    parts.append(f"My search: {mine}.")
    if expected:
        parts.append("I expected you to pick: " + ", ".join(f"{o} {100 * s:.0f}%" for o, s in expected) +
                     (" (? = a move I guessed you have)" if any("?" in o for o, _ in expected) else "") + ".")
    if act is not None and act.item == GenData.UNKNOWN_ITEM and last["items"]:
        guess = " or ".join(_item_names().get(i, i) for i, _ in last["items"].most_common(2))
        parts.append(f"Guessed your {mon(act.species)}'s item: {guess}.")
    return " ".join(parts)


class SearchPlayer(Player):
    """poke-env Player driven by a SearchPolicy, for live play (e.g. against Foul Play).
    hold_requests: against this repo's Showdown fork (requests before the turn's log). explain: post
    each decision (describe_decision) once its turn has been played, in explain_room (joined on
    first use) or else the battle's room, where Showdown only lets registered users talk."""

    def __init__(self, policy=None, hold_requests=False, explain=False, explain_room=None, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy or SearchPolicy()
        self.policy.explain = explain
        self.explain_room, self.joined = explain_room, False
        self.notes = {}   # battle tag -> the last decision's explanation, not posted yet
        if hold_requests:
            hold_requests_until_log(self)

    def choose_move(self, battle):
        try:
            asyncio.get_running_loop()
            return self._async_choose_move(battle)
        except RuntimeError:
            if getattr(battle, "_wait", False):
                return DefaultBattleOrder()
            return self.policy.order(battle)

    async def _async_choose_move(self, battle):
        if not self.policy.explain:
            return self.policy.order(battle)
        note = self.notes.pop(battle.battle_tag, None)
        if note:   # the previous decision: its turn has been played since
            if self.explain_room and not self.joined:
                await self.ps_client.send_message(f"/join {self.explain_room}")
                self.joined = True
            await self.ps_client.send_message(note, self.explain_room or battle.battle_tag)
        self.policy.last = None
        order = self.policy.order(battle)
        if self.policy.last and "opp" in self.policy.last:
            try:
                self.notes[battle.battle_tag] = describe_decision(battle, self.policy.last, order,
                                                                  self.policy.leaf_scale)
            except Exception as e:  # noqa: BLE001 - an explanation must never cost the move
                self.policy.stats[f"explain failed: {type(e).__name__}: {str(e)[:80]}"] += 1
        return order
