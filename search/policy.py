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
import random
import time
from collections import Counter, defaultdict

import numpy as np
import poke_engine as pe
from poke_env.player import DefaultBattleOrder, Player

from envs.rl_player_v20 import action_mask, action_to_order, hold_requests_until_log
from search.sampling import get_prior, observations, sample_opponent
from search.state import build_state, option_to_action, tracker


class SearchPolicy:
    def __init__(self, samples=2, ms=100, early_samples=4, early_ms=50, threads=1, near_best=0.75,
                 deterministic=False, seed=None, iterations=0, iters_per_ms=0.0, net=None, prior_weight=0.0,
                 set_gamma=0.0, infer_items=False, algo="mcts", emm_depth=2, emm_samples=8, worst_weight=1.0,
                 lock_choice=True):
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
            seconds in complex positions, so depth 2 on 8 samples is the default."""
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
        self.rng = random.Random(seed)
        self.debug_path = None   # a file: log every search whose engine options weren't all legal
        self.prior = get_prior()
        self.stats = Counter()
        self.latency = []
        self.sample_iters = []   # MCTS iterations of each sample
        self.last = None   # details of the last decision: {action: share}, iterations, samples

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
        iters = 0
        for _ in range(n):
            sets, _ = sample_opponent(self.prior, obs, self.rng, weight_fn)
            state, _ = build_state(battle, sets, self.lock_choice)
            if self.iterations or self.iters_per_ms:
                its = self.iterations or max(1, int(self.iters_per_ms * ms))
                r = pe.monte_carlo_tree_search(state, iterations=its, threads=self.threads)
            else:
                r = pe.monte_carlo_tree_search(state, ms, threads=self.threads)
            total = max(r.total_visits, 1)
            iters += r.total_visits
            self.sample_iters.append(r.total_visits)
            bad = []
            for x in r.side_one:
                a = option_to_action(x.move_choice, battle, state)
                if a >= 0 and mask[a]:
                    shares[a] += x.visits / total / n
                else:
                    kind = "none" if x.move_choice in ("none", "No Move") else \
                        ("tera" if x.move_choice.endswith("-tera") else
                         ("switch" if x.move_choice.startswith("switch ") else "move"))
                    self.stats[f"illegal option visits dropped ({kind})"] += x.visits
                    bad.append(x.move_choice)
            if bad and self.debug_path:
                self._debug(battle, mask, state, r, bad)
        self.last = {"shares": dict(shares), "iterations": iters, "samples": n}
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


class SearchPlayer(Player):
    """poke-env Player driven by a SearchPolicy, for live play (e.g. against Foul Play).
    hold_requests: against this repo's Showdown fork (requests before the turn's log)."""

    def __init__(self, policy=None, hold_requests=False, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy or SearchPolicy()
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
        return self.policy.order(battle)
