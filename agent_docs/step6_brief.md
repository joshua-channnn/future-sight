# Step 6 brief: search at decision time

Written October 9, 2026, after step 4 (v21) and its A/Bs. Everything here was measured or read in
that session unless it says otherwise.

> **Status: approved by the user on October 9, 2026, as written** (section 8). Nothing is built
> yet; the next session builds it, starting at milestone 1 (section 5).

## 0. Read before starting
1. This file.
2. `CLAUDE.md`: how to run things, ports, the rules for counting wins and noise.
3. `agent_docs/poke_env_gotchas.md`: items 9-15 matter for the converter (request-before-log,
   PP, `first_turn`, weather start, own ability and tera type only from the request, Illusion).
4. `agent_docs/v21.md`: v21's featurizer, labels and aux heads (Phase 2 uses them); how the
   two-sided env gives the true state (`tools/check_v21_env.py` is the model for the converter
   checks).
5. `agent_docs/evaluation.md` and `agent_docs/step4_handoff.md` section 6: running Foul Play in
   Docker and the Foul Play eval (`eval/foulplay_eval.py`).
6. Optional: `agent_docs/step4_brief.md` for how step 4 was planned and judged.

State at handoff: branch `v21` (nothing pushed; `v20` is 8 commits ahead of `origin/v20`), no
servers or runs going, poke-engine still 0.0.46 (not upgraded yet), Docker image
`foul-play:latest` built.

## 1. Why search, and why now
- Every PPO setting tried on v20 and every network variant of step 4 (v21: battle history, a
  transformer, aux heads; and its A/Bs) converged to the same level: ~84% vs SimpleHeuristics and
  ~50% head-to-head against each other at equal steps (`v20.md`, `v21.md`).
- Foul Play wins 86.5% vs v20 (v20: 13.5% ± 4.7%, 200 games, `evaluation.md`), roughly 300 Elo,
  with no learned network at all: poke-engine's MCTS on a few sampled guesses of the opponent's
  hidden sets, ~200 ms per decision. Jaxcalibur reports +100-150 Elo from search on top of its
  network.
- So the next big gain is likely search. The network's place in it: a prior over our actions, a
  model of the opponent's hidden set (v21's aux heads) and of its next action (v21's next-action
  head). Those are what step 4 built and couldn't cash in without search.

## 2. What we have
### poke-engine (the simulator)
- pmariglia's Rust battle engine with Python bindings, **MIT licensed** (github.com/pmariglia/
  poke-engine). Installed: `poke-engine` 0.0.46 in this `python3`.
- API (`poke_engine/__init__.py`, `poke_engine.pyi`): `State` / `Side` / `Pokemon` / `Move`
  objects with `from_string` / `to_string`; `generate_instructions(state, s1_move, s2_move)`
  returns every chance branch of one turn (`StateInstructions`: percentage + instructions) and
  `State.apply_instructions` / `reverse_instructions` step through them;
  `monte_carlo_tree_search(state, duration_ms)` returns per-side visit counts and scores for each
  option; `iterative_deepening_expectiminimax(state, duration_ms)` returns the payoff matrix
  (our options x the opponent's) at the depth reached; `calculate_damage`.
- Option strings: a move id (`earthquake`), the same with Terastallization (`earthquake-tera`),
  `switch <pokemon id>`.
- Speed (toy 2v2 state, one thread, this Mac): 61,000 MCTS iterations in 100 ms;
  expectiminimax reached depth 3 in 100 ms. Real 6v6 states will be slower; measure.
- **The installed 0.0.46 is the wrong build:** its search never offers a tera option (it applies
  `-tera` moves when given them). The Python bindings' default feature is `poke-engine/gen4`;
  Foul Play installs 0.0.48 with `--config-settings="build-args=--features
  poke-engine/terastallization --no-default-features"` (`terastallization` implies `gen9`). The
  0.0.48 source is on PyPI; Rust 1.93 is installed in `~/.cargo/bin` (not on the PATH).

### Foul Play (the reference)
- `../foul-play`, **GPL-3**, run in Docker for evaluations. Recipe (`fp/search/main.py`,
  `fp/search/random_battles.py`, `fp/modes/random_battle.py`):
  1. Track the battle from the protocol (its own `Battle` class, `fp/battle/protocol.py`).
  2. Sample N complete opponent teams: each revealed Pokemon gets a full set drawn from the sets
     still consistent with what it revealed, weighted by how often the generator produces each
     set (pkmn's randbats data, downloaded at start); unrevealed Pokemon are random sets that obey
     the team generator's constraints (no repeated species, at most 3 weak to a type, ...).
  3. Run MCTS on each sample (default `--search-time-ms 100`, `--search-parallelism 1`: 2 samples
     x 100 ms, or 4 x 50 ms while the opponent's active hasn't revealed a move).
  4. Add up each option's visit share over the samples (weighted by sample probability) and pick at
     random among options with at least 75% of the best share.
- Its helpers import in this `python3`, but copying or importing them would put GPL code in this
  repo. The recommendation is our own versions (section 4).

### Set data
- `data/gen9_randbats_sets.json` (the fork's `sets.json`) has roles, movepools, abilities and tera
  types, but no items and no frequencies.
- The fork's generator itself (`../pokemon-showdown/dist/sim/teams`, `Teams.generate(
  'gen9randombattle')`) produces complete sets (level, item, ability, 4 moves, tera type, EVs,
  IVs): 2,000 teams in 1.1 s; 12,000 sets had 3,621 distinct complete sets (a long tail). A million
  teams (~10 min) gives the exact distribution our opponents are drawn from: better than pkmn's
  data, which follows current Showdown rather than the February 2026 fork.

### Networks
- v20 (`models/ppo_pokemon_6v6_v20.pt`) and v21 at 20M (`models/ppo_pokemon_6v6_v21.pt`) are
  equally strong. Only v21 has the aux heads: item, ability, tera type and moves per revealed
  opponent (all above per-species base rates) and the opponent's next action (0.41 top-1 vs a
  0.29 base rate).

## 3. Goal and how to judge it
- **Goal:** a bot clearly stronger than v20, then stronger than Foul Play at an equal search
  budget.
- **Main measure:** games vs Foul Play (local Docker, default search). One game at a time: both
  bots search for a fixed wall-clock time, so running games side by side would weaken both
  unevenly. ~9 s per game for Foul Play plus our search time: 500 games (±4.4 points) take
  ~2-3 hours.
- Also: greedy head-to-heads vs v20 (2,000 games, both seats; the search bot slots into
  `eval/v20_eval.play_games` like any policy) and 1,000 games vs SimpleHeuristics.
- Same budget as Foul Play for the main comparisons (~200 ms of search per decision); a larger
  budget is a separate question ("does more search help").

## 4. Recommended design
### Phase 1: a Foul Play-style search bot of our own
1. **Set prior** (`data/gen9_randbats_set_counts.json`): counts of complete sets per species from
   ~1M teams of the fork's generator, with the EVs/IVs (exact stats for any sampled set).
2. **Converter** (`search/state.py`): a poke-env battle (+ its request) -> poke-engine `State`.
   Own side exact from the request (stats, item, ability, tera type, moves, the active's PP);
   both sides' HP, status, boosts, volatiles (poke-env `Effect` -> engine names), substitute
   (estimated, as Foul Play does), side conditions and their remaining turns, weather / terrain /
   Trick Room remaining turns (v20's tracker already keeps the weather start), choice lock / last
   move, force-switch and trapped flags, can-tera per side. Opponent: what's revealed, the rest
   from a sampled set.
3. **Sampler** (`search/sampling.py`): N opponent teams per decision: revealed Pokemon from the
   prior restricted to sets consistent with their species, level, revealed moves / item / ability /
   tera; unrevealed ones from the generator's distribution minus species already seen.
4. **Search** (`search/policy.py`, `SearchPolicy`): MCTS on each sample, visit shares added up,
   mapped back to our 14 actions (`earthquake-tera` -> 4 + i, `switch <id>` -> 8 + team slot);
   same interface as `V20Policy` (`action(battle)`, `order(battle)`), so every eval script works
   unchanged.

### Phase 2: the network in the loop (one A/B each, vs the Phase 1 bot and vs Foul Play)
- **(a) Prior:** combine the visit shares with the network's policy (e.g. share + lambda * pi_net,
  or shares weighted by pi_net^beta). The network has seen ~470K games; MCTS at 200 ms is noisy.
- **(b) Hidden-set model:** weight the sampled opponent sets by v21's aux-head probabilities
  (item, ability, tera type, moves of each revealed opponent) times the generator prior. The aux
  heads condition on the whole battle (e.g. a Choice item from repeated moves), the prior doesn't.
- **(c) Opponent model:** MCTS and expectiminimax treat the opponent as a searcher. With the payoff
  matrix from expectiminimax, score our options by the expected value under v21's next-action
  prediction instead of the worst case (or a mix of the two).

### Later (not in this step): the network's value at the leaves
AlphaZero-style search with v21's value at the leaves needs features for simulated states, i.e. a
featurizer from poke-engine states (or poke-env battles rebuilt from them). Large; worth it only if
Phase 2 shows the network adds to search.

## 5. Milestones, each with a check
1. **Engine build:** poke-engine 0.0.48 with gen9 + terastallization. Check: tera options appear
   in MCTS results; a few known damage numbers match the fork's damage calc.
2. **Set prior:** the counts file. Check: every species in `sets.json` is covered; sampling a set
   for a species reproduces the generator's frequencies.
3. **Converter and sampler**, on live two-sided games (`RLEnvV20`, like `check_v21_env.py`; both
   sides' requests give the true state):
   - the observer's state agrees with the true state on everything it can know;
   - **engine fidelity:** from the true state and both sides' actual choices, poke-engine's chance
     branches contain what Showdown did (HP within 1%, faints, move order, status), on most turns.
     The mismatch rate says how much to trust the engine and finds converter bugs;
   - the options poke-engine offers match the legal actions (our mask), tera included.
4. **Search bot:** legal actions only, latency per decision, 200 games without errors; then vs
   SimpleHeuristics, vs v20 (expect far above 50%) and vs Foul Play (target: ~50% at the same
   budget).
5. **Phase 2 A/Bs** (a), (b), (c).
6. **Docs, and ask** before changing what the ladder or play bots run (they still run v18.3 and
   v18.2).

## 6. Costs and risks
- Phase 1 is mostly the converter and its checks (~500 lines); most of the risk is in details the
  engine needs and poke-env doesn't track exactly (substitute HP, some volatile durations, PP of
  benched Pokemon, wish / future sight timing). The fidelity check measures what they cost.
- poke-engine's mechanics are its own: gaps vs the fork (new moves, rare abilities) cost search
  quality silently. The fidelity check is the guard.
- Evaluations are slow (one game at a time vs Foul Play) and noisy: 500 games are ±4.4 points, so
  small Phase 2 effects need 1,000+ games.
- Illusion (Zoroark) breaks the opponent model as it breaks the features (gotcha 15).
- Wall-clock search depends on CPU load: run the evals with nothing else on the Mac (the
  fanless Air also throttles under sustained load).

## 7. References
- Foul Play: github.com/pmariglia/foul-play (local copy `../foul-play`, GPL-3).
- poke-engine: github.com/pmariglia/poke-engine (MIT); source 0.0.48 on PyPI.
- Jaxcalibur: jaxcalibur.github.io (search adds ~100-150 Elo to the network).

## 8. Decisions (confirmed by the user, October 9, 2026)
1. **Our own converter, sampler and search loop** (MIT-clean; Foul Play's code as a reference
   only), not importing Foul Play's GPL code.
2. **Upgrade the global poke-engine** 0.0.46 -> 0.0.48 built with gen9 + terastallization
   (reversible: `pip install poke-engine==0.0.46`).
3. **Set prior from the fork's own generator** (~1M teams), not pkmn's downloaded data.
4. **Budget:** Foul Play's default (~200 ms per decision) for the main comparisons.
5. **Network for Phase 2:** v21 at 20M (it has the aux heads).
6. **Branch:** `search` off `v21`; nothing pushed.

## 9. Working with this user
- Solo project, short messages. Give a clear recommendation and act on it, but explain the reasoning:
  the user questions methodology and expects straight answers, including when an earlier call was
  wrong.
- Runs and evals go on the MacBook (free). Confirm before any cloud spend. Check the Mac is on power
  (`pmset -g batt`) before long jobs.
- Short reports as results come in (each eval or check), and when something goes wrong.
- Git: work on branches, never commit to `main`, don't push without asking. Commit once per
  finished piece of work, ending the message with the session's `Co-Authored-By` line.
- Their Showdown ladder password is in `players/ladder_bot.py` and git history; don't copy it.
- Shell on this Mac: zsh doesn't split an unquoted `$VAR`; `pgrep -f` patterns need a `[x]yz`
  bracket; foreground `sleep` is blocked in agent sessions (use `run_in_background` loops that exit
  on a condition); stop the Showdown servers you start (each ~1.2 GB).
