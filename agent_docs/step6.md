# Step 6: search at decision time (poke-engine MCTS over sampled opponent teams)

Built from `step6_brief.md` (October 9, 2026). Phase 1 is a Foul Play-style search bot of our own
(MIT-clean: Foul Play's GPL code was read as a reference, nothing copied or imported). Phase 2 puts
the v21 network in the loop.

## Files
- `search/sampling.py`: the set prior (`SetPrior`, from `data/gen9_randbats_set_counts.json`),
  battle species -> sets.json entries, candidate filtering with relaxation, opponent team sampling
  (`observations`, `sample_opponent`, `SetPrior.fill_team`).
- `search/state.py`: poke-env battle -> poke-engine `State`: `LogTracker` (what poke-env doesn't
  keep, read from the event log), `build_state(battle, opp_sets)` (the observer's state),
  `build_true_state(battle, other)` (the true state, for checks), `option_to_action`.
- `search/policy.py`: `SearchPolicy` (V20Policy's interface: `action`, `order`, `choose_move`),
  `SearchPlayer` (a poke-env Player, for live play and Foul Play).
- `search/net.py`: `NetModel`, v21 in the loop (Phase 2: policy prior, hidden-set model, opponent
  model).
- `search/engine_patch/emm_matrix.patch`: a 20-line addition to poke-engine's Python bindings
  (`emm_matrix`: the root payoff matrix of a fixed-depth expectiminimax, every cell evaluated),
  for Phase 2 (c). The installed build otherwise is plain 0.0.48.
- `eval/search_eval.py` (vs SimpleHeuristics / random / a model / another search bot),
  `eval/foulplay_eval.py --model search`.
- Builders and checks: `tools/build_set_prior.py` + `tools/gen_set_counts.js` (the prior),
  `tools/check_engine_build.py` + `tools/showdown_damage.js` (milestone 1),
  `tools/check_set_prior.py` (milestone 2), `tools/check_search_state.py` (milestone 3).
  Logs: `logs/search_checks/`.

## Engine build (milestone 1)
poke-engine 0.0.48 from the PyPI sdist, built in this `python3` with
`PATH=$HOME/.cargo/bin:$PATH python3 -m pip install ./poke_engine-0.0.48.tar.gz --config-settings="build-args=--features poke-engine/terastallization --no-default-features"`
(back to the old one: `pip install poke-engine==0.0.46`). `python3 -m tools.check_engine_build`:
- Tera: MCTS offers `<move>-tera` options while no Pokemon on the side has terastallized, and none
  once one has (the gen4 default build never offers them).
- Damage vs the fork's simulator (getDamage with the roll fixed) on 30 matchups (type chart, STAB
  and Tera STAB, items, abilities, weather, terrain, screens, boosts, burn, Body Press, Foul Play,
  Psyshock, Knock Off, Freeze-Dry, Ogerpon's mask, Eruption...): the engine's highest roll is
  within 1% on 25, exact on 7, 1-2% off on the other 5 (rounding: Multiscale, Technician, Grassy
  Terrain, Fluffy). One engine bug: Body Press's crit roll ignores the user's Defense boosts.
- MCTS speed on real 6v6 states (this Mac, one thread): median 62K iterations per 100 ms sample
  (Foul Play's own build in Docker: median 69K over its searches), so equal wall-clock budgets
  are close to equal search.

Engine facts the converter depends on:
- Unknown species / move / item / ability names silently become NONE (no error); the prior's names
  are all known except three items (Leppa Berry, Light Clay, Scope Lens: "unknown item").
- During search the engine uses 92.5% of the highest damage roll, with a separate branch only when
  the rolls straddle a KO, plus crit branches; 2-5 hit moves always hit 3 times (4 with Loaded
  Dice, 5 with Skill Link).
- It panics on some values: a taunt duration outside 0-2, yawn outside 0-1, rest turns outside
  0-3, Encore without a last move, boosts beyond +-6; a sleep counter above 3 divides by zero.
- Choice lock = the other moves `disabled`; switch options are `switch <species id>`; a side can't
  terastallize once any of its Pokemon has `terastallized` set.

## Set prior (milestone 2)
`data/gen9_randbats_set_counts.json` (1.0 MB): 1M teams of the fork's generator
(`Teams.getGenerator('gen9randombattle')`, 8 worker processes, 80 s), counted by complete set
(level, item, ability, 4 moves, tera type, EVs, IVs) per sets.json entry, with how often each set
was a team's lead, the forme it was generated as, and the generator's team-building data (base-
species pool and weights, each species' types and weaknesses, level). 507 species, 5,979 distinct
sets: the support is small (3 sets seen once in 6M), so 1M teams cover it.

`python3 -m tools.check_set_prior --teams 200000` (a fresh holdout run of the generator):
- Coverage: all 507 sets.json species; every name known to the engine but the 3 items.
- Set frequencies, per species, total variation vs the prior: 0.025 for the holdout's non-lead
  sets against 0.023 for the same number of sets drawn by the sampler (pure sampling noise); leads
  0.056 vs 0.050. No holdout set missing from the prior.
- Filtering: 20,000 holdout sets with observations drawn from themselves (level, some moves, item,
  ability, tera type) keep the true set among the candidates every time, nothing relaxed.
- Teams: `fill_team`'s species frequencies match the generator's (TV 0.009 vs 0.009 noise), and so
  do team-level statistics (max Pokemon of one type, weaknesses, level-100s). Given a revealed
  Pokemon, its teammates are predicted better than by ignoring it for 59 of the 60 most common
  species.

The sampler: a revealed opponent gets a set drawn from its species' sets consistent with its
level, the moves it used, its original item, its ability and its tera type (once terastallized),
weighted by count (by lead count for the opponent's first Pokemon). If nothing fits, constraints
are dropped in order (item, ability, tera, then the sets with the most of the shown moves): 1% of
draws in live games. Unrevealed Pokemon are drawn the way the generator builds teams: a base
species from its weighted pool, rejected when it breaks the generator's limits (species clause, 2
per type, 3 weak to a type, 1 double-weak, 4 weak to Freeze-Dry, one level-100, one Tera Blast
species) given the team so far, then a set.

## Converter (milestone 3)
Side one (the player): exact from the request (stats, HP, item, ability, moves and the active's PP
and disabled moves, tera type and whether used) plus the log. Side two (the opponent): species /
forme, level, HP (Showdown shows ceil(100 * hp / maxhp)%: the middle of that range is used),
status, boosts, volatiles, revealed moves / item / ability / tera from the log; everything else
from the sampled set (stats from its EVs/IVs and the current forme's base stats). Both sides:
volatiles (poke-env Effect -> engine names, with durations in the engine's own units: taunt and
encore by moves made since, yawn, Slow Start, locked moves), substitute HP (1/4 of max HP, 1/10
once hit, as Foul Play), toxic counter (toxic ticks since switch-in), sleep and Rest turns,
charge turns, choice lock / disabled moves, last move this stint, side conditions with remaining
turns (Light Clay from the setter's item), weather / terrain / Trick Room remaining turns (from
when and by whom they were set: rocks, Terrain Extender; set before turn 1 or after a turn's
residual phase they don't count that turn), Wish and Future Sight timing, force-switch (a pivot,
not a faint), trapped, Baton Pass / Shed Tail, slow U-turn (the opponent's move still to come),
can-tera (kept after the terastallized Pokemon faints: the request then clears the flag). A
benched opponent with Regenerator (revealed or sampled) gets its third back.

Fixed after the Phase 1 evals (October 9, found with the search bot's own games; on by default since):
- A recharge turn skipped by Truant left poke-env's `must_recharge` set: the engine saw a Slaking that
  couldn't move while the request offered moves and switches, and the bot fell back to its first
  legal action. "Must recharge" now comes from the request (own side: only Recharge listed) and from
  the log for the opponent (`-mustrecharge`, cleared by its next move, cant or switch).
- The opponent's choice lock: an active holding a Choice item (shown or sampled) that has used one
  move since it switched in can only use that move (`lock_choice`, as Foul Play's `lock_moves`;
  before, the search let a locked opponent pick any move). Live games: when the opponent is really
  locked, the observer locks the same moves in 58-59% of decisions (the rest: its sampled item isn't
  a Choice item). Same for a locked move (Outrage): the engine only stops the locked side switching.
- `infer_items` also infers Heavy-Duty Boots (or Magic Guard) when an opponent switches into Stealth
  Rock without damage (as Foul Play does). On live games every inference is right: 15,359 ruled-out
  items, 415 inferred Boots, none wrong.

`python3 -m tools.check_search_state --games 200 --port 8123` on live two-sided games (random vs
SimpleHeuristics / random; 198 games without Illusion, 17,108 decisions, 9,139 turns):
- Observer vs truth: species / forme, level, fainted, status, revealed moves (all true moves), known
  items, tera, active's boosts, hazards, screens, weather, terrain, Trick Room all agree; HP within
  1% on all but 86 of 79,266 Pokemon (a hidden heal), active volatiles on all but 13 of ~17,000.
  The sampled set's hidden parts are right this often: item 80%, ability 95%, unrevealed moves 81%,
  tera type 60%; all six stats exact 96%.
- Engine fidelity (from the true state and both sides' actual choices, is one of the engine's
  chance branches what Showdown did): 97.8% of turns match on faints, status, actives, boosts,
  hazards, weather and terrain; of those, 97.2% also have every Pokemon's HP within the damage-roll
  spread (9% of the HP change + 1% of max HP): 95.1% overall. HP error on matching turns: median
  0, 90th percentile 2.5% of max HP, 99th 12%. The mismatches are engine mechanics, not the
  converter: simultaneous switches always resolve side one first (Intimidate, weather setters),
  multi-hit counts and Triple Axel, Rest at full HP doesn't fail, Sheer Force doesn't stop Life
  Orb recoil, Heal Block from Psychic Noise doesn't stop Leftovers or draining moves, Explosion
  into a Ghost doesn't faint the user, Take Heart / Beat Up / Armor Cannon / Toxic Chain not
  modeled, Intrepid Sword every switch-in (once per battle in gen9), residual order (burn before
  Leftovers), damage rolls near KO thresholds.
- Options: the engine's options for the deciding side map onto exactly the legal actions on all
  but 1 of ~17,000 decisions, apart from known cases: switches while "maybeTrapped" by a hidden
  Arena Trap / Magnet Pull (the engine is right: Showdown would reject the switch; 39), a locked
  move also offered with tera (one legal action, not searched; 27), Struggle (no engine option;
  9), Revival Blessing (not searched; 1).
- Illusion games (5 of 200 in some runs) are counted apart: the disguise breaks both views
  (gotcha 15), as expected.
- Building a state (sampling + observer state) takes ~0.35 ms.

## Search bot (milestone 4)
`SearchPolicy` per decision: one legal action (a locked move, recharge, a single switch) or
Revival Blessing -> no search; otherwise N samples (2 x 100 ms, or 4 x 50 ms while at most 3
opponents are revealed and its active hasn't used a move: Foul Play's defaults), MCTS on each (one
thread), each option's share of the root visits summed over samples, options mapped to actions
(illegal ones dropped), then a random pick among the options with at least 75% of the best share,
weighted by share (Foul Play's selection). Latency 230 ms per decision (p95 240 ms); engine panics
are caught (none seen) and fall back to the first legal action.

Results (October 9, 2026; Phase 1 bot = Foul Play's defaults, its near-best random pick):

| Opponent | Games | Search bot | v20 (for comparison) |
|---|---|---|---|
| Foul Play (local Docker, default search: 2 x 100 ms) | 500 | **47.4% ± 4.4%** | 13.5% ± 4.7% (200 games) |
| v20 at 20M (greedy), both seats | 2,000 | **80.0% ± 1.8%** (80.8% / 79.2% by seat) | |
| SimpleHeuristics | 1,000 | **91.5% ± 1.7%** (seat 2, 300 games: 90.7%) | 83.9% ± 2.3% |

- Against Foul Play the bot is level within the noise (the brief's target: ~50% at the same
  budget; 47.4% is about -18 Elo against v20's ~-320). Wall clock, one game at a time, nothing else
  on the Mac; our bot as p1 (the challenger); latency mean 237 ms, p95 252 ms, max 485 ms; MCTS
  median 61K iterations per sample (Foul Play's own log: median 69K). 2 of 16,965 searches fell
  back to the first legal action (no legal option visited), no errors. ~12 s per game.
- vs v20 and vs SimpleHeuristics: a fixed budget of 600 MCTS iterations per ms of Foul Play's
  budget (60K per 100 ms sample, the wall-clock median), so several evals could run side by side
  without the load changing anyone's search (the latencies these runs report are under load and
  not comparable). 39 of 52,981 searches fell back to the first legal action.

## Phase 2: the network in the loop (v21 at 20M, `search/net.py`)
`NetModel` runs v21 once per decision (featurize + forward: 1.75 ms) and provides:
- (a) Prior: v21's action probabilities; the bot chooses by visit share + `prior_weight` * pi_net
  (poke-engine's MCTS takes no priors, so they're combined after the search).
- (b) Hidden-set model: each revealed opponent's candidate sets reweighted by the aux heads. A
  candidate's weight is its prior weight times, for each attribute the observation doesn't show
  yet (item, ability, each unrevealed move), the ratio of the head's probability of the
  candidate's value to the prior's own probability of it among the candidates (ratios clipped to
  [1/20, 20], tempered by `set_gamma`). Where the head only knows the per-species rates nothing
  changes; where it knows more, the candidates shift. Plus `infer_items`: items ruled out by what
  the opponent did (two different moves in one stint: no Choice item; a status move: no Assault
  Vest; hazard damage: no Heavy-Duty Boots), as Foul Play's inference does.
- (c) Opponent model: the next-action head mapped onto the opponent's poke-engine options
  (moves, tera moves, switches to revealed Pokemon; the placeholder tokens' mass shared by the
  sampled unrevealed ones), 5% uniform mixed in. Scored on the root payoff matrix of a fixed-depth
  expectiminimax (`emm_matrix`, the engine patch: poke-engine's own expectiminimax prunes the root,
  so only the best row is complete): value = `worst_weight` * worst case + (1 - `worst_weight`) *
  expected value under the network's prediction.

`python3 -m tools.check_set_model --games 40 --port 8123` (how much probability the candidates put on
the opponent's true hidden set, on live positions; mean log-probability, top-1 accuracy in brackets;
~7,000 hidden items, ~30,000 hidden moves):

| | item | unrevealed moves | exact set |
|---|---|---|---|
| prior | -0.410 (0.772) | -0.345 (0.783) | -2.815 (0.279) |
| prior + inferred items | -0.388 (0.787) | -0.338 (0.784) | -2.801 (0.283) |
| prior + aux heads, gamma 0.5 | -0.379 (0.804) | -0.334 (0.808) | -2.796 (0.321) |
| both | -0.354 (0.815) | -0.328 (0.811) | -2.780 (0.327) |

gamma 1 overshoots (worse log-probability than 0.5 on every attribute in a first run of 40 games);
the tera-type head is no better than the prior on hidden tera types (-0.766 vs -0.777 at gamma
0.25, lower top-1), so it isn't used. Ability: too few errors to tell (a few changed abilities
dominate the mean).

### Phase 2 A/Bs (October 9, 2026)
Head-to-heads against the Phase 1 bot, both seats, both bots on the fixed budget of 600 MCTS
iterations per ms (so several runs could share the Mac), 1,000 games unless noted (±3.1 points):

| Variant | Change | vs Phase 1 |
|---|---|---|
| (a) prior 0.5 | visit share + 0.5 * pi_net | **45.4%** (46.0% / 44.8% by seat) |
| (a) prior 0.15 | visit share + 0.15 * pi_net | 52.1% (53.2% / 51.0%) |
| (b) hidden sets | aux-head reweighting (gamma 0.5) + inferred items | 52.6% (52.6% / 52.6%) |
| (c) opponent model | expectiminimax depth 2 x 16 samples, 0.5 worst case + 0.5 expected under v21's next-action head | 37.0% (seat 1, 500 games); 34.9% in 350 of seat 2, stopped |
| (c) vs its own baseline | the same against worst-case expectiminimax (depth 2 x 16) | 50.7% |

- (a): at 0.5 the network overrides the search's top option in 20% of decisions (v21's top action
  agrees with the search's only 45% of the time), and that costs ~5 points: v21 plays ~300 Elo
  below Foul Play-level search, so its preferences are mostly worse than the search's. At 0.15 it
  changes the top option in 8% of decisions: +2.1 points, inside the noise.
- (b): +2.6 points, inside the noise. The better set predictions (item top-1 77% -> 82%) are real
  but small, and a small change in opponent sets moves the search little.
- (c): the next-action head adds nothing measurable to expectiminimax (50.7% ± 3.1% against the
  same search scoring the worst case), and expectiminimax at the depth that fits the budget (2;
  depth 3 takes seconds in complex positions) is far weaker than MCTS.

### Phase 1 settings (October 9, 2026; same setup as the Phase 2 A/Bs)
| Variant | vs Phase 1 |
|---|---|
| deterministic: always the most visited option (Phase 1: Foul Play's random pick among options within 75% of the best) | 50.5% ± 3.1% (49.4% / 51.6%) |
| 3x samples (6 x 100 ms, 12 x 50 ms early: 3x the budget) | 48.0% (seat 1, 300 games, stopped) |
| 2.5x MCTS iterations per sample (2.5x the budget) | 55.0% ± 4.4% (seat 1, 500 games); seat 2 52% after 100 (stopped) |

More sampled opponent teams don't help; more search per team helps a little.

Stopped on October 9 at 16:12 (paused at the user's request; the eval scripts can't resume, so these
restart from scratch), partial: the combined Phase 2 bot ((a) 0.15 + (b)) 49.6% after 250 games of
seat 1; the choice-lock fix vs the same bot without it 48.7% after 300; the final candidate (both,
plus the locks and Boots inference) vs Phase 1 without the choice lock 48.0% after 50. Not run yet:
Phase 2 against Foul Play.
