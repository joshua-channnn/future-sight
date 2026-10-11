# Step 7: learning from the search bot (self-play data, distillation, a learned leaf evaluation)

Built from `step7_brief.md` (October 10-11, 2026).

**Result: a learned leaf evaluation makes the search bot stronger and level with Foul Play;
distilling the search into the network barely helps the network, and hurts inside the search.**
- Search self-play data: round 0 (step 6's best bot) 11,150 games, 759K decisions; round 1 (the bot
  with the learned evaluation) 5,202 games, 356K decisions. Both sides, with the search's visit
  shares, per-action values and the true engine state of every decision; every shard check passes.
- Distillation into v21. Imitation (policy to the visit shares) makes the network agree more with
  the search (top-1 44 -> 54%) and play worse: 44.2% ± 2.2% vs v20 (v21: 50.9%). Anchored to v21
  (the search's per-action values as a KL-regularized improvement step, `--target qadv --beta 100`),
  replicated on all the data: 52.0% ± 2.2% vs v21 and 53.6% ± 2.2% vs v20, 84.4% vs
  SimpleHeuristics (v21: 85.5%), a gain of ~2 points at the edge of the noise. As the search's
  network (prior and set model), distilled networks lose: 47.4% ± 3.1% and 48.0% ± 4.9%.
- The learned leaf evaluation (a linear model on 110 state features computed in Rust, fit to the
  results, used by MCTS in place of Foul Play's hand-written evaluation) at the same time budget:
  **53.6% ± 2.2% vs step 6's best bot over 2,000 games** (~25 Elo), **50.9% ± 3.1% vs Foul Play
  over 1,000 games** (step 6's best: 47.4% ± 3.1%), 81.1% ± 2.4% vs v20, 94.4% ± 1.4% vs
  SimpleHeuristics. `players/ladder_bot.py` and `players/play_bot.py` use it.
- One round of the AlphaZero loop (the new bot's games -> a new evaluator): no gain. Evaluators
  refit on round 1's games predict results no better than `lin1_86` (log-loss within 0.001 on 302
  unseen games), and the A/B is 50.1% ± 2.2% over 2,000 games. The linear model on these features
  is saturated; `lin1_86` stays.

## Files
- `training/gen_search_games.py`: the generator (format in its docstring); `tools/check_search_games.py`:
  shard checks and statistics.
- `training/distill_v21.py`: distillation into v21 (`--target shares | mix | qadv`, `--data` takes
  several directories; whole shards held out; validation: KL / top-1 against the visit shares, the
  greedy pick's regret by the search's own per-action values, value and aux metrics).
- `search/engine_patch/learned_eval.patch` (applied after `emm_matrix.patch` by `build.sh`): state
  features in Rust (`features_from_strings`, `feature_names`) and an antisymmetric model the MCTS can
  use at the leaves (`load_evaluator`, `mcts_with`, `evaluator_values`); the engine's own search is
  unchanged. `search/leaf_eval.py` registers weights, `best_settings()` gives the deployed bot's
  arguments (or none, with a warning, when the engine build or the weights are missing);
  `SearchPolicy(leaf_eval=<.npz>, leaf_mode=2, leaf_scale=1.4)` searches with them.
  `training/train_leaf_eval.py` fits them.
- `tools/check_distill_shift.py`: a network plays v20 while the search labels its positions; the
  regret of its moves and other networks' on those positions.
- `tools/foulplay_loop.sh`: Foul Play evals that survive its crashes (below).
- `eval/search_eval.py`: `"net_path"` in `--config` puts another v21 checkpoint in the search.
- Data (gitignored): `datasets/search_selfplay/` (round 0), `datasets/search_selfplay_r1/` (round 1).
- Models: `models/leaf_eval/lin1_86.npz` (the deployed evaluator, trained on 88 round-0 shards; in
  git); gitignored: `lin_68.npz` (the first one, 67 shards), `lin_r1.npz` / `lin_r01.npz` (round 2:
  round-1 data / both rounds), the distilled `models/v21_distill_p*.pt` (pilots) and
  `models/v21_distill_final.pt` (the replication).
- Logs: `logs/gen_search.log`, `logs/gen_search_r1.log`, `logs/check_search_games_r1.log`,
  `logs/distill_*.log`, `logs/eval_p*_*.log`, `logs/eval_final_*.log`, `logs/nearbest_*.log`,
  `logs/leaf_*.log` (`leaf_r2_*`: the round-2 A/B), `logs/leafnet_pQ100_*.log`,
  `logs/prior_pilot1_*.log`, `logs/shift_test_pA.log`, `logs/foulplay_*.log`, `logs/fpleaf_run1.log`.

## 1. The generator
As the brief: 5 forkserver workers on one Showdown server, each with an `RLEnvV20` and two
`SearchPolicy` objects with the best bot's settings (v21 prior 0.15, aux-head set model 0.5, item
inference) on a fixed budget of 600 MCTS iterations per ms of Foul Play's budget (2 x 60K per
decision, 4 x 30K early), sharing one `NetModel`. Per decision and side: the v21 observation, the
visit shares (before the prior; one-hot when there was one legal action), the action played (near-
best pick), the root's mean score and each action's mean score `q`, iterations, v21's aux labels
(set labels, the other side's next action), the outcome, and the true engine state (both sides
exact, the decider as side one) in the shard's `.states.gz`. Shards of 100 games, written
atomically; a restart is a new run that adds shards (`--out` picks the directory, `--config` the
`SearchPolicy` arguments).

Robustness: no websocket keepalive pings (the bindings hold the GIL during a search), EnvTimeout ->
drop the game and rebuild the env, dead workers restarted, pause on battery (`pmset`), stop below 5
GB free, SIGTERM finishes the current games. None of it was needed: 0 errors in 25.5 h.

| round | bot | run | games | samples | p1 wins |
| --- | --- | --- | --- | --- | --- |
| 0 | step 6's best (hand-written evaluation) | 20261010-032848-cc32, 17.7 h | 11,150 | 758,795 | 50.1% ± 0.9% |
| 1 | + the learned evaluation (`lin1_86`, 565 it/ms) | 20261011-011342-cf62, 7.8 h | 5,202 | 355,773 | 50.6% ± 1.4% |

Both stopped by hand: round 0 for the Foul Play eval, round 1 because the evaluator didn't need more
data (section 4) and the A/Bs needed the Mac. 97.5% of decisions searched, 34 per side per game, 0.31
KB per sample (states 0.04 KB), 15 samples/s alone on the Mac (~1.3M a day), 9-13 with evals
alongside. `python3 -m tools.check_search_games <dir>`: every check passes on both (shapes, targets
sum to 1 over legal actions, outcomes per game and side, labels, every state round-trips through
`State.from_string`, side-1/side-2 pairs are exact mirror images: 227K and 156K). The visit shares
(round 0): 8.3 legal actions, entropy 1.30 nats, the top action's share 0.54 on average (the gap to
the second is under 0.1 in a quarter of decisions), the action played is the most visited 87% of the
time; round 1 the same within 0.01.

## 2. Distillation
`training/distill_v21.py`: v21 from its 20M weights, AdamW, batch 1024, cosine LR with warmup,
3 epochs; policy cross-entropy to the target, value MSE to the result (coefficient 0.5), v21's aux
losses (PPO's coefficients); only searched decisions; ~5 min per 450K samples on MPS.

Pilots (round 0, 445-650K samples; validation = 2 held-out shards, 200 games; greedy evals with
`eval/ppo_test.py`, both seats, ±2.2 at 2,000 games):

| network | target | top-1 | KL | regret (> 0.05) | vs v20 | vs v21 | vs SimpleHeuristics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v21 (start) | - | 44.2% | 0.911 | 0.034 (21.5%) | 50.9% | - | 85.5% |
| pA | shares, lr 1e-4 | 53.5% | 0.354 | 0.024 (13.8%) | **44.2%** | 45.9% (1,000) | 79.7% |
| pB | shares, lr 3e-4 | 55.2% | 0.335 | 0.023 (13.2%) | 45.1% | | |
| pQ | qadv beta 30 | 50.1% | 0.947 | 0.026 (15.9%) | 50.0% | | |
| pQ100 | qadv beta 100 | 53.5% | 0.740 | 0.023 (13.6%) | 52.4% | **53.1%** | 86.7% |
| pQ300 | qadv beta 300 | 54.8% | 0.606 | 0.023 (13.2%) | | 49.4% | |
| pM | mix 0.5 | 50.0% | 0.435 | 0.027 (16.4%) | 52.8% | 52.2% | |
| pM75 / pM25 | mix 0.75 / 0.25 | | | | | 50.1% / 49.4% | |
| **final** | qadv beta 100, both rounds (1.05M) | 53.5% | 0.755 | 0.024 (14.4%) | **53.6%** | **52.0%** | 84.4% |

(top-1: agreement with the most visited action; regret: the search's q of its best visited action
minus q of the network's greedy pick; the final's validation is 3 shards of both rounds.) pA played
with the search's near-best rule instead of greedily: 42.9% ± 3.1% vs v20, so it isn't
argmax-of-a-mixed-strategy.

Why imitation makes the network weaker. `tools/check_distill_shift.py`: pA played 100 games against
v20 while the search labeled each of its 2,719 decisions. On those positions (the network's own) pA's
moves still have lower regret than v21's by the search's measure (0.039 vs 0.042, big mistakes 21% vs
24%), so it is not distribution shift: the distilled network does pick what the search would, and
loses more. The search's values are values under the search's own continuation (its next moves are
searched too); a move that is right for a player that will search the follow-up isn't right for a
network that won't. DAgger-style data (the network plays, the search labels) has the same labels, so
it wasn't tried. What works is staying close to the network: v21's policy reweighted toward the actions
the search values more (`qadv`: ref * exp(beta * (q - max q))) or mixed half and half with the visit
shares; the pull toward the search must be moderate (beta 300, or 75% shares: level again). pQ100 was
the best of five variants (+3.1 vs v21; the best of five noisy results reads ~2 points high), so the
final run repeats it on all the data with fresh evals: +2.0 vs v21, 53.6% vs v20 (v21 itself 50.9%).
A real gain of ~2 points (~15 Elo), small next to the search's ~250 Elo over the network.

As the search's network (prior 0.15 and the set model, in place of v21):
- pB (imitation) vs step 6's best bot: 48.0% ± 4.9% (400 games). It agrees with the search more
  (55% vs 45% top option), so the prior changes fewer decisions (4% vs 8%), and those changes don't
  help.
- pQ100 in the bot with the learned evaluation, vs the same bot with v21 (565 iterations per ms
  each): 47.4% ± 3.1% (1,000 games).
Neither helps, and together (47.5% ± 2.6% over 1,400 games) they probably hurt: a prior that echoes
the search's own preferences adds no independent information.

## 3. A learned leaf evaluation
poke-engine's MCTS expands one node per iteration and scores it with
`sigmoid(0.0125 * (evaluate(leaf) - evaluate(root)))`, Foul Play's hand-written evaluation
(HP, alive, status, items, hazards, the actives' boosts and volatiles, screens, -75 for a used tera).
The patch adds:
- **Features** of a state in Rust, 110 floats: per side (52) one-hot of the number alive, HP sum
  and sum of squares, the active's HP, items, tera used / active terastallized, statuses (burn weighted
  like `evaluate`'s, the good-status abilities apart), hazards (Stealth Rock count and damage, Spikes,
  Toxic Spikes, Sticky Web on the Pokemon they hit), the active's boosts (`evaluate`'s table),
  accuracy / evasion, volatiles (Leech Seed, Substitute, confusion, Taunt, Encore, Yawn, Perish,
  trapped, Salt Cure), screens, Safeguard, Tailwind, Healing Wish, Wish, Future Sight, a stat total
  of the living, force-switch; cross (6) `evaluate(state) / 100`, who is faster (Trick Room
  included), and a rough best-move damage estimate of each active against the other (damage formula
  with boosts, STAB, type chart, ability immunities, burn; as a fraction of the target's HP) with
  KO and KO-first versions. ~15K states/s from strings.
- **The model**: v(x) = g(x) - g(swap(x)), the logit of side one winning, g linear (or one ReLU
  layer); swap exchanges the sides and negates the cross features, so v(mirror) = -v exactly.
  Leaf score: mode 2 `sigmoid(scale * (v(leaf) - v(root)))` (mode 1: `sigmoid(v(leaf))`).
- Cost: the linear model is ~6% slower per MCTS iteration (a 16-unit MLP ~18%), so the A/Bs give it
  565 iterations per ms against the opponent's 600 (median 57K vs 60K per sample; equal latency).

`training/train_leaf_eval.py`: features of every stored state (the decider as side one), logistic
loss to the result, linear by L-BFGS. On 1,400 held-out round-0 games: log-loss 0.524 (72.3% of
results called right) against 0.536 (71.6%) for the hand-written evaluation with its best single
scale (a = 0.86 logits per 100 points, so the engine's 0.0125 per point is 1.44 logits); a 16-unit
MLP and 4 extra bench-matchup features per side (feature set 2, tried and removed): no better.

A/Bs against step 6's best bot (`eval/search_eval.py --opponents search`, fixed budget, both seats):

| evaluator | games | win rate |
| --- | --- | --- |
| linear (67 shards), mode 2, scale 1.4 | 1,000 | 53.4% ± 3.1% |
| linear (88 shards, `lin1_86`), mode 2, scale 1.4 (replication) | 1,000 | 53.8% ± 3.1% |
| both | 2,000 | **53.6% ± 2.2%** |
| linear (67), mode 2, scale 1.0 / 2.0 | 250 / 200 | 43.2% / 47.0% |
| linear (67), mode 1 (absolute) | 250 | 45.6% |
| linear + bench features (88), scale 1.4 | 400 | 51.0% ± 4.9% |

The scale matters as much as the model: 1.4 is the hand-written evaluation's own sharpness in logits,
and both softer and sharper lose (the MCTS's exploration constant is tuned to that spread). The
learned evaluation changes the search's top move in ~15% of positions (agreement with the hand-written
search 81.5%, against 96% between two hand-written runs).

The bot with `lin1_86` (prior 0.15, set model, item inference, scale 1.4), Foul Play's budget:

| opponent | games | win rate | step 6's bot |
| --- | --- | --- | --- |
| Foul Play (local Docker, default search; wall clock, alone on the Mac) | 1,000 | **50.9% ± 3.1%** | 47.4% ± 3.1% (best) |
| v20 (greedy), 565 iterations per ms | 1,000 | 81.1% ± 2.4% | 80.0% ± 1.8% (Phase 1, 2,000) |
| SimpleHeuristics, 565 iterations per ms | 1,000 | 94.4% ± 1.4% | 91.5% ± 1.7% (Phase 1) |

Against Foul Play: latency mean 237 ms (p95 251), median 58K iterations per 100 ms sample. Foul
Play's engine crashed twice (`PanicException: Invalid PokemonIndex: 6`, after ~575 and ~180 games),
so the 1,000 games are three runs (296/570, 83/180, 130/250); `eval/foulplay_eval.py` now stops when
a challenge isn't accepted (it used to wait forever), and `tools/foulplay_loop.sh` restarts Foul Play
until N games are done. Taken alone, 50.9% vs 47.4% is within the noise; the head-to-head (53.6% ±
2.2%) is the measurement of the gain.

## 4. The AlphaZero loop, round 1
Round 1's games (the bot with `lin1_86`, both sides) trained two round-2 evaluators: `lin_r1` on
round 1 alone (49 shards, 335K states) and `lin_r01` on both rounds (161 shards, 1.08M states).
Log-loss on games none of them trained on:

| evaluator | round 0 (200 games) | round 1 (302 games) |
| --- | --- | --- |
| hand-written, best scale | 0.5255 | 0.5535 |
| `lin1_86` (round 1's) | 0.5146 | 0.5261 |
| `lin_r1` | 0.5114 | 0.5254 |
| `lin_r01` | 0.5122 | 0.5251 |

`lin_r1` - `lin1_86` on the round-1 games: -0.0007 (95% bootstrap over games -0.007 .. +0.006). The
first comparison, on one held-out round-1 shard (100 games), had `lin_r1` ahead by 0.004 and chose it
for the A/B; on more games the three learned evaluators are level. A/B, `lin_r1` vs `lin1_86` (565
iterations per ms each, both seats): **50.1% ± 2.2%** (1,002 of 2,000; the four 500-game runs
52.2%, 47.8%, 50.4%, 50.0%). No gain, so `lin1_86` stays deployed. 110 features with one weight each
have said what they can: a stronger bot's games give the same fit, as did a 16-unit MLP and the bench
features (section 3). More rounds of the loop need a model that sees more of the position.

## 5. What next
The search bot with `lin1_86` is the best bot, level with Foul Play at its budget. In order:
1. **A matchup-aware leaf evaluation** (recommended; days of work). What a linear model of per-side
   totals can't express is the matchup: how much each Pokemon's HP is worth depends on what it faces.
   Keep the leaves linear and let a small network set their weights once per search: it looks at the
   root (the sampled opponent team included) and outputs per-Pokemon weights for HP, alive and
   status, which every leaf of that search uses. Mode 2 scores a leaf relative to the root, so only
   terms that change within the search matter, and these are the ones that do. Cost: one network call
   per sampled team, next to 100 ms of search. Needs: per-Pokemon features in the Rust feature set,
   `mcts_with` taking a weight vector per call instead of a registered evaluator, and training on pairs
   from the stored games (weights from the state at one decision, features of a state a few turns
   later, the game's result); the data is already there (`states.gz`, both rounds).
2. **Tune the search around the new evaluation** (cheap, hours per setting; can run while building 1).
   The scale is the most sensitive setting (1.0: 43%, 1.4: 53.6%, 2.0: 47%); 1.2 and 1.6 haven't been
   tried. The exploration constant was chosen with the hand-written evaluation and is fixed in
   poke-engine's `src/mcts.rs` (an argument of `mcts_with` would expose it). Expect a point or two at
   most, which takes ~2,000 games per setting to see.
3. **More rounds of the loop only after 1.** The pieces work as they are: `gen_search_games.py --out
   datasets/search_selfplay_r2 --config <the new bot>`, `train_leaf_eval.py --data` several directories.
4. **No more distillation.** The network alone gained ~2 points and the bot doesn't use it alone;
   inside the search, distilled networks lost.
5. **The ladder** is the real test against people and other bots (`players/ladder_bot.py`: 2 x 250 ms
   per decision with the learned evaluation). It plays in public, so it's the user's call.
