# Step 8: matchup weights for the leaf evaluation

October 11-12, 2026; the first item of `step7.md` section 5.

**Result: no gain in play; lin1_86 stays. A matchup-aware evaluation predicts results better
offline, but the search with it is weaker (39-43% against the bot with lin1_86) and, with the
matchup term scaled down to 0.3, level (51.2% ± 3.1% over 1,000 games).**
- The idea: `lin1_86` values every Pokemon's HP, status and boosts alike. A small network looks at
  the root of each search once and sets per-Pokemon weights (and a table for the active pair) that
  every leaf of that search uses; the leaves stay linear and fast (+4.9% per MCTS iteration).
- What the network sees matters more than its size. With species / move / item ids it recognizes
  games and memorizes their results (validation worse from the first quarter epoch). Without ids,
  with moves described by their type, power, category and priority and with pairwise matchup
  features computed by the engine (damage each way, KOs, speed, type effectiveness), it predicts
  results from what changes in the next 1-6 decisions 0.0063 better in log-loss than the linear
  evaluation alone (95% CI 0.0023-0.0107, 1,300 held-out games).
- In the search the full term loses 7-10 points at every scale tried; scaled by 0.3 it is level.
  The term's per-Pokemon weights are fit to how HP changes along the lines the bot played
  predicted the result, and the search uses them to choose between lines that weren't played.
  Everything is built and checked (engine patch, network, trainer, `tools/check_matchup_eval.py`,
  `SearchPolicy(matchup=...)`), and the default bot is unchanged (`BEST_MATCHUP = None`).

## Files
- `search/engine_patch/learned_eval.patch` (now also step 8's part; `build.sh` builds it):
  `Evaluator::with_root_weights` and `mcts_with(state, ms, iterations, evaluator, root_weights)`,
  `evaluator_values(..., root_weights)`; `leaf_features` (16 per Pokemon: alive, HP, HP squared,
  five statuses, item, tera, active, the active's five boosts by `evaluate`'s table); the network's
  inputs `matchup_inputs` (per Pokemon its ids, types, stats, state; the field; lin's 110
  features), `pair_features` (36 pairs x 9), `move_table`; `matchup_layout`, `pair_feature_names`,
  `leaf_features_from_strings`.
- `search/matchup_eval.py`: `MatchupNet` (the network; antisymmetric by construction),
  `leaf_phi` (the engine's leaf features in torch), `MatchupModel` (a trained model for the
  search: `root_weights(state)`).
- `search/policy.py`: `SearchPolicy(matchup=<.pt>)` (its linear part replaces `leaf_eval`).
  `search/leaf_eval.py`: `best_settings()` returns it when `BEST_MATCHUP` is set.
- `training/train_matchup_eval.py`: the trainer (pairs of states, below); `tools/check_matchup_eval.py`:
  the engine's features and values against the network's, antisymmetry, searches with root weights.
- Models (gitignored unless deployed): `models/leaf_eval/mu*.pt` with their `_lin.npz` (the linear
  part) and `_val.npz` (validation losses per pair, for paired comparisons).
- Logs: `logs/mu*.log`; the A/Bs `logs/ab_mu5_*.log` (scale 1.4), `ab_mu5s10_*` (1.0), `ab_mu9s12_*`,
  `ab_mu5w0_*` (a = 0), `ab_mu5w03_*` (a = 0.3), `ab_mu3_*` (stopped before its first readout).

## 1. The model
In mode 2 the search scores a leaf `sigmoid(1.4 * (v(leaf) - v(root)))`. With matchup weights

    v(s) = lin(x(s)) + sum over side one's Pokemon p of u_p . phi_p(s) - the same over side two's
           + W[side one's active][side two's active] (both alive)

where `u` (6 x 16 per side) and `W` (6 x 6) come from the network at the root and stay fixed for
the search. Only what changes within the search counts (mode 2 subtracts the root), and the
per-Pokemon terms are exactly what changes: which Pokemon lose HP, faint, get statused, boost.
Antisymmetry as `lin`: side two's `u` come from the same network with the sides exchanged, and
`W = m(p, q) - m(q, p)`, so the mirrored state has the opposite value exactly.

The network (`ids none, move_props, pairs`, 21K parameters): per Pokemon a token from its types,
tera type, status, 25 floats (HP, level, stats, active, boosts, sleep / rest turns, which moves are
usable), its moves (type, category, power / 100, accuracy, priority, usable; summed after a small
MLP) and aggregates of its pair features over the living opponents (means of the 9, maxima of
damage and effectiveness each way); a global token from lin's features and the field; one
transformer layer (d = 32, 4 heads) over the 13 tokens; `u` from each own Pokemon's output,
`W`'s `m(p, q)` from both outputs and the pair's 9 features. Cost at the root: 0.58 ms per search
(2 per decision).

## 2. Training on pairs of states
`training/train_matchup_eval.py`: a root (a stored decision's state) and a leaf (the same side's
state k = 1..6 decisions later in the same game), logistic loss on the game's result for the
root's decider (ties half):

    logit = lin(x(leaf)) + [m(leaf) - m(root)]     (m: the matchup term with the root's weights)

with lin initialized from `lin1_86` and trained too (lr 1e-4). A root-value head (`logit = lin(root)
+ c(root) + the change`, so c absorbs what the root already says) memorized games at once and is
off (`--no-value`). Data: both rounds of `step7.md`, 156 shards (1.03M states) for training, 13
held out by name (`--val-mod 10`: 1,300 games, 472K pairs; some of them were in `lin1_86`'s
training, which affects the baseline and every model alike). AdamW, batch 1024, cosine LR, MPS,
validation every quarter epoch, the best kept (~2 min per epoch).

Validation pair log-loss (start: `lin1_86` itself, 0.5091; baseline: the same training with m = 0,
`mu0_base`, 0.5086; differences paired by game, 95% bootstrap CI):

| model | network inputs | regularization | best at epoch | log-loss | vs baseline |
| --- | --- | --- | --- | --- | --- |
| mu1 | ids (species, ability, item, moves), value head c | dropout 0, wd 1e-4 | 0 (worse from the first check) | | |
| mu2 | ids, no c | dropout 0.1 | 0.25, then worse | 0.5079 | -0.0007 (-0.0023 .. +0.0008) |
| mu3 | ids zeroed: types, status, stats; d 32, 1 layer | dropout 0.1 | 0.25, flat | 0.5073 | -0.0013 (-0.0027 .. +0.0000) |
| mu4 | no ids, move properties, pair features | dropout 0.1 | 0.5, then worse | 0.5026 | -0.0060 (-0.0101 .. -0.0021) |
| **mu5** | as mu4 | dropout 0.3, wd 0.01 | 1.5 (the end) | **0.5023** | **-0.0063 (-0.0107 .. -0.0023)** |
| mu6 | as mu5 + ability / item embeddings | as mu5 | 0.75, then worse | 0.5045 | -0.0042 |
| mu7 | as mu5, d 64, 2 layers | as mu5 | 0.75, then worse | 0.5043 | -0.0043 |
| mu8 | as mu5, 3 epochs | as mu5 | 1.75 | 0.5031 | -0.0055 |
| mu9 | `MatchupLinear`: u linear in the pair aggregates, HP, stats, active; W linear in the pair features (419 parameters) | none | 2 (the end) | 0.5075 | -0.0011 (-0.0021 .. -0.0002) |

Memorization is the whole story: ~70 states per game share one result, and anything that
identifies the teams (ids; the value head most directly) lets the network learn results instead
of matchups. mu1's validation (3 shards) got worse from 0.49 to 0.99 while its training loss fell
to 0.27. For scale, `lin1_86`'s gain over the hand-written evaluation was ~0.011 (state log-loss).

## 3. In the search
`SearchPolicy(matchup=<.pt>, matchup_weight=a, leaf_scale=s)` against the bot with `lin1_86`
(`eval/search_eval.py --opponents search`, both seats; the matchup bot gets 536 iterations per ms
against 565 for its 4.9% slower iterations and the network's 0.58 ms per search). Early readouts
after 200 games (4 x 50, +-6.9 points):

| model | a | scale | win rate |
| --- | --- | --- | --- |
| mu5 | 1 | 1.4 | 39.5% (79 / 200) |
| mu5 | 1 | 1.0 (spread matched) | 41.0% (82 / 200) |
| mu9 (MatchupLinear) | 1 | 1.2 (spread matched) | 43.0% (86 / 200) |
| mu5 | 0 (its lin part alone; 565 it/ms) | 1.4 | 48.0% (96 / 200) |
| mu5 | 0.3 | 1.4 | 54.0% (108 / 200) |

mu5 with a = 0.3, the full run: **51.2% ± 3.1% (512 / 1,000; 125, 126, 129, 132 of 250)**, so
the 54% of its first 200 games was noise. Between a = 0 (48% after 200) and a = 1 (39.5%) the
curve may peak somewhere near 0.3, but a gain under ~2 points would take 4,000+ games to see.

Why the full term loses although it predicts better. Its changes are large (std 0.40 logits per
1-2 decisions against 0.56 for lin's, correlation +0.40), so with it the leaf values spread 1.46x
wider: at scale 1.4 that is lin1_86 at ~2.0, which lost 7 points in step 7. Lowering the scale to
match the spread doesn't help (41%): the lin part then plays at 1.0, which lost ~10 points in step
7, and mu5's lin alone at scale 1.0 changes the search's top option in 10% of positions. Its own
decisions are wrong too: on 160 stored positions the full term changes the top option in 16-22%
(two runs of the same bot: ~4%), mostly through the per-Pokemon HP and alive weights (without W,
the active weights and the boosts: still 21%). They are not noise from the sampled opponent teams
(our own Pokemon's weights vary 6-17% across the samples of one decision against their spread over
positions): they are systematic. Trained on the bot's own games, they say how a Pokemon's HP
changes along the lines that were played predict the result; the search uses them to choose
between lines, and most of those were not played (which Pokemon takes a hit, who switches in).
Small doses help where the full term misleads.

## 4. What next
- The matchup idea needs training data that shows what the moves that weren't played lead to;
  more of the same self-play games won't fix the bias. Two ways: (1) exploration in the generator
  (some decisions played off the best line, e.g. a random option among the legal ones, flagged
  in the shards), so outcomes after unplayed alternatives exist; (2) train the leaf evaluation on
  the search's own per-action values (`q` in the shards, both rounds): from each root, the engine's
  successor states for each option (`generate_instructions`), and a loss that makes
  `v(successor)` rank the options as the 120K-iteration search did. (2) needs no new games and
  attacks the problem directly; it is the one to try if this line continues.
- Otherwise the cheaper items of `step7.md` section 5: the leaf scale between 1.2 and 1.6, the
  exploration constant, the ladder.
- Reuse: `mcts_with(..., root_weights)` takes any per-search weights, so a better way of setting
  them plugs into the same engine code; `MatchupLinear` (419 parameters) is the low-capacity
  starting point, `MatchupNet` with `--ids none --move-props --pairs --no-value` the network.
