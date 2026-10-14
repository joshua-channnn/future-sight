# Step 8: matchup weights for the leaf evaluation

October 11-14, 2026; the first item of `step7.md` section 5.

**Result: no gain in play; lin1_86 stays. A matchup-aware evaluation predicts results better
offline, but the search with it is weaker (39-43% against the bot with lin1_86) and, with the
matchup term scaled down to 0.3, level (51.2% ± 3.1% over 1,000 games). Trained instead on the
search's own per-option values (section 4), it values the options one turn ahead much more like
the deep search does (its best option is the search's 54% of the time against 48% for lin1_86),
and the search with it is still level (50.5% ± 3.1%). Trained toward every searched node's value
(TreeStrap, section 5), a small network agrees with the search best of all (correlation 0.898 vs
0.858) and is level as well (50.5% ± 3.1%): it changes the search's pick in 8.6% of decisions,
against 6.4% for the search's own noise.**
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
- Training on the search's values, so the evaluation learns what unplayed options lead to, fixes
  the offline side but not play: the search already explores the root's options deeply itself, and
  the evaluation it needs is the one at its frontier, several turns away (section 4).
- TreeStrap aims at that frontier. The engine exports the nodes of each search (`pe.mcts_tree`).
  The evaluation is trained so that each node's value relative to the root matches what the
  search below that node found. The fit is much closer, but the search's picks barely move: the
  leaf evaluation has little leverage left at this budget (section 5).
- More search doesn't measurably help either. At the ladder's budget (2 x 250 ms per decision),
  twice the iterations wins 51.5% ± 2.2% over 2,000 games against the same bot (section 6), so the
  ladder stays at 2 x 250 ms.

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
- Section 4: `pe.option_features_from_strings` (in the patch: each option of side one played one
  turn ahead against side two's weighted options, the successors' expected features);
  `training/train_q_eval.py` (the successor cache `datasets/q_cache/`, the linear fits) and
  `training/train_q_matchup.py` (the matchup network on the same objective, cache
  `datasets/q_cache_mu/`, 25% of the training decisions). Both caches (gitignored, 4.7 GB) were
  deleted after the step: `train_q_eval.py` rebuilds its own on the first run, the network's is
  built as the `train_q_matchup.py` docstring shows (about 5 minutes each).
- Section 5 (TreeStrap):
  - `pe.mcts_tree(state, ms, iterations, evaluator, min_visits, max_depth=30)` (in the patch)
    returns the search's result and its nodes: features, depth, parent, visits, `q`, whether the
    battle is over.
  - `tools/check_treestrap.py` checks it.
  - `training/gen_treestrap.py` generates the data in `datasets/treestrap_r0/` (gitignored,
    802 MB; deleted after the step). Rebuilding it from the self-play shards takes about an hour
    with 4 workers: `python3 -m training.gen_treestrap --out datasets/treestrap_r0`.
  - `training/train_treestrap.py` fits it (`--mode ls | combined | mlp`).
  - `tools/compare_picks.py` counts how often an evaluation changes the search's pick.
- Models (gitignored unless deployed): `models/leaf_eval/mu*.pt` with their `_lin.npz` (the linear
  part) and `_val.npz` (validation losses per pair, for paired comparisons); section 5's
  `models/leaf_eval/ts_r0.npz` (`_raw`: before the rescaling), `ts_r0_mlp8.npz`, `ts_r0_mlp16.npz`.
- Logs: `logs/mu*.log`; the A/Bs `logs/ab_mu5_*.log` (scale 1.4), `ab_mu5s10_*` (1.0), `ab_mu9s12_*`,
  `ab_mu5w0_*` (a = 0), `ab_mu5w03_*` (a = 0.3), `ab_mu3_*` (stopped before its first readout);
  section 4: `logs/q_cache*.log`, `logs/linq_fit.log`, `logs/qmu*.log`, `logs/ab_linqls_*`,
  `logs/ab_qmu1_*`, `logs/ab_qmu1w05_*`; section 5: `logs/treestrap_gen_r0.log`,
  `logs/treestrap_fit_*.log`, `logs/ab_tsmlp16_eq_*`; section 6: `logs/ab_budget2x_seat*` (the
  first 1,000 games) and `logs/ab_budget2x_r2_*` (the second).

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

## 4. Training on the search's own values
The shards keep, for every decision, the search's per-option mean score `q` (120K iterations,
the hand-written evaluation in round 0, lin1_86 at scale 1.4 in round 1). From each decision the
engine plays every option of ours that the search visited at least 0.5% of the time one turn
ahead (`pe.option_features_from_strings`): against the opponent's recorded choice that turn (its
own search's visit shares, its top 4 options of at least 5%), over the turn's chance outcomes
(branching on damage, as MCTS does at the root). The evaluation is linear in the successors'
features once the root's weights are set, so each option needs only their expectation.
1.0M decisions, 8.1 options each (skipped: 20% unsearched, ~2% where the opponent's row is another
moment of the turn, ~1% whose options didn't map); decisions whose turn may end the battle are left
out. Targets: t = logit(q) / 1.44 (round 0) or / 1.4 (round 1), the search's values in logits of
its evaluation. The loss compares only a decision's options with each other (centered), with a
free scale alpha: a deep value is compressed against one-turn values (alpha = 0.42 for lin1_86).
The targets differ between the options of one decision, so memorizing a game's result can't help.

Validation (13 held-out shards, 80K decisions, 640K options): how the one-turn lookahead with each
evaluation values a decision's options against the search (mean per-decision correlation of the
centered values; how often its best option is the search's best by q, and its most visited):

| evaluation | correlation | = best q | = most visited | spread | in play vs lin1_86 |
| --- | --- | --- | --- | --- | --- |
| lin1_86 | 0.663 | 47.7% | 48.9% | 0.31 | |
| lin, least squares on t (rescaled to lin1_86's spread: `linq_ls`) | 0.694 | 50.8% | 52.2% | 0.31 | 48.8% ± 4.9% (195 / 400, stopped) |
| lin, outcome loss + 10 x this loss (`train_q_eval.py`) | 0.694 | 50.7% | 52.1% | 0.57 | |
| matchup network on t, lin1_86 fixed (`qmu1`, 4 epochs) | 0.729 | 53.5% | 55.1% | 0.46 | 47.5% (95 / 200, stopped) |
| the same, matchup term x 0.5 | | | | | **50.5% ± 3.1% (505 / 1,000)** |
| `qmu2` (8 epochs) / `qmu3` (d 64, 2 layers) | 0.733 / 0.737 | 53.8% / 54.1% | 55.4% / 55.6% | 0.46 | |

(The matchup bots get 536 iterations per ms against 565, as in section 3; `linq_ls` 565.) The linear
model can't use the better targets much; the network can, and with ~250K decisions it saturates
near 0.73. Neither plays better. A one-turn lookahead that agrees with the deep search about the
root's options is something the search computes anyway: it explores those options with 120K
iterations. What its leaves need is a better evaluation at the frontier, three to six turns from
the root, and weights fit to the root's one-turn successors don't carry there.

## 5. TreeStrap: training toward every searched node's value
October 12-13, 2026. Section 4 aimed at the root's one-turn successors; this aims at the nodes
where the search uses its evaluation. After a search, `pe.mcts_tree` (in the patch; MCTS nodes now
keep a `total_score`) walks the tree and returns every node with at least `min_visits` visits: its
engine features, depth, parent, visits, whether the battle is over, and side one's mean score `q`
over the iterations through it. `tools/check_treestrap.py`: a node's visits times `q` is its own
rollout (scored from its dumped features) plus its children's, its visits are its children's plus
0 or 1 (~6% of nodes never rolled out themselves: chance outcomes first reached after their pair
was expanded), the root matches the search's result and `features_from_strings`, and the filters
hold.

Data (`training/gen_treestrap.py`, `datasets/treestrap_r0/`): every 10th searched decision of both
self-play rounds (the true state, the decider as side one), searched again with lin1_86 at scale
1.4 and 56,500 iterations (the A/Bs' budget). Each search has ~610 nodes with at least 25 visits,
of which the root and 32 random others are kept. 108,697 searches, 3.3M nodes, 802 MB, 62 minutes
on 4 processes. By depth: 1 15% (median 119 visits), 2 41%, 3 27%, 4 12%, 5+ 6% (~45 visits).
Nodes where the battle is over (5%) are left out. The target is `t = logit(q) / 1.4`: mode 2
scores a leaf `sigmoid(1.4 * (v(leaf) - v(root)))`, so `t` is the `v(n) - v(root)` that a leaf at
n would need to score what the search below n found. `training/train_treestrap.py` fits the
model's `v(n) - v(root)` to `t` by least squares:
- `--mode ls`: lin's 58 coefficients, closed form.
- `--mode mlp`: `train_leaf_eval`'s antisymmetric one-layer ReLU network, started as lin1_86 with
  two of its units (`--init-lin`), so it learns corrections.

Both fits are rescaled to lin1_86's spread of node values, since the leaf scale is tuned to it.
They come out 10-15% narrower, as any least-squares fit does: its spread is the correlation times
the targets' spread.

Validation: 13 held-out shards, 8,558 searches, 259K nodes. The columns are the correlation of
`v(n) - v(root)` with `t` (overall, then by depth), R^2 at the best scale, and the roots' log-loss
against the game results:

| evaluation | correlation | depth 1 | 2 | 3+ | R^2 | roots' log-loss |
| --- | --- | --- | --- | --- | --- | --- |
| lin1_86 | 0.858 | 0.696 | 0.856 | 0.883 | 0.736 | 0.5301 |
| lin, least squares (`ts_r0`; cosine 0.956 to lin1_86) | 0.868 | 0.714 | 0.869 | 0.890 | 0.754 | 0.5323 |
| network, 8 units (`ts_r0_mlp8`) | 0.894 | 0.755 | 0.888 | 0.916 | 0.799 | 0.5296 |
| network, 16 units (`ts_r0_mlp16`) | 0.898 | 0.767 | 0.892 | 0.920 | 0.806 | 0.5298 |

On the first 41 shards (28 for training) the linear fit is the same, and the network reaches
0.885 with 8 units, 0.887 with 16 and 0.887 with 32. The linear model can't use these targets;
the network can, without losing anything on the results. What it learns isn't a squashing of
the value: the best monotone `h` in `h(v_lin(n)) - h(v_lin(root))` gets 0.857, and a free table
of (`v_lin(n)`, `v_lin(root)`) 0.855. It learns interactions between features, which matter most
in lopsided positions: 0.727 -> 0.827 where `|v(root)| > 3`, 0.903 -> 0.921 where it is under 0.5.

In the search, the 16-unit network at lin1_86's 565 iterations per ms (equal iterations, the
fairest test of the evaluation itself): **50.5% ± 3.1% (505 / 1,000; 125, 114, 130, 136 of
250)**. Cost, measured in one process on an idle M3, in iterations per ms: 589 for the
hand-written evaluation, 546 for lin1_86, 543 for `ts_r0` and 463 for the 16-unit network. The
network is 15% slower than lin1_86, so a level result at equal iterations is a loss at equal time,
and the equal-time A/B wasn't run.

Why closer agreement doesn't change play: the search's picks hardly depend on it.
`tools/compare_picks.py` searches 210 stored decisions (3 held-out shards of round 1, true states,
56,500 iterations) with each evaluation and counts how often the most visited option differs from
lin1_86's. The table averages two runs, since one run of 210 moves by about 2 points:

| evaluation | pick differs from lin1_86's (two runs) |
| --- | --- |
| lin1_86, a second run (the search's own noise) | 6.4% (5.7%, 7.1%) |
| the 16-unit network | 8.6% (7.6%, 9.5%) |
| the linear fit | 8.8% (9.0%, 8.6%) |
| the hand-written evaluation (lin1_86 beat it by 3.6 points in step 7) | 21.0% (21.4%, 20.5%) |

Section 3's matchup term at full weight changed 16-22% and lost 7-10 points. The network changes
about 2% of decisions beyond the noise, against ~15% for the hand-written evaluation. lin1_86's
3.6 points came with ~15%. If the network's changes were worth as much per decision, they would
give about half a point, which 1,000 games can't measure.

## 6. Twice the search at the ladder's budget
Step 6 measured more search at Foul Play's budget: 2.5x the iterations per sample won 53.5% ±
3.1% against the same bot (~25 Elo). The ladder bot has used 2.5x the budget since: 2 x 250 ms
per decision, 4 x 125 ms early. Here the question is whether doubling it again would pay on the
ladder, where time is free (~0.6 s per decision now, ~1.1 s at 2 x 500 ms).

The A/B: the deployed bot (lin1_86 at scale 1.4, the v21 prior at 0.15, set_gamma 0.5, item
inference) at 2 x 500 ms (4 x 250 ms early) against itself at the ladder's 2 x 250 ms. Both run on
fixed iterations, 565 per ms (282,500 against 141,250 per sample). Each batch is 4 processes of 250
games, two per seat:

    B='"prior_weight": 0.15, "set_gamma": 0.5, "infer_items": true, "net": true,
       "leaf_eval": "models/leaf_eval/lin1_86.npz", "leaf_mode": 2, "leaf_scale": 1.4,
       "iters_per_ms": 565'
    python3 -m eval.search_eval --opponents search --config "{$B, \"ms\": 500, \"early_ms\": 250}" \
        --opp-config "{$B, \"ms\": 250, \"early_ms\": 125}" --budget-iters-per-ms 565 \
        --ms 250 --early-ms 125 --battles 250 --seat 1 --port 8123

| batch | result | wins per process (of 250) |
| --- | --- | --- |
| first 1,000 | 51.4% ± 3.1% | 115, 139 (seat 1); 144, 116 (seat 2) |
| second 1,000 | 51.6% ± 3.1% | 134, 129 (seat 1); 132, 121 (seat 2) |
| **all 2,000** | **51.5% ± 2.2% (1,030 / 2,000)** | seat 1 51.7%, seat 2 51.3% |

The result is level within the noise: about +10 Elo, with a 95% CI of -5 to +26. More search pays
less as the budget grows. At the ladder's budget, doubling it isn't measurable in 2,000 games. The
ladder stays at 2 x 250 ms. Running the two samples in parallel would also double the search (it
needs the engine binding to release the GIL), so it isn't worth building for this.

Cost: 5.4 and 6.0 hours per batch on the M3. With 4 processes, decisions took 1.5-1.6 s at
2 x 500 ms and 0.7-0.8 s at 2 x 250 ms, 40-50% slower than one process alone (1.1 s and 0.5 s).
Two single decisions took 14 s and 79 s, against ~2 s at the 95th percentile. The Mac was deep in
swap afterwards (9.4 of 10 GB used), the likely cause, but this wasn't checked.

## 7. What next
- Four learned evaluations have failed to beat lin1_86 in play:
  - the matchup network trained on game results (sections 2-3);
  - the linear model trained on the search's per-option values (section 4);
  - the matchup network trained on the same values (section 4);
  - TreeStrap's network (section 5). It agrees with the search best, and changes its picks in
    only ~2% of decisions beyond the noise.

  At this budget the leaf evaluation has little leverage left. Before an A/B of a new
  evaluation, check with `tools/compare_picks.py` how often it changes the search's pick beyond
  the ~6% noise.
- More search per decision isn't measurable either: twice the ladder's budget is level
  (section 6).
- The cheaper items of `step7.md` section 5: the leaf scale between 1.2 and 1.6, the exploration
  constant, the ladder. No change since lin1_86 has gained measurably, so the ladder (play
  against people, not run yet) is the most informative test left.
- Reuse:
  - `mcts_with(..., root_weights)` takes any per-search weights.
  - `option_features_from_strings` does any one-turn lookahead.
  - `MatchupNet` with `--ids none --move-props --pairs` is the network.
  - `mcts_tree` exports any statistics of the search tree.
  - `train_treestrap.py --mode mlp --init-lin` fits a network that starts as lin1_86.
