# Step 4 brief: attention over the battle tokens, battle history and auxiliary heads (v21)

For the agent that builds step 4 of `roadmap.md`. Written October 4, 2026, at the end of the
session that ran v20's hyperparameter A/Bs and extended v20 to 20M steps. Everything here was
measured in that session or in the v20 session unless it says otherwise.

The user asked for step 4 in a new chat. Read this file, then confirm the open questions in
section 9 with the user before writing code.

## 0. Read before starting
1. This file.
2. `CLAUDE.md`: how to run things, ports, the rules for counting wins and noise.
3. `agent_docs/v20.md`: v20's design (observation, actions, network, training loop) and every run's
   results. Step 4 extends this code.
4. `agent_docs/poke_env_gotchas.md`: 15 silent bugs this project has hit. Items 9-15 matter most for
   v20-style features and for the labels below.
5. `agent_docs/step4_handoff.md`, sections 2-4 and 6-9: how runs are started, monitored and judged on
   this Mac, and what went wrong before (section 7 here repeats the essentials).
6. `agent_docs/roadmap.md`: the plan, the known limits, the references.

## 1. Where things stand
- **Git:** branch `v20` (off `v19-baseline`), not merged; `origin/v20` is behind (the latest
  commits aren't pushed; `git status` shows how many). Start a new branch off `v20`, e.g. `v21`.
  Never commit to `main`; don't push without asking.
- **Best model: v20 at 20M steps**, `models/ppo_pokemon_6v6_v20.pt` (tracked in git; the 10M model is
  in git history). All head-to-heads below are greedy vs greedy (section 2 says why that matters):

  | Opponent | Games | v20 at 20M | v20 at 10M |
  |---|---|---|---|
  | SimpleHeuristics | 1,000 | 83.9% ± 2.3% | 83.7% ± 2.3% |
  | v19 (SB3, 10M) | 2,000, both seats | 75.4% ± 1.9% | 73.9% ± 1.9% |
  | v18.3 (SB3, ~210M steps) | 2,000, both seats | 67.2% ± 2.1% | 62.8% ± 2.1% |
  | v20 at 10M | 2,000, both seats | 52.3% ± 2.2% | |
  | RandomPlayer | 1,000 | 99.8% ± 0.3% | 99.8% ± 0.3% |
  | Foul Play (local, default search) | 200 | not run | 13.5% ± 4.7% |

- **Equal-step baselines for comparisons:** `models/eval_snapshots_v20/step_*.pt` (gitignored), one
  per 1M from 1M to 20M. The ones you'll use most: 5M `step_0005019854.pt` (77.4% ± 2.6% vs
  SimpleHeuristics over 1,000 games), 10M `step_0010034773.pt` (83.7%), 20M `step_0020025160.pt`
  (83.9%).
- **What's been ruled out: PPO settings.** Each A/B was v20 from scratch with one change, judged by
  greedy head-to-heads against v20 at equal steps (details in `v20.md`):

  | Change | Result |
  |---|---|
  | LR x3 (`lr_initial` 1.766e-4) | Slower early (37% vs 54% at 1M), level at 5M and 10M (47.3%, 49.4%) |
  | No value clip (`clip_range_vf=None`) | Faster start (58.6% at 1M, 55.5% at 2M), level by 5M (51.6%) |
  | Entropy bonus 0.02 (v20: 0.0588) | Much sharper policy, level at 5M and 10M (49.4%, 49.5%) |
  | GAE λ 0.95 (v20: 0.754) | Clearly worse (35.6% at 5M) |
  | v20 itself from 10M to 20M | Small gain: 52.3% vs its 10M self, +4.4 points vs v18.3 |

  Explained variance sat at 0.81 in every run but the λ one. The level, ~83-84% vs SimpleHeuristics,
  didn't move. That points at the network and its inputs, which is step 4.
- **Nothing is running:** no Showdown servers, no training. Docker Desktop may be up (only Foul Play
  needs it).

## 2. Goal and how to judge it
- **Goal:** a v21 that is clearly stronger than v20 at equal training steps, and then a new best
  model.
- **Like for like:** same PPO settings and 20M learning-rate schedule as v20 (`CONFIG` in
  `training/ppo_v20.py`, value clip included), same opponent mix and pool seeds, same 6 workers on 3
  servers. Then a gain belongs to the architecture. x-axis = learner samples (two-sided games give
  both sides' samples), as in v20's logs.
- **The verdict comes from two checks at equal steps**, run after the run stops at that step, on free
  ports (~7 minutes in parallel):
  ```
  python3 -m eval.ppo_test --model models/eval_snapshots_v21/step_<~10M>.pt --opponents model \
      --opp-model models/eval_snapshots_v20/step_0010034773.pt --opp-deterministic --battles 1000 --seat both --port 8000
  python3 -m eval.ppo_test --model models/eval_snapshots_v21/step_<~10M>.pt --opponents simple --battles 1000 --port 8001
  ```
  2,000 head-to-head games are good to ±2.2 points, so "clearly ahead" means above ~53%. 1,000 games
  vs SimpleHeuristics are good to ±2.5.
- **Always pass `--opp-deterministic` to head-to-heads.** Without it the opponent model samples its
  actions while `--model` plays greedily, and `--seat both` only swaps seats: v20@5M greedy beats
  itself sampling 67.7%. The deployed bots play greedily.
- **In-training evals (500 games vs SimpleHeuristics every 1M) are only for the curve's shape.**
  Single points differ by chance by up to ~6 points; at 5M they said 75.0% vs 80.6% for two models
  that 1,000-game evals put at 77.8% vs 77.4%.
- **Horizon:** check at 5M (stop only if clearly worse or broken), take the verdict at 10M (v20 gained
  only ~2-3 points head-to-head from 10M to 20M, so level at 10M means no gain), continue to 20M if
  clearly ahead. Then the final checks: `MODEL=... bash tools/final_checks_v20.sh` (1,000 vs
  SimpleHeuristics, greedy head-to-heads vs v19 and v18.3), a greedy head-to-head vs v20@20M, and
  200 games vs Foul Play (`step4_handoff.md` section 6 has the Docker command).

## 3. What step 4 is, and why
v20's network (`networks/v20_policy.py`, 1.08M parameters) has three gaps that no PPO setting can fix:
1. **Tokens don't interact.** Each Pokémon, move and field token is encoded alone by an MLP. The only
   information shared between tokens is one pooled context vector (field, both actives, and mean and
   max over each team), concatenated into the heads. A move's score sees its own token and that
   context, not the opposing bench. A switch's score sees its own token, the opposing active and the
   context.
2. **No memory.** Apart from a few counters (each Pokémon's last move this stint, switch-in turns,
   weather start, own PP), nothing from earlier turns reaches the network: not the opponent's
   switching habits, not which moves it has used into which Pokémon, not what happened last turn.
3. **No auxiliary losses.** The network is never asked to predict hidden information (the opponent's
   sets, its next action), so it learns about them only through the win/loss reward.

**Step 4, as planned in `roadmap.md`:** an entity-token transformer (3-4 layers, d=128-192) over the
field, the 6+6 Pokémon, the 4 moves and the last 8-16 battle events, with auxiliary heads for the
opponent's next action and its hidden item, ability, moves and tera type. In self-play the labels are
free (section 4.3).

**What Jaxcalibur did** (jaxcalibur.github.io, read October 4, 2026; it peaked at 2557 Elo, #1 on the
ladder, with search adding ~100-150):
- Tokens: 1 field; 2 active Pokémon ("stats, moves, damage calcs"); 5 candidate moves (Struggle gets
  its own); 6 own team; 6 opposing team ("masked if they aren't known yet"); 16 history tokens ("the
  last 16 game events (moves/switch-ins)"); 24 matchup tokens ("damage calculations for a pair of
  Pokémon").
- 8.5M parameters; a non-causal transformer, pre-RMSNorm, with "a position-based mixture-of-experts
  setup" (Pokémon positions use different MLP weights from moves, events, etc.).
- Actions exactly like v20's: each team position emits a switch logit; each move position emits a
  "use" and a "Terastallize and use" logit.
- Heads: win probability (the value); "the item, ability, moves, and Tera type of each opposing
  Pokémon"; "the opponent's next action. This one made the bot substantially stronger."
- PPO with GAE, undiscounted 1/0 reward, entropy plus the zero-avoiding term (v20 has both).
- "Almost 100 million games" in about a week on one H100. "Smaller networks trained for longer
  outperformed larger networks trained for fewer steps."

**Scale here:** v20 plays ~230K games per 10M steps, about 4 hours on this MacBook Air. Keep v21
small, and measure its cost before committing to a size (section 4.7).

## 4. Recommended design
### 4.1 Version, files and compatibility
- New version **v21**, in new files, leaving v20's code working: `envs/rl_player_v21.py` (a
  featurizer that extends v20's), `networks/v21_policy.py`, and v21 classes next to `V20Policy`/
  `V20Player` (same interface: `action(battle)` returns an int, `order(battle)` an order).
- **Checkpoints and loading.** `model_state` (`training/ppo_v20.py`) writes `{"arch": "v20",
  "net_config", "layout": LAYOUT, "model", ...}`, and `players/v20_player.load_checkpoint` rejects any
  other arch or layout. v21 needs arch `"v21"`, its own `LAYOUT`, and one loader that routes by arch.
  v20 checkpoints must keep loading: every comparison is a head-to-head against a v20 snapshot, and
  the pool may hold v20 models.
- **The trainer** imports `V20Net`, `Featurizer` and `LAYOUT` directly and writes arch "v20".
  Recommended: generalize it with a small registry (featurizer, network, policy and label extractor
  per arch) instead of copying its 860 lines. Re-run v20's checks afterwards
  (`tools/check_v20_selfplay.py`, `tools/check_v20_resume.sh`) to show v20 still trains.
- **Entry points that assume v20.** Make each arch-aware:
  - `eval/ppo_test.py` routes every `.pt` to `main_v20` (`V20Policy.load`).
  - The in-training eval (`eval_main`, `training/ppo_v20.py:358`) calls `V20Policy.load` directly.
  - `Worker.opponent` goes through `players/v20_player.load_opponent`, whose `is_v20_checkpoint`
    is just `.endswith(".pt")`.
  - `opponent_decision` dispatches on `isinstance(opponent, V20Policy)`. A v21 policy would fall
    through to `.choose_move`, raise, and abandon every v21 pool game.
  - `eval/foulplay_eval.py` uses `V20Player` for `.pt` models.

  **These failures are silent.** A worker that can't load a pool model plays live self-play instead
  (`pick_opponent`), and an abandoned game just rebuilds the env. A routing bug changes the opponent
  mix without stopping the run. Watch the update line's opponent counts (`'pool_v20'` etc.) and the
  error counters.

  `players/ladder_bot.py` and `players/play_bot.py` still run SB3 models (v18.3 and v18.2) through
  `PPOPlayer`; neither was ever switched to v20.

### 4.2 Observation: v20's tokens plus history
- **Keep v20's 17 tokens as they are** (12 Pokémon, 4 moves, 1 field: `idx int64[140]`,
  `flt float32[1259]`, `mask bool[14]`). They are checked against ground truth
  (`tools/check_v20_env.py`).
- **History tokens:** the last H battle events (H = 16 as in Jaxcalibur; 8 is cheaper), padded to a
  fixed H with a mask.
  - Source: `battle.observations[turn].events` for finished turns, plus
    `battle.current_observation.events` for the turn in progress. Each event is a list of protocol
    fields, e.g. `['', 'move', 'p1a: Garchomp', 'Earthquake', 'p2a: Heatran']`.
    `disable_turn_snapshots()`, which every training and eval process calls, only skips poke-env's
    per-turn copies of the Pokémon; the events stay.
  - Suggested token: one per move or switch-in, with: own/opponent side, the actor's species, kind
    (move, switch, drag), the move or switched-in species, how many turns ago, and the outcome
    folded in from the messages that follow (damage dealt as a fraction of HP, effectiveness, crit,
    miss or immunity, status inflicted, faint, Terastallization).
  - Parse incrementally, as v20's `_Tracker` does: it keeps a cursor into the events, because
    `featurize` runs at every decision.
  - Folding outcomes in has traps:
    - Not every `-damage` after a move is that move's damage: `[from] item: Life Orb`, Rocky Helmet,
      recoil, hazards and weather all show up as `-damage` too.
    - Own HP arrives as `cur/max`, the opponent's as `x/100`.
    - The battle objects hold only the latest state, so the tracker has to remember each
      Pokémon's previous HP itself.
  - Protocol idents are `p1a: Name`; v20's `_Tracker._key` turns them into the `p1: Name` keys of
    `battle.team` / `battle.opponent_team`. Own or opponent depends on `battle.player_role`.
  - Keep `hold_requests_until_log` (gotcha 9) on, or a decision can see a turn's events half
    applied.
- **Matchup tokens (optional, costly).** Jaxcalibur has 24 pairwise damage tokens. poke-env's
  `calculate_damage` is one ~30 µs call per move (`Featurizer._damage`), and v20 only computes the
  active matchup. A full 6x6 grid both ways is up to 288 calls (~9 ms), ~35x the whole featurizer
  (0.26 ms). Bench opponents would also need the temporary `_stats` fill that `featurize` does for the
  opposing active, or the calc fails. If you add any, keep to a few pairs, e.g. each own Pokémon vs
  the opposing active and the reverse.
- **Storage:** training stores `flt` as float16 and rounds the features *before* acting, so the stored
  sample is exactly what the policy saw and the PPO ratio starts at 1. Do the same for any new float
  arrays.

### 4.3 Auxiliary labels: free from the other side's battle object
Every game runs both agents in one `RLEnvV20`. So for each recorded decision, the other side's battle
object holds the truth, in every kind of game (live self-play, pool, SimpleHeuristics): side 1's
labels come from `env.battle2`, side 2's from `env.battle1`.
- **The opponent's hidden set, per opponent token.** v20 orders opponent tokens by first sighting
  (`battle.opponent_team` order), then placeholders. `battle.opponent_team` keys and the other side's
  `battle.team` keys are the same (`p2: Name`). Unrevealed slots get no label (mask).
  - Moves: `truth.moves`.
  - Tera type: from the other side's request (`Featurizer.own_request_info(other_battle)`); poke-env
    drops own tera types (gotcha 14).
  - Ability: `own_request_info` returns the *current* ability before `baseAbility`
    (`envs/rl_player_v20.py:858`), e.g. what Trace copied. Use the request's `baseAbility` for the
    set label.
  - Item: `truth.item` is the *current* item, `""` after Knock Off or a used berry. Decide whether
    the label is the current or the original item.
- **Exceptions handled in `tools/check_v20_env.py` (`opp_tokens`), reuse that code:**
  - Forme changes the two sides report differently (species differ): skip.
  - Abilities that change in battle: Trace, As One, Imposter, Ogerpon's tera.
  - Transform/Imposter copying moves.
- **Illusion is *not* handled there.** `main()` only tags those games
  (`illusion = any(m.ability == "illusion" or m.base_species in ("zoroark", "zorua") ...)`). Zoroark
  (~5% of games) breaks the identity mapping on both sides (gotcha 15). Mask every label of such a
  game with that test.
- **The opponent's next action:** what the other side chose at the same decision (turns are
  simultaneous).
  - **Build it after the action loop in `Worker.play_game`.** `acts` starts as `{a1: 0, a2: 0}`, and
    in live games `acts[a2]` is filled only after side 1's sample has been appended, so reading it
    while appending gives side 1 the label 0 every time.
  - Mask it unless the other side is choosing at this step (`env.agent1_to_move` /
    `env.agent2_to_move`; a forced switch on one side only gives no label).
  - Opponent kinds: v20/v21 policies give an int; SimpleHeuristics and SB3 opponents give a
    BattleOrder (`order_to_action(order, other_battle)`, which returns -1 for default orders: mask).
  - Convert to the observer's frame before `env.step` changes the battles: action `a` < 8 is
    `other.available_moves[a % 4]` (tera if `a` ≥ 4); `a` ≥ 8 is `team_list(other)[a - 8]`, then find
    that Pokémon in the observer's `opponent_team`. Revival Blessing switch actions point at fainted
    Pokémon.
  - Define the class space up front. A factored head is simplest: kind (move or switch); for a move,
    its move vocab index (354) and a tera flag; for a switch, one of the 6 opponent token slots or
    "unrevealed".
- **Measuring the aux heads:** compare each accuracy with a per-species baseline. A global majority
  class is beaten at once by memorizing species. `data/gen9_randbats_sets.json` has each species'
  abilities, tera types and movepools, but no items. Report accuracy both overall and on attributes
  not yet revealed in the observation.
- Labels feed the loss only, never the observation.
- **Storage:** samples are positional 6-tuples in `Worker.play_game`
  (`(idx, flt, mask, action, logp, value)`), turned into named arrays in `collect`, and read by name
  and indexed `[mb]` in `ppo_update`. `tools/check_v20_selfplay.py` unpacks exactly 6 fields, so carry
  labels alongside (e.g. a separate per-sample label tuple) rather than lengthening the tuple, or
  v20's own check breaks.

### 4.4 Network
- **Embeddings:** reuse v20's tables (species 48, move 32, ability/item/type 16) and per-type encoders
  (Pokémon, move, field) up to width d. Add an encoder for history tokens, and learned token-type and
  slot embeddings.
- **Trunk:** a non-causal transformer, 3-4 layers, pre-norm (`nn.RMSNorm` exists in this torch 2.10),
  with a padding mask for placeholders, absent moves and empty history slots. A query row that is all
  padding gives NaN: let padded positions attend to a CLS token or to themselves.
- **Heads, keeping v20's 14 actions:** each move token gives (use, tera + use); each own Pokémon token
  gives a switch logit; the value comes from a CLS (or field) token. Aux heads: item (65 classes),
  ability (206), tera type (21) and moves (multi-label over 354) on each opponent token; the next
  action from the CLS token.
- Keep v20's details: orthogonal init with gain 0.01 on the policy logits, masked logits at -1e8, and
  `evaluate_actions`' entropy and zero-avoiding KL over legal actions.
- **No dropout.** `nn.TransformerEncoderLayer` defaults to `dropout=0.1`. `ppo_update` trains in
  `model.train()` while workers act in eval mode, so dropout would make the PPO ratio noisy from the
  first step (gotcha 5).
- **Inference is cheap enough at the planned sizes** (measured, one CPU thread, batch 1): v20's whole
  network 0.30 ms per decision; a bare d=128, 3-layer trunk over 41 tokens 0.35 ms; d=192, 4 layers
  0.61 ms.

### 4.5 Training loop
- PPO exactly as v20 (`ppo_update`), plus the aux losses: total loss = PPO loss + Σ coef × masked
  cross-entropy (BCE for the moves). Jaxcalibur gives no weights; start small enough that the PPO
  loss still dominates. Log each aux loss and accuracy every update, next to a base-rate predictor's.
  They are the first sign the network is learning anything about hidden information.
- **Gradient clipping is shared.** `clip_grad_norm_(..., 0.543)` clips the summed gradient, and every
  loss shares Adam's moments. Aux losses over 6 tokens × hundreds of classes can dominate the norm and
  shrink the PPO step even when they look small. Log the PPO-only gradient norm next to the total.
  Otherwise the comparison with v20 isn't like for like.
- **Fixed tensor shapes on the GPU.** MPS caches a compiled graph per input shape (~57 MB each). A
  partial minibatch of a new size every update once grew the learner to 7.6 GB. Pad history and labels
  to fixed sizes. `ppo_update` already uses only full 1,024-sample minibatches. Mask the aux losses by
  multiplying with the mask (or `ignore_index`), never by boolean indexing (`logits[mask]`): that
  makes a new shape every minibatch, the same leak.
- **New settings must be `CONFIG` keys, and listed in `RESUME_CHECK_KEYS`.** That covers the arch,
  the history length and the aux weights. `--set` only accepts existing keys, and a resumed run only
  compares the keys in `RESUME_CHECK_KEYS` (`net_config` is one), so anything else changes silently on
  resume.
- Workers stay single-threaded and already batch the live sides of a game into one forward pass.

### 4.6 The update is the bottleneck on this Mac (measured October 4)
- A PPO update is 7 epochs × 40 minibatches of 1,024 = 280 training steps. v20 does it in 7-10 s on MPS
  next to ~45 s of collection: 700-800 samples/s overall, a 10M run in ~4 hours.
- One training step of a bare transformer trunk (`nn.TransformerEncoder`, pre-norm, feed-forward 2d) at
  batch 1,024 × 41 tokens:

  | Trunk | MPS fp32 | MPS fp16 / bf16 autocast | CPU, 8 threads |
  |---|---|---|---|
  | d=128, 3 layers (0.40M params) | 208 ms | 190 / 193 ms | 426 ms |
  | d=192, 4 layers (1.19M params) | 449 ms | | |

  Times 280, that adds ~56 s (d=128) or ~126 s (d=192) to every update: a 10M run of roughly 7 or 12
  hours. These are estimates from the bare trunk; measure the real network early.
- Ways to cut it, each a decision for the user: a smaller trunk (d=96 or 2 layers), fewer tokens
  (H=8), fewer PPO epochs (changes PPO, so the comparison with v20 is no longer like for like), or a
  cloud GPU for the update (roadmap step 5; costs money; `cloud/` handles v20 but has never run it).

## 5. Milestones, each with a check
1. **Featurizer v21:** history tokens and label extraction. Check on live battles like
   `tools/check_v20_env.py`: history tokens against the battle log, labels against the other side's
   true team, v20's 17 tokens unchanged, time per call.
2. **Network:** shapes and masks (no NaN when a whole token group is padding), save/load round trip
   through the arch-routing loader, batch-1 CPU time, and the MPS update time at the real size.
3. **Trainer:** a self-play check like `tools/check_v20_selfplay.py` (labels present, aligned and
   masked; both sides recorded in live games; each side's last return equals its terminal reward) and
   a resume check like `tools/check_v20_resume.sh`. Then re-run v20's own checks.
4. **Smoke run to ~1M steps:** aux accuracies rise above their base rates; the policy learns at least
   about as fast as v20 (v20 at 1M: 54.0% in-training); throughput and learner memory as expected.
5. **The comparison run**, from scratch with v20's settings: checks at 5M, verdict at 10M, on to 20M if
   clearly ahead (section 2).
6. **If v21 wins:** final checks, Foul Play, then docs (README table, a `v21.md`, roadmap). Ask the
   user before changing what the scripts load by default: `eval/foulplay_eval.py` defaults to
   `models/ppo_pokemon_6v6_v20.pt`, while the ladder bot still runs v18.3 (section 4.1).

## 6. Reference numbers
- v20 in-training evals (500 games vs SimpleHeuristics, ±4 points): 54.0, 68.0, 73.0, 75.2, 80.6,
  81.2, 83.0, 81.8, 84.4, 83.4 at 1-10M; 84.6, 81.8, 81.6, 82.6, 83.8, 84.4, 88.0, 84.4, 83.8, 84.4 at
  11-20M.
- v20 diagnostics: approx_kl 0.0016 → 0.0004 over 0-10M (0.0002 at 20M), clip fraction 0.12 → 0.03,
  entropy 1.68 → 1.04 (0.98 at 20M), explained variance 0.81 from 1M on, ~28 decisions per side per
  game late in training, ~22 per game vs SimpleHeuristics.
- Throughput with 6 workers on 3 servers: 900-1,280 samples/s collecting early in a run, 550-1,000
  later; updates 7-10 s on MPS; featurizer 0.26 ms and network 0.30 ms per decision.
- Memory: each Showdown server ~1.2 GB fresh, 1.5-1.8 GB after ~125K games; the six workers' model
  caches grow with the pool (3.4 GB at 15M); the learner grows ~155 MB per 1M steps. At ~10.7 GB the
  Mac swapped and updates doubled. Restart the stack around every 5M (section 7).
- Disk: ~24 GB free. Each run's pool copies the five SB3 seeds (~270 MB); eval snapshots are ~4 MB.

## 7. Pitfalls already hit (don't repeat them)
- **Head-to-heads without `--opp-deterministic`** pit a greedy model against a sampling one (section
  2). v20's first-reported numbers were 5-7 points too high because of it.
- **500-game evals decide nothing** on their own (section 2).
- **The worst bugs here don't crash.** A checkpoint-routing bug quietly changes the opponent mix
  (section 4.1). A misaligned label (section 4.3) just trains a useless head. Before trusting a run,
  check the update line's opponent counts, and the aux accuracies against per-species baselines.
- **MPS and tensor shapes:** keep every GPU input shape fixed (section 4.5).
- **Memory growth on long runs:** when updates slow from ~10 s to ~20 s and the Mac swaps, restart the
  stack: kill the runner, SIGTERM the trainer (it saves), restart the Showdown servers, start the
  runner again with the same arguments. Stopping at 5M for the checks is a natural restart point.
- **The fork's request-before-log order** (gotcha 9): keep `hold_requests_until_log` for local play.
- **poke-env gotchas 10-15:** type chart orientation, PP never decremented, `first_turn`, weather
  start, own ability and tera type only from the request, Illusion.
- **Resuming:** a run resumes from `models/ppo_pokemon_6v6_<version>_latest.pt` and refuses different
  settings: repeat the same `--set KEY=VALUE` flags. Don't edit the trainer, env, network or player
  code while a run is live (eval processes and resumed segments import the code on disk).
- **The trainer rewrites `models/ppo_pokemon_6v6_<version>.pt`** when a run reaches
  `total_timesteps` (20M). For version v20 that's the tracked best model; check before committing.
- **Standalone evals die on a stalled battle** (`EnvTimeout`, about once in 20 runs): rerun them.
  Training recovers on its own.
- **Shell:** zsh doesn't split an unquoted `$VAR` into words (`kill $PIDS` fails silently; use
  `xargs` or `bash -c`); BSD sed has no `\|` alternation (use `grep -E`); `pgrep -f` patterns need a
  `[x]yz` bracket so they don't match your own shell; a background-job session can't sleep in the
  foreground (use `run_in_background` loops that exit on a condition).
- **Monitoring:** after starting a run, start the eval watcher straight away. Once, a run finished
  with no reports at 1M-4M because no watcher was set. The pattern:
  `n0=$(grep -ac '>>> Eval @' LOG); until [ "$(grep -ac '>>> Eval @' LOG)" -gt "$n0" ] || ! pgrep -f "[r]un_v20_mac" > /dev/null; do sleep 20; done`.

## 8. Working with this user
- Solo project, short messages. Give a clear recommendation and act on it, but explain the reasoning:
  the user questions methodology (e.g. why A/Bs stopped at 5M, whether 10M is a fair horizon) and
  expects straight answers, including when an earlier call of yours was wrong.
- Runs go on the MacBook (free). Confirm before any cloud spend.
- They want a short report at every 1M eval and whenever a check finishes, and they may ask to pause:
  kill the runner, SIGTERM the trainer, stop the servers; resume with the same command later.
- Git: work on branches, never commit to `main`, don't push without asking. Commit once per finished
  piece of work (not along the way), ending the message with the session's `Co-Authored-By` line.
- Their Showdown ladder password is in `players/ladder_bot.py` and git history; don't copy it.

## 9. Confirm with the user before building
1. **All at once or one part per run.** Recommended: trunk, history and aux heads together in one v21
   run, then ablate only if it fails. They reinforce each other (history feeds the next-action head;
   attention combines both with the current state), it's the fastest route to a stronger bot, and
   step 3 did the same (v20 bundled every change and won 74% head-to-head vs v19). The alternative,
   one part per run, attributes gains cleanly but costs ~3x the runs at 7+ hours each.
2. **Size and compute (section 4.6).** d=128, 3 layers, H=16 is about 7 hours per 10M run on the Mac.
   Smaller is faster; a cloud GPU for the update would make runs Mac-speed or better but costs money.
3. **Aux heads and weights:** all of Jaxcalibur's (item, ability, moves and tera type per opposing
   Pokémon; the opponent's next action), at small weights.
4. **Settings:** v20's exactly, value clip included, for a clean comparison. Dropping the value clip
   only gave a faster start.
5. **Version and branch:** v21, on a new branch off `v20`.
