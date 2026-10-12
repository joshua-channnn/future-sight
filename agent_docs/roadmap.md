# Roadmap (from the September 2026 review)

## Where things stand
- **Step 8 (matchup weights for the leaf evaluation) is done, with no gain** (`step8.md`). A small
  network sets per-Pokemon leaf weights once per search; trained on pairs of the self-play games'
  states it predicts results 0.0063 better in log-loss than lin1_86's linear evaluation (after
  removing what let it memorize games), but the search with it loses 7-10 points against the bot
  with lin1_86 and, scaled down to 0.3, is level (51.2% ± 3.1% over 1,000 games). The weights fit
  the lines the bot played; the search chooses between lines it didn't. Trained instead on the
  search's own per-option values (every option played one turn ahead), the evaluation agrees with
  the deep search about the best option 54% of the time against lin1_86's 48%, and the search with
  it is still level (50.5% ± 3.1%): the search explores the root's options itself, and its leaves
  are several turns deeper. lin1_86 stays.
- **Step 7 (learning from the search) is done: a learned leaf evaluation is the gain** (`step7.md`).
  The search bot played itself (16K games, 1.1M decisions recorded with visit shares, values and
  engine states). A linear model of 110 engine-state features, fit to the results and used by the
  MCTS at the leaves in place of Foul Play's hand-written evaluation, wins 53.6% ± 2.2% against
  step 6's best bot (2,000 games) and 50.9% ± 3.1% vs Foul Play (1,000; step 6: 47.4%); the ladder
  and play bots use it. Distilling the search into v21 gave the network ~2 points with a target
  anchored to v21 (52.0% ± 2.2% vs v21) and made it weaker with plain imitation (44.2% vs v20);
  distilled networks inside the search lost ~2.5. A second round of the loop (the new bot's games,
  a refit) gave nothing (50.1% ± 2.2%): the linear model is saturated.
- **Step 6 (search at decision time) is done: the best bot is now the search bot** (`step6.md`,
  `search/`). poke-engine MCTS over opponent teams sampled from the fork's own generator, Foul
  Play's recipe in our own code, at Foul Play's default budget: 47.4% ± 4.4% vs Foul Play (v20:
  13.5%), 80.0% ± 1.8% vs v20, 91.5% ± 1.7% vs SimpleHeuristics. Putting v21 in the loop (policy
  prior, hidden-set model, opponent model) adds little: the best combination wins 55.8% vs the Phase
  1 bot head-to-head but stays at 47.4% ± 3.1% vs Foul Play over 1,000 games. Step 5 (cloud
  scale-out) wasn't done: step 6 was chosen over it.
- **Step 4 is done, with no gain** (`v21.md`). v21 (v20's tokens plus 16 battle-history tokens, a
  3-layer d=128 transformer, auxiliary heads for the opponent's hidden set and next action), from
  scratch with v20's settings, is level with v20 at equal steps: 48.1% ± 2.2% head-to-head vs v20
  at 10M and 50.9% ± 2.2% at 20M (2,000 games each, both greedy); 85.5% vs SimpleHeuristics at 20M
  (v20: 83.9%). It started slower (34% vs 54% at 1M) and caught up only slowly, the aux heads
  learned (all above per-species base rates), and each update cost ~6x v20's on the Mac. v20 stays
  the best model.
- **Step 3 is done: v20 is the best model** (`models/ppo_pokemon_6v6_v20.pt`; details in
  `v20.md`). Trained from scratch on the MacBook with v19's PPO settings. At 10M steps it won
  83.7% ± 2.3% of 1,000 games vs SimpleHeuristics, 73.9% ± 1.9% vs v19 and 62.8% ± 2.1% vs v18.3
  (2,000 games each, both seats, both greedy); continued to 20M on its own schedule, it wins
  83.9%, 75.4% and 67.2%, and 52.3% ± 2.2% against its 10M self. Its in-training evals: 54.0% at
  1M (v19: 13.4%), 68.0% at 2M (34.0%), 80.6% at 5M (51.0%), 83.4% at 10M (63.0%), then 82-88%
  to 20M (500-game evals can't show the slow late gain). A 3x learning rate and dropping the
  value clip changed how fast it learns, not where it ends up, a lower entropy bonus made it
  sharper, not stronger, and GAE λ 0.95 made it clearly worse (`v20.md`).
- v19 = the v18 architecture plus bug fixes (see `poke_env_gotchas.md`), sparse reward and Wang's
  full PPO recipe. Trained from scratch to 10M steps on a GCP VM (20 envs) and stopped there on
  purpose. In-training evals vs SimpleHeuristics (500 games each): 13.4% at 1M, 34.0% at 2M,
  51.0% at 5M, 61.8% at 7M, 63.0% at 10M, flat from about 7M. Over 1,000 games the 10M model
  (`models/ppo_pokemon_6v6_v19_best`) wins 60.7% ± 3.0% vs SimpleHeuristics and about 44% ± 2.2%
  head-to-head vs v18.3 (2,000 games, both seats). It is the from-scratch baseline for step 3;
  v18.3 stays the best model.
- Step 1 below is done. The opponent-remaining count is fixed, the extractor encodes all 12 Pokémon
  in one call (same weights, so every v18 checkpoint still loads), and `PPOPlayer` uses the
  training mask.
- v18.3_best predates the fixes. On the current env it wins 67.9% ± 2.9% of 1,000 games vs
  SimpleHeuristics, against 70.4% ± 2.8% before the opponent-count fix: no measurable change,
  though its games got shorter (34 vs 43 decisions). Gains have to come from retraining.

## Known limits of the v18 architecture
1. Switches are chosen blind. The bench is max+mean pooled, so the only per-slot information is
   10 type-effectiveness floats in `_encode_global`. The policy can't see which switch slot holds
   which HP, status or species, forced switches included. Wang kept per-Pokémon vectors in place
   and keyed switches by species; Jaxcalibur scores each switch from that Pokémon's token.
2. Moves are learned per slot. All 13 logits come from one linear layer, and Showdown shuffles
   move order when it builds sets (`../pokemon-showdown/data/random-battles/gen9/teams.ts:1586`),
   so every move has to be learned in all 4 slots.
3. Policy and value have separate extractors, so value learning (or any auxiliary loss) never
   shapes the policy's representation. Wang and Jaxcalibur share a trunk.
4. Missing inputs:
   - Own tera type. Showdown sends `teraType` for every Pokémon in each request, but poke-env's
     `Pokemon.update_from_request` drops it, so `tera_type` is None until you terastallize. Read
     it from `battle.last_request`.
   - Level (64-100 in gen9 randbats) and actual stats. Ours are in `pokemon.stats`; the
     opponent's follow from base stats and level, since randbats uses 85 EVs and 31 IVs (0 Atk
     for special attackers, 0 Spe for Trick Room sets).
   - The opponent's possible sets: 1-3 per species in
     `../pokemon-showdown/data/random-battles/gen9/sets.json`.
   - Damage estimates (`poke_env.calc.damage_calc_gen9`), turn counters (weather, terrain,
     screens, Trick Room, Tailwind, sleep, Protect), last move used, and a distinct marker for
     unrevealed opponents (today they are all-zero blocks).
5. The heuristic masks block legal, often good moves. They judge immunity by the move's base type
   and block the tera version too, which is wrong for Tera Blast (37 of 869 gen9 sets), -ate
   abilities, Scrappy/Mind's Eye, and Mold Breaker/Teravolt/Turboblaze into Levitate (13 species).
6. No memory beyond hand-made history features, and those are rough: the opponent-history "move
   type" is the type of the last revealed move, not the last one used.
7. No auxiliary prediction heads.

## Throughput (measured September 2026 on this MacBook Air M3)
- Websockets are not the cap. poke-env with one local Showdown server runs ~900 decisions/s one
  game at a time and ~2,200/s with concurrent games (SimpleHeuristics vs itself).
- One training env against a v18.3 opponent: 160-220 steps/s before step 1, when about half of
  each step was batch-1 inference for the learner and the opponent. Batching the encoder made
  that inference 5.7x faster and env steps 1.3-1.7x faster (alternating runs); PPO minibatches
  didn't get faster.
- 4 envs in parallel: 250-500 steps/s combined before step 1, 590-750 after (alternating runs).
  Four env processes plus their Showdown servers fill all 8 cores (4P+4E), and the fanless Air
  throttles under sustained load (`kernel_task` climbs), so more envs won't help on this machine.
  Compare speeds with alternating back-to-back runs.
- Cloud (t2d-standard-32, 20 envs): ~540 steps/s collecting, only about 1.2x this Air, because
  SubprocVecEnv steps every env together and waits for the slowest. Env processes need one torch
  thread each (the training script sets this): with the default one-per-core, 20 envs pushed a
  32-core VM's load average past 360.
- Over the 10M run, updates slowed from ~95 s to ~159 s. Games got shorter (51 → 26 decisions,
  so more resets), and the old 3-model opponent cache missed more as the pool grew: 22% → 44% of
  resets reloaded a model (~90 ms), and each reload stalls every env. `opponent_cache_size` now
  keeps the whole pool loaded (~19 MB per model per env); on this Air one late-run update went
  from ~146 s to ~107 s.
- Self-play games average ~39 decisions per side; SimpleHeuristics vs itself, ~24.
- A 3-layer, d=128 transformer trunk over 25 tokens takes 0.23 ms at batch 1 on one CPU thread.

## Planned steps, in order
1. Done: fix the opponent-remaining count, batch the Pokémon encoder, drop `PPOPlayer`'s tera
   block.
2. Done: v19 trained from scratch to 10M steps as the clean baseline (results above). Compare step 3
   against its eval curve at equal steps, with the same 20M learning-rate schedule.
3. Done (September 2026; `v20.md`, brief in `step3_handoff.md`). Replace SB3 with a small CleanRL-style PPO loop
   (~300 lines), trained from scratch, with:
   per-token action heads (each bench Pokémon scores its own switch; one shared scorer gives each
   move a move logit and a tera+move logit), a shared trunk, two-sided self-play (record both
   players when both run the current policy, as Wang did), legality-only masks plus Jaxcalibur's
   zero-avoiding term, and the missing inputs.
   Then (September 30-October 4): A/Bs of the learning rate (x3), the value clip (off), the
   entropy bonus (0.02) and `gae_lambda` (0.95), and v20 extended to 20M. None raised v20's level;
   only training longer helped, a little (`v20.md`). The v20 tokens are the input step 4's
   attention trunk needs.
4. Done (October 2026; `v21.md`, brief in `step4_brief.md`), no gain. Entity-token transformer
   (3 layers, d=128) over the field, 6+6 Pokémon, 4 moves and the last 16 events from
   `battle.observations`, with auxiliary heads for the opponent's next action and its hidden item,
   ability, moves and tera type (labels free from the other side's battle object). Level with v20
   at 10M (48.1% ± 2.2% head-to-head) and at 20M (50.9% ± 2.2%); each PPO update ~60-95 s on the
   Mac instead of 7-12 s. A/Bs of v21 at 5M: v20's context vector in the heads +3 points vs v21
   (level with v20), no aux losses level (after a 1.2M-step stall), half the LR clearly worse
   (43% vs v21) despite a v20-like start. Nothing got past v20.
5. Scale out on a many-core cloud machine (~2 cores per env), with the PPO update on a GPU.
6. Done (October 9-10, 2026; `step6.md`; chosen over step 5, brief in `step6_brief.md`): search at
   decision time. Phase 1, a Foul Play-style bot of our own on poke-engine (MIT): our converter
   from poke-env battles to engine states, opponent sets sampled from the fork's own generator,
   MCTS per sample. Phase 2, the network in the loop: policy prior, v21's aux heads for the hidden
   sets, its next-action head as the opponent model. Search added about 100-150 Elo on top of
   Jaxcalibur's network; Foul Play's search alone is ~300 Elo above v20. Result: Phase 1 is level
   with Foul Play (47.4%) and wins 80% vs v20; Phase 2 adds no measurable gain against Foul Play.
7. Done (October 10-11, 2026; `step7.md`, brief in `step7_brief.md`): learning from the search.
   Search self-play data, distillation into v21 (~2 points for the network alone, a loss inside the
   search), a learned leaf evaluation in the engine (+3.6 points vs step 6's bot, level with Foul
   Play), one more round of the AlphaZero loop (no gain with a linear model).
8. Done (October 11-12, 2026; `step8.md`), no gain: a matchup-aware leaf evaluation (a small
   network sets per-Pokémon weights once per search, the leaves stay linear). Better offline
   prediction, a weaker (full weight) or level (0.3) search.
   Then trained on the search's own per-option values (`step8.md` section 4): better one-turn
   agreement with the search, no gain in play.
9. Next (`step8.md` section 5): the cheap items, a finer sweep of the leaf scale (1.2-1.6), the
   exploration constant, the ladder; or, to keep at the evaluation, TreeStrap (train toward the
   values of every searched node, which needs a node dump from the engine).

Two Wang settings tuned on 3v3 battles, which are half as long, were A/B'd on v20: `clip_range_vf`
(None only gave a faster start) and `gae_lambda` (0.95 was clearly worse). Results in `v20.md`.

Fold the cleanup into step 3, since it needs a from-scratch run anyway: remove the hand-made
features (oscillation detector, "opponent seems passive", game-phase flags, hand-picked item and
ability flags), the heuristic masks, the shaping code once sparse reward has proven itself, v17
checkpoint support, the embedding-LR split and the monkeypatched LR scheduler.

## References
- Wang (2024), MIT thesis, DSpace 1721.1/153888, Gen 4 randbats: per-Pokémon 896-d vectors
  concatenated into a 3-layer MLP, one head over ~495 actions keyed by move and species ID, a shared
  policy/value trunk, both players' trajectories recorded, and hyperparameters tuned on 3v3 battles.
  The raw policy won 78.6% vs SimpleHeuristics after about 3M battles, 90.8% with MCTS; the agent
  peaked at 1693 Elo (#8) on the gen4 ladder.
- Jaxcalibur (https://jaxcalibur.github.io/ and Smogon thread 3787537): 8.5M-parameter non-causal
  transformer over 66 tokens, per-token action logits, auxiliary heads (the opponent's next action
  "made the bot substantially stronger"; no efficiency number is given), PPO with an undiscounted
  1/0 reward, entropy plus a zero-avoiding term, about 100M self-play games in a week on one H100
  with a JAX engine. Peaked at 2557 Elo; search added about 100-150, and the author guesses ~2400
  without it. Smaller networks trained longer beat larger ones.
- Foul Play (https://github.com/pmariglia/foul-play): MCTS over poke-engine with set sampling.
  Peaked at about 2341 Elo.
