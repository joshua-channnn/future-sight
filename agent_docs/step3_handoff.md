# Step 3 handoff: custom PPO loop with per-token action heads (v20)

For the agent that builds step 3 of `roadmap.md`. Written September 28, 2026, at the end of the
session that reviewed the architecture, fixed the v19 bugs, trained v19 on Google Cloud and
measured it. Everything here was measured or checked in that session unless it says otherwise.

## 0. Read before starting
1. This file.
2. `CLAUDE.md`: how to run things, ports, and the rules for counting wins and noise.
3. `agent_docs/poke_env_gotchas.md`: eight silent bugs this project has hit. Keep them fixed.
4. `agent_docs/roadmap.md`: the review's diagnosis, known limits, throughput numbers, the plan,
   and what Wang (2024) and Jaxcalibur did.
5. When you get to them: `training.md`, `evaluation.md`, `cloud.md`.

Before writing code, confirm the open questions in section 9 with the user.

## 1. Where things stand
- **Git:** branch `v19-baseline`, 2 commits ahead of `main` (`acf4104` v19 fixes and docs,
  `7e61858` cloud scripts). Not merged, not pushed. Work on a new branch off it (`main` is the
  default branch; don't commit there).
- **Best bot: v18.3** (`models/ppo_pokemon_6v6_v18.3_best.zip`; it and
  `ppo_pokemon_6v6_v18.2.zip` are the only models tracked in git).
  End of a chain of warm-started runs, about 210M steps in total.
- **Baseline for step 3: v19**, the v18 architecture trained from scratch with the fixed env,
  sparse reward and Wang's PPO settings. Stopped on purpose at 10M steps. Its files are in
  `models/` (ignored by git; about 200 MB each, over GitHub's 100 MB limit):
  `ppo_pokemon_6v6_v19_best.zip` (the 10M model), `_latest.zip`, `training_state_v19.pkl`,
  `optimizer_state_v19.pt`, and `models/selfplay_pool_v19/` (10 checkpoints at 1M..10M plus the
  5 seed models). TensorBoard logs and `train_v19.log` are in `logs/`.
- **Cloud:** the VM is deleted and the GCP project `future-sight-training` has no resources. The
  gcloud CLI is installed on the Mac and logged in. `cloud/` holds the working scripts.

| Model | vs SimpleHeuristics | vs Random | Head-to-head |
|---|---|---|---|
| v18.3 | 67.9% ± 2.9% (1,000) | 94.7% ± 1.4% | 60.4% ± 3.0% vs v17.3sp |
| v19 (10M) | 60.7% ± 3.0% (1,000) | not run | about 44% ± 2.2% vs v18.3 (2,000 games, both seats) |

v19's in-training evals (500 games each, about ±4.3 points, deterministic, vs SimpleHeuristics):

| Steps | 1M | 2M | 3M | 4M | 5M | 6M | 7M | 8M | 9M | 10M |
|---|---|---|---|---|---|---|---|---|---|---|
| Win rate | 13.4% | 34.0% | 43.2% | 48.6% | 51.0% | 57.0% | 61.8% | 59.6% | 60.4% | 63.0% |

v19 training diagnostics (TensorBoard, `logs/MaskablePPO_1/`):
- `approx_kl` stayed at 0.0013-0.0018 through 5M and fell to 0.0008 by 10M, far below the ~0.01
  that `training.md` aims for. The policy moved slowly all run. Clip fraction 0.09-0.12 early,
  0.04 at 10M.
- Explained variance reached 0.86 by 2M and held about 0.81: the value function was fine
  despite `clip_range_vf=0.0184`.
- Learning rate 4.6e-5 at 0.5M, 5.4e-6 at 10M (Wang's schedule over a 20M horizon).
- Episode length fell from 51 to 26 decisions as the policy stopped playing randomly.

## 2. Goal and how to judge it
- **Goal:** a v20 architecture and training loop that learns clearly faster or higher than v19 at
  equal training steps. That is the route to beating v18.3; more training of the v18 design is not.
- **Compare like with like:**
  - Same in-training eval: 500 deterministic games vs SimpleHeuristics every 1M steps.
  - Same learning-rate schedule horizon: Wang's schedule over 20M, compared at 1M..10M.
  - Keep v19's PPO hyperparameters for the first v20 run, so gains can be attributed to the
    architecture and loop. A/B hyperparameters afterwards.
  - x-axis = learner samples (timesteps). With two-sided self-play one game gives two sides'
    samples; report games played too.
- **Noise:** each 500-game point is about ±4.3 points, so single points differing by less than about
  6 mean nothing. Look for a curve that is above v19's at several points.
- **Final checks:** 1,000 games vs SimpleHeuristics, and head-to-heads vs v19 and vs v18.3 with
  1,000 games in each seat.

## 3. What step 3 is, and why
The review's diagnosis of the v18 design (details and evidence in `roadmap.md`):
1. **Switches are chosen blind.** The bench is max+mean pooled; the only per-slot information
   the action layer gets is 10 type-effectiveness floats. It can't see which switch slot has
   which HP, status or species, forced switches included.
2. **Moves are learned per slot.** All 13 logits come from one linear layer, and Showdown shuffles
   move order when it builds sets (`../pokemon-showdown/data/random-battles/gen9/teams.ts:1586`),
   so each move must be learned in all 4 slots.
3. **Separate policy and value extractors**, so value learning never shapes the policy's features.
4. **Missing inputs:** own tera type, level (64-100 in gen9 randbats), actual stats, an
   unrevealed-opponent marker, turn counters, last move, damage estimates.
5. **Heuristic masks block good moves** (Tera Blast, -ate abilities, Scrappy/Mind's Eye, Mold
   Breaker into Levitate).
6. **SB3 lockstep caps throughput:** SubprocVecEnv steps every env together and waits for the
   slowest, so 20 envs on 32 cores collected only about 540 steps/s.
7. **SB3 can't easily do** per-token heads, auxiliary losses or two-sided data.

**Step 3 (v20) scope, as recommended:**
- A new CleanRL-style PPO loop that replaces SB3.
- Per-token action heads: each own-team Pokémon scores its own switch, and one shared scorer
  gives each move a "use" and a "tera + use" logit.
- One shared trunk for policy and value.
- The missing inputs, legality-only masks, and Jaxcalibur's zero-avoiding term.
- Collection without per-step lockstep.
- Two-sided self-play: record both players when both run the current policy, as Wang did.

**Not in step 3** (that's step 4 and later): the transformer, history tokens, auxiliary heads,
search. Build the inputs as tokens now so step 4 can add attention without redoing the env.

## 4. Recommended design
These are recommendations; section 9 lists what to confirm with the user.

### 4.1 Observation: a new env class (e.g. `envs/rl_player_v20.py`)
Keep `envs/rl_player_6v6_v18.py` and `networks/embedding_extractor_v18.py` unchanged: v18/v19
checkpoints need them, both for evaluation and as self-play opponents. Subclass poke-env's
`PokeEnv` as `RL6v6PlayerV18` does, and reuse its helpers where they fit (`switch_targets`,
`_opponent_remaining`, `_is_discarded_perspective`).

Token groups (index fields feed embeddings; keep them separate from floats):
- **Field (1 token):** weather, terrain, hazards on both sides, screens, Trick Room, Tailwind,
  the turn number, and tera availability for both sides. poke-env's `battle.weather`,
  `battle.fields` and the side-condition dicts hold start turns or layer counts; check which
  before deriving turns remaining.
- **Own team (6 tokens, in `battle.team` order):**
  - Identity: species, types (current and tera type), level, item, ability, 4 moves each with
    type, category, base power and PP.
  - State: actual stats from `pokemon.stats`, HP fraction, status with sleep and toxic
    counters, boosts, volatiles, active and fainted flags.
  - **Own tera type:** Showdown sends `teraType` for every Pokémon in each request, but poke-env's
    `Pokemon.update_from_request` drops it. Read it from
    `battle.last_request["side"]["pokemon"][i]["teraType"]`, matched on `ident`.
- **Opponent team (6 tokens):** revealed Pokémon, plus an explicit "unrevealed" token rather
  than today's all-zero block. Per revealed Pokémon: species, level, revealed moves, item,
  ability, HP, status, boosts, and tera state and type.
  - **Stats are computable:** gen9 randbats uses 85 EVs and 31 IVs (`teams.ts:1502-1503`), 0 Atk
    EV/IV for sets without physical attacks and 0 Spe for Trick Room / Gyro Ball sets
    (`teams.ts:1566-1578`), and no nature (so neutral). `poke_env.stats.compute_raw_stats(species, evs, ivs,
    level, nature, data)` does the arithmetic.
  - **Possible sets:** `../pokemon-showdown/data/random-battles/gen9/sets.json` has, per species,
    `level` and 1-3 sets of `{role, movepool, abilities, teraTypes}`. Candidate moves and tera
    types are an optional feature; items are chosen in code, not listed there.
- **Active moves (4 tokens):** token i is `battle.available_moves[i]`, matching the current
  action mapping. It is not `pokemon.moves` order: a Choice-locked Pokémon has one available
  move, and Struggle and recharge show up there too. Features: move id, type, category, base
  power, accuracy, priority, PP fraction, type effectiveness vs the opponent's active, STAB.
  `poke_env.calc.damage_calc_gen9.calculate_damage(attacker_identifier, defender_identifier,
  move, battle)` returns a damage range; check how it treats the opponent's unknown stats and
  items before relying on it.
- **Later (step 4), for history:** `battle.observations[turn].events` holds each turn's raw
  events (moves, switches).

Check every new feature against live battles before training. Last session's method: play
SimpleHeuristics vs SimpleHeuristics on a local server, recompute each feature from the battle at
every decision, and count mismatches against ground truth. In self-play, the opponent's
`battle2` has its true stats, items and tera types, which makes opponent estimates checkable
exactly.

### 4.2 Actions: 14, with switches keyed by team slot
- 4 moves, 4 tera moves, and **6 switches, one per `battle.team` slot**, masked when active,
  fainted or not in `battle.available_switches`. Each own-team token then scores its own switch
  action, with no shifting index.
  - The current 13-action scheme uses "k-th switch target" (team order minus active and
    fainted), which shifts as Pokémon faint.
  - Never index `available_switches` with an action: it follows Showdown's request order, which
    reshuffles (gotcha 2).
- **Revival Blessing:** when `battle.reviving`, the legal targets are fainted team members.
- **Legality-only masks:** a move slot is legal if it exists in `available_moves`; a tera slot
  also needs `battle.can_tera`; a switch needs its Pokémon in `available_switches`. Drop the
  immunity and redundant-status heuristics.
- This changes the observation and action meaning, so per `CLAUDE.md` bump the version (v20) and
  give the new model its own loading path.

### 4.3 Network (e.g. `networks/v20_policy.py`)
- **Embeddings:** species, move, ability, item and type tables, shared across token types.
  Vocabulary files are in `data/` (508 species, 351 moves, 204 abilities, 63 items).
- **Encoders:** per token type (Pokémon with a side flag, move, field), each a small MLP.
  Batch all tokens of a type in one call: 12 separate calls made the v18 extractor about 6x
  slower at batch 1.
- **Context vector:** the field encoding plus own-active and opponent-active encodings, plus
  mean and max over each team's encodings, through an MLP.
- **Heads:**
  - Move head: MLP([move_i, context]) gives 2 logits per move (use, tera + use).
  - Switch head: MLP([own_k, opponent_active, context]) gives 1 logit per team slot.
  - Value head: MLP(context).
  - One trunk, one optimizer. The 10x embedding learning rate is a legacy knob, optional.
- **Size:** 0.5-1.5M parameters is plenty; Jaxcalibur found smaller nets trained longer beat
  bigger ones. Target under 0.5 ms at batch 1 on one CPU thread (the batched v18 extractor is
  0.17 ms), because self-play opponents run batch-1 inference on every step.
- **Initialization:** CleanRL-style orthogonal init, with 0.01 gain on the policy logits.

### 4.4 Training loop (e.g. `training/ppo_v20.py`)
**Collection without per-step lockstep.**
- W worker processes (forkserver; one torch thread each). Each owns one or more envs on its own
  Showdown port and a CPU copy of the current weights, and collects T steps per update.
- The learner broadcasts weights at the start of each update and waits for all workers only at
  the update boundary. A slow reset or opponent load then delays only its own worker.
  - In the SB3 loop, each ~90 ms opponent reload and ~25 ms battle creation stalled all 20 envs.

**Two-sided self-play: use `PokeEnv`'s two-agent API directly, not `SingleAgentWrapper`.**
- `reset()` and `step({agent: action})` return per-agent dicts; `agent1_to_move` and
  `agent2_to_move` say who actually has to decide.
- Record a transition for a side only when that side decided (gotcha 3). Reward is ±1 at the end
  from each side's own perspective; compute GAE per side.
- When the opponent is the current policy, record both sides. When it is a pool snapshot or
  SimpleHeuristics, record only the learner.

**PPO**, starting from v19's `CONFIG` in `training/ppo_embed_train_selfplay.py`:
- gamma 0.9999, lambda 0.754, clip 0.0829, value clip 0.0184, 7 epochs, minibatch 1024.
- About 40,960 steps per update, max grad norm 0.543, entropy 0.0588, value coefficient 0.4375.
- Wang's LR schedule `10^-4.23 / (8x+1)^1.5` over a 20M horizon.
- Add the zero-avoiding term, a small KL toward uniform over legal actions; Jaxcalibur gives no
  coefficient, so pick a small one and log it.
- Given v19's approx_kl of ~0.001-0.002, a higher learning rate is the first A/B after the
  like-for-like run. `roadmap.md` also lists `clip_range_vf` (None vs 0.0184) and `gae_lambda`
  (0.754 vs 0.9-0.95).

**Opponents** (as in v19):
- Mix: 10% SimpleHeuristics; the rest from the pool, with 50% of those the newest opponent and
  the rest uniform over the pool.
- Pool: a snapshot every 1M steps, at most 30, seeded with the 5 old best models.
- Keep every pool model loaded in each worker (LRU). Measure memory; a v18-type model is about
  19 MB.
- For two-sided data, "newest" should mean the live weights.

**Checkpoints and resume.**
- Save `torch.save({model, optimizer, config, step, eval_state})`. That's a few MB, not 200: SB3
  saves were bloated because the monkeypatched `_update_learning_rate` pickled the whole model
  and its rollout buffer.
- Resume from the latest checkpoint with the same LR horizon, and save every 200K steps.

**Eval:** every 1M steps, 500 deterministic games vs SimpleHeuristics.
- Create a fresh env for each evaluation and close it right after the last game. A reused eval
  env hung v19's training for 3 hours at the second eval.
- Close with `force=True, wait=False` so the close itself can't block.

**Logging** (keep this compatible with `cloud/run_training.sh`, or update the script):
- TensorBoard with the same tags where they apply: `eval/win_rate`, `train/approx_kl`,
  `train/clip_fraction`, `train/explained_variance`, `rollout/ep_len_mean`, `time/fps`.
- One stdout line per update containing `total_timesteps`; the stall guard counts these.
- Eval lines formatted `>>> Eval @ 1,000,000: 13.4% (67/500)`; `STOP_AT` parses these.

**Watchdogs:** a local test once saw a single Showdown `/challenge` take 15 minutes (cause
unknown). Give each env a timeout: if a battle makes no progress in about 60 s, abandon it and
rebuild that env.

### 4.5 Player and eval integration
- `players/ppo_player.py`, `eval/ppo_test.py` and self-play loading assume SB3 `MaskablePPO`
  zips, with `ARCH_REGISTRY` keyed on the final MLP's input width (444, 700, 725).
- v20 needs its own player (e.g. `players/v20_player.py`) with sync `choose_move` (as an opponent
  inside env steps) and async `choose_move` (live play and ladder); see
  `PPOPlayer.choose_move` for how it dispatches.
- Give `eval/ppo_test.py` a path for v20 checkpoints (detect by file format), keeping v18/v19
  evaluation working.
- v20's pool must be able to hold v18/v19 SB3 opponents alongside v20 snapshots, including the
  5 seed models.

## 5. Suggested milestones, each with a check
1. **Env v20.** Token observation and 14-action mapping.
   - Check: features against live battles (mismatch counts must be zero).
   - Check: every legal Showdown choice maps to exactly one unmasked action and back.
   - Check: `action_to_order` sends the intended order.
2. **Network.** Forward shapes, masking (masked logits get zero probability), and batch-1 latency
   under 0.5 ms.
3. **Loop against SimpleHeuristics only**, 1-2M steps on the Mac.
   - Check: win rate rises clearly. For scale, v19 with self-play was at 34% by 2M.
   - Check: approx_kl, entropy and explained variance look sane.
   - Check: steps/s vs the SB3 loop, measured with alternating back-to-back runs.
4. **Self-play:** two-sided data, the pool (including loading v18/v19 SB3 opponents),
   checkpoints and resume. Check: kill mid-run and resume.
5. **Eval integration:** the v20 player plus `eval/ppo_test.py`. Check: evaluate a v20 checkpoint
   vs SimpleHeuristics and vs v19.
6. **The 10M comparison run** (local or cloud, as the user decides; see `cloud.md`), then the
   final checks in section 2.

## 6. Reference numbers
**Machines**
- **The Mac:** MacBook Air M3, 16 GB RAM, 8 cores (4 performance, 4 efficiency), fanless. It
  throttles under sustained load (watch `kernel_task`) and identical runs varied up to 2x. Compare
  speeds only with alternating back-to-back runs.
- **Cloud VM:** `t2d-standard-32` at about $1.35/hour on-demand (Spot about $0.81, and it was
  preempted after 2 minutes).
  - 20 envs collected about 540 steps/s in the SB3 loop: only about 1.2x the Mac, so lockstep
    is the ceiling.
  - Each env process needs one torch thread: with the default one-per-core, 20 envs pushed the
    load average to 363.

**Per-operation costs (Mac unless noted)**

| Operation | Cost |
|---|---|
| poke-env + one Showdown server, one game at a time | ~900 decisions/s |
| same, with concurrent games | ~2,200 decisions/s |
| One SB3 training env step | 2-5 ms |
| Battle reset | 14-25 ms |
| Loading a self-play opponent | ~90 ms |
| Batched v18 extractor at batch 1 | 0.17 ms |
| 3-layer, d=128 transformer trunk, 25 tokens, batch 1 | 0.23 ms |

**SB3 loop on the Mac (4 envs)**
- After this session's speedups: about 490 steps/s collecting.
- A late-run update took ~107 s with the full opponent cache, versus ~146 s with the old
  3-model cache.

**Costs so far**
- The v19 cloud run cost about $19.70 in total. The user's GCP credit is $10/month, and they pay
  the rest.
- 10M steps took about 9 hours of VM time in the SB3 loop (1.4 hours to 2M, 7.5 from 2M to 10M,
  as updates slowed), about $12. The rest of the $19.70 was setup, the aborted Spot start, and
  3 hours hung at the old eval bug.

## 7. Pitfalls already hit (don't repeat them)
**Training correctness**
- Everything in `poke_env_gotchas.md`: the perspective bug, `available_switches` reshuffling,
  observations with nothing to decide, DummyVecEnv auto-reset, dropout skewing PPO ratios, two
  code paths building observations, `battle.won` being None for ties, and `opponent_team`
  holding only revealed Pokémon.
- Count wins with `battle.won`, never the reward sign.

**Processes and threads**
- On Linux, forked children lose poke-env's event-loop thread and hang: use `forkserver`.
- Torch defaults to one thread per core in every process. Pin env processes to one thread; give
  the learner all threads only for the update.

**Eval and opponents**
- A battle left idle between evals can no longer be forfeited, and the reset's forfeit then
  waits forever. Never leave battles idle.
- A small FIFO opponent cache reloads constantly as the pool grows, and in lockstep every reload
  stalls all envs.

**Showdown servers**
- Several servers run from one Showdown directory race on shared files
  (`config/chat-plugins/*.json`, `logs/lastbattle.txt`). They log `CRASH: ENOENT`, and
  `logs/errors.txt` is 4.9 MB of it. So far this has been harmless.
- `pokemon-showdown start` rebuilds every time. Build once (`node build`) before starting many
  servers on a fresh tree.

**Tooling**
- macOS `tar` adds `._*` files unless `COPYFILE_DISABLE=1`; they broke the Showdown build on the VM.
- Python puts a script's own folder on `sys.path`, not the working directory. Run helper scripts
  from `pokemon-rl/` with `PYTHONPATH=.`, or as modules.
- `pkill -f pattern` inside an SSH command can kill the SSH session itself. Use `'[p]attern'`.
- `gcloud compute scp` dropped a 2.4 GB fetch partway. Use `cloud/fetch_results.sh` (rsync).

**Sessions and scheduling**
- In this Claude Code background-job session, recurring `CronCreate` checks didn't fire on time;
  they ran late or all at once.
- What worked for hourly checks: a `run_in_background` Bash command that sleeps until :07, checks
  the VM read-only, then exits (its exit wakes the agent), re-armed after each report.
- A boot-time auto-restart hook on the VM was refused by the permission system as persistence.
  Don't retry it.

## 8. Working with this user
- **Solo project**, short messages ("ok", "yes"). Give a clear recommendation and act on it.
- **Money needs their decision:** confirm before any cloud spend, and report the running cost.
  They picked the cloud for long runs and capped v19 at 10M to save money.
- **Long runs:** they want hourly progress checks during long runs (see section 7 for what worked).
- **Destructive or public actions:** ask before deleting data or VMs and before pushing. Work on
  branches, never commit to `main`, and end commit messages with the `Co-Authored-By` line from
  the session's instructions.
- **Security:** their Showdown ladder password is in `players/ladder_bot.py` and in git history
  (`bb8b58d`); they've been told. Don't copy credentials into docs or logs.

## 9. Confirm with the user before building
1. **Scope:** v20 = new loop + per-token heads + new inputs + two-sided self-play + no lockstep, as
   recommended. The alternative is a smaller first step, e.g. per-token heads in the existing
   SB3 loop through a custom policy class, which gives a cleaner comparison but none of the
   throughput or two-sided gains.
2. **Action space:** 13 → 14 (switch by team slot).
3. **Where the 10M comparison runs:** the Mac (free; the machine is busy for most of a day) or
   GCP (about $10-15, which they pay beyond the $10 credit). Say which is faster once the new
   loop's steps/s is measured.
4. **Hyperparameters:** keep v19's for the first run (recommended, so the comparison isolates the
   architecture), with A/Bs afterwards.
