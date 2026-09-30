# Handoff after step 3 (v20): what to do next

Written September 30, 2026, at the end of the session that built and trained v20. Everything here
was measured in that session unless it says otherwise. Read `v20.md` first (design, checks,
results), then `CLAUDE.md` and `poke_env_gotchas.md` (items 9-15 are new).

## 1. Where things stand
- **Git:** branch `v20`, off `v19-baseline` (neither merged nor pushed). The v20 work is committed
  there at the end of the session.
- **Best model: v20** at 10M steps, `models/ppo_pokemon_6v6_v20.pt` (4.3 MB torch file; tracked in
  git like v18.3). 83.7% ± 2.3% vs SimpleHeuristics (1,000 games), 79.3% ± 1.8% vs v19 and
  70.1% ± 2.0% vs v18.3 (2,000 games each, both seats), 99.8% vs RandomPlayer.
- **Run artifacts** (gitignored): `models/ppo_pokemon_6v6_v20_latest.pt` (resumable, 10.03M steps,
  optimizer state), `models/eval_snapshots_v20/` (1M..10M), `models/selfplay_pool_v20/` (10 v20
  snapshots + 5 SB3 seeds), `logs/v20/` (TensorBoard), `logs/train_v20.log`,
  `logs/final_checks_step_0010034773/`.
- In-training evals (500 games vs SimpleHeuristics): 54, 68, 73, 75, 81, 81, 83, 82, 84, 83% at
  1..10M, vs v19's 13, 34, 43, 49, 51, 57, 62, 60, 60, 63%.

## 2. Recommended next steps, in order
1. **Learning-rate A/B.** The like-for-like run kept v19's settings, and approx_kl was 0.0018 at 1M
   falling to 0.0004 at 10M (the ~0.01 target in `training.md` is 5-25x higher); clip fraction
   0.13 → 0.03. Try `lr_initial` x3 and x10 over the same 20M horizon, from scratch, compared at
   equal steps with the v20 curve. Change one thing per run.
2. Then `clip_range_vf` (0.0184 vs None; explained variance sat at ~0.81), `gae_lambda` (0.754 vs
   0.9-0.95), `zero_avoid_coef` (0.005 vs 0 and 0.02), `embed_lr_mult` (10 vs 1).
3. **Train longer.** The 10M curve was still inching up (81-84% from 5M on). A run to 20M (the
   schedule's end) or a fresh one with a longer horizon is cheap on the Mac: ~3.5-4 hours per 10M
   when plugged in.
4. **Step 4** (`roadmap.md`): an attention trunk over the v20 tokens (12 Pokémon + 4 moves + field
   already come as separate token arrays) plus history tokens from `battle.observations[t].events`
   and auxiliary heads. Two-sided self-play gives free labels: the opponent's actual next action and
   its hidden item/ability/moves/tera type are in `env.battle2`. Keep the batch-1 cost in mind: the
   featurizer is 0.26 ms and the v20 net 0.27 ms per decision; a 3-layer d=128 transformer over 25
   tokens measured 0.23 ms.
5. Possible features not in v20: the opponent's candidate sets from `data/gen9_randbats_sets.json`
   (roles, possible tera types and moves, filtered by revealed moves), damage estimates for the
   opponent's moves against our bench (poke-env's calc is ~30 µs per call), opponent PP.

## 3. Things that matter on this machine
- **Memory, not CPU, is the Mac's limit.** Each Showdown server is ~10 node processes and ~1.2 GB;
  stop unused ones (`kill` the launcher's process tree; they share a process group with the others,
  so don't kill by group). With other apps open the Air was at 16-20 GB of swap all session.
- **MPS caches a graph per tensor shape (~57 MB each).** Anything that sends a new shape to the GPU
  every update leaks: the learner grew to 7.6 GB before the fixed-size-minibatch fix. Keep device
  shapes fixed (gather on the CPU, fixed batch sizes); watch `footprint <pid>`.
- **Battery:** the Air drained ~1% per minute under training. `tools/run_v20_mac.sh` pauses at 20%
  on battery and resumes on power; it stops for good when the trainer exits by itself (so to restart
  the learner by hand, kill the runner first, then SIGTERM the trainer, then start a new runner).
- Throughput with 6 workers on 3 servers: 900-1,280 samples/s collecting early in a run, 550-900
  later (shorter games, more v20 pool opponents); PPO updates 7-10 s on MPS (2.7x the CPU). The SB3
  loop managed 274-427 on the same machine. The machine drifts; compare speeds with alternating runs.
- The cloud scripts (`cloud/run_training.sh`, `fetch_results.sh`) handle v20 now but were not run
  with it. On a VM without a GPU the update runs on the CPU (~17 s per update on the Mac's CPU).

## 4. Checks to rerun after changes
- `python3 -m tools.check_v20_env --games 50 --port 8123`: features vs ground truth (the other
  side's battle and the request), legal-order ↔ action mapping, orders executed. Outside Illusion
  games expect 0-1 mismatches (ability changes like Skill Swap); Illusion games are reported apart.
- `python3 -m tools.check_v20_selfplay --port 8123` and `PORT=8123 bash tools/check_v20_resume.sh`.
- A featurizer change that should not change outputs: compare against the previous version on live
  battles (the session did this for the speed-up: 2,687 decisions, 0 differing).
- Changing the observation or action layout: checkpoints store `LAYOUT` and refuse to load with a
  different one, so bump the version and keep the old featurizer importable if old models must
  still play (e.g. as pool opponents).

## 5. Pitfalls hit in step 3 (not to repeat)
- The fork's request-before-log order (gotcha 9): without `hold_requests_until_log`, own HP was
  stale in ~0.03% of decisions, more under load.
- Opponent Atk estimate: Showdown zeroes Atk from the 4 chosen moves (fixed-damage moves like
  Seismic Toss don't count; Porygon2's Tera Blast does).
- Editing `training/ppo_v20.py` or the env while a run is live: eval processes and resumed
  segments import the code on disk fresh.
- `pkill -f` patterns can hit a running job's processes; target PIDs.
- A background-job session can't sleep in the foreground; use `run_in_background` waits that exit
  on a condition (e.g. the next `>>> Eval @` line).

## 6. Working with this user
- Solo project, short messages. Give a clear recommendation and act on it.
- Money needs their decision: confirm before any cloud spend. When they say they're away and not to
  prompt, run on the Mac (free) rather than the cloud.
- They want progress checks during long runs.
- Git: work on branches, never commit to `main`, don't push without asking. They asked not to
  commit along the way: commit at the very end of a task. End commit messages with the session's
  `Co-Authored-By` line.
- Their Showdown ladder password is in `players/ladder_bot.py` and git history; don't copy it.
