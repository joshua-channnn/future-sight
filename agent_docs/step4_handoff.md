# Handoff after step 3 (v20): the learning-rate run, then what follows

For the agent that picks this up in a new chat. Its first job is the **3x learning-rate run**
(section 2). Written September 30, 2026, at the end of the session that built, trained and
evaluated v20. Everything here was measured in that session unless it says otherwise.

## 0. Start here
1. Read this file, then `CLAUDE.md`. Skim `v20.md` (design, checks, results) and items 9-15 of
   `poke_env_gotchas.md`. You don't need to read the step 3 brief (`step3_handoff.md`).
2. The user has decided on this run and asked for it in the new chat: no need to re-confirm the
   plan. It runs on the MacBook (free); don't use the cloud without their say-so.
3. Check the Mac is on power first (`pmset -g batt`): training drains the battery ~1% per minute;
   the runner pauses at 20% and resumes on power.
4. Start it (section 2), monitor it (section 3), judge it (section 4), report to the user as the
   1M evals come in.

## 1. Where things stand
- **Git:** branch `v20` off `v19-baseline`, not merged, not pushed; the working tree is clean at
  the start of the next chat. Work on this branch or a new one off it; never commit to `main`.
- **Best model: v20** at 10M steps, `models/ppo_pokemon_6v6_v20.pt` (tracked in git).

  | Opponent | Games | v20 |
  |---|---|---|
  | SimpleHeuristics | 1,000 | 83.7% ± 2.3% |
  | v19 (10M) | 2,000, both seats | 79.3% ± 1.8% |
  | v18.3 (previous best, ~210M steps) | 2,000, both seats | 70.1% ± 2.0% |
  | RandomPlayer | 1,000 | 99.8% ± 0.3% |
  | Foul Play (local, default search) | 200 | 13.5% ± 4.7% |

- In-training evals (500 deterministic games vs SimpleHeuristics, ±4 points):

  | Steps (M) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | v19 | 13.4 | 34.0 | 43.2 | 48.6 | 51.0 | 57.0 | 61.8 | 59.6 | 60.4 | 63.0 |
  | v20 | 54.0 | 68.0 | 73.0 | 75.2 | 80.6 | 81.2 | 83.0 | 81.8 | 84.4 | 83.4 |

- **Run artifacts** (gitignored): `models/ppo_pokemon_6v6_v20_latest.pt` (resumable at 10.03M
  steps, with the optimizer), `models/eval_snapshots_v20/step_*.pt` (1M..10M, for head-to-heads at
  equal steps), `models/selfplay_pool_v20/`, `logs/train_v20.log`, `logs/v20/` (TensorBoard),
  `logs/final_checks_step_0010034773/`, `logs/foulplay/`.
- Nothing is running: no Showdown servers, no training. Docker Desktop is up (only needed for Foul
  Play).

## 2. The learning-rate run
- **What:** v20 exactly, from scratch, with one change: `lr_initial` x3, 1.766e-4 instead of
  5.888e-5 (Wang's 10^-4.23). Same 20M schedule (lr / (8x+1)^1.5), embeddings still x10 on top,
  every other setting v20's. Version `v20lr3`, so it gets its own files.
- **Why:** v20's policy barely moved per update: approx_kl 0.0018 at 1M falling to 0.0004 at 10M
  (the usual target is ~0.01), clip fraction 0.13 → 0.03, and the eval curve flattened from 5M as
  the LR decayed (5.9e-5 → 5.3e-6 by 10M). The LR, tuned by Wang for a different setup, is the most
  likely bottleneck, and training v20 longer on its schedule would add a couple of points at best.
- **Length:** to 5M first (~2-2.5 hours on power: v20 ran 1,000-1,280 samples/s collecting early,
  ~600-750 overall later). Continue to 10M if it's ahead (section 4).
- **Commands**, from `pokemon-rl/`:
  ```
  cd ../pokemon-showdown && for p in 8000 8001 8002 8003; do (nohup node pokemon-showdown start --no-security --port $p > /tmp/showdown_$p.log 2>&1 &); sleep 2; done; cd ../pokemon-rl
  LOG=logs/train_v20lr3.log nohup bash tools/run_v20_mac.sh --version v20lr3 --workers 6 --servers 3 \
      --base-port 8000 --eval-port 8003 --stop-at 5000000 --set lr_initial=1.766e-4 > /dev/null 2>&1 &
  ```
  Start the servers one at a time (each `start` rebuilds). The log's first lines should show
  `--set lr_initial = 0.0001766` and `lr 1.766e-04 (embeddings x10)`.
- **Continuing or resuming:** the same command with a higher `--stop-at` resumes from
  `models/ppo_pokemon_6v6_v20lr3_latest.pt`. Repeat `--set lr_initial=1.766e-4`: the trainer
  refuses to resume a checkpoint with different settings (tested). `--set KEY=VALUE` overrides any
  `CONFIG` entry (`training/ppo_v20.py`), e.g. `--set clip_range_vf=None` for a later A/B.
- **Restarting the learner by hand** (e.g. to apply a code change): kill the runner
  (`bash tools/run_v20_mac.sh`) first, then SIGTERM the trainer (it saves), then start a new runner.
  The runner treats a clean exit as "done" and stops, so killing only the trainer ends the run.
- Don't edit `training/ppo_v20.py`, the env, network or player while a run is live: eval
  processes and resumed segments import the code on disk fresh.

## 3. Monitoring
- `bash tools/train_status.sh logs/train_v20lr3.log`: power, swap, latest update, all evals,
  runner messages, errors.
- Evals finish ~2 minutes after each 1M boundary (`>>> Eval @ <step>: <rate>`); training keeps
  going meanwhile. To wait in a background-job session, use a `run_in_background` loop that exits
  on a new eval line, e.g.
  `n0=$(grep -ac '>>> Eval @' LOG); until [ "$(grep -ac '>>> Eval @' LOG)" -gt "$n0" ] || ! pgrep -f "[r]un_v20_mac" > /dev/null; do sleep 10; done`.
  Keep the brackets in `pgrep -f "[r]un..."`: without them the pattern matches the waiting shell
  itself and the loop never ends.
- What to watch on the update lines: `kl` (approx_kl; ~0.003-0.01 is the hope; above ~0.03, or
  `clip` above ~0.3, means x3 is too hot), `ent` (entropy; v20 went 1.75 → 1.03 over 10M; a fast
  collapse is a bad sign), `ev` (explained variance; v20 held ~0.81), `win` (vs the training mix).
- Learner memory: `footprint <learner pid>` should stay around 0.7-2 GB. If it climbs every
  update, something is sending new tensor shapes to MPS again (section 6).

## 4. How to judge it
- Against v20 at equal steps: 54.0, 68.0, 73.0, 75.2, 80.6% at 1-5M. Single 500-game points
  differ by chance up to ~6 points; look for a curve above v20's at several points.
- If it looks ahead by 3-5M, confirm with a head-to-head at equal steps, 1,000 games per seat
  (after the run, on a free server port; the step number is in the file name):
  `python3 -m eval.ppo_test --model models/eval_snapshots_v20lr3/step_<~5M>.pt --opponents model --opp-model models/eval_snapshots_v20/step_0005019854.pt --battles 1000 --seat both --port 8000`
- Then:
  - **Clearly ahead:** continue it to 10M, then the final checks (`MODEL=... bash
    tools/final_checks_v20.sh`: 1,000 vs SimpleHeuristics, head-to-heads vs v19 and v18.3), a
    head-to-head vs v20's 10M model, and 200 games vs Foul Play (section 6). If approx_kl stayed
    low, x10 is the next arm.
  - **About the same:** the LR isn't what limits v20. Try the other A/Bs (section 5), or extend
    v20 itself to 20M (`--version v20 --stop-at 20000000`, no `--set`).
  - **Worse or unstable:** try x2, or keep Wang's LR.

## 5. After the LR run
1. More A/Bs, one change per run, compared at equal steps: `clip_range_vf` (0.0184 vs None;
   explained variance sat at ~0.81), `gae_lambda` (0.754 vs 0.9-0.95), `zero_avoid_coef` (0.005 vs
   0 and 0.02), `embed_lr_mult` (10 vs 1).
2. Train the winner longer (20M or more).
3. **Step 4** (`roadmap.md`): an attention trunk over the v20 tokens (12 Pokémon + 4 moves + field
   already come as separate token arrays), history tokens from `battle.observations[t].events`, and
   auxiliary heads. Two-sided self-play gives free labels: the opponent's actual next action and its
   hidden item/ability/moves/tera type are in `env.battle2`. Batch-1 costs now: featurizer 0.26 ms,
   v20 net 0.27 ms; a 3-layer d=128 transformer over 25 tokens measured 0.23 ms.
4. Features not in v20: the opponent's candidate sets from `data/gen9_randbats_sets.json` (roles,
   tera types, moves, filtered by revealed moves), the opponent's damage against our bench
   (poke-env's calc is ~30 µs per call), opponent PP.
5. Search (roadmap step 6) is what makes Foul Play strong; the network alone is ~320 Elo behind it.

## 6. Things that matter on this machine
- **Disk:** 28 GB free on September 30 (94% used). Swap lives on the same disk, and a new run's
  pool copies the five SB3 seed models (~270 MB).
- **Memory, not CPU, is the limit.** Each Showdown server is ~10 node processes and ~1.2 GB. Stop
  the ones you started when done: kill each launcher's process tree
  (`pgrep -f "pokemon-showdown start --no-security --port 8000$"` and its children); they share a
  process group with each other, so don't kill by group. Other apps kept 12-20 GB in swap.
- **MPS caches a compiled graph per tensor shape (~57 MB each).** A partial last minibatch of a new
  size every update grew the learner to 7.6 GB and the Mac swapped; `ppo_update` now uses only
  full, CPU-gathered 1,024-sample minibatches. Keep shapes on the GPU fixed.
- **Battery:** `tools/run_v20_mac.sh` pauses at 20% on battery (SIGTERM, the trainer saves) and
  resumes on AC or at 30%. The Mac slept while paused; it resumed when plugged in.
- Throughput with 6 workers on 3 servers: 900-1,280 samples/s collecting early in a run, 550-900
  later (shorter games, more v20 pool opponents); updates 7-10 s on MPS. Background apps make the
  numbers drift; compare speeds with alternating runs.
- **Foul Play:** `docker run -d --rm --name foulplay-eval foul-play:latest --websocket-uri ws://host.docker.internal:8005/showdown/websocket --ps-username FoulPlayLocal7291 --bot-mode accept_challenge --pokemon-format gen9randombattle --run-count 250 --log-level INFO`
  with a server on 8005, then `python3 -m eval.foulplay_eval --model <.pt> --games 200 --port 8005 --opponent FoulPlayLocal7291`
  (about 9 s per game). Nothing else should run meanwhile: Foul Play searches for a fixed time, so a
  busy Mac weakens it. If `docker` returns "500 Internal Server Error", Docker Desktop's VM didn't
  come up: quit Docker Desktop and open it again (`evaluation.md`).
- The cloud scripts handle v20 but weren't run with it; a VM without a GPU does the update on the
  CPU (~17 s per update).

## 7. Checks to rerun after code changes
- `python3 -m tools.check_v20_env --games 50 --port 8123` (features vs ground truth, action mapping,
  orders executed; outside Illusion games expect 0-1 mismatches), `python3 -m
  tools.check_v20_selfplay --port 8123`, `PORT=8123 bash tools/check_v20_resume.sh`.
- A featurizer change that shouldn't change outputs: compare against the previous version on live
  battles (the speed-up was checked this way: 2,687 decisions, 0 differing).
- Changing the observation or action layout: checkpoints store `LAYOUT` and refuse to load with a
  different one; bump the version and keep the old featurizer importable if old models must play.

## 8. Pitfalls hit so far
- The fork's request-before-log order (gotcha 9): keep `hold_requests_until_log` on for local play.
- zsh doesn't split an unquoted `$ARGS` into words; run multi-argument scripts under `bash -c` or
  write the arguments out.
- `pkill -f` / `pgrep -f` patterns can match your own shell or a running job; use `[x]yz` brackets
  and target PIDs.
- A background-job session can't sleep in the foreground; use `run_in_background` loops that exit
  on a condition.
- Opponent Atk estimate: Showdown zeroes Atk from the 4 chosen moves (fixed-damage moves like
  Seismic Toss don't count; Porygon2's Tera Blast does).

## 9. Working with this user
- Solo project, short messages. Give a clear recommendation and act on it.
- Money needs their decision: confirm before any cloud spend. When they're away and ask not to be
  prompted, run on the Mac.
- They want progress reports during long runs (each eval is a good moment).
- Git: work on branches, never commit to `main`, don't push without asking. Don't commit along
  the way; commit once at the very end of a task, ending the message with the session's
  `Co-Authored-By` line.
- Their Showdown ladder password is in `players/ladder_bot.py` and git history; don't copy it.
