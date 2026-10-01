# Handoff after step 3 (v20): hyperparameter A/Bs, then step 4

For the agent that picks this up in a new chat. Written September 30, 2026, at the end of the
session that built, trained and evaluated v20, and updated that night after the **3x
learning-rate run** (section 2: no gain; the LR isn't what limits v20). Which run comes next is
the user's call; the recommendation is the `clip_range_vf` A/B (section 5). Everything here was
measured on September 29-30 unless it says otherwise.

## 0. Start here
1. Read this file, then `CLAUDE.md`. Skim `v20.md` (design, checks, results, the LR run) and
   items 9-15 of `poke_env_gotchas.md`. You don't need to read the step 3 brief (`step3_handoff.md`).
2. Runs go on the MacBook (free); don't use the cloud without the user's say-so.
3. Check the Mac is on power first (`pmset -g batt`): training drains the battery ~1% per minute;
   the runner pauses at 20% and resumes on power.
4. Start a run (section 2), monitor it (section 3), judge it (section 4), report to the user as the
   1M evals come in.

## 1. Where things stand
- **Git:** branch `v20` off `v19-baseline`, not merged; the latest commits aren't pushed. Work on
  this branch or a new one off it; never commit to `main`.
- **Best model: v20** at 10M steps, `models/ppo_pokemon_6v6_v20.pt` (tracked in git).
  Head-to-heads with both sides greedy (section 4 says why that matters):

  | Opponent | Games | v20 |
  |---|---|---|
  | SimpleHeuristics | 1,000 | 83.7% ± 2.3% |
  | v19 (10M) | 2,000, both seats | 73.9% ± 1.9% (first reported 79.3%, with v19 sampling) |
  | v18.3 (previous best, ~210M steps) | 2,000, both seats | 62.8% ± 2.1% (first reported 70.1%) |
  | v20lr3 (10M) | 2,000, both seats | 50.6% ± 2.2% |
  | RandomPlayer | 1,000 | 99.8% ± 0.3% |
  | Foul Play (local, default search) | 200 | 13.5% ± 4.7% |

- In-training evals (500 deterministic games vs SimpleHeuristics, ±4 points):

  | Steps (M) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | v19 | 13.4 | 34.0 | 43.2 | 48.6 | 51.0 | 57.0 | 61.8 | 59.6 | 60.4 | 63.0 |
  | v20 | 54.0 | 68.0 | 73.0 | 75.2 | 80.6 | 81.2 | 83.0 | 81.8 | 84.4 | 83.4 |
  | v20lr3 | 37.2 | 59.8 | 63.8 | 74.8 | 75.0 | 76.4 | 80.6 | 80.8 | 83.0 | 80.8 |

- **Run artifacts** (gitignored). v20: `models/ppo_pokemon_6v6_v20_latest.pt` (resumable at
  10.03M steps, with the optimizer), `models/eval_snapshots_v20/step_*.pt` (1M..10M, for
  head-to-heads at equal steps), `models/selfplay_pool_v20/`, `logs/train_v20.log`, `logs/v20/`
  (TensorBoard), `logs/final_checks_step_0010034773/`, `logs/greedy_h2h/`, `logs/foulplay/`.
  v20lr3: the same names with `v20lr3` (resumable at 10.04M), `logs/lr3_checks_5M/`,
  `logs/final_checks_step_0010041096/`.
- Nothing is running: no Showdown servers, no training. Docker Desktop is up (only needed for Foul
  Play).

## 2. The learning-rate run (done) and how to run the next A/B
- **Result:** `lr_initial` x3 (1.766e-4, version `v20lr3`, from scratch to 10M, every other
  setting v20's) gave no gain. It was well behind early (37% vs 54% at 1M), level at 5M (77.8% vs
  77.4% vs SimpleHeuristics over 1,000 games; 47.3% ± 2.2% in a greedy head-to-head) and level at
  10M (49.4% ± 2.2% head-to-head; 80.4% vs 83.7% vs SimpleHeuristics). The clip range (0.0829)
  caps how far an update moves the policy, so v20's low approx_kl never meant the LR was too low;
  x3 pushed the clip fraction to 0.3-0.4 and learned slower. Late in the run its updates were ~4x
  v20's, and it still gained no ground. Details and diagnostics: `v20.md`, "Learning rate x3".
- **An A/B run:** v20 from scratch with one `--set` change and its own version name, to 5M first
  (~1 h 45 min on power at 750-1,000 samples/s), continued to 10M if it's ahead (section 4). From
  `pokemon-rl/`, e.g. for the `clip_range_vf` A/B:
  ```
  cd ../pokemon-showdown && for p in 8000 8001 8002 8003; do (nohup node pokemon-showdown start --no-security --port $p > /tmp/showdown_$p.log 2>&1 &); sleep 2; done; cd ../pokemon-rl
  LOG=logs/train_v20vf.log nohup bash tools/run_v20_mac.sh --version v20vf --workers 6 --servers 3 \
      --base-port 8000 --eval-port 8003 --stop-at 5000000 --set clip_range_vf=None > /dev/null 2>&1 &
  ```
  Start the servers one at a time (each `start` rebuilds). The log's first lines should show the
  `--set` line and the new value in the `PPO:` line (here `vf-clip None`).
- **Continuing or resuming:** the same command with a higher `--stop-at` resumes from
  `models/ppo_pokemon_6v6_<version>_latest.pt`. Repeat the `--set` flags: the trainer refuses to
  resume a checkpoint with different settings (tested). `--set KEY=VALUE` overrides any `CONFIG`
  entry (`training/ppo_v20.py`).
- **Restarting the learner by hand** (e.g. to apply a code change): kill the runner
  (`bash tools/run_v20_mac.sh`) first, then SIGTERM the trainer (it saves), then start a new runner.
  The runner treats a clean exit as "done" and stops, so killing only the trainer ends the run.
- Don't edit `training/ppo_v20.py`, the env, network or player while a run is live: eval
  processes and resumed segments import the code on disk fresh.

## 3. Monitoring
- `bash tools/train_status.sh logs/train_<version>.log`: power, swap, latest update, all evals,
  runner messages, errors.
- Evals finish ~2 minutes after each 1M boundary (`>>> Eval @ <step>: <rate>`); training keeps
  going meanwhile. To wait in a background-job session, use a `run_in_background` loop that exits
  on a new eval line, e.g.
  `n0=$(grep -ac '>>> Eval @' LOG); until [ "$(grep -ac '>>> Eval @' LOG)" -gt "$n0" ] || ! pgrep -f "[r]un_v20_mac" > /dev/null; do sleep 10; done`.
  Keep the brackets in `pgrep -f "[r]un..."`: without them the pattern matches the waiting shell
  itself and the loop never ends.
- What to watch on the update lines: `clip` (clip fraction: v20 0.10-0.14 early and 0.03 by 10M;
  v20lr3's 0.3-0.4 was too hot), `kl` (approx_kl: ~0.0015 early with this clip range), `ent`
  (entropy; v20 went 1.75 → 1.03 over 10M; a fast collapse is a bad sign), `ev` (explained
  variance; 0.81 from 1M on in both runs), `win` (vs the training mix).
- Learner memory: `footprint <learner pid>` should stay around 0.5-2 GB. If it climbs every
  update, something is sending new tensor shapes to MPS again (section 6).

## 4. How to judge it
- In-training evals are 500 games: single points differ by chance up to ~6 points, and at 5M they
  said v20lr3 75.0% vs v20 80.6% when 1,000-game evals said 77.8% vs 77.4%. Read the curve's
  shape from them; decide with the checks below.
- At 5M (and 10M), on free ports after the run stops, two checks in parallel (~7 minutes; the step
  number is in the file name, v20's are 0005019854 and 0010034773):
  `python3 -m eval.ppo_test --model models/eval_snapshots_<version>/step_<~5M>.pt --opponents model --opp-model models/eval_snapshots_v20/step_0005019854.pt --opp-deterministic --battles 1000 --seat both --port 8000`
  and `python3 -m eval.ppo_test --model models/eval_snapshots_<version>/step_<~5M>.pt --opponents simple --battles 1000 --port 8001`
  (v20@5M: 77.4% ± 2.6%; v20@10M: 83.7% ± 2.3%).
- **Always pass `--opp-deterministic` to head-to-heads.** Without it the opponent model samples
  while `--model` plays greedily, and `--seat both` swaps seats, not roles: v20@5M greedy beats
  itself sampling 67.7%, and v20lr3@5M scored 65.6% against v20@5M sampling but 47.3% with both
  greedy. The deployed bots play greedily.
- Then:
  - **Clearly ahead** (head-to-head above ~53%, outside its ±2.2 noise): continue to 10M, then the
    final checks (`MODEL=... bash tools/final_checks_v20.sh`: 1,000 vs SimpleHeuristics, greedy
    head-to-heads vs v19 and v18.3), a greedy head-to-head vs v20's 10M model, and 200 games vs
    Foul Play (section 6).
  - **About the same or worse:** keep v20's value and move to the next A/B.

## 5. What next
1. A/Bs, one change per run, compared at equal steps (section 4), in order of promise:
   - `clip_range_vf=None` (vs 0.0184): explained variance sat at 0.81 from 1M on in v20 and
     v20lr3 alike, so the value fit isn't LR-limited; SB3's value clip lets each prediction move
     only ±0.0184 per update.
   - `gae_lambda` 0.9-0.95 (vs 0.754): with ±1 only at the end of a ~30-decision game, λ = 0.754
     leans hard on the value function for credit assignment.
   - `ent_coef` lower (vs 0.0588, e.g. 0.02): the sampled policy, which plays all training games,
     is far weaker than the greedy one (67.7% greedy vs itself sampling at 5M).
   - `zero_avoid_coef` (0.005 vs 0 and 0.02), `embed_lr_mult` (10 vs 1).
   - Not the LR: x3 gave nothing, and x2 or an LR floor would test the same thing.
2. Train the winner longer (20M or more), or extend v20 itself (`--version v20 --stop-at 20000000`,
   no `--set`); its curve was nearly flat from 7M.
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
- **Disk:** 24 GB free on the night of September 30 (95% used). Swap lives on the same disk (it
  grew from 13 to 19 GB during the LR run), and a new run's pool copies the five SB3 seed models
  (~270 MB).
- **Memory, not CPU, is the limit.** Each Showdown server is ~10 node processes and ~1.2 GB. Stop
  the ones you started when done: kill each launcher's process tree
  (`pgrep -f "pokemon-showdown start --no-security --port 8000$"` and its children, e.g. collected
  with `pgrep -P` into a file and passed to `xargs kill`); they share a process group with each
  other, so don't kill by group. Other apps kept 12-20 GB in swap.
- **MPS caches a compiled graph per tensor shape (~57 MB each).** A partial last minibatch of a new
  size every update grew the learner to 7.6 GB and the Mac swapped; `ppo_update` now uses only
  full, CPU-gathered 1,024-sample minibatches. Keep shapes on the GPU fixed.
- **Battery:** `tools/run_v20_mac.sh` pauses at 20% on battery (SIGTERM, the trainer saves) and
  resumes on AC or at 30%. The Mac slept while paused; it resumed when plugged in.
- Throughput with 6 workers on 3 servers: 900-1,280 samples/s collecting early in a run, 550-1,000
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
- Head-to-heads without `--opp-deterministic` pit our greedy model against a sampling opponent
  (section 4); the v20 numbers first reported were inflated by 5-7 points this way.
- zsh doesn't split an unquoted `$ARGS` into words; run multi-argument scripts under `bash -c` or
  write the arguments out. The same goes for `kill $PIDS` (it fails silently with `2>/dev/null`).
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
