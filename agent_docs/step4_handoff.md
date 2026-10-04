# Handoff after step 3 (v20): hyperparameter A/Bs, then step 4

For the agent that picks this up in a new chat. Written September 30, 2026, at the end of the
session that built, trained and evaluated v20, and updated after three A/Bs and an extension
(section 2): the **3x learning rate** (no gain), **no value clip** (a faster start, level with v20
by 5M), **v20 extended to 20M** (still learning, slowly; the 20M model is the new best) and a
**lower entropy bonus** (sharper, not stronger). The user agreed the `gae_lambda` A/B next; it
started October 4 (`logs/train_v20gl.log`, section 5). Everything here was measured on
September 29-October 4 unless it says otherwise.

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
- **Best model: v20** at 20M steps, `models/ppo_pokemon_6v6_v20.pt` (tracked in git; the 10M one
  is in git history). Head-to-heads with both sides greedy (section 4 says why that matters):

  | Opponent | Games | v20 at 20M | v20 at 10M |
  |---|---|---|---|
  | SimpleHeuristics | 1,000 | 83.9% ± 2.3% | 83.7% ± 2.3% |
  | v19 (10M) | 2,000, both seats | 75.4% ± 1.9% | 73.9% ± 1.9% (first reported 79.3%, with v19 sampling) |
  | v18.3 (previous best, ~210M steps) | 2,000, both seats | 67.2% ± 2.1% | 62.8% ± 2.1% (first reported 70.1%) |
  | v20 at 10M | 2,000, both seats | 52.3% ± 2.2% | |
  | v20lr3 (10M) | 2,000, both seats | | 50.6% ± 2.2% |
  | RandomPlayer | 1,000 | 99.8% ± 0.3% | 99.8% ± 0.3% |
  | Foul Play (local, default search) | 200 | | 13.5% ± 4.7% |

- In-training evals (500 deterministic games vs SimpleHeuristics, ±4 points):

  | Steps (M) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | v19 | 13.4 | 34.0 | 43.2 | 48.6 | 51.0 | 57.0 | 61.8 | 59.6 | 60.4 | 63.0 |
  | v20 | 54.0 | 68.0 | 73.0 | 75.2 | 80.6 | 81.2 | 83.0 | 81.8 | 84.4 | 83.4 |
  | v20lr3 | 37.2 | 59.8 | 63.8 | 74.8 | 75.0 | 76.4 | 80.6 | 80.8 | 83.0 | 80.8 |
  | v20vf | 65.8 | 71.4 | 78.2 | 76.8 | 79.2 | | | | | |
  | v20ent | 56.0 | 68.2 | 76.0 | 73.6 | 78.4 | 80.2 | 82.2 | 82.0 | 80.4 | 80.4 |

  v20 from 11M to 20M: 84.6, 81.8, 81.6, 82.6, 83.8, 84.4, 88.0, 84.4, 83.8, 84.4.

- **Run artifacts** (gitignored). v20: `models/ppo_pokemon_6v6_v20_latest.pt` (resumable at
  20.03M steps, with the optimizer; the 10.03M state is in
  `models/backup_v20_latest_step_0010034773.pt`), `models/eval_snapshots_v20/step_*.pt` (1M..20M,
  for head-to-heads at equal steps), `models/selfplay_pool_v20/`, `logs/train_v20.log`,
  `logs/v20/` (TensorBoard), `logs/final_checks_step_0010034773/`,
  `logs/final_checks_step_0020025160/`, `logs/v20_extension_checks/`, `logs/greedy_h2h/`,
  `logs/foulplay/`.
  v20lr3: the same names with `v20lr3` (resumable at 10.04M), `logs/lr3_checks_5M/`,
  `logs/final_checks_step_0010041096/`. v20vf: the same names with `v20vf` (resumable at 5.02M),
  `logs/vf_checks/`. v20ent: the same names with `v20ent` (resumable at 10.03M),
  `logs/ent_checks/`.
- Nothing is running: no Showdown servers, no training. Docker Desktop is up (only needed for Foul
  Play).

## 2. The A/Bs so far, and how to run the next one
- **Learning rate x3:** `lr_initial` 1.766e-4 (version `v20lr3`, from scratch to 10M, every other
  setting v20's) gave no gain. It was well behind early (37% vs 54% at 1M), level at 5M (77.8% vs
  77.4% vs SimpleHeuristics over 1,000 games; 47.3% ± 2.2% in a greedy head-to-head) and level at
  10M (49.4% ± 2.2% head-to-head; 80.4% vs 83.7% vs SimpleHeuristics). The clip range (0.0829)
  caps how far an update moves the policy, so v20's low approx_kl never meant the LR was too low;
  x3 pushed the clip fraction to 0.3-0.4 and learned slower. Late in the run its updates were ~4x
  v20's, and it still gained no ground. Details and diagnostics: `v20.md`, "Learning rate x3".
- **No value clip:** `clip_range_vf=None` (version `v20vf`, to 5M) learned faster early
  (head-to-heads vs v20 at equal steps: 58.6% at 1M, 55.5% at 2M) and was level by 5M (51.6% ±
  2.2%; 79.0% vs 77.4% vs SimpleHeuristics). Explained variance rose sooner but leveled off at the
  same 0.81. It never did worse, so use it in the next long run (`--set clip_range_vf=None`;
  changing the `CONFIG` default would make resuming v20 and v20lr3 need
  `--set clip_range_vf=0.0184`). Keep the A/Bs to one change vs v20, so they compare with v20's
  full curve and snapshots. Details: `v20.md`, "Value clip off".
- Both settings changed how fast v20 learns, not the level it reaches (~83-84% vs
  SimpleHeuristics by 7-10M).
- **v20 extended to 20M** (its own schedule, no change): still learning, slowly. 14M beats 10M
  53.0%, 20M beats 14M 53.8%, 20M beats 10M 52.3% (each ± 2.2%, both greedy); vs v18.3 it went
  from 62.8% to 67.2%, vs v19 from 73.9% to 75.4%, vs SimpleHeuristics unchanged. The 500-game
  in-training evals can't see gains this small. Details: `v20.md`, "Extension to 20M".
- **Lower entropy bonus:** `ent_coef=0.02` (version `v20ent`, to 10M) made the policy much
  sharper (entropy 0.80 vs 1.04 at 10M) but no stronger: head-to-heads vs v20 49.4% at 5M and
  49.5% at 10M; vs SimpleHeuristics 80.2% vs 77.4% at 5M, 81.5% vs 83.7% at 10M. Details:
  `v20.md`, "Entropy bonus 0.02".
- **An A/B run:** v20 from scratch with one `--set` change and its own version name, stopping at
  5M for the checks (1 h 45 min to 2 h 10 min on power at 650-1,000 samples/s), then on to 10M
  unless it's clearly worse (section 4). From `pokemon-rl/`, e.g. for the `ent_coef` A/B:
  ```
  cd ../pokemon-showdown && for p in 8000 8001 8002 8003; do (nohup node pokemon-showdown start --no-security --port $p > /tmp/showdown_$p.log 2>&1 &); sleep 2; done; cd ../pokemon-rl
  LOG=logs/train_v20ent.log nohup bash tools/run_v20_mac.sh --version v20ent --workers 6 --servers 3 \
      --base-port 8000 --eval-port 8003 --stop-at 5000000 --set ent_coef=0.02 > /dev/null 2>&1 &
  ```
  Stopping at 5M also restarts the stack, which keeps memory in check (section 6).
  Start the servers one at a time (each `start` rebuilds). The log's first lines should show the
  `--set` line and the new value in the `PPO:` line (here `ent 0.02`).
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
- At 5M, 10M (and 20M), two checks in parallel (~7 minutes on free ports; the step number is in
  the file name, v20's are 0005019854, 0010034773 and 0020025160):
  `python3 -m eval.ppo_test --model models/eval_snapshots_<version>/step_<~5M>.pt --opponents model --opp-model models/eval_snapshots_v20/step_0005019854.pt --opp-deterministic --battles 1000 --seat both --port 8000`
  and `python3 -m eval.ppo_test --model models/eval_snapshots_<version>/step_<~5M>.pt --opponents simple --battles 1000 --port 8001`
  (v20@5M: 77.4% ± 2.6%; v20@10M: 83.7% ± 2.3%; v20@20M: 83.9% ± 2.3%). While a run is live, the
  eval port (8003) is free between its 1M evals; player names are random, so overlaps don't clash.
- **Always pass `--opp-deterministic` to head-to-heads.** Without it the opponent model samples
  while `--model` plays greedily, and `--seat both` swaps seats, not roles: v20@5M greedy beats
  itself sampling 67.7%, and v20lr3@5M scored 65.6% against v20@5M sampling but 47.3% with both
  greedy. The deployed bots play greedily.
- Then (the 5M check only saves time; settings that change where training ends up show late, and
  v20 was still learning at 20M):
  - **Clearly worse or unstable at 5M:** stop and keep v20's value.
  - **Otherwise:** continue to 10M and check against v20@10M. That's the verdict: v20 gained only
    ~2-3 points head-to-head from 10M to 20M, so level at 10M means no gain (v20ent stopped
    there). Clearly ahead at 10M: continue to 20M to measure the gain against v20@20M.
  - **Clearly ahead at the end** (head-to-head above ~53%, outside its ±2.2 noise): the final
    checks (`MODEL=... bash tools/final_checks_v20.sh`: 1,000 vs SimpleHeuristics, greedy
    head-to-heads vs v19 and v18.3) and 200 games vs Foul Play (section 6).
  - The v20vf run stopped at 5M under an older rule (continue only if ahead); its training matched
    v20's from 3.5M on, so it's unlikely to differ later. It resumes from 5.02M if needed.

## 5. What next
1. A/Bs, one change per run vs v20, compared at equal steps, to 10M by default (section 4; about
   4 hours each). The two so far only changed how fast v20 learns (section 2), so try the settings
   that change what it converges to:
   - `gae_lambda=0.95` (vs 0.754), **running since October 4** as version `v20gl`
     (`logs/train_v20gl.log`; to 5M for checks, then on to 10M): with ±1 only at the end of a
     ~30-decision game, λ = 0.754 leans hard on the value function, whose explained variance stops
     at 0.81 whatever the LR, value clip or entropy bonus.
   - `zero_avoid_coef` (0.005 vs 0 and 0.02), `embed_lr_mult` (10 vs 1): less promising.
   - Done: the LR (x3; x2 or an LR floor would test the same thing), the value clip, and the
     entropy bonus (0.02). Three null results in a row point past PPO settings, to the network
     and its inputs (items 3-4).
2. Train the winner longer. Without a winner, v20 itself keeps gaining slowly (~20-25 Elo per
   5M steps at 10-20M): `--version v20 --stop-at 30000000`, no `--set`. Its LR schedule ends at
   20M (`total_timesteps`), so the LR stays at 2.2e-6 from there. A `--version v20` run that
   reaches `total_timesteps` rewrites `models/ppo_pokemon_6v6_v20.pt` (the tracked best) when it
   stops (other versions write their own, gitignored `ppo_pokemon_6v6_<version>.pt`): check the
   new model before committing it.
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
- **Memory grows over a long run:** the servers with games played (1.5-1.8 GB each after ~125K
  games), the workers' model caches with the pool (3.4 GB for six at 15M), the learner by ~155 MB
  per 1M steps. When updates slow from ~10 s to ~20 s and the Mac swaps, restart the stack: kill
  the runner, SIGTERM the trainer (it saves), restart the servers, start the runner again with the
  same arguments (done at v20's 15.17M; `v20.md`). On a 10M run, plan one restart near the middle.
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
