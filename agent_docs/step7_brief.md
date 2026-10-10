# Step 7 brief: learning from the search bot (search self-play data, then distillation)

Written October 10, 2026, at the end of the step 6 session. The user approved the plan below
("lets do it") and asked for it to run in a new session. Nothing of it is built yet: start at
step 1 (section 3).

## 0. Read before starting
1. This file.
2. `CLAUDE.md` (how to run things, rules for counting wins and noise).
3. `agent_docs/step6.md`: the search bot (`search/`), its checks and every result; the "What next"
   section is where this step comes from.
4. `agent_docs/v21.md`: the v21 featurizer, labels and aux heads (the network this step trains).
5. `agent_docs/poke_env_gotchas.md` items 9-15, `agent_docs/step4_handoff.md` sections 3 and 6
   (monitoring runs, memory, Foul Play on this Mac).

## 1. Where things stand
- Branch `search` (off `v21`), not pushed: Phase 1 (201e122), converter fixes + Phase 2 (cd38478),
  results (f0d33bc), ladder / play bots on the search bot (a635da0), this brief.
- Nothing running: no Showdown servers, no Foul Play container (image `foul-play:latest` exists).
- poke-engine in this `python3`: 0.0.48 built with gen9 + Terastallization plus one added binding,
  `emm_matrix` (`search/engine_patch/`, `bash search/engine_patch/build.sh`).
- The best bot: `SearchPolicy(prior_weight=0.15, set_gamma=0.5, infer_items=True, net=NetModel())`
  (MCTS over sampled opponent teams, 2 x 100 ms at Foul Play's budget): 47.4% ± 3.1% vs Foul Play
  over 1,000 games (level), 55.8% vs the Phase 1 bot, the Phase 1 bot 80.0% vs v20.
- Why this step: PPO has plateaued (~84% vs SimpleHeuristics, every setting and network), the search
  bot is ~300 Elo above the network, and putting v21 into the search (prior, set model, opponent
  model) added a few points at most. PPO can't learn from the search's moves (they aren't the
  network's own samples); expert iteration can: record the search playing itself, train the network
  to predict the search's choices and the results.

## 2. The plan (approved October 10, 2026)
1. **Generator** (this session's job): the search bot plays itself, both sides recorded, 2-3 days on
   the Mac (~1M samples a day). It's data collection, not training: the bot doesn't learn while it
   plays.
2. **Distillation** (offline, hours on MPS): train v21 from its 20M weights on the data: policy
   cross-entropy to the search's visit shares, value to the game result, the aux heads on the
   labels; hold out whole games. Then measure the distilled network alone (greedy) vs v20 (2,000
   games, both seats) and vs SimpleHeuristics, and as the search's prior (`prior_weight`) vs the
   best bot.
3. **Only if 2 works: a learned leaf evaluator inside the engine.** poke-engine's MCTS doesn't play
   games out: `src/mcts.rs` `rollout` scores a leaf as `sigmoid(evaluate(state) - root_eval)` with the
   hand-written `evaluate` (`src/genx/evaluate.rs`: HP, status, items, boosts, hazards). Replacing it
   means a small model in Rust (the v21 network is far too slow at ~600 MCTS iterations per ms) on
   features of engine states: expose a `features(state)` binding so training (Python) and search
   (Rust) compute the same features, train on the stored states and results, A/B vs the best bot and
   vs Foul Play. Then the AlphaZero loop (better evaluator -> stronger search -> new games).

## 3. Step 1: `training/gen_search_games.py`
Workers (forkserver, as `training/ppo_v20.py`), each with its own `RLEnvV20` on a local server and two
`SearchPolicy` objects (the best bot's settings, `iters_per_ms=600`: a fixed budget equal to the
wall-clock one, so 4-6 workers sharing the Mac don't weaken the search) sharing one `NetModel`
(`torch.set_num_threads(1)`). Per decision, per side:
- v21 observation: `idx` int16[188], `flt` float16[1755] (training rounds features to float16 too),
  `mask` bool[14] (the v21 featurizer: `NetModel().fz`, keyed by battle).
- `target` float32[14]: the visit shares (`policy.last["shares"]`, summed over the sampled teams, a
  copy taken before the prior is added; normalize over legal actions). Set `policy.last = None`
  before each `action()`: one-legal-action decisions aren't searched (`searched` False; keep them
  with a one-hot target, training can drop them).
- `action` (the near-best pick actually played), `value` (`policy.last["value"]`: the root's mean
  MCTS score for us, 0..1, added in this commit), `iterations`.
- `lab` int16[43]: v21's `set_labels(battle, other)` and, once both sides' orders are known,
  `next_action_label` at `LAB_NEXT` (as the trainer's `Worker.play_game`); all -1 in Illusion games
  (`illusion_game`).
- `outcome` (+1 / -1 / 0 from `battle.won`, filled at the end), game id, side, turn.
- The true engine state, `build_true_state(battle, other)[0].to_string()` (both sides exact), one
  line per sample in a gzip file next to the shard: what step 3 trains on.

Output in `datasets/search_selfplay/` (add `datasets/` to `.gitignore`: `data/` is tracked): shards of
~100 games (`np.savez_compressed` + `.states.gz`), written atomically, new shard names per run so a
restart just adds shards; a meta file with the settings and the git commit. Measure the size of a
first shard before the long run: 21 GB were free on October 10; store states only every k-th
decision if they don't fit.

Robustness, for a multi-day run:
- `EnvTimeout` (no battle update for 60 s): catch it, rebuild the env, continue. poke-engine's
  bindings hold the GIL during a search, so one pathological fixed-iteration search on a loaded Mac
  (a 63 s one was seen) starves poke-env's websocket thread; one step 6 eval crashed this way.
- Pause while on battery (`pmset -g batt`, "AC Power"), as `tools/run_v20_mac.sh` does for training.
- SIGTERM: finish the current game, write the shard, exit.
- Progress lines (games, samples, samples/s, p1 win rate, decisions per game, errors) for
  monitoring with `run_in_background` loops; a test run of 1 worker x 2 games first (check shapes,
  targets sum to 1 over legal actions, outcomes, state strings round-trip with
  `poke_engine.State.from_string`).
Throughput seen in step 6: two search bots with ~5 processes on the Mac, ~25 s per game per process
(~35 decisions per side): 5 workers ~17K games and ~1.2M samples a day.

## 4. Things to know
- Servers: `cd ../pokemon-showdown && node pokemon-showdown start --no-security --port 8123`; one
  server handled 5 search processes. Stop the ones you start (kill the launcher and its children).
- Evals: `python3 -m eval.search_eval` (`--config` / `--opp-config` JSON of `SearchPolicy` arguments,
  `--budget-iters-per-ms 600` for parallel runs), `python3 -m eval.foulplay_eval --model search
  --config ...` (wall clock, alone on the Mac, ~12 s per game). Noise: ±3.1 points at 1,000 games.
- Memory: swap ~10 GB used with the user's apps; Docker Desktop's VM holds ~8 GB while open.
- Search bot latency ~240 ms per decision at Foul Play's budget (the ladder bot uses 2 x 250 ms).
- Engine gaps found by `tools/check_search_state.py` (simultaneous switches always side one first,
  2-5 hit moves always 3 hits, Rest at full HP, Sheer Force + Life Orb, Heal Block, ...) are in
  `step6.md`; they're the same for both sides of the self-play games.

## 5. Working with this user
- Solo project, short messages: give a clear recommendation, act on it, explain the reasoning; straight
  answers, including when an earlier call was wrong.
- Runs go on the MacBook (free); confirm before any cloud spend; check the Mac is on power before long
  jobs; they may ask to pause runs (the eval scripts can't resume: stop them and rerun).
- Short reports as results come in.
- Git: work on branches (stay on `search` or branch off it), never commit to `main`, don't push without
  asking, commit once per finished piece of work with the session's `Co-Authored-By` line.
- The Showdown ladder password is in `players/ladder_bot.py` and git history; don't copy it.
- Shell: zsh doesn't split an unquoted `$VAR`; `pgrep -f` patterns need a `[x]yz` bracket; no
  foreground `sleep` (use `run_in_background` loops that exit on a condition, or `perl -e
  'select(undef,undef,undef,N)'` in short waits).
