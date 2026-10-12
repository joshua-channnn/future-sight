# CLAUDE.md

Future Sight is a PPO self-play bot for Pokémon Showdown **Gen 9 Random Battles** (singles),
built on poke-env. v17-v19 use Stable-Baselines3 `MaskablePPO`; v20 (step 3 of the roadmap) has its
own PPO loop, entity-token observations and per-token action heads. The goal is a stronger bot,
measured by *true* win rate against fixed opponents (SimpleHeuristics, older models, Foul Play).

## Map
v20 (details in `agent_docs/v20.md`):
- `envs/rl_player_v20.py`: `Featurizer` (12 Pokémon, 4 move and 1 field token; own tera type and
  ability from the request, estimated opponent stats, damage estimates), the 14-action mapping
  (4 moves, 4 tera moves, 6 switches by `battle.team` slot), legality-only masks, and `RLEnvV20`
  (two-agent `PokeEnv` with timeouts and the request-order fix).
- `networks/v20_policy.py`: `V20Net` (shared trunk, move head, switch head, value head).
- `training/ppo_v20.py`: the v20 trainer (worker processes, two-sided self-play, pool, evals in
  their own process, checkpoints and resume). All settings in `CONFIG`.
- `players/v20_player.py`: `V20Policy`, `V20Player`, `load_opponent` (v20 `.pt` or SB3 zip).
- `eval/v20_eval.py`: evaluation games for v20; `eval/ppo_test.py` routes `.pt` models there.
- `tools/`: checks against live battles (`check_v20_*`), `bench_sb3.py`, `run_v20_mac.sh`.

v21 (step 4: v20 plus battle history, a transformer and auxiliary heads; level with v20, details
in `agent_docs/v21.md`):
- `envs/rl_player_v21.py`: `Featurizer` (v20's observation plus 16 history tokens), the aux labels
  (`set_labels`, `next_action_label`, read from the other side's battle object).
- `networks/v21_policy.py`: `V21Net` (net_config `head_context`: v20's context vector in the
  heads, off by default), the aux losses and metrics, per-species base rates. A partial
  `--set "net_config={...}"` is merged into the arch's defaults.
- `networks/registry.py`: arch name (`v20`, `v21`) -> featurizer, network, layout. Checkpoints
  store their arch; `V20Policy.load` and everything that loads models route through it.
- Trained by `training/ppo_v20.py --arch v21`; checks `tools/check_v21_env.py`,
  `tools/check_v21_net.py`, `tools/check_v20_selfplay.py --arch v21`,
  `ARCH=v21 bash tools/check_v20_resume.sh`.

Step 6 (search at decision time, details in `agent_docs/step6.md`):
- `search/sampling.py`: the set prior (`data/gen9_randbats_set_counts.json`: complete sets of the
  fork's generator over 1M teams, built by `tools/build_set_prior.py`) and opponent team sampling.
- `search/state.py`: poke-env battle -> poke-engine `State` (`build_state`, `build_true_state` for
  checks, `LogTracker` for what poke-env doesn't keep), engine options -> our 14 actions.
- `search/policy.py`: `SearchPolicy` (MCTS over sampled opponent teams, Foul Play's recipe; same
  interface as `V20Policy`), `SearchPlayer` (live play). `search/net.py`: v21 in the loop (Phase 2).
- poke-engine is 0.0.48 built with gen9 + Terastallization and two additions (the `emm_matrix`
  binding, step 6; the learned leaf evaluation, step 7) by `bash search/engine_patch/build.sh`;
  `pip install poke-engine==0.0.46` undoes it.
- Evals: `python3 -m eval.search_eval --opponents simple|model|search ...`,
  `python3 -m eval.foulplay_eval --model search ...`. Checks: `tools/check_engine_build.py`,
  `tools/check_set_prior.py`, `tools/check_search_state.py`, `tools/check_set_model.py`.

Step 7 (learning from the search, details in `agent_docs/step7.md`):
- `training/gen_search_games.py`: search self-play data (both sides: visit shares, per-action
  values, aux labels, the true engine state of every decision) in `datasets/` (gitignored:
  `search_selfplay/` round 0, `search_selfplay_r1/` round 1); `tools/check_search_games.py` checks
  shards.
- `training/distill_v21.py`: distillation into v21 (`--target shares | mix | qadv`).
- The learned leaf evaluation (+3.6 points vs step 6's bot, level with Foul Play):
  `search/engine_patch/learned_eval.patch` (state features and an antisymmetric linear model in
  Rust, used by MCTS at the leaves), `search/leaf_eval.py` (`best_settings()`: the deployed
  arguments, `leaf_eval="models/leaf_eval/lin1_86.npz", leaf_mode=2, leaf_scale=1.4`; falls back to
  the hand-written evaluation without the engine patch), `training/train_leaf_eval.py`.
- `tools/foulplay_loop.sh`: Foul Play evals of the search bot that restart Foul Play when it crashes.

Step 8 (matchup weights for the leaf evaluation; built and tested, no gain in play, details in
`agent_docs/step8.md`):
- `search/matchup_eval.py`: a network that sets per-Pokemon leaf weights once per search
  (`MatchupNet`, `MatchupLinear`, `MatchupModel`); the engine side is in `learned_eval.patch`
  (`mcts_with(..., root_weights)`, `matchup_inputs`, `pair_features`, `move_table`).
  `SearchPolicy(matchup=<.pt>, matchup_weight=a)`; `search/leaf_eval.py` `BEST_MATCHUP` (None).
- `training/train_matchup_eval.py` (pairs of states; ids let it memorize games: use
  `--ids none --move-props --pairs --no-value`); checks: `tools/check_matchup_eval.py`.
- Trained on the search's own per-option values instead (`pe.option_features_from_strings` plays
  each option one turn ahead): `training/train_q_eval.py` (linear; cache `datasets/q_cache/`),
  `training/train_q_matchup.py` (the network; cache `datasets/q_cache_mu/`). Better agreement with
  the deep search one turn ahead, no gain in play either.

v17-v19 (SB3):
- `envs/rl_player_6v6_v18.py`: the environment. Observation (677 floats: 12 Pokémon blocks,
  active-move details, global features), reward (`reward_mode`: `sparse` or `shaped`), action
  masks, and the 13-action mapping (4 moves, 4 tera moves, 5 switches).
- `envs/wrappers.py`: `MaskablePokeEnvWrapper`, which sits between poke-env and SB3.
- `networks/embedding_extractor_v18.py`: feature extractor (embeddings, per-Pokémon encoder, pooling).
- `training/ppo_embed_train_selfplay.py`: the only training script; all settings live in `CONFIG`.
- `players/ppo_player.py`: loads any saved model (env/extractor chosen via `ARCH_REGISTRY`). Used
  for self-play opponents and, with `--model <zip>`, `players/play_bot.py` (local play) and
  `players/ladder_bot.py` (both run the search bot by default since October 10, 2026, with step
  7's learned leaf evaluation since October 11).
- `eval/ppo_test.py` (vs Random / SimpleHeuristics / another model) and `eval/foulplay_eval.py`.
- `envs/rl_player_6v6.py`, `networks/embedding_extractor.py`, `networks/embedding_extractor_sp.py`:
  v17 code, kept only so v17 checkpoints still load.
- `cloud/`: scripts for training on a Google Cloud VM (pack, set up, run with a stall guard, fetch).
- Outside this repo: `../pokemon-showdown` (local Showdown server, a Feb 2026 fork) and
  `../foul-play` (Foul Play, locally patched, run via Docker).
- `models/` is gitignored except `ppo_pokemon_6v6_v20.pt` (the best model: v20 at 20M steps),
  `ppo_pokemon_6v6_v18.3_best.zip` (the best SB3 model) and `leaf_eval/lin1_86.npz` (the search
  bot's leaf evaluation). v20 checkpoints are torch files:
  `models/ppo_pokemon_6v6_<version>_{latest,best}.pt`, pool `models/selfplay_pool_<version>/`,
  eval snapshots `models/eval_snapshots_<version>/`.

## Running things
- Use `python3` (3.14, with poke-env 0.11 and SB3 2.7.1). `python` is Anaconda and lacks the deps.
  Run from this directory as modules, e.g. `python3 -m eval.ppo_test`.
- Everything needs a local Showdown server:
  `cd ../pokemon-showdown && node pokemon-showdown start --no-security --port 8123`.
- SB3 training occupies ports 8000-8004; v20 training uses `--servers` ports from 8000 plus the
  next one for its evals (8000-8003 with `--servers 3`). Use another port (e.g. 8123) for
  experiments and evals so you never collide with a running training job; `eval/ppo_test.py`
  defaults to 8004.
- v20: `python3 -m training.ppo_v20 --workers 6 --servers 3 --stop-at 10000000` (resumes from
  `_latest.pt` automatically; `--set KEY=VALUE` overrides any `CONFIG` entry, and a resumed run
  must repeat the same `--set` flags or it refuses to start; `--arch v21` trains v21, as version
  `v21` unless `--version` says otherwise). On the MacBook, `tools/run_v20_mac.sh` wraps it: it pauses the run
  when the battery gets low and resumes on power. Each Showdown server is ~10 node processes and
  ~1.2 GB; don't leave unused ones running on the 16 GB Mac.
- Evaluate a v20 model: `python3 -m eval.ppo_test --model models/X.pt --opponents simple --battles 1000 --port 8123`
  (head-to-heads: `--opponents model --opp-model models/Y --opp-deterministic --seat both`;
  without `--opp-deterministic` the opponent samples its actions, which inflates our win rate).
- There is no test suite. To check a change: `python3 -m py_compile` the touched files, then run a
  short eval, `python3 -m eval.ppo_test --opponents simple --battles 400 --port 8123`. For v20 env
  or trainer changes also run `python3 -m tools.check_v20_env --games 50 --port 8123` (features
  vs ground truth, action mapping, orders) and `python3 -m tools.check_v20_selfplay --port 8123`
  (v21: `tools.check_v21_env`, `tools.check_v21_net`, `tools.check_v20_selfplay --arch v21`).

## Rules for every task
- Count wins with `battle.won` / `info["battle_won"]`, never with the sign of the episode reward.
- A 400-game eval is only good to about ±4.7 points. Don't report differences smaller than the noise.
- Changing the observation layout or action meaning breaks old checkpoints: bump
  `CONFIG["version"]` and add an `ARCH_REGISTRY` entry (SB3), or for v20-style models keep the old
  featurizer importable (checkpoints store `LAYOUT` and refuse to load with a different one).
- Current baseline numbers are in the README's Performance table.

## Read first when working on
- `envs/`, the wrapper, or anything touching poke-env internals: `agent_docs/poke_env_gotchas.md`
- training runs or hyperparameters: `agent_docs/training.md`
- evaluation or Foul Play: `agent_docs/evaluation.md`
- training on Google Cloud: `agent_docs/cloud.md`
- architecture changes or deciding what to do next: `agent_docs/roadmap.md`
- v20 (design, checks, throughput, results): `agent_docs/v20.md`; the step 3 brief it was built
  from: `agent_docs/step3_handoff.md`
- v21 (step 4: attention over the tokens, battle history, auxiliary heads; built and run, no
  gain): `agent_docs/v21.md`; the brief it was built from: `agent_docs/step4_brief.md`
- the v20 A/Bs and how runs are started, monitored and judged on this Mac:
  `agent_docs/step4_handoff.md`
- step 6 (search at decision time on poke-engine, the network as prior, set model and opponent
  model): `agent_docs/step6.md`; the brief it was built from: `agent_docs/step6_brief.md`
- step 7 (search self-play data, distilling the search into the network, the learned leaf
  evaluation): `agent_docs/step7.md`; the brief it was built from: `agent_docs/step7_brief.md`
- step 8 (matchup weights for the leaf evaluation; why better prediction didn't make the search
  stronger): `agent_docs/step8.md`
