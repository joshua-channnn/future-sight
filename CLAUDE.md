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

v17-v19 (SB3):
- `envs/rl_player_6v6_v18.py`: the environment. Observation (677 floats: 12 Pokémon blocks,
  active-move details, global features), reward (`reward_mode`: `sparse` or `shaped`), action
  masks, and the 13-action mapping (4 moves, 4 tera moves, 5 switches).
- `envs/wrappers.py`: `MaskablePokeEnvWrapper`, which sits between poke-env and SB3.
- `networks/embedding_extractor_v18.py`: feature extractor (embeddings, per-Pokémon encoder, pooling).
- `training/ppo_embed_train_selfplay.py`: the only training script; all settings live in `CONFIG`.
- `players/ppo_player.py`: loads any saved model (env/extractor chosen via `ARCH_REGISTRY`). Used
  for self-play opponents, `players/play_bot.py` (local play) and `players/ladder_bot.py`.
- `eval/ppo_test.py` (vs Random / SimpleHeuristics / another model) and `eval/foulplay_eval.py`.
- `envs/rl_player_6v6.py`, `networks/embedding_extractor.py`, `networks/embedding_extractor_sp.py`:
  v17 code, kept only so v17 checkpoints still load.
- `cloud/`: scripts for training on a Google Cloud VM (pack, set up, run with a stall guard, fetch).
- Outside this repo: `../pokemon-showdown` (local Showdown server, a Feb 2026 fork) and
  `../foul-play` (Foul Play, locally patched, run via Docker).
- `models/` is gitignored except `ppo_pokemon_6v6_v20.pt` (the best model: v20 at 20M steps) and
  `ppo_pokemon_6v6_v18.3_best.zip` (the best SB3 model). v20 checkpoints are torch files:
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
  must repeat the same `--set` flags or it refuses to start). On the MacBook, `tools/run_v20_mac.sh` wraps it: it pauses the run
  when the battery gets low and resumes on power. Each Showdown server is ~10 node processes and
  ~1.2 GB; don't leave unused ones running on the 16 GB Mac.
- Evaluate a v20 model: `python3 -m eval.ppo_test --model models/X.pt --opponents simple --battles 1000 --port 8123`
  (head-to-heads: `--opponents model --opp-model models/Y --opp-deterministic --seat both`;
  without `--opp-deterministic` the opponent samples its actions, which inflates our win rate).
- There is no test suite. To check a change: `python3 -m py_compile` the touched files, then run a
  short eval, `python3 -m eval.ppo_test --opponents simple --battles 400 --port 8123`. For v20 env
  or trainer changes also run `python3 -m tools.check_v20_env --games 50 --port 8123` (features
  vs ground truth, action mapping, orders) and `python3 -m tools.check_v20_selfplay --port 8123`.

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
- building step 4 (v21: attention over the tokens, battle history, auxiliary heads), the next job:
  `agent_docs/step4_brief.md`
- the v20 A/Bs and how runs are started, monitored and judged on this Mac:
  `agent_docs/step4_handoff.md`
