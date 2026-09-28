# CLAUDE.md

Future Sight is a PPO self-play bot for Pokémon Showdown **Gen 9 Random Battles** (singles),
built on poke-env and Stable-Baselines3 `MaskablePPO`. The goal is a stronger bot, measured by
*true* win rate against fixed opponents (SimpleHeuristics, older models, Foul Play).

## Map
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
- `models/` is gitignored except `ppo_pokemon_6v6_v18.3_best.zip`, the current baseline.

## Running things
- Use `python3` (3.14, with poke-env 0.11 and SB3 2.7.1). `python` is Anaconda and lacks the deps.
  Run from this directory as modules, e.g. `python3 -m eval.ppo_test`.
- Everything needs a local Showdown server:
  `cd ../pokemon-showdown && node pokemon-showdown start --no-security --port 8123`.
- Training occupies ports 8000-8004. Use another port (e.g. 8123) for experiments and evals so you
  never collide with a running training job; `eval/ppo_test.py` defaults to 8004.
- There is no test suite. To check a change: `python3 -m py_compile` the touched files, then run a
  short eval, `python3 -m eval.ppo_test --opponents simple --battles 400 --port 8123`.

## Rules for every task
- Count wins with `battle.won` / `info["battle_won"]`, never with the sign of the episode reward.
- A 400-game eval is only good to about ±4.7 points. Don't report differences smaller than the noise.
- Changing the observation layout or action meaning breaks old checkpoints: bump
  `CONFIG["version"]` and add an `ARCH_REGISTRY` entry.
- Current baseline numbers are in the README's Performance table.

## Read first when working on
- `envs/`, the wrapper, or anything touching poke-env internals: `agent_docs/poke_env_gotchas.md`
- training runs or hyperparameters: `agent_docs/training.md`
- evaluation or Foul Play: `agent_docs/evaluation.md`
- training on Google Cloud: `agent_docs/cloud.md`
- architecture changes or deciding what to do next: `agent_docs/roadmap.md`
