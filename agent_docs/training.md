# Training

This page is about the SB3 trainer (v17-v19). For v20 (`training/ppo_v20.py`) see the
"v20" section at the end and `agent_docs/v20.md`.

## Running
- Start 5 servers (ports 8000-8003 for 4 training envs, 8004 for the eval env) from the Showdown
  directory: `cd ../pokemon-showdown && ../pokemon-rl/start_servers.sh 5`. Logs go to
  `/tmp/showdown_<port>.log`; stop them with `../pokemon-rl/start_servers.sh 5 stop`.
- Then, from this directory:
  `python3 -m training.ppo_embed_train_selfplay [--base_model models/X --reset_value_head] [--n_envs N] [--timesteps T]`.
  `--n_envs N` needs servers on ports 8000..8000+N.
- A run resumes automatically when `models/training_state_<version>.pkl` and
  `models/ppo_pokemon_6v6_<version>_latest.zip` exist for `CONFIG["version"]`.
- Use `--reset_value_head` when warm-starting after the reward has changed: it keeps the policy
  weights and re-initializes the value network.

## Settings (`CONFIG` in `training/ppo_embed_train_selfplay.py`)
- The PPO values are the full bundle from Wang (2024, MIT thesis, Table A.3), tuned together with
  a sparse ±1 reward and about 40K steps per update. Change them together, not one at a time.
- `steps_per_update` is split across the envs.
- Opponents: `heuristic_schedule` sets the SimpleHeuristics share (10%). The rest come from the
  snapshot pool (saved every `pool_save_freq`, at most `max_pool_size`, seeded with older best
  models); `latest_opponent_ratio` of those games use the newest checkpoint. Each env keeps up to
  `opponent_cache_size` pool models loaded (~19 MB each); keep it at least `max_pool_size`, since
  every reload stalls all envs.

## What to watch (TensorBoard, `logs/`)
- `train/approx_kl` and `train/clip_fraction`: the usual "aim for about 0.01" assumes a clip
  range near 0.2. With Wang's 0.0829 the clip binds first: v18.x and v20 sat at about 0.0017
  early, and tripling the LR (v20lr3, `v20.md`) only took kl to 0.005-0.006 over the first 3M
  while the clip fraction rose to 0.3-0.4 and learning slowed (37% vs 54% at 1M); it ended
  level with v20 at 10M. A low kl alone is no reason to raise the LR. v20's clip fraction:
  0.10-0.14 early, 0.03 by 10M.
- `eval/win_rate`: true wins over 500 games vs SimpleHeuristics, every `eval_freq` steps.
- `train/win_rate`: true wins against the training opponent mix.
- Throughput with 4 envs on the MacBook Air: collection ran at about 450 steps/s in a one-update
  benchmark, so a 41K-step update took about 2 minutes (about 240 steps/s and 3.3 minutes before
  the encoder was batched). The fanless Air throttles under sustained load, so speeds drift;
  compare them with back-to-back runs.

## Outputs
- `models/ppo_pokemon_6v6_<version>_latest.zip`, `..._best.zip`, `models/selfplay_pool_<version>/`,
  `models/training_state_<version>.pkl`, `models/optimizer_state_<version>.pt`.
- `_best` is the maximum over many noisy evals. Re-evaluate it with 1,000+ games before trusting it.

## v20 (`training/ppo_v20.py`)
- Servers: `--servers S` servers on ports 8000..8000+S-1 for the workers, plus port 8000+S for
  evals (start them one at a time: every `pokemon-showdown start` rebuilds).
- Run: `python3 -m training.ppo_v20 --workers 6 --servers 3 --stop-at 10000000`. It resumes from
  `models/ppo_pokemon_6v6_<version>_latest.pt` when that exists (`--version` picks the name). On
  the MacBook use `tools/run_v20_mac.sh` with the same arguments (pauses on low battery).
- One stdout line per update (`update N | total_timesteps ... | fps ... | kl ... | win ...`), and
  `>>> Eval @ <step>: <rate> (<wins>/<games>)` when an eval finishes; evals run in their own
  process while training continues.
- Settings: `CONFIG` in `training/ppo_v20.py`; override any entry with `--set KEY=VALUE`
  (repeatable; a resumed run needs the same flags). The PPO values are v19's (Wang's bundle); v20 adds
  `zero_avoid_coef` (0.005), `update_device` (the GPU when there is one) and the opponent mix
  (`heuristic_ratio`, `latest_opponent_ratio`, where "latest" is the live policy with both sides
  recorded).
- TensorBoard: `logs/<version>/`, with SB3's tag names (`train/approx_kl`,
  `train/clip_fraction`, `train/explained_variance`, `rollout/ep_len_mean`, `time/fps`,
  `eval/win_rate`) plus `rollout/win_rate_vs_{live,pool_sb3,pool_v20,heuristic}`,
  `train/zero_avoid_kl`, `time/collect_s`, `time/update_s`.
- Throughput on the MacBook Air (alternating runs, September 2026): 6 workers on 3 servers
  collect 835-1,164 samples/s (720-950 including the 7-8 s MPS update); the SB3 loop with 4 envs
  collects 274-427 (237-384 with updates).
