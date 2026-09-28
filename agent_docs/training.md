# Training

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
- `train/approx_kl`: aim for about 0.01. v18.x sat at about 0.0017, which means the policy
  barely moved. If it stays far below 0.01, raise `base_lr_initial`.
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
