"""Throughput of the SB3 (v19) training loop, for comparisons with training/ppo_v20.py.

Runs training.ppo_embed_train_selfplay for a few updates under a throwaway version and temporary
directories, so no v19 model, pool or training-state file is touched. Needs Showdown servers on
ports 8000..8000+n_envs-1. Prints SB3's log (time/fps includes the PPO update).

    python3 -m tools.bench_sb3 --n_envs 4 --updates 2
"""
import argparse
import os
import shutil
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--updates", type=int, default=2)
    args = ap.parse_args()
    import training.ppo_embed_train_selfplay as T
    tmp = tempfile.mkdtemp(prefix="sb3bench_")
    cfg = dict(T.CONFIG)
    cfg.update(version="sb3bench", n_envs=args.n_envs, model_dir=os.path.join(tmp, "models"),
               log_dir=os.path.join(tmp, "logs"), pool_dir=os.path.join(tmp, "pool"),
               total_timesteps=cfg["steps_per_update"] * args.updates,
               eval_freq=10 ** 12, save_freq=10 ** 12, pool_save_freq=10 ** 12)
    # same 20M learning-rate horizon as the real runs: only the step count is cut short
    cfg["total_timesteps"] = cfg["steps_per_update"] * args.updates
    try:
        T.train(cfg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
