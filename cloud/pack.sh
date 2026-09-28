#!/bin/bash
# On the Mac: build the two upload archives for a cloud run, pokemon-rl.tar (code plus the seed
# models the training script copies into a fresh pool) and showdown.tar (the Showdown fork's source).
# Archives land in $OUT (default: the current directory).
set -euo pipefail
OUT=${OUT:-$PWD}
cd "$(dirname "$0")/../.."
# Without this, macOS tar adds ._* metadata files, and the Showdown build fails on them
export COPYFILE_DISABLE=1
tar cf "$OUT/pokemon-rl.tar" --exclude=pokemon-rl/models --exclude=pokemon-rl/logs \
    --exclude=.git --exclude=__pycache__ pokemon-rl
# seed_models in training/ppo_embed_train_selfplay.py
tar rf "$OUT/pokemon-rl.tar" pokemon-rl/models/ppo_pokemon_6v6_{v17.2,v17.3sp,v18.1,v18.2,v18.3}_best.zip
tar cf "$OUT/showdown.tar" --exclude=.DS_Store \
    pokemon-showdown/{build,config,data,lib,sim,server,tools,translations,package.json,package-lock.json,pokemon-showdown,tsconfig.json}
ls -lh "$OUT/pokemon-rl.tar" "$OUT/showdown.tar"
