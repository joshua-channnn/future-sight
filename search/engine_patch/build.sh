#!/bin/bash
# Builds poke-engine 0.0.48 with gen9 + Terastallization and two additions into this python3:
# the emm_matrix binding (emm_matrix.patch, step 6) and a learned leaf evaluation
# (learned_eval.patch, step 7: features_from_strings, feature_names, load_evaluator,
# evaluator_values, mcts_with; search/leaf_eval.py). The engine's own search is unchanged.
# Rust (1.93) is in ~/.cargo/bin.
#   bash search/engine_patch/build.sh                      # build and install
#   WHEEL_DIR=/some/dir bash search/engine_patch/build.sh  # only build the wheel into that dir
# Back to the PyPI version: python3 -m pip install poke-engine==0.0.46 (or ==0.0.48 for the
# unpatched sdist, built with the same --config-settings).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
cd "$WORK"
python3 -m pip download --no-deps --no-binary :all: poke-engine==0.0.48 -d . > /dev/null
tar xzf poke_engine-0.0.48.tar.gz
cd poke_engine-0.0.48
patch -p1 < "$HERE/emm_matrix.patch"
patch -p1 < "$HERE/learned_eval.patch"
FEATURES="build-args=--features poke-engine/terastallization --no-default-features"
if [ -n "${WHEEL_DIR:-}" ]; then
    PATH="$HOME/.cargo/bin:$PATH" python3 -m pip wheel . --no-deps -w "$WHEEL_DIR" --config-settings="$FEATURES"
else
    PATH="$HOME/.cargo/bin:$PATH" python3 -m pip install . --config-settings="$FEATURES"
    python3 -c "import poke_engine as pe; assert hasattr(pe, 'emm_matrix') and hasattr(pe, 'mcts_with'); print('poke-engine with emm_matrix and the learned evaluation installed')"
fi
rm -rf "$WORK"
