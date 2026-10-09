#!/bin/bash
# Builds poke-engine 0.0.48 with gen9 + Terastallization and the emm_matrix binding
# (emm_matrix.patch) into this python3. Rust (1.93) is in ~/.cargo/bin.
#   bash search/engine_patch/build.sh
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
PATH="$HOME/.cargo/bin:$PATH" python3 -m pip install . \
    --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
python3 -c "import poke_engine as pe; assert hasattr(pe, 'emm_matrix'); print('poke-engine with emm_matrix installed')"
rm -rf "$WORK"
