#!/bin/bash
# On a fresh Ubuntu 24.04 VM, once: unpack the archives from cloud/pack.sh (uploaded to ~) and
# install Showdown and the Python environment. Takes a few minutes.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
cd ~
sudo apt-get update -q > /dev/null
sudo apt-get install -y -q python3-venv python3-pip nodejs npm git tmux curl rsync > /dev/null
tar xf pokemon-rl.tar
tar xf showdown.tar
# In case the archives were made without COPYFILE_DISABLE: ._* files break the Showdown build
find ~/pokemon-showdown ~/pokemon-rl -name '._*' -delete
cd ~/pokemon-showdown
npm install --no-audit --no-fund --loglevel=error > /dev/null
# Build once here: every `pokemon-showdown start` rebuilds, and many servers starting together on
# an unbuilt tree would all write dist/ at once
node build > /dev/null
cd ~
python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q poke-env==0.11.0 stable-baselines3==2.7.1 sb3-contrib==2.7.1 gymnasium==1.2.3 tensorboard tqdm rich
echo "node $(node --version)"
python -c "import torch, stable_baselines3, sb3_contrib, poke_env; print('torch', torch.__version__, '| sb3', stable_baselines3.__version__, '| sb3-contrib', sb3_contrib.__version__)"
echo SETUP_DONE
