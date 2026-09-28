#!/bin/bash
# On the Mac: copy a run's results from the (running) VM into pokemon-rl/, resuming partial files
# and retrying; plain `gcloud compute scp` failed partway through a 2.4 GB fetch. Puts the models,
# training state, the run's self-play checkpoints, TensorBoard logs and the training log where the
# training and eval scripts expect them. The pool's seed models are left out (they are copies of
# local models).
VERSION=${VERSION:-v19}
INSTANCE=${INSTANCE:-fs-train}
ZONE=${ZONE:-us-west1-a}
cd "$(dirname "$0")/.." || exit 1
IP=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
ID=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --format="value(id)")
USER_ON_VM=${USER_ON_VM:-$USER}
# gcloud's own key and known-hosts entry, so the VM's host key is still checked
SSH="ssh -i $HOME/.ssh/google_compute_engine -o HostKeyAlias=compute.$ID -o UserKnownHostsFile=$HOME/.ssh/google_compute_known_hosts -o StrictHostKeyChecking=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
R=$USER_ON_VM@$IP
fetch() {
  for try in 1 2 3 4 5; do
    rsync -a --partial --timeout=60 -e "$SSH" "$@" && { echo "ok: ${@: -2:1}"; return 0; }
    echo "retry $try: ${@: -2:1}"; sleep 5
  done
  echo "FAILED: ${@: -2:1}"; return 1
}
mkdir -p "models/selfplay_pool_$VERSION" logs
for f in "ppo_pokemon_6v6_${VERSION}_best.zip" "ppo_pokemon_6v6_${VERSION}_latest.zip" \
         "training_state_$VERSION.pkl" "optimizer_state_$VERSION.pt"; do
  fetch "$R:pokemon-rl/models/$f" models/
done
# Only written when a run reaches its last step
rsync -a --partial --timeout=60 -e "$SSH" "$R:pokemon-rl/models/ppo_pokemon_6v6_$VERSION.zip" models/ 2>/dev/null \
  && echo "ok: final model" || echo "no final model (the run was stopped before its last step)"
fetch --exclude='seed_*' "$R:pokemon-rl/models/selfplay_pool_$VERSION/" "models/selfplay_pool_$VERSION/"
fetch "$R:pokemon-rl/logs/" logs/
fetch "$R:train_$VERSION.log" logs/
echo FETCH_DONE
