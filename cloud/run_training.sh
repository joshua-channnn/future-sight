#!/bin/bash
# On the VM: one training run. Starts the Showdown servers, trains (resuming from the last
# checkpoint if one exists), then powers the VM off so it stops billing. Run it inside tmux:
#   tmux new -d -s train 'VERSION=v19 TIMESTEPS=20000000 bash ~/pokemon-rl/cloud/run_training.sh'
# VERSION must match CONFIG["version"]; it names the log (~/train_$VERSION.log) and the markers.
# STOP_AT=10000000 stops right after the first eval at or past that step (the learning-rate
# schedule still follows TIMESTEPS).
VERSION=${VERSION:-v19}
TIMESTEPS=${TIMESTEPS:-20000000}
N_ENVS=${N_ENVS:-20}
STALL_MIN=${STALL_MIN:-20}
STOP_AT=${STOP_AT:-}
LOG=~/train_$VERSION.log
exec > >(tee -a "$LOG") 2>&1

source ~/venv/bin/activate
cd ~/pokemon-showdown && bash ../pokemon-rl/start_servers.sh $((N_ENVS + 1)) > /tmp/start_servers.log 2>&1
for p in $(seq 8000 $((8000 + N_ENVS))); do
  for i in $(seq 1 120); do curl -s "http://localhost:$p" > /dev/null && break; sleep 1; done
done
cd ~/pokemon-rl
python -m training.ppo_embed_train_selfplay --timesteps "$TIMESTEPS" --n_envs "$N_ENVS" &
pid=$!

# Stall guard: SB3 logs a total_timesteps line after every rollout, about 2 minutes apart (about
# 5 around an eval). If none appears for STALL_MIN minutes, kill training and power off rather
# than bill for a hung run; running this script again resumes from the last periodic checkpoint.
last=$(grep -a -c total_timesteps "$LOG"); idle=0; stalled=0; stopped=0
while kill -0 $pid 2>/dev/null; do
  sleep 60
  now=$(grep -a -c total_timesteps "$LOG")
  if [ "$now" -gt "$last" ]; then last=$now; idle=0; else idle=$((idle + 1)); fi
  if [ $idle -ge $STALL_MIN ]; then
    echo "STALL GUARD: no progress for $STALL_MIN minutes, killing training at $(date -u)"
    stalled=1; kill -9 $pid; break
  fi
  if [ -n "$STOP_AT" ] && grep -a -o -E '>>> Eval @ [0-9,]+' "$LOG" | tr -d , \
      | awk -v s="$STOP_AT" '$4 >= s {f = 1} END {exit !f}'; then
    echo "STOP_AT: eval at or past $STOP_AT steps is logged, stopping training at $(date -u)"
    stopped=1; touch ~/${VERSION}_STOPPED_AT_$STOP_AT; kill -9 $pid; break
  fi
done
wait $pid; status=$?
echo "training exited with status $status at $(date -u)"
[ $status -eq 0 ] && [ $stalled -eq 0 ] && [ $stopped -eq 0 ] && touch ~/${VERSION}_DONE
sudo shutdown -h now
