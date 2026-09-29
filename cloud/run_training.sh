#!/bin/bash
# On the VM: one training run. Starts the Showdown servers, trains (resuming from the last
# checkpoint if one exists), then powers the VM off so it stops billing. Run it inside tmux:
#   tmux new -d -s train 'VERSION=v20 TIMESTEPS=20000000 STOP_AT=10000000 bash ~/pokemon-rl/cloud/run_training.sh'
# VERSION names the log (~/train_$VERSION.log), the model files and the markers. v20 and later use
# training/ppo_v20.py; VERSION=v19 runs the SB3 script (its CONFIG["version"] must match).
# STOP_AT=10000000 stops right after the first eval at or past that step (the learning-rate
# schedule still follows TIMESTEPS).
VERSION=${VERSION:-v20}
TIMESTEPS=${TIMESTEPS:-20000000}
STALL_MIN=${STALL_MIN:-20}
STOP_AT=${STOP_AT:-}
LOG=~/train_$VERSION.log
exec > >(tee -a "$LOG") 2>&1

source ~/venv/bin/activate
if [ "$VERSION" = "v19" ]; then
  N_ENVS=${N_ENVS:-20}
  N_SERVERS=$((N_ENVS + 1))                    # one per env, plus the eval env's
  CMD="python -m training.ppo_embed_train_selfplay --timesteps $TIMESTEPS --n_envs $N_ENVS"
else
  # v20 workers don't step in lockstep, so they share servers; the eval uses the next port
  N_WORKERS=${N_WORKERS:-24}
  N_SERVERS=${N_SERVERS:-8}
  CMD="python -m training.ppo_v20 --version $VERSION --timesteps $TIMESTEPS --workers $N_WORKERS --servers $N_SERVERS"
  N_SERVERS=$((N_SERVERS + 1))
fi
cd ~/pokemon-showdown && bash ../pokemon-rl/start_servers.sh $N_SERVERS > /tmp/start_servers.log 2>&1
for p in $(seq 8000 $((8000 + N_SERVERS - 1))); do
  for i in $(seq 1 120); do curl -s "http://localhost:$p" > /dev/null && break; sleep 1; done
done
cd ~/pokemon-rl
$CMD &
pid=$!

# Stall guard: both trainers log a total_timesteps line after every update, a minute or two apart
# (more around an eval in the SB3 script). If none appears for STALL_MIN minutes, kill training and
# power off rather than bill for a hung run; running this script again resumes from the last
# periodic checkpoint.
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
    stopped=1; touch ~/${VERSION}_STOPPED_AT_$STOP_AT; kill -TERM $pid; sleep 120; kill -9 $pid 2>/dev/null; break
  fi
done
wait $pid; status=$?
echo "training exited with status $status at $(date -u)"
[ $status -eq 0 ] && [ $stalled -eq 0 ] && [ $stopped -eq 0 ] && touch ~/${VERSION}_DONE
sudo shutdown -h now
