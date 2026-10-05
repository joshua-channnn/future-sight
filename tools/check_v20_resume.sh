#!/bin/bash
# Kill-and-resume check for training/ppo_v20.py (milestone 4 of agent_docs/step3_handoff.md):
# start a small run, SIGKILL it after its second checkpoint, restart it, and check that it resumes
# from that checkpoint's step and keeps going. Needs a Showdown server on $PORT. ARCH=v21 checks a
# v21 run (aux labels, base-rate tables in the checkpoint).
#   PORT=8123 bash tools/check_v20_resume.sh
#   ARCH=v21 PORT=8123 bash tools/check_v20_resume.sh
set -u
PORT=${PORT:-8123}
ARCH=${ARCH:-v20}
V=resumecheck_$RANDOM
LOG=/tmp/$V.log
cd "$(dirname "$0")/.."
ARGS="--arch $ARCH --version $V --workers 2 --servers 1 --base-port $PORT --eval-port $PORT --steps-per-update 2048 --save-freq 2000 --eval-freq 6000 --eval-episodes 10"
python3 -m training.ppo_v20 $ARGS --stop-at 200000 > $LOG 2>&1 &
pid=$!
until [ "$(grep -c '>>> Checkpoint' $LOG)" -ge 2 ]; do sleep 2; kill -0 $pid 2>/dev/null || break; done
saved=$(grep '>>> Checkpoint' $LOG | tail -1 | grep -o '[0-9,]* steps' | tr -d ', steps')
kill -9 $pid  # its workers exit when their pipe to the learner closes
sleep 2
echo "killed after the checkpoint at $saved steps"
python3 -m training.ppo_v20 $ARGS --stop-at $((saved + 4000)) > $LOG.2 2>&1
resumed=$(grep -o 'Resumed .* at [0-9,]* steps' $LOG.2 | grep -o 'at [0-9,]*' | tr -d 'at ,')
final=$(grep 'total_timesteps' $LOG.2 | tail -1 | grep -o 'total_timesteps [0-9,]*' | tr -d 'total_timesps ,')
echo "resumed at ${resumed:-?} steps, reached ${final:-?}"
if [ "${resumed:-x}" = "$saved" ] && [ "${final:-0}" -gt "$saved" ]; then echo "RESULT: resume ok"; else echo "RESULT: FAILED"; tail -20 $LOG.2; fi
rm -rf models/ppo_pokemon_6v6_${V}_* models/selfplay_pool_$V models/eval_snapshots_$V logs/$V $LOG $LOG.2
