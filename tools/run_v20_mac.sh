#!/bin/bash
# Run a v20 training job on the MacBook without draining the battery: pause (SIGTERM: the trainer
# saves and exits) when running on battery at or below LOW %, and resume (the trainer resumes from
# its checkpoint) once on AC power or at RESUME % or more. Keeps the Mac awake while training.
# Stops for good when training exits by itself (done) or fails MAX_FAILS times in a row.
#   LOG=logs/train_v20.log bash tools/run_v20_mac.sh --version v20 --workers 6 --servers 3 --stop-at 10000000
LOW=${LOW:-20}
RESUME=${RESUME:-30}
MAX_FAILS=${MAX_FAILS:-3}
LOG=${LOG:-logs/train_v20.log}
cd "$(dirname "$0")/.."

battery() {  # prints "<AC|BAT> <percent>"
  local s; s=$(pmset -g batt)
  local pct; pct=$(echo "$s" | grep -o '[0-9]*%' | head -1 | tr -d %)
  if echo "$s" | grep -q "AC Power"; then echo "AC ${pct:-100}"; else echo "BAT ${pct:-100}"; fi
}

fails=0
while true; do
  read src pct < <(battery)
  if [ "$src" = "BAT" ] && [ "$pct" -lt "$RESUME" ]; then
    echo "[guard $(date '+%H:%M')] on battery at ${pct}%: waiting for power (resume at ${RESUME}% or AC)" >> "$LOG"
    until read src pct < <(battery) && { [ "$src" = "AC" ] || [ "$pct" -ge "$RESUME" ]; }; do sleep 60; done
  fi
  echo "[guard $(date '+%H:%M')] starting training (${src} ${pct}%)" >> "$LOG"
  python3 -m training.ppo_v20 "$@" >> "$LOG" 2>&1 &
  pid=$!
  caffeinate -i -w $pid &
  paused=0
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    read src pct < <(battery)
    if [ "$src" = "BAT" ] && [ "$pct" -le "$LOW" ]; then
      echo "[guard $(date '+%H:%M')] on battery at ${pct}%: pausing training (SIGTERM, it saves first)" >> "$LOG"
      kill -TERM $pid; paused=1
      for i in $(seq 1 180); do kill -0 $pid 2>/dev/null || break; sleep 1; done
      kill -9 $pid 2>/dev/null
      break
    fi
  done
  wait $pid; status=$?
  if [ $paused -eq 1 ]; then continue; fi
  if [ $status -eq 0 ]; then echo "[guard $(date '+%H:%M')] training finished" >> "$LOG"; break; fi
  fails=$((fails + 1))
  echo "[guard $(date '+%H:%M')] training exited with status $status (failure $fails of $MAX_FAILS)" >> "$LOG"
  [ $fails -ge $MAX_FAILS ] && break
  sleep 30
done
