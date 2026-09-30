#!/bin/bash
# One-screen status of a v20-style training run: power, swap, the latest update line, every eval,
# the battery runner's messages, problems, and whether the trainer is running.
#   bash tools/train_status.sh logs/train_v20lr3.log
LOG=${1:-logs/train_v20.log}
cd "$(dirname "$0")/.."
echo "== $(date '+%Y-%m-%d %H:%M') | $(pmset -g batt | grep -o "'.*'" | head -1) $(pmset -g batt | grep -o '[0-9]*%' | head -1) | swap used $(sysctl -n vm.swapusage | awk '{print $6}')"
grep -a "^update" "$LOG" | tail -1 | cut -c1-330
grep -a ">>> Eval @" "$LOG" | cut -c1-80
grep -a "\[guard\|\[manual" "$LOG" | tail -4
grep -a "Traceback\|failed\|died\|silent\|abandoned\|STALL" "$LOG" | tail -3 | cut -c1-200
if pgrep -f "[t]raining.ppo_v20" > /dev/null; then echo "trainer running"; else echo "trainer NOT running"; fi
