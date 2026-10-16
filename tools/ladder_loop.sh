#!/bin/bash
# Plays N ladder games with players/ladder_parallel.py across the Mac's sleeps (agent_docs/ladder.md).
# Whenever the lid has been open for 2 minutes straight, it runs ladder_parallel.py for the games
# the account still has to play, and again each time that exits with 3 (the server dropped the
# connection, or no game started for 10 minutes). The Mac sleeps when the lid closes, whatever
# caffeinate holds, and its brief wakes while closed are too short for games. Run the loop under
# caffeinate -i, or with the lid open on battery the Mac idle-sleeps before a run starts. The target
# counts the account's games on the ladder (pokemonshowdown.com/users/<account>.json), so games lost
# to a dropped connection count too. It stops at the target or on any other exit; after a Python
# traceback that isn't a dropped connection it stops via the stop file (the games in progress
# finish). Extra arguments go to ladder_parallel.py.
#
#   nohup caffeinate -i bash tools/ladder_loop.sh 200 logs/ladder_run > logs/ladder_run_loop.log 2>&1 &
#   touch logs/ladder.stop        # stop after the games in progress
cd "$(dirname "$0")/.."
N=${1:?number of games}; LOGBASE=${2:?log prefix}; shift 2
ACCOUNT=$(grep -E '^ *USERNAME = "' players/ladder_bot.py | sed -E 's/.*"(.*)".*/\1/')
STOP=logs/ladder.stop

games() {
  curl -s -m 20 "https://pokemonshowdown.com/users/$ACCOUNT.json" | python3 -c \
    "import json, sys; r = json.load(sys.stdin)['ratings']['gen9randombattle']; print(r['w'] + r['l'])"
}

lid_open() {
  ioreg -r -k AppleClamshellState -d 4 | grep -q '"AppleClamshellState" = No'
}

unexpected_tracebacks() {   # Python tracebacks in $1 other than a dropped connection
  python3 -c '
import sys
blocks = open(sys.argv[1], errors="replace").read().split("Traceback (most recent call last):")[1:]
print(sum("ConnectionClosed" not in b[:4000] for b in blocks))' "$1"
}

if [ -f $STOP ]; then echo "removing an old $STOP"; rm -f $STOP; fi
start=$(games) || { echo "can't read $ACCOUNT's games"; exit 1; }
target=$((start + N)); part=1
echo "$(date +%T) $ACCOUNT has played $start games; target $target"
while true; do
  ok=0
  until [ $ok -ge 2 ]; do
    if lid_open; then ok=$((ok + 1)); else ok=0; fi
    perl -e 'select(undef, undef, undef, 60)'
  done
  played=$(games) || { echo "$(date +%T) can't read $ACCOUNT's games; retrying"; continue; }
  left=$((target - played))
  if [ $left -le 0 ]; then echo "$(date +%T) done: $played games"; exit 0; fi
  log=${LOGBASE}_part$part.log
  echo "$(date +%T) part $part: $left games left -> $log"
  caffeinate -i -s python3 -u -m players.ladder_parallel --games $left --stop-file $STOP "$@" \
    > $log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    perl -e 'select(undef, undef, undef, 30)'
    if [ ! -f $STOP ] && [ "$(unexpected_tracebacks $log)" -gt 0 ]; then
      echo "$(date +%T) a traceback in $log: no new games"; touch $STOP
    fi
  done
  wait $pid; code=$?
  echo "$(date +%T) part $part exited with $code"
  [ $code -eq 3 ] || exit $code
  part=$((part + 1))
done
