#!/bin/bash
# Runs eval/foulplay_eval.py --model search until TARGET games have finished, restarting Foul Play
# when it crashes (its engine panics now and then, "Invalid PokemonIndex: 6", about once per 500
# games; the game in progress is dropped). Needs Docker Desktop, the foul-play:latest image and a
# Showdown server on port 8123; the search bot gets the Mac to itself (wall-clock budget).
#   bash tools/foulplay_loop.sh 1000 foulplay_leaf '{"prior_weight": 0.15, "set_gamma": 0.5, "infer_items": true, "net": true, "leaf_eval": "models/leaf_eval/lin1_86.npz", "leaf_mode": 2, "leaf_scale": 1.4}'
# One log per container, logs/<PREFIX>_run<k>.log; a line per container and the total on stdout.
set -u
TARGET=$1; PREFIX=$2; CFG=$3
cd "$(dirname "$0")/.."
mkdir -p logs
done_n=0; wins=0; run=0
while [ "$done_n" -lt "$TARGET" ]; do
    run=$((run + 1))
    name="fp-${PREFIX}-${run}"
    user="FoulPlayLocal$((RANDOM % 9000 + 1000))"   # a fresh name each time: the last one may still be logged in
    docker rm -f "$name" >/dev/null 2>&1
    docker run -d --name "$name" foul-play:latest --websocket-uri ws://host.docker.internal:8123/showdown/websocket \
        --ps-username "$user" --bot-mode accept_challenge --pokemon-format gen9randombattle \
        --run-count 2000 --log-level INFO >/dev/null
    sleep 15
    log="logs/${PREFIX}_run${run}.log"
    python3 -u -m eval.foulplay_eval --model search --config "$CFG" --games $((TARGET - done_n)) --port 8123 \
        --opponent "$user" --game-timeout 120 > "$log" 2>&1
    line=$(grep "^vs Foul Play" "$log" | tail -1)
    w=$(echo "$line" | sed -E 's/^vs Foul Play: ([0-9]+)\/([0-9]+).*/\1/')
    n=$(echo "$line" | sed -E 's/^vs Foul Play: ([0-9]+)\/([0-9]+).*/\2/')
    if [ -z "$line" ] || [ -z "$n" ]; then echo "run $run: no result line"; w=0; n=0; fi
    wins=$((wins + w)); done_n=$((done_n + n))
    echo "$(date '+%m-%d %H:%M') run $run: $w/$n | total $wins/$done_n | container" \
        "$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$name" 2>/dev/null)"
    docker logs "$name" 2>&1 | grep -m1 -i "panic\|error" | cut -c1-200
    docker rm -f "$name" >/dev/null 2>&1
    if [ "$n" -eq 0 ] && [ "$run" -ge 3 ]; then echo "no progress: stopping"; break; fi
done
echo "TOTAL $wins/$done_n"
