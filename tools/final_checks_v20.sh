#!/bin/bash
# Final checks for a v20 model (step3_handoff.md section 2): 1,000 games vs SimpleHeuristics and
# head-to-heads vs v19 and v18.3 with 1,000 games in each seat, both sides greedy, run in parallel on
# three Showdown servers (ports 8000-8002 by default; nothing else may be using them).
#   MODEL=models/eval_snapshots_v20/step_0010012345.pt bash tools/final_checks_v20.sh
# Without --opp-deterministic the opponent model samples its actions while ours plays greedily,
# which favors our side by ~18 points against an identical model (evaluation.md).
MODEL=${MODEL:?set MODEL to a v20 .pt checkpoint}
N=${N:-1000}
P1=${P1:-8000}; P2=${P2:-8001}; P3=${P3:-8002}
OUT=${OUT:-logs/final_checks_$(basename "$MODEL" .pt)}
mkdir -p "$OUT"
cd "$(dirname "$0")/.."
python3 -m eval.ppo_test --model "$MODEL" --opponents simple --battles $N --port $P1 > "$OUT/vs_simple.txt" 2>&1 &
python3 -m eval.ppo_test --model "$MODEL" --opponents model --opp-model models/ppo_pokemon_6v6_v19_best.zip \
    --opp-deterministic --battles $N --seat both --port $P2 > "$OUT/vs_v19.txt" 2>&1 &
python3 -m eval.ppo_test --model "$MODEL" --opponents model --opp-model models/ppo_pokemon_6v6_v18.3_best.zip \
    --opp-deterministic --battles $N --seat both --port $P3 > "$OUT/vs_v18.3.txt" 2>&1 &
wait
for f in vs_simple vs_v19 vs_v18.3; do
  echo "== $f"; grep -E "Win rate|Both seats" "$OUT/$f.txt"
done
