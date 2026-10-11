# Evaluation

## Against Random, SimpleHeuristics or another model
`python3 -m eval.ppo_test --model models/X --opponents simple --battles 1000 --port 8123`
- `--opponents` is `random`, `simple`, `model` (needs `--opp-model models/Y`) or `all`.
- The architecture is detected from the checkpoint, and wins are counted with `battle.won`.
- The default port is 8004, which a running training job uses for its eval env; pass `--port`.

## v20 models
`python3 -m eval.ppo_test --model models/X.pt --opponents simple --battles 1000 --port 8123`
- `.pt` files go through `eval/v20_eval.py` (the v20 env, deterministic v20 policy).
- `--opponents model --opp-model` takes a v20 `.pt` or an SB3 zip; `--seat both` plays the given
  number of games in each seat (1: the challenger, p1; 2: the challenged side) and reports both.
- **Pass `--opp-deterministic` for head-to-heads.** Without it the opponent model samples its
  actions while `--model` plays greedily (`--seat both` swaps seats, not roles), and the greedy
  side wins far more: v20@5M greedy vs the same model sampling 67.7% ± 2.0%; v20lr3@5M vs v20@5M
  65.6% with v20 sampling but 47.3% ± 2.2% with both greedy (2,000 games each, September 30, 2026).
  The deployed players (`ladder_bot.py`, `foulplay_eval.py`) are greedy, and the SB3 path of
  `ppo_test.py` always made the opponent greedy, so greedy vs greedy is the comparable number.
  The v20 head-to-heads first reported (79.3% vs v19, 70.1% vs v18.3) were greedy vs sampling;
  with both greedy they are 73.9% ± 1.9% and 62.8% ± 2.1%. `tools/final_checks_v20.sh` now passes
  the flag.
- An SB3 model can also be evaluated against a v20 opponent (`--opp-model models/Y.pt`).
- A battle that stalls for 60 s ends the whole eval with `EnvTimeout` (training rebuilds the env
  instead; the eval path has no recovery). It's rare, about once in 20 runs of 1,000-2,000 games:
  rerun it.
- Every eval env handles requests after the turn's log (`poke_env_gotchas.md` item 9).

## How many games
95% CI ≈ 1.96·sqrt(p(1−p)/n). At p ≈ 0.65 that's ±6.6 points for 200 games, ±4.7 for 400 and
±3.0 for 1,000. Detecting a 3-point gain takes about 3,800 games per model, or about 2,200
head-to-head games against the previous model.

## Against Foul Play
Full steps are in the docstring of `eval/foulplay_eval.py`. In short: start Docker Desktop, build
the image in `../foul-play`, run the container in `accept_challenge` mode with a username that
isn't registered on the main Showdown server, then run the script. A game takes about 11 seconds
at Foul Play's default search settings.
- The script challenges one game at a time: `Player.send_challenges(n)` sends the next challenge
  while a battle is still running, and Foul Play ignores it, which deadlocks.
- `--model` takes a v20 `.pt` (the default is `models/ppo_pokemon_6v6_v20.pt`; it plays with
  `hold_requests=True`, see gotcha 9) or an SB3 zip.
- If `docker` commands return "500 Internal Server Error" and Docker's host log
  (`~/Library/Containers/com.docker.docker/Data/log/host/com.docker.backend.log`) shows "no route
  to host" to 192.168.65.7, Docker Desktop's VM didn't come up: quit Docker Desktop
  (`osascript -e 'quit app "Docker Desktop"'`) and open it again.
- Foul Play's engine panics now and then (`PanicException: Invalid PokemonIndex: 6`, about once
  per 500 games in October 2026) and the container exits. The script then stops after
  `--game-timeout` (a challenge to a user who left is never answered) and prints the games so far;
  `tools/foulplay_loop.sh N PREFIX CONFIG` runs the search bot until N games have finished,
  restarting Foul Play after each crash. Run the container without `--rm` to keep its log.
- Foul Play targets current Showdown. Against this Feb 2026 fork it needs the local patches in
  `../foul-play`: request/log reordering in `fp/websocket_client.py` and the Snow -> Snowscape
  rename in `fp/battle/protocol.py`. If it crashes on another mismatch, running it against an
  up-to-date Showdown server is cleaner than adding more patches.

## Reference points
- v18.3_best (true wins): see the README's Performance table. Against Foul Play it won 2 of 10,
  too few games to say more than that Foul Play is much stronger.
- v20 at 10M (`models/eval_snapshots_v20/step_0010034773.pt`; the best model then) vs Foul Play
  at its default search settings (100 ms per sampled state): 27 of 200 = 13.5% ± 4.7% (September
  30, 2026; 8 of the first 100, 19 of the second). Foul Play is still far stronger, roughly 320
  Elo at these settings. The 20M model (now `models/ppo_pokemon_6v6_v20.pt`, 52.3% vs the 10M
  one) hasn't played it; a gain that small wouldn't show in 200 games.
- The search bots at Foul Play's budget (2 x 100 ms), 1,000 games each: step 6's best 47.4% ± 3.1%
  (`agent_docs/step6.md`), step 7's with the learned leaf evaluation 50.9% ± 3.1%
  (`agent_docs/step7.md`).
- Public ladder: Jaxcalibur peaked at 2557 Elo (#1, Aug 2026); Foul Play peaked at 2341 Elo / 88% GXE.
