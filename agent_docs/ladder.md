# The search bot on the public ladder

The deployed search bot (`players/ladder_bot.py`: step 7's lin1_86 leaf evaluation, 2 x 250 ms per
decision, the v21 prior and set model, item inference) on the public Showdown server's Gen 9 Random
Battle ladder, in three runs:

| run | games | account record | Elo | GXE | Glicko |
| --- | --- | --- | --- | --- | --- |
| 1: October 14-15, 2026, the fork's set prior | 100 | 69-31 | 1162 -> 1869 | 38 -> 89.7 | 1408 ± 35 -> 1910 ± 42 |
| 2: October 15-16, the current generator's set prior | 200 | 112-88 | 1869 -> 2227 | 89.7 -> 88.5 | 1910 ± 42 -> 1885 ± 25 |
| 3: October 16, the same, recording every decision | 100 | 53-46 | 2227 -> 2258 | 88.5 -> 88.6 | 1885 ± 25 -> 1885 ± 25 |

After run 3 the account was 118th in the ladder's top 500 (Elo 2258; #1 had 2535, #100 2274 and
#500 2107). The rating has settled: the bot won 53% of run 3's games and 50% of run 2's last 46,
and Glicko (which GXE comes from) has not moved since the end of run 1. Elo, which the ladder ranks
by, rose until the bot won about half of its games.

## Run 1: October 14-15 (100 games, the fork's set prior)

**Result: 69 wins, 31 losses. Elo 1162 -> 1869, GXE 38 -> 89.7, Glicko 1408 ± 35 -> 1910 ± 42.**
Before this run the account had played 182 games with the PPO models (71-111). The ladder's top
500 started at Elo 2106 at the time (the top player had 2523), so the bot wasn't in it yet.

The bot's own record counts 98 real games (67-31). The other 2 ended by turn 2, when the opponent
forfeited: both count as wins on the account, whose record went from 71-111 to 140-142.

| games | wins |
| --- | --- |
| 1-25 | 23 (92%) |
| 26-50 | 18 (72%) |
| 51-75 | 13 (52%) |
| 76-98 | 13 of 23 (57%) |

The win rate falls as the rating catches up with the bot. From about game 50 its opponents are near
its own rating. At 57% over the last 23 games, the rating was probably still rising a little.

- Games averaged 25.1 turns. The logs show no errors and no invalid choices.
- The first 3 games ran one at a time (`players/ladder_bot.py`). The other 97 ran three at a time
  (`players/ladder_parallel.py`) and took 1 h 53 min.
- Logs: `logs/ladder_search_20261014.log` (games 1-3) and `logs/ladder_search_20261014_par.log`
  (the other 97; its own count restarts at 1).

## The set prior from the current generator

The search draws the opponent's hidden sets from a prior: the complete sets of 1M teams from a
random-battle generator (`search/sampling.py`). The fork's generator dates from February 9, 2026,
and Showdown's has changed since. The clone of smogon/pokemon-showdown at 3a766f2 (October 13,
2026) has:
- 2 new species (Bisharp, Stantler) and 39 level changes;
- different sets for 211 of the 507 species (by more than 2% total variation). Those species make up
  40% of the Pokémon the generator produces;
- one new move (Burning Jealousy), item (Expert Belt) and ability (Gale Wings), all of which
  poke-engine knows.

`data/gen9_randbats_set_counts_live.json` is that generator's prior: 1M teams, 509 species, 5,846
distinct sets. `tools/check_prior_replays.py` reads 300 recent public replays the way the bot reads
its own games, which gives 3,030 opponent Pokémon:

| prior | Pokémon none of its sets fit | log p |
| --- | --- | --- |
| the fork's (Feb 2026) | 305 (10.1%): level 225, moves 32, ability 19, species 15, tera 8, item 6 | -0.750 |
| current (Oct 2026) | 29 (1.0%): ability 21, moves 8 | -0.733 |

- The current prior's misses come from reading the log, and the fork's prior has them too:
  - Gardevoir's Trace and Calyrex's As One show another ability;
  - Terapagos changes ability with its forms;
  - Ditto transforms.
- What a miss does in the bot:
  - a level miss keeps all of the species' sets, whose stats are then a level off;
  - a species miss gets a stand-in set: the moves it has shown, neutral defaults otherwise;
  - a move, ability, item or tera miss drops constraints.
- log p: the mean log share of the species' sets that fit what was shown, on the 2,725 Pokémon both
  priors fit.
- Nothing needs retraining, since the search reads the prior at decision time. The v21 set model,
  which reweights the candidate sets, learned the February sets, but it can only reweight the
  candidates the prior gives it.
- Check: 30 local games of the ladder bot with the new prior against SimpleHeuristics went 28-2,
  with no errors. The local server still produces the fork's sets, so this only shows that it runs.

To rebuild it (a few minutes):

    git clone --depth 1 https://github.com/smogon/pokemon-showdown <dir>
    cd <dir> && npm ci && node build && cd -
    python3 -m tools.build_set_prior --showdown <dir> --out data/gen9_randbats_set_counts_live.json

Building and running a fresh clone of outside code needs the user: an agent's attempt was blocked.

## Run 2: October 15-16 (200 games, the current set prior)

`players/ladder_parallel.py`, three games at a time with `data/gen9_randbats_set_counts_live.json`,
under `tools/ladder_loop.sh`.

**Result: the account went 112-88 (140-142 -> 252-230). Elo 1869 -> 2227 (2258 after the first 70
games), GXE 89.7 -> 88.5, Glicko 1910 ± 42 -> 1885 ± 25; 172nd in the top 500.**

The bot's own record counts 196 complete games, 110-86 (56%). Of the other 4, 2 ended by turn 2
when the opponent forfeited (wins on the account) and 2 were lost when the Mac slept (losses).

| games | wins |
| --- | --- |
| 1-50 | 31 (62%) |
| 51-100 | 29 (58%) |
| 101-150 | 27 (54%) |
| 151-196 | 23 of 46 (50%) |

- The ladder can't show whether the new prior helped: the opponents got stronger as the rating
  rose, and a few points of win rate are within its noise (±6.9 at 200 games). The evidence for the
  prior is the replay check above (10.1% of opponent Pokémon missed -> 1.0%).
- Games averaged 26.8 turns.
- Three things stopped the run, each fixed during it:
  1. The lid closed and the Mac slept (15:19). The server dropped the connection at 15:25 (a
     keepalive ping timeout), the 2 games in progress were lost, and `ladder_parallel.py` hung.
     It now exits with 3 when the connection drops, and `tools/ladder_loop.sh` starts it again
     once the lid is open.
  2. Showdown's ladder seasons began (October 15), and with them a `|badge|` message at the start
     of each game, on which poke-env raises (18:50). The loop stopped queuing games, as it does
     after any unexpected traceback. `search/policy.py` now skips message types poke-env doesn't
     know (`poke_env_gotchas.md`, item 16).
  3. Three games ended within seconds of each other, and the server's throttle dropped the bot's
     next `/search` (19:21). With no game in progress, the run then waited 3 hours for a game that
     never came. Messages now go out at least 0.65 s apart, and `ladder_parallel.py` exits with 3
     after 10 minutes with no game in progress (`poke_env_gotchas.md`, item 17).
- After the fixes, the last 103 games ran in one go (1 h 34 min), with no warnings.
- Logs (each one's count restarts at 1): `logs/ladder_search_20261015_live.log` (games 1-14 and
  the 2 lost), `logs/ladder_search_20261015_live_part1.log` (15-66 and the 2 forfeits),
  `logs/ladder_search_20261015_live_r2_part1.log` (67-93),
  `logs/ladder_search_20261015_live_r3_part1.log` (94-196). The loop's logs end in `_loop.log`.

## Run 3: October 16 (100 games, recorded)

Run 2's bot and set prior, under `tools/ladder_loop.sh`, with
`players/ladder_parallel.py --record logs/ladder_games_20261016`. That keeps each game's log and
every search decision for the loss review (`loss_review.md`).

**Result: the bot's record 53-47. Elo 2227 -> 2258, 118th in the top 500; GXE 88.5 -> 88.6;
Glicko unchanged at 1885 ± 25.**

| games | wins |
| --- | --- |
| 1-25 | 13 |
| 26-50 | 13 |
| 51-75 | 12 |
| 76-100 | 15 |

- All 100 games were complete, with no forfeits and none lost to the Mac sleeping. They averaged
  27.5 turns.
- The run took 1 h 57 min in one go. The only warning was poke-env's, about an opponent's name with
  an apostrophe.
- The account's page showed 305-276 afterwards, up from 252-230. That's 53-46, one loss short
  of the bot's count.
- Logs: `logs/ladder_search_20261016_rec_part1.log` and its `_loop.log`. The records are in
  `logs/ladder_games_20261016/`.

## How to run

    nohup caffeinate -i bash tools/ladder_loop.sh 200 logs/ladder_run > logs/ladder_run_loop.log 2>&1 &
    caffeinate -i -s python3 -m players.ladder_parallel --games 200 --stop-file logs/ladder.stop
    python3 -m players.ladder_bot --games 100        # one at a time, with the fork's set prior

- On the laptop, use `tools/ladder_loop.sh` for unattended runs. The Mac sleeps when its lid
  closes, and caffeinate can't stop that. The server then drops the connection, the games in
  progress are lost, and `ladder_parallel.py` exits with 3. The loop runs `ladder_parallel.py`
  whenever the lid has been open for 2 minutes, for the games the account still has to play, and
  again after each exit with 3. Its target counts the account's games, so lost games count.
- `ladder_parallel.py` wraps `ladder_bot.py` (the account, the bot) and sets poke-env's
  `max_concurrent_battles` (`--concurrent`, default 3). It uses
  `data/gen9_randbats_set_counts_live.json` when that file exists; `--set-prior` picks another.
- It sends messages to the server at least 0.65 s apart, which keeps them under the server's
  throttle. With no game in progress for 10 minutes (the server lost or refused the search), it
  exits with 3, and the loop starts it again.
- `touch logs/ladder.stop` (the `--stop-file`) stops new games. The games in progress finish and the
  run ends with its results. Stopping the process forfeits the games in progress.
- `--record DIR` (passed through by the loop) keeps each game's log, as poke-env's replay, and every
  search decision in `DIR/decisions.jsonl`. Each record has the win chance, the options' shares and
  win chances, and what the search expected the opponent to pick. Nothing is posted in the battle.
  - `python3 -m tools.review_ladder_games DIR` shows the turns where each loss was lost, plus the
    search's calibration and how well it predicted the opponent (`loss_review.md`).
  - `--game <battle tag> --top 10` shows one game in detail.
- On battery below 30% (`--min-battery`), it queues no new games until the Mac is on AC power. Run
  it under `caffeinate -i -s`, or the Mac sleeps after a minute idle.
- The bot's per-battle state is keyed by battle: the trackers are in `WeakKeyDictionary`s, and the
  network's cache is keyed by `id(battle)`. The search is single-threaded, so overlapping games
  wait for each other's decisions (~0.6 s each).
- The rating: `https://pokemonshowdown.com/users/<account>.json`; the top 500:
  `https://pokemonshowdown.com/ladder/gen9randombattle.json`.

## What next
- What the losses show, and the fixes they suggest: `loss_review.md`.
- The rating has settled, so more games would measure the same thing again. A ladder run can't
  measure a change of a few points either (±6.9 at 200 games, against opponents that change with
  the rating), so local evals stay the measure of strength.
- Read right the abilities behind the current prior's misses (1% of opponent Pokémon): Trace and
  As One, Terapagos's forms, Ditto's Transform (`observations` in `search/sampling.py`).
- Rebuild the set prior when Showdown's generator changes, and check it with
  `tools/check_prior_replays.py`.
- The cheap tuning items in `step8.md` section 7.
