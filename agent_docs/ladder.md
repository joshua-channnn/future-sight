# The search bot on the public ladder

October 14-15, 2026: 100 Gen 9 Random Battles on the public Showdown server with the deployed
search bot (`players/ladder_bot.py`: step 7's lin1_86 leaf evaluation, 2 x 250 ms per decision,
the v21 prior and set model, item inference).

**Result: 69 wins, 31 losses. Elo 1162 -> 1869, GXE 38 -> 89.7, Glicko 1408 ± 35 -> 1910 ± 42.**
Before this run the account had played 182 games with the PPO models (71-111). The ladder's top
500 started at Elo 2106 at the time (the top player had 2523), so the bot isn't in it.

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

## How to run

    python3 -m players.ladder_parallel --concurrent 3 --games 100
    python3 -m players.ladder_bot --games 100                       # one at a time

- `ladder_parallel.py` wraps `ladder_bot.py` (the account, the bot) and sets poke-env's
  `max_concurrent_battles`. The bot's per-battle state is keyed by battle: the trackers are in
  `WeakKeyDictionary`s, and the network's cache is keyed by `id(battle)`. The search is
  single-threaded, so overlapping games wait for each other's decisions (~0.6 s each).
- Stopping the run mid-game forfeits the games in progress. To change settings, stop it right after
  a game ends, while the bot is still searching for the next one.
- The rating: `https://pokemonshowdown.com/users/<account>.json`; the top 500:
  `https://pokemonshowdown.com/ladder/gen9randombattle.json`.

## What next
- More games, to see where the rating settles.
- A set prior from the current generator. The bot's prior is still the Feb 2026 fork's, and the
  public server's sets may have changed
  (`tools/build_set_prior.py --showdown <a current checkout>`).
- The cheap tuning items in `step8.md` section 7.
