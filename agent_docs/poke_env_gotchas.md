# poke-env gotchas

Each of these caused a real, silent bug in this project. All were fixed in v19; keep them fixed.

1. **poke-env scores both players on one object.** `PokeEnv.reset` and `PokeEnv.step` (poke-env
   `environment/env.py`) call `embed_battle` and `calc_reward` for the opponent's battle
   (`battle2`) too, on the same env object. Any per-episode state on the env gets overwritten
   unless the opponent's perspective is skipped: see `_is_discarded_perspective` in
   `envs/rl_player_6v6_v18.py`. Before v19 this turned the shaped reward into a per-turn
   "being ahead" bonus (79% of the return) and made the opponent-history features garbage.

2. **`battle.available_switches` reshuffles.** It follows Showdown's request order, which changes
   after every switch, while `battle.team` keeps first-seen order. Switch action `8 + k` means
   `switch_targets(battle)[k]` (team order, not active, not fainted), the same order as the
   per-slot bench features in `_encode_global`. Convert with `_switch_action_to_request_index`
   and `order_to_action`; never index `available_switches` with an action. The extractor pools
   the bench, so those per-slot features are the only way the policy knows which Pokémon is
   behind each switch action.

3. **Observations arrive when you have nothing to decide.** When only the opponent must act (for
   example their forced switch), poke-env still returns an observation and ignores your action.
   `MaskablePokeEnvWrapper.step` plays through those turns using `rl_env.agent1_to_move`.
   `SingleAgentWrapper` also asks the opponent every step, so `PPOPlayer._sync_choose_move`
   returns a default order when `battle._wait` is set.

4. **`DummyVecEnv` auto-resets on done.** Calling `env.reset()` once per episode forfeits the
   battle that was just started, and `rl_env.battle1` after `done` is already the next battle.
   Reset once, then read results from the terminal step's `info` (see `evaluate` in
   `eval/ppo_test.py`).

5. **SB3 collects rollouts in eval mode and trains in train mode.** Dropout in the policy makes
   the PPO ratio noisy before any learning happens. Keep `embed_dropout` at 0.

6. **Two code paths build observations.** Training goes through the env; live play and self-play
   opponents go through `PPOPlayer._predict_action`, which calls `embed_battle` once per decision.
   Keep any per-battle state consistent between the two, and use the same action mask: until v19
   `PPOPlayer` also blocked tera on turns 0-1, which training never did.

7. **`battle.won` is `None`** for ties and unfinished battles.

8. **`battle.opponent_team` only holds revealed Pokémon.** Random battles have no team preview,
   so the opponent's other Pokémon are missing until they switch in. Counting its healthy members
   said 1 of 6 on turn 1 and undercounted at 92% of decisions, in both the observation and the
   shaped reward. Count the fainted instead (`_opponent_remaining`) and treat unrevealed Pokémon
   as healthy.

Found while building v20 (September 2026); v20 handles them, v18/v19 code is left as it was:

9. **This Showdown fork sends each `|request|` before the turn's battle log** (0.1-0.3 ms earlier;
   current Showdown sends the log first). poke-env acts on a request as soon as it is parsed, so
   a decision (and an env observation) can see the state before the turn's events or halfway
   through them: 4 of 12,700 own-HP checks disagreed with the request before the fix, 0 after.
   `envs.rl_player_v20.hold_requests_until_log(player)` handles each request after the battle's
   next log message (or after 50 ms when none follows, as on a re-request after an invalid
   choice). `RLEnvV20` and `eval/ppo_test.py` install it. Don't use it against the public server:
   there every request would wait 50 ms. Foul Play needed the same fix (`../foul-play`).
10. **`GenData.type_chart` is `[defender][attacker]`.** v18's `get_type_multiplier` and
    `_calc_hazard_damage` index it the other way, so v18/v19's bench matchup features and Stealth
    Rock damage estimates are reversed (a Fire type's SR damage comes out as 1/16 instead of
    1/4). Kept for checkpoint compatibility; `Pokemon.damage_multiplier` and v20's `type_mult`
    are right.
11. **poke-env never decrements `Move.current_pp`** (the log parser calls `moved(..., use=False)`),
    so v18's PP features were always 1. The request has the active's real PP.
12. **`Pokemon.first_turn` stays True after a normal move** (only `cant_move` and switching out
    clear it). v20 derives it from the switch-in turn in the log.
13. **`battle.weather[w]` is not the start turn**: every `-weather ... [upkeep]` message resets it
    to the current turn. v20 tracks the start from the events. `battle.fields` and the side
    conditions do hold start turns (or layer counts).
14. **Own abilities can be overwritten by log announcements** (As One shows up as Unnerve) and
    **own tera types are dropped**; the request has both (`ability`, `teraType`).
    `SimpleHeuristicsPlayer` checks `active.tera_type`, which is None until it terastallizes, so
    it never terastallizes.
15. **Illusion (Zoroark) breaks both views**: the opponent's disguised Pokémon collects the real
    one's moves and HP, and poke-env's own-team tracking for the Zoroark side goes wrong. About 5%
    of games have a Zoroark; `tools/check_v20_env.py` reports them separately.
16. **poke-env raises `NotImplementedError` on a message type it doesn't know**, and that abandons
    the rest of the server's message, which can hold the turn's log. The public server sends types
    this repo's fork doesn't: `|badge|p2|bronze|gen9randombattle|...` at the start of games since
    its ladder seasons (October 15, 2026). `search/policy.py` adds `badge` to
    `AbstractBattle.MESSAGES_TO_IGNORE` and skips any other unknown type, warning once per type. The
    v20/v21 and SB3 players don't import it, so on the public server they still raise.
17. **The public server throttles a regular account's messages** (moves, `/search`, `/leave`, chat):
    one every 600 ms, with up to 5 waiting. It drops any beyond that with `|raw|...Your message was
    not sent because you've been typing too quickly.`, which poke-env only logs as an unhandled
    message. A dropped move leaves the game waiting on the timer; a dropped `/search` leaves
    `Player.ladder` waiting for a game forever. Several games at once can burst past it: each
    finished game's `/leave`, the next `/utm` and `/search`, `/timer on`, and the other games' moves
    (October 15, 2026: three games ended within seconds and the run sat idle for 3 hours).
    `players/ladder_parallel.py` sends at least 0.65 s apart and exits with 3 after 10 minutes with
    no game in progress. Local servers started with `--no-security` don't throttle (`nothrottle`).
