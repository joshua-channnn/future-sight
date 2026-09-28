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
