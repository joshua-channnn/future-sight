# Why the search bot loses on the ladder

October 16, 2026. This review answers two questions:
- Where does the ladder bot (`ladder.md`) lose? Sections 3-5 cover 59 lost games: 12 public replays
  from run 2 and the 47 losses of run 3, which recorded every search decision.
- How often do the mechanics poke-engine gets wrong come up? Section 1 counts them over the 1.1M
  decisions of the search self-play data (`step7.md`).

**Result: the engine's wrong mechanics aren't what loses games. The search's picture of the
opponent is.**
- **The known engine gaps:** each comes up in under 0.5% of decisions, and none decided any of the
  59 losses.
- **Four new problems** turned up in the recorded losses: Rage Fist's power, Unburden, Encore into
  a 0-PP move, and items not inferred from damage. One of them decided a loss.
- **The opponent model:** the search predicts the opponent as its own best response, and people
  set up and use status moves far more than that.
  - The search gives the opponent's boosting moves a median share of 17%, against 42% for
    attacks.
  - 18 of the 47 recorded losses turned on a setup the bot let happen, 10 of them with the
    search's win chance still 50% or more after it.
  - Five were leads of 80% or more, lost to one Pokémon in a long war of boosts and heals.
- **Close calls:** in 27 of the 47 losses a decisive choice was one where the search rated the
  options within a point or two. Whatever broke the tie decided it: the visit count, near_best's
  random pick or the v21 network prior.
- **Boosts:** lin1_86 does value them. +2 is worth +0.29 logits, about 7 points of win chance; an
  earlier reading that it doesn't was wrong. What the leaves can't see is a boosted Pokémon's
  threat to the rest of the team.
- **Luck** decided 14 of the 47.
- **The search's win chance is calibrated within the noise:** 5.8 points optimistic, 95% CI -1.7
  to +13.4.
- **First fixes:** section 6. Record the engine states so the fixes can be tested on positions from
  people's games, prune options that can't help, and settle the close calls by value rather than
  by visits or the network prior. The opponent model is the largest fix and a project of its own.

## 1. The engine's wrong mechanics are rare

`tools/count_engine_gaps.py` reads the search self-play data: `datasets/search_selfplay` and
`_r1`, 169 shards, 16,352 games and 1,114,568 decisions, each stored with the true engine state.
The fidelity check (`step6.md`) found mechanics that poke-engine 0.0.48 gets wrong. For each one,
the tool counts:
- the decisions where it's in play for either active, for example a 2-5 hit move in its moveset;
- the games with at least one such decision;
- the decisions where the decider played the move or switch involved.

The search's simulation is wrong only on the lines that use the mechanic.

| mechanic the engine gets wrong | decisions in play | games | decisions played |
| --- | --- | --- | --- |
| Population Bomb / Triple Axel / Triple Kick: hits approximated | 38,597 (3.46%) | 3,569 (21.8%) | 5,447 (0.49%) |
| Loaded Dice: always 4 hits (4-5 in Showdown) | 26,464 (2.37%) | 2,680 (16.4%) | 4,289 (0.38%) |
| 2-5 hit moves always hit 3 times | 5,191 (0.47%) | 654 (4.0%) | 550 (0.05%) |
| Heal Block (Psychic Noise) doesn't stop Leftovers or drain, against a foe that heals that way | 18,413 (1.65%) | 2,023 (12.4%) | 2,366 (0.21%) |
| Sheer Force + Life Orb: the engine charges the recoil | 13,498 (1.21%) | 1,431 (8.8%) | 3,536 (0.32%) |
| Take Heart (a no-op), Beat Up (one hit), Armor Cannon (no self-drop) | 11,977 (1.07%) | 1,038 (6.3%) | 866 (0.08%) |
| Toxic Chain: modeled as a 30% secondary, so only edge cases | 15,766 (1.41%) | 1,289 (7.9%) | n/a |
| Burn or poison before Leftovers, for a holder at 1/8 HP or less | 3,227 (0.29%) | 979 (6.0%) | n/a |
| Intrepid Sword again on a later switch-in (once per battle in gen 9) | 2,401 (0.22%) | 185 (1.1%) | 242 |
| Rest at full HP with a status: it cures in the engine, fails in Showdown | 907 (0.08%) | 141 (0.9%) | 36 |
| Explosion into a Ghost doesn't faint the user | 434 (0.04%) | 119 (0.7%) | 3 |
| Both sides switch at once: side one always goes first | 12,270 (1.10%) | 5,035 (30.8%) | all |
| ... and either brings in Intimidate, Download or Trace | 708 (0.06%) | 336 (2.1%) | all |
| ... and both bring in a weather setter / a terrain setter | 2 / 0 | 1 / 0 | all |
| Illusion (observation): a Zoroark on either team | 25,711 (2.31%) | 404 (2.5%) | n/a |
| Transform or Imposter (observation) on either active | 194 (0.02%) | 169 (1.0%) | n/a |

- **The ones that come up:**
  - multi-hit determinism (a fifth of games have Population Bomb, Triple Axel or Triple Kick, and
    a sixth have Loaded Dice);
  - Heal Block against Leftovers or drain (12% of games);
  - Sheer Force with Life Orb (9%);
  - the three unmodeled moves (6%).

  Even these are played in under 0.5% of decisions each.
- **The rare ones:**
  - Rest at full HP: 36 plays in 1.1M decisions. Another 365 Rests at full HP were by a sleeping
    Pokémon, and 365 by one with no status; those fail in Showdown and change nothing.
  - Explosion into a Ghost, Intrepid Sword and the order of abilities in a double switch.
- **None of them decided one of the 59 losses** (sections 4-5). Loaded Dice, Beat Up and Triple
  Axel came up without deciding anything. The losses turned up four problems this list doesn't
  have (section 4), one of which decided a loss.
- **Testing:** each fix alone is far too rare to show in an A/B (±2.2 points at 2,000 games).
  - Fix them as a batch in `search/engine_patch/`.
  - Check the batch with the fidelity check (`tools/check_search_state.py`) and with how often it
    changes the search's pick (`tools/compare_picks.py`).

## 2. What lin1_86 values at the leaves (measured)

`v` is the leaf evaluation's logit of the bot winning. It sits at the MCTS leaves as
sigmoid(1.4 x (v - v(root))). To measure what it values, `tools/leaf_values.py` edits real
self-play states one field at a time and reads `v` back from the engine (`pe.evaluator_values`).
It used 3,000 states from 20 shards, with side one as the decider.

| change (own side; the foe's is the same with the sign flipped) | mean change of v | 10%-90% |
| --- | --- | --- |
| active's Attack +1 / +2 / +6 | +0.15 / +0.29 / +0.48 | +0.39 to +0.74 at +6 |
| active's Special Attack +2 / +6 | +0.29 / +0.48 | |
| active's Speed +2 / +6 | +0.35 / +0.52 | |
| active's Defense +2 / +6 | +0.25 / +0.42 | |
| the foe's active from full HP to 1 HP | +1.13 | +0.87 to +1.46 |
| the foe's active from 1 HP to fainted (the KO) | +0.20 | -0.31 to +0.85 |
| own active from full HP to 50% | -0.61 | -0.78 to -0.47 |
| Substitute | +0.89 | |
| Leech Seed on the active | -0.77 | |
| Yawn on the active | -0.31 | |
| Reflect / Light Screen / Aurora Veil | +0.27 / -0.02 / +0.52 | |
| Tailwind | +0.52 | |
| a layer of Spikes / Toxic Spikes; Stealth Rock; Sticky Web, on own side | -0.29 / -0.36; -0.32; -0.44 | |
| the active partially trapped (Whirlpool, Infestation) | +0.29 | |
| the active Taunted / Salt Cured | +0.11 / +0.09 | |

At 50%, +0.29 is about 7 points of win chance.

- **Boosts do count.** +2 in an attacking stat is worth about 7 points of win chance at 50%.
  - The leaves value them through three terms:
    - the boost features;
    - the hand-written evaluation, which is one of lin1_86's features;
    - the damage and KO features, which use the boosted stats.
  - What the leaves can't see is a boosted Pokémon's threat to the rest of the team. The cross
    features look only at the active pair, and each damage term is capped at a KO of the current
    defender. That is why +6 is worth only about 1.7 times +2.
  - **A correction:** the first reading of the weights (and the public review below) put +6
    Attack at 0.06-0.20 and a Substitute at 0.39-0.45. That reading took `w1` per side. The
    engine uses `g(x) - g(swap(x))`, which doubles every side weight, and that reading also left
    out the damage terms.
- **Finishing a 1-HP foe is worth little at the leaves:** +0.20 on average, less than healing 20%
  of the active's HP. Leaving a foe at 1 HP is worth nearly as much as the KO. That's right only
  if it can't act again.
- **Some values are implausible:**
  - being trapped is good (+0.29), and so are Taunt (+0.11) and Salt Cure (+0.09);
  - Light Screen is worth nothing, while Reflect is worth +0.27;
  - a Substitute (+0.89) is worth nearly twice +6 Attack.

  These are rare features fit on self-play, probably carrying correlations. For example, a
  Pokémon that gets trapped may be one the trapper can't otherwise beat.

## 3. Run 3: 100 recorded games (53-47)

`players/ladder_parallel.py --record` kept all 3,096 search decisions of run 3 (`ladder.md`).
`tools/review_ladder_games.py` gives the turning points and the overall figures. One-off scripts
gave the rest; they are named in the full review.

### 3.1 Counts from the logs: 47 losses against 53 wins

| | losses | wins |
| --- | --- | --- |
| the bot's Pokémon KO'd by a foe at +2 or more (all stats summed) | 105 of 282 (37%) | 28 of 135 (21%) |
| the foe's Pokémon KO'd by the bot's at +2 or more | 34 of 145 (23%) | 82 of 277 (30%) |
| setup moves by the opponent (per game) | 116 (2.5) | 118 (2.2) |
| ... where the setter took no direct damage that turn | 66 (1.4) | 55 (1.0) |
| the bot's voluntary switches per turn | 0.171 | 0.148 |
| switching back to the Pokémon it just left, within 2 turns | 54 | 14 |
| ... with the search's win chance at 40% or more | 11 | 10 |
| the bot's moves that failed / went into an immunity of a foe already in | 26 / 9 | 7 / 3 |
| crits against the bot / by the bot | 48 / 27 | 26 / 33 |
| the bot fully paralyzed | 18 | 5 |
| decisions where the bot didn't play the search's most-visited option | 16.0% | 10.5% |
| games where the bot terastallized | 45 | 49 |

- **Setup:** opponents set up about as often in wins as in losses. What differs is what came next.
  - In wins, a third of the setters were KO'd within 2 turns of their last setup move: 33 of 70.
  - In losses, 11 of 56 were. 22 of the 56 setters in losses went on to take 2 or more KOs,
    against 4 of 70 in wins.
- **Switch loops:** mostly a symptom.
  - Of the 54 switch-backs in losses, 32 came when the win chance was already under 15%. The
    search's values go flat there, and the switches cost nothing.
  - With the game still live (40% or more), losses and wins had about as many (11 and 10).
  - 51 of the 54 were the search's top option, not near_best's random pick. The search rated the
    switch about 1 point above staying in.
- **Close calls:** when the bot played an option other than the most-visited one, the search had
  rated it within 0.3 points on average.

### 3.2 The search's win chance

| the search's win chance | decisions | share of those decisions' games won |
| --- | --- | --- |
| 0-10% | 383 | 0.8% |
| 10-20% | 176 | 9.1% |
| 20-30% | 193 | 16.6% |
| 30-40% | 250 | 22.4% |
| 40-50% | 371 | 37.7% |
| 50-60% | 433 | 50.1% |
| 60-70% | 380 | 61.3% |
| 70-80% | 310 | 67.4% |
| 80-90% | 257 | 77.8% |
| 90-100% | 343 | 93.3% |

- **Mild optimism:** on average the search's win chance is 5.8 points above the share of games
  won. The 95% CI, bootstrapped over games, is -1.7 to +13.4, so this could be noise.
- **By whether the active Pokémon are boosted:**
  - neither side boosted: +5.1 points;
  - the foe's active at +2 or more (attack, special attack and speed summed): +8.8 (CI -4.6 to
    +21.5);
  - the bot's boosted: +9.8.

  So it isn't detectably more optimistic against a boosted foe.
- **Leads:** 19 of the 47 losses reached 70% at some point. Of the 70 games that reached 70%,
  the bot won 51 (73%), about what a calibrated estimate gives. 10 of the 53 wins came back from
  30% or less.

### 3.3 Predicting the opponent

The search's prediction of the opponent comes from the same MCTS: the side-two root visits. That
is its best response for the opponent, not a model of how people play. Over the 2,418 decisions
where the log shows what the opponent chose:
- the search gave that option a mean share of 36%; it was the top option for 49% of the moves;
- it was the top option for 25% of the switches to a Pokémon already seen;
- switches to one not seen yet got 39%, the summed share of all such switches.

By the kind of move the opponent played:

| the opponent's move | n | mean share | median | under 10% |
| --- | --- | --- | --- | --- |
| an attack | 1,498 | 46% | 42% | 18% |
| a heal (Recover, Roost, Wish, Rest, ...) | 128 | 39% | 39% | 33% |
| Protect and the like | 27 | 35% | 28% | 37% |
| a boosting move | 225 | 32% | 17% | 41% |
| other status moves (hazards, status, Encore, ...) | 166 | 25% | 12% | 48% |

People set up, and use other status moves, far more often than the search's best response
expects.

## 4. Run 3's 47 losses, one by one

Every loss was reviewed turn by turn against its search notes: the options' shares and win chances,
and what the search expected the opponent to do. The full write-up is in
`logs/loss_review_recorded_20261016.md`.

**Who it lost to:** in the losses, the opponents averaged 2225 and the bot 2227. 30 of the 47 losses
were to lower-rated players.

Causes (more than one where they combined):
- (a) bad luck;
- (b) a clear bot misplay;
- (c) an engine or observation problem;
- (d) outplayed or a bad matchup.

| cause | first cause | first or a contributing cause |
| --- | --- | --- |
| (d) outplayed or a bad matchup | 27 | 42 |
| (a) bad luck | 13 | 23 |
| (b) a clear bot misplay | 6 | 25 |
| (c) an engine or observation problem | 1 | 4 |

What decided the losses (more than one per game):

| pattern | losses |
| --- | --- |
| a setup the bot let happen, then a sweep | 18 |
| the opponent's choice that decided it got a low share from the search | 19 |
| a decisive choice between options the search rated within 1-3 points | 27 |
| a chance event against the bot decided an even game | 14 |
| an opponent Tera flipped an exchange | 9 |
| a choice lifted by the network prior (outside near_best's band) cost something | 10 |
| a common opponent set missing from both samples | 4 |
| a Tera of the bot's backfired | 5 |
| a switch loop decided it | 3 |

**Setup (18 losses).** People set up when the bot's active can't punish it, and in long boost
and heal wars the bot keeps attacking.
- In 13 of the 18, the search gave a setup move 25% or less of the opponent's share at some point.
  In 10, its win chance was still 50% or more after the setup.
- Five were leads of 80% or more, each lost to one Pokémon:
  - a +1 Altaria at 92%;
  - a Curse/Recover Pincurchin at 82%, with Taunt and Spore on the bench;
  - Calm Mind/Wish/Protect Sylveon, against which the bot was 4 to 1 at 93%;
  - Calm Mind/Moonlight Cresselia, 4 to 1 at 97%: 20 Iron Heads in a row;
  - Calm Mind Necrozma, with the search at 72-78% through +5/+5.
- In the others the setup came on a forced or predicted switch, or the team had no answer.

**The opponent model.** The search's own prediction of the opponent (section 3.3) runs through
most of these:
- the setup and healing turns of the stall wars, predicted at 0-3%;
- 9 opponent Teras that flipped an exchange, most at 0-12%;
- the switch-ins it didn't foresee.

**Close calls.** The bot often chose between options the search rated within a point or two, and
the visit share decided.
- Sometimes the most-visited option had a clearly lower win chance than one with fewer visits:
  - Brave Bird at 36% of visits and a 29% win chance, over Tera Close Combat at 29% and 53%;
  - a switch at 36% and 28%, over a sure-KO Liquidation at 14% and 35%;
  - a switch at 37% and 29%, over a +2 Psychic at 22% and 36%.
- Several times it played a no-op the engine knew was one:
  - Thunder Wave into a paralyzed foe;
  - Encore while the foe was already Encored;
  - Shadow Ball into a Normal type already in;
  - Toxic into a poisoned foe.

**The network prior.** The v21 prior is added to the shares at 0.15 x its probabilities, so it
decides between options the search visited very differently but rated alike.
- By the search's own estimate, the options it lifted had a win chance only 0.1 points below the
  top option on average.
- Prior-lifted choices were 3.7% of the decisions in losses and 2.8% in wins. Step 6 measured the
  prior at +2.1 points, inside the noise.
- In hindsight, what it lifted cost something in 10 losses:
  - two Flamethrowers into a Heatran whose Flash Fire was known (no leaf feature sees Flash Fire);
  - Close Combat into a predicted Ghost switch-in;
  - a Tera Iron Defense at 3% of the visits;
  - a Power Whip into the Knock Off that the search's own pick survived.

**Two samples.** Each decision searches two sampled opponent teams. A set the prior gives 50% is
missing from both a quarter of the time, and four losses turned on one that was:
- Drednaw's Crunch set (50%): Surf was chosen over a sure-KO Earthquake;
- Terapagos's Earth Power (49%);
- Choice Band Cinderace (44%);
- Great Tusk's Ice Spinner (26%): 88% to 39%.

**Engine and observation problems.** One loss was decided by one and three more were affected.
None of the gaps in section 1 decided a loss; Loaded Dice, Beat Up and Triple Axel came up
without deciding anything. The new ones:
- **Rage Fist is a flat 50 BP** in poke-engine. `choices.rs` has no power callback for it and
  the state keeps no hit counter.
  - The bot fought a Bulk Up war against an Annihilape whose Rage Fist was at 150 BP and lost
    Okidogi to it (68% to 36%).
- **Unburden never activates.** The engine checks an UNBURDEN volatile that nothing sets, and
  `search/state.py` doesn't read it from the log. After Drifblim ate its Sitrus Berry, the search
  thought it was slower than it was.
- **Encore into a move with 0 PP succeeds in the engine** (`generate_instructions.rs`, the Encore
  case only checks for a last move). In Showdown it fails, and the bot's Hypno lost its turn and
  was OHKO'd (54% to 15%).
- **Items aren't inferred from damage.** An 81% Brave Bird didn't make Dodrio's Choice Band more
  likely than the guessed Heavy-Duty Boots (`observations()` in `search/sampling.py` rules items
  out only from actions).
- **Dauntless Shield and Intrepid Sword fire on every simulated switch-in** (`abilities.rs`).
  That is pessimistic for the bot, and it decided nothing.

**What the earlier review found, on this bigger sample:**
- **Setup is the main leak:** confirmed, with the threat underrated rather than the boosts.
- **Switch loops:** mostly a symptom of lost positions (section 3.1); decisive in 3 losses.
- **Tera misreads:** confirmed for the opponent's Teras, less so for the bot's own.
- **The Ditto and Rest bugs:** neither came up.

| battle | opponent (rating; bot) | turns | peak | causes | what decided it |
| --- | --- | --- | --- | --- | --- |
| 2699646784 | Eluding_Fate (2115; 2227) | 28 | 96% | d, b | +1 Altaria not seen as a threat at 92%; a resist-switch loop gave it a second Dragon Dance and a Roost; it swept 6 |
| 2699647045 | Aunche (2212; 2227) | 17 | 55% | d, b | two free Dragon Dances (shares 5%, 11%) for a Loaded Dice Lapras, which swept 5 |
| 2699647621 | Helsa06 (2154; 2206) | 31 | 86% | a, c | three flinches and four full paralyses; Unburden untracked, so Drifblim looked slower than it was |
| 2699648823 | rozinante (2268; 2159) | 31 | 70% | b, a, d | a prior-lifted Tera Triple Axel missed; Plusle got 4 free Nasty Plots, 2 while Encore locked the bot into a failing Slack Off |
| 2699653166 | slynch64 (2240; 2256) | 33 | 67% | a | two 80% Gunk Shot misses in a row |
| 2699653815 | squadlecuisto (2106; 2271) | 38 | 87% | b, c, a | Curse/Recover Pincurchin reached +4 (Taunt unused, Spore overridden by the prior); Encore into a 0-PP move |
| 2699655254 | RainbowTrainerZ (2242; 2269) | 16 | 47% | d | a Specs-locked move into an unseen Weavile, then Swords Dance and Tera Ice; nothing outsped it |
| 2699655621 | SmurfAccountN9999 (2430; 2240) | 21 | 60% | b, d | switched out instead of a +2 Psychic KO; a Tera Fire Sunny Day (0%) swept |
| 2699655892 | bonomia (2255; 2256) | 26 | 89% | d, b, a | Swords Dance at 0% and an even OHKO roll; then Minun's predicted Nasty Plots given free |
| 2699658052 | TheLastOne (2202; 2259) | 28 | 93% | d, b | 4v1 at 93% lost to Calm Mind/Wish/Protect Sylveon; four attacks into Protect |
| 2699658436 | Antumbral (2466; 2239) | 28 | 80% | d | two unseen Pokémon and a Tera Fighting Focus Blast (2%) |
| 2699659093 | Eluding_Fate (2136; 2255) | 23 | 80% | d, b | an unseen Choice Band Wugtrio swept the last three; the Tera had gone on a resisted Focus Blast |
| 2699660717 | schizoryan (2206; 2230) | 28 | 71% | d, b, a | Rain Dance (23%) Swift Swim Poliwrath; a prior-lifted Power Whip lost Brambleghast; a crit |
| 2699661257 | BlueMustang12 (2121; 2205) | 19 | 54% | d, a | Glare and a full paralysis; a predicted Tera Dragon Swords Dance Sandslash swept 5 |
| 2699661949 | JustHaxRelax (2209; 2197) | 32 | 75% | b, d | Bisharp stayed in for a 30% Iron Head into the Flamethrower it expected |
| 2699662699 | BigDaddyKyle2 (2137; 2178) | 26 | 59% | a, d | a crit Close Combat on Zacian-Crowned at 54% |
| 2699666368 | Vegeta's Hairlinee (2157; 2171) | 29 | 68% | a, d | +2 Volcarona's Fire Blast missed an OHKO at 68% |
| 2699666984 | squadlecuisto (2146; 2207) | 38 | 94% | a, b, d | a 52% KO roll the search rated 88%, then Tera Electric (2%) Raikou; its Great Tusk answer sacked earlier |
| 2699668560 | Skeptikal_MonkE (2024; 2200) | 42 | 97% | d, b | 4v1 at 97% lost to Calm Mind/Moonlight Cresselia: 20 Iron Heads, its boosts predicted at 0-3% |
| 2699670867 | SmurfAccountN9999 (2418; 2232) | 55 | 49% | a, d, (b) | a turn-1 Gunk Shot miss; Freeze-Dry into eight predicted Roosts instead of Taunt |
| 2699671315 | Antumbral (2350; 2248) | 31 | 61% | a, d | a crit and two full paralyses on +2 Annihilape |
| 2699672646 | fireicedragon25 (2185; 2219) | 59 | 75% | a, b | prior-lifted Flamethrowers into Flash Fire; at 75%, an Earth Power neither sample had, with a crit |
| 2699673073 | marper8 (2176; 2234) | 41 | 46% | d, a | walled by Will-O-Wisp Arceus-Steel and Water Absorb Poliwrath; crits |
| 2699674003 | los chipirones (2334; 2252) | 17 | 49% | d, b | a turn-1 Tera Fire read by a Diancie switch-in; Rain Dance over a KO |
| 2699674131 | Galak0 (2214; 2228) | 19 | 70% | d, c | Choice Band Dodrio taken for Boots (no inference from damage); Tera Flying (3%) |
| 2699674685 | JustHaxRelax (2224; 2214) | 16 | 58% | a, d | a crit on Chansey, the right answer to a predicted Swords Dance |
| 2699676626 | MichaelderBeste2 (2429; 2212) | 37 | 70% | a, d | Iron Defense at 0% and 2% after a Will-O-Wisp miss; a crit Body Press |
| 2699676688 | SmurfAccountN9999 (2426; 2226) | 44 | 80% | d | Tera Ground (3%) Calm Mind Necrozma; the search at 72-78% through +5/+5 |
| 2699677143 | No 5580 (2177; 2206) | 31 | 60% | d, b | Shadow Tag trapped Salazzle; a 29% Smeargle set hazards; Tera Water Barraskewda (12%) |
| 2699678658 | belabourer (2008; 2168) | 29 | 68% | a, d | a Scald burn on +2 Gyarados; Calm Mind Suicune swept 5 |
| 2699679037 | squadlecuisto (2123; 2207) | 25 | 58% | d, b | Shadow Ball into a revealed Normal type; Brave Bird (29%) over Tera Close Combat (53%) |
| 2699680605 | No 5580 (2198; 2197) | 57 | 68% | c, b | Rage Fist at 50 BP in the engine: a Bulk Up war lost to a 150 BP Rage Fist |
| 2699682181 | calmalt (2173; 2214) | 23 | 67% | d, b | an unsampled Choice Band Cinderace; a switch loop over a sure Liquidation KO |
| 2699683714 | los chipirones (2253; 2189) | 19 | 84% | d, a | Hypnosis and a 3-turn sleep; Tidy Up (0%) Maushold OHKO'd the last four |
| 2699683810 | yirak (2161; 2210) | 35 | 65% | d, b | a Glare/Prankster stall team; Sandaconda's Stone Edge at 6% |
| 2699685206 | classicahri (1998; 2170) | 26 | 50% | b, d | Choice Band Flygon's resisted Earthquake over Outrage; +2 Leafeon swept 4 |
| 2699686353 | Alpha Tw (2094; 2209) | 30 | 90% | d | at 88%, an unrevealed Ice Spinner (under 1%) OHKO'd Toedscruel |
| 2699688302 | moto gible (2180; 2222) | 54 | 54% | d | neither sample had Drednaw's 50% Crunch set: Surf over a sure-KO Earthquake |
| 2699689354 | JustHaxRelax (2246; 2200) | 21 | 79% | d, b | a dominated Thunderbolt into a switch-in; Tera Ground Beartic's Earthquake (0%) |
| 2699690425 | Emkatt (2343; 2264) | 32 | 44% | d, a | two Toxic misses while Sticky Web and Toxic Spikes went up; Trick Room |
| 2699691975 | SmurfAccountN9999 (2387; 2302) | 52 | 70% | d, b | a prior-lifted Close Combat into a Ghost; switching while Zamazenta-Crowned Iron Defended to +6 |
| 2699693353 | gyuuoy (2245; 2283) | 34 | 68% | a, d | two 85% Fire Blast misses |
| 2699693917 | Antumbral (2414; 2269) | 24 | 57% | b, a, d | Swords Dance over a near-sure KO (High Jump Kick at 1%), then a crit |
| 2699694939 | totodilehola (2099; 2236) | 23 | 50% | d, a | a well-built lower-rated team; no clear bot error |
| 2699695952 | Antumbral (2445; 2289) | 25 | 73% | a, d | a 70% Focus Blast missed at 73% |
| 2699696193 | elodin (2200; 2248) | 41 | 60% | d | Tera Dragon Salt Cure/Recover Garganacl walled the team |
| 2699696919 | Breedlove123 (2229; 2265) | 30 | 64% | d, b | six Cosmic Powers, predicted at 0-7% after the first two; +6/+6 Arceus took all six |

## 5. Run 2's public replays: 12 losses

Before `--record` existed, 25 of run 2's games had public replays (`logs/ladder_replays_public/`).
The bot went 13-12 in them, 3-6 against Antumbral (rated 2283-2405). The 12 losses were reviewed
from the logs alone (`logs/loss_review_public_20261016.md`).

**Primary causes:**
- clear bot misplay: 6;
- outplayed or a bad matchup: 3;
- bad luck: 2;
- mixed: 1.

**The misplays:**
- A RestTalk Snorlax with no move that hits Ghosts stayed in for 30 turns while a Misdreavus went
  to +6/+6. A Prankster Encore Illumise sat on the bench.
- A paralyzed Mew broke a Belly Drum Eiscue's Ice Face instead of switching to the Delphox that
  outspeeds and OHKOs it. Eiscue took all six.
- Sinistcha, which can't damage Girafarig, Calm Minded while Girafarig Nasty Plotted.
- A 1-HP Ribombee and a 6% Grimmsnarl were left alive. One later paralyzed the bot's Pokémon, and
  the other set Light Screen.
- Sunny Day was used over a Fire Blast KO.

**Other patterns:**
- switch loops in 4 losses (11 of the 12 switch-backs in all 25 games were in losses);
- 3 of the bot's Teras backfired, and opponent Teras decided an exchange in 4;
- one bug: the bot's own Ditto, after Transform, is modeled with Ditto's stats and Normal type.
  The request gives Ditto's own stats (Showdown's `baseStoredStats`), and `search/state.py` takes
  the types from the species. It got Choice-locked into Recover and Protect.

**Antumbral:** no repeatable exploit. Antumbral set up in front of whatever couldn't punish it,
switched into absorbers and resists, read the bot's obvious move with Tera, and ground the bot
down with bulky pivots that drew its switch loops.

This review's lin1_86 figures were too low (section 2). Its conclusion that "boosts are cheap at
the leaves" doesn't hold; what does hold is that the leaves don't see the threat to the rest of
the team.

## 6. What to fix, ranked

Ranked by how many of the losses each fix might have changed, and by what it costs.

**On testing:** the local A/Bs play against the bot itself (2,000 games, ±2.2) and Foul Play (1,000
games, ±3.1). Neither plays like people, so a fix aimed at human play can be level there and still
help on the ladder, and the ladder can't measure a few points either (±6.9 at 200 games). So the
first step is a position set from people's games.

0. **Record the engine states (cheap; it makes the rest testable).** `--record` keeps the notes
   but not the states the search saw.
   - Saving each sample's `state.to_string()`, or the request, would turn these losses' turning
     points into positions.
   - Each change can then be re-searched there to see whether it picks differently: does it punish
     the setup turns, does it take the KO?
1. **A model of how people play, for the opponent (the largest fix; a project).**
   - It bears on most of the 18 setup losses, the 9 opponent Teras that flipped an exchange and
     the 19 decisive mispredictions.
   - Fit a model of people's choices (the kind of move, given the position) on public replays;
     `tools/check_prior_replays.py` already reads them the way the bot reads its games.
   - Use it to weight side two's options at the root. poke-engine's MCTS takes no priors, so that
     needs an engine patch: root priors for side two, or a share of the root iterations playing
     the model's pick.
   - Test it on the position set, then A/B it to check it costs nothing.
2. **Prune the options that can't help (cheap).** These are no-ops or self-harm that the engine
   state already shows:
   - a status move into a foe that already has a status;
   - Encore into a foe that's already Encored;
   - a heal at full HP;
   - a move into a revealed immunity or absorbing ability (Flash Fire, Water or Volt Absorb, Sap
     Sipper).

   They were chosen in flat positions. Pruning them also takes them away from near_best and the
   prior. Check with `tools/compare_picks.py`, then an A/B.
3. **Close calls (cheap; an A/B each).**
   - When the most-visited option's win chance is clearly below that of another option with
     substantial visits, choose by value: for example, the best win chance among options with 10%
     of the visits or more.
   - Set `prior_weight` to 0, or let the prior only break ties inside near_best's band. It measured
     +2.1 points in step 6, inside the noise, and its picks cost something in 10 of the 47 losses.
4. **Leaves that see threats (medium).**
   - **New features:** the active's threat to the other side's bench, such as how many of the
     remaining foes it outspeeds and OHKOs or 2HKOs at its current boosts, and whether it out-heals
     what the other side can do to it.
   - **A refit with sign constraints:** this also fixes section 2's odd values (trapped, Taunt,
     Salt Cure, Light Screen) and the small value of a KO. Four recorded losses and three public
     ones skipped an available KO.
   - **Testing:** self-play states rarely contain people's setup lines, so test on the position set
     before the A/Bs.
5. **Cover the opponent's likely sets (cheap to medium).** Four losses turned on a set of 26-50%
   that neither sample had.
   - Draw the second sample to differ from the first in the opponent active's unrevealed moves and
     item, or add a third sample when its set is uncertain.
   - Step 6's 3x samples, at 3x the budget, was inconclusive (48.0% over 300 games).
6. **Engine and observation fixes, as one batch.** Each is cheap and too rare for an A/B. Check the
   batch with `tools/check_search_state.py` and `tools/compare_picks.py`.
   - Rage Fist's power, which needs a hit counter.
   - Unburden: set the volatile when the holder's item goes, both in the engine and from the log.
   - Encore failing into a 0-PP move.
   - The bot's own Ditto after Transform: the target's stats, types and ability.
   - Dauntless Shield and Intrepid Sword once per battle.
   - Items from damage: Choice Band and Life Orb.
   - Section 1's list: Rest at full HP, Sheer Force + Life Orb, Heal Block against Leftovers and
     drain, Take Heart, Armor Cannon, Beat Up, Loaded Dice's 4-5 hits, and 2-5 hit moves.

The rest of the losses are luck (14 of the 47) and opponents who played well or had the better
team.

## Files
- `tools/count_engine_gaps.py`: section 1. Its output is in `logs/count_engine_gaps_20261016.log`.
  - `python3 -m tools.count_engine_gaps datasets/search_selfplay datasets/search_selfplay_r1 --workers 6`
- `tools/leaf_values.py`: section 2, for any leaf evaluation (`--eval`). Run it after a refit, for
  example with sign constraints, to see what the new weights value.
- `players/ladder_parallel.py --record DIR`: keeps the games' logs and every search decision
  (`ladder.md`).
- `tools/review_ladder_games.py DIR`:
  - for each loss, the turns where the win chance fell the most, with the log and the search's
    notes;
  - over all games, the leads lost, calibration and the opponent predictions.
- The data (gitignored):
  - `logs/ladder_games_20261016/`: run 3's 100 games and 3,096 decisions;
  - `logs/ladder_replays_public/`: run 2's public replays;
  - the two full per-game reviews this summarizes: `logs/loss_review_public_20261016.md` (its
    weight figures predate the correction in section 2) and `logs/loss_review_recorded_20261016.md`.
