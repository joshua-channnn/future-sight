# Future Sight — RL Agent for Pokemon Showdown

A reinforcement learning bot for Gen 9 Random Battles on [Pokemon Showdown](https://pokemonshowdown.com/), trained with PPO and self-play.

## Performance

`models/ppo_pokemon_6v6_v20.pt` (v20, 20M steps from scratch), counting actual wins (`battle.won`):

| Opponent | Games | Win Rate (95% CI) |
|----------|-------|-------------------|
| RandomPlayer | 1,000 | 99.8% ± 0.3% |
| SimpleHeuristics | 1,000 | 83.9% ± 2.3% |
| v18.3 (previous best, ~210M steps) | 2,000 (both seats, both greedy) | 67.2% ± 2.1% |
| v19 (v18 architecture, 10M steps) | 2,000 (both seats, both greedy) | 75.4% ± 1.9% |
| v20 at 10M steps | 2,000 (both seats, both greedy) | 52.3% ± 2.2% |
| Foul Play (MCTS bot, local, default search), measured with v20 at 10M | 200 | 13.5% ± 4.7% |

v20 at 10M steps (the best model until October 2, 2026, still in git history) won 83.7% vs
SimpleHeuristics, 73.9% vs v19 and 62.8% vs v18.3.

v21 (v20 plus battle history, a transformer over the tokens and auxiliary prediction heads,
October 2026) is level with v20 at equal steps: 48.1% ± 2.2% head-to-head at 10M steps, 80.6% ±
2.5% vs SimpleHeuristics (`agent_docs/v21.md`).

The previous best, `models/ppo_pokemon_6v6_v18.3_best.zip`, wins 94.7% ± 1.4% vs RandomPlayer,
67.9% ± 2.9% vs SimpleHeuristics and 60.4% ± 3.0% vs v17.3sp (1,000 games each). Earlier versions
of this table counted episodes with positive reward as wins, which overstated results (e.g. 73.2%
vs SimpleHeuristics), and reported v20's head-to-heads with the opponent sampling its actions
(79.3% vs v19, 70.1% vs v18.3); both models play greedily now, as the deployed bots do.

## How It Works

v20 reads the battle each turn as tokens: one per Pokemon on each side (the opponent's unrevealed ones as placeholders), one per available move, and one for the field. Learned embeddings for species, moves, abilities, items and types feed per-token encoders and a shared trunk. Each move token scores "use" and "Terastallize and use", each of our Pokemon scores "switch to me", and a value head scores the position: 14 actions in all, masked to the legal ones. It is trained with its own PPO loop (worker processes, self-play against itself and a pool of past models) instead of Stable-Baselines3. v17-v19 used a flat feature vector, 13 actions and SB3 `MaskablePPO`; their code is kept so those models still load. Details: `agent_docs/v20.md`.

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+

### Install

```bash
# Python dependencies
pip install poke-env stable-baselines3 sb3-contrib gymnasium torch numpy

# Pokemon Showdown server
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
```

## Usage

All scripts should be run from the project root directory.

### Play against the bot locally

Start your local Showdown server:

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```

In a separate terminal, start the bot:

```bash
python -m players.play_bot
```

Open `http://localhost:8000` in your browser and challenge `MyPPOBot` to a Gen 9 Random Battle.

### Run on the Pokemon Showdown ladder

```bash
python -m players.ladder_bot
```

Edit `players/ladder_bot.py` to set your username, password, and number of games. Connects to the public Showdown server and queues for ranked Gen 9 Random Battles.

### Evaluate against bots

```bash
python -m eval.ppo_test --model models/ppo_pokemon_6v6_v20.pt                          # vs Random + SimpleHeuristics
python -m eval.ppo_test --model models/ppo_pokemon_6v6_v20.pt --opponents model \
    --opp-model models/ppo_pokemon_6v6_v18.3_best.zip --opp-deterministic --seat both # head-to-head, both seats
```

### Train

v20 (needs Showdown servers on ports 8000-8003: 3 for the workers, 1 for evals):

```bash
python -m training.ppo_v20 --workers 6 --servers 3 --stop-at 10000000   # resumes automatically
```

The SB3 trainer for v17-v19 needs servers on ports 8000-8004 (4 training envs + 1 eval env):

```bash
python -m training.ppo_embed_train_selfplay                      # from scratch
python -m training.ppo_embed_train_selfplay \
    --base_model models/ppo_pokemon_6v6_v18.3_best --reset_value_head   # warm start, fresh value head
```

### Evaluate against Foul Play

See the instructions at the top of `eval/foulplay_eval.py`.

## Project Structure

```
├── players/                          # Player wrappers + deployment bots
│   ├── v20_player.py                 #   v20/v21 policy / player, and loading of any model as an opponent
│   ├── ppo_player.py                 #   SB3 inference player (auto-detects v17-v19 models)
│   ├── play_bot.py                   #   Play against the bot locally
│   └── ladder_bot.py                 #   Run on the Showdown ladder
├── envs/                             # RL environments
│   ├── rl_player_v20.py              #   v20 tokens, 14-action mapping, two-agent env
│   ├── rl_player_v21.py              #   v21: v20's tokens plus battle history, aux labels
│   ├── rl_player_6v6_v18.py          #   v18/v19 observation space (677-dim)
│   ├── rl_player_6v6.py              #   v17 observation space (640-dim), for loading v17 models
│   └── wrappers.py                   #   Maskable action wrapper (SB3)
├── networks/
│   ├── v20_policy.py                 #   v20 network (per-token heads, shared trunk)
│   ├── v21_policy.py                 #   v21 network (transformer, per-token and aux heads)
│   ├── registry.py                   #   arch name -> featurizer and network, for loading
│   └── embedding_extractor*.py       #   v17-v19 feature extractors
├── training/
│   ├── ppo_v20.py                    #   v20/v21 PPO loop (workers, two-sided self-play, pool, evals)
│   └── ppo_embed_train_selfplay.py   #   SB3 self-play training (v17-v19)
├── eval/
│   ├── ppo_test.py                   #   Evaluate models vs Random / SimpleHeuristics / other models
│   ├── v20_eval.py                   #   Evaluation games for v20 models
│   └── foulplay_eval.py              #   Evaluate against Foul Play (MCTS bot) running in Docker
├── tools/                            # Checks against live battles, benchmarks, Mac runner
├── cloud/                            # Google Cloud training scripts
├── agent_docs/                       # Notes for agents working on the project
├── data/                             # Embedding vocabularies, random-battle sets
└── models/                           # Trained model checkpoints
```

## References

- Huang & Lee. "PPO and Self-Play for Pokemon Showdown." CoG 2019.
- Wang. "Pokemon Battle Agent with RL." MIT Thesis 2024.
- pmariglia. [Foul Play](https://github.com/pmariglia/foul-play) — search-based Pokemon Showdown bot.
