# Future Sight — RL Agent for Pokemon Showdown

A reinforcement learning bot for Gen 9 Random Battles on [Pokemon Showdown](https://pokemonshowdown.com/), trained with PPO and self-play over 100 million timesteps.

## Performance

| Opponent | Win Rate (500 games) |
|----------|---------------------|
| RandomPlayer | 100% |
| SimpleHeuristics | 73.2% |
| Best prior self-play agent | 61.8% |

## How It Works

The bot uses a PPO policy network with learned embeddings for Pokemon, moves, abilities, and items. It reads the battle state each turn, encodes it into a feature vector, and picks the highest-value action from 13 options (4 moves, 4 moves with Terastallize, 5 switches). Action masking enforces legal moves and blocks obviously bad ones like Electric moves into Ground types.

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
python -m eval.ppo_test                                                              # vs Random + SimpleHeuristics
python -m eval.ppo_test --opponents model --opp-model models/ppo_pokemon_6v6_v17.3sp # head-to-head
```

## Project Structure

```
├── players/                          # Player wrappers + deployment bots
│   ├── ppo_player.py                 #   PPO inference player (auto-detects model version)
│   ├── play_bot.py                   #   Play against the bot locally
│   ├── ladder_bot.py                 #   Run on the Showdown ladder
│   ├── search_player.py              #   Value-network-guided search player
│   ├── engine_search_player.py       #   PPO + poke-engine MCTS search
│   ├── nn_mcts_player.py             #   Conservative NN + engine MCTS
│   └── wang_mcts_player.py           #   Wang-style MCTS with NN value leaf
├── envs/                             # RL environments + gym wrappers
│   ├── rl_player_6v6.py              #   v17 observation space (640-dim)
│   ├── rl_player_6v6_v18.py          #   v18 observation space (677-dim)
│   └── wrappers.py                   #   Maskable action wrapper, curriculum wrapper
├── networks/                         # Neural network feature extractors
│   ├── embedding_extractor.py        #   v17 feature extractor
│   ├── embedding_extractor_sp.py     #   v17 self-play variant (max+mean pool)
│   ├── embedding_extractor_v18.py    #   v18 feature extractor
│   └── pretrain_embed.py             #   Embedding pretraining (types → effectiveness)
├── training/                         # Training scripts
│   ├── ppo_embed_train_selfplay.py   #   v18 mixed self-play + curriculum
│   ├── ppo_embed_train_parallel.py   #   v18 heuristic-only parallel training
│   ├── ppo_train.py                  #   v17 curriculum training
│   └── ppo_train_selfplay.py         #   v13 self-play training
├── eval/                             # Evaluation + testing
│   ├── ppo_test.py                   #   Evaluate models vs various opponents
│   ├── test_battle.py                #   Simple battle smoke test
│   └── debug.py                      #   Dataset diagnostics
├── utils/                            # Utilities
│   ├── state_bridge.py               #   poke-env → search server state conversion
│   ├── battle_cloner.py              #   JS search state → observation vectors
│   └── train_log.py                  #   Training monitor (TensorBoard logs)
├── data/                             # Embedding vocabularies + config
└── models/                           # Trained model checkpoints
```

## References

- Huang & Lee. "PPO and Self-Play for Pokemon Showdown." CoG 2019.
- Wang. "Pokemon Battle Agent with RL." MIT Thesis 2024.
- pmariglia. [Foul Play](https://github.com/pmariglia/foul-play) — search-based Pokemon Showdown bot.
