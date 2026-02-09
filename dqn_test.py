from RLPlayer2v2 import MaxDamage2v2Player
from RLPlayer6v6 import RL6v6Player
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from masked_dqn import MaskedDQN

def evaluate(model_path, opponent, n_battles=500):
    env = RL6v6Player(battle_format="gen9randombattle")
    wrapped_env = SingleAgentWrapper(env, opponent)
    
    model = QRDQN.load(model_path, env=wrapped_env)
    
    wins = 0
    for i in range(n_battles):
        obs, info = wrapped_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1
    
    return wins

if __name__ == "__main__":
    model_path = "models/qrdqn_pokemon_6v6_v12"
    n_battles = 1000
    # Test vs Random
    wins = evaluate(model_path, RandomPlayer(battle_format="gen9randombattle"), n_battles)
    print(f"vs RandomPlayer: {wins}/{n_battles} ({100*wins/n_battles:.1f}%)")
    
    # Test vs MaxDamage
    wins = evaluate(model_path, MaxDamage2v2Player(battle_format="gen9randombattle"), n_battles)
    print(f"vs MaxDamage: {wins}/{n_battles} ({100*wins/n_battles:.1f}%)")

    # Test vs SimpleHeuristics
    wins = evaluate(model_path, SimpleHeuristicsPlayer(battle_format="gen9randombattle"), n_battles)
    print(f"vs SimpleHeuristics: {wins}/{n_battles} ({100*wins/n_battles:.1f}%)")