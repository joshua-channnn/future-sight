"""
evaluate_v17.py — Evaluation script for v17.2 embedding-based models.

Tests the model against multiple opponents (Random, MaxDamage, SimpleHeuristics)
with detailed action distribution analysis.

Usage:
  python evaluate_v17.py                                    # Use default best model path
  python evaluate_v17.py --model models/ppo_pokemon_6v6_v17.2_best.zip
  python evaluate_v17.py --battles 500
  python evaluate_v17.py --opponents simple                 # Only test vs SimpleHeuristics
"""

import os
import sys
import uuid
import argparse
import zipfile
import io
import numpy as np
from collections import defaultdict

import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from RLPlayer6v6 import RL6v6PlayerV17
from embedding_extractor import V17FeatureExtractor
from wrappers import MaskablePokeEnvWrapper


# ============================================================================
# Configuration
# ============================================================================

EVAL_PORT = 8004  # Use port 8004 (separate from training ports 8000-8003)

def get_server_config(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://127.0.0.1:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

MODEL_CONFIG = {
    "features_dim": 256,
    "pokemon_repr_dim": 128,
    "embed_dropout": 0.05,
    "net_arch_pi": [256, 128],
    "net_arch_vf": [256, 128],
    "share_features_extractor": False,
    "data_dir": "data",
}


# ============================================================================
# Model Loading (handles differential LR optimizer)
# ============================================================================

def load_v17_model(model_path: str, env):
    """
    Load a v17.2 model properly.
    
    v17.2 models were saved with a 2-group differential optimizer,
    but SB3's MaskablePPO.load() expects a 1-group optimizer.
    We create a fresh model with the correct architecture, then
    load only the policy weights from the saved file.
    """
    # Resolve path
    load_path = model_path
    if not load_path.endswith(".zip"):
        if os.path.exists(load_path + ".zip"):
            load_path = load_path + ".zip"
    
    if not os.path.exists(load_path):
        print(f"ERROR: Model not found at {load_path}")
        sys.exit(1)
    
    print(f"Loading model from {load_path}...")
    
    # Create fresh model with correct architecture
    policy_kwargs = {
        "features_extractor_class": V17FeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": MODEL_CONFIG["features_dim"],
            "config_path": f"{MODEL_CONFIG['data_dir']}/embedding_config.json",
            "pokemon_repr_dim": MODEL_CONFIG["pokemon_repr_dim"],
            "embed_dropout": MODEL_CONFIG["embed_dropout"],
        },
        "net_arch": dict(
            pi=MODEL_CONFIG["net_arch_pi"],
            vf=MODEL_CONFIG["net_arch_vf"],
        ),
        "share_features_extractor": MODEL_CONFIG["share_features_extractor"],
        "activation_fn": torch.nn.ReLU,
    }
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    
    # Load only policy weights from saved file
    with zipfile.ZipFile(load_path, "r") as zip_f:
        with zip_f.open("policy.pth") as f:
            buffer = io.BytesIO(f.read())
            policy_state = torch.load(buffer, map_location="cpu", weights_only=False)
    
    model.policy.load_state_dict(policy_state)
    model.policy.eval()  # Set to eval mode (disables dropout)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  ✓ Loaded {total_params:,} parameters")
    
    return model


# ============================================================================
# Environment Creation
# ============================================================================

def create_eval_env(opponent, port=EVAL_PORT):
    """Create evaluation environment with specified opponent."""
    server_config = get_server_config(port)
    uid = uuid.uuid4().hex[:6]
    
    rl_env = RL6v6PlayerV17(
        battle_format="gen9randombattle",
        data_dir=MODEL_CONFIG["data_dir"],
        server_configuration=server_config,
        account_configuration1=AccountConfiguration.generate(f"EvalBot-{uid}", rand=True),
        account_configuration2=AccountConfiguration.generate(f"EvalOpp-{uid}", rand=True),
    )
    
    wrapped = SingleAgentWrapper(rl_env, opponent)
    env = MaskablePokeEnvWrapper(wrapped, rl_env)
    env = TimeLimit(env, max_episode_steps=500)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    return vec_env, rl_env


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, env, n_battles=500, verbose=True):
    """
    Evaluate model against a specific opponent with detailed stats.
    """
    wins = 0
    total_reward = 0
    episode_lengths = []
    rewards_list = []
    action_counts = defaultdict(int)
    
    for i in range(n_battles):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            masks = env.env_method("action_masks")[0]
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            action_counts[int(action[0])] += 1
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            done = done[0]
        
        total_reward += episode_reward
        rewards_list.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_reward > 0:
            wins += 1
        
        if verbose and (i + 1) % 100 == 0:
            current_wr = wins / (i + 1)
            print(f"    {i+1}/{n_battles} — running win rate: {current_wr:.1%}")
    
    win_rate = wins / n_battles
    avg_reward = total_reward / n_battles
    avg_length = np.mean(episode_lengths)
    std_reward = np.std(rewards_list)
    
    # 95% confidence interval for win rate
    ci = 1.96 * np.sqrt(win_rate * (1 - win_rate) / n_battles)
    
    return {
        'wins': wins,
        'n_battles': n_battles,
        'win_rate': win_rate,
        'ci_95': ci,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'action_counts': dict(action_counts),
    }


def analyze_actions(action_counts):
    """Analyze action distribution to understand agent behavior."""
    total = sum(action_counts.values())
    if total == 0:
        return
    
    moves_no_tera = sum(action_counts.get(i, 0) for i in range(4))
    moves_with_tera = sum(action_counts.get(i, 0) for i in range(4, 8))
    switches = sum(action_counts.get(i, 0) for i in range(8, 13))
    
    print(f"\n  Action Distribution ({total:,} total actions):")
    print(f"    Moves (no Tera):   {moves_no_tera:>7,} ({100*moves_no_tera/total:>5.1f}%)")
    print(f"    Moves (with Tera): {moves_with_tera:>7,} ({100*moves_with_tera/total:>5.1f}%)")
    print(f"    Switches:          {switches:>7,} ({100*switches/total:>5.1f}%)")
    
    print(f"\n  Individual Actions:")
    action_names = {
        0: "Move 1", 1: "Move 2", 2: "Move 3", 3: "Move 4",
        4: "Move 1+Tera", 5: "Move 2+Tera", 6: "Move 3+Tera", 7: "Move 4+Tera",
        8: "Switch 1", 9: "Switch 2", 10: "Switch 3", 11: "Switch 4", 12: "Switch 5",
    }
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        name = action_names.get(action, f"Action {action}")
        bar = "█" * int(40 * count / total)
        print(f"    {name:>12}: {count:>7,} ({100*count/total:>5.1f}%) {bar}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate v17.2 Pokemon RL model")
    parser.add_argument("--model", type=str, 
                        default="models/ppo_pokemon_6v6_v17.2",
                        help="Path to model (with or without .zip)")
    parser.add_argument("--battles", type=int, default=500,
                        help="Number of battles per opponent")
    parser.add_argument("--opponents", type=str, default="all",
                        choices=["all", "random", "simple"],
                        help="Which opponents to test against")
    parser.add_argument("--port", type=int, default=EVAL_PORT,
                        help="Pokemon Showdown server port")
    args = parser.parse_args()
    
    n_battles = args.battles
    
    print("=" * 70)
    print(f"Pokemon RL v17.2 — Evaluation")
    print(f"Model: {args.model}")
    print(f"Battles per opponent: {n_battles}")
    print("=" * 70)
    
    results = {}
    opponents = {}
    
    if args.opponents in ("all", "random"):
        opponents["RandomPlayer"] = RandomPlayer(
            battle_format="gen9randombattle",
            server_configuration=get_server_config(args.port),
            start_listening=False,
        )
    
    if args.opponents in ("all", "simple"):
        opponents["SimpleHeuristics"] = SimpleHeuristicsPlayer(
            battle_format="gen9randombattle",
            server_configuration=get_server_config(args.port),
            start_listening=False,
        )
    
    # Create env with first opponent just for model loading
    first_opp_name = list(opponents.keys())[0]
    env, rl_env = create_eval_env(opponents[first_opp_name], port=args.port)
    model = load_v17_model(args.model, env)
    
    try:
        env.close()
    except:
        pass
    
    # Evaluate against each opponent
    for idx, (opp_name, opponent) in enumerate(opponents.items()):
        print(f"\n[{idx+1}/{len(opponents)}] Testing vs {opp_name}...")
        
        env, rl_env = create_eval_env(opponent, port=args.port)
        result = evaluate(model, env, n_battles=n_battles, verbose=True)
        results[opp_name] = result
        
        print(f"  Win rate: {result['win_rate']:.1%} ± {result['ci_95']:.1%} (95% CI)")
        print(f"  Avg reward: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Avg length: {result['avg_length']:.1f} turns")
        
        try:
            env.close()
        except:
            pass
    
    # Action analysis for the last opponent tested
    last_opp = list(results.keys())[-1]
    analyze_actions(results[last_opp]['action_counts'])
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Opponent':<25} {'Wins':>6} {'Win Rate':>10} {'95% CI':>10} {'Avg Rew':>10} {'Avg Len':>10}")
    print("-" * 70)
    for opp_name, r in results.items():
        print(f"{opp_name:<25} {r['wins']:>6} {r['win_rate']:>9.1%} "
              f"±{r['ci_95']:>7.1%} {r['avg_reward']:>10.2f} {r['avg_length']:>10.1f}")
    print("=" * 70)
    
    # Comparison with baselines
    print(f"\nBaselines:")
    print(f"  v13 (hand-crafted):  72%")
    print(f"  v17.1 (frozen embed): 69.5% peak")
    print(f"  v17.2 training best:  74.0%")
    
    return results


if __name__ == "__main__":
    results = main()