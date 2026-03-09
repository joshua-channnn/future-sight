import os
import sys
import uuid
import argparse
import zipfile
import io
import importlib
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

from envs.wrappers import MaskablePokeEnvWrapper
from players.ppo_player import PPOPlayer


ARCH_REGISTRY = {
    444: {
        "name": "v17.2",
        "extractor_module": "networks.embedding_extractor",
        "extractor_class": "V17FeatureExtractor",
        "rl_env_module": "envs.rl_player_6v6",
        "rl_env_class": "RL6v6PlayerV17",
        "obs_size": 640,
    },
    700: {
        "name": "v17.3sp",
        "extractor_module": "networks.embedding_extractor_sp",
        "extractor_class": "V17SPFeatureExtractor",
        "rl_env_module": "envs.rl_player_6v6",
        "rl_env_class": "RL6v6PlayerV17",
        "obs_size": 640,
    },
    725: {
        "name": "v18",
        "extractor_module": "networks.embedding_extractor_v18",
        "extractor_class": "V18FeatureExtractor",
        "rl_env_module": "envs.rl_player_6v6_v18",
        "rl_env_class": "RL6v6PlayerV18",
        "obs_size": 677,
    },
}


def _detect_architecture(state_dict):
    """Detect model architecture from saved state dict by checking final MLP input size."""
    for prefix in ["features_extractor", "pi_features_extractor", "vf_features_extractor"]:
        ln_key = f"{prefix}.final_mlp.0.weight"
        if ln_key in state_dict:
            return state_dict[ln_key].shape[0]
    return 444  # Default to v17.2


def _import_class(module_name, class_name):
    """Dynamically import a class from a module."""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


EVAL_PORT = 8004

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


def load_model(model_path: str, env):
    """
    Load a v17.2/v17.3sp/v18 model with auto-detection.

    Models were saved with a 2-group differential optimizer,
    but SB3's MaskablePPO.load() expects a 1-group optimizer.
    We create a fresh model with the correct architecture, then
    load only the policy weights from the saved file.
    """
    load_path = model_path
    if not load_path.endswith(".zip"):
        if os.path.exists(load_path + ".zip"):
            load_path = load_path + ".zip"

    if not os.path.exists(load_path):
        print(f"ERROR: Model not found at {load_path}")
        sys.exit(1)

    print(f"Loading model from {load_path}...")

    with zipfile.ZipFile(load_path, "r") as zip_f:
        with zip_f.open("policy.pth") as f:
            buffer = io.BytesIO(f.read())
            policy_state = torch.load(buffer, map_location="cpu", weights_only=False)

    final_mlp_size = _detect_architecture(policy_state)
    if final_mlp_size not in ARCH_REGISTRY:
        print(f"ERROR: Unknown architecture with final_mlp size {final_mlp_size}")
        print(f"  Known: {list(ARCH_REGISTRY.keys())}")
        sys.exit(1)

    arch_info = ARCH_REGISTRY[final_mlp_size]
    extractor_class = _import_class(arch_info["extractor_module"], arch_info["extractor_class"])

    print(f"  ✓ Detected {arch_info['name']} ({final_mlp_size}-dim final MLP, {arch_info['obs_size']}-dim obs)")

    policy_kwargs = {
        "features_extractor_class": extractor_class,
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

    model.policy.load_state_dict(policy_state)
    model.policy.eval()

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  ✓ Loaded {total_params:,} parameters")

    return model, arch_info


def create_eval_env(opponent, arch_info, port=EVAL_PORT):
    """Create evaluation environment with the correct RL env for the model architecture."""
    server_config = get_server_config(port)
    uid = uuid.uuid4().hex[:6]

    rl_env_class = _import_class(arch_info["rl_env_module"], arch_info["rl_env_class"])

    rl_env = rl_env_class(
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


def evaluate(model, env, n_battles=500, verbose=True):
    """Evaluate model against a specific opponent with detailed stats."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate v17.2/v17.3sp/v18 Pokemon RL model")
    parser.add_argument("--model", type=str,
                        default="models/ppo_pokemon_6v6_v18.3_best",
                        help="Path to model (auto-detects architecture)")
    parser.add_argument("--battles", type=int, default=500,
                        help="Number of battles per opponent")
    parser.add_argument("--opponents", type=str, default="all",
                        choices=["all", "random", "simple", "model"],
                        help="Which opponents to test against")
    parser.add_argument("--opp-model", type=str, default=None,
                        help="Path to opponent model (auto-detects v17.2/v17.3sp/v18)")
    parser.add_argument("--opp-deterministic", action="store_true",
                        help="Use deterministic actions for opponent model")
    parser.add_argument("--port", type=int, default=EVAL_PORT,
                        help="Pokemon Showdown server port")
    args = parser.parse_args()

    n_battles = args.battles

    load_path = args.model
    if not load_path.endswith(".zip"):
        if os.path.exists(load_path + ".zip"):
            load_path = load_path + ".zip"
    if not os.path.exists(load_path):
        print(f"ERROR: Model not found at {load_path}")
        sys.exit(1)

    with zipfile.ZipFile(load_path, "r") as zip_f:
        with zip_f.open("policy.pth") as f:
            policy_state = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)

    final_mlp_size = _detect_architecture(policy_state)
    if final_mlp_size not in ARCH_REGISTRY:
        print(f"ERROR: Unknown architecture with final_mlp size {final_mlp_size}")
        sys.exit(1)
    arch_info = ARCH_REGISTRY[final_mlp_size]

    print("=" * 70)
    print(f"Pokemon RL — Evaluation ({arch_info['name']})")
    print(f"Model: {args.model}")
    print(f"Architecture: {arch_info['name']} ({arch_info['obs_size']}-dim obs)")
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

    if args.opponents in ("all", "model"):
        opp_model_path = args.opp_model
        if opp_model_path is None:
            if args.opponents == "model":
                print("ERROR: --opp-model is required when opponents='model'")
                sys.exit(1)
            # Skip model opponent if --opponents=all and no --opp-model given
        else:
            opponents[f"Model({os.path.basename(opp_model_path)})"] = PPOPlayer(
                model_path=opp_model_path,
                data_dir=MODEL_CONFIG["data_dir"],
                deterministic=True,
                battle_format="gen9randombattle",
                server_configuration=get_server_config(args.port),
                start_listening=False,
            )

    if not opponents:
        print("ERROR: No opponents selected")
        sys.exit(1)

    first_opp_name = list(opponents.keys())[0]
    env, rl_env = create_eval_env(opponents[first_opp_name], arch_info, port=args.port)
    model, _ = load_model(args.model, env)

    try:
        env.close()
    except Exception:
        pass

    for idx, (opp_name, opponent) in enumerate(opponents.items()):
        print(f"\n[{idx+1}/{len(opponents)}] Testing vs {opp_name}...")

        env, rl_env = create_eval_env(opponent, arch_info, port=args.port)
        result = evaluate(model, env, n_battles=n_battles, verbose=True)
        results[opp_name] = result

        print(f"  Win rate: {result['win_rate']:.1%} ± {result['ci_95']:.1%} (95% CI)")
        print(f"  Avg reward: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Avg length: {result['avg_length']:.1f} turns")

        try:
            env.close()
        except Exception:
            pass

    last_opp = list(results.keys())[-1]
    analyze_actions(results[last_opp]['action_counts'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Opponent':<25} {'Wins':>6} {'Win Rate':>10} {'95% CI':>10} {'Avg Rew':>10} {'Avg Len':>10}")
    print("-" * 70)
    for opp_name, r in results.items():
        print(f"{opp_name:<25} {r['wins']:>6} {r['win_rate']:>9.1%} "
              f"±{r['ci_95']:>7.1%} {r['avg_reward']:>10.2f} {r['avg_length']:>10.1f}")
    print("=" * 70)

    print(f"\nBaselines:")
    print(f"  v13 (hand-crafted):      72%")
    print(f"  v17.2 (diff LR):         74% peak / 67% smoothed")
    print(f"  v17.3sp (self-play):      69.5% peak / 60% smoothed")
    print(f"  v18 (cat eff + mixed SP): 71% peak / 62% smoothed")

    return results


if __name__ == "__main__":
    results = main()