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

from envs.wrappers import MaskablePokeEnvWrapper
from players.ppo_player import PPOPlayer, ARCH_REGISTRY, _detect_architecture, _import_class


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

    # This Showdown fork sends each request before the turn's log; decide after the log
    from envs.rl_player_v20 import hold_requests_until_log
    hold_requests_until_log(rl_env.agent1)
    hold_requests_until_log(rl_env.agent2)

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

    # Reset once: the VecEnv auto-resets after every game, and resetting again would
    # forfeit that freshly started battle.
    obs = env.reset()
    for i in range(n_battles):
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

        # Count real wins; the episode return is not a reliable win signal
        if info[0].get("battle_won"):
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


def opponent_player(path, deterministic, port):
    """An env-stepped opponent for either checkpoint kind: V20Player for v20 .pt files, PPOPlayer
    for SB3 zips."""
    if path.endswith(".pt"):
        from players.v20_player import V20Player
        return V20Player(model_path=path, deterministic=deterministic,
                         battle_format="gen9randombattle",
                         server_configuration=get_server_config(port), start_listening=False)
    return PPOPlayer(model_path=path, data_dir=MODEL_CONFIG["data_dir"], deterministic=deterministic,
                     battle_format="gen9randombattle", server_configuration=get_server_config(port),
                     start_listening=False)


def main_v20(args):
    """Evaluate a torch checkpoint (.pt, v20 or v21): the RLEnvV20 env, the checkpoint's policy
    (its arch picks the featurizer and network), any opponent."""
    from players.v20_player import V20Policy, load_opponent
    from eval.v20_eval import play_games, print_actions
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
    policy = V20Policy.load(args.model, deterministic=True)
    opponents = {}
    if args.opponents in ("all", "random"):
        opponents["RandomPlayer"] = RandomPlayer(battle_format="gen9randombattle", start_listening=False)
    if args.opponents in ("all", "simple"):
        opponents["SimpleHeuristics"] = SimpleHeuristicsPlayer(battle_format="gen9randombattle",
                                                               start_listening=False)
    if args.opponents in ("all", "model") and args.opp_model:
        opponents[f"Model({os.path.basename(args.opp_model)})"] = load_opponent(
            args.opp_model, deterministic=args.opp_deterministic, data_dir=MODEL_CONFIG["data_dir"])
    elif args.opponents == "model":
        print("ERROR: --opp-model is required when opponents='model'")
        sys.exit(1)
    seats = {"1": [1], "2": [2], "both": [1, 2]}[args.seat]
    print("=" * 70)
    print(f"Pokemon RL - Evaluation ({policy.arch})\nModel: {args.model}\nBattles per opponent per seat: "
          f"{args.battles} | seats: {seats}")
    print("=" * 70)
    results = {}
    for name, opp in opponents.items():
        for seat in seats:
            print(f"\nTesting vs {name} (seat {seat})...")
            r = play_games(policy, opp, args.battles, args.port, seat=seat, progress=100)
            results[(name, seat)] = r
            print(f"  Win rate: {r['win_rate']:.1%} ± {r['ci_95']:.1%} (95% CI) | "
                  f"W/L/T {r['wins']}/{r['losses']}/{r['ties']} | avg length {r['mean_len']:.1f} decisions")
        if len(seats) == 2:
            w = sum(results[(name, s)]["wins"] for s in seats)
            n = sum(results[(name, s)]["games"] for s in seats)
            p = w / max(n, 1)
            print(f"  Both seats: {p:.1%} ± {1.96 * np.sqrt(p * (1 - p) / max(n, 1)):.1%} over {n} games")
    if results:
        print_actions(list(results.values())[-1]["action_counts"])
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    print(f"{'Opponent':<34} {'Seat':>4} {'Wins':>6} {'Games':>6} {'Win Rate':>9} {'95% CI':>8}")
    for (name, seat), r in results.items():
        print(f"{name:<34} {seat:>4} {r['wins']:>6} {r['games']:>6} {r['win_rate']:>8.1%} ±{r['ci_95']:>6.1%}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Pokemon RL model (v17-v19 SB3 zips, v20/v21 .pt)")
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
    parser.add_argument("--seat", choices=["1", "2", "both"], default="1",
                        help=".pt models: play as the challenger (1, p1), the challenged side (2), or both")
    args = parser.parse_args()
    if args.model.endswith(".pt") or os.path.exists(args.model + ".pt"):
        if not args.model.endswith(".pt"):
            args.model += ".pt"
        return main_v20(args)

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
            opponents[f"Model({os.path.basename(opp_model_path)})"] = opponent_player(
                opp_model_path, deterministic=True, port=args.port)

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

    print(f"\nBaseline (v18.3_best, 1,000 games, true wins):")
    print(f"  RandomPlayer 94.7% | SimpleHeuristics 67.9% | v17.3sp 60.4%")

    return results


if __name__ == "__main__":
    results = main()