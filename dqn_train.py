from RLPlayer6v6 import RL6v6Player
from RLPlayer2v2 import MaxDamage2v2Player
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CurriculumCallback(BaseCallback):
    """Switches opponents during training without resetting exploration."""
    
    def __init__(self, env, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.stages = [
            (0, RandomPlayer(battle_format="gen9randombattle"), "RandomPlayer"),
            (50000, MaxDamage2v2Player(battle_format="gen9randombattle"), "MaxDamagePlayer"),
            (100000, SimpleHeuristicsPlayer(battle_format="gen9randombattle"), "SimpleHeuristicsPlayer"),
        ]
        self.current_stage = 0
    
    def _on_step(self):
        # Check if we should advance to next stage
        if self.current_stage < len(self.stages) - 1:
            next_threshold = self.stages[self.current_stage + 1][0]
            if self.num_timesteps >= next_threshold:
                self.current_stage += 1
                _, opponent, name = self.stages[self.current_stage]
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Step {self.num_timesteps}: Switching to {name}")
                    print(f"{'='*60}\n")
                new_wrapped = SingleAgentWrapper(self.env, opponent)
                self.model.set_env(new_wrapped)
                
                self.model.env.reset()
                self.model._last_obs = self.model.env.reset()[0]
        return True


class EvalCallback(BaseCallback):
    """Periodically evaluates win rate against SimpleHeuristics."""
    
    def __init__(self, eval_freq=25000, n_eval=100, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.eval_env = None
        self.results = []
    
    def _on_training_start(self):
        # Create separate eval environment
        eval_base = RL6v6Player(battle_format="gen9randombattle")
        self.eval_env = SingleAgentWrapper(
            eval_base, 
            SimpleHeuristicsPlayer(battle_format="gen9randombattle")
        )
    
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            wins = 0
            for _ in range(self.n_eval):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                if reward > 0:
                    wins += 1
            
            win_rate = wins / self.n_eval
            exploration_rate = self.model.exploration_rate
            
            self.results.append({
                'step': self.num_timesteps,
                'win_rate': win_rate,
                'epsilon': exploration_rate
            })
            
            if self.verbose:
                print(f"\n>>> Eval @ {self.num_timesteps:,} steps: "
                      f"{win_rate:.1%} vs SimpleHeuristics | ε={exploration_rate:.3f}\n")
        
        return True


def train():
    # Base environment
    env = RL6v6Player(battle_format="gen9randombattle")
    
    # Start with RandomPlayer
    initial_opponent = RandomPlayer(battle_format="gen9randombattle")
    wrapped_env = SingleAgentWrapper(env, initial_opponent)
    
    # Model with tuned hyperparameters
    model = QRDQN(
        "MlpPolicy",
        wrapped_env,
        learning_rate=0.0001,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.25,
        exploration_final_eps=0.02,
        policy_kwargs={"net_arch": [512, 512]},
        verbose=1,
        tensorboard_log="./logs/qrdqn_pokemon/",
    )
    
    # Callbacks
    curriculum_cb = CurriculumCallback(env, verbose=1)
    eval_cb = EvalCallback(eval_freq=50000, n_eval=100, verbose=1)
    
    print("="*60)
    print("Starting training: 400k steps with curriculum learning")
    print("  - Stage 1 (0-50k): vs RandomPlayer")
    print("  - Stage 2 (50k-100k): vs MaxDamagePlayer")
    print("  - Stage 3 (100k-400k): vs SimpleHeuristicsPlayer")
    print("="*60 + "\n")
    
    # Single learn call - exploration decays smoothly across all stages
    model.learn(
        total_timesteps=700000,
        callback=[curriculum_cb, eval_cb],
        progress_bar=True,
    )
    
    # Save model
    model.save("models/qrdqn_pokemon_6v6_v12")
    
    # Print summary
    print("\n" + "="*60)
    print("Training complete! Results summary:")
    print("="*60)
    for r in eval_cb.results:
        print(f"  Step {r['step']:>7,}: {r['win_rate']:>5.1%} win rate (ε={r['epsilon']:.3f})")
    
    print(f"\nModel saved to: models/qrdqn_pokemon_6v6_v12")
    
    return model, eval_cb.results


if __name__ == "__main__":
    model, results = train()