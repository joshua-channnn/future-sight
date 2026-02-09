"""
Deploy your bot on the REAL Pokemon Showdown ladder.

WARNING: This connects to the actual Pokemon Showdown server.
You need a registered account.

Instructions:
1. Create an account at https://play.pokemonshowdown.com/
2. Set your username and password below
3. Run this script
4. The bot will automatically search for ladder games
"""

import asyncio
import time
import numpy as np
from datetime import datetime
from sb3_contrib import MaskablePPO

from poke_env.player import Player
from stable_baselines3.common.vec_env import VecNormalize
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration
from poke_env.concurrency import POKE_LOOP

from RLPlayer6v6_v13 import RL6v6Player


# Needed to unpickle VecNormalize saved from training (__main__.MaskableVecNormalize)
class MaskableVecNormalize(VecNormalize):
    pass


class LadderBotPlayer(Player):
    """Bot that plays on the real Pokemon Showdown ladder."""
    
    def __init__(
        self,
        model_path: str,
        vec_normalize_path: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Load model
        self.model = MaskablePPO.load(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load normalization stats
        self.obs_rms = None
        if vec_normalize_path:
            import pickle
            try:
                with open(vec_normalize_path, "rb") as f:
                    vec_data = pickle.load(f)
                if hasattr(vec_data, 'obs_rms'):
                    self.obs_rms = vec_data.obs_rms
                    print(f"Loaded normalization stats")
            except Exception as e:
                print(f"Warning: Could not load vec_normalize: {e}")
        
        # Create RL env for embedding/action conversion
        self.rl_env = RL6v6Player(battle_format="gen9randombattle")
        self.clip_obs = 10.0
        
        # Track stats
        self.wins = 0
        self.losses = 0
        self.ignored = 0
        self.battle_history = []
        self._last_leave_ts = 0.0
    
    def choose_move(self, battle):
        # Embed battle state
        obs = self.rl_env.embed_battle(battle)
        
        # Normalize observation
        if self.obs_rms is not None:
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        
        # Get action mask
        mask = self.rl_env.get_action_mask(battle)
        
        # Predict action
        action, _ = self.model.predict(
            obs.reshape(1, -1).astype(np.float32),
            deterministic=True,
            action_masks=mask.reshape(1, -1)
        )
        
        return self.rl_env.action_to_order(int(action[0]), battle)
    
    def _battle_finished_callback(self, battle):
        """Called when a battle ends."""
        # Ensure we leave the battle room to close the tab
        if hasattr(self.ps_client, "websocket"):
            now = time.time()
            if now - self._last_leave_ts >= 2.0:
                self._last_leave_ts = now
                async def _leave():
                    # Give PS a moment to finalize the battle end
                    await asyncio.sleep(0.25)
                    try:
                        # Prefer sending /leave in the room itself
                        await self.ps_client.send_message("/leave", battle.battle_tag)
                    except Exception:
                        # Fallback to global /leave <room>
                        await self.ps_client.send_message(f"/leave {battle.battle_tag}")
                    try:
                        # Extra fallback: /part is also accepted
                        await self.ps_client.send_message("/part", battle.battle_tag)
                    except Exception:
                        await self.ps_client.send_message(f"/part {battle.battle_tag}")
                    # Retry once in case the first message was dropped
                    await asyncio.sleep(1.0)
                    try:
                        await self.ps_client.send_message("/leave", battle.battle_tag)
                    except Exception:
                        await self.ps_client.send_message(f"/leave {battle.battle_tag}")
                    try:
                        await self.ps_client.send_message("/part", battle.battle_tag)
                    except Exception:
                        await self.ps_client.send_message(f"/part {battle.battle_tag}")
                asyncio.run_coroutine_threadsafe(_leave(), POKE_LOOP)
        if battle.turn <= 4:
            self.ignored += 1
            self.battle_history.append({
                'time': datetime.now(),
                'result': "IGNORED",
                'turns': battle.turn,
                'battle_tag': battle.battle_tag,
            })
            print(f"\n{'='*50}")
            print(f"Battle ignored (turn {battle.turn})")
            print(f"Ignored: {self.ignored}")
            print(f"{'='*50}\n")
            return

        won = battle.won
        if won:
            self.wins += 1
            result = "WIN"
        else:
            self.losses += 1
            result = "LOSS"
        
        self.battle_history.append({
            'time': datetime.now(),
            'result': result,
            'turns': battle.turn,
            'battle_tag': battle.battle_tag,
        })
        
        total = self.wins + self.losses
        win_rate = self.wins / total * 100 if total > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"Battle #{total} - {result}")
        print(f"Turns: {battle.turn}")
        print(f"Record: {self.wins}W - {self.losses}L ({win_rate:.1f}%) | Ignored: {self.ignored}")
        print(f"{'='*50}\n")


async def main():
    # ======================
    # CONFIGURATION - EDIT THESE
    # ======================
    MODEL_PATH = "models/ppo_pokemon_6v6_v14"
    VEC_PATH = "models/vec_normalize_pokemon_6v6_v14.pkl"
    
    # Your Pokemon Showdown credentials
    # Create an account at https://play.pokemonshowdown.com/ first!
    USERNAME = "vibecoded"  # CHANGE THIS
    PASSWORD = "opus45"  # CHANGE THIS
    
    # Number of ladder games to play
    N_GAMES = 30
    # ======================
    
    if USERNAME == "YourBotUsername":
        print("ERROR: Please edit the script and set your Pokemon Showdown credentials!")
        print("1. Create an account at https://play.pokemonshowdown.com/")
        print("2. Edit USERNAME and PASSWORD in this script")
        return
    
    print("=" * 60)
    print("Pokemon RL Bot - REAL LADDER MODE")
    print("=" * 60)
    print(f"Username: {USERNAME}")
    print(f"Games to play: {N_GAMES}")
    print(f"Format: gen9randombattle")
    print("=" * 60)
    print()
    print("WARNING: This will play on the real Pokemon Showdown server!")
    print("Your rating will be affected.")
    print()
    input("Press Enter to continue (Ctrl+C to cancel)...")

    # Ensure SSL certs are available (macOS python sometimes lacks trust store)
    try:
        import certifi
        import os
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass
    
    # Create bot player
    bot = LadderBotPlayer(
        model_path=MODEL_PATH,
        vec_normalize_path=VEC_PATH,
        account_configuration=AccountConfiguration(USERNAME, PASSWORD),
        battle_format="gen9randombattle",
        server_configuration=ShowdownServerConfiguration,
        max_concurrent_battles=4,
        start_timer_on_battle_start=True,
    )
    
    print(f"\nBot '{bot.username}' connecting to Pokemon Showdown...")
    print(f"Starting ladder search for {N_GAMES} games...\n")
    
    # Play ladder games
    await bot.ladder(N_GAMES)
    
    # Print final stats
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total games: {bot.wins + bot.losses}")
    print(f"Wins: {bot.wins}")
    print(f"Losses: {bot.losses}")
    print(f"Ignored (turn <=4): {bot.ignored}")
    print(f"Win rate: {bot.wins / (bot.wins + bot.losses) * 100:.1f}%")
    print("=" * 60)
    
    # Save results
    results_file = f"ladder_results_{USERNAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Username: {USERNAME}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Total: {bot.wins + bot.losses}\n")
        f.write(f"Wins: {bot.wins}\n")
        f.write(f"Losses: {bot.losses}\n")
        f.write(f"Ignored (turn <=4): {bot.ignored}\n")
        f.write(f"Win Rate: {bot.wins / (bot.wins + bot.losses) * 100:.1f}%\n\n")
        f.write("Battle History:\n")
        for b in bot.battle_history:
            f.write(f"  {b['time']} - {b['result']} in {b['turns']} turns\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped.")