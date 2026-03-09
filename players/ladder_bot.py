import asyncio
from players.ppo_player import PPOPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration
from poke_env.battle.abstract_battle import AbstractBattle

class LadderPPOPlayer(PPOPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_wins = 0
        self.real_losses = 0
        self.forfeits = 0

    def _battle_finished_callback(self, battle: AbstractBattle):
        if battle.turn <= 2:
            self.forfeits += 1
            print(f"  [Forfeit at turn {battle.turn} — ignoring]")
            return

        if battle.won:
            self.real_wins += 1
            result = "WIN"
        else:
            self.real_losses += 1
            result = "LOSS"

        total = self.real_wins + self.real_losses
        wr = self.real_wins / total * 100 if total > 0 else 0
        print(f"\n  Battle #{total}: {result} (turn {battle.turn})")
        print(f"  Record: {self.real_wins}W - {self.real_losses}L ({wr:.1f}%) | Forfeits ignored: {self.forfeits}")

async def main():
    USERNAME = "vibecoded"
    PASSWORD = "opus45"
    N_GAMES = 30

    try:
        import certifi, os
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass

    bot = LadderPPOPlayer(
        model_path="models/ppo_pokemon_6v6_v18.3_best",
        data_dir="data",
        deterministic=True,
        account_configuration=AccountConfiguration(USERNAME, PASSWORD),
        battle_format="gen9randombattle",
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=True,
    )

    print(f"\nBot '{bot.username}' connecting to Pokemon Showdown...")
    print(f"Playing {N_GAMES} ladder games...\n")
    await bot.ladder(N_GAMES)

    total = bot.real_wins + bot.real_losses
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Real games: {total}")
    print(f"Wins: {bot.real_wins}")
    print(f"Losses: {bot.real_losses}")
    print(f"Forfeits ignored: {bot.forfeits}")
    if total > 0:
        print(f"Win rate: {bot.real_wins / total:.1%}")
    print(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())