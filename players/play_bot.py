import asyncio
from stable_baselines3.common.vec_env import VecNormalize
from players.ppo_player import PPOPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


# Needed to unpickle VecNormalize saved from training (__main__.MaskableVecNormalize)
class MaskableVecNormalize(VecNormalize):
    pass

async def main():
    bot = PPOPlayer(
        model_path="models/ppo_pokemon_6v6_v18.2",
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration("MyPPOBot", None),
        server_configuration=LocalhostServerConfiguration,
    )
    # Accept any challenges indefinitely
    await bot.accept_challenges(None, n_challenges=999999)

if __name__ == "__main__":
    asyncio.run(main())