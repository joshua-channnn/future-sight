import sys

sys.path.append("../src")

from poke_env import RandomPlayer
from poke_env.data import GenData
from poke_env.player import Player        
import asyncio

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)

            if battle.can_tera:
                return self.create_order(best_move, terastallize=True)

            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

async def main():
    random_player_5 = RandomPlayer()
    second_player_5 = RandomPlayer()
    max_dmg_player = MaxDamagePlayer()
    await max_dmg_player.battle_against(second_player_5, n_battles=10)

    # n_won_battles and n_finished_battles

    print(
        f"Player {max_dmg_player.username} won {max_dmg_player.n_won_battles} out of {max_dmg_player.n_finished_battles} played"
    )
    print(
        f"Player {second_player_5.username} won {second_player_5.n_won_battles} out of {second_player_5.n_finished_battles} played"
    )

    # Looping over battles

    for battle_tag, battle in random_player_5.battles.items():
        print(battle_tag, battle.won)

# Run the main function
asyncio.run(main())