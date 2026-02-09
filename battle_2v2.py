from poke_env.player import (
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    cross_evaluate,
    background_cross_evaluate,
    background_evaluate_player,
)
import asyncio
import time
        

class MaxDamage2v2Player(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)

            if battle.can_tera:
                return self.create_order(best_move, terastallize=True)

            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)
'''
# Create players
random_player = RandomPlayer(
    battle_format="gen9randombattle2v2",
)
random_player_2 = RandomPlayer(
    battle_format="gen9randombattle2v2",
)
greedy_player = MaxDamage2v2Player(
    battle_format="gen9randombattle2v2",
)
simple_heuristics_player = SimpleHeuristicsPlayer(
    battle_format="gen9randombattle2v2",
)

# Run a battle
async def main():
    await random_player.battle_against(simple_heuristics_player, n_battles=20)
    print(f"Result: {random_player.n_won_battles} wins, {simple_heuristics_player.n_won_battles} losses")
    await random_player.battle_against(greedy_player, n_battles=20)
    print(f"Result: {random_player.n_won_battles} wins, {greedy_player.n_won_battles} losses")
    	

asyncio.run(main())
'''