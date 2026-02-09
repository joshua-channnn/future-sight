from poke_env.environment.env import PokeEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player import RandomPlayer, SingleBattleOrder, DefaultBattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from battle_2v2 import MaxDamage2v2Player


import numpy as np



class RL2v2Player(PokeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def action_spaces(self):
        return {agent: Discrete(5) for agent in self.possible_agents}
    
    @property
    def observation_spaces(self):
        return {agent: Box(low=0, high=1, shape=(21,), dtype=np.float32) for agent in self.possible_agents}
    
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        state = []

        my_pokemon = battle.active_pokemon
        # player pokemon current HP (default to fainted)
        state.append(my_pokemon.current_hp_fraction if my_pokemon else 0.0)
        
        opp_pokemon = battle.opponent_active_pokemon
        # opponent pokemon current HP (default to fully healthy)
        state.append(opp_pokemon.current_hp_fraction if opp_pokemon else 1.0)

        # player bench HP
        my_bench = [mon for mon in battle.team.values() if not mon.active]
        state.append(my_bench[0].current_hp_fraction if my_bench else 0.0)

        # opponent bench HP
        opp_bench = [mon for mon in battle.opponent_team.values() if not mon.active]
        state.append(opp_bench[0].current_hp_fraction if opp_bench else 1.0)

        max_power = max([m.base_power or 0 for m in battle.available_moves], default=0)
        for i in range(4):
            if i < len(battle.available_moves): # check if 4 available moves
                move = battle.available_moves[i]
                state.append(move.base_power / 150 if move.base_power else 0.0)

                # is strongest move
                is_strongest = 1.0 if (move.base_power or 0) == max_power else 0.0
                state.append(is_strongest)

                # STAB check
                has_stab = 1.0 if my_pokemon and move.type in my_pokemon.types else 0.0
                state.append(has_stab)

                # type effectiveness, (0 -> 1) = 0x -> 4x
                if opp_pokemon:
                    multiplier = opp_pokemon.damage_multiplier(move)
                    state.append(multiplier / 4) 
                else:
                    state.append(0.5) # 0.5 = neutral


            else:
                state.append(0.0) # append 0 if move not available
                state.append(0.0) # is_strongest
                state.append(0.0)  # STAB
                state.append(0.5) # append 0.5 for neutral type effectiveness

        # switch ability state, 1 if can switch, 0 if not
        state.append(1.0 if battle.available_switches else 0.0)

        return np.array(state, dtype = np.float32)
    

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value = 2.0,
            hp_value = 1.0,
            victory_value = 15.0,
        )

    def action_to_order(self, action: int, battle: AbstractBattle, fake: bool = False, strict: bool = True) -> BattleOrder:
        # Action 0-3 = use a move
        if action < 4 and action < len(battle.available_moves):
            return SingleBattleOrder(battle.available_moves[action])
        
        # Action 4 = switch
        if battle.available_switches:
            return SingleBattleOrder(battle.available_switches[0])
        
        # fallback, pick random
        if battle.available_moves:
            return SingleBattleOrder(battle.available_moves[0])
        if battle.available_switches:
            return SingleBattleOrder(battle.available_switches[0])
        
        return DefaultBattleOrder()
    
    def order_to_action(self, order: BattleOrder, battle: AbstractBattle, fake: bool = False, strict: bool = True) -> int:
        # if move, find move idx
        if order.order in battle.available_moves:
            return battle.available_moves.index(order.order)

        # if switch, return 4
        if order.order in battle.available_switches:
            return 4

        # unknown
        return -1


