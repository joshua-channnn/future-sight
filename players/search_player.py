import json
import time
import requests
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from poke_env.player import Player, SingleBattleOrder, DefaultBattleOrder, BattleOrder
from poke_env.battle.abstract_battle import AbstractBattle

from players.ppo_player import PPOPlayer, _detect_architecture, _import_class, ARCH_REGISTRY
from utils.state_bridge import battle_to_search_request, get_showdown_actions


PER_POKEMON = 44       # Each pokemon block is 44 dims
N_POKEMON = 12         # 6 ours + 6 opponent
POKEMON_BLOCK = N_POKEMON * PER_POKEMON  # 528

# Offsets within a pokemon block (after 13 index dims)
IDX_HP = 19            # HP fraction (1 float)
IDX_STATUS = 20        # Status one-hot (7 floats: none, brn, frz, par, psn, slp, tox)
IDX_BOOSTS = 27        # Boosts (7 floats: atk, def, spa, spd, spe, acc, eva)
IDX_ALIVE = 34         # Alive flag (1 float)

# Pokemon ordering in observation
# 0: our active, 1-5: our bench, 6: opp active, 7-11: opp bench

STATUS_ORDER = ['none', 'brn', 'frz', 'par', 'psn', 'slp', 'tox']


class SearchPlayer(PPOPlayer):
    """
    PPO player enhanced with depth-1 search using value network scoring.

    For each decision:
    1. Gets policy probs and V(current) from the trained model
    2. Simulates each action forward via search server
    3. Patches observation with simulated outcomes
    4. Scores patched observations with V(s')
    5. Combines policy prior + value advantage

    Falls back to pure policy if search server is unavailable.
    """

    def __init__(
        self,
        model_path: str,
        search_server_url: str = "http://localhost:9001",
        search_weight: float = 0.5,
        n_opponent_samples: int = 3,
        min_tera_turn: int = 2,
        search_timeout: float = 2.0,
        verbose: bool = False,
        **kwargs,
    ):
        # Set attributes BEFORE super().__init__() — parent may trigger choose_move
        self.search_server_url = search_server_url
        self.search_weight = search_weight
        self.n_opponent_samples = n_opponent_samples
        self.min_tera_turn = min_tera_turn
        self.search_timeout = search_timeout
        self.verbose = verbose
        self._search_stats = {
            'total_calls': 0,
            'successful_searches': 0,
            'fallback_to_policy': 0,
            'avg_search_time_ms': 0,
            'search_overrides': 0,
        }

        super().__init__(model_path=model_path, **kwargs)

    def _get_value(self, obs: np.ndarray) -> float:
        """Get V(s) from the trained value network."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if hasattr(self.policy, 'predict_values'):
                value = self.policy.predict_values(obs_tensor)
                return value.item()
            # Fallback: manual forward through vf
            features = self.policy.extract_features(obs_tensor, self.policy.vf_features_extractor)
            vf_latent = self.policy.mlp_extractor.forward_critic(features)
            value = self.policy.value_net(vf_latent)
            return value.item()

    def _get_policy_scores(self, obs: np.ndarray, action_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get policy probabilities AND raw logits for all actions."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            dist = self.policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits.clone()
            raw_logits = logits.squeeze(0).numpy().copy()
            logits[~mask_tensor] = float('-inf')
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        return probs, raw_logits

    def _patch_obs_with_outcome(
        self,
        obs: np.ndarray,
        outcome: dict,
    ) -> np.ndarray:
        """
        Create a modified observation by patching HP/status/boosts/alive
        from the simulation outcome.

        The outcome dict has:
          p1hp: [{species, hp, maxhp, hpFraction, fainted, status, active}, ...]
          p2hp: [{species, hp, maxhp, hpFraction, fainted, status, active}, ...]
          p1active: {species, hp, maxhp, status, boosts, ...} or null
          p2active: {species, hp, maxhp, status, boosts, ...} or null
          p1alive: int
          p2alive: int
          ended: bool
          winner: str or null

        We patch:
          - HP fractions for all pokemon
          - Status one-hot for active pokemon
          - Boosts for active pokemon
          - Alive flags
        """
        patched = obs.copy()

        p1hp = outcome.get('p1hp', [])
        for i, mon in enumerate(p1hp):
            if i >= 6:
                break
            base = i * PER_POKEMON
            patched[base + IDX_HP] = mon.get('hpFraction', 0.0)
            patched[base + IDX_ALIVE] = 0.0 if mon.get('fainted', False) else 1.0

        p2hp = outcome.get('p2hp', [])
        for i, mon in enumerate(p2hp):
            if i >= 6:
                break
            base = (6 + i) * PER_POKEMON
            patched[base + IDX_HP] = mon.get('hpFraction', 0.0)
            patched[base + IDX_ALIVE] = 0.0 if mon.get('fainted', False) else 1.0

        p1active = outcome.get('p1active')
        if p1active:
            base = 0  # Our active is pokemon 0
            self._patch_status(patched, base, p1active.get('status'))
            self._patch_boosts(patched, base, p1active.get('boosts', {}))
            if p1active.get('maxhp', 0) > 0:
                patched[base + IDX_HP] = p1active['hp'] / p1active['maxhp']

        p2active = outcome.get('p2active')
        if p2active:
            base = 6 * PER_POKEMON  # Opponent active is pokemon 6
            self._patch_status(patched, base, p2active.get('status'))
            self._patch_boosts(patched, base, p2active.get('boosts', {}))
            if p2active.get('maxhp', 0) > 0:
                patched[base + IDX_HP] = p2active['hp'] / p2active['maxhp']

        return patched

    def _patch_status(self, obs: np.ndarray, base: int, status: Optional[str]):
        """Patch status one-hot at the given pokemon block offset."""
        obs[base + IDX_STATUS: base + IDX_STATUS + 7] = 0.0
        if status is None or status == '' or status == 'none':
            obs[base + IDX_STATUS] = 1.0  # 'none' flag
        elif status in STATUS_ORDER:
            idx = STATUS_ORDER.index(status)
            obs[base + IDX_STATUS + idx] = 1.0
        else:
            obs[base + IDX_STATUS] = 1.0  # default to none

    def _patch_boosts(self, obs: np.ndarray, base: int, boosts: dict):
        """Patch boost values (normalized to [-1, 1] by dividing by 6)."""
        boost_names = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
        for j, name in enumerate(boost_names):
            val = boosts.get(name, 0)
            obs[base + IDX_BOOSTS + j] = val / 6.0

    def _search_actions(
        self,
        battle: AbstractBattle,
        obs: np.ndarray,
        action_mask: np.ndarray,
    ) -> Optional[Dict[int, float]]:
        """
        Depth-1 search: simulate each action, score resulting states with V(s').

        Returns dict of {action_index: V(s')} or None if search fails.
        """
        actions = get_showdown_actions(battle, action_mask)
        if not actions:
            return None

        choice_strings = [choice for _, choice in actions]

        request = battle_to_search_request(
            battle,
            p1_actions=choice_strings,
            n_opponent_samples=self.n_opponent_samples,
        )
        if request is None:
            return None

        try:
            t0 = time.time()
            response = requests.post(
                f"{self.search_server_url}/evaluate",
                json=request,
                timeout=self.search_timeout,
            )
            elapsed_ms = (time.time() - t0) * 1000

            if response.status_code != 200:
                if self.verbose:
                    print(f"  Search server error: {response.status_code}")
                return None

            data = response.json()
            results = data.get('results', data)
            if isinstance(results, dict) and 'error' in results:
                if self.verbose:
                    print(f"  Search error: {results['error']}")
                return None

            choice_to_idx = {choice: idx for idx, choice in actions}

            value_scores = {}
            for r in results:
                action_str = r.get('action')
                if action_str not in choice_to_idx:
                    continue

                action_idx = choice_to_idx[action_str]
                samples = r.get('samples', [])

                if not samples:
                    continue

                values = []
                for sample in samples:
                    if sample.get('ended'):
                        winner = sample.get('winner', '')
                        if 'p1' in str(winner).lower() or 'searchp1' in str(winner).lower():
                            values.append(1.0)
                        elif 'p2' in str(winner).lower() or 'searchp2' in str(winner).lower():
                            values.append(-1.0)
                        else:
                            values.append(0.0)
                    else:
                        patched_obs = self._patch_obs_with_outcome(obs, sample)
                        v_prime = self._get_value(patched_obs)
                        values.append(v_prime)

                if values:
                    value_scores[action_idx] = float(np.mean(values))

            self._search_stats['total_calls'] += 1
            self._search_stats['successful_searches'] += 1
            alpha = 0.1
            self._search_stats['avg_search_time_ms'] = (
                (1 - alpha) * self._search_stats['avg_search_time_ms']
                + alpha * elapsed_ms
            )

            if self.verbose:
                meta = data.get('meta', {})
                print(f"  Search: {len(value_scores)} actions scored in {elapsed_ms:.0f}ms "
                      f"(server: {meta.get('elapsed_ms', '?')}ms)")
                v_current = self._get_value(obs)
                for action_idx, v in sorted(value_scores.items(), key=lambda x: -x[1]):
                    choice = dict(actions).get(action_idx, f"action_{action_idx}")
                    advantage = v - v_current
                    print(f"    {choice}: V={v:.3f} (adv={advantage:+.3f})")

            return value_scores

        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"  Search server unavailable: {e}")
            self._search_stats['total_calls'] += 1
            self._search_stats['fallback_to_policy'] += 1
            return None

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Combines policy probabilities + value network search scores.

        Strategy:
        1. Get policy probs from trained model
        2. Simulate each action → get V(s') from value network
        3. Convert value advantages to a probability distribution via softmax
        4. Mix: final = (1-w) * policy_probs + w * value_probs
        5. Pick argmax

        The softmax temperature controls how peaked the value distribution is.
        Higher temperature → more uniform → less aggressive overrides.
        """
        obs = self._rl_env.embed_battle(battle)
        action_mask = self._rl_env.get_action_mask(battle)

        if self.min_tera_turn > 0 and battle.turn < self.min_tera_turn:
            tera_backup = action_mask[4:8].copy()
            action_mask[4:8] = 0.0
            if action_mask.sum() == 0:
                action_mask[4:8] = tera_backup

        policy_probs, raw_logits = self._get_policy_scores(obs, action_mask)
        v_current = self._get_value(obs)

        search_values = self._search_actions(battle, obs, action_mask)

        if search_values is not None and len(search_values) > 0:
            legal_actions = [i for i in range(13) if action_mask[i] > 0]
            
            value_logits = np.full(13, float('-inf'), dtype=np.float32)
            for i in legal_actions:
                if i in search_values:
                    # Advantage = V(s') - V(s)
                    value_logits[i] = search_values[i] - v_current
                else:
                    # No search data: use 0 advantage (neutral)
                    value_logits[i] = 0.0

            # Temperature-scaled softmax over advantages
            # temperature=1.0 → raw advantages as logits
            # temperature=0.5 → more peaked (trust search more)
            # temperature=2.0 → flatter (trust search less)
            temperature = 1.0
            value_logits_scaled = value_logits / temperature
            # Stable softmax
            max_logit = value_logits_scaled[value_logits_scaled != float('-inf')].max()
            exp_logits = np.where(
                value_logits_scaled != float('-inf'),
                np.exp(value_logits_scaled - max_logit),
                0.0
            )
            value_probs = exp_logits / exp_logits.sum()

            w = self.search_weight
            combined = (1 - w) * policy_probs + w * value_probs

            combined = combined * (action_mask > 0)
            total = combined.sum()
            if total > 0:
                combined = combined / total

            action = int(combined.argmax())

            if self.verbose:
                policy_choice = int(policy_probs.argmax())
                if action != policy_choice:
                    actions_map = dict(get_showdown_actions(battle, action_mask))
                    best_choice = actions_map.get(action, f"action_{action}")
                    policy_best = actions_map.get(policy_choice, f"action_{policy_choice}")
                    p_old = combined[policy_choice]
                    p_new = combined[action]
                    print(f"  Turn {battle.turn}: Search overrode policy! "
                          f"{policy_best}({p_old:.3f}) -> {best_choice}({p_new:.3f}) "
                          f"(V_cur={v_current:.3f})")
                    self._search_stats['search_overrides'] += 1
        else:
            self._search_stats['fallback_to_policy'] += 1
            if self.deterministic:
                masked_logits = np.where(action_mask > 0, policy_probs, -np.inf)
                action = int(masked_logits.argmax())
            else:
                probs = policy_probs * action_mask
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                    action = np.random.choice(13, p=probs)
                else:
                    action = int(action_mask.argmax())

        return self._rl_env.action_to_order(action, battle)

    def get_search_stats(self) -> dict:
        """Return search performance statistics."""
        stats = self._search_stats.copy()
        if stats['total_calls'] > 0:
            stats['search_rate'] = stats['successful_searches'] / stats['total_calls']
            stats['override_rate'] = (
                stats['search_overrides'] / stats['successful_searches']
                if stats['successful_searches'] > 0 else 0
            )
        return stats


class ValueRerankPlayer(PPOPlayer):
    """
    Heuristic reranking without search server.
    Uses type effectiveness and matchup analysis to adjust policy.
    """

    def __init__(self, model_path: str, min_tera_turn: int = 2,
                 rerank_strength: float = 0.3, verbose: bool = False, **kwargs):
        self.min_tera_turn = min_tera_turn
        self.rerank_strength = rerank_strength
        self.verbose = verbose
        super().__init__(model_path=model_path, **kwargs)

    def _compute_action_adjustments(self, battle: AbstractBattle) -> np.ndarray:
        adj = np.zeros(13, dtype=np.float32)
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        moves = battle.available_moves or []
        switches = battle.available_switches or []

        if not my or not opp:
            return adj

        best_eff = 0.0
        for i, move in enumerate(moves):
            if i >= 4:
                break
            try:
                eff = opp.damage_multiplier(move)
            except Exception:
                eff = 1.0

            if eff >= 2.0:
                adj[i] += 0.3; adj[i + 4] += 0.3
            elif eff >= 1.0:
                adj[i] += 0.1
            elif 0 < eff < 1.0:
                adj[i] -= 0.2; adj[i + 4] -= 0.2
            elif eff == 0:
                adj[i] -= 1.0; adj[i + 4] -= 1.0

            best_eff = max(best_eff, eff)
            if move.type in my.types:
                adj[i] += 0.05
            if my.current_hp_fraction < 0.35 and hasattr(move, 'id') and move.id.lower() in {
                'swordsdance', 'nastyplot', 'calmmind', 'dragondance',
                'quiverdance', 'shellsmash', 'bulkup', 'irondefense',
            }:
                adj[i] -= 0.4

        for i, switch in enumerate(switches):
            if i >= 5:
                break
            s_idx = 8 + i
            try:
                from envs.rl_player_6v6_v18 import get_type_multiplier
                off = get_type_multiplier(switch.types, opp.types)
                def_ = get_type_multiplier(opp.types, switch.types)
            except ImportError:
                off = def_ = 1.0

            if def_ < 1.0 and off > 1.0: adj[s_idx] += 0.3
            elif def_ < 1.0: adj[s_idx] += 0.15
            elif def_ > 1.5: adj[s_idx] -= 0.3
            if best_eff >= 2.0: adj[s_idx] -= 0.15

            try:
                from poke_env.battle.side_condition import SideCondition
            except ImportError:
                from poke_env.environment.side_condition import SideCondition
            if SideCondition.STEALTH_ROCK in battle.side_conditions:
                adj[s_idx] -= 0.1

        return adj

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        obs = self._rl_env.embed_battle(battle)
        action_mask = self._rl_env.get_action_mask(battle)

        if self.min_tera_turn > 0 and battle.turn < self.min_tera_turn:
            tera_backup = action_mask[4:8].copy()
            action_mask[4:8] = 0.0
            if action_mask.sum() == 0:
                action_mask[4:8] = tera_backup

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            dist = self.policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits.clone()
            logits[~mask_tensor] = float('-inf')
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        adjustments = self._compute_action_adjustments(battle) * self.rerank_strength
        combined = probs + adjustments
        combined = np.where(action_mask > 0, combined, -np.inf)
        action = int(combined.argmax())

        if self.verbose and action != int(probs.argmax()):
            actions_map = dict(get_showdown_actions(battle, action_mask))
            old = actions_map.get(int(probs.argmax()), "?")
            new = actions_map.get(action, "?")
            print(f"  Turn {battle.turn}: Rerank changed {old} -> {new}")

        return self._rl_env.action_to_order(action, battle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_pokemon_6v6_v18.2")
    parser.add_argument("--mode", choices=["search", "rerank"], default="search")
    parser.add_argument("--server", default="http://localhost:9001")
    parser.add_argument("--search-weight", type=float, default=0.3,
                        help="Weight for value-search probs in mix (0=pure policy, 1=pure search)")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--n-battles", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from poke_env.player import SimpleHeuristicsPlayer

    if args.mode == "search":
        try:
            r = requests.get(f"{args.server}/health", timeout=1)
            health = r.json()
            print(f"Search server: {health}")
        except Exception as e:
            print(f"WARNING: Search server not available at {args.server}: {e}")
            print("Falling back to rerank mode")
            args.mode = "rerank"

    if args.mode == "search":
        player = SearchPlayer(
            model_path=args.model,
            search_server_url=args.server,
            search_weight=args.search_weight,
            n_opponent_samples=args.n_samples,
            deterministic=True,
            verbose=args.verbose,
            battle_format="gen9randombattle",
        )
    else:
        player = ValueRerankPlayer(
            model_path=args.model,
            deterministic=True,
            verbose=args.verbose,
            battle_format="gen9randombattle",
        )

    opponent = SimpleHeuristicsPlayer(battle_format="gen9randombattle")

    import asyncio

    async def main():
        mode_str = "search" if args.mode == "search" else "rerank"
        print(f"\nTesting {mode_str} player ({args.n_battles} battles vs SimpleHeuristics)...")
        if args.mode == "search":
            print(f"  search_weight={args.search_weight}, n_samples={args.n_samples}")
        await player.battle_against(opponent, n_battles=args.n_battles)

        wins = sum(1 for b in player.battles.values() if b.won)
        total = len(player.battles)
        print(f"\nResult: {wins}/{total} ({wins/total*100:.1f}%)")

        if hasattr(player, 'get_search_stats'):
            stats = player.get_search_stats()
            print(f"\nSearch stats:")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")

    asyncio.run(main())