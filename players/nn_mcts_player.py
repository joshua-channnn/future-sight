import time
import numpy as np
import torch
from typing import Dict, Optional

from poke_env.player import BattleOrder
from poke_env.battle.abstract_battle import AbstractBattle

from players.ppo_player import PPOPlayer

from players.engine_search_player import (
    battle_to_engine_state,
    _map_mcts_to_actions,
    _norm,
)

from poke_engine import (
    State,
    mcts as engine_mcts,
)


def _action_name(idx: int, battle: AbstractBattle) -> str:
    moves = battle.available_moves or []
    switches = battle.available_switches or []
    if idx < 4 and idx < len(moves):
        return moves[idx].id
    if idx < 8 and (idx - 4) < len(moves):
        return f"{moves[idx-4].id}+tera"
    if idx >= 8 and (idx - 8) < len(switches):
        return f"switch {switches[idx-8].species}"
    return f"action_{idx}"


class NNMCTSPlayer(PPOPlayer):
    """
    Conservative search player: engine MCTS scores + policy prior,
    override only when confident.
    """

    def __init__(
        self,
        model_path: str,
        search_time_ms: int = 50,
        override_threshold: float = 0.15,
        min_tera_turn: int = 2,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            search_time_ms: Time for poke-engine MCTS per decision
            override_threshold: Only override policy when engine's best action
                has Q-value at least this much higher than policy's choice.
                0.0 = always use engine best (aggressive)
                0.15 = moderate (override ~20-30% of decisions)  
                0.3 = conservative (override ~10% of decisions)
        """
        self.search_time_ms = search_time_ms
        self.override_threshold = override_threshold
        self.min_tera_turn = min_tera_turn
        self._verbose = verbose

        self._stats = {
            'total_decisions': 0,
            'search_successes': 0,
            'search_failures': 0,
            'overrides': 0,
            'policy_kept': 0,
            'avg_search_ms': 0,
            'avg_engine_iters': 0,
            'override_margin_sum': 0,  # Track how big the margins are when we override
        }

        super().__init__(model_path=model_path, **kwargs)

    def _get_policy_probs(self, battle, action_mask):
        obs = self._rl_env.embed_battle(battle)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.clone()
            logits[~mask_t] = float('-inf')
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        return probs

    def _run_engine_search(self, battle, action_mask):
        """
        Run poke-engine MCTS and return normalized Q-values per action.
        Returns {action_idx: q_value} where q_value is in [0, 1].
        """
        state, reason = battle_to_engine_state(battle)
        if state is None:
            return None, 0

        try:
            state_str = state.to_string()
            state = State.from_string(state_str)
        except BaseException:
            return None, 0

        try:
            t0 = time.time()
            result = engine_mcts(state, self.search_time_ms)
            elapsed = (time.time() - t0) * 1000

            scores = _map_mcts_to_actions(result, battle)
            if not scores:
                return None, 0

            filtered = {k: v for k, v in scores.items() if action_mask[k] > 0}
            if not filtered:
                return None, 0

            return filtered, result.iteration_count

        except BaseException:
            return None, 0

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        action_mask = self._rl_env.get_action_mask(battle)

        if self.min_tera_turn > 0 and battle.turn < self.min_tera_turn:
            tera_backup = action_mask[4:8].copy()
            action_mask[4:8] = 0.0
            if action_mask.sum() == 0:
                action_mask[4:8] = tera_backup

        self._stats['total_decisions'] += 1

        policy_probs = self._get_policy_probs(battle, action_mask)
        policy_choice = int(policy_probs.argmax())

        engine_scores, iterations = self._run_engine_search(battle, action_mask)

        if engine_scores is None:
            self._stats['search_failures'] += 1
            masked = np.where(action_mask > 0, policy_probs, -np.inf)
            return self._rl_env.action_to_order(int(masked.argmax()), battle)

        self._stats['search_successes'] += 1
        alpha = 0.1
        self._stats['avg_engine_iters'] = (
            (1 - alpha) * self._stats['avg_engine_iters'] + alpha * iterations)

        vals = list(engine_scores.values())
        v_min, v_max = min(vals), max(vals)
        v_range = v_max - v_min
        if v_range > 0.001:
            engine_q = {k: (v - v_min) / v_range for k, v in engine_scores.items()}
        else:
            engine_q = {k: 0.5 for k in engine_scores}

        engine_best = max(engine_q, key=engine_q.get)
        engine_best_q = engine_q[engine_best]

        policy_q = engine_q.get(policy_choice, None)

        if policy_q is not None:
            margin = engine_best_q - policy_q
            should_override = (
                engine_best != policy_choice
                and margin > self.override_threshold
            )
        else:
            # Policy's choice wasn't in engine results — use engine best
            should_override = True
            margin = engine_best_q

        if should_override:
            action = engine_best
            self._stats['overrides'] += 1
            self._stats['override_margin_sum'] += margin
            if self._verbose:
                print(f"  Turn {battle.turn}: OVERRIDE "
                      f"{_action_name(policy_choice, battle)}"
                      f"->{_action_name(action, battle)} "
                      f"(policy_p={policy_probs[policy_choice]:.3f}, "
                      f"margin={margin:.3f})")
        else:
            action = policy_choice
            self._stats['policy_kept'] += 1

        if self._verbose:
            moves = battle.available_moves or []
            switches = battle.available_switches or []
            print(f"  T{battle.turn}: ", end="")
            items = sorted(engine_q.items(), key=lambda x: -x[1])
            for idx, q in items[:5]:
                name = _action_name(idx, battle)
                p = policy_probs[idx]
                raw = engine_scores.get(idx, 0)
                marker = " <<<" if idx == action else ""
                print(f"{name}(Q={q:.2f},p={p:.2f},raw={raw:.3f}){marker}  ", end="")
            print()

        return self._rl_env.action_to_order(action, battle)

    def get_search_stats(self) -> dict:
        s = {**self._stats}
        total = s['total_decisions']
        success = s['search_successes']
        overrides = s['overrides']
        if total > 0:
            s['search_rate'] = success / total
        if success > 0:
            s['override_rate'] = overrides / success
        if overrides > 0:
            s['avg_override_margin'] = s['override_margin_sum'] / overrides
        return s


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Conservative NN+Engine Search v4")
    parser.add_argument("--model", default="models/ppo_pokemon_6v6_v18.2")
    parser.add_argument("--search-time-ms", type=int, default=50)
    parser.add_argument("--override-threshold", type=float, default=0.15,
                        help="Min Q-margin to override policy (0=aggressive, 0.3=conservative)")
    parser.add_argument("--n-battles", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare-pure-ppo", action="store_true")
    parser.add_argument("--sweep-thresholds", action="store_true",
                        help="Test multiple threshold values")
    args = parser.parse_args()

    from poke_env.player import SimpleHeuristicsPlayer

    async def run_test(threshold, n_battles, label=""):
        player = NNMCTSPlayer(
            model_path=args.model,
            search_time_ms=args.search_time_ms,
            override_threshold=threshold,
            deterministic=True,
            verbose=args.verbose,
            battle_format="gen9randombattle",
        )
        opp = SimpleHeuristicsPlayer(battle_format="gen9randombattle")
        await player.battle_against(opp, n_battles=n_battles)
        wins = sum(1 for b in player.battles.values() if b.won)
        total = len(player.battles)
        stats = player.get_search_stats()
        return wins, total, stats

    async def main():
        if args.sweep_thresholds:
            print("=" * 70)
            print("Threshold Sweep")
            print("=" * 70)

            # First run pure PPO baseline
            ppo = PPOPlayer(model_path=args.model, deterministic=True,
                            battle_format="gen9randombattle")
            opp_base = SimpleHeuristicsPlayer(battle_format="gen9randombattle")
            await ppo.battle_against(opp_base, n_battles=args.n_battles)
            pw = sum(1 for b in ppo.battles.values() if b.won)
            pt = len(ppo.battles)
            print(f"  Pure PPO baseline: {pw}/{pt} ({pw/pt*100:.1f}%)")
            print()

            for threshold in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
                wins, total, stats = await run_test(threshold, args.n_battles)
                override_rate = stats.get('override_rate', 0)
                avg_margin = stats.get('avg_override_margin', 0)
                print(f"  threshold={threshold:.2f}: {wins}/{total} ({wins/total*100:.1f}%) "
                      f"override={override_rate:.1%} avg_margin={avg_margin:.3f}")

        else:
            print("=" * 60)
            print(f"Conservative NN+Engine Search v4")
            print(f"  search_time={args.search_time_ms}ms, "
                  f"threshold={args.override_threshold}")
            print("=" * 60)

            wins, total, stats = await run_test(
                args.override_threshold, args.n_battles)
            print(f"\nResult: {wins}/{total} ({wins/total*100:.1f}%)")
            print("\nStats:")
            for k, v in sorted(stats.items()):
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

            if args.compare_pure_ppo:
                print(f"\n{'='*60}")
                ppo = PPOPlayer(model_path=args.model, deterministic=True,
                                battle_format="gen9randombattle")
                opp2 = SimpleHeuristicsPlayer(battle_format="gen9randombattle")
                print(f"Running {args.n_battles} battles: Pure PPO...")
                await ppo.battle_against(opp2, n_battles=args.n_battles)
                pw = sum(1 for b in ppo.battles.values() if b.won)
                pt = len(ppo.battles)
                print(f"Pure PPO: {pw}/{pt} ({pw/pt*100:.1f}%)")
                print(f"\n  Search: {wins}/{total} ({wins/total*100:.1f}%)")
                print(f"  PPO:    {pw}/{pt} ({pw/pt*100:.1f}%)")
                print(f"  Delta:  {(wins/total - pw/pt)*100:+.1f}%")

    asyncio.run(main())