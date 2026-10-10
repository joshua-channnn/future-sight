"""Plays ladder games on the public Showdown server.

    python3 -m players.ladder_bot                       # the search bot (search/policy.py), 30 games
    python3 -m players.ladder_bot --games 10 --model models/ppo_pokemon_6v6_v18.3_best

--model: "search" (default), an SB3 zip (v17-v19) or a v20/v21 .pt. The search bot plays with the
settings that measured best in step 6 (agent_docs/step6.md: the v21 network's prior and set model,
item inference) at 2.5x Foul Play's budget (2 x 250 ms per decision; the ladder clock allows far more).
Its set prior comes from the Feb 2026 Showdown fork's generator; the public server's sets may have
changed since (unmatched sets fall back to the closest ones, unknown species to stand-in sets).
"""
import argparse
import asyncio
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration
from poke_env.battle.abstract_battle import AbstractBattle


class _Record:
    """Win/loss record, ignoring games that ended by turn 2 (forfeits)."""

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


def make_player(model="search", hold_requests=False, **kwargs):
    """A ladder player for `model` ("search", an SB3 zip or a v20/v21 .pt). hold_requests: only
    against this repo's Showdown fork (it sends requests before the turn's log), never the public
    server (agent_docs/poke_env_gotchas.md, item 9)."""
    if model == "search":
        import torch
        torch.set_num_threads(1)
        from search.net import NetModel
        from search.policy import SearchPlayer, SearchPolicy

        class LadderSearchPlayer(_Record, SearchPlayer):
            pass

        policy = SearchPolicy(samples=2, ms=250, early_samples=4, early_ms=125, prior_weight=0.15,
                              set_gamma=0.5, infer_items=True, net=NetModel())
        return LadderSearchPlayer(policy=policy, hold_requests=hold_requests, **kwargs)
    if model.endswith(".pt"):
        from players.v20_player import V20Player

        class LadderV20Player(_Record, V20Player):
            pass

        return LadderV20Player(model_path=model, deterministic=True, hold_requests=hold_requests, **kwargs)
    from players.ppo_player import PPOPlayer

    class LadderPPOPlayer(_Record, PPOPlayer):
        pass

    return LadderPPOPlayer(model_path=model, data_dir="data", deterministic=True, **kwargs)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="search", help='"search", an SB3 zip or a v20/v21 .pt')
    ap.add_argument("--games", type=int, default=30)
    args = ap.parse_args()
    USERNAME = "vibecoded"
    PASSWORD = "opus45"
    N_GAMES = args.games

    try:
        import certifi, os
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass

    bot = make_player(
        args.model,
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