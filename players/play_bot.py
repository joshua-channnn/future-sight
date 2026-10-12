"""A bot on the local Showdown server that accepts any challenge (to play against it yourself).

    python3 -m players.play_bot                                   # the search bot (steps 6-7)
    python3 -m players.play_bot --explain                         # ... telling you its reasoning
    python3 -m players.play_bot --model models/ppo_pokemon_6v6_v18.2

--model: "search" (default; the same settings as players/ladder_bot.py), an SB3 zip or a v20/v21 .pt.
--explain (search bot): after each turn, it posts in the Lobby what it chose, its options' shares
of the search with the win chance it gave each, what it expected you to pick, and the item it
guessed for your active Pokemon (search.policy.describe_decision). Not in the battle's chat: Showdown
only lets registered users talk in battles, and nobody is registered on a local server.
"""
import argparse
import asyncio

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from players.ladder_bot import make_player


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="search", help='"search", an SB3 zip or a v20/v21 .pt')
    ap.add_argument("--username", default="MyPPOBot")
    ap.add_argument("--explain", action="store_true", help="search bot: post its reasoning after each turn")
    args = ap.parse_args()
    if args.explain and args.model != "search":
        ap.error("--explain needs the search bot")
    bot = make_player(
        args.model,
        # this repo's Showdown fork sends each request before the turn's log (SB3 players ignore it)
        hold_requests=True,
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration(args.username, None),
        server_configuration=LocalhostServerConfiguration,
        **({"explain": True, "explain_room": "lobby"} if args.explain else {}),
    )
    # Accept any challenges indefinitely
    await bot.accept_challenges(None, n_challenges=999999)


if __name__ == "__main__":
    asyncio.run(main())
