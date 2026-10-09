"""Phase 2 (b) of agent_docs/step6_brief.md, before any games: does reweighting the sampler's
candidate sets with v21's aux heads (search/net.py, NetModel.set_weight_fn) put more probability on
the opponent's true hidden set than the generator prior alone?

Plays live two-sided games (side 1 random, side 2 SimpleHeuristics / random) and at each decision,
for each revealed opponent Pokemon with something hidden, compares the candidates' distribution
(prior, and reweighted for each gamma) on the truth (the other side's request): mean log-probability
and top-1 accuracy of its item, ability and tera type where not shown yet, of each unrevealed move
(probability that the set contains it), and of the exact set.

    python3 -m tools.check_set_model --games 40 --port 8123 --gammas 0.5,1
"""
import argparse
import math
import random
import uuid
from collections import defaultdict

import numpy as np
from poke_env.data import to_id_str
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

from envs.rl_player_v20 import RLEnvV20, action_mask, action_to_order
from envs.rl_player_v21 import illusion_game
from eval.v20_eval import server_config
from search.net import NetModel
from search.sampling import get_prior, observations
from search.state import tracker


def truth_of(other, key):
    req = other.last_request or {}
    for p in (req.get("side") or {}).get("pokemon", []):
        if p["ident"] == key:
            return {"item": to_id_str(p.get("item") or ""), "ability": to_id_str(p.get("baseAbility") or ""),
                    "tera": to_id_str(p.get("teraType") or ""), "moves": [to_id_str(m) for m in p.get("moves", [])]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--gammas", default="0.5,1")
    args = ap.parse_args()
    gammas = [float(g) for g in args.gammas.split(",")]
    prior = get_prior()
    net = NetModel()
    rng = random.Random(0)
    uid = uuid.uuid4().hex[:4]
    env = RLEnvV20(server_configuration=server_config(args.port),
                   account_configuration1=AccountConfiguration.generate(f"SmA{uid}", rand=True),
                   account_configuration2=AccountConfiguration.generate(f"SmB{uid}", rand=True))
    sh = SimpleHeuristicsPlayer(battle_format="gen9randombattle", start_listening=False)
    # stats[model][attr] = [sum log p, n, n top-1 right]
    stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0]))
    for g in range(args.games):
        env.reset()
        if illusion_game(env.battle1, env.battle2):
            pass
        while not env.battle1.finished:
            acts = {}
            for agent, b, other, tm, use_sh in ((env.agent1.username, env.battle1, env.battle2, env.agent1_to_move, False),
                                                (env.agent2.username, env.battle2, env.battle1, env.agent2_to_move, g % 2 == 1)):
                if not tm or b.finished:
                    acts[agent] = 0
                    continue
                if not illusion_game(env.battle1, env.battle2):
                    score(b, other, prior, net, gammas, stats)
                acts[agent] = sh.choose_move(b) if use_sh else action_to_order(
                    int(rng.choice(np.flatnonzero(action_mask(b)))), b)
            env.step(acts)
    env.close()
    print(f"{args.games} games: mean log-probability of the truth (top-1 accuracy), prior vs reweighted")
    attrs = ["item", "ability", "tera", "moves", "set"]
    print(f"{'model':>24} " + " ".join(f"{a:>22}" for a in attrs))
    for model in ["prior"] + [f"gamma {g:g}" for g in gammas] + ["inferred items"] + \
            [f"inferred + gamma {g:g}" for g in gammas]:
        row = []
        for a in attrs:
            s, n, right = stats[model][a]
            row.append(f"{s / max(n, 1):8.3f} ({right / max(n, 1):.3f}) n={n}")
        print(f"{model:>24} " + " ".join(f"{x:>22}" for x in row))


def score(b, other, prior, net, gammas, stats):
    tr = tracker(b)
    obs = observations(b, tr)
    fns = {f"gamma {g:g}": net.set_weight_fn(b, gamma=g, tera=True) for g in gammas}
    for j, (o, (key, mon)) in enumerate(zip(obs, list(b.opponent_team.items())[:6])):
        t = truth_of(other, key)
        if t is None or len(t["moves"]) != 4 or "transform" in t["moves"]:
            continue
        keys = prior.keys_for(o["species"])
        if len(keys) != 1:
            continue
        cands, w, relaxed = prior.candidates(keys[0], o["level"], o["moves"], o["item"], o["ability"], o["tera"], o["lead"])
        if relaxed or len(cands) < 2:
            continue
        models = {"prior": np.array(w, dtype=np.float64)}
        for name, fn in fns.items():
            models[name] = np.array(fn(j, cands, w), dtype=np.float64)
        # rule-based: items ruled out by what it did (the same candidate list, ruled-out items at 0)
        no_items = (tr.mons.get(key).no_items if tr.mons.get(key) else set())
        wi = np.array([wt if s.item not in no_items else 0.0 for s, wt in zip(cands, w)], dtype=np.float64)
        if wi.sum() > 0:
            models["inferred items"] = wi
            for name, fn in fns.items():
                models[f"inferred + {name}"] = np.array(fn(j, cands, list(wi)), dtype=np.float64)
        else:
            models["inferred items"] = models["prior"]
            for name in fns:
                models[f"inferred + {name}"] = models[name]
        hidden_moves = [m for m in t["moves"] if m not in mon.moves]
        true_set = (t["item"], t["ability"], t["tera"], tuple(sorted(t["moves"])))
        for name, ww in models.items():
            p = ww / ww.sum()
            st = stats[name]
            checks = []
            if o["item"] is None and mon.item == "unknown_item":
                checks.append(("item", lambda s: s.item == t["item"], lambda s: s.item))
            if not mon.ability:
                checks.append(("ability", lambda s: s.ability == t["ability"], lambda s: s.ability))
            if o["tera"] is None:
                checks.append(("tera", lambda s: s.tera == t["tera"], lambda s: s.tera))
            for attr, ok, val in checks:
                mass = sum(pi for pi, s in zip(p, cands) if ok(s))
                st[attr][0] += math.log(max(mass, 1e-6))
                st[attr][1] += 1
                agg = defaultdict(float)
                for pi, s in zip(p, cands):
                    agg[val(s)] += pi
                st[attr][2] += max(agg, key=agg.get) == val(next(s for s in cands if ok(s))) if mass > 0 else 0
            for m in hidden_moves:
                mass = sum(pi for pi, s in zip(p, cands) if m in s.moves)
                st["moves"][0] += math.log(max(mass, 1e-6))
                st["moves"][1] += 1
                st["moves"][2] += mass > 0.5
            mass = sum(pi for pi, s in zip(p, cands)
                       if (s.item, s.ability, s.tera, s.moves) == true_set)
            st["set"][0] += math.log(max(mass, 1e-6))
            st["set"][1] += 1
            st["set"][2] += int(np.argmax(p)) == next((i for i, s in enumerate(cands)
                                                         if (s.item, s.ability, s.tera, s.moves) == true_set), -1)


if __name__ == "__main__":
    main()
