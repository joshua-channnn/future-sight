"""Search at decision time on poke-engine (agent_docs/step6_brief.md).

sampling.py: the set prior (data/gen9_randbats_set_counts.json) and opponent team sampling.
state.py: poke-env battle -> poke-engine State.
policy.py: SearchPolicy, MCTS over sampled opponent teams, with V20Policy's interface.
"""
