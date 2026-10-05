"""Checkpoint architectures: the featurizer, network and observation layout behind each "arch".

Torch checkpoints store {"arch", "net_config", "layout", "model", ...}. The trainer, the policy
loader (players/v20_player.py) and the evaluation scripts build everything through `get(arch)`, so
v20 checkpoints keep loading next to v21 ones.
"""


class _V20:
    name = "v20"
    has_labels = False

    @staticmethod
    def layout(cfg=None):
        from envs.rl_player_v20 import LAYOUT
        return LAYOUT

    @staticmethod
    def featurizer(layout):
        from envs.rl_player_v20 import Featurizer
        return Featurizer()

    @staticmethod
    def net(net_config, layout):
        from networks.v20_policy import V20Net
        return V20Net(net_config)

    @staticmethod
    def default_net_config():
        from networks.v20_policy import DEFAULT_CONFIG
        return dict(DEFAULT_CONFIG)


class _V21:
    name = "v21"
    has_labels = True

    @staticmethod
    def layout(cfg=None):
        from envs.rl_player_v21 import DEFAULT_HISTORY, make_layout
        return make_layout((cfg or {}).get("history_len") or DEFAULT_HISTORY)

    @staticmethod
    def featurizer(layout):
        from envs.rl_player_v21 import Featurizer
        return Featurizer(history_len=layout["hist_len"])

    @staticmethod
    def net(net_config, layout):
        from networks.v21_policy import V21Net
        return V21Net(net_config, layout)

    @staticmethod
    def default_net_config():
        from networks.v21_policy import DEFAULT_CONFIG
        return dict(DEFAULT_CONFIG)


ARCHS = {"v20": _V20, "v21": _V21}


def get(arch):
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r} (known: {sorted(ARCHS)})")
    return ARCHS[arch]


def expected_layout(arch, saved_layout):
    """The layout this code produces for a checkpoint of `arch` (v21: with its history length)."""
    spec = get(arch)
    if arch == "v21":
        return spec.layout({"history_len": (saved_layout or {}).get("hist_len")})
    return spec.layout()
