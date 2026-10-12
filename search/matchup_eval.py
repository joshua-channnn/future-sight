"""Matchup weights for the learned leaf evaluation (agent_docs/step7.md section 5; step 8).

lin1_86 (search/leaf_eval.py) values every Pokemon's HP, status and boosts alike. Here a network
looks at the root of each search once and sets weights for per-Pokemon leaf features, which the
engine keeps for every leaf of that search (engine::learned, `root_weights` of pe.mcts_with):

    v(leaf) = lin(x(leaf)) + sum over side one's Pokemon p of u_p . phi_p(leaf)
              - the same over side two's + W[side one's active][side two's active] (both alive)

and the leaf score is sigmoid(scale * (v(leaf) - v(root))) as before (mode 2). phi_p
(pe.matchup_layout()'s names: alive, HP, status, item, tera, active, the active's boosts) changes
within a search, so u and W say how much each change is worth in this matchup. The network's inputs
come from the engine (pe.matchup_inputs: per Pokemon its species, ability, item, types, moves and
state; the field; the 110 features of lin), so training and search read the same thing. It is
antisymmetric by construction: side two's u come from the same network with the sides exchanged,
W = m(side one's p, side two's q) - m(q, p), so v(mirror) = -v exactly.

Trained by training/train_matchup_eval.py (a .pt with the network, its config and the linear part;
the linear part also goes to an .npz that search/leaf_eval.py registers with the engine). Result
(agent_docs/step8.md): better prediction of results offline, but no stronger search (the full term
loses 7-10 points, scaled by 0.3 it is level), so the deployed bot doesn't use it.
"""
import numpy as np
import poke_engine as pe
import torch
import torch.nn as nn

(N_POKE_INTS, N_POKE_FLOATS, N_FIELD, M_LEAF, N_X, N_ROOT_WEIGHTS), LEAF_NAMES = pe.matchup_layout() \
    if hasattr(pe, "matchup_layout") else ((11, 25, 17, 16, 110, 228), [])
N_INTS = 12 * N_POKE_INTS + 2
N_FLOATS = 12 * N_POKE_FLOATS + N_FIELD + N_X
N_SIDE = 52   # lin's features per side (engine::learned::SIDE_NAMES); then 6 cross features
# embedding table sizes (engine enums: PokemonName ~1455, Abilities ~322, Items ~239, Choices ~885)
VOCAB = {"species": 1600, "ability": 400, "item": 256, "type": 32, "status": 8, "move": 1000}
# engine::learned::boost_multiplier, indexed by boost + 6
BOOST_TABLE = torch.tensor([-3.3, -3.15, -3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 3.15, 3.3])
STATUS = {"burn": 1, "sleep": 2, "freeze": 3, "paralyze": 4, "poison": 5, "toxic": 6}   # PokemonStatus


def has_engine_support():
    return hasattr(pe, "matchup_inputs")


def split_inputs(ints, floats):
    """(B, N_INTS), (B, N_FLOATS) -> per-Pokemon ints (B, 12, 11), floats (B, 12, 25), active slots
    (B, 2), field (B, 17), lin's features x (B, 110)."""
    b = ints.shape[0]
    pk = ints[:, :12 * N_POKE_INTS].reshape(b, 12, N_POKE_INTS)
    act = ints[:, 12 * N_POKE_INTS:]
    pf = floats[:, :12 * N_POKE_FLOATS].reshape(b, 12, N_POKE_FLOATS)
    o = 12 * N_POKE_FLOATS
    return pk, pf, act, floats[:, o:o + N_FIELD], floats[:, o + N_FIELD:]


def leaf_phi(pk, pf, act):
    """engine::learned::leaf_features from the inputs: phi (B, 12, M_LEAF) and the one-hot of the
    active pair when both are alive (B, 36). Float tensors in, float32 out."""
    pf = pf.float()
    alive, hp, active = pf[..., 0], pf[..., 1], pf[..., 9]
    st = pk[..., 6].long()
    phi = torch.zeros(pf.shape[:-1] + (M_LEAF,), device=pf.device)
    phi[..., 0] = alive
    phi[..., 1] = hp * alive
    phi[..., 2] = hp * hp * alive
    phi[..., 3] = (st == STATUS["burn"]) * alive
    phi[..., 4] = (st == STATUS["paralyze"]) * alive
    phi[..., 5] = (st == STATUS["sleep"]) * alive
    phi[..., 6] = (st == STATUS["freeze"]) * alive
    phi[..., 7] = ((st == STATUS["poison"]) | (st == STATUS["toxic"])) * alive
    phi[..., 8] = (pk[..., 2] != 0) * alive
    phi[..., 9] = pf[..., 10] * alive
    on = active * alive
    phi[..., 10] = on
    boosts = torch.round(pf[..., 11:16] * 6).long().clamp(-6, 6) + 6
    phi[..., 11:16] = BOOST_TABLE.to(pf.device)[boosts] * on[..., None]
    a1, a2 = act[:, 0].long(), act[:, 1].long()
    rows = torch.arange(pk.shape[0], device=pf.device)
    both = alive[rows, a1] * alive[rows, 6 + a2]
    pair = torch.zeros(pk.shape[0], 36, device=pf.device)
    pair[rows, a1 * 6 + a2] = both
    return phi, pair


N_PAIR = 9   # engine::learned::PAIR_NAMES
PAIR_SWAP = [1, 0, 3, 2, 4, 6, 5, 8, 7]   # a pair's features seen from the other side ("faster" flips sign)
N_AGG = N_PAIR + 4


def pair_views(pairs):
    """pe.pair_features (B, 36 * N_PAIR), side one's slot i vs side two's slot j -> (B, 6, 6, N_PAIR)
    from side one's view (own i, opponent j) and from side two's (own j, opponent i)."""
    p1 = pairs.float().reshape(-1, 6, 6, N_PAIR)
    sign = torch.ones(N_PAIR, device=p1.device)
    sign[4] = -1
    return p1, p1.transpose(1, 2)[..., PAIR_SWAP] * sign


def pair_aggregates(p, opp_alive):
    """Per own Pokemon over the living opponents: the mean of each pair feature and the max of
    dmg, dmg_back, eff, eff_back (B, 6, N_AGG)."""
    n = opp_alive.sum(1).clamp(min=1)[:, None, None]
    mean = p.sum(2) / n
    alive = opp_alive[:, None, :, None] > 0
    big = p[..., [0, 1, 5, 6]].masked_fill(~alive, -3.0).max(2).values
    big = torch.where(opp_alive.sum(1)[:, None, None] > 0, big, torch.zeros_like(big))
    return torch.cat([mean, big], -1)


class MatchupNet(nn.Module):
    """Root matchup inputs -> (c, u1, u2, W): a correction c (B,) to lin's value of the root
    (training only: the pairs' loss is about what changes between the root and a later state, and
    c absorbs what the root already says), per-Pokemon weights u1 / u2 for
    side one's / side two's 6 Pokemon (B, 6, M_LEAF) and W (B, 6, 6). Plus the linear part of the
    evaluation (lin, antisymmetric: side one's 52 weights apply to x_s1 - x_s2, 6 cross weights).

    ids: which ids get embeddings ("all": species, ability, item, moves; "items": ability and item;
    "none"); move_props: moves described by their type, category, power, accuracy and priority
    (pe.move_table) instead of an id embedding; pairs: pe.pair_features as inputs (per-Pokemon
    aggregates over the living opponents, and each pair's features in W's head). Embeddings of ids
    let the network recognize games and memorize their results (training/train_matchup_eval.py)."""

    def __init__(self, d=64, layers=2, heads=4, dropout=0.0, ids="all", move_props=False, pairs=False,
                 x_mu=None, x_sigma=None, lin_side=None, lin_cross=None):
        super().__init__()
        self.config = {"d": d, "layers": layers, "heads": heads, "dropout": dropout, "ids": ids,
                       "move_props": move_props, "pairs": pairs}
        self.ids, self.move_props, self.pairs = ids, move_props, pairs
        self.ty = nn.Embedding(VOCAB["type"], 8)
        self.st = nn.Embedding(VOCAB["status"], 4)
        embs = [self.ty, self.st]
        n_in = 16 + 4 + 32 + N_POKE_FLOATS + (N_AGG if pairs else 0)
        if ids == "all":
            self.sp = nn.Embedding(VOCAB["species"], 32)
            self.mv = nn.Embedding(VOCAB["move"], 24)
            embs += [self.sp, self.mv]
            n_in += 32
        if ids in ("all", "items"):
            k = 16 if ids == "all" else 8
            self.ab = nn.Embedding(VOCAB["ability"], k)
            self.it = nn.Embedding(VOCAB["item"], k)
            embs += [self.ab, self.it]
            n_in += 2 * k
        for e in embs:
            nn.init.normal_(e.weight, std=0.1)
        n_move = (24 if ids == "all" else 0) + 1
        if move_props:
            ti, tf = pe.move_table()
            self.register_buffer("mv_int", torch.tensor(ti, dtype=torch.int64).reshape(-1, 3))
            self.register_buffer("mv_flt", torch.tensor(tf, dtype=torch.float32).reshape(-1, 2))
            self.cat = nn.Embedding(4, 4)
            nn.init.normal_(self.cat.weight, std=0.1)
            n_move += 8 + 4 + 3
        self.mv_mlp = nn.Sequential(nn.Linear(n_move, 32), nn.ReLU(), nn.Linear(32, 32))
        self.poke = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.side = nn.Parameter(torch.zeros(2, d))   # own, opponent
        nn.init.normal_(self.side, std=0.1)
        self.glob = nn.Sequential(nn.Linear(N_FIELD + N_X, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.u = nn.Linear(d, M_LEAF)
        self.pair = nn.Sequential(nn.Linear(2 * d + (N_PAIR if pairs else 0), d), nn.ReLU(), nn.Linear(d, 1))
        self.val = nn.Linear(d, 1)
        for lin in (self.u, self.pair[-1], self.val):   # start as the plain linear evaluation
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("x_mu", torch.zeros(N_X) if x_mu is None else torch.as_tensor(x_mu, dtype=torch.float32))
        self.register_buffer("x_sigma", torch.ones(N_X) if x_sigma is None else torch.as_tensor(x_sigma, dtype=torch.float32))
        self.lin_side = nn.Parameter(torch.zeros(N_SIDE) if lin_side is None else torch.as_tensor(lin_side, dtype=torch.float32))
        self.lin_cross = nn.Parameter(torch.zeros(N_X - 2 * N_SIDE) if lin_cross is None
                                      else torch.as_tensor(lin_cross, dtype=torch.float32))

    # ---- the linear part ------------------------------------------------------------------------
    @staticmethod
    def swap_x(x):
        return torch.cat([x[:, N_SIDE:2 * N_SIDE], x[:, :N_SIDE], -x[:, 2 * N_SIDE:]], 1)

    def lin(self, x):
        """lin1_86's v(x): the logit of side one winning from the 110 features."""
        x = x.float()
        return (x[:, :N_SIDE] - x[:, N_SIDE:2 * N_SIDE]) @ self.lin_side + x[:, 2 * N_SIDE:] @ self.lin_cross

    def lin_export(self):
        """w1 (1, 110) for pe.load_evaluator (hidden 0), whose v is (w1_s1 - w1_s2) . (x_s1 - x_s2)
        + 2 w1_cross . x_cross: with w1_s2 = 0 and the cross weights halved, exactly lin()."""
        w = np.zeros((1, N_X), np.float32)
        w[0, :N_SIDE] = self.lin_side.detach().cpu().numpy()
        w[0, 2 * N_SIDE:] = self.lin_cross.detach().cpu().numpy() / 2
        return w

    # ---- the network ----------------------------------------------------------------------------
    def _tokens(self, pk, pf, field, x, agg=None):
        """Own side's 6 Pokemon first (pk / pf / agg already ordered), the global token from x (oriented)."""
        pk = pk.long()
        pf = pf.float()
        ids = lambda t, n: t.clamp(0, n - 1)
        types = self.ty(ids(pk[..., 3], VOCAB["type"])) + self.ty(ids(pk[..., 4], VOCAB["type"]))
        moves = pk[..., 7:11]
        mv = [pf[..., 21:25, None]]   # usable
        if self.ids == "all":
            mv.append(self.mv(ids(moves, VOCAB["move"])))
        if self.move_props:
            m = moves.clamp(0, self.mv_int.shape[0] - 1)
            mi, mf = self.mv_int[m], self.mv_flt[m]
            mv += [self.ty(ids(mi[..., 0], VOCAB["type"])), self.cat(mi[..., 1].clamp(0, 3)), mf,
                   (mi[..., 2:3].float() / 3).clamp(-2, 2)]
        mv = self.mv_mlp(torch.cat(mv, -1)).sum(-2)
        z = [types, self.ty(ids(pk[..., 5], VOCAB["type"])), self.st(ids(pk[..., 6], VOCAB["status"])), mv, pf]
        if self.ids == "all":
            z.append(self.sp(ids(pk[..., 0], VOCAB["species"])))
        if self.ids in ("all", "items"):
            z += [self.ab(ids(pk[..., 1], VOCAB["ability"])), self.it(ids(pk[..., 2], VOCAB["item"]))]
        if self.pairs:
            z.append(agg)
        h = self.poke(torch.cat(z, -1))
        h = h + torch.cat([self.side[0].expand(6, -1), self.side[1].expand(6, -1)])[None]
        g = self.glob(torch.cat([field.float(), (x.float() - self.x_mu) / self.x_sigma], -1))[:, None]
        hs = self.norm(self.tr(torch.cat([g, h], 1)))
        return hs[:, 0], hs[:, 1:7], hs[:, 7:13]

    def forward(self, ints, floats, pairs=None):
        pk, pf, act, field, x = split_inputs(ints, floats)
        b = pk.shape[0]
        perm = torch.tensor(list(range(6, 12)) + list(range(6)), device=pk.device)
        agg, view = None, None
        if self.pairs:
            if pairs is None:
                raise ValueError("this network needs pe.pair_features")
            p1, p2 = pair_views(pairs)
            alive = pf[..., 0].float()
            a1, a2 = pair_aggregates(p1, alive[:, 6:]), pair_aggregates(p2, alive[:, :6])
            agg = torch.cat([torch.cat([a1, a2], 1), torch.cat([a2, a1], 1)])
            view = torch.cat([p1, p2])
        # both perspectives in one batch: side one as "own", then side two
        g, own, opp = self._tokens(torch.cat([pk, pk[:, perm]]), torch.cat([pf, pf[:, perm]]),
                                   torch.cat([field, field]), torch.cat([x.float(), self.swap_x(x.float())]), agg)
        val = self.val(g).squeeze(-1)
        u = self.u(own)
        hp = [own[:, :, None].expand(-1, -1, 6, -1), opp[:, None, :].expand(-1, 6, -1, -1)]
        if self.pairs:
            hp.append(view)
        m = self.pair(torch.cat(hp, -1)).squeeze(-1)
        c = val[:b] - val[b:]
        w = m[:b] - m[b:].transpose(1, 2)
        return c, u[:b], u[b:], w

    @staticmethod
    def root_weight_vector(u1, u2, w):
        """The engine's layout (engine::learned::Evaluator::with_root_weights): (B, N_ROOT_WEIGHTS)."""
        b = u1.shape[0]
        return torch.cat([u1.reshape(b, -1), u2.reshape(b, -1), w.reshape(b, -1)], 1)

    @staticmethod
    def matchup_value(u1, u2, w, phi, pair):
        """The matchup term of v for states with these phi / pair (B,)."""
        return (u1 * phi[:, :6]).sum((1, 2)) - (u2 * phi[:, 6:]).sum((1, 2)) + (w.reshape(-1, 36) * pair).sum(1)


class MatchupLinear(nn.Module):
    """A low-capacity alternative to MatchupNet with the same interface: each Pokemon's weights u
    are linear in its matchup statistics (pair_aggregates over the living opponents, its HP,
    stats and whether it is active), W linear in the pair's features; no root value. ~360
    parameters: it can't memorize games, and its weights vary smoothly with the matchup."""

    N_IN = N_AGG + 9

    def __init__(self, x_mu=None, x_sigma=None, lin_side=None, lin_cross=None, **_):
        super().__init__()
        self.config = {"arch": "linear"}
        self.pairs = True
        self.u = nn.Linear(self.N_IN, M_LEAF, bias=False)
        self.wp = nn.Linear(N_PAIR, 1, bias=False)
        nn.init.zeros_(self.u.weight)
        nn.init.zeros_(self.wp.weight)
        self.register_buffer("x_mu", torch.zeros(N_X) if x_mu is None else torch.as_tensor(x_mu, dtype=torch.float32))
        self.register_buffer("x_sigma", torch.ones(N_X) if x_sigma is None else torch.as_tensor(x_sigma, dtype=torch.float32))
        self.lin_side = nn.Parameter(torch.zeros(N_SIDE) if lin_side is None else torch.as_tensor(lin_side, dtype=torch.float32))
        self.lin_cross = nn.Parameter(torch.zeros(N_X - 2 * N_SIDE) if lin_cross is None
                                      else torch.as_tensor(lin_cross, dtype=torch.float32))

    lin, lin_export = MatchupNet.lin, MatchupNet.lin_export
    swap_x = staticmethod(MatchupNet.swap_x)
    root_weight_vector = staticmethod(MatchupNet.root_weight_vector)
    matchup_value = staticmethod(MatchupNet.matchup_value)

    def forward(self, ints, floats, pairs=None):
        pk, pf, act, field, x = split_inputs(ints, floats)
        pf = pf.float()
        p1, p2 = pair_views(pairs)
        alive = pf[..., 0]
        a1, a2 = pair_aggregates(p1, alive[:, 6:]), pair_aggregates(p2, alive[:, :6])
        own = lambda a, f: torch.cat([a, f[..., 1:2], f[..., 3:9], f[..., 9:10], torch.ones_like(f[..., :1])], -1)
        u1, u2 = self.u(own(a1, pf[:, :6])), self.u(own(a2, pf[:, 6:]))
        w = self.wp(p1).squeeze(-1) - self.wp(p2).squeeze(-1).transpose(1, 2)
        return torch.zeros(pk.shape[0], device=pk.device), u1, u2, w


class MatchupModel:
    """A trained matchup evaluation for the search: the network (CPU, one thread) and the engine
    evaluator of its linear part."""

    def __init__(self, path, mode=2, scale=1.4, weight=1.0):
        """weight: a factor on the matchup term (1: as trained)."""
        from search.leaf_eval import evaluator_id
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["layout"] != [N_POKE_INTS, N_POKE_FLOATS, N_FIELD, M_LEAF, N_X, N_ROOT_WEIGHTS]:
            raise ValueError(f"{path}: its layout doesn't match this engine build's")
        cfg = dict(ck["config"])
        self.net = MatchupLinear() if cfg.pop("arch", "net") == "linear" else MatchupNet(**cfg)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.leaf_id = evaluator_id(ck["lin_npz"], mode, scale)
        self.path, self.weight = path, float(weight)

    @torch.no_grad()
    def root_weights(self, state):
        """The root weights of one search from its root State: a list for pe.mcts_with."""
        ints, floats = pe.matchup_inputs(state)
        pairs = torch.tensor([pe.pair_features(state)]) if self.net.pairs else None
        c, u1, u2, w = self.net(torch.tensor([ints], dtype=torch.int64), torch.tensor([floats]), pairs)
        rw = self.net.root_weight_vector(u1, u2, w)[0]
        return (rw * self.weight if self.weight != 1.0 else rw).tolist()
