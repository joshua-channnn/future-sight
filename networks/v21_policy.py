"""v21 policy/value network: v20's token encoders, a transformer over all tokens, per-token action
heads and auxiliary heads.

Input is the flat (idx, flt, mask) triple from envs.rl_player_v21.Featurizer: v20's 12 Pokemon,
4 move and 1 field token plus H history tokens. The network:
  - embeds and encodes the v20 tokens as V20Net does (one MLP per token type, tables shared), and
    each history token (actor, move, other Pokemon, outcome floats) with a fourth MLP, all to width d,
  - adds a learned token-type embedding (own Pokemon, opponent Pokemon, move, field, history, CLS)
    and a recency embedding to history tokens, and appends a CLS token,
  - runs a non-causal pre-LayerNorm transformer (no dropout: rollouts run in eval mode, updates in
    train mode). Absent moves and empty history slots are masked as keys; CLS and the field token
    always attend, so no attention row is empty. Unrevealed opponent placeholders stay visible:
    their number is information,
  - scores each move token twice (use, Terastallize + use) and each own Pokemon token once (switch
    to that team slot) from [token output, CLS output]; the value comes from the CLS output,
  - auxiliary heads (training signal only): item, base ability, tera type and moves of each
    opponent token; the opponent's next action from CLS (moves and tera moves by vocab index) plus
    one pointer logit per opponent token (switch to it). Labels: envs.rl_player_v21.
LayerNorm, not RMSNorm: nn.RMSNorm isn't a fused kernel on MPS and made the update 30% slower.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.rl_player_v20 import ABILITY_UNKNOWN, FLT_MV_END, FLT_PK_END, IDX_PK_END, ITEM_UNKNOWN, TYPE_UNKNOWN
from envs.rl_player_v21 import (LAB_ABILITY, LAB_ITEM, LAB_MOVES, LAB_NEXT, LAB_TERA, N_NEXT,
                                NEXT_SWITCH0, NEXT_UNREVEALED, make_layout)
from networks.v20_policy import MASK_VALUE, _mlp

DEFAULT_CONFIG = {
    "d": 128,             # token width
    "n_layers": 3,
    "n_heads": 4,
    "ff_mult": 2,         # feed-forward width = ff_mult * d
    "species_dim": 48, "move_dim": 32, "ability_dim": 16, "item_dim": 16, "type_dim": 16,
    "head_hidden": 256,
}
N_TOKEN_TYPES = 6  # own Pokemon, opponent Pokemon, move, field, history, CLS
AUX_HEADS = ("next", "item", "ability", "tera", "moves")


class Block(nn.Module):
    """Pre-LayerNorm transformer layer: x + MHA(LN(x)), then x + FF(LN(x))."""

    def __init__(self, d, n_heads, ff):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, x, key_bias):
        """x (B, T, d); key_bias (B, 1, 1, T): 0 for keys to attend to, a large negative otherwise."""
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D // self.n_heads)) + key_bias
        x = x + self.proj((att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class V21Net(nn.Module):
    has_aux = True

    def __init__(self, config=None, layout=None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        self.config = cfg
        L = dict(layout or make_layout())
        self.layout = L
        V = L["vocab"]
        d, hh = cfg["d"], cfg["head_hidden"]
        self.n_pk, self.n_mv = L["n_pk"], L["n_mv"]
        self.n_pk_idx, self.n_pk_flt = L["n_pk_idx"], L["n_pk_flt"]
        self.n_mv_idx, self.n_mv_flt = L["n_mv_idx"], L["n_mv_flt"]
        self.H, self.n_hs_idx, self.n_hs_flt = L["hist_len"], L["n_hs_idx"], L["n_hs_flt"]
        self.v20_n_idx, self.v20_n_flt = L["v20_n_idx"], L["v20_n_flt"]

        self.species_emb = nn.Embedding(V["species"], cfg["species_dim"], padding_idx=0)
        self.move_emb = nn.Embedding(V["move"], cfg["move_dim"], padding_idx=0)
        self.ability_emb = nn.Embedding(V["ability"], cfg["ability_dim"], padding_idx=0)
        self.item_emb = nn.Embedding(V["item"], cfg["item_dim"], padding_idx=0)
        self.type_emb = nn.Embedding(V["type"], cfg["type_dim"], padding_idx=0)

        pk_in = (cfg["species_dim"] + cfg["item_dim"] + cfg["ability_dim"] + 2 * cfg["type_dim"]
                 + 2 * cfg["move_dim"] + self.n_pk_flt)
        mv_in = cfg["move_dim"] + cfg["type_dim"] + self.n_mv_flt
        hs_in = 2 * cfg["species_dim"] + cfg["move_dim"] + self.n_hs_flt
        self.pk_enc = _mlp([pk_in, 256, d], final_relu=False)
        self.mv_enc = _mlp([mv_in, 128, d], final_relu=False)
        self.fd_enc = _mlp([L["n_fd_flt"], 128, d], final_relu=False)
        self.hs_enc = _mlp([hs_in, 128, d], final_relu=False)
        self.token_type = nn.Parameter(torch.zeros(N_TOKEN_TYPES, d))
        self.hist_pos = nn.Parameter(torch.zeros(self.H, d))
        self.cls = nn.Parameter(torch.zeros(d))
        types = [0] * 6 + [1] * 6 + [2] * self.n_mv + [3] + [4] * self.H + [5]
        self.register_buffer("types", torch.tensor(types), persistent=False)

        self.blocks = nn.ModuleList([Block(d, cfg["n_heads"], cfg["ff_mult"] * d)
                                     for _ in range(cfg["n_layers"])])
        self.norm = nn.LayerNorm(d)

        self.move_head = _mlp([2 * d, hh, 2], final_relu=False)
        self.switch_head = _mlp([2 * d, hh, 1], final_relu=False)
        self.value_head = _mlp([d, hh, 1], final_relu=False)
        self.aux_item = nn.Linear(d, V["item"])
        self.aux_ability = nn.Linear(d, V["ability"])
        self.aux_tera = nn.Linear(d, V["type"])
        self.aux_moves = nn.Linear(d, V["move"])
        self.aux_next_move = nn.Linear(d, NEXT_SWITCH0)
        self.aux_next_switch = nn.Linear(d, 1)
        self._init_weights()

    def _init_weights(self):
        # encoders and heads as in v20 (orthogonal sqrt(2), 0.01 for policy logits, 1 for value);
        # the trunk GPT-2 style (normal 0.02, residual projections scaled down by depth)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        for head, gain in ((self.move_head[-1], 0.01), (self.switch_head[-1], 0.01), (self.value_head[-1], 1.0),
                           (self.aux_item, 0.01), (self.aux_ability, 0.01), (self.aux_tera, 0.01),
                           (self.aux_moves, 0.01), (self.aux_next_move, 0.01), (self.aux_next_switch, 0.01)):
            nn.init.orthogonal_(head.weight, gain)
        resid = 0.02 / math.sqrt(2 * len(self.blocks))
        for b in self.blocks:
            for lin, std in ((b.qkv, 0.02), (b.ff1, 0.02), (b.proj, resid), (b.ff2, resid)):
                nn.init.normal_(lin.weight, 0.0, std)
                nn.init.zeros_(lin.bias)
        for p in (self.token_type, self.hist_pos):
            nn.init.normal_(p, 0.0, 0.02)
        for e in (self.species_emb, self.move_emb, self.ability_emb, self.item_emb, self.type_emb):
            nn.init.normal_(e.weight, 0.0, 0.1)
            with torch.no_grad():
                e.weight[0].zero_()

    def embedding_parameters(self):
        return [self.species_emb.weight, self.move_emb.weight, self.ability_emb.weight,
                self.item_emb.weight, self.type_emb.weight]

    def other_parameters(self):
        emb = {id(p) for p in self.embedding_parameters()}
        return [p for p in self.parameters() if id(p) not in emb]

    # ---- trunk ------------------------------------------------------------------------------
    def tokens(self, idx, flt):
        """Token outputs (B, T, d), T = 12 Pokemon + 4 moves + field + H history + CLS."""
        B = idx.shape[0]
        L = self.layout
        pk_idx = idx[:, :IDX_PK_END].view(B, self.n_pk, self.n_pk_idx)
        mv_idx = idx[:, IDX_PK_END:self.v20_n_idx].view(B, self.n_mv, self.n_mv_idx)
        hs_idx = idx[:, self.v20_n_idx:].view(B, self.H, self.n_hs_idx)
        pk_flt = flt[:, :FLT_PK_END].view(B, self.n_pk, self.n_pk_flt)
        mv_flt = flt[:, FLT_PK_END:FLT_MV_END].view(B, self.n_mv, self.n_mv_flt)
        fd_flt = flt[:, FLT_MV_END:self.v20_n_flt]
        hs_flt = flt[:, self.v20_n_flt:].view(B, self.H, self.n_hs_flt)

        # Pokemon tokens, as V20Net: [species, item, ability, tera, type1, type2, move0-3, last_move]
        moves = pk_idx[..., 6:10]
        me = self.move_emb(moves)
        present = (moves != 0).unsqueeze(-1).to(me.dtype)
        me = (me * present).sum(-2) / present.sum(-2).clamp(min=1.0)
        types = self.type_emb(pk_idx[..., 4]) + self.type_emb(pk_idx[..., 5])
        pk = self.pk_enc(torch.cat([
            self.species_emb(pk_idx[..., 0]), self.item_emb(pk_idx[..., 1]),
            self.ability_emb(pk_idx[..., 2]), self.type_emb(pk_idx[..., 3]), types,
            me, self.move_emb(pk_idx[..., 10]), pk_flt], dim=-1))                     # (B, 12, d)
        mv = self.mv_enc(torch.cat([self.move_emb(mv_idx[..., 0]), self.type_emb(mv_idx[..., 1]), mv_flt], -1))
        fd = self.fd_enc(fd_flt).unsqueeze(1)
        hs = self.hs_enc(torch.cat([self.species_emb(hs_idx[..., 0]), self.move_emb(hs_idx[..., 1]),
                                    self.species_emb(hs_idx[..., 2]), hs_flt], -1)) + self.hist_pos
        x = torch.cat([pk, mv, fd, hs, self.cls.expand(B, 1, -1)], 1) + self.token_type[self.types]

        own_valid = pk_flt[:, :6, L["pk_placeholder"]] < 0.5
        always = torch.ones(B, 1, dtype=torch.bool, device=idx.device)
        valid = torch.cat([own_valid, torch.ones_like(own_valid), mv_flt[..., L["mv_exists"]] > 0.5, always,
                           hs_flt[..., L["hs_present"]] > 0.5, always], 1)
        key_bias = torch.zeros(valid.shape, dtype=x.dtype, device=x.device).masked_fill(~valid, -1e9)
        key_bias = key_bias[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, key_bias)
        return self.norm(x)

    def forward(self, idx, flt, mask, aux=False):
        """idx (B, N_IDX) long, flt (B, N_FLT) float, mask (B, 14) bool -> masked logits (B, 14),
        value (B,) [, aux logits: dict of (B, 6, C) per opponent token, and "next" (B, N_NEXT)]."""
        x = self.tokens(idx, flt)
        B = x.shape[0]
        cls = x[:, -1]
        own, opp, mv = x[:, :6], x[:, 6:12], x[:, 12:12 + self.n_mv]
        move_logits = self.move_head(torch.cat([mv, cls.unsqueeze(1).expand(B, self.n_mv, -1)], -1))
        switch_logits = self.switch_head(torch.cat([own, cls.unsqueeze(1).expand(B, 6, -1)], -1)).squeeze(-1)
        logits = torch.cat([move_logits[..., 0], move_logits[..., 1], switch_logits], dim=-1)
        logits = torch.where(mask, logits, torch.full_like(logits, MASK_VALUE))
        value = self.value_head(cls).squeeze(-1)
        if not aux:
            return logits, value
        out = {"item": self.aux_item(opp), "ability": self.aux_ability(opp), "tera": self.aux_tera(opp),
               "moves": self.aux_moves(opp),
               "next": torch.cat([self.aux_next_move(cls), self.aux_next_switch(opp).squeeze(-1)], -1)}
        return logits, value, out

    @torch.no_grad()
    def act(self, idx, flt, mask, deterministic=False):
        """Sample (or argmax) actions: returns action (B,), log-prob (B,), value (B,)."""
        logits, value = self.forward(idx, flt, mask)
        logp_all = F.log_softmax(logits, dim=-1)
        if deterministic:
            action = logits.argmax(-1)
        else:
            action = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
        return action, logp_all.gather(-1, action.unsqueeze(-1)).squeeze(-1), value

    def evaluate_actions(self, idx, flt, mask, actions, aux=False):
        """For the PPO update: log-prob of `actions`, entropy, value, KL(uniform over legal actions ||
        policy) (the zero-avoiding term) [, aux logits]."""
        res = self.forward(idx, flt, mask, aux=aux)
        logits, value = res[0], res[1]
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        p = logp_all.exp()
        zero = torch.zeros_like(logp_all)
        entropy = -torch.where(mask, p * logp_all, zero).sum(-1)
        n_legal = mask.sum(-1).clamp(min=1).to(logp_all.dtype)
        kl_uniform = -torch.log(n_legal) - torch.where(mask, logp_all, zero).sum(-1) / n_legal
        if aux:
            return logp, entropy, value, kl_uniform, res[2]
        return logp, entropy, value, kl_uniform


# =============================================================================================
# Auxiliary losses and metrics. Every tensor keeps a fixed shape (masks multiply, never index):
# MPS compiles a graph per input shape.
def aux_obs(idx, flt, layout):
    """What the aux losses and metrics read from the observation, per opponent token (B, 6):
    species, the item / ability / tera indices it shows, its revealed moves (B, 6, 4), whether it
    is a placeholder or active."""
    B = idx.shape[0]
    n_idx, n_flt = layout["n_pk_idx"], layout["n_pk_flt"]
    oi = idx[:, 6 * n_idx:12 * n_idx].view(B, 6, n_idx)
    of = flt[:, 6 * n_flt:12 * n_flt].view(B, 6, n_flt)
    return {"species": oi[..., 0], "item": oi[..., 1], "ability": oi[..., 2], "tera": oi[..., 3],
            "moves": oi[..., 6:10], "placeholder": of[..., layout["pk_placeholder"]] > 0.5,
            "active": of[..., layout["pk_active"]] > 0.5}


def _ce(logits, target):
    """Cross-entropy per target (0 where target < 0) and the validity mask."""
    valid = target >= 0
    nll = -F.log_softmax(logits, -1).gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return nll * valid, valid


def _mean(x, valid):
    return x.sum() / valid.sum().clamp(min=1)


def aux_terms(out, lab, obs, metrics=False):
    """Losses {head: mean loss} for the aux logits `out` against labels `lab` (B, N_LAB) long, and
    with metrics=True also {name: (sum, count)} accuracies, overall and on what the observation
    doesn't show yet (`_unrev`). Works for any logits of the right shapes, so the per-species
    base-rate tables are scored by the same code."""
    losses, mets = {}, {}
    B = lab.shape[0]
    unknown = {"item": ITEM_UNKNOWN, "ability": ABILITY_UNKNOWN, "tera": TYPE_UNKNOWN}
    for name, start in (("item", LAB_ITEM), ("ability", LAB_ABILITY), ("tera", LAB_TERA)):
        target = lab[:, start:start + 6]
        nll, valid = _ce(out[name], target)
        losses[name] = _mean(nll, valid)
        if metrics:
            correct = (out[name].argmax(-1) == target) & valid
            unrev = valid & (obs[name] == unknown[name])
            mets[name] = (correct.sum(), valid.sum())
            mets[name + "_unrev"] = ((correct & unrev).sum(), unrev.sum())

    # moves: cross-entropy against the set's moves, 1/k each (k = 4 for nearly every set)
    logits = out["moves"]
    n_moves = logits.shape[-1]
    mv = lab[:, LAB_MOVES:LAB_MOVES + 24].view(B, 6, 4)
    mvalid = mv >= 0
    true = torch.zeros(B, 6, n_moves, device=lab.device).scatter_add_(-1, mv.clamp(min=0), mvalid.float())
    k = mvalid.sum(-1)
    tok = k > 0
    nll = -(true / k.clamp(min=1).unsqueeze(-1) * F.log_softmax(logits, -1)).sum(-1)
    losses["moves"] = _mean(nll * tok, tok)
    if metrics:
        true_b = true > 0
        hits = true_b.gather(-1, logits.topk(4, -1).indices).sum(-1)
        mets["moves"] = ((hits * tok).sum(), (k * tok).sum())
        # unrevealed moves: are they in the top k_u of the moves not shown yet?
        shown = torch.zeros(B, 6, n_moves, dtype=torch.bool, device=lab.device).scatter_(
            -1, obs["moves"].long(), True)
        shown[..., 0] = False
        unrev_true = true_b & ~shown
        k_u = unrev_true.sum(-1, keepdim=True)
        order = logits.masked_fill(shown, -1e9).argsort(-1, descending=True)
        rank = torch.empty_like(order).scatter_(-1, order, torch.arange(n_moves, device=lab.device).expand_as(order))
        mets["moves_unrev"] = ((unrev_true & (rank < k_u)).sum(), k_u.sum())

    # next action: one target class, or for a switch to an unrevealed Pokemon every placeholder
    logits = out["next"]
    t = lab[:, LAB_NEXT]
    single = (t >= 0) & (t < N_NEXT)
    target = F.one_hot(t.clamp(0, N_NEXT - 1), N_NEXT).bool() & single.unsqueeze(-1)
    ph = torch.cat([torch.zeros(B, NEXT_SWITCH0, dtype=torch.bool, device=lab.device), obs["placeholder"]], -1)
    target = target | (ph & (t == NEXT_UNREVEALED).unsqueeze(-1))
    valid = target.any(-1)
    nll = torch.logsumexp(logits, -1) - torch.logsumexp(logits.masked_fill(~target, -1e9), -1)
    losses["next"] = _mean(nll * valid, valid)
    if metrics:
        pred = logits.argmax(-1, keepdim=True)
        correct = target.gather(-1, pred).squeeze(-1) & valid
        mets["next"] = (correct.sum(), valid.sum())
        kind_ok = (pred.squeeze(-1) < NEXT_SWITCH0) == ((t >= 0) & (t < NEXT_SWITCH0))
        mets["next_kind"] = ((kind_ok & valid).sum(), valid.sum())
    return losses, mets


class AuxBaseline:
    """Per-species base rates for the aux heads, from the labels of earlier updates: the most
    frequent item / ability / tera type, the 4 most frequent moves of each species, and the most
    frequent move of the opponent's active species (it never predicts a switch). An aux head has to
    beat these to have learned more than species -> set."""

    def __init__(self, layout):
        V = layout["vocab"]
        S = V["species"]
        self.layout = layout
        self.tables = {"item": torch.zeros(S, V["item"]), "ability": torch.zeros(S, V["ability"]),
                       "tera": torch.zeros(S, V["type"]), "moves": torch.zeros(S, V["move"]),
                       "next": torch.zeros(S, NEXT_SWITCH0)}

    def device_tables(self, device):
        return {k: v.to(device) for k, v in self.tables.items()}

    @staticmethod
    def logits(obs, tables):
        """Base-rate 'logits' in aux_terms' shapes. `obs` from aux_obs; `tables` from device_tables
        on the same device."""
        sp = obs["species"].long()
        out = {k: tables[k][sp] for k in ("item", "ability", "tera", "moves")}
        act = (sp * obs["active"].long()).sum(-1)
        out["next"] = torch.cat([tables["next"][act],
                                 torch.full((sp.shape[0], 6), -1e9, device=sp.device)], -1)
        return out

    def update(self, idx, flt, lab):
        """Add one rollout's labels (numpy arrays from collect) to the counts."""
        obs = aux_obs(torch.as_tensor(idx).long(), torch.as_tensor(flt).float(), self.layout)
        lab = torch.as_tensor(lab).long()
        sp = obs["species"].long()
        for name, start in (("item", LAB_ITEM), ("ability", LAB_ABILITY), ("tera", LAB_TERA)):
            t = lab[:, start:start + 6]
            ok = t >= 0
            self.tables[name].index_put_((sp[ok], t[ok]), torch.ones(int(ok.sum())), accumulate=True)
        mv = lab[:, LAB_MOVES:LAB_MOVES + 24].view(-1, 6, 4)
        ok = mv >= 0
        self.tables["moves"].index_put_((sp.unsqueeze(-1).expand_as(mv)[ok], mv[ok]), torch.ones(int(ok.sum())),
                                        accumulate=True)
        t = lab[:, LAB_NEXT]
        act = (sp * obs["active"].long()).sum(-1)
        ok = (t >= 0) & (t < NEXT_SWITCH0)
        self.tables["next"].index_put_((act[ok], t[ok]), torch.ones(int(ok.sum())), accumulate=True)

    def state_dict(self):
        return {k: v.to(torch.int32) for k, v in self.tables.items()}

    def load_state_dict(self, state):
        for k, v in state.items():
            if k in self.tables and self.tables[k].shape == v.shape:
                self.tables[k] = v.float()


def build_from_checkpoint(ckpt):
    """V21Net with the weights of a checkpoint dict ({"model", "net_config", "layout", ...})."""
    net = V21Net(ckpt.get("net_config"), ckpt.get("layout"))
    net.load_state_dict(ckpt["model"])
    return net
