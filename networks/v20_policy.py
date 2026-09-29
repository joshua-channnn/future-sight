"""v20 policy/value network: per-token encoders, one shared trunk, per-token action heads.

Input is the flat (idx, flt, mask) triple from envs.rl_player_v20.Featurizer. The network:
  - embeds species, items, abilities, types and moves (tables shared by every token type),
  - encodes each of the 12 Pokemon tokens with one MLP (a side flag tells own from opponent),
    each of the 4 move tokens with another, and the field token with a third,
  - builds a context vector from the field, both actives, and mean + max over each team,
  - scores each move token twice (use, Terastallize + use), each own Pokemon token once (switch
    to that team slot), and the context once (value).
All tokens of a type go through their encoder in one call; there are no per-token Python loops.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.rl_player_v20 import LAYOUT, IDX_PK_END, FLT_PK_END, FLT_MV_END

MASK_VALUE = -1e8

DEFAULT_CONFIG = {
    "d": 128,             # token width
    "ctx": 256,           # context width
    "species_dim": 48, "move_dim": 32, "ability_dim": 16, "item_dim": 16, "type_dim": 16,
    "head_hidden": 256,
}


def _mlp(sizes, final_relu=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class V20Net(nn.Module):
    def __init__(self, config=None, layout=LAYOUT):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        self.config = cfg
        self.layout = dict(layout)
        L = layout
        V = L["vocab"]
        d, ctx, hh = cfg["d"], cfg["ctx"], cfg["head_hidden"]
        self.n_pk, self.n_mv = L["n_pk"], L["n_mv"]
        self.n_pk_idx, self.n_pk_flt = L["n_pk_idx"], L["n_pk_flt"]
        self.n_mv_idx, self.n_mv_flt = L["n_mv_idx"], L["n_mv_flt"]
        self.pk_active = L["pk_active"]

        self.species_emb = nn.Embedding(V["species"], cfg["species_dim"], padding_idx=0)
        self.move_emb = nn.Embedding(V["move"], cfg["move_dim"], padding_idx=0)
        self.ability_emb = nn.Embedding(V["ability"], cfg["ability_dim"], padding_idx=0)
        self.item_emb = nn.Embedding(V["item"], cfg["item_dim"], padding_idx=0)
        self.type_emb = nn.Embedding(V["type"], cfg["type_dim"], padding_idx=0)

        pk_in = (cfg["species_dim"] + cfg["item_dim"] + cfg["ability_dim"] + 2 * cfg["type_dim"]
                 + 2 * cfg["move_dim"] + self.n_pk_flt)
        mv_in = cfg["move_dim"] + cfg["type_dim"] + self.n_mv_flt
        self.pk_enc = _mlp([pk_in, 256, d])
        self.mv_enc = _mlp([mv_in, 128, d])
        self.fd_enc = _mlp([L["n_fd_flt"], 128, d])
        self.ctx_norm = nn.LayerNorm(7 * d)
        self.ctx_mlp = _mlp([7 * d, 512, ctx])
        self.move_head = _mlp([d + ctx, hh, 2], final_relu=False)
        self.switch_head = _mlp([2 * d + ctx, hh, 1], final_relu=False)
        self.value_head = _mlp([ctx, hh, 1], final_relu=False)
        self._init_weights()

    def _init_weights(self):
        # CleanRL-style: orthogonal sqrt(2) for hidden layers, 0.01 for the policy logits, 1 for value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        for head, gain in ((self.move_head, 0.01), (self.switch_head, 0.01), (self.value_head, 1.0)):
            nn.init.orthogonal_(head[-1].weight, gain)
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

    def forward(self, idx, flt, mask):
        """idx (B, N_IDX) long, flt (B, N_FLT) float, mask (B, 14) bool ->
        masked logits (B, 14), value (B,)."""
        B = idx.shape[0]
        pk_idx = idx[:, :IDX_PK_END].view(B, self.n_pk, self.n_pk_idx)
        mv_idx = idx[:, IDX_PK_END:].view(B, self.n_mv, self.n_mv_idx)
        pk_flt = flt[:, :FLT_PK_END].view(B, self.n_pk, self.n_pk_flt)
        mv_flt = flt[:, FLT_PK_END:FLT_MV_END].view(B, self.n_mv, self.n_mv_flt)
        fd_flt = flt[:, FLT_MV_END:]

        # Pokemon tokens: [species, item, ability, tera, type1, type2, move0-3, last_move]
        moves = pk_idx[..., 6:10]
        me = self.move_emb(moves)
        present = (moves != 0).unsqueeze(-1).to(me.dtype)
        me = (me * present).sum(-2) / present.sum(-2).clamp(min=1.0)
        types = self.type_emb(pk_idx[..., 4]) + self.type_emb(pk_idx[..., 5])
        pk_in = torch.cat([
            self.species_emb(pk_idx[..., 0]), self.item_emb(pk_idx[..., 1]),
            self.ability_emb(pk_idx[..., 2]), self.type_emb(pk_idx[..., 3]), types,
            me, self.move_emb(pk_idx[..., 10]), pk_flt], dim=-1)
        pk = self.pk_enc(pk_in)                                   # (B, 12, d)
        own, opp = pk[:, :6], pk[:, 6:]
        own_act = (own * pk_flt[:, :6, self.pk_active:self.pk_active + 1]).sum(1)
        opp_act = (opp * pk_flt[:, 6:, self.pk_active:self.pk_active + 1]).sum(1)

        mv_in = torch.cat([self.move_emb(mv_idx[..., 0]), self.type_emb(mv_idx[..., 1]), mv_flt], dim=-1)
        mv = self.mv_enc(mv_in)                                   # (B, 4, d)
        fd = self.fd_enc(fd_flt)                                  # (B, d)

        ctx = self.ctx_mlp(self.ctx_norm(torch.cat([
            fd, own_act, opp_act, own.mean(1), own.max(1).values, opp.mean(1), opp.max(1).values],
            dim=-1)))                                             # (B, ctx)

        move_logits = self.move_head(torch.cat([mv, ctx.unsqueeze(1).expand(B, self.n_mv, -1)], -1))
        switch_in = torch.cat([own, opp_act.unsqueeze(1).expand(B, 6, -1),
                               ctx.unsqueeze(1).expand(B, 6, -1)], -1)
        switch_logits = self.switch_head(switch_in).squeeze(-1)  # (B, 6)
        logits = torch.cat([move_logits[..., 0], move_logits[..., 1], switch_logits], dim=-1)
        logits = torch.where(mask, logits, torch.full_like(logits, MASK_VALUE))
        value = self.value_head(ctx).squeeze(-1)
        return logits, value

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

    def evaluate_actions(self, idx, flt, mask, actions):
        """For the PPO update: log-prob of `actions`, entropy, value, and KL(uniform over legal
        actions || policy), the zero-avoiding term."""
        logits, value = self.forward(idx, flt, mask)
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        p = logp_all.exp()
        zero = torch.zeros_like(logp_all)
        entropy = -torch.where(mask, p * logp_all, zero).sum(-1)
        n_legal = mask.sum(-1).clamp(min=1).to(logp_all.dtype)
        kl_uniform = -torch.log(n_legal) - torch.where(mask, logp_all, zero).sum(-1) / n_legal
        return logp, entropy, value, kl_uniform


def build_from_checkpoint(ckpt):
    """V20Net with the weights of a checkpoint dict ({"model", "net_config", ...})."""
    net = V20Net(ckpt.get("net_config"))
    net.load_state_dict(ckpt["model"])
    return net
