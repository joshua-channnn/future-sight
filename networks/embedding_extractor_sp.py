import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

NUM_TYPES = 19


class SharedPokemonEncoder(nn.Module):
    """Encodes one Pokemon (13 indices + 31 floats) into a fixed-size repr."""

    def __init__(self, pokemon_embed, move_embed, ability_embed, item_embed, type_embed,
                 pokemon_embed_dim=48, move_embed_dim=24, ability_embed_dim=16,
                 item_embed_dim=16, type_embed_dim=16, n_float_features=31, output_dim=128):
        super().__init__()
        self.pokemon_embed = pokemon_embed
        self.move_embed = move_embed
        self.ability_embed = ability_embed
        self.item_embed = item_embed
        self.type_embed = type_embed

        total = (pokemon_embed_dim + move_embed_dim + ability_embed_dim + item_embed_dim
                 + type_embed_dim * 2 + type_embed_dim + n_float_features)
        # 48 + 24 + 16 + 16 + 32 + 16 + 31 = 183

        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(total, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.ReLU(),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, species_idx, move_indices, ability_idx, item_idx,
                type_indices, move_type_indices, float_features):
        se = self.pokemon_embed(species_idx)
        me = self.move_embed(move_indices)
        mm = (move_indices != 0).unsqueeze(-1).float()
        me = (me * mm).sum(1) / mm.sum(1).clamp(min=1)
        ae = self.ability_embed(ability_idx)
        ie = self.item_embed(item_idx)
        t1e = self.type_embed(type_indices[:, 0])
        t2e = self.type_embed(type_indices[:, 1])
        mte = self.type_embed(move_type_indices)
        mtm = (move_type_indices != 0).unsqueeze(-1).float()
        mte = (mte * mtm).sum(1) / mtm.sum(1).clamp(min=1)
        return self.mlp(torch.cat([se, me, ae, ie, t1e, t2e, mte, float_features], dim=-1))


class ActiveMatchupHead(nn.Module):
    """Processes active matchup: 2 Pokemon reprs + 4 move details."""

    def __init__(self, pokemon_repr_dim, move_embed, move_embed_dim=24, output_dim=128):
        super().__init__()
        self.move_embed = move_embed
        per_move = move_embed_dim + 12
        total = pokemon_repr_dim * 2 + per_move * 4
        self.mlp = nn.Sequential(
            nn.Linear(total, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.ReLU(),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, our_repr, opp_repr, move_indices, move_floats):
        me = self.move_embed(move_indices)
        mf = torch.cat([me, move_floats], dim=-1)
        return self.mlp(torch.cat([our_repr, opp_repr, mf.view(mf.size(0), -1)], dim=-1))


class V17SPFeatureExtractor(BaseFeaturesExtractor):
    """
    v17.3 Feature Extractor with Max+Mean Pooling.

    Final input to MLP:
      matchup(128) + our_bench_maxmean(256) + opp_team_maxmean(256) + global(60) = 700
    
    vs v17.2:
      matchup(128) + our_bench_max(128) + opp_team_max(128) + global(60) = 444
    """

    def __init__(self, observation_space, features_dim=256,
                 config_path="data/embedding_config.json",
                 pokemon_repr_dim=128, embed_dropout=0.05):
        super().__init__(observation_space, features_dim)

        with open(config_path) as f:
            config = json.load(f)
        self.config = config

        self.pokemon_embed_dim = 48
        self.move_embed_dim = 24
        self.ability_embed_dim = 16
        self.item_embed_dim = 16
        self.type_embed_dim = 16

        self.pokemon_embed = nn.Embedding(config["pokemon_vocab_size"], 48, padding_idx=0)
        self.move_embed = nn.Embedding(config["move_vocab_size"], 24, padding_idx=0)
        self.ability_embed = nn.Embedding(config["ability_vocab_size"], 16, padding_idx=0)
        self.item_embed = nn.Embedding(config["item_vocab_size"], 16, padding_idx=0)
        self.type_embed = nn.Embedding(NUM_TYPES, 16, padding_idx=0)

        for e in [self.pokemon_embed, self.move_embed, self.ability_embed,
                  self.item_embed, self.type_embed]:
            nn.init.normal_(e.weight, 0, 0.1); e.weight.data[0].zero_()

        self.pokemon_encoder = SharedPokemonEncoder(
            self.pokemon_embed, self.move_embed, self.ability_embed,
            self.item_embed, self.type_embed,
            n_float_features=31,
            output_dim=pokemon_repr_dim,
        )

        self.matchup_head = ActiveMatchupHead(
            pokemon_repr_dim, self.move_embed,
            move_embed_dim=24, output_dim=pokemon_repr_dim,
        )

        self.embed_dropout = nn.Dropout(embed_dropout)
        self.n_global_features = 60

        # our_bench: max(128) + mean(128) = 256
        # opp_team:  max(128) + mean(128) = 256
        # Total: matchup(128) + our_bench(256) + opp_team(256) + global(60) = 700
        final_input = pokemon_repr_dim + pokemon_repr_dim * 2 * 2 + self.n_global_features  # 700

        self.final_mlp = nn.Sequential(
            nn.LayerNorm(final_input),
            nn.Linear(final_input, 512), nn.ReLU(),
            nn.Linear(512, features_dim), nn.ReLU(),
        )
        for m in self.final_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _encode_pokemon_batch(self, obs, offset, batch_size):
        idx = offset
        species = obs[:, idx].long().clamp(0, self.config["pokemon_vocab_size"]-1); idx+=1
        moves = obs[:, idx:idx+4].long().clamp(0, self.config["move_vocab_size"]-1); idx+=4
        ability = obs[:, idx].long().clamp(0, self.config["ability_vocab_size"]-1); idx+=1
        item = obs[:, idx].long().clamp(0, self.config["item_vocab_size"]-1); idx+=1
        types = obs[:, idx:idx+2].long().clamp(0, NUM_TYPES-1); idx+=2
        mtypes = obs[:, idx:idx+4].long().clamp(0, NUM_TYPES-1); idx+=4
        floats = obs[:, idx:idx+31]; idx+=31
        return self.pokemon_encoder(species, moves, ability, item, types, mtypes, floats)

    def forward(self, observations):
        bs = observations.shape[0]
        PER = 44

        reprs = []
        for i in range(12):
            reprs.append(self._encode_pokemon_batch(observations, i*PER, bs))
        reprs = torch.stack(reprs, dim=1)
        reprs = self.embed_dropout(reprs)

        our_active = reprs[:, 0]
        our_bench = reprs[:, 1:6]
        opp_active = reprs[:, 6]
        opp_bench = reprs[:, 7:12]

        md_offset = 12 * PER  # 528
        mi_offset = md_offset + 48  # 576
        gl_offset = mi_offset + 4   # 580

        move_indices = observations[:, mi_offset:mi_offset+4].long().clamp(0, self.config["move_vocab_size"]-1)
        move_floats = observations[:, md_offset:md_offset+48].view(bs, 4, 12)

        matchup = self.matchup_head(our_active, opp_active, move_indices, move_floats)

        our_bench_max = our_bench.max(dim=1).values      # (batch, 128)
        our_bench_mean = our_bench.mean(dim=1)            # (batch, 128)
        our_bench_pool = torch.cat([our_bench_max, our_bench_mean], dim=-1)  # (batch, 256)

        opp_team = torch.cat([opp_active.unsqueeze(1), opp_bench], dim=1)
        opp_team_max = opp_team.max(dim=1).values         # (batch, 128)
        opp_team_mean = opp_team.mean(dim=1)               # (batch, 128)
        opp_team_pool = torch.cat([opp_team_max, opp_team_mean], dim=-1)  # (batch, 256)

        global_feats = observations[:, gl_offset:gl_offset+self.n_global_features]

        combined = torch.cat([matchup, our_bench_pool, opp_team_pool, global_feats], dim=-1)
        return self.final_mlp(combined)