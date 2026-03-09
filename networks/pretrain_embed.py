import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str


TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]
TYPE_TO_IDX = {t: i + 1 for i, t in enumerate(TYPES)}
TYPE_TO_IDX[""] = 0
TYPE_TO_IDX[None] = 0
NUM_TYPES = 19

# For verification
IDX_TO_TYPE = {v: k for k, v in TYPE_TO_IDX.items()}

CATEGORY_TO_IDX = {"physical": 0, "special": 1, "status": 2}

EFFECTIVENESS_WEIGHTS = torch.tensor([
    8.0,   # 0: immune (3.8%)
    12.0,  # 1: 0.25x (2.0%)
    1.5,   # 2: 0.5x (19.1%)
    0.3,   # 3: 1x (56.5%) - heavily downweight
    1.5,   # 4: 2x (17.5%)
    20.0,  # 5: 4x (1.2%)
])


class PokemonTypeDataset(Dataset):
    """Pokemon -> Types (multi-label classification)"""
    
    def __init__(self, data_dir: str, gen: int = 9):
        self.gen_data = GenData.from_gen(gen)
        
        with open(Path(data_dir) / "pokemon_to_idx.json") as f:
            self.pokemon_to_idx = json.load(f)
        
        self.samples = []
        self._build_samples()
        print(f"PokemonTypeDataset: {len(self.samples)} samples")
    
    def _build_samples(self):
        pokedex = self.gen_data.pokedex
        
        for pokemon_name, poke_info in pokedex.items():
            pokemon_id = to_id_str(pokemon_name)
            pokemon_idx = self.pokemon_to_idx.get(pokemon_id, 0)
            if pokemon_idx == 0:
                continue
            
            pokemon_types = poke_info.get("types", [])
            if not pokemon_types:
                continue
            
            # Store type indices directly for easier accuracy calculation
            type1_idx = TYPE_TO_IDX.get(pokemon_types[0].lower(), 0)
            type2_idx = TYPE_TO_IDX.get(pokemon_types[1].lower(), 0) if len(pokemon_types) > 1 else 0
            
            # Multi-hot encoding
            type_labels = [0.0] * NUM_TYPES
            if type1_idx > 0:
                type_labels[type1_idx] = 1.0
            if type2_idx > 0:
                type_labels[type2_idx] = 1.0
            
            self.samples.append({
                "pokemon_idx": pokemon_idx,
                "type_labels": type_labels,
                "type1_idx": type1_idx,
                "type2_idx": type2_idx,
                "num_types": 1 if type2_idx == 0 else 2,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "pokemon_idx": torch.tensor(s["pokemon_idx"], dtype=torch.long),
            "type_labels": torch.tensor(s["type_labels"], dtype=torch.float32),
            "type1_idx": torch.tensor(s["type1_idx"], dtype=torch.long),
            "type2_idx": torch.tensor(s["type2_idx"], dtype=torch.long),
            "num_types": torch.tensor(s["num_types"], dtype=torch.long),
        }


class MoveTypeDataset(Dataset):
    """Move -> Type, Power, Category"""
    
    def __init__(self, data_dir: str, gen: int = 9):
        self.gen_data = GenData.from_gen(gen)
        
        with open(Path(data_dir) / "move_to_idx.json") as f:
            self.move_to_idx = json.load(f)
        
        self.samples = []
        self._build_samples()
        print(f"MoveTypeDataset: {len(self.samples)} samples")
    
    def _build_samples(self):
        moves = self.gen_data.moves
        
        for move_name, move_info in moves.items():
            move_id = to_id_str(move_name)
            move_idx = self.move_to_idx.get(move_id, 0)
            if move_idx == 0:
                continue
            
            move_type = move_info.get("type", "Normal").lower()
            type_idx = TYPE_TO_IDX.get(move_type, 0)
            
            power = move_info.get("basePower", 0) / 150.0
            category = move_info.get("category", "Status").lower()
            category_idx = CATEGORY_TO_IDX.get(category, 2)
            
            self.samples.append({
                "move_idx": move_idx,
                "type_idx": type_idx,
                "power": power,
                "category_idx": category_idx,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "move_idx": torch.tensor(s["move_idx"], dtype=torch.long),
            "type_idx": torch.tensor(s["type_idx"], dtype=torch.long),
            "power": torch.tensor(s["power"], dtype=torch.float32),
            "category_idx": torch.tensor(s["category_idx"], dtype=torch.long),
        }


class EffectivenessDataset(Dataset):
    """Pokemon + Move -> Effectiveness class"""
    
    def __init__(self, data_dir: str, gen: int = 9):
        self.gen_data = GenData.from_gen(gen)
        self.type_chart = self.gen_data.type_chart
        
        with open(Path(data_dir) / "pokemon_to_idx.json") as f:
            self.pokemon_to_idx = json.load(f)
        with open(Path(data_dir) / "move_to_idx.json") as f:
            self.move_to_idx = json.load(f)
        
        self.samples = []
        self._build_samples()
        print(f"EffectivenessDataset: {len(self.samples)} samples")
    
    def _get_effectiveness(self, attack_type: str, defend_types: List[str]) -> float:
        multiplier = 1.0
        attack_type = attack_type.upper()
        
        for dtype in defend_types:
            if dtype:
                dtype_upper = dtype.upper()
                if attack_type in self.type_chart and dtype_upper in self.type_chart[attack_type]:
                    multiplier *= self.type_chart[attack_type][dtype_upper]
        
        return multiplier
    
    def _effectiveness_to_class(self, eff: float) -> int:
        if eff == 0:
            return 0
        elif eff <= 0.25:
            return 1
        elif eff <= 0.5:
            return 2
        elif eff <= 1.0:
            return 3
        elif eff <= 2.0:
            return 4
        else:
            return 5
    
    def _build_samples(self):
        pokedex = self.gen_data.pokedex
        moves = self.gen_data.moves
        
        for pokemon_name, poke_info in pokedex.items():
            pokemon_id = to_id_str(pokemon_name)
            pokemon_idx = self.pokemon_to_idx.get(pokemon_id, 0)
            if pokemon_idx == 0:
                continue
            
            pokemon_types = poke_info.get("types", [])
            if not pokemon_types:
                continue
            
            for move_name, move_info in moves.items():
                move_id = to_id_str(move_name)
                move_idx = self.move_to_idx.get(move_id, 0)
                if move_idx == 0:
                    continue
                
                move_type = move_info.get("type", "Normal")
                effectiveness = self._get_effectiveness(move_type, pokemon_types)
                eff_class = self._effectiveness_to_class(effectiveness)
                
                self.samples.append({
                    "pokemon_idx": pokemon_idx,
                    "move_idx": move_idx,
                    "effectiveness_class": eff_class,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "pokemon_idx": torch.tensor(s["pokemon_idx"], dtype=torch.long),
            "move_idx": torch.tensor(s["move_idx"], dtype=torch.long),
            "effectiveness_class": torch.tensor(s["effectiveness_class"], dtype=torch.long),
        }


class PokemonStatsDataset(Dataset):
    """Pokemon -> Base stats"""
    
    def __init__(self, data_dir: str, gen: int = 9):
        self.gen_data = GenData.from_gen(gen)
        
        with open(Path(data_dir) / "pokemon_to_idx.json") as f:
            self.pokemon_to_idx = json.load(f)
        
        self.samples = []
        self._build_samples()
        print(f"PokemonStatsDataset: {len(self.samples)} samples")
    
    def _build_samples(self):
        pokedex = self.gen_data.pokedex
        
        for pokemon_name, poke_info in pokedex.items():
            pokemon_id = to_id_str(pokemon_name)
            pokemon_idx = self.pokemon_to_idx.get(pokemon_id, 0)
            if pokemon_idx == 0:
                continue
            
            base_stats = poke_info.get("baseStats", {})
            if not base_stats:
                continue
            
            stats = [
                base_stats.get("hp", 80) / 255.0,
                base_stats.get("atk", 80) / 255.0,
                base_stats.get("def", 80) / 255.0,
                base_stats.get("spa", 80) / 255.0,
                base_stats.get("spd", 80) / 255.0,
                base_stats.get("spe", 80) / 255.0,
            ]
            
            self.samples.append({
                "pokemon_idx": pokemon_idx,
                "stats": stats,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "pokemon_idx": torch.tensor(s["pokemon_idx"], dtype=torch.long),
            "stats": torch.tensor(s["stats"], dtype=torch.float32),
        }


class EmbeddingPretrainerV3(nn.Module):
    """
    Two-phase pretraining model.
    
    Phase 1: Train type prediction heads
    Phase 2: Freeze embeddings, train effectiveness head
    """
    
    def __init__(self, config_path: str):
        super().__init__()
        
        with open(config_path) as f:
            config = json.load(f)
        
        self.config = config
        
        pokemon_dim = config["pokemon_embed_dim"]  # 64
        move_dim = config["move_embed_dim"]        # 32
        
        # Embeddings
        self.pokemon_embed = nn.Embedding(
            config["pokemon_vocab_size"], pokemon_dim, padding_idx=0
        )
        self.move_embed = nn.Embedding(
            config["move_vocab_size"], move_dim, padding_idx=0
        )
        self.ability_embed = nn.Embedding(
            config["ability_vocab_size"], config["ability_embed_dim"], padding_idx=0
        )
        self.item_embed = nn.Embedding(
            config["item_vocab_size"], config["item_embed_dim"], padding_idx=0
        )
        self.type_embed = nn.Embedding(NUM_TYPES, 16, padding_idx=0)
        
        # Pokemon -> Types (predict both types)
        self.pokemon_type_head = nn.Sequential(
            nn.Linear(pokemon_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_TYPES),
        )
        
        # Move -> Type
        self.move_type_head = nn.Sequential(
            nn.Linear(move_dim, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_TYPES),
        )
        
        # Move -> Power
        self.move_power_head = nn.Sequential(
            nn.Linear(move_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Move -> Category
        self.move_category_head = nn.Sequential(
            nn.Linear(move_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        
        # Pokemon -> Stats
        self.pokemon_stats_head = nn.Sequential(
            nn.Linear(pokemon_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid(),
        )
        
        self.effectiveness_head = nn.Sequential(
            nn.Linear(pokemon_dim + move_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for embed in [self.pokemon_embed, self.move_embed, 
                      self.ability_embed, self.item_embed, self.type_embed]:
            nn.init.normal_(embed.weight, mean=0.0, std=0.1)
            embed.weight.data[0].zero_()
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward_pokemon_types(self, pokemon_idx):
        emb = self.pokemon_embed(pokemon_idx)
        return self.pokemon_type_head(emb)
    
    def forward_move_type(self, move_idx):
        emb = self.move_embed(move_idx)
        return self.move_type_head(emb)
    
    def forward_move_power(self, move_idx):
        emb = self.move_embed(move_idx)
        return self.move_power_head(emb).squeeze(-1)
    
    def forward_move_category(self, move_idx):
        emb = self.move_embed(move_idx)
        return self.move_category_head(emb)
    
    def forward_pokemon_stats(self, pokemon_idx):
        emb = self.pokemon_embed(pokemon_idx)
        return self.pokemon_stats_head(emb)
    
    def forward_effectiveness(self, pokemon_idx, move_idx):
        pokemon_emb = self.pokemon_embed(pokemon_idx)
        move_emb = self.move_embed(move_idx)
        combined = torch.cat([pokemon_emb, move_emb], dim=-1)
        return self.effectiveness_head(combined)
    
    def freeze_embeddings(self):
        """Freeze embedding layers (call before Phase 2)."""
        for embed in [self.pokemon_embed, self.move_embed]:
            for param in embed.parameters():
                param.requires_grad = False
        print("Embeddings frozen")
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding layers."""
        for embed in [self.pokemon_embed, self.move_embed]:
            for param in embed.parameters():
                param.requires_grad = True
        print("Embeddings unfrozen")


def compute_pokemon_type_accuracy(logits, type1_idx, type2_idx, num_types):
    """
    Compute accuracy: do top-k predictions contain the actual types?
    """
    batch_size = logits.size(0)
    correct = 0
    
    probs = torch.sigmoid(logits)
    
    for i in range(batch_size):
        n_types = num_types[i].item()
        top_k = torch.topk(probs[i], k=max(n_types, 2)).indices.tolist()
        
        t1 = type1_idx[i].item()
        t2 = type2_idx[i].item()
        
        if n_types == 1:
            if t1 in top_k[:1]:
                correct += 1
        else:
            if t1 in top_k and t2 in top_k:
                correct += 1
    
    return correct


def phase1_training(
    model: EmbeddingPretrainerV3,
    pokemon_type_loader: DataLoader,
    move_type_loader: DataLoader,
    stats_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    target_pokemon_acc: float = 0.90,
    target_move_acc: float = 0.95,
):
    """Phase 1: Train embeddings to encode type information."""
    print("\n" + "="*60)
    print("PHASE 1: Learning Type Information")
    print("="*60)
    print(f"Target Pokemon Type Acc: {target_pokemon_acc:.0%}")
    print(f"Target Move Type Acc: {target_move_acc:.0%}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    
    best_pokemon_acc = 0
    best_move_acc = 0
    
    for epoch in range(epochs):
        model.train()
        
        pokemon_correct = 0
        pokemon_total = 0
        move_correct = 0
        move_total = 0
        
        # Train Pokemon type prediction
        for batch in pokemon_type_loader:
            pokemon_idx = batch["pokemon_idx"].to(device)
            type_labels = batch["type_labels"].to(device)
            type1_idx = batch["type1_idx"].to(device)
            type2_idx = batch["type2_idx"].to(device)
            num_types = batch["num_types"].to(device)
            
            optimizer.zero_grad()
            logits = model.forward_pokemon_types(pokemon_idx)
            loss = bce_loss_fn(logits, type_labels)
            loss.backward()
            optimizer.step()
            
            pokemon_correct += compute_pokemon_type_accuracy(logits, type1_idx, type2_idx, num_types)
            pokemon_total += len(pokemon_idx)
        
        # Train Move type prediction
        for batch in move_type_loader:
            move_idx = batch["move_idx"].to(device)
            type_idx = batch["type_idx"].to(device)
            power = batch["power"].to(device)
            category_idx = batch["category_idx"].to(device)
            
            optimizer.zero_grad()
            
            type_logits = model.forward_move_type(move_idx)
            type_loss = ce_loss_fn(type_logits, type_idx)
            
            power_pred = model.forward_move_power(move_idx)
            power_loss = mse_loss_fn(power_pred, power)
            
            cat_logits = model.forward_move_category(move_idx)
            cat_loss = ce_loss_fn(cat_logits, category_idx)
            
            loss = type_loss + 0.5 * power_loss + 0.5 * cat_loss
            loss.backward()
            optimizer.step()
            
            move_correct += (type_logits.argmax(dim=-1) == type_idx).sum().item()
            move_total += len(move_idx)
        
        # Train Pokemon stats (auxiliary)
        for batch in stats_loader:
            pokemon_idx = batch["pokemon_idx"].to(device)
            stats = batch["stats"].to(device)
            
            optimizer.zero_grad()
            pred = model.forward_pokemon_stats(pokemon_idx)
            loss = mse_loss_fn(pred, stats)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        pokemon_acc = pokemon_correct / max(pokemon_total, 1)
        move_acc = move_correct / max(move_total, 1)
        
        best_pokemon_acc = max(best_pokemon_acc, pokemon_acc)
        best_move_acc = max(best_move_acc, move_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Pokemon Type Acc: {pokemon_acc:.2%} (best: {best_pokemon_acc:.2%})")
            print(f"  Move Type Acc:    {move_acc:.2%} (best: {best_move_acc:.2%})")
        
        # Early stopping if targets reached
        if pokemon_acc >= target_pokemon_acc and move_acc >= target_move_acc:
            print(f"\n✓ Targets reached at epoch {epoch + 1}!")
            break
    
    print(f"\nPhase 1 Complete:")
    print(f"  Best Pokemon Type Acc: {best_pokemon_acc:.2%}")
    print(f"  Best Move Type Acc: {best_move_acc:.2%}")
    
    return best_pokemon_acc, best_move_acc


def phase2_training(
    model: EmbeddingPretrainerV3,
    eff_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 5e-4,
    freeze_embeddings: bool = True,
):
    """Phase 2: Train effectiveness prediction with frozen embeddings."""
    print("\n" + "="*60)
    print("PHASE 2: Learning Type Effectiveness")
    print("="*60)
    
    if freeze_embeddings:
        model.freeze_embeddings()
        # Only optimize effectiveness head
        optimizer = optim.AdamW(model.effectiveness_head.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    eff_weights = EFFECTIVENESS_WEIGHTS.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=eff_weights)
    
    best_acc = 0
    best_epoch = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        
        correct = 0
        total = 0
        class_correct = [0] * 6
        class_total = [0] * 6
        
        for batch in eff_loader:
            pokemon_idx = batch["pokemon_idx"].to(device)
            move_idx = batch["move_idx"].to(device)
            target = batch["effectiveness_class"].to(device)
            
            optimizer.zero_grad()
            logits = model.forward_effectiveness(pokemon_idx, move_idx)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += len(target)
            
            for c in range(6):
                mask = target == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == target) & mask).sum().item()
        
        scheduler.step()
        
        acc = correct / max(total, 1)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_state = {
                "pokemon_embed": model.pokemon_embed.state_dict(),
                "move_embed": model.move_embed.state_dict(),
                "ability_embed": model.ability_embed.state_dict(),
                "item_embed": model.item_embed.state_dict(),
                "type_embed": model.type_embed.state_dict(),
                "effectiveness_head": model.effectiveness_head.state_dict(),
                "epoch": epoch,
                "accuracy": acc,
            }
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Acc: {acc:.2%} (best: {best_acc:.2%})")
            # Per-class accuracy
            class_names = ["0x", "0.25x", "0.5x", "1x", "2x", "4x"]
            class_accs = [class_correct[i] / max(class_total[i], 1) for i in range(6)]
            print(f"  Per-class: " + " | ".join(f"{class_names[i]}:{class_accs[i]:.0%}" for i in range(6)))
    
    print(f"\nPhase 2 Complete:")
    print(f"  Best Accuracy: {best_acc:.2%} at epoch {best_epoch + 1}")
    
    return best_state, best_acc


def pretrain(
    data_dir: str = "data",
    phase1_epochs: int = 100,
    phase2_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    save_path: str = "models/pretrained_embeddings_v3.pt",
    device: str = None,
    phase1_only: bool = False,
):
    """Run two-phase pretraining."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    pokemon_type_dataset = PokemonTypeDataset(data_dir)
    move_type_dataset = MoveTypeDataset(data_dir)
    stats_dataset = PokemonStatsDataset(data_dir)
    eff_dataset = EffectivenessDataset(data_dir)
    
    pokemon_type_loader = DataLoader(pokemon_type_dataset, batch_size=batch_size, shuffle=True)
    move_type_loader = DataLoader(move_type_dataset, batch_size=batch_size, shuffle=True)
    stats_loader = DataLoader(stats_dataset, batch_size=batch_size, shuffle=True)
    eff_loader = DataLoader(eff_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    print("\n=== Creating Model ===")
    config_path = Path(data_dir) / "embedding_config.json"
    model = EmbeddingPretrainerV3(str(config_path)).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Phase 1: Type learning
    pokemon_acc, move_acc = phase1_training(
        model, pokemon_type_loader, move_type_loader, stats_loader,
        device, epochs=phase1_epochs, lr=lr,
        target_pokemon_acc=0.90, target_move_acc=0.95,
    )
    
    if phase1_only:
        # Save phase 1 checkpoint
        torch.save({
            "pokemon_embed": model.pokemon_embed.state_dict(),
            "move_embed": model.move_embed.state_dict(),
            "ability_embed": model.ability_embed.state_dict(),
            "item_embed": model.item_embed.state_dict(),
            "type_embed": model.type_embed.state_dict(),
            "phase": 1,
            "pokemon_type_acc": pokemon_acc,
            "move_type_acc": move_acc,
        }, save_path.replace(".pt", "_phase1.pt"))
        print(f"\nPhase 1 checkpoint saved to: {save_path.replace('.pt', '_phase1.pt')}")
        return model
    
    # Phase 2: Effectiveness learning
    best_state, eff_acc = phase2_training(
        model, eff_loader, device,
        epochs=phase2_epochs, lr=lr * 0.5,
        freeze_embeddings=True,
    )
    
    # Save final model
    if best_state:
        best_state["pokemon_type_acc"] = pokemon_acc
        best_state["move_type_acc"] = move_acc
        torch.save(best_state, save_path)
        print(f"\nModel saved to: {save_path}")
    
    return model


def verify_embeddings(
    pretrained_path: str = "models/pretrained_embeddings_v3.pt",
    data_dir: str = "data",
):
    """Verify pretrained embeddings."""
    print("\n=== Verifying Pretrained Embeddings ===")
    
    config_path = Path(data_dir) / "embedding_config.json"
    model = EmbeddingPretrainerV3(str(config_path))
    
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    model.pokemon_embed.load_state_dict(checkpoint["pokemon_embed"])
    model.move_embed.load_state_dict(checkpoint["move_embed"])
    
    # Load effectiveness head if available
    if "effectiveness_head" in checkpoint:
        model.effectiveness_head.load_state_dict(checkpoint["effectiveness_head"])
    
    model.eval()
    
    print(f"Loaded checkpoint:")
    print(f"  Pokemon Type Acc: {checkpoint.get('pokemon_type_acc', 'N/A'):.2%}" if isinstance(checkpoint.get('pokemon_type_acc'), float) else f"  Pokemon Type Acc: N/A")
    print(f"  Move Type Acc: {checkpoint.get('move_type_acc', 'N/A'):.2%}" if isinstance(checkpoint.get('move_type_acc'), float) else f"  Move Type Acc: N/A")
    print(f"  Effectiveness Acc: {checkpoint.get('accuracy', 'N/A'):.2%}" if isinstance(checkpoint.get('accuracy'), float) else f"  Effectiveness Acc: N/A")
    
    with open(Path(data_dir) / "pokemon_to_idx.json") as f:
        pokemon_to_idx = json.load(f)
    with open(Path(data_dir) / "move_to_idx.json") as f:
        move_to_idx = json.load(f)
    
    # Test Pokemon type predictions
    print("\n--- Pokemon Type Predictions ---")
    test_pokemon = [
        ("charizard", ["fire", "flying"]),
        ("gyarados", ["water", "flying"]),
        ("gengar", ["ghost", "poison"]),
        ("garchomp", ["dragon", "ground"]),
        ("ferrothorn", ["grass", "steel"]),
    ]
    
    for pokemon, expected_types in test_pokemon:
        pokemon_idx = torch.tensor([pokemon_to_idx.get(pokemon, 0)])
        with torch.no_grad():
            logits = model.forward_pokemon_types(pokemon_idx)
            probs = torch.sigmoid(logits[0])
            top2 = torch.topk(probs, k=2).indices.tolist()
            pred_types = [IDX_TO_TYPE.get(t, "?") for t in top2]
        
        match = set(pred_types) == set(expected_types)
        status = "✓" if match else "✗"
        print(f"{status} {pokemon:12}: pred={pred_types}, expected={expected_types}")
    
    # Test effectiveness predictions
    print("\n--- Effectiveness Predictions ---")
    test_cases = [
        ("charizard", "energyball", "0.25x", 1),
        ("charizard", "hydropump", "2x", 4),
        ("gyarados", "thunderbolt", "4x", 5),
        ("gengar", "tackle", "0x", 0),
        ("blastoise", "thunderbolt", "2x", 4),
        ("garchomp", "icebeam", "4x", 5),
        ("ferrothorn", "flamethrower", "4x", 5),
        ("tyranitar", "closecombat", "4x", 5),
        ("dragonite", "rockslide", "1x", 3),
        ("scizor", "flamethrower", "4x", 5),
    ]
    
    correct = 0
    class_names = ["0x", "0.25x", "0.5x", "1x", "2x", "4x"]
    
    for pokemon, move, expected_mult, expected_class in test_cases:
        pokemon_idx = torch.tensor([pokemon_to_idx.get(pokemon, 0)])
        move_idx = torch.tensor([move_to_idx.get(move, 0)])
        
        with torch.no_grad():
            logits = model.forward_effectiveness(pokemon_idx, move_idx)
        
        pred_class = logits.argmax().item()
        is_correct = pred_class == expected_class
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {pokemon:12} vs {move:12}: pred={class_names[pred_class]:5}, expected={expected_mult}")
    
    print(f"\nEffectiveness Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
    
    if correct >= 7:
        print("✓ Embeddings are working well!")
    elif correct >= 5:
        print("⚠ Embeddings partially working - may still help training")
    else:
        print("✗ Embeddings not learning type relationships properly")


def load_pretrained_into_extractor(extractor, pretrained_path: str):
    """Load pretrained embeddings into HybridEmbeddingExtractor."""
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    
    extractor.pokemon_embed.load_state_dict(checkpoint["pokemon_embed"])
    extractor.move_embed.load_state_dict(checkpoint["move_embed"])
    extractor.ability_embed.load_state_dict(checkpoint["ability_embed"])
    extractor.item_embed.load_state_dict(checkpoint["item_embed"])
    extractor.type_embed.load_state_dict(checkpoint["type_embed"])
    
    print(f"Loaded pretrained embeddings from {pretrained_path}")
    acc = checkpoint.get('accuracy', 0)
    if acc:
        print(f"  (effectiveness accuracy: {acc:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-phase embedding pretraining")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--phase1-epochs", type=int, default=100)
    parser.add_argument("--phase2-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="models/pretrained_embeddings_v3.pt")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.verify:
        verify_embeddings(args.save_path, args.data_dir)
    else:
        pretrain(
            data_dir=args.data_dir,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.save_path,
            device=args.device,
            phase1_only=args.phase1_only,
        )