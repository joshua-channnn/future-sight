import json

# Load randbats sets data
with open("data/random-battles/gen9/sets.json", "r") as f:
    randbats_data = json.load(f)

# Load items data
with open("gen9_randbats_items_100000.json", "r") as f:
    items_data = json.load(f)

# Collect entities
pokemon_set = set()
moves_set = set()
abilities_set = set()

for pokemon_name, data in randbats_data.items():
    pokemon_set.add(pokemon_name)
    for set_data in data.get("sets", []):
        for move in set_data.get("movepool", []):
            moves_set.add(move)
        for ability in set_data.get("abilities", []):
            abilities_set.add(ability)

items_set = set(items_data["itemCounts"].keys())

# Create index mappings (0 = unknown)
def normalize(name):
    """Convert to poke-env ID format"""
    return name.lower().replace(" ", "").replace("-", "").replace("'", "")

def create_mapping(entities, name):
    mapping = {"<UNK>": 0}
    for i, entity in enumerate(sorted(entities), start=1):
        mapping[normalize(entity)] = i
    
    with open(f"data/{name}_to_idx.json", "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"{name}: {len(mapping)} entries")
    return mapping

import os
os.makedirs("data", exist_ok=True)

pokemon_to_idx = create_mapping(pokemon_set, "pokemon")
move_to_idx = create_mapping(moves_set, "move")
ability_to_idx = create_mapping(abilities_set, "ability")
item_to_idx = create_mapping(items_set, "item")

# Also save vocab sizes for model config
config = {
    "pokemon_vocab_size": len(pokemon_to_idx),
    "move_vocab_size": len(move_to_idx),
    "ability_vocab_size": len(ability_to_idx),
    "item_vocab_size": len(item_to_idx),
    "pokemon_embed_dim": 64,
    "move_embed_dim": 32,
    "ability_embed_dim": 32,
    "item_embed_dim": 16,
}

with open("data/embedding_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\nConfig: {config}")