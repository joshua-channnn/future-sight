from networks.pretrain_embed import TypeEffectivenessDataset

dataset = TypeEffectivenessDataset("data")

class_counts = [0] * 6
for i in range(len(dataset)):
    sample = dataset[i]
    class_idx = sample["effectiveness_class"].item()
    class_counts[class_idx] += 1

print("Class distribution:")
print(f"  0 (immune):  {class_counts[0]:>6} ({class_counts[0]/len(dataset)*100:.1f}%)")
print(f"  1 (0.25x):   {class_counts[1]:>6} ({class_counts[1]/len(dataset)*100:.1f}%)")
print(f"  2 (0.5x):    {class_counts[2]:>6} ({class_counts[2]/len(dataset)*100:.1f}%)")
print(f"  3 (1x):      {class_counts[3]:>6} ({class_counts[3]/len(dataset)*100:.1f}%)")
print(f"  4 (2x):      {class_counts[4]:>6} ({class_counts[4]/len(dataset)*100:.1f}%)")
print(f"  5 (4x):      {class_counts[5]:>6} ({class_counts[5]/len(dataset)*100:.1f}%)")
print(f"\nTotal: {len(dataset)}")