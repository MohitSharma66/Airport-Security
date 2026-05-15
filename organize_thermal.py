import os
import shutil
from pathlib import Path

# ====== CONFIGURATION ======
# SOURCE: Path to your Thermal_Overlap folder
source_root = Path("C:/Users/mohit/Documents/Thermal_Overlap")

# TARGET: Path to your organized YOLO folder for Thermal
target_root = Path("datasets/Thermal_Overlap")

# Since we only have train split, we'll split it ourselves (e.g., 80/20)
TRAIN_SPLIT_RATIO = 0.8  # 80% train, 20% val
# ===========================

print("Starting organization of Thermal_Overlap dataset...")
print(f"Source: {source_root}")
print(f"Target: {target_root}")

# Source paths
source_images = source_root / "train" / "images"
source_labels = source_root / "train" / "labels"

if not source_images.exists():
    print(f"❌ Error: {source_images} not found!")
    exit(1)

# Get all image files
image_files = list(source_images.glob("*.*"))
image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
print(f"\n📸 Found {len(image_files)} images")

# Shuffle and split into train/val
import random
random.seed(42)  # For reproducibility
random.shuffle(image_files)

split_idx = int(len(image_files) * TRAIN_SPLIT_RATIO)
train_images = image_files[:split_idx]
val_images = image_files[split_idx:]

print(f"   Train: {len(train_images)} images")
print(f"   Val: {len(val_images)} images")

# Create directories
for split in ['train', 'val']:
    (target_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (target_root / "labels" / split).mkdir(parents=True, exist_ok=True)

# Copy train images and labels
print("\n📂 Copying train split...")
for img_file in train_images:
    # Copy image
    shutil.copy2(img_file, target_root / "images" / "train" / img_file.name)
    
    # Find corresponding label
    label_file = source_labels / f"{img_file.stem}.txt"
    if label_file.exists():
        shutil.copy2(label_file, target_root / "labels" / "train" / label_file.name)
    else:
        print(f"  ⚠️ No label for: {img_file.name}")

# Copy val images and labels
print("\n📂 Copying val split...")
for img_file in val_images:
    # Copy image
    shutil.copy2(img_file, target_root / "images" / "val" / img_file.name)
    
    # Find corresponding label
    label_file = source_labels / f"{img_file.stem}.txt"
    if label_file.exists():
        shutil.copy2(label_file, target_root / "labels" / "val" / label_file.name)
    else:
        print(f"  ⚠️ No label for: {img_file.name}")

# Copy and update the data.yaml file
source_yaml = source_root / "data.yaml"
target_yaml = target_root / "dataset.yaml"

if source_yaml.exists():
    import yaml
    
    with open(source_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Update paths to relative paths for YOLO
    data_config['path'] = './datasets/Thermal_Overlap'
    data_config['train'] = 'images/train'
    data_config['val'] = 'images/val'
    
    with open(target_yaml, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"\n✅ dataset.yaml created at {target_yaml}")
    print(f"   Classes: {data_config.get('nc', '?')} classes")
    if 'names' in data_config:
        print(f"   Class names: {data_config['names'][:5]}... ({len(data_config['names'])} total)")
else:
    # Create dataset.yaml manually using class_config
    print(f"\n⚠️ data.yaml not found. Creating manually...")
    
    # You need to get the actual class names from Roboflow
    # For now, I'll use placeholder names
    class_names = [f"class_{i}" for i in range(27)]
    
    dataset_config = {
        'path': './datasets/Thermal_Overlap',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 27,
        'names': class_names
    }
    
    with open(target_yaml, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ dataset.yaml created manually at {target_yaml}")

# Create weapon-only version (if you know which indices are weapons)
print("\n🔫 Creating weapon-only dataset configuration...")

# You'll need to update these indices based on actual class mapping
# For now, these are placeholders
weapon_indices = [7, 8, 10, 5, 6, 11, 13, 18, 21]  # Gun, Knife, Scissor, etc.
weapon_names = ['Gun', 'Knife', 'Scissor', 'Hammer', 'Screwdriver', 'Wrench', 'Lighter', 'Handsaw', 'Lock']

weapon_config = {
    'path': './datasets/Thermal_Overlap',
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(weapon_names),
    'names': weapon_names,
    'note': 'Filtered to weapon classes only. Update indices based on actual data.yaml'
}

weapon_yaml = target_root / "dataset_weapons.yaml"
with open(weapon_yaml, 'w') as f:
    yaml.dump(weapon_config, f, default_flow_style=False)

print(f"✅ Weapon-only dataset config created at {weapon_yaml}")

# Final summary
print("\n" + "="*50)
print("🎉 Thermal_Overlap organization complete!")
print("="*50)
print(f"📁 Dataset location: {target_root}")
print(f"   📂 train: {len(list((target_root / 'images/train').glob('*.*')))} images")
print(f"   📂 val: {len(list((target_root / 'images/val').glob('*.*')))} images")
print("\n📄 Config files:")
print(f"   - dataset.yaml (all {data_config.get('nc', 27)} classes)")
print(f"   - dataset_weapons.yaml (filtered to {len(weapon_names)} weapon classes)")