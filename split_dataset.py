import os
import random
import shutil
from pathlib import Path

# Paths
dataset_root = Path("datasets/STCray")
train_images = dataset_root / "images" / "train"
train_labels = dataset_root / "labels" / "train"

val_images = dataset_root / "images" / "val"
val_labels = dataset_root / "labels" / "val"

# Create validation folders
val_images.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

# Get all image files
image_files = list(train_images.glob("*.*"))
print(f"Total images: {len(image_files)}")

# Shuffle and split (80% train, 20% val)
random.seed(42)  # For reproducibility
random.shuffle(image_files)

split_idx = int(0.8 * len(image_files))
val_files = image_files[split_idx:]

print(f"Moving {len(val_files)} images to validation set...")

# Move files to validation folders
for img_path in val_files:
    # Move image
    target_img = val_images / img_path.name
    shutil.move(str(img_path), str(target_img))
    
    # Move corresponding label
    label_name = img_path.stem + ".txt"
    label_path = train_labels / label_name
    target_label = val_labels / label_name
    
    if label_path.exists():
        shutil.move(str(label_path), str(target_label))
    else:
        print(f"Warning: No label found for {img_path.name}")

print(f"\nFinal counts:")
print(f"Training images: {len(list(train_images.glob('*.*')))}")
print(f"Validation images: {len(list(val_images.glob('*.*')))}")
print(f"Training labels: {len(list(train_labels.glob('*.txt')))}")
print(f"Validation labels: {len(list(val_labels.glob('*.txt')))}")