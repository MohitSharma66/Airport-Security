# augment_thermal_dataset.py - FIXED VERSION
import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil
from tqdm import tqdm
import random

# ====== CONFIGURATION ======
SOURCE_DIR = Path("datasets/Thermal_Overlap")
TARGET_DIR = Path("datasets/Thermal_Overlap_Augmented")

# Augmentation multiplier (how many variations per image)
AUGMENTATIONS_PER_IMAGE = 5  # 5x dataset size

# Classes to keep
FOCUS_CLASSES = {
    0: "Gun",
    1: "Knife",
    2: "Scissor",
    3: "Hammer",
    4: "Screwdriver",
    5: "Wrench",
    6: "Lighter",
    7: "Handsaw",
    8: "Lock"
}

# ====== DEFINE AUGMENTATIONS (FIXED) ======
def get_augmentation_pipeline():
    """Define augmentation transformations - Fixed for recent albumentations"""
    return A.Compose([
        # Geometric transformations
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        
        # Scaling (replaced RandomSizedCrop with RandomResizedCrop)
        A.RandomResizedCrop(height=640, width=640, scale=(0.6, 1.0), p=0.4),
        
        # Color/Intensity transformations (important for thermal)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        
        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        
        # Blur (your teacher mentioned this)
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.MedianBlur(blur_limit=5, p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
        
        # Other augmentations
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        # Additional augmentations for thermal
        A.RandomToneCurve(scale=0.1, p=0.3),
        
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# ====== SIMPLIFIED AUGMENTATION (MORE RELIABLE) ======
def get_simple_augmentation_pipeline():
    """Simpler augmentation that won't cause errors"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.4),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# ====== MAIN AUGMENTATION FUNCTION ======
def augment_dataset():
    """Generate augmented dataset"""
    
    print("=" * 60)
    print("🔥 THERMAL DATASET AUGMENTATION")
    print("=" * 60)
    
    # Check if source exists
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        print("   Please make sure you ran organize_thermal.py first")
        return
    
    # Create target directories
    for split in ['train', 'val']:
        (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Use simpler augmentation pipeline (more reliable)
    aug_pipeline = get_simple_augmentation_pipeline()
    
    total_original = 0
    total_augmented = 0
    
    # Process train and validation splits
    for split in ['train', 'val']:
        print(f"\n📂 Processing {split} split...")
        
        source_images = SOURCE_DIR / "images" / split
        source_labels = SOURCE_DIR / "labels" / split
        
        if not source_images.exists():
            print(f"   ⚠️ {source_images} not found, skipping")
            continue
        
        image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png")) + list(source_images.glob("*.jpeg"))
        print(f"   Found {len(image_files)} original images")
        
        # Copy original images (without augmentation)
        print("   Copying original images...")
        for img_file in tqdm(image_files, desc="   Copying originals"):
            # Copy image
            shutil.copy2(img_file, TARGET_DIR / "images" / split / img_file.name)
            
            # Copy corresponding label
            label_file = source_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, TARGET_DIR / "labels" / split / label_file.name)
            
            total_original += 1
        
        # Generate augmented versions
        print(f"   Generating {AUGMENTATIONS_PER_IMAGE}x augmented images...")
        
        for img_file in tqdm(image_files, desc="   Augmenting"):
            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # Resize to 640x640 if needed
            if image.shape[0] != 640 or image.shape[1] != 640:
                image = cv2.resize(image, (640, 640))
            
            # Read labels (YOLO format: class x_center y_center width height)
            label_file = source_labels / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                labels = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append([class_id, x_center, y_center, width, height])
            
            if len(labels) == 0:
                continue
            
            # Generate multiple augmented versions
            success_count = 0
            for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
                try:
                    # Prepare for augmentation
                    bboxes = [[x_center, y_center, width, height] for _, x_center, y_center, width, height in labels]
                    class_labels = [class_id for class_id, _, _, _, _ in labels]
                    
                    # Apply augmentation
                    augmented = aug_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    
                    # Skip if no bounding boxes remain
                    if len(aug_bboxes) == 0:
                        continue
                    
                    # Save augmented image
                    aug_filename = f"{img_file.stem}_aug{aug_idx:02d}{img_file.suffix}"
                    aug_image_path = TARGET_DIR / "images" / split / aug_filename
                    cv2.imwrite(str(aug_image_path), aug_image)
                    
                    # Save augmented labels
                    aug_label_path = TARGET_DIR / "labels" / split / f"{img_file.stem}_aug{aug_idx:02d}.txt"
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_id in zip(aug_bboxes, aug_labels):
                            x_c, y_c, w, h = bbox
                            # Ensure values are within [0,1]
                            x_c = max(0, min(1, x_c))
                            y_c = max(0, min(1, y_c))
                            w = max(0, min(1, w))
                            h = max(0, min(1, h))
                            f.write(f"{int(class_id)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    
                    success_count += 1
                    total_augmented += 1
                    
                except Exception as e:
                    # Silently skip failed augmentations
                    continue
            
            # If no augmentations succeeded, create simple flips
            if success_count == 0:
                print(f"   ⚠️ Using simple augmentation for {img_file.name}")
                # Simple horizontal flip
                aug_image = cv2.flip(image, 1)
                aug_filename = f"{img_file.stem}_flip{img_file.suffix}"
                cv2.imwrite(str(TARGET_DIR / "images" / split / aug_filename), aug_image)
                
                # Adjust bounding boxes for flip
                with open(TARGET_DIR / "labels" / split / f"{img_file.stem}_flip.txt", 'w') as f:
                    for class_id, x_c, y_c, w, h in labels:
                        x_c_flip = 1 - x_c
                        f.write(f"{int(class_id)} {x_c_flip:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                total_augmented += 1
    
    # ====== CREATE DATASET YAML ======
    print("\n📝 Creating dataset configuration...")
    
    dataset_config = {
        'path': './datasets/Thermal_Overlap_Augmented',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 9,
        'names': list(FOCUS_CLASSES.values())
    }
    
    try:
        import yaml
        with open(TARGET_DIR / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        print("   ✅ dataset.yaml created")
    except Exception as e:
        print(f"   ⚠️ Could not create yaml: {e}")
        # Write manually
        with open(TARGET_DIR / 'dataset.yaml', 'w') as f:
            f.write(f"path: {dataset_config['path']}\n")
            f.write(f"train: {dataset_config['train']}\n")
            f.write(f"val: {dataset_config['val']}\n")
            f.write(f"nc: {dataset_config['nc']}\n")
            f.write("names:\n")
            for name in dataset_config['names']:
                f.write(f"  - {name}\n")
    
    # ====== SUMMARY ======
    print("\n" + "=" * 60)
    print("✅ AUGMENTATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Original images:  {total_original}")
    print(f"📊 Augmented images: {total_augmented}")
    print(f"📊 Total images:     {total_original + total_augmented}")
    print(f"📁 Output directory: {TARGET_DIR}")
    print("=" * 60)
    
    # Count class distribution
    print("\n📈 Class distribution in augmented dataset (train split):")
    class_counts = {name: 0 for name in FOCUS_CLASSES.values()}
    
    train_labels_dir = TARGET_DIR / "labels" / "train"
    if train_labels_dir.exists():
        for label_file in train_labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id in FOCUS_CLASSES:
                                class_counts[FOCUS_CLASSES[class_id]] += 1
            except:
                pass
        
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} instances")

if __name__ == "__main__":
    # Install albumentations if not installed
    try:
        import albumentations
        import yaml
        print(f"✅ Albumentations version: {albumentations.__version__}")
    except ImportError:
        print("Installing albumentations...")
        os.system("pip install albumentations tqdm pyyaml")
        import albumentations
        import yaml
    
    augment_dataset()
    
    print("\n🚀 Next steps:")
    print("   1. Train on augmented dataset:")
    print("      yolo detect train data=datasets/Thermal_Overlap_Augmented/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8 name=thermal_augmented")
    print("\n   2. Or train with just 50 epochs first to test:")
    print("      yolo detect train data=datasets/Thermal_Overlap_Augmented/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8 name=thermal_augmented_test")