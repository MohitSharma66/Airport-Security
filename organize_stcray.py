import os
import shutil
from pathlib import Path

# ====== CONFIGURATION ======
# SOURCE: Path to your main downloaded 'train' folder
source_root = Path("C:/Users/YourName/Downloads/STCray/train")  # <<< CHANGE THIS

# TARGET: Path to your new organized YOLO folder
target_root = Path("datasets/STCray")  # <<< Ensure this is correct

# Define class names and their YOLO IDs based on your folder list
# IMPORTANT: This order MUST match your final dataset.yaml file
class_config = {
    "Class 1_Explosive": 0,
    "Class 2_Gun": 1,
    "Class_3_3D Gun": 2,
    "Class 4_Knife": 3,
    "Class_5_Cutter": 4,
    "Class 6_Blade": 5,
    "Class_7_Shaving Razor": 6,
    "Class 8_Lighter": 7,
    "Class_9_Injection": 8,
    "Class 10_Battery": 9,
    "Class_11_Nail Cutter": 10,
    "Class 12_Other Sharp Item": 11,
    "Class_13_Powerbank": 12,
    "Class 14_Scissors": 13,
    "Class_15_Hammer": 14,
    "Class 16_Pliers": 15,
    "Class_17_Wrench": 16,
    "Class 18_Screwdriver": 17,
    "Class_19_Handcuffs": 18,
    "Class 20_Bullet": 19,
    "Class_21_Multilabel Threat": 20,
    "Class 22_Non Threat": 21,
}
# ===========================

# Create target directories
images_target = target_root / "images" / "train"
labels_target = target_root / "labels" / "train"
images_target.mkdir(parents=True, exist_ok=True)
labels_target.mkdir(parents=True, exist_ok=True)

# Track processed files to handle duplicates (same image in multiple classes)
processed_images = set()
label_file_extensions = ['.txt', '.json']  # Add other extensions if needed

print("Starting organization...")

for class_folder_name, yolo_class_id in class_config.items():
    class_folder_path = source_root / class_folder_name
    
    if not class_folder_path.exists():
        print(f"Warning: Folder {class_folder_path} not found. Skipping.")
        continue
    
    print(f"Processing: {class_folder_name} -> YOLO class {yolo_class_id}")
    
    # Process each file in the class folder
    for item in class_folder_path.iterdir():
        if item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # This is an image file
            target_image_path = images_target / item.name
            
            # Handle duplicates: If image already copied, skip copying again
            if item.name not in processed_images:
                shutil.copy2(item, target_image_path)
                processed_images.add(item.name)
            
            # Now look for its corresponding label file
            label_found = False
            for ext in label_file_extensions:
                # Look for label in the same class folder (common structure)
                possible_label = class_folder_path / f"{item.stem}{ext}"
                if possible_label.exists():
                    # Create YOLO format label file
                    target_label_path = labels_target / f"{item.stem}.txt"
                    
                    # Read the original label and convert to YOLO format
                    # ASSUMPTION: Original label is already in YOLO format (class x_center y_center width height)
                    # If it's in a different format, this part needs adjustment
                    with open(possible_label, 'r') as src_file, open(target_label_path, 'a') as tgt_file:
                        for line in src_file:
                            parts = line.strip().split()
                            if len(parts) == 5:  # Standard YOLO format
                                # Replace original class ID with current yolo_class_id
                                parts[0] = str(yolo_class_id)
                                tgt_file.write(' '.join(parts) + '\n')
                    
                    label_found = True
                    break
            
            if not label_found:
                print(f"  Warning: No label file found for {item.name}")
    
    print(f"  Completed: {class_folder_name}")

print(f"\nOrganization complete!")
print(f"Total unique images copied: {len(processed_images)}")
print(f"Images in: {images_target}")
print(f"Labels in: {labels_target}")