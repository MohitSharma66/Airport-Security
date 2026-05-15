import json
import shutil
from pathlib import Path

# ====== CONFIGURATION ======
source_root = Path("C:/Users/mohit/Documents/STCray_TrainSet")  # Your TrainSet folder
target_root = Path("datasets/STCray")  # Target YOLO folder

# Class mapping - VERIFY THESE NAMES IN YOUR JSON FILES
class_mapping = {
    "Explosive": 0,
    "Gun": 1,
    "3D Gun": 2,
    "Knife": 3,
    "Cutter": 4,
    "Blade": 5,
    "Shaving Razor": 6,
    "Lighter": 7,
    "Injection": 8,
    "Battery": 9,
    "Nail Cutter": 10,
    "Other Sharp Item": 11,
    "Powerbank": 12,
    "Scissors": 13,
    "Hammer": 14,
    "Pliers": 15,
    "Wrench": 16,
    "Screwdriver": 17,
    "Handcuffs": 18,
    "Bullet": 19,
    "Multilabel Threat": 20,
    "Non Threat": 21
}
# ===========================

# Create target directories
images_target = target_root / "images" / "train"
labels_target = target_root / "labels" / "train"
images_target.mkdir(parents=True, exist_ok=True)
labels_target.mkdir(parents=True, exist_ok=True)

print(f"Source: {source_root}")
print(f"Target images: {images_target}")
print(f"Target labels: {labels_target}")

# Track progress
processed_images = set()
total_annotations = 0
errors = []

# Get all class folders from Images directory
images_dir = source_root / "Images"
if not images_dir.exists():
    print(f"ERROR: Images directory not found at {images_dir}")
    exit(1)

class_folders = [f for f in images_dir.iterdir() if f.is_dir() and f.name.startswith("Class")]
print(f"Found {len(class_folders)} class folders in Images")

for class_folder in class_folders:
    class_name = class_folder.name
    print(f"\nProcessing: {class_name}")
    
    # Get corresponding Json_BB folder
    jsonbb_class_dir = source_root / "Json_BB" / class_name
    if not jsonbb_class_dir.exists():
        print(f"  Warning: No Json_BB folder for {class_name} at {jsonbb_class_dir}")
        continue
    
    # Process each image in the class folder
    image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.png"))
    print(f"  Found {len(image_files)} images")
    
    for img_idx, image_path in enumerate(image_files):
        if img_idx % 100 == 0 and img_idx > 0:
            print(f"    Processed {img_idx} images...")
        
        image_name = image_path.name
        image_stem = image_path.stem
        
        # Find corresponding JSON file
        json_path = jsonbb_class_dir / f"{image_stem}.json"
        
        if not json_path.exists():
            # Try alternative naming (some datasets use _BB suffix)
            alt_json_path = jsonbb_class_dir / f"{image_stem}_BB.json"
            if alt_json_path.exists():
                json_path = alt_json_path
            else:
                errors.append(f"No JSON for {image_name} in {class_name}")
                continue
        
        # Load JSON data
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON {json_path.name}: {e}")
            continue
        
        # Get image dimensions
        image_height = data.get("imageHeight")
        image_width = data.get("imageWidth")
        
        if not image_height or not image_width:
            # Try to extract from image name or use default
            image_height = 330  # Default from your example
            image_width = 608   # Default from your example
            print(f"    Warning: Using default dimensions for {image_name}")
        
        # Copy image to target
        target_image_path = images_target / image_name
        
        # Handle duplicate filenames (if same image appears in multiple classes)
        if image_name in processed_images:
            # Add class prefix to avoid overwriting
            unique_name = f"{class_name}_{image_name}"
            target_image_path = images_target / unique_name
            image_stem = unique_name.rsplit('.', 1)[0]
        
        shutil.copy2(image_path, target_image_path)
        processed_images.add(image_name)
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            shape_type = shape.get("shape_type", "")
            points = shape.get("points", [])
            
            # Skip if not rectangle or label not in mapping
            if shape_type != "rectangle":
                continue
            
            if label not in class_mapping:
                print(f"    Warning: Unknown label '{label}' in {json_path.name}")
                # Try to add it dynamically
                if label not in class_mapping and label:  # If new label found
                    new_id = len(class_mapping)
                    class_mapping[label] = new_id
                    print(f"    Added new label '{label}' as class {new_id}")
            
            if label not in class_mapping:
                continue
            
            # Convert rectangle points to YOLO format
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Convert to YOLO format (normalized 0-1)
                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = abs(x2 - x1) / image_width
                height = abs(y2 - y1) / image_height
                
                # Clamp to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                class_id = class_mapping[label]
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
                total_annotations += 1
        
        # Write YOLO label file
        if yolo_lines:
            label_filename = f"{image_stem}.txt"
            target_label_path = labels_target / label_filename
            
            with open(target_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
        else:
            errors.append(f"No valid annotations in {json_path.name}")

print(f"\n{'='*50}")
print("CONVERSION SUMMARY")
print(f"{'='*50}")
print(f"Total images processed: {len(processed_images)}")
print(f"Total annotations converted: {total_annotations}")
print(f"Target images folder: {images_target}")
print(f"Target labels folder: {labels_target}")

# Show final class mapping
print(f"\nFinal class mapping (update your dataset.yaml with this):")
for label, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
    print(f"  {class_id}: {label}")

if errors:
    print(f"\nWarnings/Errors ({len(errors)}):")
    for i, error in enumerate(errors[:10]):  # Show first 10 errors
        print(f"  {i+1}. {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors)-10} more")

# Verify with a few files
print(f"\nVerification - checking 3 random label files:")
label_files = list(labels_target.glob("*.txt"))[:3]
for label_file in label_files:
    print(f"\n{label_file.name}:")
    try:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if content:
                for line in content.split('\n'):
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_name = [k for k, v in class_mapping.items() if v == class_id]
                        print(f"  Class {class_id} ({class_name[0] if class_name else '?'}): {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            else:
                print("  (empty file)")
    except Exception as e:
        print(f"  Error reading: {e}")