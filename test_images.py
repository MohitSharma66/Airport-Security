import os
import random
import cv2
import numpy as np

# =====================================================
# CONFIG
# =====================================================

INPUT_FOLDER = "datasets/Thermal_Overlap/images/train"
OUTPUT_FOLDER = "queue_test_images"

NUM_IMAGES_PER_GROUP = 5
QUEUE_SIZES = [2, 3, 4]

TARGET_HEIGHT = 640

# Smaller spacing = more realistic
BASE_SPACING = 15

# Blend width between images
BLEND_WIDTH = 25

# =====================================================
# SETUP
# =====================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

image_files = [
    os.path.join(INPUT_FOLDER, f)
    for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} images")


# =====================================================
# FUNCTION: RESIZE IMAGE
# =====================================================

def resize_to_height(img, target_height):

    h, w = img.shape[:2]

    scale = target_height / h

    new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, target_height))

    return resized


# =====================================================
# FUNCTION: CREATE THERMAL-LIKE BACKGROUND
# =====================================================

def create_thermal_background(height, width):

    # Base gray thermal tone
    bg = np.random.randint(
        25,
        45,
        (height, width, 3),
        dtype=np.uint8
    )

    # Smooth noise
    bg = cv2.GaussianBlur(bg, (31, 31), 0)

    return bg


# =====================================================
# FUNCTION: BLEND IMAGES
# =====================================================

def blend_region(base, overlay, x_start):

    h, w = overlay.shape[:2]

    x_end = x_start + w

    # Normal paste
    base[:, x_start:x_end] = overlay

    # Blend left edge
    if x_start > 0:

        blend_start = max(0, x_start - BLEND_WIDTH)

        for i in range(BLEND_WIDTH):

            alpha = i / BLEND_WIDTH

            bx = blend_start + i

            if bx >= base.shape[1]:
                break

            base[:, bx] = (
                (1 - alpha) * base[:, bx] +
                alpha * overlay[:, i]
            ).astype(np.uint8)

    return base


# =====================================================
# FUNCTION: STITCH REALISTIC QUEUE
# =====================================================

def create_queue_scene(image_paths):

    resized_images = []

    total_width = 0

    # Resize images
    for path in image_paths:

        img = cv2.imread(path)

        if img is None:
            continue

        img = resize_to_height(img, TARGET_HEIGHT)

        resized_images.append(img)

        total_width += img.shape[1]

    # Add spacing
    total_width += BASE_SPACING * (len(resized_images) - 1)

    # Create realistic thermal background
    canvas = create_thermal_background(
        TARGET_HEIGHT,
        total_width
    )

    current_x = 0

    for img in resized_images:

        h, w = img.shape[:2]

        # Slight overlap for realism
        overlap_shift = random.randint(0, 10)

        current_x -= overlap_shift

        current_x = max(current_x, 0)

        canvas = blend_region(
            canvas,
            img,
            current_x
        )

        current_x += w + BASE_SPACING

    return canvas


# =====================================================
# GENERATE DATASET
# =====================================================

for queue_size in QUEUE_SIZES:

    print(f"\nGenerating {queue_size}-person queues...")

    for i in range(NUM_IMAGES_PER_GROUP):

        selected = random.sample(
            image_files,
            queue_size
        )

        queue_img = create_queue_scene(selected)

        filename = f"queue_{queue_size}people_{i+1}.jpg"

        save_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        cv2.imwrite(save_path, queue_img)

        print(f"Saved: {save_path}")

print("\nAll queue images generated successfully!")