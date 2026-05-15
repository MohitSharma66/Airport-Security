import os
import random
import cv2
import numpy as np

# =====================================================
# CONFIG
# =====================================================

# Folder containing single bag images
INPUT_FOLDER = "datasets/STCray/images/train"

# Output folder
OUTPUT_FOLDER = "bag_queue_test_images"

# Generate:
# 5 images with 2 bags
# 5 images with 3 bags
# 5 images with 4 bags
NUM_IMAGES_PER_GROUP = 5
QUEUE_SIZES = [2, 3, 4]

# Resize settings
TARGET_HEIGHT = 480

# Small spacing for realism
BASE_SPACING = 20

# Blend edges slightly
BLEND_WIDTH = 20

# =====================================================
# SETUP
# =====================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

image_files = [
    os.path.join(INPUT_FOLDER, f)
    for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} bag images")


# =====================================================
# RESIZE FUNCTION
# =====================================================

def resize_to_height(img, target_height):

    h, w = img.shape[:2]

    scale = target_height / h

    new_w = int(w * scale)

    resized = cv2.resize(
        img,
        (new_w, target_height)
    )

    return resized


# =====================================================
# CREATE X-RAY STYLE BACKGROUND
# =====================================================

def create_xray_background(height, width):

    # Dark bluish-gray background feel
    bg = np.random.randint(
        10,
        35,
        (height, width, 3),
        dtype=np.uint8
    )

    # Smooth texture
    bg = cv2.GaussianBlur(bg, (41, 41), 0)

    return bg


# =====================================================
# BLEND BAG INTO BACKGROUND
# =====================================================

def blend_region(base, overlay, x_start):

    h, w = overlay.shape[:2]

    x_end = x_start + w

    # Safety
    if x_end > base.shape[1]:
        return base

    # Paste image
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
# CREATE SIDE-VIEW BAG QUEUE
# =====================================================

def create_bag_queue(image_paths):

    resized_images = []

    total_width = 0

    # Resize all bag images
    for path in image_paths:

        img = cv2.imread(path)

        if img is None:
            continue

        img = resize_to_height(
            img,
            TARGET_HEIGHT
        )

        resized_images.append(img)

        total_width += img.shape[1]

    # Add spacing
    total_width += BASE_SPACING * (
        len(resized_images) - 1
    )

    # Create background
    canvas = create_xray_background(
        TARGET_HEIGHT,
        total_width
    )

    current_x = 0

    for img in resized_images:

        h, w = img.shape[:2]

        # Tiny random overlap
        overlap_shift = random.randint(0, 8)

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
# GENERATE BAG QUEUES
# =====================================================

for queue_size in QUEUE_SIZES:

    print(f"\nGenerating {queue_size}-bag scenes...")

    for i in range(NUM_IMAGES_PER_GROUP):

        selected = random.sample(
            image_files,
            queue_size
        )

        queue_img = create_bag_queue(
            selected
        )

        filename = (
            f"bags_{queue_size}people_{i+1}.jpg"
        )

        save_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        cv2.imwrite(
            save_path,
            queue_img
        )

        print(f"Saved: {save_path}")

print("\nAll bag queue images generated!")