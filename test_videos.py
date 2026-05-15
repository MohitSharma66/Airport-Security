import os
import cv2
import random
import numpy as np

# =====================================================
# CONFIG
# =====================================================

BASE_FOLDER = "thermal_videos"

GUN_FOLDER = os.path.join(BASE_FOLDER, "Handgun")
NO_GUN_FOLDER = os.path.join(BASE_FOLDER, "No_Gun")

OUTPUT_FOLDER = "generated_queue_videos"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VIDEOS_PER_GROUP = 5
GROUP_SIZES = [2, 3, 4]

# =====================================================
# VIDEO SETTINGS
# =====================================================

OUTPUT_HEIGHT = 720

# No gap between videos
BASE_SPACING = 0

# Slight realism
MAX_Y_OFFSET = 15

# Slight size variation
MIN_SCALE = 0.95
MAX_SCALE = 1.05

VALID_EXTENSIONS = (".mp4", ".avi", ".mov")

# =====================================================
# LOAD VIDEOS RECURSIVELY
# =====================================================

def get_all_videos(folder):

    video_paths = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(VALID_EXTENSIONS):

                full_path = os.path.join(root, file)

                video_paths.append(full_path)

    return video_paths


gun_videos = get_all_videos(GUN_FOLDER)
no_gun_videos = get_all_videos(NO_GUN_FOLDER)

print(f"Gun videos found: {len(gun_videos)}")
print(f"No-gun videos found: {len(no_gun_videos)}")

if len(gun_videos) == 0:
    raise Exception("No handgun videos found!")

if len(no_gun_videos) == 0:
    raise Exception("No no-gun videos found!")

# =====================================================
# RESIZE FRAME
# =====================================================

def resize_frame(frame, target_height):

    h, w = frame.shape[:2]

    scale = target_height / h

    new_w = int(w * scale)

    resized = cv2.resize(frame, (new_w, target_height))

    return resized

# =====================================================
# THERMAL BACKGROUND
# =====================================================

def create_background(height, width):

    bg = np.random.randint(
        20,
        40,
        (height, width, 3),
        dtype=np.uint8
    )

    bg = cv2.GaussianBlur(bg, (41, 41), 0)

    return bg

# =====================================================
# VIDEO STITCHER
# =====================================================

def create_stitched_video(video_paths, output_path):

    caps = [cv2.VideoCapture(v) for v in video_paths]

    fps = int(caps[0].get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 30

    frame_counts = [
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for cap in caps
    ]

    min_frames = min(frame_counts)

    # =================================================
    # RANDOM PERSON HEIGHTS
    # =================================================

    person_heights = []

    for _ in video_paths:

        scale = random.uniform(
            MIN_SCALE,
            MAX_SCALE
        )

        height = int(OUTPUT_HEIGHT * scale)

        person_heights.append(height)

    # =================================================
    # GET WIDTHS
    # =================================================

    widths = []

    for idx, cap in enumerate(caps):

        ret, frame = cap.read()

        if not ret:
            continue

        resized = resize_frame(
            frame,
            person_heights[idx]
        )

        widths.append(resized.shape[1])

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # =================================================
    # CANVAS SIZE
    # =================================================

    total_width = sum(widths)

    canvas_height = OUTPUT_HEIGHT + 60

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (total_width, canvas_height)
    )

    # =================================================
    # RANDOM DELAYS
    # =================================================

    delays = [
        random.randint(0, fps // 2)
        for _ in video_paths
    ]

    y_offsets = [
        random.randint(0, MAX_Y_OFFSET)
        for _ in video_paths
    ]

    print(f"\nGenerating: {output_path}")

    # =================================================
    # FRAME LOOP
    # =================================================

    for frame_idx in range(min_frames):

        canvas = create_background(
            canvas_height,
            total_width
        )

        current_x = 0

        for i, cap in enumerate(caps):

            target_frame = frame_idx - delays[i]

            # Empty space until person enters
            if target_frame < 0:

                current_x += widths[i]

                continue

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                target_frame
            )

            ret, frame = cap.read()

            if not ret:

                current_x += widths[i]

                continue

            frame = resize_frame(
                frame,
                person_heights[i]
            )

            h, w = frame.shape[:2]

            y = canvas_height - h - y_offsets[i]

            # =================================================
            # SAFETY FIXES
            # =================================================

            # Prevent negative y
            if y < 0:
                y = 0

            # Prevent height overflow
            if y + h > canvas_height:

                h = canvas_height - y

                frame = frame[:h, :]

            # Prevent width overflow
            if current_x + w > total_width:

                w = total_width - current_x

                frame = frame[:, :w]

            # Skip invalid frame
            if h <= 0 or w <= 0:
                continue

            # =================================================
            # PLACE FRAME
            # =================================================

            canvas[
                y:y+h,
                current_x:current_x+w
            ] = frame

            current_x += w + BASE_SPACING

        writer.write(canvas)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{min_frames}")

    # =================================================
    # CLEANUP
    # =================================================

    writer.release()

    for cap in caps:
        cap.release()

    print(f"Saved: {output_path}")

# =====================================================
# VIDEO SELECTION
# =====================================================

def choose_videos(group_size):

    num_gun = group_size // 2

    num_no_gun = group_size - num_gun

    selected_gun = random.sample(
        gun_videos,
        num_gun
    )

    selected_no_gun = random.sample(
        no_gun_videos,
        num_no_gun
    )

    combined = selected_gun + selected_no_gun

    random.shuffle(combined)

    return combined

# =====================================================
# GENERATE DATASET
# =====================================================

for group_size in GROUP_SIZES:

    print(f"\nGenerating {group_size}-person videos...")

    for i in range(VIDEOS_PER_GROUP):

        selected_videos = choose_videos(group_size)

        print("\nSelected Videos:")

        for vid in selected_videos:
            print(vid)

        output_name = (
            f"queue_{group_size}people_{i+1}.mp4"
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_name
        )

        create_stitched_video(
            selected_videos,
            output_path
        )

print("\nAll stitched videos generated successfully!")