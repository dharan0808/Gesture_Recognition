import cv2
import os
import numpy as np

# ----- CONFIG -----
video_path = 'sample_gesture.mp4'     # Your test video
output_dir = 'test_clip'              # Output folder with 20 frames
target_frames = 20                    # Fixed frame count

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames < target_frames:
    raise ValueError(f"Video has only {total_frames} frames, expected at least {target_frames}.")

# Compute frame indices to sample uniformly
indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

saved = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in indices:
        resized = cv2.resize(frame, (224, 224))
        out_path = os.path.join(output_dir, f"{saved:03d}.jpg")
        cv2.imwrite(out_path, resized)
        saved += 1

    frame_idx += 1
    if saved >= target_frames:
        break

cap.release()
print(f"Extracted frames!")
