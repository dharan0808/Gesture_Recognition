import cv2
import numpy as np

def extract_frames_from_video(video_path, num_frames=20, resize_dim=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    current_frame = 0
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == frame_indices[idx]:
            resized = cv2.resize(frame, resize_dim)
            frames.append(resized)
            idx += 1

            if idx >= len(frame_indices):
                break

        current_frame += 1

    cap.release()
    return frames
