import cv2
import numpy as np
from PIL import Image
import os

# Dummy landmark facts
LANDMARK_FACTS = {
    "Eiffel Tower": {
        "name": "Eiffel Tower",
        "facts": [
            "Built in 1889 for the World's Fair",
            "Originally intended to be temporary",
            "Made of iron, weighs 10,000 tons"
        ]
    },
    "Statue of Liberty": {
        "name": "Statue of Liberty",
        "facts": [
            "Gift from France to the US",
            "Designed by Frédéric Auguste Bartholdi",
            "Symbol of freedom and democracy"
        ]
    }
}


def identify_landmark(frame):
    # Placeholder: always return Eiffel Tower
    # In real implementation, use vision API
    return "Eiffel Tower"


def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames


def compute_tracking_data(frames):
    # Placeholder: assume building is in center, fixed size
    height, width = frames[0].shape[:2]
    center_x, center_y = width // 2, height // 2
    size = min(width, height) // 4

    tracking = []
    for i, frame in enumerate(frames):
        # Dummy tracking: slight movement
        offset_x = int(10 * np.sin(i * 0.1))
        offset_y = int(5 * np.cos(i * 0.1))
        tracking.append({
            "frame": i,
            "x": center_x + offset_x,
            "y": center_y + offset_y,
            "width": size,
            "height": size
        })

    return tracking


def analyze_video(video_path):
    frames = extract_frames(video_path)
    if not frames:
        return {"error": "No frames extracted"}

    landmark = identify_landmark(frames[0])
    facts = LANDMARK_FACTS.get(
        landmark, {"name": "Unknown", "facts": ["No facts available"]})
    tracking = compute_tracking_data(frames)

    return {
        "landmark": facts,
        "tracking": tracking,
        "frame_count": len(frames)
    }
