# face_source.py
import numpy as np
import cv2
import math

def get_next_face_frame_rgba(t: float, w: int = 256, h: int = 256) -> np.ndarray:
    """
    Returns a single RGBA frame of your animated face at time t.
    Replace this later with sprites, ElevenLabs-driven mouth, etc.
    """
    img = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    # Blink
    blink = 0.2 if (math.sin(t * 2.0) > 0.97) else 1.0

    # Eyes
    ex = int(0.18 * w)
    ey = int(0.10 * h)
    eye_w = int(0.13 * w)
    eye_h = max(1, int(0.08 * h * blink))

    left = (cx - ex, cy - ey)
    right = (cx + ex, cy - ey)

    cv2.ellipse(img, left, (eye_w, eye_h), 0, 0, 360, (255, 255, 255, 255), -1)
    cv2.ellipse(img, right, (eye_w, eye_h), 0, 0, 360, (255, 255, 255, 255), -1)

    # Pupils drift
    px = int(eye_w * 0.35 * math.sin(t * 1.7))
    pr = max(1, int(0.03 * w))
    cv2.circle(img, (left[0] + px, left[1]), pr, (0, 0, 0, 255), -1)
    cv2.circle(img, (right[0] + px, right[1]), pr, (0, 0, 0, 255), -1)

    # Mouth driven by time (swap to audio amplitude later)
    open_amt = 0.2 + 0.5 * (0.5 + 0.5 * math.sin(t * 3.0))
    mw = int(0.22 * w)
    mh = max(2, int(0.10 * h * open_amt))
    mouth_center = (cx, cy + int(0.18 * h))
    cv2.ellipse(img, mouth_center, (mw, mh), 0, 0, 180, (0, 0, 0, 255), 3)

    return img
