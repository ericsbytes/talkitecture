# face_source.py
import numpy as np
import cv2
import math
import librosa
from scipy.signal import hilbert
_AUDIO = None
_SR = None
_ENVELOPE = None
_DURATION = None

def init_audio(audio_path: str, smooth_ms: float = 50.0):
    """
    Load audio (wav or mp3) and compute a smooth amplitude envelope.
    """
    global _AUDIO, _SR, _ENVELOPE, _DURATION

    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio) + 1e-6)

    # Amplitude envelope via Hilbert transform
    analytic = hilbert(audio)
    envelope = np.abs(analytic)

    # Smooth envelope (moving average)
    win = int(sr * smooth_ms / 1000)
    if win > 1:
        kernel = np.ones(win) / win
        envelope = np.convolve(envelope, kernel, mode="same")

    envelope /= envelope.max() + 1e-6

    _AUDIO = audio
    _SR = sr
    _ENVELOPE = envelope
    _DURATION = len(audio) / sr


def mouth_openness_at_time(t: float, silence_thresh: float = 0.08) -> float:
    """
    Returns mouth openness in [0,1] at time t.
    """
    if _ENVELOPE is None:
        return 0.0

    if t < 0 or t > _DURATION:
        return 0.0

    idx = int(t * _SR)
    idx = min(idx, len(_ENVELOPE) - 1)

    a = _ENVELOPE[idx]

    # Gate silence
    if a < silence_thresh:
        return 0.0

    return float(a)

def get_audio_for_playback():
    """Returns (audio_mono_float32, sr) or (None, None) if not initialized."""
    if _AUDIO is None or _SR is None:
        return None, None
    return _AUDIO, _SR

import cv2
import numpy as np

# Global style setting for boldness
OUTLINE_BOLD = 5 
OUTLINE_THIN = 3

def draw_eyelashes(img, cx, cy, eye_w, eye_h, blink, side):
    """
    Draws 3-4 distinct lashes on the outer corner.
    side: -1 for left eye, 1 for right eye
    """
    if blink < 0.3: return  # Don't draw individual lashes when closed
    
    AA = cv2.LINE_AA
    # The 'start' point for lashes is the outer upper edge of the eye
    # We draw 3 lashes with varying angles
    for i in range(3):
        angle_deg = -20 - (i * 20)  # Angles fanning upward
        angle_rad = np.deg2rad(angle_deg)
        
        # Start point (on the eyelid curve)
        start_x = cx + (side * eye_w * 0.8)
        start_y = cy - (eye_h * 0.6)
        
        # End point (fanning out and up)
        length = 15 + (i * 5)
        end_x = int(start_x + (side * length * np.cos(angle_rad)))
        end_y = int(start_y + (length * np.sin(angle_rad)))
        
        cv2.line(img, (int(start_x), int(start_y)), (end_x, end_y), (10, 10, 10, 255), 3, AA)

def draw_styled_eye(img, center, eye_w, eye_h_open, blink, iris_color):
    cx, cy = center
    eye_h = int(eye_h_open * (blink ** 0.8))
    AA = cv2.LINE_AA
    
    # Determine if this is the left or right eye relative to the face center
    side = -1 if cx < (img.shape[1] // 2) else 1

    if blink < 0.15:
        cv2.ellipse(img, (cx, cy), (eye_w, int(eye_h_open*0.25)), 0, 0, 180, (20, 20, 20, 255), 5, AA)
        return

    # Sclera & Iris (Your existing logic)
    cv2.ellipse(img, (cx, cy), (eye_w, eye_h), 0, 0, 360, (255, 255, 255, 255), -1, AA)
    cv2.ellipse(img, (cx, cy), (eye_w, eye_h), 0, 0, 360, (10, 10, 10, 255), 5, AA)
    
    iris_r = int(eye_h * 0.75)
    cv2.circle(img, (cx, cy), iris_r, iris_color, -1, AA)
    cv2.circle(img, (cx, cy), int(iris_r * 0.5), (10, 10, 10, 255), -1, AA)
    cv2.circle(img, (cx - int(iris_r*0.3), cy - int(iris_r*0.3)), int(iris_r*0.3), (255, 255, 255, 255), -1, AA)

    # --- ADD EYELASHES HERE ---
    draw_eyelashes(img, cx, cy, eye_w, eye_h, blink, side)
    
    # Bold Upper Eyelid (covers the lash start points for a clean look)
    cv2.ellipse(img, (cx, cy), (eye_w, eye_h), 0, 190, 350, (10, 10, 10, 255), 7, AA)

def draw_eyebrow(img, center, width, lift_amt):
    cx, cy = center
    lift_y = cy - int(lift_amt * 20)
    
    # Using a polyline for a thicker, tapered look
    pts = np.array([
        [cx - width, lift_y + 8],
        [cx, lift_y],
        [cx + width, lift_y + 8]
    ], np.int32)
    cv2.polylines(img, [pts], False, (10, 10, 10, 255), OUTLINE_BOLD + 1, cv2.LINE_AA)

def get_next_face_frame_rgba(t: float, w: int = 512, h: int = 512) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    mid_x, mid_y = w // 2, h // 2
    
    # Function to get openness (Ensure this is defined in your environment)
    open_amt = mouth_openness_at_time(t) 
    blink = 0.0 if (np.sin(t * 3.5) > 0.97) else 1.0 
    
    # --- Features ---
    # Eyebrows
    draw_eyebrow(img, (mid_x - 85, mid_y - 100), 50, open_amt)
    draw_eyebrow(img, (mid_x + 85, mid_y - 100), 50, open_amt)

    # Eyes
    eye_color = (220, 160, 120, 255) # Blue-ish
    draw_styled_eye(img, (mid_x - 90, mid_y - 35), 55, 35, blink, eye_color)
    draw_styled_eye(img, (mid_x + 90, mid_y - 35), 55, 35, blink, eye_color)

    # Mouth
    m_cx, m_cy = mid_x, mid_y + 100
    mw, mh = 70, int(5 + 65 * open_amt)

    if open_amt > 0.05:
        # Thick Outer Lip/Border
        cv2.ellipse(img, (m_cx, m_cy), (mw + 4, mh + 4), 0, 0, 360, (60, 40, 220, 255), -1, cv2.LINE_AA)
        # Inner Mouth
        cv2.ellipse(img, (m_cx, m_cy), (mw, mh), 0, 0, 360, (30, 20, 20, 255), -1, cv2.LINE_AA)
        
        # Teeth (Simplified bold block)
        #tw = int(mw * 0.7)
        #cv2.rectangle(img, (m_cx - tw, m_cy - mh + 2), (m_cx + tw, m_cy - mh + 15), (255, 255, 255, 255), -1, cv2.LINE_AA)
    else:
        # Thick expressive smile line
        cv2.ellipse(img, (m_cx, m_cy - 10), (mw, 20), 0, 10, 170, (60, 40, 220, 255), OUTLINE_BOLD, cv2.LINE_AA)

    return img