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

def draw_feminine_eye_rgba(
    img: np.ndarray,
    center: tuple[int, int],
    eye_w: int,
    eye_h_open: int,
    blink: float,
    iris_color_bgra=(120, 170, 200, 255),  # soft blue-gray (B,G,R,A)
):
    """
    Draws a cute, feminine eye that blinks nicely.
    img is RGBA.
    blink: 1.0 = fully open, 0.0 = closed
    """
    cx, cy = center

    # Eye height shrinks with blink; keep >=1 so OpenCV doesn't complain
    eye_h = max(1, int(eye_h_open * blink))

    # If almost closed: draw a soft eyelid line only (no scary pupil)
    if blink < 0.15:
        # Slight smile-shaped eyelid
        cv2.ellipse(img, (cx, cy), (eye_w, max(1, int(0.20 * eye_h_open))), 0, 200, 340, (0, 0, 0, 255), 2)
        return

    # --- Sclera (white) ---
    cv2.ellipse(img, (cx, cy), (eye_w, eye_h), 0, 0, 360, (255, 255, 255, 255), -1)

    # --- Upper lid outline (makes it less "stare") ---
    cv2.ellipse(img, (cx, cy), (eye_w, eye_h), 0, 200, 340, (0, 0, 0, 255), 2)

    # --- Iris (colored) ---
    iris_r = max(1, int(0.55 * eye_h))  # scale with openness
    iris_center = (cx, cy + int(0.05 * eye_h))  # slightly lower = cuter
    cv2.circle(img, iris_center, iris_r, iris_color_bgra, -1)

    # --- Pupil (not too big) ---
    pupil_r = max(1, int(0.45 * iris_r))
    cv2.circle(img, iris_center, pupil_r, (10, 10, 10, 255), -1)

    # --- Highlight (sparkle) ---
    hl_r = max(1, int(0.25 * pupil_r))
    cv2.circle(img, (iris_center[0] - int(0.30 * iris_r), iris_center[1] - int(0.30 * iris_r)), hl_r, (255, 255, 255, 230), -1)

    # --- Lower lash / subtle shadow (softens eye) ---
    cv2.ellipse(img, (cx, cy + int(0.10 * eye_h)), (eye_w, max(1, int(0.55 * eye_h))), 0, 20, 160, (0, 0, 0, 120), 1)

    # --- Eyelashes (only when open; scale with blink) ---
    # Anchor along the upper lid arc. As blink closes, lashes shorten and fade out.
    lash_len = int((0.55 + 0.45 * blink) * (0.75 * eye_h_open))
    lash_len = max(2, int(0.35 * lash_len))  # keep them tasteful
    lash_th = 2

    # Place a few lashes on the outer half to feel more feminine
    # (Angles chosen to fan upward/outward)
    lashes = [
        (-0.55, -0.90),
        (-0.25, -1.00),
        ( 0.05, -0.95),
        ( 0.35, -0.80),
    ]
    # eyelid y position (top of sclera)
    lid_y = cy - int(0.65 * eye_h)

    for dx_norm, dy_norm in lashes:
        x0 = cx + int(dx_norm * eye_w)
        y0 = lid_y
        x1 = x0 + int(0.35 * lash_len * dx_norm)
        y1 = y0 - int(lash_len * (0.9 + 0.2 * (-dy_norm)))  # upward
        cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0, int(180 * blink)), lash_th)

    # Optional: tiny eyeliner at upper lid
    cv2.ellipse(img, (cx, cy), (eye_w, eye_h), 0, 200, 340, (0, 0, 0, 200), 2)

def get_next_face_frame_rgba(t: float, w: int = 256, h: int = 256) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    # --- Eyes (same as before) ---
    # --- Eyes (feminine + blink-friendly) ---
    blink = 0.25 if (np.sin(t * 2.0) > 0.97) else 1.0  # keep your blink logic

    cx, cy = w // 2, h // 2
    ex = int(0.20 * w)
    ey = int(0.10 * h)

    eye_w = int(0.14 * w)
    eye_h_open = int(0.09 * h)

    left_center = (cx - ex, cy - ey)
    right_center = (cx + ex, cy - ey)

    draw_feminine_eye_rgba(img, left_center, eye_w, eye_h_open, blink, iris_color_bgra=(140, 180, 210, 255))
    draw_feminine_eye_rgba(img, right_center, eye_w, eye_h_open, blink, iris_color_bgra=(140, 180, 210, 255))

    # --- Mouth driven by audio ---
    open_amt = mouth_openness_at_time(t)  # 0..1

    mw = int(0.23 * w)                    # mouth half-width
    mh = int(0.02 * h + 0.14 * h * open_amt)  # height grows with audio
    cx, cy = w // 2, h // 2
    mouth_center = (cx, cy + int(0.20 * h))

    # Mouth cavity (inside)
    if open_amt > 0.03:
        cv2.ellipse(img, mouth_center, (mw, mh), 0, 0, 360, (10, 10, 10, 255), -1)

        # Add a little tongue when open
        tongue_h = int(0.45 * mh)
        tongue_w = int(0.65 * mw)
        tongue_center = (mouth_center[0], mouth_center[1] + int(0.25 * mh))
        cv2.ellipse(img, tongue_center, (tongue_w, tongue_h), 0, 0, 360, (0, 0, 0, 255), -1)

        # Upper teeth hint (a small light strip)
        teeth_h = max(2, int(0.20 * mh))
        teeth_top = mouth_center[1] - int(0.55 * mh)
        cv2.rectangle(img,
                    (mouth_center[0] - int(0.7 * mw), teeth_top),
                    (mouth_center[0] + int(0.7 * mw), teeth_top + teeth_h),
                    (230, 230, 230, 230), -1)

    # Lip outline (draw on top)
    cv2.ellipse(img, mouth_center, (mw, max(2, mh)), 0, 0, 360, (64, 40, 241, 255), 3)

    # If basically closed, draw a smile line instead
    if open_amt <= 0.03:
        cv2.ellipse(img, mouth_center, (mw, max(2, int(0.05*h))), 0, 0, 180, (0,0,0,255), 3)
    return img
