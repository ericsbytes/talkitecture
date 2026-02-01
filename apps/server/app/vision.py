from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

# YOLO from Ultralytics
# pip install ultralytics
from ultralytics import YOLO
import time
from face_source import get_next_face_frame_rgba
import json
from pathlib import Path
import sounddevice as sd
from face_source import init_audio, get_audio_for_playback

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class Config:
    video_path: str = "app/data/ny.mov"     # .mov or .mp4
    model_path: str = "models/yolo11m.pt"       # your partner's trained model
    target_label: str = "empire_state"          # MUST match your model's class name
    conf_thresh: float = 0.35
    iou_thresh: float = 0.45
    reference_json: str = "app/data/reference_points.json"
    orb_nfeatures: int = 1500
    orb_ratio_test: float = 0.75
    orb_min_matches: int = 20


    # Tracker settings (bbox tracker still available as fallback)
    tracker_type: str = "CSRT"

    # Redetection logic (bbox mode)
    detect_every_n_frames_when_tracking: int = 30
    max_consecutive_lost: int = 10

    # Display
    window_name: str = "Building Tracker"
    draw_label: bool = True
    show_debug_overlays: bool = False

    # 4-point planar tracking
    lk_win_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    lk_criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    # Pose from homography (approx intrinsics)
    use_pose_decomposition: bool = True  # set False if you only want quad tracking


# ---------------------------
# Existing helper functions
# ---------------------------

def create_tracker(tracker_type: str):
    """
    Create an OpenCV object tracker.
    Note: some OpenCV builds require opencv-contrib-python for these.
    """
    tracker_type = tracker_type.upper()
    legacy = getattr(cv2, "legacy", None)

    def _make(name: str):
        if hasattr(cv2, name):
            return getattr(cv2, name)()
        if legacy is not None and hasattr(legacy, name):
            return getattr(legacy, name)()
        return None

    if tracker_type == "CSRT":
        t = _make("TrackerCSRT_create")
    elif tracker_type == "KCF":
        t = _make("TrackerKCF_create")
    elif tracker_type == "MOSSE":
        t = _make("TrackerMOSSE_create")
    else:
        raise ValueError("tracker_type must be CSRT, KCF, or MOSSE")

    if t is None:
        raise RuntimeError("Could not create tracker. Try: pip install opencv-contrib-python")
    return t


def xyxy_to_xywh(xyxy: np.ndarray) -> BBox:
    x1, y1, x2, y2 = xyxy.astype(int).tolist()
    x = int(x1)
    y = int(y1)
    w = int(max(1, x2 - x1))
    h = int(max(1, y2 - y1))
    return (x, y, w, h)


def clamp_bbox(bbox: BBox, w: int, h: int) -> BBox:
    x, y, bw, bh = bbox
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    return (x, y, bw, bh)


def detect_target_bbox(
    model: YOLO,
    frame_bgr: np.ndarray,
    target_label: str,
    conf_thresh: float,
    iou_thresh: float,
) -> Optional[Tuple[BBox, float]]:
    """
    Runs YOLO on a single frame and returns best bbox for target_label if found.

    Returns:
        (bbox_xywh, confidence) or None
    """
    results = model.predict(
        source=frame_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
    )

    if not results:
        return None

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    names = r.names
    best = None

    for box in r.boxes:
        cls_id = int(box.cls.item())
        label = names.get(cls_id, str(cls_id))
        conf = float(box.conf.item())

        if label != target_label:
            continue

        xyxy = box.xyxy[0].cpu().numpy()
        bbox = xyxy_to_xywh(xyxy)

        if best is None or conf > best[1]:
            best = (bbox, conf)

    return best


def draw_bbox(frame: np.ndarray, bbox: BBox, text: Optional[str] = None):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    if text:
        cv2.putText(
            frame,
            text,
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


# ---------------------------
# New: 4-point planar tracking
# ---------------------------

def approx_camera_matrix(W: int, H: int) -> np.ndarray:
    """
    Approximate camera intrinsics if you don't have calibration.
    Good enough for demo pose decomposition (yaw/pitch/roll-ish).
    """
    fx = 0.9 * W
    fy = 0.9 * W
    cx = W / 2.0
    cy = H / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def collect_4_points(frame: np.ndarray, window: str = "Init Quad (4 clicks)") -> np.ndarray:
    """
    Collect 4 points by mouse clicks in this strict order:
      1: top-left
      2: top-right
      3: bottom-right
      4: bottom-left

    Returns:
      pts: (4, 2) float32
    """
    clone = frame.copy()
    pts: List[Tuple[int, int]] = []
    instructions = [
        "Click 1: TOP-LEFT corner of facade",
        "Click 2: TOP-RIGHT corner of facade",
        "Click 3: BOTTOM-RIGHT corner of facade",
        "Click 4: BOTTOM-LEFT corner of facade",
    ]

    def on_mouse(event, x, y, flags, param):
        nonlocal clone, pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                clone,
                str(len(pts)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        view = clone.copy()
        msg = instructions[len(pts)] if len(pts) < 4 else "Done! Press ENTER to confirm, or 'c' to clear"
        cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, view)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(pts) == 4:  # Enter
            break
        if key == ord("c"):
            pts = []
            clone = frame.copy()
        if key in (27, ord("q")):  # cancel
            break
        

    cv2.destroyWindow(window)

    if len(pts) != 4:
        raise RuntimeError("Point collection cancelled or incomplete (need 4 points).")

    return np.array(pts, dtype=np.float32)


def track_points_klt(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_pts: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track points using Lucas–Kanade optical flow.
    prev_pts: (N, 2) float32
    Returns:
      next_pts: (N, 2) float32
      status: (N, 1) uint8 where 1 means tracked successfully
    """
    prev_pts_ = prev_pts.reshape(-1, 1, 2)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts_,
        None,
        winSize=cfg.lk_win_size,
        maxLevel=cfg.lk_max_level,
        criteria=cfg.lk_criteria,
    )
    return next_pts.reshape(-1, 2), status


def draw_quad(frame: np.ndarray, pts4: np.ndarray, color=(0, 255, 0), thickness: int = 2):
    pts = pts4.astype(int).reshape(-1, 2)
    for i in range(4):
        a = tuple(pts[i])
        b = tuple(pts[(i + 1) % 4])
        cv2.line(frame, a, b, color, thickness)
    for i, (x, y) in enumerate(pts, start=1):
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv2.putText(frame, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def order_quad_tl_tr_br_bl(quad: np.ndarray) -> np.ndarray:
    """
    Reorders 4 points into TL, TR, BR, BL.
    quad: (4,2)
    """
    pts = quad.astype(np.float32)
    s = pts.sum(axis=1)          # x+y
    d = pts[:, 0] - pts[:, 1]    # x-y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def init_points_in_quad(gray: np.ndarray, quad: np.ndarray, max_pts: int = 250) -> Optional[np.ndarray]:
    """
    Detect strong corners INSIDE the quad to track with optical flow.
    Returns pts: (N,2) float32 or None if not enough points.
    """
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, poly, 255)

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_pts,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=7,
        mask=mask,
        useHarrisDetector=False,
    )
    if pts is None or len(pts) < 25:
        return None
    return pts.reshape(-1, 2).astype(np.float32)


def auto_init_quad_from_corners(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Auto-select a stable quad on a (likely planar) facade by:
      1) finding lots of corners
      2) choosing the densest region (grid cell)
      3) fitting a min-area rectangle to that region -> quad

    Returns:
      quad (4,2) float32 in TL,TR,BR,BL order OR None
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    # Heuristic: ignore top 20% of image (often sky)
    y0 = int(0.20 * H)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[int(0.30*H):int(0.90*H), :] = 255

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1200,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=7,
        mask=mask,
    )
    if pts is None or len(pts) < 80:
        return None

    pts2 = pts.reshape(-1, 2)

    # Grid-bin corners to find densest region
    gx, gy = 10, 8  # grid resolution
    xs = np.clip((pts2[:, 0] / W * gx).astype(int), 0, gx - 1)
    ys = np.clip((pts2[:, 1] / H * gy).astype(int), 0, gy - 1)

    counts = np.zeros((gy, gx), dtype=np.int32)
    for xbin, ybin in zip(xs, ys):
        counts[ybin, xbin] += 1

    yb, xb = np.unravel_index(np.argmax(counts), counts.shape)

    # Take points in the densest cell AND its neighbors for stability
    keep = []
    for i in range(len(pts2)):
        if abs(xs[i] - xb) <= 1 and abs(ys[i] - yb) <= 1:
            keep.append(i)
    cluster = pts2[keep]

    if cluster.shape[0] < 40:
        return None

    # Fit a rotated rectangle
    rect = cv2.minAreaRect(cluster.astype(np.float32))
    box = cv2.boxPoints(rect)  # (4,2)
    quad = order_quad_tl_tr_br_bl(box)

    # Slight shrink-in to avoid grabbing sky/edges (optional)
    center = quad.mean(axis=0, keepdims=True)
    quad = center + 0.90 * (quad - center)

    return quad.astype(np.float32)


def update_quad_via_homography(quad_ref: np.ndarray, Hmat: np.ndarray) -> np.ndarray:
    """
    Uses homography Hmat (ref -> current) to move quad corners into the current frame.
    """
    q = quad_ref.reshape(-1, 1, 2).astype(np.float32)
    q2 = cv2.perspectiveTransform(q, Hmat).reshape(-1, 2)
    return order_quad_tl_tr_br_bl(q2)


def smooth_points(prev: np.ndarray, new: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """
    Exponential smoothing to reduce jitter.
    alpha close to 1 = smoother but laggier.
    """
    return (alpha * prev + (1 - alpha) * new).astype(np.float32)

def warp_face_to_quad(face_rgba: np.ndarray, quad_pts: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Warp an RGBA face image (fh, fw, 4) into the video frame using the destination quad points.
    quad_pts must be in order: TL, TR, BR, BL.
    Returns an RGBA image (out_h, out_w, 4) aligned to the building.
    """
    fh, fw = face_rgba.shape[:2]
    src = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]], dtype=np.float32)
    dst = quad_pts.astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        face_rgba,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),  # transparent
    )
    return warped


def alpha_blend_rgba_onto_bgr(frame_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """
    Alpha blend overlay_rgba (H,W,4) onto frame_bgr (H,W,3). Returns blended frame.
    """
    out = frame_bgr.astype(np.float32)
    overlay = overlay_rgba.astype(np.float32)

    alpha = overlay[:, :, 3:4] / 255.0  # (H,W,1)
    out = out * (1.0 - alpha) + overlay[:, :, :3] * alpha
    return out.astype(np.uint8)

def shrink_quad(quad: np.ndarray, scale: float = 1.5) -> np.ndarray:
    c = quad.mean(axis=0, keepdims=True)
    return (c + scale * (quad - c)).astype(np.float32)

def pose_from_homography(K: np.ndarray, H: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Decompose homography into candidate rotations/translations.
    Returns a (yaw, pitch, roll) estimate in degrees for one plausible solution.
    This is 'demo orientation' (not perfect, but good enough to show the plane rotates).

    Notes:
    - Homography decomposition returns multiple solutions. We pick the one with
      the largest positive t_z (in front of camera) as a heuristic.
    """
    try:
        n_solutions, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return None

    best_idx = None
    best_tz = -1e18
    for i in range(n_solutions):
        tz = float(Ts[i][2, 0])
        if tz > best_tz:
            best_tz = tz
            best_idx = i

    if best_idx is None:
        return None

    R = Rs[best_idx]
    # yaw/pitch/roll from rotation matrix (one convention)
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(yaw), float(pitch), float(roll)

def load_reference_json(json_path: str) -> tuple[str, np.ndarray]:
    """
    Loads {"image_path": "...", "quad": [[x,y],...]}.
    Returns (image_path, quad_ref_pts (4,2) float32 in TL,TR,BR,BL order).
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    img_path = data["image_path"]
    quad = np.array(data["quad"], dtype=np.float32)
    quad = order_quad_tl_tr_br_bl(quad)
    return img_path, quad


def orb_detect_and_compute(gray: np.ndarray, nfeatures: int = 1500):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, des = orb.detectAndCompute(gray, None)
    return kps, des


def estimate_homography_orb(
    ref_gray: np.ndarray,
    ref_kps,
    ref_des: np.ndarray,
    frame_gray: np.ndarray,
    nfeatures: int = 1500,
    ratio: float = 0.75,
    min_matches: int = 20,
) -> Optional[np.ndarray]:
    """
    Estimates homography H such that: ref -> frame.
    Returns H (3x3) or None.
    """
    # Compute ORB for current frame
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps2, des2 = orb.detectAndCompute(frame_gray, None)
    if des2 is None or ref_des is None or len(kps2) < 10 or len(ref_kps) < 10:
        return None

    # Match using Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(ref_des, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_matches:
        return None

    src_pts = np.float32([ref_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    # Optional: require enough inliers
    if inliers is not None and int(inliers.sum()) < min_matches:
        return None

    return H


# ---------------------------
# Main
# ---------------------------

def main(cfg: Config):
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg.video_path}")

    model = YOLO(cfg.model_path)

    # --- reference init assets (ORB on a reference image) ---
    ref_img_path, ref_quad = load_reference_json(cfg.reference_json)
    ref_img_path = str(Path(ref_img_path))  # normalize

    ref_bgr = cv2.imread(ref_img_path)
    if ref_bgr is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_img_path}")

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_kps, ref_des = orb_detect_and_compute(ref_gray, nfeatures=cfg.orb_nfeatures)


    # --- bbox tracking state (your old mode) ---
    tracker = None
    tracking_bbox: Optional[BBox] = None
    tracking = False
    lost_count = 0

    # --- planar tracking state (homography from many points) ---
    mode = "bbox"  # "bbox" or "quad"
    prev_gray: Optional[np.ndarray] = None

    quad_ref: Optional[np.ndarray] = None      # (4,2) initial quad
    quad_curr: Optional[np.ndarray] = None     # (4,2) current quad

    pts_ref: Optional[np.ndarray] = None       # (N,2) points in reference frame
    pts_curr: Optional[np.ndarray] = None      # (N,2) points tracked to current frame

    quad_lost = 0
    frame_idx = 0 

    print("Controls:")
    print("  q or ESC: quit")
    print("  t: init quad from reference image (ORB homography)")
    print("  a: auto-init planar quad (no clicking)")
    print("  r: force re-detect with YOLO (drops bbox tracker)")
    print("  m: manually select ROI (bbox tracker)")
    print("  p: planar mode init (click 4 corners on current frame), then track quad thereafter")
    print("  b: switch back to bbox mode (keeps running YOLO+CSRT pipeline)")
    print("  c: clear quad tracking (if in quad mode)")
    print()

    init_audio("app/data/voice.mp3")  # or .wav
    audio_data, audio_sr = get_audio_for_playback()

    audio_playing = False
    audio_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        Hh, Ww = frame.shape[:2]

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        if key == ord("b"):
            mode = "bbox"
        if key == ord("c"):
            prev_gray = None
            quad_ref = None
            quad_curr = None
            pts_ref = None
            pts_curr = None
            quad_lost = 0


        # Init quad planar tracking on current frame
        if key == ord("p"):
            try:
                q = collect_4_points(frame, window="Init Quad (4 clicks)")
                q = order_quad_tl_tr_br_bl(q)

                gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pts0 = init_points_in_quad(gray0, q, max_pts=300)
                if pts0 is None:
                    raise RuntimeError("Not enough corners inside quad. Try clicking a more textured facade area.")

                mode = "quad"
                prev_gray = gray0
                quad_ref = q.copy()
                quad_curr = q.copy()
                pts_ref = pts0.copy()
                pts_curr = pts0.copy()
                quad_lost = 0

                # Disable bbox tracker when entering quad mode
                tracking = False
                tracker = None
                tracking_bbox = None
                lost_count = 0
            except RuntimeError:
                pass


        # BBox tracker manual init
        if key == ord("m"):
            bbox = cv2.selectROI("Select Building ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Building ROI")
            bbox = tuple(map(int, bbox))
            if bbox[2] > 0 and bbox[3] > 0:
                tracker = create_tracker(cfg.tracker_type)
                bbox = clamp_bbox(bbox, Ww, Hh)
                tracker.init(frame, bbox)
                tracking_bbox = bbox
                tracking = True
                lost_count = 0
                mode = "bbox"

        # Force re-detect resets bbox tracker
        if key == ord("r"):
            tracking = False
            tracker = None
            tracking_bbox = None
            lost_count = 0
        if key == ord("a"):
            # Auto-init quad + points
            q = auto_init_quad_from_corners(frame)
            if q is not None:
                gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pts0 = init_points_in_quad(gray0, q, max_pts=300)

                if pts0 is not None:
                    mode = "quad"
                    prev_gray = gray0
                    quad_ref = q.copy()
                    quad_curr = q.copy()
                    pts_ref = pts0.copy()
                    pts_curr = pts0.copy()
                    quad_lost = 0

                    # Disable bbox pipeline to avoid confusion
                    tracking = False
                    tracker = None
                    tracking_bbox = None
                    lost_count = 0

        if key == ord("t"):
            # Try to find the reference plane in THIS frame, then init quad tracker
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            Href = estimate_homography_orb(
                ref_gray=ref_gray,
                ref_kps=ref_kps,
                ref_des=ref_des,
                frame_gray=frame_gray,
                nfeatures=cfg.orb_nfeatures,
                ratio=cfg.orb_ratio_test,
                min_matches=cfg.orb_min_matches,
            )

            if Href is None:
                cv2.putText(
                    frame,
                    "Reference match failed. Try a clearer frame or press 'a'/'p'.",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                # Map the reference quad into current frame
                q = cv2.perspectiveTransform(ref_quad.reshape(-1, 1, 2), Href).reshape(-1, 2)
                q = order_quad_tl_tr_br_bl(q)

                # Seed KLT points inside that quad
                pts0 = init_points_in_quad(frame_gray, q, max_pts=300)
                if pts0 is None:
                    cv2.putText(
                        frame,
                        "Matched ref, but not enough corners inside quad. Try another frame.",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    # Enter quad tracking mode
                    mode = "quad"
                    audio_start_time = time.time()
                    if audio_data is not None and audio_sr is not None:
                        sd.stop()  # stop any previous playback
                        sd.play(audio_data, audio_sr, blocking=False)
                        audio_playing = True
                    prev_gray = frame_gray
                    quad_ref = q.copy()
                    quad_curr = q.copy()
                    pts_ref = pts0.copy()
                    pts_curr = pts0.copy()
                    quad_lost = 0

                    # Disable bbox tracker to avoid confusion
                    tracking = False
                    tracker = None
                    tracking_bbox = None
                    lost_count = 0
        if key == ord("s"):
            sd.stop()
            audio_playing = False
            audio_start_time = None


        # -----------------------------
        # MODE 1: Planar quad tracking
        # -----------------------------
        if mode == "quad":
            if prev_gray is None or quad_ref is None or quad_curr is None or pts_ref is None or pts_curr is None:
                cv2.putText(
                    frame,
                    "Quad mode: press 'a' (auto) or 'p' (manual) to init",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (50, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Track many points forward
                next_pts, status = track_points_klt(prev_gray, gray, pts_curr, cfg)
                good = status.reshape(-1) == 1

                pts_ref_g = pts_ref[good]
                next_pts_g = next_pts[good]

                if len(pts_ref_g) < 25:
                    quad_lost += 1
                    cv2.putText(
                        frame,
                        f"Planar tracking weak ({quad_lost}). Press 'a'/'p' to re-init.",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Auto-repair: try reseeding points inside current quad
                    if quad_lost >= 5:
                        pts_new = init_points_in_quad(gray, quad_curr, max_pts=300)
                        if pts_new is not None:
                            # Reset reference to current so homography continues smoothly
                            prev_gray = gray
                            quad_ref = quad_curr.copy()
                            pts_ref = pts_new.copy()
                            pts_curr = pts_new.copy()
                            quad_lost = 0
                else:
                    quad_lost = 0

                    # Homography from reference points -> current points
                    Hmat, inliers = cv2.findHomography(
                        pts_ref_g.astype(np.float64),
                        next_pts_g.astype(np.float64),
                        method=cv2.RANSAC,
                        ransacReprojThreshold=6.0,
                    )

                    if Hmat is None:
                        cv2.putText(
                            frame,
                            "Homography failed. Press 'a'/'p' to re-init.",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        # Update quad from homography
                        new_quad = update_quad_via_homography(quad_ref, Hmat)
                        quad_curr = smooth_points(quad_curr, new_quad, alpha=0.80)

                        # Draw quad (debug)
                        if cfg.show_debug_overlays:
                            draw_quad(frame, quad_curr, color=(0, 255, 0), thickness=2)

                        # Project your animated face into quad_curr
                        t_anim = time.time() - audio_start_time if audio_start_time is not None else 0.0
                        face_rgba = get_next_face_frame_rgba(t_anim)
                        quad_face = shrink_quad(quad_curr, scale=1.2)
                        warped_face = warp_face_to_quad(face_rgba, quad_face, Ww, Hh)
                        frame = alpha_blend_rgba_onto_bgr(frame, warped_face)

                        # Optional: show inlier count for debugging
                        if cfg.show_debug_overlays and inliers is not None:
                            cv2.putText(
                                frame,
                                f"inliers: {int(inliers.sum())}/{len(pts_ref_g)}",
                                (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                        # Update point sets to only good points (prevents drift)
                        pts_ref = pts_ref_g.astype(np.float32)
                        pts_curr = next_pts_g.astype(np.float32)
                        # after updating pts_curr
                        if cfg.show_debug_overlays:
                            for (x, y) in pts_curr.astype(int):
                                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)


                prev_gray = gray

            cv2.putText(
                frame,
                "MODE: QUAD (b=bbox, a=auto-init, p=manual-init, c=clear)",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 255, 255),
                2,
                cv2.LINE_AA,
            )


        # -----------------------------
        # MODE 2: YOLO + bbox tracking
        # -----------------------------
        else:
            # Decide whether to run detection
            should_detect = False
            if not tracking:
                should_detect = True
            else:
                if cfg.detect_every_n_frames_when_tracking > 0 and frame_idx % cfg.detect_every_n_frames_when_tracking == 0:
                    should_detect = True
                if lost_count >= cfg.max_consecutive_lost:
                    should_detect = True

            if should_detect:
                det = detect_target_bbox(
                    model=model,
                    frame_bgr=frame,
                    target_label=cfg.target_label,
                    conf_thresh=cfg.conf_thresh,
                    iou_thresh=cfg.iou_thresh,
                )

                if det is not None:
                    bbox, conf = det
                    bbox = clamp_bbox(bbox, Ww, Hh)

                    tracker = create_tracker(cfg.tracker_type)
                    tracker.init(frame, bbox)

                    tracking_bbox = bbox
                    tracking = True
                    lost_count = 0

                    if cfg.draw_label:
                        draw_bbox(frame, bbox, f"DETECTED {cfg.target_label} ({conf:.2f})")
                else:
                    if not tracking:
                        cv2.putText(
                            frame,
                            f"Searching for '{cfg.target_label}'... (press 'm' manual ROI, 'p' quad init)",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

            # Tracking update
            if tracking and tracker is not None:
                success, bbox = tracker.update(frame)
                if success:
                    bbox = tuple(map(int, bbox))
                    bbox = clamp_bbox(bbox, Ww, Hh)
                    tracking_bbox = bbox
                    lost_count = 0
                    label_text = f"TRACKING {cfg.target_label}" if cfg.draw_label else None
                    draw_bbox(frame, bbox, label_text)
                else:
                    lost_count += 1
                    cv2.putText(
                        frame,
                        f"Tracking lost ({lost_count}/{cfg.max_consecutive_lost})... redetecting soon",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if lost_count >= cfg.max_consecutive_lost:
                        tracking = False
                        tracker = None
                        tracking_bbox = None

            cv2.putText(frame, "MODE: BBOX (press 'p' for quad planar tracking)", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2, cv2.LINE_AA)

        # Overlay frame index
        cv2.putText(
            frame,
            f"frame: {frame_idx}",
            (20, Hh - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(cfg.window_name, frame)
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config(
        video_path="app/data/ny.mov",
        model_path="models/yolo11m.pt",
        target_label="empire_state",
    )
    main(cfg)

from .landmarks import find_visible_landmarks


def analyze_video(video_path):
    """Placeholder function for video analysis"""
    pass


def analyze_ar_frame(frame_path, latitude, longitude, accuracy, alpha, beta, gamma, use_api=False):
    """Analyze AR frame and return visible landmarks based on GPS and orientation

    Args:
        frame_path: Path to the video frame (currently unused)
        latitude: User's latitude
        longitude: User's longitude
        accuracy: GPS accuracy in meters
        alpha: Device rotation around Z axis (degrees)
        beta: Device rotation around X axis (degrees)
        gamma: Device rotation around Y axis (degrees)
        use_api: Whether to use external APIs for landmark data

    Returns:
        Dictionary containing landmark info, face region, location, and orientation
    """
    # Find visible landmarks based on GPS and orientation
    visible_landmarks = find_visible_landmarks(
        latitude, longitude, alpha, use_api=use_api)

    if visible_landmarks:
        # Return the closest visible landmark
        landmark = visible_landmarks[0]
        return {
            "landmark": {
                "name": landmark['name'],
                "facts": landmark['facts']
            },
            "face_region": {
                "x": 200,
                "y": 150,
                "width": 120,
                "height": 120
            },
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "accuracy": accuracy
            },
            "orientation": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            },
            "landmark_info": {
                "distance": landmark['distance'],
                "bearing": landmark['bearing'],
                "angle_diff": landmark['angle_diff']
            }
        }
    else:
        # No landmarks visible
        return {
            "landmark": {
                "name": "No landmarks detected",
                "facts": [
                    "Try moving to a different location",
                    "Point your camera toward nearby attractions",
                    f"Current position: {latitude:.4f}, {longitude:.4f}"
                ]
            },
            "face_region": {
                "x": 200,
                "y": 150,
                "width": 120,
                "height": 120
            },
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "accuracy": accuracy
            },
            "orientation": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            }
        }
