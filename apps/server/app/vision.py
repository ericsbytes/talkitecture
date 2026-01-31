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

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class Config:
    video_path: str = "app/data/ny.mov"     # .mov or .mp4
    model_path: str = "models/yolo11m.pt"       # your partner's trained model
    target_label: str = "empire_state"          # MUST match your model's class name
    conf_thresh: float = 0.35
    iou_thresh: float = 0.45

    # Tracker settings (bbox tracker still available as fallback)
    tracker_type: str = "CSRT"

    # Redetection logic (bbox mode)
    detect_every_n_frames_when_tracking: int = 30
    max_consecutive_lost: int = 10

    # Display
    window_name: str = "Building Tracker"
    draw_label: bool = True

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


# ---------------------------
# Main
# ---------------------------

def main(cfg: Config):
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg.video_path}")

    model = YOLO(cfg.model_path)

    # --- bbox tracking state (your old mode) ---
    tracker = None
    tracking_bbox: Optional[BBox] = None
    tracking = False
    lost_count = 0

    # --- planar quad tracking state (new mode) ---
    mode = "bbox"  # "bbox" or "quad"
    quad_pts: Optional[np.ndarray] = None  # (4,2) float32
    quad_ref: Optional[np.ndarray] = None  # initial reference points (4,2)
    prev_gray: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    quad_lost = 0  # count frames with too many lost points

    frame_idx = 0

    print("Controls:")
    print("  q or ESC: quit")
    print("  r: force re-detect with YOLO (drops bbox tracker)")
    print("  m: manually select ROI (bbox tracker)")
    print("  p: planar mode init (click 4 corners on current frame), then track quad thereafter")
    print("  b: switch back to bbox mode (keeps running YOLO+CSRT pipeline)")
    print("  c: clear quad tracking (if in quad mode)")
    print()

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
            quad_pts = None
            quad_ref = None
            prev_gray = None
            K = None
            quad_lost = 0

        # Init quad planar tracking on current frame
        if key == ord("p"):
            try:
                quad_pts = collect_4_points(frame, window="Init Quad (4 clicks)")
                quad_ref = quad_pts.copy()
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                K = approx_camera_matrix(Ww, Hh)
                mode = "quad"

                # Disable bbox tracker when entering quad mode (avoids confusion)
                tracking = False
                tracker = None
                tracking_bbox = None
                lost_count = 0
                quad_lost = 0
            except RuntimeError as e:
                # user cancelled; stay in current mode
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

        # -----------------------------
        # MODE 1: Planar quad tracking
        # -----------------------------
        if mode == "quad":
            if quad_pts is None or prev_gray is None:
                cv2.putText(frame, "Quad mode: press 'p' to initialize 4 corners", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 255), 2, cv2.LINE_AA)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                next_pts, status = track_points_klt(prev_gray, gray, quad_pts, cfg)
                good = status.reshape(-1) == 1

                if good.sum() < 4:
                    quad_lost += 1
                    cv2.putText(frame, f"Quad tracking unstable ({quad_lost}). Press 'p' to re-init.", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    quad_pts = next_pts.astype(np.float32)
                    quad_lost = 0

                    draw_quad(frame, quad_pts, color=(0, 255, 0), thickness=2)
                    # 1) generate face image for this frame
                    t = time.time()
                    face_rgba = get_next_face_frame_rgba(t)

                    # 2) warp it onto the tracked quad
                    warped_face = warp_face_to_quad(face_rgba, quad_pts, Ww, Hh)

                    # 3) blend onto the live frame
                    frame = alpha_blend_rgba_onto_bgr(frame, warped_face)


                    # Homography from reference quad -> current quad
                    if quad_ref is not None:
                        Hmat, inliers = cv2.findHomography(quad_ref.astype(np.float64),
                                                           quad_pts.astype(np.float64),
                                                           method=cv2.RANSAC,
                                                           ransacReprojThreshold=6.0)
                        if Hmat is not None and cfg.use_pose_decomposition and K is not None:
                            ypr = pose_from_homography(K, Hmat)
                            if ypr is not None:
                                yaw, pitch, roll = ypr
                                cv2.putText(frame, f"plane yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}",
                                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                prev_gray = gray

            cv2.putText(frame, "MODE: QUAD (press 'b' for bbox, 'p' to re-init)", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2, cv2.LINE_AA)

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
