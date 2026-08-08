import cv2
import numpy as np
from dt_apriltags import Detector

# =============================================================
# CONSTANTS — edit these for your setup
# =============================================================

TAG_SIZE_MM = 20
CAMERA_INDEX = 0

# Same coordinate system as before: origin (0,0) = ID 0, top-left.
TOP_EDGE_IDS = [0, 2, 4, 6]
LEFT_EDGE_IDS = [8, 10, 12]
RIGHT_EDGE_IDS = [9, 11, 13]
HORIZONTAL_SPACING_MM = 100.0
VERTICAL_SPACING_MM = 100.0

# How many nearest markers (by pixel distance to the box) to use for
# the local transform. 1 = fastest, least stable. 2-3 = good balance.
NUM_NEAREST_MARKERS = 2

# =============================================================


def build_tag_layout_mm():
    layout = {}
    for i, tag_id in enumerate(TOP_EDGE_IDS):
        layout[tag_id] = (i * HORIZONTAL_SPACING_MM, 0.0)
    for i, tag_id in enumerate(LEFT_EDGE_IDS):
        layout[tag_id] = (0.0, (i + 1) * VERTICAL_SPACING_MM)
    right_x_mm = (len(TOP_EDGE_IDS) - 1) * HORIZONTAL_SPACING_MM
    for i, tag_id in enumerate(RIGHT_EDGE_IDS):
        layout[tag_id] = (right_x_mm, (i + 1) * VERTICAL_SPACING_MM)
    return layout


TAG_LAYOUT_MM = build_tag_layout_mm()


def tag_corners_in_paper_mm(center_x_mm, center_y_mm, size_mm):
    """4 known corners of a marker in paper mm, given its known center."""
    h = size_mm / 2.0
    bl = (center_x_mm - h, center_y_mm + h)
    br = (center_x_mm + h, center_y_mm + h)
    tr = (center_x_mm + h, center_y_mm - h)
    tl = (center_x_mm - h, center_y_mm - h)
    return [bl, br, tr, tl]


def estimate_similarity_transform(src_pts, dst_pts):
    """
    Umeyama's method: closed-form least-squares fit of a similarity
    transform (uniform scale + rotation + translation) mapping
    src_pts -> dst_pts. Works with as few as 2 point pairs (exact fit)
    or more (least-squares fit, averages out noise).

    Returns (scale, R (2x2 rotation matrix), t (2,)) such that:
        dst ≈ scale * R @ src + t
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Covariance matrix
    cov = (dst_c.T @ src_c) / n

    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(cov))
    D = np.eye(2)
    if d < 0:
        D[1, 1] = -1

    R = U @ D @ Vt

    var_src = (src_c ** 2).sum() / n
    scale = (S * np.diag(D)).sum() / var_src if var_src > 1e-9 else 1.0

    t = dst_mean - scale * (R @ src_mean)

    return scale, R, t


def apply_similarity_transform(scale, R, t, points_px):
    """Apply a (scale, R, t) similarity transform to a list/array of 2D points."""
    pts = np.asarray(points_px, dtype=np.float64)
    return (scale * (R @ pts.T)).T + t


def find_nearest_markers(box_center_px, detected_markers, num_nearest):
    """
    detected_markers: list of dicts, each with keys:
        'id', 'center_px', 'corners_px' (4x2), 'known_center_mm', 'known_corners_mm'
    Returns the `num_nearest` markers whose center is closest (in pixels)
    to the box's center — these are the only ones used for this box's
    local transform.
    """
    scored = []
    for m in detected_markers:
        dist = np.linalg.norm(np.array(m["center_px"]) - np.array(box_center_px))
        scored.append((dist, m))
    scored.sort(key=lambda x: x[0])
    return [m for _, m in scored[:num_nearest]]


def map_box_to_paper_mm(box_center_px, box_corners_px, detected_markers,
                         num_nearest=NUM_NEAREST_MARKERS):
    """
    Main entry point: given a detected box (its pixel center and 4 pixel
    corners) and the list of currently detected markers, returns the
    box's estimated center (mm) and size (width_mm, height_mm) using
    only the nearest markers.

    Returns None if fewer than 1 known marker is available at all
    (can't do anything), otherwise always returns a best-effort estimate.
    """
    if len(detected_markers) == 0:
        return None

    nearest = find_nearest_markers(box_center_px, detected_markers, num_nearest)

    # Build point correspondences: pixel -> mm, pooling corners from
    # whichever nearby markers we selected (1 marker = 4 points,
    # 2 markers = 8 points, etc.)
    src_pts = []  # pixel
    dst_pts = []  # mm
    for m in nearest:
        src_pts.extend(m["corners_px"])
        dst_pts.extend(m["known_corners_mm"])

    if len(src_pts) < 2:
        # A single marker still gives 4 corners, so this only triggers
        # if something upstream filtered corners down — kept as a
        # safety guard.
        return None

    scale, R, t = estimate_similarity_transform(src_pts, dst_pts)

    box_center_mm = apply_similarity_transform(scale, R, t, [box_center_px])[0]
    box_corners_mm = apply_similarity_transform(scale, R, t, box_corners_px)

    # Estimate box size in mm from its transformed corners.
    # Assumes box_corners_px are ordered consistently (e.g. TL, TR, BR, BL).
    width_mm = np.linalg.norm(box_corners_mm[1] - box_corners_mm[0])
    height_mm = np.linalg.norm(box_corners_mm[3] - box_corners_mm[0])

    return {
        "center_mm": tuple(box_center_mm),
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "markers_used": [m["id"] for m in nearest],
    }


# =============================================================
# DEMO — replace the placeholder box detector with your fine-tuned
# model's output (pixel center + 4 pixel corners per detected box).
# =============================================================

def dummy_box_detector(frame):
    """
    PLACEHOLDER. Replace this with your actual fine-tuned box detector.
    Must return a list of dicts: {'center_px': (x,y), 'corners_px': [4 x (x,y)]}
    """
    return []


def main():
    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    print("Starting video feed. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray, estimate_tag_pose=False)

        detected_markers = []
        for tag in tags:
            if tag.tag_id not in TAG_LAYOUT_MM:
                continue
            cx_mm, cy_mm = TAG_LAYOUT_MM[tag.tag_id]
            detected_markers.append({
                "id": tag.tag_id,
                "center_px": tuple(tag.center),
                "corners_px": tag.corners.tolist(),
                "known_center_mm": (cx_mm, cy_mm),
                "known_corners_mm": tag_corners_in_paper_mm(cx_mm, cy_mm, TAG_SIZE_MM),
            })

            corners_int = tag.corners.astype(int)
            cv2.polylines(frame, [corners_int], True, (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tag.tag_id}",
                        (corners_int[0][0], corners_int[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        boxes = dummy_box_detector(frame)  # <- plug your model in here

        for box in boxes:
            result = map_box_to_paper_mm(
                box["center_px"], box["corners_px"], detected_markers
            )
            if result is not None:
                cx, cy = int(box["center_px"][0]), int(box["center_px"][1])
                label = (f"{result['center_mm'][0]:.0f},{result['center_mm'][1]:.0f}mm "
                         f"{result['width_mm']:.0f}x{result['height_mm']:.0f}mm")
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
                cv2.putText(frame, label, (cx + 8, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.imshow("Local Marker-Based Box Mapping", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()