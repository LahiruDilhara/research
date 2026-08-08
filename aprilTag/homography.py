import cv2
import numpy as np
from dt_apriltags import Detector

# =============================================================
# CONSTANTS — edit these for your setup
# =============================================================

TAG_SIZE_MM = 20
CAMERA_INDEX = 0

# -------------------------------------------------------------
# COORDINATE SYSTEM
# You must define this explicitly — the code has no way to know
# which marker is a "corner" or an "edge" on its own.
#
#   origin (0,0) = ID 0, top-left corner
#   x increases -> to the RIGHT, along the top edge
#   y increases -> DOWNWARD, along the left/right edges
#
# Based on your layout:
#   Top edge, left -> right:   ID 0 (corner), ID 2, ID 4, ID 6 (corner)
#   Left edge, top -> bottom (below ID 0):   ID 8, ID 10, ID 12
#   Right edge, top -> bottom (below ID 6):  ID 9, ID 11, ID 13
# -------------------------------------------------------------

TOP_EDGE_IDS = [0, 2, 4, 6]      # left -> right, ID 0 = top-left corner, ID 6 = top-right corner
LEFT_EDGE_IDS = [8, 10, 12]      # top -> bottom, below ID 0 (does NOT include ID 0 itself)
RIGHT_EDGE_IDS = [9, 11, 13]     # top -> bottom, below ID 6 (does NOT include ID 6 itself)

# Distance between consecutive markers ALONG the top edge (mm)
HORIZONTAL_SPACING_MM = 100.0

# Distance between consecutive markers DOWN the left/right edges (mm)
VERTICAL_SPACING_MM = 100.0

# --- Homography stability settings ---
MIN_MARKERS_REQUIRED = 1   # minimum known markers visible before trusting a fresh solve
H_SMOOTHING_ALPHA = 0.75   # 0-1, higher = smoother but more lag

PIXELS_PER_MM = 2.0

# =============================================================


def build_tag_layout_mm():
    """
    Build the ID -> (x_mm, y_mm) lookup table from the edge lists above.
    This is the ONE place that encodes "which ID is where on the paper" —
    everything else in the script just consumes this table blindly.
    """
    layout = {}

    # Top edge: evenly spaced left to right, y = 0
    for i, tag_id in enumerate(TOP_EDGE_IDS):
        x_mm = i * HORIZONTAL_SPACING_MM
        y_mm = 0.0
        layout[tag_id] = (x_mm, y_mm)

    # Left edge: x = 0, starts one spacing below ID 0 (the top-left corner)
    for i, tag_id in enumerate(LEFT_EDGE_IDS):
        x_mm = 0.0
        y_mm = (i + 1) * VERTICAL_SPACING_MM
        layout[tag_id] = (x_mm, y_mm)

    # Right edge: x = same as top-right corner, starts one spacing below ID 6
    right_x_mm = (len(TOP_EDGE_IDS) - 1) * HORIZONTAL_SPACING_MM
    for i, tag_id in enumerate(RIGHT_EDGE_IDS):
        x_mm = right_x_mm
        y_mm = (i + 1) * VERTICAL_SPACING_MM
        layout[tag_id] = (x_mm, y_mm)

    return layout


TAG_LAYOUT_MM = build_tag_layout_mm()

# Derive the paper's full extent from the layout itself, so the flat
# output canvas always matches your actual marker spread.
_all_x = [p[0] for p in TAG_LAYOUT_MM.values()]
_all_y = [p[1] for p in TAG_LAYOUT_MM.values()]
PAPER_WIDTH_MM = max(_all_x) - min(_all_x)
PAPER_HEIGHT_MM = max(_all_y) - min(_all_y)


def tag_corners_in_paper_mm(center_x_mm, center_y_mm, size_mm):
    """
    Given a tag's known center in paper mm, return its 4 corners in
    the same order dt_apriltags returns image corners:
    bottom-left, bottom-right, top-right, top-left.
    Assumes tags are printed axis-aligned with the paper.
    """
    h = size_mm / 2.0
    bl = (center_x_mm - h, center_y_mm + h)
    br = (center_x_mm + h, center_y_mm + h)
    tr = (center_x_mm + h, center_y_mm - h)
    tl = (center_x_mm - h, center_y_mm - h)
    return [bl, br, tr, tl]


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

    canvas_w = int(PAPER_WIDTH_MM * PIXELS_PER_MM)
    canvas_h = int(PAPER_HEIGHT_MM * PIXELS_PER_MM)

    smoothed_H = None

    print("Starting video feed. Press 'q' to exit.")
    print(f"Known marker layout (mm): {TAG_LAYOUT_MM}")
    print(f"Derived paper size (mm): {PAPER_WIDTH_MM} x {PAPER_HEIGHT_MM}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray, estimate_tag_pose=False)

        image_pts = []
        paper_pts_mm = []

        for tag in tags:
            corners = tag.corners.astype(int)
            center = tuple(tag.center.astype(int))

            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(frame, center, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                frame, f"ID: {tag.tag_id}", (center[0] - 15, center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

            # Only markers present in our known layout can contribute
            # to the homography. Any other detected ID is drawn but ignored.
            if tag.tag_id in TAG_LAYOUT_MM:
                cx_mm, cy_mm = TAG_LAYOUT_MM[tag.tag_id]
                known_corners_mm = tag_corners_in_paper_mm(cx_mm, cy_mm, TAG_SIZE_MM)
                image_pts.extend(tag.corners.tolist())
                paper_pts_mm.extend(known_corners_mm)

        cv2.imshow("AprilTag Live Detection", frame)

        num_markers_used = len(image_pts) // 4

        if num_markers_used >= MIN_MARKERS_REQUIRED:
            image_pts_np = np.array(image_pts, dtype=np.float32)
            paper_pts_px = np.array(
                [(x * PIXELS_PER_MM, y * PIXELS_PER_MM) for (x, y) in paper_pts_mm],
                dtype=np.float32,
            )

            H_new, mask = cv2.findHomography(image_pts_np, paper_pts_px, cv2.RANSAC, 5.0)

            if H_new is not None:
                if smoothed_H is None:
                    smoothed_H = H_new
                else:
                    smoothed_H = (
                        H_SMOOTHING_ALPHA * smoothed_H
                        + (1 - H_SMOOTHING_ALPHA) * H_new
                    )

        if smoothed_H is not None:
            flat_view = cv2.warpPerspective(frame, smoothed_H, (canvas_w, canvas_h))
            cv2.putText(
                flat_view, f"Markers used: {num_markers_used}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
            cv2.imshow("Flat Top-Down Paper View", flat_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()