import cv2
import numpy as np
from dt_apriltags import Detector

# =============================================================
# CONSTANTS — edit these for your setup
# =============================================================

# Physical size of the black square border of your printed tag,
# measured edge-to-edge of the OUTER black square, in meters.
# e.g. a 4 cm marker -> 0.04
TAG_SIZE_M = 0.04

# --- Camera intrinsics ---
# These MUST come from a proper camera calibration (checkerboard +
# cv2.calibrateCamera) for the distance/tilt numbers to be accurate.
# The placeholder values below are a rough guess based on a common
# 720p webcam FOV and WILL be off — replace them once you calibrate.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Rough guess: fx = fy ~ frame_width (assumes ~60-70 deg horizontal FOV)
FX = FRAME_WIDTH * 1.0
FY = FRAME_WIDTH * 1.0
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0

CAMERA_PARAMS = (FX, FY, CX, CY)  # (fx, fy, cx, cy) required by dt_apriltags

# Camera index for cv2.VideoCapture
CAMERA_INDEX = 0

# =============================================================


def rotation_matrix_to_euler_deg(R):
    """
    Convert a 3x3 rotation matrix (tag pose relative to camera) into
    roll, pitch, yaw in degrees. This tells you how tilted the tag
    plane is relative to the camera's view direction.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0

    return roll, pitch, yaw


def main():
    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    print("Starting video feed. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=CAMERA_PARAMS,
            tag_size=TAG_SIZE_M,
        )

        for tag in tags:
            corners = tag.corners.astype(int)
            center = tuple(tag.center.astype(int))

            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(frame, center, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                frame, f"ID: {tag.tag_id}", (center[0] - 15, center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

            # --- Pose info ---
            # pose_t: translation vector, tag center relative to camera, in meters.
            # Its norm = straight-line distance from camera to tag.
            t = tag.pose_t.flatten()
            distance_m = np.linalg.norm(t)

            # pose_R: 3x3 rotation matrix, tag orientation relative to camera.
            roll, pitch, yaw = rotation_matrix_to_euler_deg(tag.pose_R)

            info_lines = [
                f"dist: {distance_m*100:.1f} cm",
                f"roll:{roll:5.1f} pitch:{pitch:5.1f} yaw:{yaw:5.1f}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(
                    frame, line, (center[0] - 15, center[1] + 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1
                )

        cv2.imshow("AprilTag Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()