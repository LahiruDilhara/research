"""
realtimeprocess/main_realtime_ui.py

Interactive Real-Time Touch Detection HUD Dashboard & Control Center.

Features:
1. Live 12 FPS camera feed canvas with MediaPipe hand skeleton overlay.
2. Real-time sliding 5-frame window inference with 2-frame shift triggers.
3. Per-finger Touch Status Dashboard (Thumb, Index, Middle, Ring, Pinky) with confidence probability bars.
4. Interactive model selector (press 'n' or 'm' to cycle available .pth models live).
5. Device switcher (press 'd' to toggle CPU / CUDA).
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtimeprocess.model_manager import ModelManager
from realtimeprocess.camera_thread import CameraThread


class RealtimeApp:
    def __init__(self, camera_src=0, device: str = None):
        print("\n" + "="*80)
        print("  REAL-TIME TOUCH GESTURE DETECTION & MODEL INFERENCE HUD")
        print("="*80)

        self.model_manager = ModelManager(device=device)
        self.last_predictions = {f: {"touch": False, "prob": 0.0} for f in ["thumb", "index", "middle", "ring", "pinky"]}
        self.inference_latency_ms = 0.0
        self.last_trigger_time = time.perf_counter()

        self.camera_thread = CameraThread(
            src=camera_src,
            target_fps=12.0,
            callback=self.on_window_shift_trigger
        )

    def on_window_shift_trigger(self, window_5_frames, scores_5, frame_w, frame_h):
        """Triggered asynchronously whenever a 2-frame shift occurs on a full 5-frame buffer."""
        t0 = time.perf_counter()
        preds = self.model_manager.predict_window(window_5_frames, scores_5, frame_w, frame_h)
        latency = (time.perf_counter() - t0) * 1000.0

        self.last_predictions = preds
        self.inference_latency_ms = latency

    def draw_hud(self, frame, actual_fps, hand_detected):
        """Renders rich OpenCV HUD overlay with per-finger touch status cards and controls."""
        h, w, _ = frame.shape

        # Sidebar canvas width = 360px
        sidebar_w = 360
        canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        # Draw dark sidebar background
        sidebar = canvas[:, w:]
        sidebar[:] = (24, 24, 28)

        # ── Sidebar Header ───────────────────────────────────────────────────
        cv2.rectangle(sidebar, (10, 10), (sidebar_w - 10, 120), (38, 38, 45), -1)
        cv2.rectangle(sidebar, (10, 10), (sidebar_w - 10, 120), (70, 70, 85), 1)

        cv2.putText(sidebar, "TOUCH DETECTOR HUD", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        info = self.model_manager.active_info
        if info:
            arch_title = f"{info['arch_name']}"
            var_title  = f"Variant: {info['variant_name']}"
        else:
            arch_title = "No Model Loaded"
            var_title  = "Variant: None"

        cv2.putText(sidebar, arch_title, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
        cv2.putText(sidebar, var_title,  (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        dev_str = f"Device: {self.model_manager.device.type.upper()}"
        fps_str = f"FPS: {actual_fps:.1f} | Latency: {self.inference_latency_ms:.2f} ms"
        cv2.putText(sidebar, dev_str, (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)
        cv2.putText(sidebar, fps_str, (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # ── Per-Finger Touch Status Dashboard ────────────────────────────────
        cv2.putText(sidebar, "PER-FINGER TOUCH STATUS", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        y_start = 170
        card_h = 45

        for idx, f in enumerate(fingers):
            p_data = self.last_predictions.get(f, {"touch": False, "prob": 0.0})
            is_touch = p_data["touch"]
            prob = p_data["prob"]

            y1 = y_start + idx * (card_h + 10)
            y2 = y1 + card_h

            if not hand_detected:
                bg_color = (40, 40, 48)
                txt_color = (130, 130, 140)
                status_str = "NO HAND"
                bar_color = (60, 60, 70)
            elif is_touch:
                bg_color = (25, 80, 35)      # Bright Green tint
                txt_color = (0, 255, 128)
                status_str = f"TOUCH ({prob * 100:.0f}%)"
                bar_color = (0, 220, 100)
            else:
                bg_color = (35, 35, 60)      # Soft Blue/Gray
                txt_color = (180, 180, 220)
                status_str = f"UNTOUCH ({prob * 100:.0f}%)"
                bar_color = (120, 120, 160)

            # Draw card box
            cv2.rectangle(sidebar, (15, y1), (sidebar_w - 15, y2), bg_color, -1)
            cv2.rectangle(sidebar, (15, y1), (sidebar_w - 15, y2), txt_color, 1)

            # Label
            cv2.putText(sidebar, f.upper(), (25, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(sidebar, status_str, (150, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_color, 1)

            # Draw probability progress bar
            bar_x1, bar_y1 = 25, y1 + 28
            bar_w = sidebar_w - 50
            bar_x2 = bar_x1 + int(bar_w * prob) if hand_detected else bar_x1

            cv2.rectangle(sidebar, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + 8), (20, 20, 25), -1)
            if bar_x2 > bar_x1:
                cv2.rectangle(sidebar, (bar_x1, bar_y1), (bar_x2, bar_y1 + 8), bar_color, -1)

        # ── Footer Controls ──────────────────────────────────────────────────
        y_foot = h - 60
        cv2.rectangle(sidebar, (10, y_foot), (sidebar_w - 10, h - 10), (35, 35, 42), -1)
        cv2.putText(sidebar, "[N]/[M] Switch Model  |  [D] Device", (20, y_foot + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(sidebar, "[Q] Quit Application", (20, y_foot + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)

        return canvas

    def run(self):
        self.camera_thread.start()
        print("\n[RealtimeApp] Starting GUI loop. Press 'q' in OpenCV window to exit.")

        cv2.namedWindow("Real-Time MediaPipe Touch Detector", cv2.WINDOW_AUTOSIZE)

        try:
            while self.camera_thread.running:
                frame_data = self.camera_thread.get_latest_frame_data()
                annotated_frame, detected, actual_fps, w, h = frame_data

                if annotated_frame is None:
                    time.sleep(0.02)
                    continue

                canvas = self.draw_hud(annotated_frame, actual_fps, detected)
                cv2.imshow("Real-Time MediaPipe Touch Detector", canvas)

                key = cv2.waitKey(20) & 0xFF
                if key == ord('q') or key == 27:
                    print("[RealtimeApp] Quit requested by user.")
                    break
                elif key == ord('n') or key == ord('m'):
                    self.model_manager.switch_next_model()
                elif key == ord('d'):
                    cur_dev = self.model_manager.device.type
                    new_dev = "cpu" if cur_dev == "cuda" else "cuda"
                    self.model_manager.set_device(new_dev)

        finally:
            self.camera_thread.stop()
            cv2.destroyAllWindows()
            print("[RealtimeApp] Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time MediaPipe Touch Detector HUD Application.")
    parser.add_argument("--src", default=0, help="Camera index (e.g. 0) or video file path")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Target device")
    args = parser.parse_args()

    # Try parsing int camera index
    try:
        src = int(args.src)
    except ValueError:
        src = args.src

    app = RealtimeApp(camera_src=src, device=args.device)
    app.run()


if __name__ == "__main__":
    main()
