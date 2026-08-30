"""
realtimeprocess/main_realtime_ui.py

Native Desktop Real-Time Touch Detection HUD Application (CustomTkinter GUI).

Architecture:
1. Left Container: Resizable video feed display canvas showing live MediaPipe hand tracking stream.
2. Right Native UI Sidebar: Native CustomTkinter control panel & dashboard:
   - Live Model Architecture Dropdown selector.
   - Execution Device Switcher (CUDA / CPU).
   - Real-Time FPS and Inference Latency badges.
   - 5 Native Per-Finger Touch Status Cards (Thumb, Index, Middle, Ring, Pinky) with CTkProgressBars.
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtimeprocess.model_manager import ModelManager
from realtimeprocess.camera_thread import CameraThread

# Set CustomTkinter Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class RealtimeTkApp(ctk.CTk):
    """Native CustomTkinter Desktop Application for Real-Time Touch Gesture Detection."""

    def __init__(self, camera_src=0, device: str = None):
        super().__init__()

        print("\n" + "="*80)
        print("  REAL-TIME TOUCH GESTURE DETECTION NATIVE DESKTOP HUD")
        print("="*80)

        self.title("Real-Time MediaPipe Touch Gesture Detector")
        self.geometry("1280x720")
        self.minsize(960, 540)

        # Model Manager & Async Window Shift State
        self.model_manager = ModelManager(device=device)
        self.last_predictions = {f: {"touch": False, "prob": 0.0} for f in ["thumb", "index", "middle", "ring", "pinky"]}
        self.inference_latency_ms = 0.0

        # Camera Thread (12 FPS continuous capture)
        self.camera_thread = CameraThread(
            src=camera_src,
            target_fps=12.0,
            callback=self.on_window_shift_trigger
        )

        # ── Configure Grid Layout (Column 0: Video, Column 1: Native Sidebar) ─
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ── 1. Left Container: Resizable Live Video Feed ─────────────────────
        self.video_container = ctk.CTkFrame(self, corner_radius=0, fg_color="#101014")
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # ── 2. Right Container: Native UI Sidebar Controls ───────────────────
        self.sidebar = ctk.CTkFrame(self, width=380, corner_radius=0, fg_color="#18181C")
        self.sidebar.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)

        # Header Title
        self.lbl_title = ctk.CTkLabel(
            self.sidebar,
            text="TOUCH DETECTOR HUD",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FFFFFF"
        )
        self.lbl_title.pack(padx=20, pady=(20, 15), anchor="w")

        # Active Model Selector Dropdown
        self.lbl_model_select = ctk.CTkLabel(
            self.sidebar,
            text="Active Model Architecture:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#A0A0B0"
        )
        self.lbl_model_select.pack(padx=20, pady=(5, 2), anchor="w")

        model_titles = [m["display_title"] for m in self.model_manager.available_models]
        self.option_model = ctk.CTkOptionMenu(
            self.sidebar,
            values=model_titles if model_titles else ["No Models Found"],
            command=self.on_model_dropdown_select,
            fg_color="#282832",
            button_color="#3A3A4A",
            button_hover_color="#00A0E0",
            dropdown_fg_color="#1E1E24"
        )
        if model_titles:
            active_t = self.model_manager.active_info["display_title"]
            self.option_model.set(active_t)
        self.option_model.pack(padx=20, pady=(0, 8), fill="x")

        # Variant Subtitle Label
        self.lbl_variant_info = ctk.CTkLabel(
            self.sidebar,
            text="Variant: --",
            font=ctk.CTkFont(size=12),
            text_color="#00D7FF"
        )
        self.lbl_variant_info.pack(padx=20, pady=(0, 15), anchor="w")

        # Execution Device Selector (CUDA / CPU)
        self.lbl_device_select = ctk.CTkLabel(
            self.sidebar,
            text="Execution Device:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#A0A0B0"
        )
        self.lbl_device_select.pack(padx=20, pady=(5, 2), anchor="w")

        self.seg_device = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["CUDA", "CPU"],
            command=self.on_device_segment_select,
            selected_color="#00A0E0",
            selected_hover_color="#0080B0"
        )
        cur_dev_str = "CUDA" if self.model_manager.device.type == "cuda" else "CPU"
        self.seg_device.set(cur_dev_str)
        self.seg_device.pack(padx=20, pady=(0, 15), fill="x")

        # Performance Stats Badge Panel (FPS & Latency)
        self.perf_frame = ctk.CTkFrame(self.sidebar, fg_color="#22222A", corner_radius=6)
        self.perf_frame.pack(padx=20, pady=(0, 15), fill="x")

        self.lbl_fps = ctk.CTkLabel(
            self.perf_frame,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00FF80"
        )
        self.lbl_fps.pack(side="left", padx=15, pady=8)

        self.lbl_latency = ctk.CTkLabel(
            self.perf_frame,
            text="Latency: 0.00 ms",
            font=ctk.CTkFont(size=12),
            text_color="#CCCCCC"
        )
        self.lbl_latency.pack(side="right", padx=15, pady=8)

        # ── 3. Native Per-Finger Touch Status Dashboard ──────────────────────
        self.lbl_cards_title = ctk.CTkLabel(
            self.sidebar,
            text="PER-FINGER TOUCH STATUS",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FFFFFF"
        )
        self.lbl_cards_title.pack(padx=20, pady=(5, 10), anchor="w")

        self.finger_cards = {}
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            card_frame = ctk.CTkFrame(self.sidebar, fg_color="#22222A", corner_radius=8)
            card_frame.pack(padx=20, pady=5, fill="x")

            card_header = ctk.CTkFrame(card_frame, fg_color="transparent")
            card_header.pack(fill="x", padx=12, pady=(8, 4))

            lbl_name = ctk.CTkLabel(
                card_header,
                text=finger.upper(),
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#FFFFFF"
            )
            lbl_name.pack(side="left")

            lbl_status = ctk.CTkLabel(
                card_header,
                text="NO HAND",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#808090"
            )
            lbl_status.pack(side="right")

            pbar = ctk.CTkProgressBar(card_frame, height=8, progress_color="#00DC64")
            pbar.set(0.0)
            pbar.pack(fill="x", padx=12, pady=(0, 8))

            self.finger_cards[finger] = {
                "frame": card_frame,
                "status": lbl_status,
                "pbar": pbar
            }

        # ── 4. Footer Controls ────────────────────────────────────────────────
        self.lbl_footer = ctk.CTkLabel(
            self.sidebar,
            text="Hotkeys: [N]/[M] Switch Model  |  [D] Device  |  [Q] Quit",
            font=ctk.CTkFont(size=11),
            text_color="#808090"
        )
        self.lbl_footer.pack(side="bottom", padx=20, pady=15)

        # Bind Keypresses
        self.bind("<Key>", self.on_keypress)

        # Start Camera Thread
        self.camera_thread.start()

        # Update initial model info
        self.update_active_model_labels()

        # Start GUI Loop
        self.after(20, self.update_gui_loop)

    def on_window_shift_trigger(self, window_5_frames, scores_5, frame_w, frame_h):
        """Triggered asynchronously whenever a 2-frame shift occurs on a full 5-frame buffer."""
        t0 = time.perf_counter()
        preds = self.model_manager.predict_window(window_5_frames, scores_5, frame_w, frame_h)
        latency = (time.perf_counter() - t0) * 1000.0

        self.last_predictions = preds
        self.inference_latency_ms = latency

    def on_model_dropdown_select(self, selected_title: str):
        """Dropdown handler to switch active model by display title."""
        for idx, m in enumerate(self.model_manager.available_models):
            if m["display_title"] == selected_title:
                self.model_manager.load_model_by_index(idx)
                self.camera_thread.reset_buffers()
                self.update_active_model_labels()
                break

    def on_device_segment_select(self, selected_dev: str):
        """Segmented button handler to switch device between CUDA and CPU."""
        self.model_manager.set_device(selected_dev.lower())
        cur_dev_str = "CUDA" if self.model_manager.device.type == "cuda" else "CPU"
        self.seg_device.set(cur_dev_str)

    def update_active_model_labels(self):
        """Refreshes sidebar labels with active model metadata."""
        info = self.model_manager.active_info
        if info:
            self.option_model.set(info["display_title"])
            self.lbl_variant_info.configure(text=f"Variant: {info['variant_name']} (Dim: {info['feature_dim']})")

    def on_keypress(self, event):
        """Keyboard event handler for quick hotkey navigation."""
        key = event.char.lower() if event.char else ""
        if key == "q":
            self.on_closing()
        elif key in ("n", "m"):
            self.model_manager.switch_next_model()
            self.camera_thread.reset_buffers()
            self.update_active_model_labels()
        elif key == "d":
            cur_dev = self.model_manager.device.type
            new_dev = "cpu" if cur_dev == "cuda" else "cuda"
            self.model_manager.set_device(new_dev)
            self.seg_device.set("CUDA" if self.model_manager.device.type == "cuda" else "CPU")

    def update_gui_loop(self):
        """Main Tkinter GUI update loop operating at ~50 Hz."""
        if not self.camera_thread.running:
            return

        frame_data = self.camera_thread.get_latest_frame_data()
        annotated_frame, detected, actual_fps, w, h = frame_data

        if annotated_frame is not None:
            # 1. Update Video Feed in Left Container (keep aspect ratio matching container)
            container_w = max(100, self.video_container.winfo_width())
            container_h = max(100, self.video_container.winfo_height())

            # Calculate scaled image dimensions preserving aspect ratio
            img_aspect = w / float(h)
            container_aspect = container_w / float(container_h)

            if container_aspect > img_aspect:
                new_h = container_h
                new_w = int(container_h * img_aspect)
            else:
                new_w = container_w
                new_h = int(container_w / img_aspect)

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
            self.video_label.configure(image=ctk_img)

            # 2. Update Performance Badges
            self.lbl_fps.configure(text=f"FPS: {actual_fps:.1f}")
            self.lbl_latency.configure(text=f"Latency: {self.inference_latency_ms:.2f} ms")

            # 3. Update Native Per-Finger Touch Status Cards
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                card = self.finger_cards[finger]
                p_data = self.last_predictions.get(finger, {"touch": False, "prob": 0.0})
                is_touch = p_data["touch"]
                prob = p_data["prob"]

                if not detected:
                    card["frame"].configure(fg_color="#22222A")
                    card["status"].configure(text="NO HAND", text_color="#808090")
                    card["pbar"].configure(progress_color="#505060")
                    card["pbar"].set(0.0)
                elif is_touch:
                    card["frame"].configure(fg_color="#183C24")       # Dark Green tint
                    card["status"].configure(text=f"TOUCH ({prob*100:.0f}%)", text_color="#00FF80")
                    card["pbar"].configure(progress_color="#00DC64")   # Vibrant Green
                    card["pbar"].set(prob)
                else:
                    card["frame"].configure(fg_color="#222238")       # Deep Blue/Gray
                    card["status"].configure(text=f"UNTOUCH ({prob*100:.0f}%)", text_color="#B4B4DC")
                    card["pbar"].configure(progress_color="#646496")   # Soft Blue/Gray
                    card["pbar"].set(prob)

        self.after(20, self.update_gui_loop)

    def on_closing(self):
        """Clean shutdown handler."""
        print("[RealtimeTkApp] Shutting down application...")
        self.camera_thread.stop()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description="Real-Time MediaPipe Touch Gesture Detector Native Desktop Application.")
    parser.add_argument("--src", default=0, help="Camera index (e.g. 0) or video file path")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Target execution device")
    args = parser.parse_args()

    try:
        src = int(args.src)
    except ValueError:
        src = args.src

    app = RealtimeTkApp(camera_src=src, device=args.device)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
