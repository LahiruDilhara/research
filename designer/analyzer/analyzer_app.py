"""
analyzer_app.py

Main desktop application for Virtual Keyboard Layout Analyzer & Touch Detector.
Features twin containers:
- Left Panel: Always-live OpenCV Camera Feed with paper identification & live projected button overlays.
- Right Panel: Interactive 2D Paper Layout Preview with active key highlight & homography metrics.
"""

import os
import glob
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from xml_parser import parse_layout_xml
from homography_engine import HomographyEngine
from synthetic_generator import generate_synthetic_camera_frame

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def detect_available_cameras():
    """Detects connected camera devices on system silently."""
    cams = []
    video_devices = sorted(glob.glob("/dev/video*"))
    tested_indices = []

    for dev_path in video_devices:
        try:
            dev_idx = int(dev_path.replace("/dev/video", ""))
            if dev_idx in tested_indices:
                continue
            tested_indices.append(dev_idx)

            cap = cv2.VideoCapture(dev_idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cams.append(f"Camera {dev_idx}")
                cap.release()
        except Exception:
            pass

    if not cams:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cams.append("Camera 0")
                cap.release()
        except Exception:
            pass

    cams.append("Synthetic Frame Mode")
    return cams


def get_system_native_file_picker(parent_window, title="Select Layout XML File"):
    """
    Uses GTK system file manager (Zenity) or KDE (KDialog) on Linux for true OS native file picker,
    falling back to Tkinter filedialog.
    """
    # 1. Try Zenity (Linux GTK native file manager)
    try:
        cmd = ["zenity", "--file-selection", f"--title={title}", "--file-filter=XML Files (*.xml) | *.xml"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            selected = res.stdout.strip()
            if selected:
                return selected
        elif res.returncode == 1:
            return ""  # User cancelled dialog
    except Exception:
        pass

    # 2. Try KDialog (Linux KDE native file manager)
    try:
        cmd = ["kdialog", "--getopenfilename", ".", "*.xml"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            selected = res.stdout.strip()
            if selected:
                return selected
        elif res.returncode == 1:
            return ""
    except Exception:
        pass

    # 3. Standard Tkinter filedialog fallback
    return filedialog.askopenfilename(
        parent=parent_window,
        title=title,
        filetypes=[("XML Files", "*.xml"), ("All Files", "*.*")]
    )


class AnalyzerApp(ctk.CTk):
    def __init__(self, initial_xml=None):
        super().__init__()

        self.title("Virtual Keyboard Layout Analyzer & Live Camera Inspector")
        self.geometry("1400x880")
        self.minsize(1100, 700)

        # State Variables
        self.layout_data = None
        self.homography_engine = None
        self.current_frame = None
        self.display_frame = None
        self.cap = None
        self.is_camera_running = False

        # Touch event state
        self.last_touch_pixel = None
        self.last_touch_paper = None
        self.last_pressed_button = None
        self.last_touch_result = None

        self._build_ui()

        # Detect connected cameras
        self.available_cams = detect_available_cameras()
        self.combo_cam_source.configure(values=self.available_cams)
        self.combo_cam_source.set(self.available_cams[0])

        # ALWAYS START LIVE CAMERA FEED IMMEDIATELY
        if self.available_cams[0].startswith("Camera"):
            cam_idx = int(self.available_cams[0].split()[-1])
            self.start_webcam(cam_idx)
        else:
            self.update_synthetic_frame()

        # Load initial XML ONLY if explicitly passed as argument
        if initial_xml and os.path.exists(initial_xml):
            self.load_xml_file(initial_xml)

    def _build_ui(self):
        # -------------------------------------------------------------
        # TOP CONTROL BAR (XML selector, Camera source controls)
        # -------------------------------------------------------------
        top_bar = ctk.CTkFrame(self, corner_radius=10, height=60)
        top_bar.pack(side="top", fill="x", padx=15, pady=(15, 10))

        btn_xml = ctk.CTkButton(
            top_bar, text="📁 Load Layout XML", command=self.on_open_xml_clicked,
            font=("Segoe UI", 13, "bold"), fg_color="#1E88E5", hover_color="#1565C0", width=170
        )
        btn_xml.pack(side="left", padx=15, pady=12)

        self.lbl_xml_info = ctk.CTkLabel(
            top_bar, text="Camera feed is live. Click 'Load Layout XML' to select your exported layout.xml file.",
            font=("Segoe UI", 12, "bold"), text_color="#FFB74D"
        )
        self.lbl_xml_info.pack(side="left", padx=10)

        # Camera Controls on Right side of Top Bar
        self.btn_toggle_cam = ctk.CTkButton(
            top_bar, text="⏹ Stop Feed", command=self.toggle_camera,
            font=("Segoe UI", 12, "bold"), fg_color="#C62828", hover_color="#8E0000", width=120
        )
        self.btn_toggle_cam.pack(side="right", padx=15, pady=12)

        self.combo_cam_source = ctk.CTkOptionMenu(
            top_bar, values=["Synthetic Frame Mode"],
            command=self.on_cam_source_changed, width=170
        )
        self.combo_cam_source.pack(side="right", padx=10)

        lbl_src = ctk.CTkLabel(top_bar, text="Camera Feed:", font=("Segoe UI", 12, "bold"))
        lbl_src.pack(side="right", padx=(10, 2))

        # -------------------------------------------------------------
        # MAIN PANELS SPLIT CONTAINER (Left: Camera, Right: Layout Preview)
        # -------------------------------------------------------------
        main_split = ctk.CTkFrame(self, fg_color="transparent")
        main_split.pack(side="top", fill="both", expand=True, padx=15, pady=(0, 15))
        main_split.columnconfigure(0, weight=6)
        main_split.columnconfigure(1, weight=5)
        main_split.rowconfigure(0, weight=1)

        # LEFT CONTAINER: OpenCV Live Camera Feed
        left_container = ctk.CTkFrame(main_split, corner_radius=12)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        lbl_cam_title = ctk.CTkLabel(
            left_container, text="📷 Live OpenCV Camera Feed (Real-Time AprilTag & Paper Identification)",
            font=("Segoe UI", 13, "bold")
        )
        lbl_cam_title.pack(side="top", anchor="w", padx=15, pady=(12, 5))

        # Camera Canvas Frame
        cam_frame = ctk.CTkFrame(left_container, fg_color="#121212", corner_radius=8)
        cam_frame.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 10))

        self.cam_canvas = tk.Canvas(
            cam_frame, bg="#121212", highlightthickness=0, cursor="crosshair"
        )
        self.cam_canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.cam_canvas.bind("<Button-1>", self.on_camera_canvas_clicked)
        self.cam_canvas.bind("<Configure>", self.on_cam_canvas_resized)

        # Status / Instructions at bottom of Camera container
        self.lbl_cam_status = ctk.CTkLabel(
            left_container, text="Camera live. Load a layout XML to start real-time paper & button tracking.",
            font=("Segoe UI", 11), text_color="#90A4AE"
        )
        self.lbl_cam_status.pack(side="bottom", anchor="w", padx=15, pady=(0, 10))

        # RIGHT CONTAINER: Layout Preview & Touch Diagnostics
        right_container = ctk.CTkFrame(main_split, corner_radius=12)
        right_container.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        lbl_layout_title = ctk.CTkLabel(
            right_container, text="🗺️ Paper Layout Inspector (Key Highlight)",
            font=("Segoe UI", 13, "bold")
        )
        lbl_layout_title.pack(side="top", anchor="w", padx=15, pady=(12, 5))

        # Paper Layout Canvas Frame
        layout_canvas_frame = ctk.CTkFrame(right_container, fg_color="#1E1E1E", corner_radius=8)
        layout_canvas_frame.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 10))

        self.layout_canvas = tk.Canvas(
            layout_canvas_frame, bg="#1E1E1E", highlightthickness=0
        )
        self.layout_canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.layout_canvas.bind("<Configure>", self.on_layout_canvas_resized)

        # Bottom Sidebar Box: Touch Metrics & Results
        self.metrics_box = ctk.CTkFrame(right_container, corner_radius=8, fg_color="#181824")
        self.metrics_box.pack(side="bottom", fill="x", padx=12, pady=(0, 10))

        self._build_metrics_sidebar()

    def _build_metrics_sidebar(self):
        # Result Header Badge
        self.lbl_result_badge = ctk.CTkLabel(
            self.metrics_box, text="WAITING FOR LAYOUT XML FILE",
            font=("Segoe UI", 12, "bold"), fg_color="#37474F", text_color="#ECEFF1", corner_radius=6, height=28
        )
        self.lbl_result_badge.pack(side="top", fill="x", padx=10, pady=(10, 8))

        grid_frame = ctk.CTkFrame(self.metrics_box, fg_color="transparent")
        grid_frame.pack(side="top", fill="x", padx=10, pady=(0, 10))
        grid_frame.columnconfigure((0, 1), weight=1)

        # Column 0: Pressed Key & Position
        lbl_key_heading = ctk.CTkLabel(grid_frame, text="PRESSED KEY", font=("Segoe UI", 10, "bold"), text_color="#78909C")
        lbl_key_heading.grid(row=0, column=0, sticky="w")

        self.lbl_key_val = ctk.CTkLabel(grid_frame, text="None", font=("Segoe UI", 16, "bold"), text_color="#00E676")
        self.lbl_key_val.grid(row=1, column=0, sticky="w")

        lbl_paper_heading = ctk.CTkLabel(grid_frame, text="PAPER MM COORD", font=("Segoe UI", 10, "bold"), text_color="#78909C")
        lbl_paper_heading.grid(row=2, column=0, sticky="w", pady=(5, 0))

        self.lbl_paper_val = ctk.CTkLabel(grid_frame, text="X: -- mm, Y: -- mm", font=("Segoe UI", 12, "bold"), text_color="#E0E0E0")
        self.lbl_paper_val.grid(row=3, column=0, sticky="w")

        # Column 1: Camera Pixel & Homography Stats
        lbl_pixel_heading = ctk.CTkLabel(grid_frame, text="CAMERA PIXEL COORD", font=("Segoe UI", 10, "bold"), text_color="#78909C")
        lbl_pixel_heading.grid(row=0, column=1, sticky="w")

        self.lbl_pixel_val = ctk.CTkLabel(grid_frame, text="x: -- px, y: -- px", font=("Segoe UI", 12, "bold"), text_color="#E0E0E0")
        self.lbl_pixel_val.grid(row=1, column=1, sticky="w")

        lbl_homo_heading = ctk.CTkLabel(grid_frame, text="HOMOGRAPHY & BUTTON STATUS", font=("Segoe UI", 10, "bold"), text_color="#78909C")
        lbl_homo_heading.grid(row=2, column=1, sticky="w", pady=(5, 0))

        self.lbl_homo_val = ctk.CTkLabel(grid_frame, text="Markers: 0 | Inliers: 0", font=("Segoe UI", 12), text_color="#B0BEC5")
        self.lbl_homo_val.grid(row=3, column=1, sticky="w")

        self.lbl_buttons_val = ctk.CTkLabel(grid_frame, text="Buttons Identified: 0 / 0", font=("Segoe UI", 12, "bold"), text_color="#00E5FF")
        self.lbl_buttons_val.grid(row=4, column=1, sticky="w")
    # -------------------------------------------------------------
    # XML LOADING LOGIC
    # -------------------------------------------------------------
    def on_open_xml_clicked(self):
        filepath = get_system_native_file_picker(self)
        if filepath:
            self.load_xml_file(filepath)

    def load_xml_file(self, xml_path):
        try:
            self.layout_data = parse_layout_xml(xml_path)
            self.homography_engine = HomographyEngine(self.layout_data, ransac_thresh_mm=5.0)

            filename = os.path.basename(xml_path)
            info_str = (
                f"Layout: {filename} ({self.layout_data.paper_width_mm}x{self.layout_data.paper_height_mm}mm) | "
                f"AprilTags: {len(self.layout_data.markers)} | Buttons: {len(self.layout_data.buttons)}"
            )
            self.lbl_xml_info.configure(text=info_str, text_color="#81C784")
            self.lbl_result_badge.configure(
                text="LAYOUT LOADED - CLICK CAMERA FEED TO TOUCH",
                fg_color="#0288D1", text_color="#FFFFFF"
            )

            # Redraw Layout Preview Canvas
            self.draw_layout_preview()

            # Refresh camera frame if synthetic mode active
            if self.combo_cam_source.get() == "Synthetic Frame Mode":
                self.update_synthetic_frame()

        except Exception as e:
            messagebox.showerror("XML Loading Error", f"Failed to parse XML file:\n{str(e)}")

    # -------------------------------------------------------------
    # REAL-TIME OPENCV CAMERA FEED LOGIC
    # -------------------------------------------------------------
    def on_cam_source_changed(self, choice):
        if choice.startswith("Camera"):
            cam_idx = int(choice.split()[-1])
            self.start_webcam(cam_idx)
        else:
            self.stop_webcam()
            self.update_synthetic_frame()

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_webcam()
        else:
            source = self.combo_cam_source.get()
            if source.startswith("Camera"):
                cam_idx = int(source.split()[-1])
                self.start_webcam(cam_idx)
            else:
                self.update_synthetic_frame()

    def start_webcam(self, camera_idx):
        self.stop_webcam()

        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Could not open Camera index {camera_idx}.")
            return

        self.is_camera_running = True
        self.btn_toggle_cam.configure(text="⏹ Stop Feed", fg_color="#C62828", hover_color="#8E0000")
        self._webcam_loop()

    def stop_webcam(self):
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_toggle_cam.configure(text="▶ Start Feed", fg_color="#2E7D32", hover_color="#1B5E20")

    def _webcam_loop(self):
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_frame = frame
            self.render_camera_frame(frame)

        self.after(30, self._webcam_loop)

    def update_synthetic_frame(self):
        if self.layout_data is None:
            # If no layout loaded yet in synthetic mode, generate blank frame
            blank_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 30
            cv2.putText(blank_frame, "SYNTHETIC FEED ACTIVE", (450, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            cv2.putText(blank_frame, "Click 'Load Layout XML' to load your layout.", (410, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            self.current_frame = blank_frame
        else:
            frame = generate_synthetic_camera_frame(self.layout_data, frame_w=1280, frame_h=720, skew_amount=0.15)
            self.current_frame = frame

        self.render_camera_frame(self.current_frame)

    def render_camera_frame(self, frame):
        if frame is None:
            return

        annotated_frame = frame.copy()

        # If homography engine is ready (layout XML loaded), process detection & projected overlays
        if self.homography_engine is not None:
            active_btn_id = self.last_pressed_button["id"] if self.last_pressed_button else None

            # Unified single-pass execution for maximum FPS performance
            H, info = self.homography_engine.process_frame(frame, active_button_id=active_btn_id)
            annotated_frame = info["annotated_frame"]

            # Update live homography & button identification UI labels
            if info["success"]:
                self.lbl_homo_val.configure(
                    text=f"Markers: {info['detected_markers_count']} | Inliers: {info['inliers_count']}"
                )
                self.lbl_buttons_val.configure(
                    text=f"Buttons Identified: {info['identified_buttons_count']} / {info['total_buttons']}",
                    text_color="#00E5FF"
                )
            else:
                self.lbl_homo_val.configure(text=f"Markers: {info['detected_markers_count']} | Inliers: 0")
                self.lbl_buttons_val.configure(
                    text=f"Buttons Identified: 0 / {info['total_buttons']}",
                    text_color="#FF5252"
                )

        # 4. If a touch click point exists, draw visual target crosshair
        if self.last_touch_pixel:
            tx, ty = self.last_touch_pixel
            cv2.circle(annotated_frame, (tx, ty), 12, (0, 230, 118), 3)
            cv2.circle(annotated_frame, (tx, ty), 3, (0, 230, 118), -1)
            cv2.line(annotated_frame, (tx - 18, ty), (tx + 18, ty), (0, 230, 118), 2)
            cv2.line(annotated_frame, (tx, ty - 18), (tx, ty + 18), (0, 230, 118), 2)

        self.display_frame = annotated_frame

        # Resize and render onto tkinter canvas
        cw = self.cam_canvas.winfo_width()
        ch = self.cam_canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 720, 480

        h, w = annotated_frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        resized = cv2.resize(annotated_frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.cam_canvas.delete("all")

        # Center image on canvas
        offset_x = (cw - nw) // 2
        offset_y = (ch - nh) // 2

        self.cam_canvas.create_image(offset_x, offset_y, anchor="nw", image=img_tk)
        self.cam_canvas.image_ref = img_tk  # Prevent garbage collection

        # Save rendering transform params for click mapping
        self.cam_img_offset = (offset_x, offset_y)
        self.cam_img_scale = scale

    def on_cam_canvas_resized(self, event):
        if self.current_frame is not None:
            self.render_camera_frame(self.current_frame)

    # -------------------------------------------------------------
    # CAMERA CLICK TOUCH DETECTION HANDLER
    # -------------------------------------------------------------
    def on_camera_canvas_clicked(self, event):
        if self.current_frame is None or self.homography_engine is None:
            messagebox.showinfo("Analyzer Info", "Please click 'Load Layout XML' first to select your layout file.")
            return

        offset_x, offset_y = getattr(self, "cam_img_offset", (0, 0))
        scale = getattr(self, "cam_img_scale", 1.0)

        # Map canvas click coordinates (event.x, event.y) back to camera image pixel coordinates
        img_pixel_x = int((event.x - offset_x) / scale)
        img_pixel_y = int((event.y - offset_y) / scale)

        h, w = self.current_frame.shape[:2]
        if 0 <= img_pixel_x < w and 0 <= img_pixel_y < h:
            # Execute Homography + Touch Detection Pipeline
            result = self.homography_engine.process_touch_event(
                self.current_frame, img_pixel_x, img_pixel_y
            )
            self._apply_touch_result(result)

    def _apply_touch_result(self, result):
        self.last_touch_result = result
        self.last_touch_pixel = result["pixel_point"]
        self.last_touch_paper = result["paper_point_mm"]
        self.last_pressed_button = result["button"]

        # Re-render camera frame to show click target overlay & pressed key highlight
        if self.current_frame is not None:
            self.render_camera_frame(self.current_frame)

        # Update Right Panel Metrics
        info = result["info"]
        px, py = result["pixel_point"]

        self.lbl_pixel_val.configure(text=f"x: {px} px, y: {py} px")
        self.lbl_homo_val.configure(
            text=f"Markers: {info['detected_markers_count']} | Inliers: {info['inliers_count']}"
        )
        self.lbl_buttons_val.configure(
            text=f"Buttons Identified: {info.get('identified_buttons_count', 0)} / {info.get('total_buttons', 0)}",
            text_color="#00E5FF" if info.get('success', False) else "#FF5252"
        )

        if not result["success"]:
            self.lbl_result_badge.configure(
                text="❌ HOMOGRAPHY FAILED (NO MARKERS DETECTED)",
                fg_color="#C62828", text_color="#FFFFFF"
            )
            self.lbl_key_val.configure(text="ERROR", text_color="#FF5252")
            self.lbl_paper_val.configure(text="X: -- mm, Y: -- mm")
            self.lbl_cam_status.configure(
                text=f"Error: {result['error_message']}", text_color="#FF5252"
            )
        else:
            paper_x, paper_y = result["paper_point_mm"]
            self.lbl_paper_val.configure(text=f"X: {paper_x:.1f} mm, Y: {paper_y:.1f} mm")

            btn = result["button"]
            if btn is not None:
                btn_name = btn["text"] if btn["text"] else btn["id"]
                self.lbl_result_badge.configure(
                    text=f"✅ KEY PRESSED: {btn_name.upper()}",
                    fg_color="#2E7D32", text_color="#FFFFFF"
                )
                self.lbl_key_val.configure(text=f"'{btn_name}'", text_color="#00E676")
                self.lbl_cam_status.configure(
                    text=f"Touch mapped to paper ({paper_x:.1f}mm, {paper_y:.1f}mm) -> Key '{btn_name}' pressed!",
                    text_color="#81C784"
                )
            else:
                self.lbl_result_badge.configure(
                    text="⚠️ TOUCHED OUTSIDE BUTTONS",
                    fg_color="#F57F17", text_color="#FFFFFF"
                )
                self.lbl_key_val.configure(text="None (Paper Margins)", text_color="#FFD54F")
                self.lbl_cam_status.configure(
                    text=f"Touch mapped to paper ({paper_x:.1f}mm, {paper_y:.1f}mm) -> No button at this coordinate.",
                    text_color="#FFD54F"
                )

        # Redraw Paper Layout Canvas to highlight pressed key
        self.draw_layout_preview()

    # -------------------------------------------------------------
    # RIGHT PANEL: 2D PAPER LAYOUT PREVIEW CANVAS
    # -------------------------------------------------------------
    def draw_layout_preview(self):
        if self.layout_data is None:
            return

        self.layout_canvas.delete("all")

        cw = self.layout_canvas.winfo_width()
        ch = self.layout_canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 600, 420

        pw_mm = self.layout_data.paper_width_mm
        ph_mm = self.layout_data.paper_height_mm

        # Scale paper to fit inside canvas with margin
        margin_px = 25
        scale = min((cw - 2 * margin_px) / pw_mm, (ch - 2 * margin_px) / ph_mm)

        paper_w_px = pw_mm * scale
        paper_h_px = ph_mm * scale

        offset_x = (cw - paper_w_px) / 2.0
        offset_y = (ch - paper_h_px) / 2.0

        def mm_to_px(x_mm, y_mm):
            return offset_x + x_mm * scale, offset_y + y_mm * scale

        # Draw A4 Paper Sheet
        px1, py1 = mm_to_px(0, 0)
        px2, py2 = mm_to_px(pw_mm, ph_mm)
        self.layout_canvas.create_rectangle(
            px1, py1, px2, py2, fill="#FFFFFF", outline="#B0BEC5", width=2
        )

        # Draw AprilTag Markers
        for m_id, m_info in self.layout_data.markers.items():
            corners = m_info["corners_mm"]
            poly_pts = []
            for (cx, cy) in corners:
                poly_pts.extend(mm_to_px(cx, cy))

            # Fill marker square with dark gray and draw ID
            self.layout_canvas.create_polygon(poly_pts, fill="#212121", outline="#000000")
            mcx, mcy = mm_to_px(m_info["center_x_mm"], m_info["center_y_mm"])
            self.layout_canvas.create_text(
                mcx, mcy, text=str(m_id), fill="#FFFFFF", font=("Segoe UI", max(7, int(7 * (scale / 2.0))), "bold")
            )

        # Draw Buttons
        for btn in self.layout_data.buttons:
            bx1, by1 = mm_to_px(btn["x_mm"], btn["y_mm"])
            bx2, by2 = mm_to_px(btn["x_max_mm"], btn["y_max_mm"])

            # Check if this button is currently pressed!
            is_pressed = (self.last_pressed_button is not None) and (self.last_pressed_button["id"] == btn["id"])

            fill_color = "#00E676" if is_pressed else "#E3F2FD"
            outline_color = "#00C853" if is_pressed else "#1565C0"
            text_color = "#000000" if is_pressed else "#0D47A1"
            line_width = 3 if is_pressed else 2

            self.layout_canvas.create_rectangle(
                bx1, by1, bx2, by2, fill=fill_color, outline=outline_color, width=line_width
            )

            # Draw button label
            text = btn["text"] if btn["text"] else btn["id"]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            font_sz = max(8, int(11 * (scale / 2.0)))
            self.layout_canvas.create_text(
                bcx, bcy, text=text, fill=text_color, font=("Segoe UI", font_sz, "bold" if is_pressed else "normal")
            )

        # Draw touch point marker on paper if available
        if self.last_touch_paper is not None:
            t_x, t_y = self.last_touch_paper
            tcx, tcy = mm_to_px(t_x, t_y)

            # Target ring
            self.layout_canvas.create_oval(tcx - 8, tcy - 8, tcx + 8, tcy + 8, outline="#FF1744", width=3)
            self.layout_canvas.create_line(tcx - 12, tcy, tcx + 12, tcy, fill="#FF1744", width=2)
            self.layout_canvas.create_line(tcx, tcy - 12, tcx, tcy + 12, fill="#FF1744", width=2)

    def on_layout_canvas_resized(self, event):
        self.draw_layout_preview()


if __name__ == "__main__":
    app = AnalyzerApp()
    app.mainloop()
