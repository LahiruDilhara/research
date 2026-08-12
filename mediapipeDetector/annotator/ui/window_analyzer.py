"""
annotator/ui/window_analyzer.py

Separate interactive Toplevel window for analyzing 5-frame window kinematics — VS Code Dark+ theme.
Renders embedded Matplotlib graphs and charts for:
- All joints position & velocity trajectories over 5 frames
- Individual finger joint analysis (MCP, PIP, DIP coordinates and velocities)
- Wrist position and velocity analysis
- Full numerical data inspection table (mousewheel scroll fixed)
"""
import logging
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from annotator.constants import FINGERS, FINGER_COLORS_HEX, JOINT_LABELS
from annotator.ui.theme import (
    FF, BG, PANEL, BORDER, HDR_BG, TXT_PRI, TXT_SEC, TXT_MUT,
    BTN_SEC, BTN_HVS, BTN_GHO, BTN_GHH,
    MPL_BG, MPL_PLOT, MPL_GRID, MPL_SPINE, MPL_TICK,
)

logger = logging.getLogger("Annotator.WindowAnalyzer")


class WindowAnalyzerDialog(ctk.CTkToplevel):
    def __init__(self, parent, window_idx: int, window_frames: list, duration_ms: int) -> None:
        super().__init__(parent)
        self.window_idx = window_idx
        self.window_frames = window_frames
        self.duration_ms = duration_ms

        sf = window_frames[0]["frame_idx"]
        ef = window_frames[-1]["frame_idx"]
        sms = window_frames[0]["timestamp_ms"]
        ems = window_frames[-1]["timestamp_ms"]

        self.title(f"Window Analyzer — Window #{window_idx + 1} (Frames {sf}–{ef}, {sms}ms–{ems}ms)")
        self.geometry("1120x760")
        self.minsize(900, 600)
        self.configure(fg_color=BG)

        self.lift()
        self.focus_force()

        self._parse_data()
        self._build_ui()

    def _parse_data(self) -> None:
        """Parse raw 5-frame window data into arrays for plotting."""
        self.frame_indices = [fd["frame_idx"] for fd in self.window_frames]
        self.timestamps_ms = [fd["timestamp_ms"] for fd in self.window_frames]
        self.f_steps = [1, 2, 3, 4, 5]
        self.v_steps = [1, 2, 3, 4]

        self.wrist_pos = []
        self.finger_pos = {fn: [[] for _ in JOINT_LABELS] for fn in FINGERS}

        for fd in self.window_frames:
            hd = fd.get("hand_data")
            if hd:
                self.wrist_pos.append(hd["wrist"])
                for fn in FINGERS:
                    jpts = hd["fingers"].get(fn, [(0.0, 0.0)] * 3)
                    for j in range(3):
                        pt = jpts[j] if j < len(jpts) else (0.0, 0.0)
                        self.finger_pos[fn][j].append(pt)
            else:
                self.wrist_pos.append((0.0, 0.0))
                for fn in FINGERS:
                    for j in range(3):
                        self.finger_pos[fn][j].append((0.0, 0.0))

        self.wrist_vel = []
        self.finger_vel = {fn: [[] for _ in JOINT_LABELS] for fn in FINGERS}

        for v_idx in range(1, 5):
            if v_idx < len(self.window_frames):
                vd = self.window_frames[v_idx].get("velocity_data")
            else:
                vd = None

            if vd:
                wv = vd.get("wrist_velocity") or (0.0, 0.0)
                self.wrist_vel.append(wv)
                fvels = vd.get("finger_velocities", {})
                for fn in FINGERS:
                    jvs = fvels.get(fn, [(0.0, 0.0)] * 3)
                    for j in range(3):
                        jv = jvs[j] if (j < len(jvs) and jvs[j] is not None) else (0.0, 0.0)
                        self.finger_vel[fn][j].append(jv)
            else:
                self.wrist_vel.append((0.0, 0.0))
                for fn in FINGERS:
                    for j in range(3):
                        self.finger_vel[fn][j].append((0.0, 0.0))

    def _build_ui(self) -> None:
        # ── Header bar ────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, height=48, corner_radius=0, fg_color=HDR_BG)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        sf = self.window_frames[0]["frame_idx"]
        ef = self.window_frames[-1]["frame_idx"]
        sms = self.window_frames[0]["timestamp_ms"]
        ems = self.window_frames[-1]["timestamp_ms"]

        ctk.CTkLabel(
            hdr,
            text=f"Window #{self.window_idx + 1} Kinematics Analyzer  ·  Frames {sf}–{ef} ({sms}ms – {ems}ms)",
            font=ctk.CTkFont(family=FF, size=15, weight="bold"),
            text_color=TXT_PRI,
        ).pack(side="left", padx=16)

        ctk.CTkButton(
            hdr, text="Close", width=80, height=28, corner_radius=4,
            border_width=1, border_color=BORDER,
            fg_color=BTN_GHO, hover_color=BTN_GHH,
            text_color=TXT_SEC,
            font=ctk.CTkFont(family=FF, size=12),
            command=self.destroy,
        ).pack(side="right", padx=16)

        # ── Tab view ──────────────────────────────────────────────────────────
        self.tabs = ctk.CTkTabview(
            self,
            fg_color=PANEL,
            segmented_button_fg_color=BG,
            segmented_button_selected_color="#007acc",
            segmented_button_selected_hover_color="#0062a3",
            segmented_button_unselected_color=BTN_SEC,
            segmented_button_unselected_hover_color=BTN_HVS,
            text_color=TXT_SEC,
            text_color_disabled=TXT_MUT,
        )
        self.tabs.pack(fill="both", expand=True, padx=12, pady=10)

        tab_all     = self.tabs.add("All Joints Overview")
        tab_fingers = self.tabs.add("Individual Finger Analysis")
        tab_wrist   = self.tabs.add("Wrist Analysis")
        tab_table   = self.tabs.add("Numerical Feature Table")

        self._build_all_joints_tab(tab_all)
        self._build_finger_tab(tab_fingers)
        self._build_wrist_tab(tab_wrist)
        self._build_table_tab(tab_table)

    def _style_axes(self, *axes):
        for ax in axes:
            ax.set_facecolor(MPL_PLOT)
            ax.tick_params(colors=MPL_TICK, labelsize=8)
            ax.grid(True, color=MPL_GRID, linestyle="--", alpha=0.6)
            for spine in ax.spines.values():
                spine.set_color(MPL_SPINE)

    # ── Tab 1: All Joints Overview ────────────────────────────────────────────

    def _build_all_joints_tab(self, parent) -> None:
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=MPL_BG)
        fig.subplots_adjust(hspace=0.35, wspace=0.25, left=0.08, right=0.95, top=0.92, bottom=0.10)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        self._style_axes(ax1, ax2, ax3, ax4)

        ax1.set_title("Fingertip X Coordinates (5 Frames)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frame Step", color=MPL_TICK, fontsize=8)
        ax1.set_ylabel("Normalized X", color=MPL_TICK, fontsize=8)
        for fn in FINGERS:
            col = FINGER_COLORS_HEX.get(fn, "#ffffff")
            xs = [pt[0] for pt in self.finger_pos[fn][2]]
            ax1.plot(self.f_steps, xs, marker="o", label=fn, color=col, linewidth=1.8)
        ax1.legend(facecolor=MPL_BG, edgecolor=MPL_GRID, labelcolor=TXT_PRI, fontsize=7)

        ax2.set_title("Fingertip Y Coordinates (5 Frames)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frame Step", color=MPL_TICK, fontsize=8)
        ax2.set_ylabel("Normalized Y", color=MPL_TICK, fontsize=8)
        for fn in FINGERS:
            col = FINGER_COLORS_HEX.get(fn, "#ffffff")
            ys = [pt[1] for pt in self.finger_pos[fn][2]]
            ax2.plot(self.f_steps, ys, marker="s", label=fn, color=col, linewidth=1.8)

        ax3.set_title("Fingertip Velocity Magnitude (4 Transitions)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax3.set_xlabel("Transition Step", color=MPL_TICK, fontsize=8)
        ax3.set_ylabel("Velocity Speed |V|", color=MPL_TICK, fontsize=8)
        for fn in FINGERS:
            col = FINGER_COLORS_HEX.get(fn, "#ffffff")
            vmags = [float(np.hypot(v[0], v[1])) for v in self.finger_vel[fn][2]]
            ax3.plot(self.v_steps, vmags, marker="^", label=fn, color=col, linewidth=1.8)

        ax4.set_title("Joint Peak Velocity Magnitudes", color=TXT_PRI, fontsize=10, fontweight="bold")
        names = ["Wrist"] + FINGERS
        w_vmag = max([float(np.hypot(v[0], v[1])) for v in self.wrist_vel] or [0.0])
        f_vmags = [max([float(np.hypot(v[0], v[1])) for v in self.finger_vel[fn][2]] or [0.0]) for fn in FINGERS]
        all_peaks = [w_vmag] + f_vmags
        colors = ["#cccccc"] + [FINGER_COLORS_HEX.get(fn, "#ffffff") for fn in FINGERS]
        bars = ax4.bar(names, all_peaks, color=colors, alpha=0.85, edgecolor=MPL_SPINE)
        ax4.set_ylabel("Peak Speed", color=MPL_TICK, fontsize=8)
        for bar in bars:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.3f}",
                     ha='center', va='bottom', color=TXT_PRI, fontsize=7)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ctk.CTkFrame(parent, height=36, fg_color=HDR_BG)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    # ── Tab 2: Individual Finger Analysis ─────────────────────────────────────

    def _build_finger_tab(self, parent) -> None:
        top_ctrl = ctk.CTkFrame(parent, height=40, fg_color="transparent")
        top_ctrl.pack(fill="x", padx=10, pady=(6, 2))

        ctk.CTkLabel(
            top_ctrl, text="Select Finger to Inspect:",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            text_color=TXT_SEC,
        ).pack(side="left", padx=(0, 10))

        self.selected_finger = ctk.StringVar(value="Index")
        for fn in FINGERS:
            ctk.CTkRadioButton(
                top_ctrl, text=fn, variable=self.selected_finger, value=fn,
                command=self._update_finger_plots,
                text_color=FINGER_COLORS_HEX.get(fn, TXT_PRI),
                font=ctk.CTkFont(family=FF, size=12, weight="bold"),
                fg_color="#007acc", hover_color="#0062a3",
                border_color=BORDER,
            ).pack(side="left", padx=10)

        self.finger_fig_container = ctk.CTkFrame(parent, fg_color="transparent")
        self.finger_fig_container.pack(fill="both", expand=True)

        self._update_finger_plots()

    def _update_finger_plots(self) -> None:
        for w in self.finger_fig_container.winfo_children():
            w.destroy()

        fn = self.selected_finger.get()
        col = FINGER_COLORS_HEX.get(fn, "#38bdf8")

        fig = Figure(figsize=(10, 6), dpi=100, facecolor=MPL_BG)
        fig.subplots_adjust(hspace=0.35, wspace=0.25, left=0.08, right=0.95, top=0.92, bottom=0.10)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        self._style_axes(ax1, ax2, ax3, ax4)

        jcolors = ["#38bdf8", "#4ade80", "#f472b6"]

        ax1.set_title(f"{fn} Joints X Coordinates (5 Frames)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frame Step", color=MPL_TICK, fontsize=8)
        ax1.set_ylabel("X Position", color=MPL_TICK, fontsize=8)
        for j, jlabel in enumerate(JOINT_LABELS):
            xs = [pt[0] for pt in self.finger_pos[fn][j]]
            ax1.plot(self.f_steps, xs, marker="o", label=jlabel, color=jcolors[j], linewidth=1.8)
        ax1.legend(facecolor=MPL_BG, edgecolor=MPL_GRID, labelcolor=TXT_PRI, fontsize=7)

        ax2.set_title(f"{fn} Joints Y Coordinates (5 Frames)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frame Step", color=MPL_TICK, fontsize=8)
        ax2.set_ylabel("Y Position", color=MPL_TICK, fontsize=8)
        for j, jlabel in enumerate(JOINT_LABELS):
            ys = [pt[1] for pt in self.finger_pos[fn][j]]
            ax2.plot(self.f_steps, ys, marker="s", label=jlabel, color=jcolors[j], linewidth=1.8)

        ax3.set_title(f"{fn} Joints Vx (4 Transitions)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax3.set_xlabel("Transition Step", color=MPL_TICK, fontsize=8)
        ax3.set_ylabel("Vx", color=MPL_TICK, fontsize=8)
        for j, jlabel in enumerate(JOINT_LABELS):
            vxs = [v[0] for v in self.finger_vel[fn][j]]
            ax3.plot(self.v_steps, vxs, marker="^", label=jlabel, color=jcolors[j], linewidth=1.8)

        ax4.set_title(f"{fn} Joints Vy (4 Transitions)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax4.set_xlabel("Transition Step", color=MPL_TICK, fontsize=8)
        ax4.set_ylabel("Vy", color=MPL_TICK, fontsize=8)
        for j, jlabel in enumerate(JOINT_LABELS):
            vys = [v[1] for v in self.finger_vel[fn][j]]
            ax4.plot(self.v_steps, vys, marker="d", label=jlabel, color=jcolors[j], linewidth=1.8)

        canvas = FigureCanvasTkAgg(fig, master=self.finger_fig_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Tab 3: Wrist Analysis ─────────────────────────────────────────────────

    def _build_wrist_tab(self, parent) -> None:
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=MPL_BG)
        fig.subplots_adjust(hspace=0.35, wspace=0.25, left=0.08, right=0.95, top=0.92, bottom=0.10)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        self._style_axes(ax1, ax2, ax3, ax4)

        w_xs = [pt[0] for pt in self.wrist_pos]
        w_ys = [pt[1] for pt in self.wrist_pos]

        ax1.set_title("Wrist Position (X, Y) over 5 Frames", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frame Step", color=MPL_TICK, fontsize=8)
        ax1.plot(self.f_steps, w_xs, marker="o", label="Wrist X", color="#38bdf8", linewidth=2.0)
        ax1.plot(self.f_steps, w_ys, marker="s", label="Wrist Y", color="#f43f5e", linewidth=2.0)
        ax1.legend(facecolor=MPL_BG, edgecolor=MPL_GRID, labelcolor=TXT_PRI, fontsize=8)

        ax2.set_title("Wrist 2D Trajectory Path (X vs Y)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax2.set_xlabel("X Position", color=MPL_TICK, fontsize=8)
        ax2.set_ylabel("Y Position (Inverted)", color=MPL_TICK, fontsize=8)
        ax2.plot(w_xs, w_ys, marker="o", color="#a855f7", linewidth=2.0, linestyle="--")
        for i, (x, y) in enumerate(zip(w_xs, w_ys)):
            ax2.annotate(f"F{i+1}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', color=TXT_PRI, fontsize=8)
        ax2.invert_yaxis()

        w_vxs = [v[0] for v in self.wrist_vel]
        w_vys = [v[1] for v in self.wrist_vel]
        ax3.set_title("Wrist Velocity (Vx, Vy) over 4 Transitions", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax3.set_xlabel("Transition Step", color=MPL_TICK, fontsize=8)
        ax3.plot(self.v_steps, w_vxs, marker="^", label="Wrist Vx", color="#38bdf8", linewidth=2.0)
        ax3.plot(self.v_steps, w_vys, marker="d", label="Wrist Vy", color="#f43f5e", linewidth=2.0)
        ax3.legend(facecolor=MPL_BG, edgecolor=MPL_GRID, labelcolor=TXT_PRI, fontsize=8)

        w_mags = [float(np.hypot(v[0], v[1])) for v in self.wrist_vel]
        ax4.set_title("Wrist Overall Speed |V| (4 Transitions)", color=TXT_PRI, fontsize=10, fontweight="bold")
        ax4.set_xlabel("Transition Step", color=MPL_TICK, fontsize=8)
        ax4.plot(self.v_steps, w_mags, marker="o", color="#f59e0b", linewidth=2.2)
        ax4.fill_between(self.v_steps, w_mags, color="#f59e0b", alpha=0.2)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Tab 4: Numerical Feature Table ───────────────────────────────────────

    def _build_table_tab(self, parent) -> None:
        scroll = ctk.CTkScrollableFrame(
            parent,
            fg_color=BG,
            scrollbar_button_color=BTN_SEC,
            scrollbar_button_hover_color=BTN_HVS,
        )
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            scroll, text="Numerical Window Feature Extractor Table",
            font=ctk.CTkFont(family=FF, size=14, weight="bold"),
            text_color=TXT_PRI,
        ).pack(anchor="w", pady=(0, 10))

        # 160 Coordinates
        ctk.CTkLabel(
            scroll, text="1. 160 Landmark Coordinates (5 Frames × 16 Joints × 2)",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            text_color="#4ec9f0",
        ).pack(anchor="w", pady=(8, 4))

        coord_frame = ctk.CTkFrame(scroll, fg_color=PANEL, corner_radius=4, border_width=1, border_color=BORDER)
        coord_frame.pack(fill="x", pady=4)

        hdr_row = ctk.CTkFrame(coord_frame, fg_color=HDR_BG)
        hdr_row.pack(fill="x")
        ctk.CTkLabel(hdr_row, text="Joint", width=130, anchor="w",
                     font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_PRI).pack(side="left", padx=6, pady=4)
        for f in range(1, 6):
            ctk.CTkLabel(hdr_row, text=f"Frame {f} (X, Y)", width=155,
                         font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_PRI).pack(side="left", padx=4)

        r_wrist = ctk.CTkFrame(coord_frame, fg_color="transparent")
        r_wrist.pack(fill="x")
        ctk.CTkLabel(r_wrist, text="Wrist", width=130, anchor="w",
                     font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_SEC).pack(side="left", padx=6, pady=2)
        for pt in self.wrist_pos:
            ctk.CTkLabel(r_wrist, text=f"({pt[0]:.4f}, {pt[1]:.4f})", width=155,
                         font=ctk.CTkFont(family=FF, size=11), text_color=TXT_SEC).pack(side="left", padx=4)

        for fn in FINGERS:
            for j, jlabel in enumerate(JOINT_LABELS):
                row = ctk.CTkFrame(coord_frame, fg_color="transparent")
                row.pack(fill="x")
                ctk.CTkLabel(row, text=f"{fn} {jlabel}", width=130, anchor="w",
                             font=ctk.CTkFont(family=FF, size=11), text_color=TXT_MUT).pack(side="left", padx=6, pady=2)
                for pt in self.finger_pos[fn][j]:
                    ctk.CTkLabel(row, text=f"({pt[0]:.4f}, {pt[1]:.4f})", width=155,
                                 font=ctk.CTkFont(family=FF, size=11), text_color=TXT_SEC).pack(side="left", padx=4)

        # 128 Velocities
        ctk.CTkLabel(
            scroll, text="2. 128 Joint Velocities (4 Transitions × 16 Joints × 2)",
            font=ctk.CTkFont(family=FF, size=12, weight="bold"),
            text_color="#f47474",
        ).pack(anchor="w", pady=(16, 4))

        vel_frame = ctk.CTkFrame(scroll, fg_color=PANEL, corner_radius=4, border_width=1, border_color=BORDER)
        vel_frame.pack(fill="x", pady=4)

        v_hdr_row = ctk.CTkFrame(vel_frame, fg_color=HDR_BG)
        v_hdr_row.pack(fill="x")
        ctk.CTkLabel(v_hdr_row, text="Joint", width=130, anchor="w",
                     font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_PRI).pack(side="left", padx=6, pady=4)
        for t in range(1, 5):
            ctk.CTkLabel(v_hdr_row, text=f"Transition {t} (Vx, Vy)", width=175,
                         font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_PRI).pack(side="left", padx=4)

        rv_wrist = ctk.CTkFrame(vel_frame, fg_color="transparent")
        rv_wrist.pack(fill="x")
        ctk.CTkLabel(rv_wrist, text="Wrist", width=130, anchor="w",
                     font=ctk.CTkFont(family=FF, size=11, weight="bold"), text_color=TXT_SEC).pack(side="left", padx=6, pady=2)
        for v in self.wrist_vel:
            ctk.CTkLabel(rv_wrist, text=f"({v[0]:.4f}, {v[1]:.4f})", width=175,
                         font=ctk.CTkFont(family=FF, size=11), text_color=TXT_SEC).pack(side="left", padx=4)

        for fn in FINGERS:
            for j, jlabel in enumerate(JOINT_LABELS):
                row = ctk.CTkFrame(vel_frame, fg_color="transparent")
                row.pack(fill="x")
                ctk.CTkLabel(row, text=f"{fn} {jlabel}", width=130, anchor="w",
                             font=ctk.CTkFont(family=FF, size=11), text_color=TXT_MUT).pack(side="left", padx=6, pady=2)
                for v in self.finger_vel[fn][j]:
                    ctk.CTkLabel(row, text=f"({v[0]:.4f}, {v[1]:.4f})", width=175,
                                 font=ctk.CTkFont(family=FF, size=11), text_color=TXT_SEC).pack(side="left", padx=4)
