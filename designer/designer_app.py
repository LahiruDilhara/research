"""
designer_app.py

Drag-and-drop A4-landscape button layout designer.
Run this file directly: python3 designer_app.py

- Markers are auto-placed around the border (not editable) — see
  layout_constants.py to change marker size/spacing/margin.
- The interior area (dashed guide) is where buttons can be placed.
- Click "Add Button" to create one, drag its body to move it, drag
  its corner handles to resize it. Buttons cannot overlap or leave
  the interior area — an invalid move/resize is simply rejected.
- Edit text / font size in the side panel for the selected button.
- Save/Load keeps an editable project file (JSON).
- Export XML / Export PDF produce the final files for printing and
  for your runtime touch-detection app.
"""

import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk


def get_native_file_dialog(mode="open", title="Select File", default_ext="", filetypes=None):
    """
    Attempts to use GTK system file manager (Zenity) on Linux for native OS file selection,
    falling back to Tkinter filedialog if unavailable.
    """
    if filetypes is None:
        filetypes = []
    
    # Try Zenity native GTK system dialog on Linux
    try:
        cmd = ["zenity", "--file-selection", f"--title={title}"]
        if mode == "save":
            cmd.append("--save")
            cmd.append("--confirm-overwrite")
            if default_ext:
                cmd.append(f"--filename=project{default_ext}")
        
        # Add file filter if specified
        if filetypes:
            for desc, glob_pattern in filetypes:
                cmd.append(f"--file-filter={desc} | {glob_pattern}")
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            selected_path = res.stdout.strip()
            if mode == "save" and default_ext and not selected_path.endswith(default_ext):
                selected_path += default_ext
            return selected_path
        elif res.returncode == 1:
            return ""  # User cancelled
    except Exception:
        pass

    # Fallback to standard Tkinter filedialog
    if mode == "save":
        return filedialog.asksaveasfilename(title=title, defaultextension=default_ext, filetypes=filetypes)
    else:
        return filedialog.askopenfilename(title=title, filetypes=filetypes)

from PIL import ImageTk
from layout_constants import (
    PAPER_WIDTH_MM, PAPER_HEIGHT_MM, MARKER_SIZE_MM,
    BUTTON_MIN_WIDTH_MM, BUTTON_MIN_HEIGHT_MM, BUTTON_DEFAULT_FONT_SIZE_PT,
    BUTTON_MIN_GAP_MM,
    INTERIOR_X_MIN, INTERIOR_X_MAX, INTERIOR_Y_MIN, INTERIOR_Y_MAX,
    generate_marker_layout, rects_overlap, rect_within_interior,
)
from project_io import save_project, load_project, export_xml, export_pdf, generate_preview_image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PX_PER_MM = 3.0
HANDLE_SIZE_PX = 9  # resize-handle square size, in screen pixels

# Grid Constants & Default Settings
GRID_DEFAULT_SIZE_MM = 10.0
GRID_MIN_SIZE_MM = 2.0
GRID_MAX_SIZE_MM = 50.0


def mm_to_px(v):
    return v * PX_PER_MM


def px_to_mm(v):
    return v / PX_PER_MM


class Button:
    """One design-time button. Geometry stored in mm; canvas item ids
    for redrawing."""
    _next_num = 1

    def __init__(self, x_mm, y_mm, width_mm, height_mm, text=None, font_size_pt=None):
        self.id = f"button_{Button._next_num}"
        Button._next_num += 1
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.text = text if text is not None else self.id
        self.font_size_pt = font_size_pt if font_size_pt is not None else BUTTON_DEFAULT_FONT_SIZE_PT

        # canvas item ids, set when drawn
        self.rect_item = None
        self.text_item = None
        self.handle_items = []  # 4 corner handles, only shown when selected

    def rect_mm(self):
        return (self.x_mm, self.y_mm, self.width_mm, self.height_mm)

    def to_dict(self):
        return {
            "id": self.id, "x_mm": self.x_mm, "y_mm": self.y_mm,
            "width_mm": self.width_mm, "height_mm": self.height_mm,
            "text": self.text, "font_size_pt": self.font_size_pt,
        }

    @staticmethod
    def from_dict(d):
        b = Button(d["x_mm"], d["y_mm"], d["width_mm"], d["height_mm"],
                   d.get("text"), d.get("font_size_pt"))
        b.id = d["id"]
        return b


class DesignerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Paper Button Layout Designer")
        self.geometry(f"{int(mm_to_px(PAPER_WIDTH_MM)) + 360}x{int(mm_to_px(PAPER_HEIGHT_MM)) + 80}")

        self.buttons = []          # list[Button]
        self.selected = None       # Button or None
        self.drag_mode = None      # None | "move" | "resize"
        self.resize_corner = None  # "tl","tr","br","bl"
        self.drag_start = None     # (mouse_x_mm, mouse_y_mm)
        self.drag_orig_rect = None

        # Undo / Redo state stacks
        self.undo_stack = []
        self.redo_stack = []
        self._is_undo_redo_action = False

        # Grid settings
        self.show_grid = tk.BooleanVar(value=True)
        self.snap_to_grid = tk.BooleanVar(value=True)
        self.grid_size_mm = float(GRID_DEFAULT_SIZE_MM)
        self.grid_size_var = tk.StringVar(value=str(int(self.grid_size_mm)))

        self._build_ui()
        self._bind_shortcuts()
        self._draw_static_layout()
        self.save_state()  # save initial empty state

    # -----------------------------------------------------------
    # UI CONSTRUCTION
    # -----------------------------------------------------------
    def _build_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=330, corner_radius=12)
        self.sidebar.pack(side="left", fill="y", padx=12, pady=12)

        ctk.CTkLabel(self.sidebar, text="Layout Designer", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(12, 12))

        # Undo / Redo Buttons
        undo_redo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        undo_redo_frame.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkButton(undo_redo_frame, text="↶ Undo (Ctrl+Z)", fg_color="gray30", hover_color="gray40", command=self.undo).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ctk.CTkButton(undo_redo_frame, text="↷ Redo (Ctrl+Y)", fg_color="gray30", hover_color="gray40", command=self.redo).pack(side="right", fill="x", expand=True, padx=(2, 0))

        ctk.CTkButton(self.sidebar, text="+ Add Button", font=ctk.CTkFont(weight="bold"), command=self.add_button).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(self.sidebar, text="Duplicate Selected (Ctrl+D)", fg_color="#27AE60", hover_color="#1E8449", font=ctk.CTkFont(weight="bold"),
                      command=self.duplicate_selected).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(self.sidebar, text="Delete Selected (Ctrl+X / Del)", fg_color="#C0392B", hover_color="#962D22", font=ctk.CTkFont(weight="bold"),
                      command=self.delete_selected).pack(fill="x", padx=12, pady=4)

        sep0 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray25")
        sep0.pack(fill="x", padx=10, pady=8)

        # Grid Controls Section
        ctk.CTkLabel(self.sidebar, text="Grid Settings", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(0, 6))

        grid_switch_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        grid_switch_frame.pack(fill="x", padx=12, pady=2)

        ctk.CTkSwitch(grid_switch_frame, text="Show Grid", variable=self.show_grid, command=self.toggle_grid).pack(side="left")
        ctk.CTkSwitch(grid_switch_frame, text="Snap to Grid", variable=self.snap_to_grid).pack(side="right")

        self._labeled_entry("Grid Size (mm)", self.grid_size_var)
        ctk.CTkButton(self.sidebar, text="Update Grid", fg_color="gray30", hover_color="gray40", command=self.update_grid_size).pack(fill="x", padx=12, pady=(4, 6))

        sep1 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray25")
        sep1.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(self.sidebar, text="Selected Button", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(0, 6))

        self.text_var = tk.StringVar()
        self.font_var = tk.StringVar()
        self.width_var = tk.StringVar()
        self.height_var = tk.StringVar()

        self._labeled_entry("Text", self.text_var)
        self._labeled_entry("Font size (pt)", self.font_var)
        self._labeled_entry("Width (mm)", self.width_var)
        self._labeled_entry("Height (mm)", self.height_var)

        ctk.CTkButton(self.sidebar, text="Apply Changes", command=self.apply_property_changes).pack(fill="x", padx=12, pady=(6, 4))

        sep2 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray25")
        sep2.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(self.sidebar, text="Project", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(0, 6))
        ctk.CTkButton(self.sidebar, text="Save Project (.json)", command=self.on_save_project).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(self.sidebar, text="Load Project (.json)", command=self.on_load_project).pack(fill="x", padx=12, pady=4)

        sep3 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray25")
        sep3.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(self.sidebar, text="Export & Preview", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(0, 6))
        ctk.CTkButton(self.sidebar, text="👁 Preview Final Print", fg_color="#6366F1", hover_color="#4F46E5", font=ctk.CTkFont(weight="bold"), command=self.on_show_preview).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(self.sidebar, text="Export XML", command=self.on_export_xml).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(self.sidebar, text="Export PDF", command=self.on_export_pdf).pack(fill="x", padx=12, pady=4)

        self.status_label = ctk.CTkLabel(self.sidebar, text="", text_color="gray70", wraplength=280, justify="left")
        self.status_label.pack(side="bottom", pady=10, padx=10)

        # Canvas area
        canvas_w = int(mm_to_px(PAPER_WIDTH_MM))
        canvas_h = int(mm_to_px(PAPER_HEIGHT_MM))
        canvas_frame = ctk.CTkFrame(self, corner_radius=12)
        canvas_frame.pack(side="left", padx=12, pady=12, fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=canvas_w, height=canvas_h,
                                 bg="#1E1E2E", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)

        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)


    def _bind_shortcuts(self):
        # Keyboard Shortcuts
        self.bind("<Control-w>", lambda e: self.close_application())
        self.bind("<Control-W>", lambda e: self.close_application())
        self.bind("<Control-s>", lambda e: self.on_save_project())
        self.bind("<Control-S>", lambda e: self.on_save_project())
        self.bind("<Control-o>", lambda e: self.on_load_project())
        self.bind("<Control-O>", lambda e: self.on_load_project())
        self.bind("<Escape>", lambda e: self.close_open_windows())

        self.bind("<Control-d>", lambda e: self.duplicate_selected())
        self.bind("<Control-D>", lambda e: self.duplicate_selected())
        self.bind("<Control-x>", lambda e: self.delete_selected())
        self.bind("<Control-X>", lambda e: self.delete_selected())
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-Z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-Y>", lambda e: self.redo())
        self.bind("<Control-Shift-Z>", lambda e: self.redo())
        self.bind("<Control-Shift-z>", lambda e: self.redo())

    def close_application(self):
        self.destroy()

    def close_open_windows(self):
        # Close any open CTk Toplevel windows (dialogs, input windows, etc.)
        top_windows = [child for child in self.children.values() if isinstance(child, (ctk.CTkToplevel, tk.Toplevel))]
        if top_windows:
            for win in top_windows:
                try:
                    win.destroy()
                except Exception:
                    pass
            self._set_status("Closed open dialog.")
            return

        # Deselect any selected button and return focus to main window
        if self.selected:
            self.select_button(None)
            self._set_status("Deselected button.")


    def _labeled_entry(self, label, var):
        row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def _set_status(self, msg):
        self.status_label.configure(text=msg)

    def snap_value(self, val):
        if self.snap_to_grid.get() and self.grid_size_mm > 0:
            return round(val / self.grid_size_mm) * self.grid_size_mm
        return val

    def update_grid_size(self):
        try:
            val = float(self.grid_size_var.get())
            if val < GRID_MIN_SIZE_MM or val > GRID_MAX_SIZE_MM:
                messagebox.showerror("Invalid Grid Size", f"Grid size must be between {GRID_MIN_SIZE_MM}mm and {GRID_MAX_SIZE_MM}mm.")
                return
            self.grid_size_mm = val
            self._draw_static_layout()
            self._set_status(f"Grid size updated to {val}mm.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric value for grid size.")

    def toggle_grid(self):
        self._draw_static_layout()

    # -----------------------------------------------------------
    # STATIC LAYOUT (paper background + grid lines + markers + interior guide)
    # -----------------------------------------------------------
    def _draw_static_layout(self):
        c = self.canvas
        c.delete("static")

        # Paper background (Clean modern paper white/off-white)
        c.create_rectangle(0, 0, mm_to_px(PAPER_WIDTH_MM), mm_to_px(PAPER_HEIGHT_MM),
                            fill="#FAFAFA", outline="", tags="static")

        # Grid lines (draw anchored to paper origin (0,0) so lines perfectly match snap_value)
        if self.show_grid.get() and self.grid_size_mm > 0:
            # Vertical grid lines
            x = 0.0
            while x <= PAPER_WIDTH_MM:
                if INTERIOR_X_MIN <= x <= INTERIOR_X_MAX:
                    px = mm_to_px(x)
                    c.create_line(px, mm_to_px(INTERIOR_Y_MIN), px, mm_to_px(INTERIOR_Y_MAX),
                                  fill="#E2E8F0", width=1, tags="static")
                x += self.grid_size_mm

            # Horizontal grid lines
            y = 0.0
            while y <= PAPER_HEIGHT_MM:
                if INTERIOR_Y_MIN <= y <= INTERIOR_Y_MAX:
                    py = mm_to_px(y)
                    c.create_line(mm_to_px(INTERIOR_X_MIN), py, mm_to_px(INTERIOR_X_MAX), py,
                                  fill="#E2E8F0", width=1, tags="static")
                y += self.grid_size_mm

        # Interior guide line (Modern sleek indigo accent line)
        c.create_rectangle(
            mm_to_px(INTERIOR_X_MIN), mm_to_px(INTERIOR_Y_MIN),
            mm_to_px(INTERIOR_X_MAX), mm_to_px(INTERIOR_Y_MAX),
            outline="#6366F1", dash=(6, 4), width=1.5, tags="static"
        )

        # AprilTag Markers around the border
        for m in generate_marker_layout():
            half = MARKER_SIZE_MM / 2.0
            x0 = mm_to_px(m["x_mm"] - half)
            y0 = mm_to_px(m["y_mm"] - half)
            x1 = mm_to_px(m["x_mm"] + half)
            y1 = mm_to_px(m["y_mm"] + half)
            c.create_rectangle(x0, y0, x1, y1, fill="#0F172A", outline="", tags="static")
            c.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(m["id"]),
                          fill="#F8FAFC", font=("Segoe UI", 8, "bold"), tags="static")

        # Keep static layer behind buttons
        c.tag_lower("static")

    # -----------------------------------------------------------
    # BUTTON DRAWING
    # -----------------------------------------------------------
    def _redraw_button(self, b):
        c = self.canvas
        if b.rect_item:
            c.delete(b.rect_item)
        if b.text_item:
            c.delete(b.text_item)
        for h in b.handle_items:
            c.delete(h)
        b.handle_items = []

        x0, y0 = mm_to_px(b.x_mm), mm_to_px(b.y_mm)
        x1, y1 = mm_to_px(b.x_mm + b.width_mm), mm_to_px(b.y_mm + b.height_mm)

        is_selected = (self.selected is b)
        bg_fill = "#FFFFFF" if not is_selected else "#EEF2FF"
        outline_color = "#4F46E5" if is_selected else "#94A3B8"
        outline_w = 2.5 if is_selected else 1.5
        text_color = "#312E81" if is_selected else "#1E293B"

        b.rect_item = c.create_rectangle(x0, y0, x1, y1, fill=bg_fill,
                                          outline=outline_color, width=outline_w,
                                          tags=("button", b.id))
        
        # Calculate available dimensions for text (with 4px padding on sides)
        max_text_w = max(10, (x1 - x0) - 8)
        font_style = ("Segoe UI", max(8, int(b.font_size_pt * 0.9)), "bold" if is_selected else "normal")

        b.text_item = c.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=b.text,
                                     font=font_style,
                                     fill=text_color, width=max_text_w, justify="center",
                                     tags=("button", b.id))

        # Perform fast single-pass vertical clip check only if necessary
        max_text_h = (y1 - y0) - 4
        bbox = c.bbox(b.text_item)
        if bbox and (bbox[3] - bbox[1] > max_text_h) and len(b.text) > 3:
            truncated = b.text[:max(1, int(len(b.text) * (max_text_h / (bbox[3] - bbox[1]))))] + "…"
            c.itemconfig(b.text_item, text=truncated)

        if is_selected:
            for cx, cy, corner in [(x0, y0, "tl"), (x1, y0, "tr"), (x1, y1, "br"), (x0, y1, "bl")]:
                h = c.create_rectangle(
                    cx - HANDLE_SIZE_PX / 2, cy - HANDLE_SIZE_PX / 2,
                    cx + HANDLE_SIZE_PX / 2, cy + HANDLE_SIZE_PX / 2,
                    fill="#4F46E5", outline="#FFFFFF", width=1, tags=("handle", b.id, corner)
                )
                b.handle_items.append(h)



    def _redraw_all_buttons(self):
        for b in self.buttons:
            self._redraw_button(b)

    # -----------------------------------------------------------
    # UNDO / REDO STATE MANAGEMENT
    # -----------------------------------------------------------
    def save_state(self):
        if self._is_undo_redo_action:
            return
        snapshot = [b.to_dict() for b in self.buttons]
        if not self.undo_stack or self.undo_stack[-1] != snapshot:
            self.undo_stack.append(snapshot)
            self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) <= 1:
            self._set_status("Nothing to undo.")
            return
        self._is_undo_redo_action = True
        self.redo_stack.append(self.undo_stack.pop())
        state = self.undo_stack[-1]
        self._restore_state(state)
        self._is_undo_redo_action = False
        self._set_status("Undid last action.")

    def redo(self):
        if not self.redo_stack:
            self._set_status("Nothing to redo.")
            return
        self._is_undo_redo_action = True
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        self._restore_state(state)
        self._is_undo_redo_action = False
        self._set_status("Redid action.")

    def _restore_state(self, state):
        for b in self.buttons:
            self.canvas.delete(b.rect_item)
            self.canvas.delete(b.text_item)
            for h in b.handle_items:
                self.canvas.delete(h)
        self.buttons = [Button.from_dict(d) for d in state]
        self.selected = None
        self._clear_property_fields()
        self._redraw_all_buttons()

    # -----------------------------------------------------------
    # BUTTON MANAGEMENT
    # -----------------------------------------------------------
    def add_button(self):
        w, h = BUTTON_MIN_WIDTH_MM * 1.5, BUTTON_MIN_HEIGHT_MM * 1.5
        x, y = self._find_free_spot(w, h)
        if x is None:
            self._set_status("No free space left for a new button.")
            return
        b = Button(x, y, w, h)
        self.buttons.append(b)
        self.select_button(b)
        self._redraw_button(b)
        self.save_state()
        self._set_status(f"Added {b.id}.")

    def duplicate_selected(self):
        if self.selected is None:
            self._set_status("No button selected to duplicate.")
            return
        orig = self.selected
        w, h = orig.width_mm, orig.height_mm
        offset = self.grid_size_mm if (self.snap_to_grid.get() and self.grid_size_mm > 0) else 5.0
        
        # Try nearby spots offset right/down, or fall back to _find_free_spot
        candidate_positions = [
            (self.snap_value(orig.x_mm + offset), self.snap_value(orig.y_mm + offset)),
            (self.snap_value(orig.x_mm + offset), orig.y_mm),
            (orig.x_mm, self.snap_value(orig.y_mm + offset)),
        ]
        
        x, y = None, None
        for cx, cy in candidate_positions:
            rect = (cx, cy, w, h)
            if rect_within_interior(rect) and not any(rects_overlap(rect, ob.rect_mm(), gap_mm=BUTTON_MIN_GAP_MM) for ob in self.buttons):
                x, y = cx, cy
                break

        if x is None:
            x, y = self._find_free_spot(w, h)

        if x is None:
            self._set_status("No space left to duplicate button.")
            return

        dup = Button(x, y, w, h, text=f"{orig.text}_copy", font_size_pt=orig.font_size_pt)
        self.buttons.append(dup)
        self.select_button(dup)
        self._redraw_button(dup)
        self.save_state()
        self._set_status(f"Duplicated button as {dup.id}.")

    def _find_free_spot(self, w, h):
        step = self.grid_size_mm if (self.snap_to_grid.get() and self.grid_size_mm > 0) else 5.0
        y = self.snap_value(INTERIOR_Y_MIN)
        if y < INTERIOR_Y_MIN:
            y += step
        while y + h <= INTERIOR_Y_MAX:
            x = self.snap_value(INTERIOR_X_MIN)
            if x < INTERIOR_X_MIN:
                x += step
            while x + w <= INTERIOR_X_MAX:
                candidate = (x, y, w, h)
                if not any(rects_overlap(candidate, ob.rect_mm(), gap_mm=BUTTON_MIN_GAP_MM) for ob in self.buttons):
                    return x, y
                x += step
            y += step
        return None, None

    def delete_selected(self):
        if self.selected is None:
            return
        b = self.selected
        self.canvas.delete(b.rect_item)
        self.canvas.delete(b.text_item)
        for h in b.handle_items:
            self.canvas.delete(h)
        self.buttons.remove(b)
        self.selected = None
        self._clear_property_fields()
        self.save_state()
        self._set_status(f"Deleted {b.id}.")

    def select_button(self, b):
        prev = self.selected
        self.selected = b
        if prev is not None and prev is not b:
            self._redraw_button(prev)
        if b is not None:
            self._redraw_button(b)
            self.text_var.set(b.text)
            self.font_var.set(str(b.font_size_pt))
            self.width_var.set(f"{b.width_mm:.1f}")
            self.height_var.set(f"{b.height_mm:.1f}")
        else:
            self._clear_property_fields()

    def _clear_property_fields(self):
        self.text_var.set("")
        self.font_var.set("")
        self.width_var.set("")
        self.height_var.set("")

    def apply_property_changes(self):
        b = self.selected
        if b is None:
            return
        try:
            new_font = int(self.font_var.get())
            new_w = float(self.width_var.get())
            new_h = float(self.height_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Font size, width, and height must be numbers.")
            return

        new_w = max(new_w, BUTTON_MIN_WIDTH_MM)
        new_h = max(new_h, BUTTON_MIN_HEIGHT_MM)
        candidate = (b.x_mm, b.y_mm, new_w, new_h)

        if not rect_within_interior(candidate):
            self._set_status("Resize rejected: would leave the interior area.")
            return
        if any(rects_overlap(candidate, ob.rect_mm(), gap_mm=BUTTON_MIN_GAP_MM) for ob in self.buttons if ob is not b):
            self._set_status("Resize rejected: would overlap another button.")
            return

        b.text = self.text_var.get()
        b.font_size_pt = new_font
        b.width_mm = new_w
        b.height_mm = new_h
        self._redraw_button(b)
        self.save_state()
        self._set_status(f"Updated {b.id}.")


    # -----------------------------------------------------------
    # CANVAS MOUSE INTERACTION
    # -----------------------------------------------------------
    def on_canvas_press(self, event):
        # Check for a resize handle hit first (only relevant if something is selected)
        if self.selected is not None:
            item = self.canvas.find_withtag("current")
            tags = self.canvas.gettags(item) if item else ()
            if "handle" in tags:
                corner = tags[2]
                self.drag_mode = "resize"
                self.resize_corner = corner
                self.drag_start = (px_to_mm(event.x), px_to_mm(event.y))
                self.drag_orig_rect = self.selected.rect_mm()
                return

        # Otherwise, check if we clicked on a button body
        clicked_button = self._button_at_pixel(event.x, event.y)
        if clicked_button is not None:
            self.select_button(clicked_button)
            self.drag_mode = "move"
            self.drag_start = (px_to_mm(event.x), px_to_mm(event.y))
            self.drag_orig_rect = clicked_button.rect_mm()
        else:
            self.select_button(None)
            self.drag_mode = None

    def on_canvas_double_click(self, event):
        clicked_button = self._button_at_pixel(event.x, event.y)
        if clicked_button is not None:
            dialog = ctk.CTkInputDialog(text="Enter new button text:", title="Rename Button")
            dialog.bind("<Escape>", lambda e: dialog._cancel_event())
            
            # Pre-populate entry with current button text and focus entry
            def _setup_entry():
                if hasattr(dialog, "_entry") and dialog._entry:
                    dialog._entry.insert(0, clicked_button.text)
                    dialog._entry.focus()
                    try:
                        dialog._entry.select_range(0, tk.END)
                    except Exception:
                        pass
            dialog.after(160, _setup_entry)

            new_text = dialog.get_input()
            if new_text is not None and new_text.strip() != "":
                clicked_button.text = new_text.strip()
                self.select_button(clicked_button)
                self._redraw_button(clicked_button)
                self.save_state()
                self._set_status(f"Renamed {clicked_button.id} to '{clicked_button.text}'.")


    def _button_at_pixel(self, px, py):
        x_mm, y_mm = px_to_mm(px), px_to_mm(py)
        for b in reversed(self.buttons):  # topmost first
            if b.x_mm <= x_mm <= b.x_mm + b.width_mm and b.y_mm <= y_mm <= b.y_mm + b.height_mm:
                return b
        return None

    def on_canvas_drag(self, event):
        if self.selected is None or self.drag_mode is None:
            return

        cur_x_mm, cur_y_mm = px_to_mm(event.x), px_to_mm(event.y)
        dx = cur_x_mm - self.drag_start[0]
        dy = cur_y_mm - self.drag_start[1]
        ox, oy, ow, oh = self.drag_orig_rect

        if self.drag_mode == "move":
            target_x = self.snap_value(ox + dx)
            target_y = self.snap_value(oy + dy)
            candidate = (target_x, target_y, ow, oh)
        else:  # resize
            candidate = self._resize_rect(ox, oy, ow, oh, dx, dy, self.resize_corner)
            if candidate is None:
                return

        if not rect_within_interior(candidate):
            return  # simply ignore invalid moves — button stays at last valid spot
        if any(rects_overlap(candidate, ob.rect_mm(), gap_mm=BUTTON_MIN_GAP_MM)
               for ob in self.buttons if ob is not self.selected):
            return

        self.selected.x_mm, self.selected.y_mm, self.selected.width_mm, self.selected.height_mm = candidate
        self._redraw_button(self.selected)
        if self.selected:
            self.width_var.set(f"{self.selected.width_mm:.1f}")
            self.height_var.set(f"{self.selected.height_mm:.1f}")

    def _resize_rect(self, x, y, w, h, dx, dy, corner):
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        if corner == "tl":
            x1, y1 = self.snap_value(x + dx), self.snap_value(y + dy)
        elif corner == "tr":
            x2, y1 = self.snap_value(x + w + dx), self.snap_value(y + dy)
        elif corner == "br":
            x2, y2 = self.snap_value(x + w + dx), self.snap_value(y + h + dy)
        elif corner == "bl":
            x1, y2 = self.snap_value(x + dx), self.snap_value(y + h + dy)

        new_w = x2 - x1
        new_h = y2 - y1
        if new_w < BUTTON_MIN_WIDTH_MM or new_h < BUTTON_MIN_HEIGHT_MM:
            return None
        return (x1, y1, new_w, new_h)

    def on_canvas_release(self, event):
        if self.drag_mode is not None:
            self.save_state()
        self.drag_mode = None
        self.resize_corner = None
        self.drag_start = None
        self.drag_orig_rect = None

    # -----------------------------------------------------------
    # PROJECT / EXPORT ACTIONS
    # -----------------------------------------------------------
    def on_save_project(self):
        path = get_native_file_dialog(mode="save", title="Save Project", default_ext=".json",
                                      filetypes=[("JSON project (*.json)", "*.json")])
        if not path:
            return
        save_project(path, [b.to_dict() for b in self.buttons])
        self._set_status(f"Saved project to {path}")

    def on_load_project(self):
        path = get_native_file_dialog(mode="open", title="Load Project", default_ext=".json",
                                      filetypes=[("JSON project (*.json)", "*.json")])
        if not path:
            return
        data = load_project(path)
        for b in self.buttons:
            self.canvas.delete(b.rect_item)
            self.canvas.delete(b.text_item)
            for h in b.handle_items:
                self.canvas.delete(h)
        self.buttons = [Button.from_dict(d) for d in data]
        self.selected = None
        self._clear_property_fields()
        self._redraw_all_buttons()
        self._set_status(f"Loaded project from {path}")

    def on_export_xml(self):
        path = get_native_file_dialog(mode="save", title="Export XML Layout", default_ext=".xml",
                                      filetypes=[("XML layout (*.xml)", "*.xml")])
        if not path:
            return
        export_xml(path, [b.to_dict() for b in self.buttons])
        self._set_status(f"Exported XML to {path}")

    def on_export_pdf(self):
        path = get_native_file_dialog(mode="save", title="Export Printable PDF", default_ext=".pdf",
                                      filetypes=[("PDF document (*.pdf)", "*.pdf")])
        if not path:
            return
        export_pdf(path, [b.to_dict() for b in self.buttons])
        self._set_status(f"Exported PDF to {path}")

    def on_show_preview(self):
        buttons_dict = [b.to_dict() for b in self.buttons]
        pil_img = generate_preview_image(buttons_dict, scale=3.0)

        # Create TopLevel Modal Preview Window
        top = ctk.CTkToplevel(self)
        top.title("Print Outcome Preview (A4 Landscape)")
        top.grab_set()
        top.focus_set()
        top.bind("<Escape>", lambda e: top.destroy())

        # Render image inside a scrollable canvas / frame
        img_tk = ImageTk.PhotoImage(pil_img)

        label = ctk.CTkLabel(top, text="This preview shows how your AprilTag markers and button layout will appear on the final printed paper.",
                             font=ctk.CTkFont(size=13), text_color="gray80")
        label.pack(pady=(12, 6), padx=16)

        img_label = tk.Label(top, image=img_tk, bg="#FFFFFF", bd=2, relief="solid")
        img_label.image = img_tk  # keep reference
        img_label.pack(padx=16, pady=12)

        close_btn = ctk.CTkButton(top, text="Close Preview (Esc)", command=top.destroy)
        close_btn.pack(pady=(0, 12))
        self._set_status("Displayed print outcome preview.")


if __name__ == "__main__":
    app = DesignerApp()
    app.mainloop()
