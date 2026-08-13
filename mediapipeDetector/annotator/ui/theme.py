"""
annotator/ui/theme.py

Shared VS Code Dark+ colour tokens, font family, and scroll-routing utility
for all annotator UI screens.

Scroll routing is set up ONCE on the root CTk window via setup_scroll_routing().
It installs a bind_all handler that finds the CTkScrollableFrame canvas directly
under the cursor (via winfo_containing + tree walk) and scrolls it — no lag,
no enter/leave event races, works even when the cursor is over child widgets.
"""
import customtkinter as ctk

FF = "Helvetica"          # smooth anti-aliased vector font on Linux X11

# ── Colour palette (VS Code Dark+) ───────────────────────────────────────────
BG        = "#1e1e1e"     # main window / screen background
PANEL     = "#252526"     # card / side-panel background
BORDER    = "#3c3c3c"     # 1-px border colour
HDR_BG    = "#2d2d2d"     # top-bar / header strip
TXT_PRI   = "#ffffff"     # primary text
TXT_SEC   = "#cccccc"     # secondary / body text
TXT_MUT   = "#858585"     # muted / hints
ROW_BG    = "#2a2a2a"     # list row background
ROW_REC   = "#252526"     # saved/recorded row
ROW_CUR   = "#0d3a5c"     # current-active row (blue tint)
ROW_CUR_T = "#4ec9f0"     # current-active row text

BTN_PRI   = "#007acc"     # VS Code blue – primary actions
BTN_HVP   = "#0062a3"     # primary hover
BTN_SEC   = "#3c3c3c"     # secondary / neutral actions
BTN_HVS   = "#4c4c4c"     # secondary hover
BTN_GHO   = "transparent" # ghost button background
BTN_GHH   = "#37373d"     # ghost button hover
BTN_PUR   = "#6f42c1"     # purple – Analyze Window
BTN_PUR_H = "#5a339e"
SEL_BG    = "#007acc"     # selected-row highlight

AMBER     = "#d4a017"     # auto-save toggle / recorded label
SW_TRACK  = "#555555"     # switch off-state track colour

SECTION_C = "#9cdcfe"     # section header (VS Code variable-name blue)

# ── Matplotlib tokens (kept here for reference in window_analyzer) ────────────
MPL_BG    = "#1e1e1e"
MPL_PLOT  = "#252526"
MPL_GRID  = "#3c3c3c"
MPL_SPINE = "#555555"
MPL_TICK  = "#858585"


# ── Global scroll routing ─────────────────────────────────────────────────────

def setup_scroll_routing(root: ctk.CTk) -> None:
    """
    Install a single bind_all scroll handler on the root CTk window.

    When the user scrolls anywhere, winfo_containing() finds the widget
    directly under the cursor; we then walk up the widget tree to locate
    the nearest CTkScrollableFrame and scroll its internal canvas.

    This approach:
    - Works even when the cursor is over a child widget (button, label …)
      inside a scrollable frame — no event swallowing.
    - Has zero lag (direct yview_scroll call, no polling).
    - Requires no per-frame enter/leave bookkeeping.
    """
    def _find_scrollable_canvas(widget):
        """Walk the master-chain to find the closest CTkScrollableFrame canvas."""
        try:
            w = widget
            for _ in range(30):          # limit depth to avoid infinite loop
                if w is None:
                    break
                parent = getattr(w, "master", None)
                if parent is None:
                    break
                # CTkScrollableFrame exposes its internal canvas as _parent_canvas
                if hasattr(parent, "_parent_canvas") and hasattr(parent, "_scrollbar"):
                    return parent._parent_canvas
                w = parent
        except Exception:
            pass
        return None

    def _scroll(e, delta: int | None = None):
        try:
            x = root.winfo_pointerx()
            y = root.winfo_pointery()
            widget = root.winfo_containing(x, y)
            if widget is None:
                return
            canvas = _find_scrollable_canvas(widget)
            if canvas is None:
                return
            d = delta if delta is not None else int(-1 * (e.delta / 120))
            canvas.yview_scroll(d, "units")
        except Exception:
            pass

    root.bind_all("<Button-4>",   lambda e: _scroll(e,  -3))   # Linux scroll up
    root.bind_all("<Button-5>",   lambda e: _scroll(e,   3))   # Linux scroll down
    root.bind_all("<MouseWheel>", _scroll)                      # Windows / macOS


# ── Convenience widget factories ──────────────────────────────────────────────

def make_switch(parent, text: str, variable, text_color=TXT_SEC,
                on_color=BTN_PRI, bg=PANEL) -> ctk.CTkSwitch:
    """Return a VS Code-styled CTkSwitch that blends into the panel background."""
    return ctk.CTkSwitch(
        parent,
        text=text,
        variable=variable,
        onvalue=True,
        offvalue=False,
        font=ctk.CTkFont(family=FF, size=13),
        text_color=text_color,
        # Off-state track colour
        fg_color=SW_TRACK,
        # On-state track colour (progress fill)
        progress_color=on_color,
        # Toggle-circle colours
        button_color="#e0e0e0",
        button_hover_color="#ffffff",
        # Match background so the track doesn't float on a wrong bg
        bg_color=bg,
    )


def section_label(parent, text: str) -> ctk.CTkLabel:
    """Return a styled section header label."""
    return ctk.CTkLabel(
        parent,
        text=text,
        font=ctk.CTkFont(family=FF, size=13, weight="bold"),
        text_color=SECTION_C,
    )
