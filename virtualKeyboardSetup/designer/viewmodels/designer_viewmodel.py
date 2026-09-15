"""
Designer View Model (MVVM Architecture).
Mediates layout state, user actions (button/marker CRUD), selection, validation,
XML repository file persistence, SQLite database sync, and PDF printing.
"""

from pathlib import Path
from PySide6.QtCore import QObject, Signal

from config.app_config import AppConfig
from core.models.button_model import ButtonModel
from core.models.marker_model import MarkerModel
from core.models.paper_layout import PaperLayoutModel
from core.geometry.layout_geometry import validate_layout_geometry, rects_overlap
from services.xml_repository import XmlRepository
from services.db_repository import DbRepository
from services.pdf_exporter import PdfExporter
from services.preview_service import PreviewService


class DesignerViewModel(QObject):
    # Signals for View Layer Binding
    layout_changed = Signal(object)              # PaperLayoutModel
    button_added = Signal(object)                # ButtonModel
    marker_added = Signal(object)                # MarkerModel
    button_removed = Signal(str)                 # button id
    marker_removed = Signal(int)                 # marker id
    button_updated = Signal(object)              # ButtonModel
    marker_updated = Signal(object)              # MarkerModel
    selection_changed = Signal(object)           # ButtonModel or None
    marker_selection_changed = Signal(object)    # MarkerModel or None
    stats_changed = Signal(int, int, bool)       # key_count, marker_count, is_valid
    status_message = Signal(str, str)            # title, message
    error_message = Signal(str, str)             # title, message
    dirty_changed = Signal(bool)                 # is_dirty

    def __init__(self, config: AppConfig, parent: QObject | None = None):
        super().__init__(parent)
        self.config = config
        self._layout = PaperLayoutModel(
            paper_width_mm=config.paper_width_mm,
            paper_height_mm=config.paper_height_mm,
            paper_margin_mm=config.paper_margin_mm,
            marker_size_mm=config.marker_size_mm,
            marker_spacing_mm=config.marker_spacing_mm,
        )
        self.selected_button: ButtonModel | None = None
        self.selected_marker: MarkerModel | None = None

        self._button_counter = 1
        self._marker_counter = 0

        self._is_dirty = False
        self._current_filepath: str | None = None

        # Repositories & Exporters
        self.xml_repo = XmlRepository(self.config)
        self.db_repo = DbRepository()
        self.pdf_exporter = PdfExporter()
        self.preview_service = PreviewService()

    @property
    def layout(self) -> PaperLayoutModel:
        return self._layout

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    @property
    def current_filepath(self) -> str | None:
        return self._current_filepath

    def mark_dirty(self) -> None:
        """Mark layout state as unsaved/modified."""
        if not self._is_dirty:
            self._is_dirty = True
            self.dirty_changed.emit(True)

    def mark_clean(self) -> None:
        """Mark layout state as clean/saved."""
        if self._is_dirty:
            self._is_dirty = False
            self.dirty_changed.emit(False)


    def _find_next_available_position(self, width_mm: float, height_mm: float, is_marker: bool = False) -> tuple[float, float]:
        """Calculate a non-overlapping position within the active surface region."""
        margin_x = self.config.interior_x_min + 5.0
        margin_y = self.config.interior_y_min + 5.0
        max_x = self.config.interior_x_max - width_mm - 2.0
        max_y = self.config.interior_y_max - height_mm - 2.0

        required_gap = max(5.0, self.config.button_min_gap_mm)
        gap_mm = required_gap + 2.0
        step_x = width_mm + gap_mm
        step_y = height_mm + gap_mm

        curr_y = margin_y
        while curr_y <= max_y:
            curr_x = margin_x
            while curr_x <= max_x:
                cand_rect = (curr_x, curr_y, width_mm, height_mm)

                has_collision = False
                for btn in self._layout.buttons:
                    if rects_overlap(cand_rect, btn.rect_tuple, gap_mm=required_gap):
                        has_collision = True
                        break

                if not has_collision and self._layout.use_custom_markers:
                    for m in self._layout.custom_markers:
                        if rects_overlap(cand_rect, m.rect_tuple, gap_mm=required_gap):
                            has_collision = True
                            break

                if not has_collision:
                    return (round(curr_x, 1), round(curr_y, 1))

                curr_x += step_x
            curr_y += step_y

        return (margin_x, margin_y)

    def update_project_name(self, name: str) -> None:
        """Update layout project name and notify view canvas and window components."""
        clean_name = name.strip() if name and name.strip() else "My Paper Keyboard"
        if self._layout.project_name != clean_name:
            self._layout.project_name = clean_name
            self.mark_dirty()
        self.layout_changed.emit(self._layout)
        self._validate_and_notify_stats()

    def update_paper_dimensions(self, width_mm: float, height_mm: float) -> None:
        """Update layout paper dimensions and notify subscribers."""
        if self._layout.paper_width_mm != width_mm or self._layout.paper_height_mm != height_mm:
            self._layout.paper_width_mm = width_mm
            self._layout.paper_height_mm = height_mm
            self.mark_dirty()
        self._layout.marker_size_mm = self.config.marker_size_mm
        self._validate_and_notify_stats()

    def create_new_layout(self, project_name: str = "My Paper Keyboard") -> None:
        """Clear active design layout for a new design."""
        self._layout.project_name = project_name
        self._layout.buttons.clear()
        self._layout.custom_markers.clear()
        self.selected_button = None
        self.selected_marker = None
        self._button_counter = 1
        self._marker_counter = 0
        self._current_filepath = None
        self.selection_changed.emit(None)
        self.marker_selection_changed.emit(None)
        self.layout_changed.emit(self._layout)
        self._validate_and_notify_stats()
        self.mark_clean()
        self.status_message.emit("New Layout", f"Canvas initialized for '{project_name}'.")

    def add_button(self) -> None:
        """Add a new key button to the design canvas at a non-overlapping position."""
        b_id = f"btn_{self._button_counter}"
        self._button_counter += 1

        w_mm = max(25.0, self.config.button_min_width_mm)
        h_mm = max(18.0, self.config.button_min_height_mm)
        pos_x, pos_y = self._find_next_available_position(w_mm, h_mm, is_marker=False)

        button = ButtonModel(
            id=b_id,
            x_mm=pos_x,
            y_mm=pos_y,
            width_mm=w_mm,
            height_mm=h_mm,
            text=f"Key {self._button_counter - 1}",
            font_size_pt=self.config.default_font_size_pt,
        )
        self._layout.buttons.append(button)
        self.select_button(button)
        self.button_added.emit(button)
        self.mark_dirty()
        self._validate_and_notify_stats()

    def get_used_marker_ids(self) -> set[int]:
        """Collect all assigned AprilTag IDs across custom markers and outer perimeter anchors."""
        used = {m.id for m in self._layout.custom_markers}
        if self.config.show_outer_markers:
            from core.geometry.marker_generator import generate_marker_layout
            outer_markers = generate_marker_layout(self.config)
            used.update(m.id for m in outer_markers)
        return used

    def get_next_available_marker_id(self) -> int:
        """Find the lowest available non-conflicting AprilTag ID."""
        used_ids = self.get_used_marker_ids()
        next_id = 0
        while next_id in used_ids:
            next_id += 1
        return next_id

    def add_custom_marker(self) -> None:
        """Add a new custom interior AprilTag fiducial marker with a guaranteed unique Tag ID."""
        tag_id = self.get_next_available_marker_id()

        m_size = self.config.marker_size_mm
        half = m_size / 2.0
        pos_x, pos_y = self._find_next_available_position(m_size, m_size, is_marker=True)
        center_x = pos_x + half
        center_y = pos_y + half

        min_x = self.config.interior_x_min + half
        max_x = self.config.interior_x_max - half
        min_y = self.config.interior_y_min + half
        max_y = self.config.interior_y_max - half
        center_x = max(min_x, min(center_x, max_x))
        center_y = max(min_y, min(center_y, max_y))

        marker = MarkerModel(
            id=tag_id,
            x_mm=center_x,
            y_mm=center_y,
            size_mm=m_size,
        )
        self._layout.custom_markers.append(marker)
        self._layout.use_custom_markers = True
        self.select_marker(marker)
        self.marker_added.emit(marker)
        self.mark_dirty()
        self._validate_and_notify_stats()



    def duplicate_selected_button(self) -> None:
        """Duplicate currently selected key button."""
        if not self.selected_button:
            return

        orig = self.selected_button
        b_id = f"btn_{self._button_counter}"
        self._button_counter += 1

        new_button = ButtonModel(
            id=b_id,
            x_mm=orig.x_mm + 5.0,
            y_mm=orig.y_mm + 5.0,
            width_mm=orig.width_mm,
            height_mm=orig.height_mm,
            text=f"{orig.text}_copy",
            font_size_pt=orig.font_size_pt,
        )
        self._layout.buttons.append(new_button)
        self.button_added.emit(new_button)
        self.mark_dirty()
        self._validate_and_notify_stats()

    def remove_selected_button(self) -> None:
        """Remove currently selected key button."""
        if not self.selected_button:
            return

        btn_id = self.selected_button.id
        self._layout.buttons = [b for b in self._layout.buttons if b.id != btn_id]
        self.selected_button = None
        self.selection_changed.emit(None)
        self.button_removed.emit(btn_id)
        self.mark_dirty()
        self._validate_and_notify_stats()

    def remove_selected_marker(self) -> None:
        """Remove currently selected interior AprilTag marker."""
        if not self.selected_marker:
            return

        tag_id = self.selected_marker.id
        self._layout.custom_markers = [m for m in self._layout.custom_markers if m.id != tag_id]
        if not self._layout.custom_markers:
            self._layout.use_custom_markers = False
        self.selected_marker = None
        self.marker_selection_changed.emit(None)
        self.marker_removed.emit(tag_id)
        self.mark_dirty()
        self._validate_and_notify_stats()

    def remove_selected_element(self) -> None:
        """Remove whichever element is currently selected (key button or AprilTag marker)."""
        if self.selected_button:
            self.remove_selected_button()
        elif self.selected_marker:
            self.remove_selected_marker()


    def select_button(self, button: ButtonModel | None) -> None:
        self.selected_button = button
        if button is not None:
            self.selected_marker = None
            self.marker_selection_changed.emit(None)
        self.selection_changed.emit(button)

    def select_marker(self, marker: MarkerModel | None) -> None:
        self.selected_marker = marker
        if marker is not None:
            self.selected_button = None
            self.selection_changed.emit(None)
        self.marker_selection_changed.emit(marker)

    def update_button_properties(self, updated_button: ButtonModel) -> None:
        """Update property values for a button in the layout."""
        for idx, b in enumerate(self._layout.buttons):
            if b.id == updated_button.id:
                self._layout.buttons[idx] = updated_button
                break
        self.button_updated.emit(updated_button)
        self.mark_dirty()
        self._validate_and_notify_stats()

    def update_marker_properties(self, updated_marker: MarkerModel) -> None:
        """Update property values for a marker in the layout."""
        for idx, m in enumerate(self._layout.custom_markers):
            if m.id == updated_marker.id:
                self._layout.custom_markers[idx] = updated_marker
                break
        self.marker_updated.emit(updated_marker)
        self.mark_dirty()
        self._validate_and_notify_stats()

    def sync_layout_config(self) -> None:
        """Sync current AppConfig values into layout model for saving."""
        self._layout.paper_width_mm = self.config.paper_width_mm
        self._layout.paper_height_mm = self.config.paper_height_mm
        self._layout.paper_margin_mm = self.config.paper_margin_mm
        self._layout.marker_size_mm = self.config.marker_size_mm
        self._layout.marker_spacing_mm = self.config.marker_spacing_mm
        self._layout.marker_family = self.config.marker_family
        self._layout.marker_min_gap_ratio = self.config.marker_min_gap_ratio
        self._layout.show_outer_markers = self.config.show_outer_markers
        self._layout.button_stroke_width_mm = self.config.button_stroke_width_mm
        self._layout.button_corner_radius_mm = self.config.button_corner_radius_mm
        self._layout.button_min_width_mm = self.config.button_min_width_mm
        self._layout.button_min_height_mm = self.config.button_min_height_mm
        self._layout.button_min_gap_mm = self.config.button_min_gap_mm
        self._layout.default_font_size_pt = self.config.default_font_size_pt
        self._layout.grid_snap_enabled = self.config.grid_snap_enabled
        self._layout.grid_size_mm = self.config.grid_size_mm

    def load_project_xml(self, filepath: str) -> None:
        """Load layout design from target XML project file."""
        try:
            self._layout = self.xml_repo.load(filepath)
            self._current_filepath = str(Path(filepath).resolve())
            self.config.paper_width_mm = self._layout.paper_width_mm
            self.config.paper_height_mm = self._layout.paper_height_mm
            self.config.paper_margin_mm = self._layout.paper_margin_mm
            self.config.marker_size_mm = self._layout.marker_size_mm
            self.config.button_stroke_width_mm = self._layout.button_stroke_width_mm
            self.config.button_corner_radius_mm = self._layout.button_corner_radius_mm
            self.config.button_min_gap_mm = self._layout.button_min_gap_mm
            self.config.grid_snap_enabled = self._layout.grid_snap_enabled
            self.config.grid_size_mm = self._layout.grid_size_mm
            self.config.show_outer_markers = self._layout.show_outer_markers
            self.selected_button = None
            self.selected_marker = None
            self.selection_changed.emit(None)
            self.marker_selection_changed.emit(None)
            self.layout_changed.emit(self._layout)
            self._validate_and_notify_stats()
            self.mark_clean()
            self.status_message.emit("Project Loaded", f"Loaded layout from {Path(filepath).name}")
        except Exception as e:
            self.error_message.emit("Error Loading Project", str(e))

    def save_project_xml(self, filepath: str) -> None:
        """Save layout design directly into target XML project file."""
        self.sync_layout_config()
        if not validate_layout_geometry(self._layout, self.config):
            self.error_message.emit(
                "Cannot Save Layout",
                "One or more key buttons or custom markers are placed outside the active surface zone or overlapping. Please move red-highlighted elements into valid positions.",
            )
            return
        try:
            self.xml_repo.save(filepath, self._layout)
            self._current_filepath = str(Path(filepath).resolve())
            self.mark_clean()
            self.status_message.emit("Project Saved", f"Saved layout to {Path(filepath).name}")
        except Exception as e:
            self.error_message.emit("Error Saving Project", str(e))

        except Exception as e:
            self.error_message.emit("Error Saving Project", str(e))

    def sync_to_database(self) -> None:
        """Synchronize current layout model with SQLite database."""
        self.sync_layout_config()
        if not validate_layout_geometry(self._layout, self.config):
            self.error_message.emit(
                "Cannot Sync Database",
                "One or more key buttons or custom markers are placed outside the active surface zone or overlapping. Please move red-highlighted elements into valid positions.",
            )
            return
        try:
            self.db_repo.save("default_layout", self._layout)
            self.status_message.emit("Database Sync", "Saved current layout into SQLite database.")
        except Exception as e:
            self.error_message.emit("Database Error", str(e))

    def export_pdf(self, filepath: str) -> None:
        """Export printable PDF layout sheet."""
        if not validate_layout_geometry(self._layout, self.config):
            self.error_message.emit(
                "Cannot Export PDF",
                "One or more key buttons or custom markers are placed outside the active surface zone or overlapping. Please move red-highlighted elements into valid positions.",
            )
            return
        try:
            self.pdf_exporter.export(filepath, self._layout, self.config)
            self.status_message.emit("PDF Exported", f"Exported printable PDF to {Path(filepath).name}")
        except Exception as e:
            self.error_message.emit("Export Error", str(e))

    def export_png(self, filepath: str) -> None:
        """Export high-resolution PNG image layout sheet."""
        if not validate_layout_geometry(self._layout, self.config):
            self.error_message.emit(
                "Cannot Export PNG",
                "One or more key buttons or custom markers are placed outside the active surface zone or overlapping. Please move red-highlighted elements into valid positions.",
            )
            return
        try:
            pil_img = self.preview_service.render_preview(self._layout, self.config, scale=4.0)
            pil_img.save(filepath, "PNG")
            self.status_message.emit("PNG Exported", f"Exported high-resolution PNG image to {Path(filepath).name}")
        except Exception as e:
            self.error_message.emit("Export Error", str(e))


    def _validate_and_notify_stats(self) -> None:
        """Recalculate layout geometry validity and emit stats_changed signal."""
        valid = validate_layout_geometry(self._layout, self.config)
        num_markers = len(self._layout.custom_markers) if self._layout.use_custom_markers else 0
        self.stats_changed.emit(len(self._layout.buttons), num_markers, valid)
