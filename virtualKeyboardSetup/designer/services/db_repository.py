"""
SQLite Database Repository Service.
Implements IProjectRepository using DatabaseManager connection class.
"""

from pathlib import Path
from config.constants import (
    PAPER_MARGIN_MM,
    MARKER_SPACING_MM,
)
from core.interfaces.repository_interface import IProjectRepository
from core.models.paper_layout import PaperLayoutModel
from core.models.button_model import ButtonModel
from core.models.marker_model import MarkerModel
from db.connection import DatabaseManager


class DbRepository(IProjectRepository):
    def __init__(self, db_manager: DatabaseManager | None = None):
        self.db_manager = db_manager or DatabaseManager()

    def save(self, project_id: str | Path, layout: PaperLayoutModel) -> None:
        p_id = str(project_id)
        with self.db_manager.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO projects (id, name, paper_width_mm, paper_height_mm, paper_margin_mm, marker_size_mm, marker_spacing_mm, use_custom_markers, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    paper_width_mm=excluded.paper_width_mm,
                    paper_height_mm=excluded.paper_height_mm,
                    paper_margin_mm=excluded.paper_margin_mm,
                    marker_size_mm=excluded.marker_size_mm,
                    marker_spacing_mm=excluded.marker_spacing_mm,
                    use_custom_markers=excluded.use_custom_markers,
                    updated_at=CURRENT_TIMESTAMP;
                """,
                (
                    p_id,
                    p_id,
                    layout.paper_width_mm,
                    layout.paper_height_mm,
                    layout.paper_margin_mm,
                    layout.marker_size_mm,
                    layout.marker_spacing_mm,
                    1 if layout.use_custom_markers else 0,
                ),
            )

            # Save Buttons
            cursor.execute("DELETE FROM buttons WHERE project_id = ?;", (p_id,))
            for btn in layout.buttons:
                cursor.execute(
                    """
                    INSERT INTO buttons (id, project_id, x_mm, y_mm, width_mm, height_mm, text, font_size_pt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (btn.id, p_id, btn.x_mm, btn.y_mm, btn.width_mm, btn.height_mm, btn.text, btn.font_size_pt),
                )

            # Save Custom Markers
            cursor.execute("DELETE FROM markers WHERE project_id = ?;", (p_id,))
            for m in layout.custom_markers:
                cursor.execute(
                    """
                    INSERT INTO markers (id, project_id, x_mm, y_mm, size_mm)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (m.id, p_id, m.x_mm, m.y_mm, m.size_mm),
                )

    def load(self, project_id: str | Path) -> PaperLayoutModel:
        p_id = str(project_id)
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM projects WHERE id = ?;", (p_id,))
            project_row = cursor.fetchone()
            if not project_row:
                return PaperLayoutModel()

            cursor.execute("SELECT * FROM buttons WHERE project_id = ?;", (p_id,))
            button_rows = cursor.fetchall()

            buttons = [
                ButtonModel(
                    id=row["id"],
                    x_mm=row["x_mm"],
                    y_mm=row["y_mm"],
                    width_mm=row["width_mm"],
                    height_mm=row["height_mm"],
                    text=row["text"],
                    font_size_pt=row["font_size_pt"],
                )
                for row in button_rows
            ]

            cursor.execute("SELECT * FROM markers WHERE project_id = ?;", (p_id,))
            marker_rows = cursor.fetchall()
            custom_markers = [
                MarkerModel(
                    id=row["id"],
                    x_mm=row["x_mm"],
                    y_mm=row["y_mm"],
                    size_mm=row["size_mm"],
                )
                for row in marker_rows
            ]

            keys = project_row.keys()
            p_margin = project_row["paper_margin_mm"] if "paper_margin_mm" in keys else PAPER_MARGIN_MM
            m_spacing = project_row["marker_spacing_mm"] if "marker_spacing_mm" in keys else MARKER_SPACING_MM
            use_custom = bool(project_row["use_custom_markers"]) if "use_custom_markers" in keys else bool(custom_markers)

            return PaperLayoutModel(
                paper_width_mm=project_row["paper_width_mm"],
                paper_height_mm=project_row["paper_height_mm"],
                paper_margin_mm=p_margin,
                marker_size_mm=project_row["marker_size_mm"],
                marker_spacing_mm=m_spacing,
                use_custom_markers=use_custom,
                buttons=buttons,
                custom_markers=custom_markers,
            )
