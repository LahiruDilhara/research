"""
Database table schema definitions for SQLite database.
"""

CREATE_PROJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    paper_width_mm REAL NOT NULL,
    paper_height_mm REAL NOT NULL,
    paper_margin_mm REAL DEFAULT 10.0,
    marker_size_mm REAL NOT NULL,
    marker_spacing_mm REAL DEFAULT 40.0,
    use_custom_markers INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_BUTTONS_TABLE = """
CREATE TABLE IF NOT EXISTS buttons (
    id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    x_mm REAL NOT NULL,
    y_mm REAL NOT NULL,
    width_mm REAL NOT NULL,
    height_mm REAL NOT NULL,
    text TEXT DEFAULT '',
    font_size_pt INTEGER DEFAULT 14,
    PRIMARY KEY (id, project_id),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
"""

CREATE_MARKERS_TABLE = """
CREATE TABLE IF NOT EXISTS markers (
    id INTEGER NOT NULL,
    project_id TEXT NOT NULL,
    x_mm REAL NOT NULL,
    y_mm REAL NOT NULL,
    size_mm REAL NOT NULL,
    PRIMARY KEY (id, project_id),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
"""

CREATE_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
