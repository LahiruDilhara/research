#!/usr/bin/env bash
# ==============================================================================
# run_designer.sh
# Launches the Paper Layout Designer (PySide6 Fluent Window application)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGNER_DIR="$SCRIPT_DIR/designer"

if [ ! -d "$DESIGNER_DIR" ]; then
    echo "[ERROR] Designer directory not found at: $DESIGNER_DIR"
    exit 1
fi

cd "$DESIGNER_DIR"

# Ensure .env exists from .env.example if missing
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "[INFO] Initializing .env from .env.example..."
    cp ".env.example" ".env"
fi

# Detect python runtime (prefer existing virtual environment, fallback to uv / system python)
if [ -f "$DESIGNER_DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$DESIGNER_DIR/.venv/bin/python3"
elif command -v uv >/dev/null 2>&1; then
    PYTHON_EXEC="uv run python3"
else
    PYTHON_EXEC="python3"
fi

echo "========================================================================"
echo "  Starting Paper Layout Designer..."
echo "  Directory : $DESIGNER_DIR"
echo "  Runtime   : $PYTHON_EXEC"
echo "========================================================================"

exec $PYTHON_EXEC main.py "$@"
