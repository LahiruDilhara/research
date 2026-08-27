"""
datacreator/annotate.py

Launcher for the lightweight 12 FPS hand landmark & touch annotator GUI.
"""

import sys
from pathlib import Path

# Add project root to sys.path for robust imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacreator.annotator.app import main

if __name__ == "__main__":
    main()
