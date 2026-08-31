"""
main.py

Entry point to launch the Virtual Keyboard Layout Analyzer & Touch Detector app.
Usage:
    python3 main.py [path/to/layout.xml]
"""

import sys
import os

# Ensure current analyzer directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer_app import AnalyzerApp

def main():
    initial_xml = sys.argv[1] if len(sys.argv) > 1 else None
    app = AnalyzerApp(initial_xml=initial_xml)
    app.mainloop()

if __name__ == "__main__":
    main()
