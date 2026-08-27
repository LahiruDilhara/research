"""
datacreator/annotator/app.py

Lightweight 12 FPS Hand Landmark Annotator Application Container.
VS Code Dark+ theme, CustomTkinter interface.
Manages smooth screen transitions between SetupScreen and AnnotationScreen.
"""

import argparse
import os
import sys
from pathlib import Path
import customtkinter as ctk

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacreator.annotator.setup_screen import SetupScreen
from datacreator.annotator.annotation_screen import AnnotationScreen

BG = "#1e1e1e"


class LightweightAnnotatorApp(ctk.CTk):
    def __init__(self, video_path: str = "", csv_path: str = "", window_size: int = 5, window_overlap: int = 2):
        super().__init__()

        self.title("Lightweight 12 FPS Window Hand Landmark Annotator")
        self.geometry("1240x860")
        self.configure(fg_color=BG)

        self.video_path = video_path
        self.csv_path = csv_path
        self.window_size = window_size
        self.window_overlap = window_overlap

        self.setup_screen = None
        self.annotation_screen = None

        self.show_setup_screen()

        # Auto-start if valid CLI file paths provided
        if video_path and csv_path and os.path.exists(video_path) and os.path.exists(csv_path):
            self.start_annotation_session(video_path, csv_path, window_size, window_overlap)

    def show_setup_screen(self):
        """Displays the initial Setup & Window Configuration Screen."""
        if self.annotation_screen:
            self.annotation_screen.pack_forget()
            self.annotation_screen.destroy()
            self.annotation_screen = None

        self.setup_screen = SetupScreen(
            self, app=self,
            initial_video=self.video_path,
            initial_csv=self.csv_path,
            window_size=self.window_size,
            window_overlap=self.window_overlap
        )
        self.setup_screen.pack(fill="both", expand=True)

    def start_annotation_session(self, video_path: str, csv_path: str, window_size: int, window_overlap: int):
        """Transitions to the main Annotation Screen."""
        self.video_path = video_path
        self.csv_path = csv_path
        self.window_size = window_size
        self.window_overlap = window_overlap

        if self.setup_screen:
            self.setup_screen.pack_forget()
            self.setup_screen.destroy()
            self.setup_screen = None

        self.annotation_screen = AnnotationScreen(
            self, app=self,
            video_path=video_path,
            csv_path=csv_path,
            window_size=window_size,
            window_overlap=window_overlap
        )
        self.annotation_screen.pack(fill="both", expand=True)


def main():
    parser = argparse.ArgumentParser(description="Lightweight 12 FPS Window Landmark & Touch Annotator GUI")
    parser.add_argument("-v", "--video", default="", help="Path to 12 FPS video file")
    parser.add_argument("-c", "--csv", default="", help="Path to raw landmarks CSV file")
    parser.add_argument("-w", "--window-size", type=int, default=5, help="Window size in frames (default: 5)")
    parser.add_argument("-o", "--overlap", type=int, default=2, help="Window overlap in frames (default: 2)")

    args = parser.parse_args()

    app = LightweightAnnotatorApp(
        video_path=args.video,
        csv_path=args.csv,
        window_size=args.window_size,
        window_overlap=args.overlap
    )
    app.mainloop()


if __name__ == "__main__":
    main()
