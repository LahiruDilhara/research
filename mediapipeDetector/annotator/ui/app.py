"""
annotator/ui/app.py

Main CustomTkinter application window with screen switching and logging.
"""
import logging
import customtkinter as ctk

logger = logging.getLogger("Annotator.App")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AnnotatorApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Setting up main CTk window properties...")
        self.title("Touch Detection Data Annotator")
        self.geometry("1200x800")
        self.minsize(960, 680)
        self._screen: ctk.CTkFrame | None = None
        self._show_setup()

    def _switch(self, frame: ctk.CTkFrame) -> None:
        frame_name = frame.__class__.__name__
        logger.info(f"Switching active UI screen to: {frame_name}")
        if self._screen is not None:
            self._screen.destroy()
        frame.pack(fill="both", expand=True)
        self._screen = frame

    # ── Navigation ────────────────────────────────────────────────────────────

    def _show_setup(self) -> None:
        logger.info("Displaying SetupScreen...")
        from annotator.ui.setup_screen import SetupScreen
        self._switch(SetupScreen(self, app=self))

    def show_setup(self) -> None:
        self._show_setup()

    def show_processing(
        self,
        video_path: str,
        mode: str,                     # "new" | "resume"
        csv_dir: str = None,           # new mode: directory for csv
        csv_base: str = None,          # new mode: base filename (no extension/hash)
        csv_path: str = None,          # resume mode: full path to existing csv
        csv_hash_expected: str = None,  # resume mode: hash from csv filename
    ) -> None:
        logger.info(f"Transitioning to ProcessingScreen (mode={mode}, video={video_path})...")
        from annotator.ui.processing_screen import ProcessingScreen
        self._switch(ProcessingScreen(
            self, app=self,
            video_path=video_path, mode=mode,
            csv_dir=csv_dir, csv_base=csv_base,
            csv_path=csv_path, csv_hash_expected=csv_hash_expected,
        ))

    def show_recovery(
        self,
        frame_data: list,
        fps: float,
        total_frames: int,
        duration_ms: int,
        csv_path: str,
        video_path: str,
        video_hash: str,
    ) -> None:
        logger.info("Transitioning to RecoveryScreen...")
        from annotator.ui.recovery_screen import RecoveryScreen
        self._switch(RecoveryScreen(
            self, app=self,
            frame_data=frame_data, fps=fps,
            total_frames=total_frames, duration_ms=duration_ms,
            csv_path=csv_path, video_path=video_path, video_hash=video_hash,
        ))

    def show_annotation(
        self,
        frame_data: list,
        fps: float,
        total_frames: int,
        duration_ms: int,
        csv_path: str,
        video_path: str,
        video_hash: str,
        start_window_idx: int = 0,
    ) -> None:
        logger.info(f"Transitioning to AnnotationScreen (start_window_idx={start_window_idx})...")
        from annotator.ui.annotation_screen import AnnotationScreen
        self._switch(AnnotationScreen(
            self, app=self,
            frame_data=frame_data, fps=fps,
            total_frames=total_frames, duration_ms=duration_ms,
            csv_path=csv_path, video_path=video_path, video_hash=video_hash,
            start_window_idx=start_window_idx,
        ))
