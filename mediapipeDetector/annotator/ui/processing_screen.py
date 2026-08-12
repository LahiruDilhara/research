"""
annotator/ui/processing_screen.py

Video processing progress screen — VS Code Dark+ theme.
Thread-safe Queue polling architecture with SHA-256 fingerprint verification.
"""
import logging
import os
import queue
import threading
from tkinter import messagebox

import customtkinter as ctk

from annotator.constants import MODEL_PATH
from annotator.csv_manager import CSVManager
from annotator.pipeline import process_video
from annotator.utils import build_csv_path, compute_file_hash, extract_hash_from_csv_filename

logger = logging.getLogger("Annotator.ProcessingScreen")
FF = "Helvetica"   # font family


class ProcessingScreen(ctk.CTkFrame):
    def __init__(
        self, parent, app,
        video_path: str,
        mode: str,                     # "new" | "resume"
        csv_dir: str = None,           # new mode
        csv_base: str = None,          # new mode
        csv_path: str = None,          # resume mode (pre-known path)
        csv_hash_expected: str = None,  # resume mode (hash from filename)
    ) -> None:
        super().__init__(parent, fg_color="#1e1e1e")
        self.app = app
        self.video_path = video_path
        self.mode = mode
        self.csv_dir = csv_dir
        self.csv_base = csv_base
        self.csv_path = csv_path
        self.csv_hash_expected = csv_hash_expected
        self.video_hash: str = ""
        self._alive = True
        self._queue = queue.Queue()
        self._mismatch_event = None
        self._mismatch_proceed = False
        self._build()
        logger.info(f"Initialized ProcessingScreen for video: {video_path} (mode={mode})")
        self._start()

    def _build(self) -> None:
        outer = ctk.CTkFrame(
            self,
            width=540,
            corner_radius=4,
            border_width=1,
            border_color="#3c3c3c",
            fg_color="#252526",
        )
        outer.place(relx=0.5, rely=0.5, anchor="center")

        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(padx=36, pady=36, fill="both", expand=True)

        ctk.CTkLabel(
            inner,
            text="Processing Video Pipeline",
            font=ctk.CTkFont(family=FF, size=20, weight="bold"),
            text_color="#ffffff",
        ).pack(anchor="w", pady=(0, 6))

        ctk.CTkLabel(
            inner,
            text="Running MediaPipe hand landmark detection, scale normalisation, 1€ filter, and velocity calculation.",
            font=ctk.CTkFont(family=FF, size=12),
            text_color="#cccccc",
            justify="left",
            wraplength=460,
        ).pack(anchor="w", pady=(0, 24))

        self._status = ctk.CTkLabel(
            inner,
            text="Initializing pipeline...",
            font=ctk.CTkFont(family=FF, size=13),
            text_color="#cccccc",
            anchor="w",
        )
        self._status.pack(anchor="w", pady=(0, 8))

        self._bar = ctk.CTkProgressBar(
            inner, width=460, height=8, corner_radius=2,
            fg_color="#3c3c3c", progress_color="#007acc",
        )
        self._bar.set(0)
        self._bar.pack(anchor="w", pady=(0, 8))

        self._frame_lbl = ctk.CTkLabel(
            inner,
            text="Frame 0 / 0",
            font=ctk.CTkFont(family=FF, size=11),
            text_color="#858585",
            anchor="w",
        )
        self._frame_lbl.pack(anchor="w")

    def _start(self) -> None:
        logger.info("Spawning background thread for video processing...")
        threading.Thread(target=self._run_worker, daemon=True).start()
        self.after(30, self._poll_queue)

    def _poll_queue(self) -> None:
        if not self._alive:
            return
        try:
            while True:
                msg = self._queue.get_nowait()
                msg_type = msg[0]

                if msg_type == "status":
                    self._status.configure(text=msg[1])

                elif msg_type == "progress":
                    _, pct, done, total = msg
                    self._bar.set(pct)
                    self._frame_lbl.configure(text=f"Frame {done} / {total}")
                    self._status.configure(text=f"Extracting features ({int(pct * 100)}%)")

                elif msg_type == "hash_mismatch":
                    _, expected, actual = msg
                    logger.warning(f"Displaying SHA-256 Hash Mismatch warning dialog: expected '{expected}', actual '{actual}'")
                    proceed = messagebox.askyesno(
                        "⚠ Video Fingerprint Mismatch",
                        f"The selected video SHA-256 hash does NOT match this CSV dataset!\n\n"
                        f"  CSV Expected Hash :  {expected}\n"
                        f"  Selected Video Hash: {actual}\n\n"
                        "Calculated velocities and landmarks will not correspond to recorded annotations.\n\n"
                        "Do you want to proceed anyway?",
                        parent=self,
                    )
                    self._mismatch_proceed = proceed
                    if self._mismatch_event:
                        self._mismatch_event.set()
                    if not proceed:
                        logger.info("User chose NOT to proceed after hash mismatch. Returning to SetupScreen.")
                        self.app.show_setup()
                        return

                elif msg_type == "done":
                    _, frame_data, fps, total_frames, duration_ms = msg
                    self._bar.set(1.0)
                    self._status.configure(text="Processing complete")
                    logger.info("Main thread received 'done' signal. Executing screen transition.")
                    self._done(frame_data, fps, total_frames, duration_ms)
                    return

                elif msg_type == "error":
                    _, err_msg = msg
                    logger.error(f"Main thread received 'error' signal: {err_msg}")
                    self._error(err_msg)
                    return

        except queue.Empty:
            pass

        self.after(30, self._poll_queue)

    def _run_worker(self) -> None:
        """Background worker thread function."""
        try:
            logger.info("Background thread started. Computing file hash...")
            self._queue.put(("status", "Computing SHA-256 video fingerprint..."))
            self.video_hash = compute_file_hash(self.video_path)
            logger.info(f"Computed SHA-256 video hash: {self.video_hash}")

            if self.mode == "new":
                self.csv_path = build_csv_path(self.csv_dir, self.csv_base, self.video_hash)
                logger.info(f"Target CSV path determined: {self.csv_path}")
            else:
                expected_hash = self.csv_hash_expected
                if not expected_hash and self.csv_path:
                    expected_hash = extract_hash_from_csv_filename(self.csv_path)

                if expected_hash and expected_hash != self.video_hash:
                    logger.warning(f"SHA-256 Hash Mismatch detected! CSV expected: '{expected_hash}', Video actual: '{self.video_hash}'")
                    self._mismatch_event = threading.Event()
                    self._mismatch_proceed = False
                    self._queue.put(("hash_mismatch", expected_hash, self.video_hash))
                    self._mismatch_event.wait()
                    if not self._mismatch_proceed:
                        logger.info("Worker thread aborting due to user rejection of hash mismatch.")
                        return

            if self.mode == "new":
                logger.info(f"Creating new CSV file headers at: {self.csv_path}")
                self._queue.put(("status", "Creating CSV file..."))
                CSVManager(self.csv_path).create()

            def progress(done: int, total: int) -> None:
                pct = done / total if total > 0 else 0.0
                self._queue.put(("progress", pct, done, total))

            logger.info("Invoking process_video pipeline...")
            self._queue.put(("status", "Running MediaPipe landmark detector..."))
            frame_data, fps, total_frames, duration_ms = process_video(
                self.video_path, MODEL_PATH, progress
            )
            logger.info(f"process_video completed: {total_frames} frames, {fps:.2f} FPS, {duration_ms} ms")

            self._queue.put(("done", frame_data, fps, total_frames, duration_ms))

        except Exception as exc:
            logger.exception(f"Error during video processing worker thread: {exc}")
            self._queue.put(("error", str(exc)))

    def _done(self, frame_data, fps, total_frames, duration_ms) -> None:
        if not self._alive:
            return
        logger.info(f"Transitioning from ProcessingScreen (frames={total_frames}, duration={duration_ms}ms)...")
        if self.mode == "new":
            self.app.show_annotation(
                frame_data, fps, total_frames, duration_ms,
                self.csv_path, self.video_path, self.video_hash,
                start_window_idx=0,
            )
        else:
            self.app.show_recovery(
                frame_data, fps, total_frames, duration_ms,
                self.csv_path, self.video_path, self.video_hash,
            )

    def _error(self, msg: str) -> None:
        messagebox.showerror("Processing Error", msg)
        self.app.show_setup()

    def destroy(self) -> None:
        self._alive = False
        super().destroy()
