# Annotator Tool — Implementation Plan

## Directory Structure
```
mediapipeDetector/
├── annotator.py                  ← entry point (root)
└── annotator/
    ├── __init__.py
    ├── constants.py              ← CSV headers, finger/joint names, filter params
    ├── pipeline.py               ← MediaPipe pipeline (mirrors live_camera.py)
    ├── video_processor.py        ← window math, JPEG-cached frame access, data extraction
    ├── csv_manager.py            ← write-close CSV ops (append / override)
    ├── utils.py                  ← SHA-256 hash, filename builder
    └── ui/
        ├── __init__.py
        ├── app.py                ← CTkApp with show_screen() switcher
        ├── setup_screen.py       ← new CSV / open CSV selection
        ├── processing_screen.py  ← threaded MediaPipe progress bar
        ├── recovery_screen.py    ← hash check, last-record summary
        ├── record_picker.py      ← modal Toplevel record list
        └── annotation_screen.py  ← main annotation UI

## Window Windowing
- WINDOW_SIZE = 5, OVERLAP = 2, STEP = 3
- window_idx → start_frame = window_idx * 3
- window_idx → end_frame   = start_frame + 4
- Drop final window if remaining new frames < 3

## CSV Filename Convention
  {user_base_name}.{video_hash_16chars}.csv
  hash extracted for verification on re-open

## CSV Record Per Window
- Identity: start_frame, end_frame, start_ms, end_ms
- Coords:   wrist + 5 fingers × 3 joints (MCP/PIP/DIP label) × (x,y) — from middle frame
- Velocities: wrist + 5 fingers × 3 joints × (vx,vy) — max-magnitude across window
- Annotations: touch flags, hand_move, pov, hand_closer, hovering, daylight, hand_visible, any_difference

## Override Logic (annotation_screen)
1. On NEXT: build new_rec from UI
2. find() by (start_frame, end_frame, start_ms, end_ms)
3. If not found → append
4. If found + data identical → just advance (no dialog)
5. If found + data different → messagebox.askyesno → override or stay

## CSV Crash Safety
- append()  : open-write-close (single record)
- override(): read-all → modify → write-all → close (atomic file replacement)
- Never keep file handle open between operations
