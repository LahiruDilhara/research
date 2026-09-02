# AGENTS.md - University Research Project & Agent Guidelines

> [!IMPORTANT]
> **CRITICAL SYSTEM TECHNICAL OVERRIDES & GUIDELINES:** Whenever reviewing project details, writing code, or drafting LaTeX thesis chapters, you **MUST ALWAYS FOLLOW** these up-to-date system technical specifications:
> - **Scale Normalization without 1€ Filtering:** Hand landmark coordinates are scale-normalized relative to unitless hand length. **DO NOT use 1€ filtering** or temporal smoothing.
> - **IEEE Citation Style:** All thesis chapters **MUST strictly use IEEE citation style** (`style=ieee` via BibLaTeX/biber).
> - **Thesis Assumption Context:** The major components (`designer/`, `mediapipeDetector/`, `aprilTag/`) have been built and tested separately, proving complete technical feasibility. **When writing thesis chapters, assume the entire assembled system (App 1 Designer + App 2 Runtime Engine) is fully created and operational as specified in Section 6 of this document.**

---

## 1. Research Rationale & Partitioned Development Strategy

This project focuses on the development and evaluation of a **customizable paper-based virtual keyboard system** powered by monocular computer vision, AprilTag fiducial homography tracking, and PyTorch deep learning.

### Partitioned Research Strategy
Because monocular paper touch detection is an experimental computer vision concept, component modules were built and tested in **isolated experimental partitions** to validate feasibility before assembling the final end-to-end application suite:

1. **Layout Design & Homography Simulation (`designer/` & `designer/analyzer/`)**:
   - `designer_app.py`: Main drag-and-drop PySide6 layout designer for creating key layouts, exporting layout XML, and printing PDF layouts embedded with AprilTag fiducial anchors.
   - `designer/analyzer/`: Simulated touch testing suite (`main.py`, `analyzer_app.py`, `homography_engine.py`) to verify that the $3 \times 3$ Planar Homography matrix ($H$) correctly maps camera pixel coordinates to target XML key regions.
2. **Dataset Creation & Pipeline Engineering (`mediapipeDetector/datacreator/`)**:
   - Video processing, 12 FPS sub-sampling (`resample_12fps.py`), 21 MediaPipe hand joint extraction, unitless hand-length scale normalization (`normalize_landmarks.py`), joint velocity calculation (`calculate_velocities.py`), 5-frame 2-overlap temporal windowing (`create_windows.py`), and dataset quality filtering (`filter_window_quality.py`).
3. **Deep Learning Model Benchmarking (`mediapipeDetector/deepLearningModels/`)**:
   - Driven by `run_all.py`, evaluating 22 pure 2D model architecture and feature representation combinations (LSTM, BiLSTM, 1D CNN, 1D ResNet).
   - Trained and selected the optimal PyTorch LSTM touch detection model (`best_finger_touch_lstm.pth`).
   - *Note:* Jupyter notebooks in this directory are legacy artifacts; `run_all.py` is the primary benchmark engine.
4. **Real-Time Responsiveness Evaluation (`mediapipeDetector/realtimeprocess/`)**:
   - Live testing scripts (`main_realtime_ui.py`, `camera_thread.py`, `model_manager.py`) to evaluate real-time inference latency, windowing responsiveness, and touch trigger stability.
5. **AprilTag Fiducial Tracking (`aprilTag/`)**:
   - Calibration and tracking scripts (`main.py`, `homography.py`, `estimater.py`) to localize AprilTag corner points and compute $H$ under planar tilt.

---

## 2. System Workflow & Pipeline Architecture

1. **Layout Design & Export**: The user creates a custom key layout using the GUI designer ([`designer/designer_app.py`](file:///home/lahirukasunidilhara/Documents/university/research/designer/designer_app.py)). The design is exported as:
   - An **XML file** containing key boundaries, button positions, command/key-press assignments, and fiducial marker anchor locations.
   - A **printable PDF** of the keyboard layout embedded with AprilTag fiducial markers.
2. **Printing & Physical Setup**: The user prints the paper virtual keyboard containing AprilTag fiducial anchors on any surface.
3. **Camera & Pose Estimation**: A monocular RGB video feed captures the physical surface. [MediaPipe Hand Landmarker](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector) extracts 3D/2D hand joint coordinates.
4. **Temporal Windowing & Feature Processing**:
   - Data is sub-sampled and scale-normalized (unitless hand-length distance normalization).
   - **No 1€ Filter / No Temporal Smoothing:** Raw coordinates are scale-normalized directly without temporal low-pass or 1€ filter smoothing.
   - Assembled into sliding windows of **5 frames with 2-frame overlap** (stride of 3 frames).
5. **Touch Detection (Custom Model)**: A custom PyTorch Deep Learning model ([best_finger_touch_lstm.pth](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/best_finger_touch_lstm.pth)) evaluates each 5-frame window to classify per-finger touch events.
6. **Homography Mapping & Key Execution**:
   - [AprilTag](file:///home/lahirukasunidilhara/Documents/university/research/aprilTag) markers on the printed paper are tracked to compute a $3 \times 3$ **Homography matrix ($H$)**.
   - When a finger touch event is confirmed, fingertip pixel coordinates ($P_{\text{pixel}}$) are mapped through $H$ into the XML layout coordinate space ($P_{\text{XML}} = H \cdot P_{\text{pixel}}$).
   - The corresponding key press or system command is executed.

---

## 3. Directory & Component Structure

Below is the layout of the project workspace:

| Directory / File                                                                                                                      | Description                                                                                                                                                                                                                                                                    | Status / Notes                                   |
| :------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------- |
| [`designer/designer_app.py`](file:///home/lahirukasunidilhara/Documents/university/research/designer/designer_app.py)                | **Main Layout Designer Application.** PySide6 GUI tool for designing layouts, mapping button IDs, exporting XML & printable AprilTag PDF.                                                                                                                                      | **Core Component (Operational)**                 |
| [`designer/analyzer/main.py`](file:///home/lahirukasunidilhara/Documents/university/research/designer/analyzer/main.py)               | Verification module for homography-based coordinate transformation using simulated finger touches.                                                                                                                                                                             | **Testing / Verification**                       |
| [`mediapipeDetector/datacreator/`](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/datacreator)      | Data pipeline: 12 FPS sub-sampling, landmark extraction, unitless scale normalization (`normalize_landmarks.py`), velocity derivation, and 5-frame 2-overlap window creation (`create_windows.py`).                                                                           | **Data Pipeline Engine**                         |
| [`mediapipeDetector/deepLearningModels/run_all.py`](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/deepLearningModels/run_all.py) | Deep learning model benchmark engine evaluating 22 pure 2D architectures (LSTM, BiLSTM, 1D CNN, 1D ResNet). Produces `best_finger_touch_lstm.pth`. (Jupyter notebooks in this dir are legacy).                                                                               | **Model Benchmark Engine**                       |
| [`mediapipeDetector/realtimeprocess/`](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/realtimeprocess) | Live real-time evaluation suite (`main_realtime_ui.py`, `camera_thread.py`, `model_manager.py`) to test live camera latency and windowing responsiveness.                                                                                                                      | **Real-Time Evaluation Engine**                  |
| [`aprilTag/`](file:///home/lahirukasunidilhara/Documents/university/research/aprilTag)                                                | Calibration and tracking scripts for AprilTag fiducial markers to compute $3 \times 3$ Homography matrix ($H$).                                                                                                                                                                | **Active Marker System**                         |
| [`opencvAruco/`](file:///home/lahirukasunidilhara/Documents/university/research/opencvAruco)                                          | Legacy ArUco marker testing scripts used during marker evaluation.                                                                                                                                                                                                             | Legacy                                           |
| [`sources/chapter-breakdown.md`](file:///home/lahirukasunidilhara/Documents/university/research/sources/chapter-breakdown.md)        | Detailed thesis chapter breakdown & section outline to follow when writing thesis chapters.                                                                                                                                                                                     | Active thesis guideline                          |
| [`sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf`](file:///home/lahirukasunidilhara/Documents/university/research/sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf) | Previous draft breakdown for Chapters 1, 2, and 3. Used for reference/context only (internal mechanisms updated; **do not cite**).                                                                                                                                           | Contextual draft reference                       |
| [`pdf-sources/`](file:///home/lahirukasunidilhara/Documents/university/research/pdf-sources)                                          | Repository storing PDF research papers and literature references.                                                                                                                                                                                                              | Paper storage                                    |
| [`research-db/`](file:///home/lahirukasunidilhara/Documents/university/research/research-db)                                          | Literature analysis database containing [`summary-matrix.md`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/summary-matrix.md) and [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib). | Thesis reference hub                             |
| [`chapters/`](file:///home/lahirukasunidilhara/Documents/university/research/chapters)                                                | Directory storing LaTeX `.tex` files for individual thesis chapters.                                                                                                                                                                                                           | Thesis writing                                   |
| [`main.tex`](file:///home/lahirukasunidilhara/Documents/university/research/main.tex)                                                 | Root LaTeX file that compiles all thesis chapters.                                                                                                                            .                                                                                                | Thesis entrypoint                                |
| [`out/`](file:///home/lahirukasunidilhara/Documents/university/research/out)                                                         | Output directory storing compiled PDF files and TeX build artifacts.                                                                                                                                                                           | Output directory                                 |
| [`figures/`](file:///home/lahirukasunidilhara/Documents/university/research/figures)                                                  | Visual assets, diagrams, charts, and figures used in the thesis.                                                                                                                                                                                | Thesis figures                                   |

---

## 4. Thesis Writing & Literature Research Protocol

When instructed to draft, research, or revise thesis chapters:

1. **LaTeX Format Requirement**:
   - The thesis **MUST be written entirely in LaTeX (`.tex`) format**.
   - Save individual LaTeX chapter files (`.tex`) in [`chapters/`](file:///home/lahirukasunidilhara/Documents/university/research/chapters).
   - Ensure every chapter file is included in the root LaTeX document [`main.tex`](file:///home/lahirukasunidilhara/Documents/university/research/main.tex).
2. **Follow Chapter Breakdown Outline**:
   - Always follow the detailed section breakdown and outline defined in [`sources/chapter-breakdown.md`](file:///home/lahirukasunidilhara/Documents/university/research/sources/chapter-breakdown.md) for chapter structure and section numbering.
3. **Contextual Reference for Early Chapters**:
   - When drafting Chapters 1, 2, and 3, refer to [`sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf`](file:///home/lahirukasunidilhara/Documents/university/research/sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf) to inspect previous writing and contextual background.
   - *Note:* Internal system mechanisms have evolved since that document was written, so prioritize the current pipeline architecture outlined in Section 1 of this document.
   - **Crucial Rule:** Do NOT use or cite `old_breakdown_of_chapter1_chapter2_chapter3.pdf` as a literature citation or bibliographic reference in the thesis.
4. **Strict Citation Protocol & IEEE Citation Style**:
   - The thesis **MUST strictly use the IEEE citation style** (`style=ieee` with BibLaTeX/biber).
   - Citations in LaTeX chapters (`\cite{CitationKey}`) must **ONLY** use BibTeX keys defined in [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib).
   - Consult [`summary-matrix.md`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/summary-matrix.md) first to map paper methodologies and findings to BibTeX keys in [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib).
5. **Targeted PDF Inspection in [`pdf-sources/`](file:///home/lahirukasunidilhara/Documents/university/research/pdf-sources)**:
   - Read specific target PDFs in [`pdf-sources/`](file:///home/lahirukasunidilhara/Documents/university/research/pdf-sources) to extract fine-grained technical details, mathematical formulations, or empirical data needed for chapter text.
6. **Figures and Visual Assets**:
   - Save all diagrams, charts, plots, and figures into [`figures/`](file:///home/lahirukasunidilhara/Documents/university/research/figures) and include them using standard LaTeX `\includegraphics` syntax.

---

## 5. TeX Live & System Toolchain & Compilation Protocol

The system has **`texlive-full`** and system development utilities installed. The available tools and their exact invocation syntax ("tool callings") are detailed below:

### Available Toolchain & Tool Callings:
- **Build Automation**:
  - `latexmk`: Automated multi-pass TeX compiler & dependency manager (`latexmk -pdf -outdir=out main.tex`)
  - Clean build artifacts: `latexmk -c -outdir=out`
- **Core Compilers**: `pdflatex -output-directory=out main.tex`, `xelatex`, `lualatex`
- **Bibliography Processor**: `biber --input-directory=out --output-directory=out main` or `bibtex out/main`
- **Inspection & Linting CLI Utilities**: `chktex main.tex`, `pdfinfo out/main.pdf`, `pdftotext out/main.pdf -`

### Mandatory Compilation & Output Protocol:
- **Output Directory (`./out`)**: All LaTeX tools, compilers, and bibliography processors **MUST** save generated PDF outputs and intermediate compilation artifacts into `./out`.
- **Always Compile & Verify**: Whenever you complete or modify writing in any `.tex` file or chapter, you **MUST run a compilation command** directing output to `./out` (e.g., `latexmk -pdf -outdir=out main.tex`).
- **Verify Output & Logs**: Inspect the compilation output and log files inside `./out` to verify that the build succeeds without fatal errors, missing references, or broken syntax.

---

## 6. Target Final System Architecture & Production Vision

> [!NOTE]
> **Thesis Writing Assumption:** Major system components have been built and tested in modular subdirectories (`designer/`, `mediapipeDetector/`, `aprilTag/`), proving complete technical feasibility. When writing the thesis, **assume that the integrated two-application suite defined below is fully implemented and operational.**

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 APPLICATION 1: LAYOUT DESIGNER               │
  │                     (designer/designer_app.py)              │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Exports Synchronized Artifacts
                                 ▼
                     ┌───────────────────────┐
                     │  • Printable PDF      │ (Printed on paper surface)
                     │  • Layout XML File    │ (Loaded into App 2)
                     └───────────┬───────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         APPLICATION 2: RUNTIME VIRTUAL KEYBOARD ENGINE       │
  ├─────────────────────────────────────────────────────────────┤
  │ 1. Setup & Action Command Mapper GUI                        │
  │    - Select Input Camera Feed                               │
  │    - Load Layout XML                                        │
  │    - Map buttons to Keystrokes or System Shell Commands     │
  │    - Save / Load Layout Action Configuration                │
  │                                                             │
  │ 2. Background Live Vision & Touch Execution Engine          │
  │    - Live Monocular Camera Watch                            │
  │    - AprilTag Tracker -> Homography Matrix (H)              │
  │    - MediaPipe Pose -> Scale Normalization (No 1€ filter)   │
  │    - 5-Frame Window -> PyTorch LSTM Touch Detection Model  │
  │    - Identify Active Finger & Fingertip Pixel Location      │
  │    - Planar Coordinate Mapping: P_XML = H * P_pixel          │
  │    - XML Key Lookup -> Trigger Mapped Keystroke / Command   │
  └─────────────────────────────────────────────────────────────┘
```

### Detailed Component Specifications

#### Application 1: Layout Designer Suite (`designer/designer_app.py`)
* **Role:** Interactive PySide6 desktop GUI tool for layout design and anchor placement.
* **Functionality:**
  * Drag-and-drop workspace for adding, resizing, and positioning key buttons on an A4 layout grid.
  * Automatic border placement of AprilTag fiducial marker anchors.
  * Dual export engine:
    1. **Printable PDF:** Vector layout sheet with embedded AprilTag anchors for physical printing.
    2. **Layout XML:** Structural specification defining key bounding coordinates, button IDs, default values, and marker anchor locations.

#### Application 2: Virtual Keyboard Runtime Engine & Command Mapper
* **Role:** Interactive setup GUI and background vision runtime engine.
* **Component A: Setup & Action Command Mapper GUI:**
  * **Camera Selector:** Allows the user to choose the active monocular RGB camera input stream.
  * **Layout Loader:** Loads any exported layout XML file.
  * **Interactive Action Mapping Table:** User maps each layout button ID to a specific action:
    * *Keystroke Action:* Single keys or key combinations (e.g., `'A'`, `'Space'`, `'Ctrl+C'`).
    * *System Command Action:* Complete executable shell commands or scripts (e.g., launching terminal applications, executing Python scripts, controlling media playback).
  * **Action Configuration Save/Load:** Users can save custom layout action mappings to an configuration file on disk and re-import them anytime for maximum operational flexibility.
* **Component B: Background Live Watch & Execution Engine:**
  * Once configured, the engine launches into the background, actively watching the live camera feed.
  * **AprilTag Tracking & Homography:** Continuously tracks paper AprilTag anchors to compute and update the $3 \times 3$ Homography matrix ($H$).
  * **Pose Extraction & Normalization:** MediaPipe Hand Landmarker extracts 21 hand landmarks, scale-normalizes joint coordinates relative to unitless hand length (without 1€ smoothing), and constructs 5-frame sliding windows (stride of 3 frames).
  * **PyTorch LSTM Classifier:** The optimal trained PyTorch LSTM model (`best_finger_touch_lstm.pth`) evaluates 5-frame window sequences to detect finger surface contact and identify the active finger.
  * **Fingertip Planar Mapping:** Translates the active fingertip pixel location ($P_{\text{pixel}}$) through $H$ into XML coordinate space ($P_{\text{XML}} = H \cdot P_{\text{pixel}}$).
  * **Key Lookup & Action Dispatch:** Locates the hit key in the layout XML and executes the user's mapped keystroke or system shell command.

### Key Architectural Advantages & Operational Flexibility Specifications

1. **Multi-Layout & Multi-Configuration Flexibility**:
   - **Arbitrary Layout Library:** Any user can design and maintain a library of layout XML/PDF templates tailored to distinct work environments (e.g., standard text entry, software shortcut pads, DAW audio control, gaming macros, or accessibility keyboards).
   - **Per-Layout Configuration Profiles:** For any single printed paper layout sheet, users can create, save, and load multiple action mapping configuration files (e.g., "Work Mode" vs. "Gaming Mode" vs. "Media Control Mode" for the exact same physical sheet), offering complete operational adaptability.

2. **Fiducial Tracking & Surface Freedom**:
   - **Partial Marker Visibility Tolerance:** The AprilTag tracking engine can maintain homography estimation even if only a subset of AprilTag corner anchors (e.g., 2 markers) are visible, eliminating the requirement for the physical paper sheet to remain entirely unobstructed.
   - **Orientation & Perspective Invariance:** The system functions reliably across arbitrary paper rotation, surface inclination, perspective tilt, and camera viewing angles because $H$ continuously rectifies perspective distortions.

3. **Environmental & Morphological Robustness**:
   - **Illumination Invariance:** MediaPipe pose estimation and keypoint localization operate reliably across dark, bright, or uneven ambient lighting conditions, completely overcoming the shadow instability of classical single-camera systems.
   - **Skin Tone & Hand Morphology Invariance:** Hand keypoint extraction relies on 3D/2D structural skeletal joint geometry rather than skin-color thresholding, ensuring unbiased performance across diverse skin tones, finger shapes, and hand sizes.
   - **Camera Distance Invariance:** Unitless hand-length scale normalization ensures spatial kinematic features remain invariant regardless of how close or far the user's hand is from the camera.

4. **Universal Hardware Accessibility (12 FPS & Resolution Independence)**:
   - **12 FPS Low-Resource Standard:** The vision and touch classification pipeline operates smoothly at 12 FPS sub-sampling, allowing real-time execution on low-cost commodity RGB webcams, legacy cameras, or budget embedded hardware without requiring high-speed camera sensors.
   - **Resolution Independence:** Because joint coordinates are normalized relative to hand length and mapped through planar homography, the system operates identically across 480p, 720p, 1080p, or 4K camera streams.
