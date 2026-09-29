# Paper-Based Virtual Keyboard System: Architecture, Data Pipeline, and Research Reference

## 1. Research Overview and Core Technical Specifications

This research project develops a paper-based virtual keyboard interaction framework using a single ordinary monocular RGB webcam and standard printed paper. The system enables users to create customizable keypad or keyboard layouts, print them on plain A4 paper with AprilTag visual markers, and interact with flat surfaces in real time using their hands.

### Core Technical Principles
* **Single Regular RGB Camera Only:** The system works strictly with a standard monocular RGB camera (such as a laptop webcam or cheap USB camera). No specialized depth sensors, Time-of-Flight cameras, infrared emitters, stereo cameras, or wearable gloves are required.
* **Low and Standard Video Resolutions:** Reliable operation is maintained across standard 480p (640x480) and 720p (1280x720) resolutions.
* **Standard Commodity CPU Execution:** The complete detection, computer vision, and neural network inference pipeline runs entirely on standard CPUs without requiring a dedicated GPU.
* **12 FPS Pipeline Standard:** Video processing and feature windowing operate at a synchronized 12 FPS rate. This gives sufficient temporal resolution to capture touch deceleration dynamics while keeping CPU usage minimal.
* **Single-Hand Interaction Standard:** The system tracks one active hand at a time (MediaPipe `num_hands=1`). This single-hand design keeps CPU latency around 29 ms and prevents hand occlusions across the four border AprilTag fiducial anchors on compact A4 paper.
* **Scale Normalization with Direct Feature Propagation:** Hand skeletal coordinates are scale-normalized relative to unitless hand length ($L_{\text{hand}}$, distance from wrist to middle MCP joint). Coordinates are sent directly to the temporal sequence model without temporal smoothing filters to avoid artificial lag.
* **Decoupled Physical Geometry and Digital Semantics:** Physical paper geometry (button boundaries and visual markers) is separated from software action semantics. The same printed paper sheet can be dynamically mapped to different digital profiles (such as text typing, developer shortcuts, DAW audio controls, or terminal commands) without reprinting the sheet.

---

## 2. Directory Structure and Responsibilities

Following the project restructure, the repository is organized into distinct, modular functional components:

```
research/
├── virtualKeyboardSetup/       # Primary application suite
│   ├── designer/              # App 1: PySide6 Paper Layout Designer
│   └── detector/              # App 2: PySide6 Virtual Keyboard Detector & Runtime
├── annotator/                 # Ground-truth video annotation GUI tool
├── dataPipeline/              # 13-step feature extraction and dataset creation pipeline
├── modelBenchmark/            # Deep learning benchmark suite and trained model weights
├── videos/                    # Raw MP4 video files and synchronized landmark/annotation CSVs
├── chapters/                  # LaTeX thesis chapters (chapter01.tex to chapter06.tex)
├── figures/                   # Visual diagrams, charts, and architectural figures
├── sources/                   # Reference literature and university guidelines
├── out/                       # LaTeX build outputs (out/main.pdf)
├── output/                    # Final exported artifacts (output/main.pdf)
├── 1_generate_raw_landmarks.sh # Stage 1 shell runner
├── 2_run_annotator.sh         # Stage 2 shell runner
├── 3_run_data_pipeline.sh     # Stage 3 shell runner
├── 4_run_model_benchmarks.sh  # Stage 4 shell runner
├── run_designer.sh            # Designer application launcher
├── run_detector.sh            # Detector application launcher
└── compile_thesis.sh          # Master LaTeX thesis build script
```

### Detailed Directory Responsibilities

#### `virtualKeyboardSetup/designer/` (Application 1: Layout Designer)
* **Purpose:** Allows users to design custom keyboard layouts visually on a 2D interactive canvas.
* **Internal Architecture:** Built using PySide6 with a Model-View-ViewModel (MVVM) architecture.
* **Key Functions:**
  * Defines paper dimensions (standard A4 or Letter).
  * Automatically places AprilTag fiducial markers along the outer margins.
  * Allows creating, resizing, dragging, and labeling interactive button keys.
  * Validates spatial constraints (ensuring buttons stay within printable margins and do not overlap markers or each other).
  * Exports clean XML layout files containing physical millimeter coordinates and metadata.
  * Generates print-ready high-resolution vector PDF files embedded with official AprilTag 36h11 markers.

#### `virtualKeyboardSetup/detector/` (Application 2: Runtime Detector)
* **Purpose:** The real-time camera tracking, touch detection, homography mapping, and key execution engine.
* **Internal Architecture:** Multi-threaded MVVM application separating camera acquisition, vision processing, inference, and UI rendering.
* **Key Components:**
  * `CameraWorker`: Background thread capturing frames, running MediaPipe HandLandmarker, and assembling 5-frame temporal sliding windows.
  * `TouchPipelineService`: Executes multi-stage filtration (hand movement filtering, quality filtering, kinematic velocity checking).
  * `ModelRegistry` & `ModelDiscoveryService`: Manages neural network plugins and weights.
  * `HomographyEngine` & `AprilTagTracker`: Continuously estimates the 3x3 planar homography matrix ($H$) from detected AprilTags.
  * `TouchResolver`: Maps detected touch points through $H$ into physical millimeter space, applies distal fingertip offsets, and identifies target keys.
  * `ActionExecutor`: Simulates system keypresses or executes custom terminal commands.

#### `annotator/` (Video Dataset Annotator)
* **Purpose:** Standalone tool for annotating ground-truth finger contact events from recorded video files.
* **Internal Architecture:** Built using CustomTkinter and OpenCV.
* **Key Functions:**
  * Plays video files frame by frame at 12 FPS alongside MediaPipe skeletal overlays.
  * Allows annotators to tag discrete touchdown, touch contact, and liftoff intervals per individual finger (Thumb, Index, Middle, Ring, Pinky).
  * Exports window annotation CSV files (`*.window_annotations.*.csv`) matching video hashes.

#### `dataPipeline/` (Data Processing Pipeline)
* **Purpose:** Converts raw video files and annotation CSVs into clean, normalized, balanced training and testing datasets for deep learning.
* **Location of Outputs:** `dataPipeline/output/train_dataset.csv` and `dataPipeline/output/test_dataset.csv`.
* **Internal Source:** Located in `dataPipeline/src/`.

#### `modelBenchmark/` (Model Benchmark Suite)
* **Purpose:** Benchmarks neural network sequence models across 22 architectural configurations spanning five core deep learning families.
* **Location of Results:**
  * `modelBenchmark/results/summary_all.csv`: Full evaluation summary across all architectures.
  * `modelBenchmark/results/model_evaluation_report.json`: JSON report detailing precision, recall, F1, and training time.
  * `modelBenchmark/weights/`: Saved PyTorch model checkpoint weights.
  * `modelBenchmark/best_models/`: The top-performing weight (`best_finger_touch_lstm.pth`).

#### `videos/` (Dataset Recordings)
* **Purpose:** Holds all recorded camera video footage (`.mp4`), generated landmark CSV files (`*.raw_landmarks.*.csv`), and human-verified annotation files (`*.window_annotations.*.csv`).

---

## 3. Data Processing Pipeline (From Raw Video to Dataset)

The end-to-end dataset creation pipeline (`3_run_data_pipeline.sh`) consists of 13 sequential processing steps:

```
[Raw MP4 Videos in ./videos/]
          │
          ▼ (Step 1: annotator/mediapipe_extractor.py)
[Raw Landmark CSVs (30 FPS -> 12 FPS, 21 Joints)]
          │
          ▼ (Step 2: dataPipeline/src/normalize_landmarks.py)
[Scale-Normalized Coordinates (Divided by L_hand)]
          │
          ▼ (Step 3: dataPipeline/src/filter_landmarks.py)
[Direct Feature Passthrough (No Low-Pass Smoothing)]
          │
          ▼ (Step 4: dataPipeline/src/create_windows.py)
[5-Frame Sliding Windows (2-Frame Overlap, Stride 3)]
          │
          ▼ (Step 5: dataPipeline/src/merge_windows.py)
[Unified Windowed Dataset]
          │
          ▼ (Step 6: dataPipeline/src/filter_hand_movement.py)
[Hand Transit Movement Filter (Threshold <= 0.155 L_hand)]
          │
          ▼ (Step 7: dataPipeline/src/calculate_velocities.py)
[Kinematic Velocity Derivation (84-D Vector: Coords + Vel + Speeds)]
          │
          ▼ (Step 8: dataPipeline/src/filter_dataset.py)
[Zero-Velocity, Desync, and Missing Hand Filtration]
          │
          ▼ (Step 9: dataPipeline/src/filter_window_quality.py)
[Tracking Quality Filtration (Avg Score >= 0.65, Frame Score >= 0.45)]
          │
          ▼ (Step 10: dataPipeline/src/split_fingers.py)
[Per-Finger Sequence Unrolling (Thumb, Index, Middle, Ring, Pinky)]
          │
          ▼ (Step 11: dataPipeline/src/split_touch.py)
[Touch vs. Untouch Candidate Separation]
          │
          ▼ (Step 12: dataPipeline/src/create_train_test_split.py)
[Stratified Train/Test Split (80/20 Touch, No Video Leakage)]
          │
          ▼ (Step 13: Export)
[Final Datasets: train_dataset.csv (2774 rows), test_dataset.csv (636 rows)]
```

### Detailed Pipeline Stages

1. **Step 1: MediaPipe Landmark Extraction (`annotator/mediapipe_extractor.py`)**
   * Video feeds are sub-sampled to 12 FPS.
   * MediaPipe HandLandmarker extracts 21 skeletal joints (X, Y, Z, visibility, presence) along with detection confidence scores.
2. **Step 2: Scale Normalization (`normalize_landmarks.py`)**
   * Hand size varies depending on distance to the camera and person.
   * Reference hand length is computed as Euclidean distance between Wrist (joint 0) and Middle MCP (joint 9):
     $$L_{\text{hand}} = \sqrt{(X_{\text{mid\_mcp}} - X_{\text{wrist}})^2 + (Y_{\text{mid\_mcp}} - Y_{\text{wrist}})^2 + (Z_{\text{mid\_mcp}} - Z_{\text{wrist}})^2}$$
   * All coordinate offsets relative to the wrist are divided by $L_{\text{hand}}$, producing unitless, scale-invariant spatial coordinates.
3. **Step 3: Direct Feature Passthrough (`filter_landmarks.py`)**
   * Coordinates pass directly to temporal assembly without smoothing filters to preserve fast contact impact transients.
4. **Step 4: 5-Frame Temporal Windowing (`create_windows.py`)**
   * Hand motions are sliced into 5-frame sliding temporal windows with a 2-frame overlap (step size / stride = 3 frames).
   * At 12 FPS, each 5-frame window covers approximately 417 ms of motion, which is ideal for capturing descent, impact, and rebound.
5. **Step 5: Window Merging (`merge_windows.py`)**
   * Consolidates windowed data chunks from multiple recording sessions into a unified CSV.
6. **Step 6: Hand Movement Filtering (`filter_hand_movement.py`)**
   * During typing, keys are struck while the palm remains stationary relative to the surface.
   * Windows where the MCP joint cluster travels beyond a stationary threshold ($0.155 L_{\text{hand}}$) represent gross hand transport across keys and are excluded from contact training.
7. **Step 7: Kinematic Velocity Calculation (`calculate_velocities.py`)**
   * Derives frame-to-frame velocity components ($\Delta X, \Delta Y, \Delta Z$) and instantaneous scalar Euclidean speed across adjacent frames for all joints.
   * Formulates the complete 84-dimensional feature vector ($21 \times 4$ values: $X, Y, Z$ and scalar speed).
8. **Step 8: Dataset Cleaning (`filter_dataset.py`)**
   * Strips out unphysical artifacts, such as zero-velocity stationary frames marked as touches, desynchronized frames, or frames where fingers fell outside camera bounds.
9. **Step 9: Quality & Confidence Filtering (`filter_window_quality.py`)**
   * Filters out low-quality MediaPipe predictions:
     * Minimum average hand confidence score across the window: $\ge 0.65$.
     * Minimum single frame confidence score: $\ge 0.45$.
     * Maximum allowed confidence drop across consecutive frames: $\le 0.35$.
10. **Step 10: Per-Finger Unrolling (`split_fingers.py`)**
    * Unrolls full-hand 5-frame windows into individual finger sample instances (`thumb`, `index`, `middle`, `ring`, `pinky`).
11. **Step 11: Touch/Untouch Separation (`split_touch.py`)**
    * Separates positive physical contact instances from negative hovering or airborne motions.
12. **Step 12: Stratified Train/Test Split (`create_train_test_split.py`)**
    * Splits dataset into 80% training and 20% test samples with zero video leakage (ensuring recordings in the test set never appeared in the training set).
13. **Step 13: Export Finalized Datasets**
    * Training Set: 2,774 samples (`train_dataset.csv`).
    * Test Set: 636 samples (`test_dataset.csv`).
    * Synchronized directly to `modelBenchmark/data/` for immediate benchmarking.

---

## 4. Deep Learning Model Benchmark Results

The benchmark suite evaluated 22 deep learning architecture and feature representation configurations across five model families:
1. **Long Short-Term Memory (LSTM)**
2. **Bidirectional LSTM (BiLSTM)**
3. **1D Convolutional Neural Networks (CNN1D)**
4. **Residual Networks (ResNet1D)**
5. **Temporal Convolutional / Attention Networks (TCN / Transformer)**

### Benchmark Results Table (from `modelBenchmark/results/summary_all.csv`)

| Architecture | Hidden Units | Layers | Dropout | Learning Rate | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Training Time (s) | Model Weight File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LSTM_All_Combined** | **48** | **2** | **0.25** | **0.0010** | **94.34%** | **93.79%** | **94.97%** | **94.37%** | **5.2s** | `LSTM_All_Combined_cfg01.pth` |
| **LSTM_Finger_Only** | 32 | 2 | 0.25 | 0.0010 | 92.92% | 93.59% | 91.82% | 92.70% | 6.3s | `LSTM_Finger_Only_cfg01.pth` |
| **BiLSTM** | 32 | 2 | 0.25 | 0.0015 | 92.77% | 91.44% | 94.03% | 92.71% | 6.9s | `BiLSTM_cfg01.pth` |
| **LSTM_All_Joints_Vel** | 32 | 2 | 0.30 | 0.0010 | 92.77% | 93.57% | 91.51% | 92.53% | 9.3s | `LSTM_All_Joints_Vel_cfg01.pth` |
| **LSTM_Finger_Wrist** | 32 | 2 | 0.25 | 0.0010 | 92.92% | 91.41% | 93.71% | 92.55% | 9.6s | `LSTM_Finger_Wrist_cfg01.pth` |
| **LSTM_Combined** | 32 | 2 | 0.25 | 0.0010 | 92.45% | 91.08% | 93.08% | 92.07% | 5.7s | `LSTM_Combined_cfg01.pth` |
| **LSTM_Vel_Speed** | 32 | 2 | 0.25 | 0.0015 | 91.98% | 91.05% | 92.77% | 91.90% | 4.8s | `LSTM_Vel_Speed_cfg01.pth` |
| **LSTM_Tip_Vel_Ratios** | 32 | 2 | 0.25 | 0.0010 | 92.14% | 91.05% | 92.77% | 91.90% | 5.9s | `LSTM_Tip_Vel_Ratios_cfg01.pth` |
| **LSTM_Coords** | 48 | 2 | 0.20 | 0.0015 | 91.51% | 90.97% | 91.82% | 91.39% | 8.9s | `LSTM_Coords_cfg01.pth` |
| **LSTM_Velocities** | 32 | 2 | 0.25 | 0.0010 | 92.14% | 90.71% | 92.14% | 91.42% | 5.8s | `LSTM_Velocities_cfg01.pth` |

### Key Benchmark Findings
* **Winning Model:** `LSTM_All_Combined` achieved the highest performance across all evaluation metrics with **94.34% Test Accuracy** and **94.37% F1-Score**.
* **Confusion Matrix (Test Set of 636 samples):**
  * True Negatives (TN): 298
  * False Positives (FP): 20
  * False Negatives (FN): 16
  * True Positives (TP): 302
* **Inference Speed:** Runs in under 0.5 ms per temporal window on a standard commodity Intel/AMD CPU, comfortably satisfying real-time interaction requirements.
* **Primary Deployment Weight:** The trained weights from `modelBenchmark/weights/LSTM_All_Combined_cfg01.pth` are deployed directly into `virtualKeyboardSetup/detector/ai_model_plugins/lstm_all_combined/LSTM_All_Combined_cfg01.pth`.

---

## 5. Application 1: Paper Layout Designer Architecture

The Paper Layout Designer (`virtualKeyboardSetup/designer/main.py`) provides a visual editing environment for configuring custom virtual keyboards:

```
[PySide6 Fluent Window UI]
         │
         ▼
[DesignerViewModel / SettingsViewModel]
         │
    ┌────┴───────────────────────────┐
    ▼                                ▼
[Layout Canvas (GraphicsScene)]    [Validation Engine]
 - Key placement & sizing           - Margin boundary check
 - AprilTag fiducial anchors        - Key-to-key overlap check
 - Visual grid snapping             - Key-to-marker overlap check
    │                                │
    └────────────────┬───────────────┘
                     ▼
           [Export Subsystem]
            ├── XML Repository (Physical mm geometry & metadata)
            └── PDF Exporter (Vector A4 printable layout with AprilTags)
```

### Key Subsystems of the Designer
1. **Interactive Canvas (`ui/components/interactive_canvas.py`):**
   * Renders the true physical millimeter surface of the selected paper format (e.g. A4: 297 mm x 210 mm).
   * Supports mouse-driven dragging, resizing, alignment, and selection of keys.
2. **Domain Models (`core/models/`):**
   * `PaperLayout`: Holds overall paper dimensions, active printable surface bounds, marker family, and collections of buttons and markers.
   * `ButtonModel`: Holds button ID, display text/label, millimeter coordinates ($X, Y, W, H$), and associated action type.
   * `MarkerModel`: Stores fiducial marker IDs and their designated physical millimeter positions.
3. **Geometric Validation (`core/geometry/layout_geometry.py`):**
   * Enforces printable margin constraints.
   * Detects and blocks collisions between keys.
   * Prevents keys from overlapping the visual fiducial markers.
4. **Layout Exporting:**
   * **XML Export (`services/xml_repository.py`):** Saves the keyboard definition into structured XML containing exact millimeter boundaries, marker positions, and default action bindings.
   * **Vector PDF Generation (`services/pdf_exporter.py`):** Renders the layout into an exact-scale printable PDF. Generates high-contrast AprilTag 36h11 markers at the corners to enable optical tracking.

---

## 6. Application 2: Virtual Keyboard Detector Internal Architecture

The runtime detector application (`virtualKeyboardSetup/detector/main.py`) processes camera video, runs real-time deep learning inference, maps coordinates, and fires system actions:

```
[Camera Feed (OpenCV Capture @ 12 FPS)]
                 │
                 ▼
       [MediaPipe HandLandmarker]
                 │ (21 Skeletal Landmarks)
                 ▼
      [CameraWorker Thread]
                 │ (5-Frame Temporal Window)
                 ▼
     [TouchPipelineService] ──► [Pipeline Filters]
                                 ├─ Filter 1: Normalizer (L_hand)
                                 ├─ Filter 2: Hand Movement Filter
                                 ├─ Filter 3: Window Quality Filter
                                 └─ Filter 4: Velocity Pre-Check
                 │ (Passed Window)
                 ▼
      [PyTorch LSTM Touch Model]
                 │ (Per-Finger Touch Probabilities)
                 ▼
       [AprilTagTracker] ──────► [3x3 Homography Matrix (H)]
                 │
                 ▼
       [TouchResolver]
        ├─ Fingertip Pixel to Millimeter Mapping (P_mm = H * P_pixel)
        ├─ Distal Vector Forward Offset (DIP -> TIP)
        └─ Bounding Box Key Containment Hit-Test
                 │
                 ▼
    [DetectorViewModel (Buffer & Release Logic)]
        ├─ Multi-Touch Independence (Per-Finger State Tracking)
        ├─ Active Touch Window Buffering
        └─ Release Confirmation (Fires on 2 Consecutive Non-Touch Windows)
                 │
                 ▼
        [ActionExecutor]
         ├─ Simulated Keystrokes (X11 / evdev)
         └─ Shell Commands / Hotkeys
```

### Detailed Detector Pipeline & Filters

#### 1. Multi-Threaded Execution Architecture
* **`CameraWorker` Thread:** Runs the OpenCV video capture loop and MediaPipe hand landmarker independently of the GUI. It maintains a 5-frame temporal sliding FIFO buffer and emits `window_ready` events every 3 frames (stride of 3, 2-frame overlap).
* **Main UI Thread (`DetectorViewModel`):** Listens for `window_ready` events, orchestrates the filtration pipeline, evaluates neural network inference, updates visual telemetry graphs, and triggers action execution.

#### 2. The Internal Filtration Chain (`TouchPipelineService`)
Before executing deep learning inference, each 5-frame window must pass through four sequential sanity filters to prevent false touches:
1. **Filter 1 (Hand Length Scale Normalization):**
   * Computes $L_{\text{hand}}$ between wrist and middle MCP.
   * Normalizes joint coordinates into unitless space.
2. **Filter 2 (Hand Transit Movement Filter):**
   * Monitors displacement of the hand's MCP joints across the 5 frames.
   * If the hand is traveling horizontally across the surface (displacement $> 0.20 L_{\text{hand}}$), the window is flagged as hand transit motion and touch evaluation is skipped.
3. **Filter 3 (Window Quality Filter):**
   * Inspects MediaPipe tracking confidence across all 5 frames.
   * Requires average confidence $\ge 0.60$, minimum frame score $\ge 0.60$, and maximum confidence drop $\le 0.35$.
4. **Filter 4 (Kinematic Velocity Pre-Check):**
   * Verifies that the candidate finger is experiencing downward vertical deceleration.
   * If fingertip speed falls below the minimum threshold ($0.1200 L_{\text{hand}}/\text{frame}$), the window is identified as resting or stationary hovering and rejected.

#### 3. Deep Learning Touch Inference
* The normalized 84-dimensional feature vector across the 5 frames is fed into the PyTorch `LSTM_All_Combined` model.
* The model produces independent touch probability scores $[0.0, 1.0]$ for each of the five fingers: Thumb, Index, Middle, Ring, and Pinky.

#### 4. AprilTag Tracking and Planar Homography
* `AprilTagTracker` scans the camera frame using OpenCV ArUco with the `DICT_APRILTAG_36h11` dictionary.
* Detects the four corner visual fiducial markers on the printed paper layout.
* Pairs detected camera pixel corners $(u, v)$ with known physical millimeter layout corners $(X, Y)$ defined in the XML layout.
* Computes the $3 \times 3$ Planar Homography matrix ($H$) using `cv2.findHomography`:
  $$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
* This matrix dynamically compensates for camera tilt, perspective distortion, and paper re-positioning.

#### 5. Touch Resolution and Distal Forward Projection (`TouchResolver`)
* When a touch is confirmed by the neural network, the fingertip pixel coordinate $(u, v)$ from the touchdown frame is transformed into physical paper millimeter coordinates $(X, Y)_{\text{mm}}$ via $H$.
* **Distal Vector Offset:** Because the physical contact surface of a typing finger is the finger pad slightly in front of the fingernail joint tracked by MediaPipe, `TouchResolver` computes the distal directional vector from the Distal Interphalangeal (DIP) joint to the Fingertip (TIP):
  $$\vec{u} = \frac{\vec{P}_{\text{TIP}} - \vec{P}_{\text{DIP}}}{\|\vec{P}_{\text{TIP}} - \vec{P}_{\text{DIP}}\|}$$
  The contact point is projected forward by a configurable offset ($5.0\text{ mm}$):
  $$\vec{P}_{\text{contact}} = \vec{P}_{\text{TIP}} + \text{offset} \cdot \vec{u}$$
* **Hit-Testing:** Checks whether $\vec{P}_{\text{contact}}$ falls inside any button bounding box defined in the layout XML. If it lands within a small tolerance ($3.0\text{ mm}$) outside a key edge, it snaps to the nearest key.

#### 6. Touch Buffering and Release Confirmation
* **Sustained Touch Buffering:** When a finger touches down, the candidate key is highlighted in the UI and stored in a candidate buffer.
* **Lift-Off Release Trigger:** The keypress action does not fire upon initial downward impact. Instead, the system waits for release confirmation (2 consecutive non-touch windows indicating the finger has lifted off).
* **Benefits:** This eliminates false double-triggers, prevents dragging across adjacent keys, and ensures natural typing behavior without requiring artificial hover dwell delays.

#### 7. Action Execution (`ActionExecutor`)
* Dispatches the confirmed keypress to the host operating system.
* Supports standard keyboard keypress simulation (via X11/evdev), keyboard shortcuts (e.g. `Ctrl+C`, `Ctrl+V`), and custom terminal command execution (e.g. launching applications or executing scripts).

---

## 7. Thesis Writing Guide and Alignment Summary

When drafting your thesis chapters, align your descriptions with the structure and terminology documented here:

| Chapter | Recommended Topics to Cover from This Guide |
| :--- | :--- |
| **Chapter 3 (System Architecture & Methodology)** | Single RGB camera requirement, commodity CPU target, 12 FPS standard, single active hand interaction model, decoupled layout geometry vs. digital semantics, mathematical formulation of $L_{\text{hand}}$ scale normalization, and $3 \times 3$ homography transformation. |
| **Chapter 4 (Design and Implementation)** | Complete 13-step data processing pipeline, the four detector runtime filters (movement, quality, velocity), PySide6 MVVM design in App 1 (Designer) and App 2 (Detector), multi-threaded `CameraWorker`, distal vector projection, and touch-release buffering logic. |
| **Chapter 5 (Experimental Results & Evaluation)** | 22-model benchmark evaluation table, performance of `LSTM_All_Combined` (94.34% accuracy, 94.37% F1), confusion matrix breakdown, CPU inference latency (0.5 ms), and end-to-end pipeline latency (29.09 ms). |
| **Chapter 6 (Discussion & Conclusion)** | Solving the six literature gaps (monocular multi-finger detection, zero dwell-time delay, CPU efficiency, lighting robustness, paper tilt resilience, action multiplexing), current single-hand interaction advantages, and future extensions to bimanual typing. |
