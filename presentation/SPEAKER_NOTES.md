# Research Viva Defense: Presentation Speaker Notes & Demonstration Guide

**Project Title:** Customizable Paper-Based Virtual Keyboard Using Monocular Vision and Deep Learning  
**Candidate Name:** G A Lahiru Dilhara  
**Degree:** BSc (Hons) in Computer Science  
**Format:** 15-Slide Presentation & Live Demonstration  
**Target Duration:** ~12 to 15 Minutes (+ Q&A)

---

## Slide-by-Slide Defense Script

### Slide 1: Title Slide
* **Greeting:** "Good morning respected members of the panel, supervisors, and examiners. Today I am presenting my final year research project: *Customizable Paper-Based Virtual Keyboard Using Monocular Vision and Deep Learning*."
* **Opening Hook:** "The goal of this research is to transform any ordinary sheet of printed paper and a standard webcam into a responsive, real-time virtual keyboard without needing expensive depth sensors, infrared projectors, or GPUs."

---

### Slide 2: Background and Problem Statement
* **Key Talking Points:**
  1. Traditional monocular vision keyboards suffer from severe limitations:
     * **Single-finger restriction:** They can only track one index finger because multi-finger shadows merge and break tracking.
     * **Dwell-time delays:** Because standard 2D webcams lack depth perception, previous systems forced users to hover their finger over a key for 500 ms to 1000 ms before registering a press.
     * **High hardware requirements:** Systems that avoid dwell time usually require specialized 3D Time-of-Flight (ToF) cameras, infrared laser projectors, or dedicated GPUs.
     * **Rigid setups:** Any paper movement or camera vibration breaks calibration.
     * **Static layouts:** Fixed QWERTY designs with no way to remap keys to developer shortcuts or audio tools.
  2. **Core Research Challenge:** Can we achieve real-time multi-finger touch detection on a commodity CPU using only a single regular RGB camera and plain printed paper?

---

### Slide 3: Research Aim and SMART Objectives
* **Aim Statement:** State the primary research question clearly.
* **The 4 Core Objectives:**
  1. **Objective 1 (Design & Calibration):** Build a visual layout designer and an AprilTag homography tracking module robust to paper movement and tilt up to $75^\circ$.
  2. **Objective 2 (Data Pipeline & Kinematics):** Construct a 12 FPS sub-sampled dataset, extract 21 MediaPipe skeletal landmarks, and normalize scale by unitless hand length ($L_{\text{hand}}$).
  3. **Objective 3 (Deep Learning Benchmarking):** Train and benchmark 22 deep learning model configurations across 5 architecture families to select the optimal model for commodity CPUs.
  4. **Objective 4 (Runtime Engine):** Implement an end-to-end interactive runtime detector with distal projection ($5.0\text{ mm}$), multi-filtering, and native OS key injection at under $30\text{ ms}$ latency.

---

### Slide 4: End-to-End System Architecture
* **Key Talking Points:**
  1. Explain the decoupled two-app design:
     * **App 1 (Layout Designer):** Used beforehand to visually create custom key layouts and export printable PDFs with border AprilTags.
     * **App 2 (Runtime Detector Engine):** The real-time execution engine running during user interaction.
  2. Trace the runtime pipeline: Monocular RGB Camera $\rightarrow$ AprilTag Detection & Homography ($H$) $\rightarrow$ MediaPipe Hand Tracking (21 Joints) $\rightarrow$ Hand-Length Scale Normalization $\rightarrow$ PyTorch LSTM Inference $\rightarrow$ 4-Stage Runtime Filtering $\rightarrow$ OS Key Dispatch.

---

### Slide 5: Hardware and Physical Interaction Setup
* **Key Talking Points:**
  1. **Strict Zero-Specialized Hardware Standard:** Standard USB webcam (even 480p or 720p) and ordinary plain A4 paper from a standard home printer.
  2. **No Wearables or Markers on Hands:** Hands remain completely free of gloves or optical markers.
  3. **Commodity CPU Execution:** Designed to run purely on standard laptop CPUs without needing a discrete GPU.
  4. **Single Active Hand Standard:** The system is optimized to track one active hand at a time. This keeps CPU latency low ($29.09\text{ ms}$) and avoids covering the 4 border AprilTag anchors.

---

### Slide 6: App 1: Custom Layout Designer and Dynamic Multiplexing
* **Key Talking Points:**
  1. Built using **PySide6 (Qt)** with a visual drag-and-drop interface.
  2. Generates print-ready PDFs with 4 border AprilTag fiducial anchors accurately positioned in millimeters.
  3. **Action Multiplexing Innovation:** Physical layout geometry is decoupled from software key bindings.
  4. A single printed paper sheet can be dynamically bound to different software profiles (standard typing, VS Code developer shortcuts, DAW audio macros, gaming keypads, or terminal commands) by loading an XML configuration file without reprinting the sheet.

---

### Slide 7: Visual Planar Tracking via AprilTag Homography
* **Key Talking Points:**
  1. The 4 border AprilTags (family `tag36h11`) establish four known spatial anchor points.
  2. Computes the $3 \times 3$ Planar Homography matrix ($H$), mapping camera pixel coordinates $(u, v)$ to normalized paper coordinates $(x_p, y_p)$.
  3. **Dynamic Motion Compensation:** If the user bumps the paper or adjusts the camera, $H$ is updated frame-by-frame.
  4. Tested across camera perspective tilt angles up to $75^\circ$, maintaining sub-millimeter localization accuracy. If 1 tag is occluded, the remaining 3 tags sustain homography tracking.

---

### Slide 8: Feature Extraction and Kinematic Normalization
* **Key Talking Points:**
  1. **21 Skeletal Landmarks:** MediaPipe extracts 21 2D joint coordinates per frame.
  2. **Hand-Length Scale Normalization ($L_{\text{hand}}$):** Joint distances are divided by the anatomical distance between the wrist and middle MCP joint. This makes the feature vector unitless and invariant to camera distance.
  3. **Kinematic Velocity Features:** First-order temporal differences $(\Delta x, \Delta y)$ capture deceleration impact upon surface contact.
  4. **Sliding Window:** 5 frames with 2-frame overlap at 12 FPS produce a $(5 \times 84)$ temporal feature tensor.

---

### Slide 9: Dataset Engineering and Custom Annotator
* **Key Talking Points:**
  1. Ground-truth collection for monocular touch is challenging; standard datasets do not exist for multi-finger paper touch.
  2. Developed a dedicated **PySide6 Video Annotator tool** to label discrete touches across all five fingers (Thumb, Index, Middle, Ring, Pinky).
  3. Executed a 13-step data engineering pipeline with 12 FPS resampling, joint extraction, normalization, and quality filtering to eliminate ambiguous edge cases and ensure class balance.

---

### Slide 10: Deep Learning Architecture for Touch Classification
* **Key Talking Points:**
  1. **PyTorch Multi-Head LSTM:** Takes sequential input tensors of shape $(B, 5, 84)$.
  2. Uses a 2-layer LSTM with 128 hidden units and dropout (0.2).
  3. **Multi-Head Binary Output:** 5 parallel linear heads with Sigmoids evaluate independent touch probabilities for Thumb, Index, Middle, Ring, and Pinky.
  4. Trained using Multi-Label Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).

---

### Slide 11: Model Benchmarking and Empirical Results
* **Key Talking Points:**
  1. Benchmarked **22 distinct model configurations** across 5 deep learning families: 1D-CNN, Attention/Transformer, ResNet, BiLSTM, and LSTM.
  2. Compared 4 feature representations: raw landmarks, velocities only, normalized landmarks, and all features combined.
  3. **Winning Model:** `LSTM_All_Combined` achieved the highest performance: **94.34% Accuracy** and **94.37% F1-Score**.
  4. The confusion matrix and per-finger metrics demonstrate high precision across all five individual fingers.

---

### Slide 12: Real-Time Performance and Latency Breakdown
* **Key Talking Points:**
  1. **Total End-to-End Latency: 29.09 ms** (equivalent to $>34$ FPS throughput).
  2. Stage breakdown:
     * Camera Capture: $3.12\text{ ms}$
     * AprilTag Tracking & Homography: $4.35\text{ ms}$
     * MediaPipe Landmark Extraction: $16.82\text{ ms}$
     * PyTorch LSTM Inference: $3.68\text{ ms}$
     * Key Mapping & Event Dispatch: $1.12\text{ ms}$
  3. Proves that deep learning-based touch classification runs comfortably in real time on a standard commodity laptop CPU without requiring a GPU.

---

### Slide 13: App 2: Real-Time Runtime Detector Engine
* **Key Talking Points:**
  1. **Distal Vector Projection ($5.0\text{ mm}$):** Under perspective tilt, the camera sees the top of the fingernail rather than the physical contact point. The system projects $5.0\text{ mm}$ forward along the distal bone axis $(P_{\text{tip}} - P_{\text{dip}})$ to pinpoint the true touch location.
  2. **4-Stage Runtime Filtering:**
     * Temporal window smoothing buffer
     * Kinematic velocity deceleration confirmation
     * Multi-finger contact probability threshold ($P \ge 0.65$)
     * $120\text{ ms}$ debounce timer to prevent accidental double triggering
  3. **OS Key Injection:** Injects native key events into active OS applications via `pynput`.

---

### Slide 14: Practical Discussion and System Limitations
* **Key Talking Points:**
  1. **Key Advantages:** Truly low-cost, zero specialized sensors, real-time CPU speed, and dynamic layout flexibility.
  2. **Honest Limitations Acknowledged:**
     * Single active hand standard (bimanual typing on A4 causes AprilTag occlusion).
     * Camera perspective tilt exceeding $75^\circ$ increases homography corner jitter.
     * Severe ambient shadows under low-contrast paper can slightly distort fingertip landmarks.
  3. **Future Work:** Extension to two-handed interaction on A3 layout sheets and micro-edge deployment.

---

### Slide 15: Conclusion and Key Research Contributions
* **Key Talking Points:**
  1. **Summary of Contributions:**
     * Proved that monocular RGB video + kinematic deceleration LSTM can achieve multi-finger touch detection without depth sensors.
     * Verified commodity CPU real-time feasibility ($29.09\text{ ms}$ latency).
     * Demonstrated dynamic action multiplexing where one printed sheet serves multiple distinct software workflows.
     * Published an empirical benchmark across 22 models validating combined temporal features.
  2. **Closing:** "Thank you for your time and attention. I am now happy to demonstrate the system live and answer any questions."

---

## Live Demonstration Guide for the Viva Panel

1. **Step 1: Physical Setup Showcase**
   * Place the printed A4 paper layout on the desk.
   * Point the ordinary laptop webcam at the paper (show that any moderate tilt angle between $30^\circ$ and $60^\circ$ works instantly).
   * Show that there are no wires, no sensors, and no markers.

2. **Step 2: App 1 (Layout Designer) Demo**
   * Launch `virtualKeyboardSetup/designer/main.py`.
   * Show adding/modifying keys and switching between profiles (e.g. QWERTY typing vs Media/Shortcuts).
   * Show the PDF preview with the 4 corner AprilTags.

3. **Step 3: App 2 (Runtime Detector) Live Typing Demo**
   * Launch `virtualKeyboardSetup/detector/main.py`.
   * Point camera at the printed paper: observe instant AprilTag green planar bounding box.
   * Open a text editor (Notepad, Gedit, or VS Code).
   * Touch keys on the paper using different fingers (Index, Middle, Thumb).
   * Demonstrate instant key appearance in the text editor with no dwell time.
   * Move or rotate the paper slightly during typing to show the dynamic AprilTag homography compensation in action.
