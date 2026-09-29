# My Research Journey: Chronological & Thematic Restructuring

This document reorganizes the full personal narrative from `sources/myStory.txt` in chronological order and logical thematic phases. Every single piece of information, technical decision, trial, failure, and insight from the original text is preserved without omission.

---

## 1. Initial Ideology, Motivation & Feasibility Concept

* **Dream and Fantasy:** When starting, it felt like a fantasy. The goal was to conduct an enjoyable, unique, and different research project that brought an old dream to life.
* **The Long-Term Vision:** An extensible touch-based interface, specifically tailored as a customizable macro keyboard.
* **Initial Feasibility Check:** An initial rough feasibility assessment suggested the project was viable: capture video from a camera, process the feed to recognize keys, and detect physical touch.

---

## 2. Phase 1: Hand-Drawn Paper Layouts and Freehand Vision (Initial Concept & Failure)

* **Initial Vision of Ultimate Freedom:** The very first concept was completely freehand and printer-free. The user would simply draw keys by hand on a plain sheet of paper without needing any printed markers or templates.
* **Proposed Workflow:**
  1. The user draws their own keyboard layout on paper.
  2. The user draws four plain black square boxes on the corners of the sheet.
  3. The user measures the physical size of the markers and the distances between them.
  4. *Initialization Phase:* The user shows the paper to the camera without their hands over it so the computer scans the drawing, identifies keys, boundaries, and markers, and loads the layout digitally.
  5. The user assigns digital commands or keys to each drawn box.
  6. The user types, and the camera detects touch events and computes homography from visible markers.
* **Problems Encountered with Freehand Drawing:**
  * **Hardware Constraint (Basic Low-Quality Camera):** The author only had a basic camera with low resolution, noticeable lens distortion, and poor image quality.
  * **OpenCV Detection Failures:** OpenCV could not reliably detect hand-drawn boxes because low camera resolution, hand-drawn lines (not neat, irregular connections), varying marker margins, ink colors, and ambient lighting caused severe distortion and missed detections.
  * **Loss of Versatility:** Supporting hand-drawn layouts would force strict limitations on camera angle, camera distance, and lighting conditions. These rigid requirements destroyed user versatility.
* **Key Design Compromise:** Struck a middle ground between complete freehand drawing and usability. To preserve the core research objective of complete layout freedom while eliminating vision errors from hand drawings, layout creation was shifted to a computer software designer followed by standard printing. Users could still create any key sizes and arrangements digitally.

---

## 3. Phase 2: Computer-Assisted Layouts & The Coordinate Projection Breakthrough

* **Initial Printed Setup with Plain Corner Markers:**
  * The user designs any layout on the computer with fixed margin sizes and distances, then prints it on plain paper.
  * Initially retained four plain black corner markers for homography calibration.
* **New Technical Hurdles:**
  * Far-away keys were still difficult to detect using low-resolution cameras.
  * Hand Occlusion: The four corner markers were frequently blocked by the user's hands. If even a single corner marker was occluded, the entire 4-point homography calculation failed completely.
* **The Major Architectural Breakthrough (Eliminating Key Detection & The Initialization Step):**
  * The author realized that since the layout is designed on the computer and loaded digitally, the exact mathematical $(x, y)$ coordinate boundaries of every key are already known by the software.
  * Instead of training a heavy vision model to detect individual printed key boxes from the camera stream, the system only needs to find the paper plane via homography and project the digital layout directly onto the camera view.
  * This eliminated key-box detection entirely. Keys could now be accurately identified even when fully occluded by hands, drastically simplifying the system pipeline.
  * **Elimination of the Initialization Step:** Because the layout is designed by the computer and loaded directly into the runtime system, an initialization step is no longer required. In early concepts, users had to present the bare paper to the camera without hands so the system could scan and discover key positions. Having the layout pre-loaded in software gave the major practical advantage of eliminating this initialization step completely, allowing users to start typing immediately with their hands already resting on the surface.

---

## 4. Phase 3: Fiducial Marker Evolution (Plain Markers $\rightarrow$ ArUco $\rightarrow$ AprilTags $\rightarrow$ Redundant Margins)

* **Moving Away from Plain Markers:** Because the layout was now printed, standard computer vision fiducial markers could be used instead of plain drawn squares.
* **Testing ArUco Markers:**
  * ArUco markers were printed on the corners.
  * A dedicated prototype testing application was developed to detect markers and visually overlay the digital layout to verify alignment.
  * *Limitation:* ArUco markers struggled with camera tilt, steep viewing angles, and distance.
* **Discovering AprilTags:**
  * Researched alternative fiducial marker systems (ArUco, AprilTag, STags). AprilTags proved to be the most promising.
  * Tested AprilTags on numerous real-world internet images under extreme lighting, distance, and steep angles; AprilTags demonstrated superior detection robustness.
* **Overcoming Hand Occlusion with Redundant Tags:**
  * Having only 4 corner markers remained brittle because a user's hand still blocked individual corners during typing.
  * Homography mathematically requires 4 points, but each individual AprilTag provides 4 distinct corner points.
  * *Solution:* Placed multiple redundant AprilTags along the outer margins of the paper sheet.
  * *Result:* Even if the user's hands cover several markers, the remaining visible tags provide more than enough corner points to maintain an accurate homography matrix ($H$). The entire layout sheet no longer needs to be fully visible; partial visibility is sufficient.
* **Physical Print Scaling Calibration:**
  * Observed that different desktop printers scale documents slightly differently during printing.
  * Added a calibration feature in the designer/runtime software where the user can measure the physical edge width of a printed marker using a standard ruler and enter that value, allowing the software to calibrate the exact physical scale of the layout.
* **Designer Software Evolution:**
  * First developed as a prototype using `customtkinter`.
  * Later rebuilt and finalized into a robust desktop suite using `PySide6`.

---

## 5. Phase 4: Touch Detection & Neural Architecture Evolution

* **Goal:** Reliable touch detection under arbitrary hand rotations, camera angles, and varying ambient conditions.
* **Trial 1: Pure 2D CNN on Video Frames:**
  * Heavy computational footprint.
  * Failed under lighting changes.
  * Required an impractically large image training dataset.
* **Realization on Temporal Dynamics:** Touch is a dynamic physical process that cannot be accurately determined from a single static video frame; it requires evaluating a temporal sequence across consecutive frames.
* **Trial 2: CNN + LSTM Hybrid:**
  * Considered extracting spatial feature maps with a CNN and passing them into an LSTM.
  * Discarded because it directly contradicted the second core research goal: **running in real-time on standard commodity CPUs without requiring a GPU**.
* **Trial 3: MediaPipe Hand Landmarks + 64x64 CNN Fingertip Patches:**
  * Found Google MediaPipe: tracks 21 3D/2D hand landmarks, pre-calibrated, highly accurate, runs natively on CPU, and is invariant to hand appearance and skin tone.
  * Realized that MediaPipe provides the ability to capture each individual finger separately with complete ease, since skeletal landmarks are pre-indexed to specific digits (Thumb, Index, Middle, Ring, Pinky).
  * Attempted cropping $64 \times 64$ pixel patches around fingertips to classify touch with a CNN.
  * *Architectural Decision on Speed:* When deciding on CNN architectures, the author initially thought to evaluate each finger separately using individual CNN models or separate processing branches. However, running separate evaluations for each finger multiplied computation and was still too heavy for smooth CPU real-time execution.
* **Trial 4: Pure Coordinate Kinematics + Unified LSTM (Eliminating Per-Finger CNNs and Shadows):**
  * Decided not to process fingers through separate CNNs. Instead, completely removed image-based CNN processing after MediaPipe.
  * Avoided shadow-based touch detection (which fails under diffuse lighting, multiple lights, or low ambient light).
  * Passed purely numerical, scale-normalized landmark coordinate kinematics of the entire hand directly into a single unified LSTM sequence model that predicts touches across all fingers concurrently.
  * This eliminated multi-model inference overhead and redundant image processing, making the entire pipeline dramatically faster and fully real-time on standard CPUs.

---

## 6. Phase 5: Normalization, Hardware Diagnostics & The 12 FPS Standard

* **Resolution and Hand Size Invariance:**
  * Designed a coordinate normalization scheme based on MediaPipe palm landmark distances (unitless hand-length normalization).
  * Ensures coordinates remain stable whether the hand is large or small, close to or far from the camera, and regardless of camera video resolution.
* **Hardware Diagnostic & The 13 FPS Discovery:**
  * Observed unexpected landmark jitter when processing video, even though the webcam software reported 30 FPS.
  * Built a custom diagnostic Python application to measure true hardware throughput and discovered the physical camera was actually only delivering 13 FPS.
* **The 12 FPS Standardization:**
  * Realized different commodity webcams operate at inconsistent real-world frame rates.
  * Because recurrent neural networks (LSTM) evaluate motion across discrete time steps, fluctuating frame rates distort perceived finger velocity and cause detection errors.
  * Capped the system pipeline at a uniform **12 FPS**. Since virtually all commodity cameras can sustain 12 FPS, this standardized the time step ($\Delta t$), guaranteed consistent velocity inputs for the LSTM, and fit comfortably within the CPU processing budget for real-time responsiveness.

---

## 7. Phase 6: Dataset Engineering, Annotation Overhaul & The 13-Step Pipeline

* **Dataset Capture Setup:**
  * Developed a dedicated Python data recording and annotation application.
  * Real Camera Rig & Physical Desk Measurements: Camera height between $20\text{ cm}$ and $35\text{ cm}$ vertically above the desk, with a horizontal distance from camera base to the paper sheet between $30\text{ cm}$ and $60\text{ cm}$. Downward camera viewing angle is determined directly by right-angle trigonometry ($\theta = \arctan(h/d) \approx 18.4^\circ\text{ to } 49.4^\circ$ from horizontal, optical line-of-sight distance $36.1\text{ to } 69.5\text{ cm}$), removing the need for manual angle measurement.
  * Captured flat-surface hand tapping videos across different cameras and frame rates.
  * Recorded 5 different individuals' hands (ensuring the model is hand-independent).
  * Video protocol: 10-second video clips collected over 5 different days at various times of day, yielding a total of **3,800 seconds of training footage**.
* **Temporal Sliding Window Construction:**
  * Initially tried non-overlapping windows, but rapid taps were frequently split across window boundaries, ruining touch detection.
  * Adopted a **5-frame window with a 2-frame overlap/stride**. This captured the complete deceleration and impact curve while keeping the interface responsive by evaluating new inferences every 2-frame shift.
* **First Annotation Iteration & The 82% Accuracy Plateau:**
  * The first version of the data collection software only extracted MCP, PIP/DIP, and fingertip coordinates.
  * Initial LSTM models were trained in Jupyter notebooks. Because text/coordinate data trains rapidly, multiple model configurations could be tested quickly.
  * Models hit an accuracy ceiling at approximately **82%**.
* **Root Cause Analysis (Missing Wrist Coordinates & False Hand-Movement Touches):**
  * Investigated the 82% ceiling and realized the model was missing wrist landmark coordinates.
  * Without wrist landmarks, the model could not differentiate between:
    1. A stationary hand resting while a finger performs an intentional tap.
    2. A moving hand translating across the keyboard (where all joints move together), leading to false touch triggers during general hand repositioning.
  * Furthermore, the initial annotator had hardcoded rigid data filtering, preventing post-processing flexibility.
* **Complete Re-Annotation & Extraction Overhaul:**
  * Rewrote the annotation tool from scratch to record all 21 raw MediaPipe landmarks without premature filtering, including the tracking confidence/accuracy score emitted by MediaPipe.
  * Re-annotated all 3,800 seconds of video footage from the ground up.
* **The 13-Step Data Filtering Pipeline:**
  * With raw data preserved, built a flexible 13-step offline data processing pipeline.
  * Systematically pruned noisy, low-confidence MediaPipe windows and separated hand-translation movement from stationary taps.
  * Tuned hyperparameters across architectures, boosting LSTM touch classification accuracy from **82% to over 94%**, while confirming that LSTM delivered the lowest latency among all evaluated sequence architectures.

---

## 8. Phase 7: Application Suite Finalization & Detector Optimizations

* **Two Production Applications Built with PySide6:**
  1. **Application 1 (Layout Designer):** Desktop GUI for custom key placement, AprilTag anchor embedding, digital layout XML export, and printable PDF generation with physical ruler scale calibration.
  2. **Application 2 (Runtime Virtual Keyboard Detector):** Multi-threaded real-time camera tracking, continuous AprilTag homography estimation, MediaPipe hand extraction, 12 FPS temporal windowing, PyTorch LSTM touch inference, and key-press dispatch.
* **Runtime Detector Enhancements:**
  * Shifted filtering logic into the runtime detector, incorporating hand movement filters to suppress false activations during hand travel.
  * Integrated multi-tag redundancy, ensuring accurate key mapping even when hands cover significant portions of the paper layout.
  * Reached a reliable, lightweight, fully CPU-driven paper virtual keyboard system operating on low-cost hardware.
