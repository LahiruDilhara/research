# IMPORTANT.MD - Critical System Technical Specifications & Overrides

> [!IMPORTANT]
> **MANDATORY FOR ALL AI AGENTS:** Always inspect this file before generating code, architectural documentation, or drafting thesis chapters. The technical specifications listed here override any historical descriptions or older documentation drafts.

---

## 1. Feature Processing & Normalization Overrides

* **NO 1€ FILTER / NO TEMPORAL SMOOTHING:**
  - The system **DOES NOT use 1€ filtering**, low-pass filtering, or temporal moving average smoothing on MediaPipe hand landmark coordinates.
  - 1€ filtering was evaluated in early experiments but has been suppressed in the active production pipeline.
  - Raw hand landmark coordinates from MediaPipe Hand Landmarker are processed directly without temporal trajectory smoothing.

* **SCALE NORMALIZATION:**
  - Hand joint coordinates are scale-normalized relative to unitless hand size (wrist-to-middle-MCP distance) to achieve camera-distance and hand-size invariance across video frames.
  - Feature vectors are normalized before being assembled into temporal sliding windows.

* **TEMPORAL WINDOWING & LSTM MODEL:**
  - Sliding temporal feature windows consist of **5 frames with 2-frame overlap** (stride of 3 frames).
  - The classifier model is a custom PyTorch LSTM network (`best_finger_touch_lstm.pth`) that evaluates each 5-frame feature window to classify per-finger touch events.

* **HOMOGRAPHY MAPPING:**
  - AprilTag fiducial anchors on the printed paper layout compute a $3 \times 3$ Homography matrix ($H$).
  - Confirmed fingertip touch pixel coordinates are transformed through $H$ directly into the XML layout coordinate space.

* **IEEE CITATION STYLE REQUIREMENT:**
  - All literature citations and references in LaTeX thesis chapters **MUST strictly follow the IEEE citation style**.
  - Use numerical citations (`\cite{CitationKey}`) configured via BibLaTeX (`style=ieee`).

