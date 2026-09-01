# AGENTS.md - University Research Project & Agent Guidelines

## 1. Project Overview & Research Goals

This project focuses on the development and evaluation of a **customizable paper-based virtual keyboard system** powered by computer vision and deep learning.

### System Workflow & Pipeline Architecture

1. **Layout Design & Export**: The user creates a custom key layout using the GUI designer ([designer](file:///home/lahirukasunidilhara/Documents/university/research/designer)). The design is exported as:
   - An **XML file** containing key boundaries, button positions, command/key-press assignments, and fiducial marker anchor locations.
   - A **printable PDF** of the keyboard layout embedded with fiducial markers.
2. **Printing & Physical Setup**: The user prints the paper virtual keyboard containing AprilTag fiducial anchors on any surface.
3. **Camera & Pose Estimation**: A video feed captures the physical surface. [MediaPipe Hand Landmarker](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector) extracts 3D/2D hand joint coordinates.
4. **Temporal Windowing & Feature Processing**:
   - Data is sub-sampled and scale-normalized (unitless hand-length).
   - Applied **1€ Filter** for denoising and joint velocity calculation.
   - Assembled into sliding windows of **5 frames with 2-frame overlap** (stride of 3 frames).
5. **Touch Detection (Custom Model)**: A custom PyTorch Deep Learning model ([best_finger_touch_lstm.pth](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/best_finger_touch_lstm.pth)) evaluates each 5-frame window to classify per-finger touch events.
6. **Homography Mapping & Key Execution**:
   - [AprilTag](file:///home/lahirukasunidilhara/Documents/university/research/aprilTag) markers on the printed paper are tracked to compute a 3x3 **Homography matrix ($H$)**.
   - When a finger touch event is confirmed, the fingertip pixel coordinates are mapped through $H$ into the XML layout coordinate space.
   - The corresponding key press or system command is executed.

---

## 2. Directory & Component Structure

Below is the layout of the project workspace:

| Directory / File                                                                                                                                                                    | Description                                                                                                                                                                                                                                                                    | Status / Notes                      |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------- |
| [`designer/`](file:///home/lahirukasunidilhara/Documents/university/research/designer)                                                                                              | PySide6 desktop GUI tool for designing custom keyboard layouts, key properties, and AprilTag anchors. Exports PDF & XML.                                                                                                                                                       | **Partially Complete**              |
| [`designer/analyzer/main.py`](file:///home/lahirukasunidilhara/Documents/university/research/designer/analyzer/main.py)                                                             | Verification module for homography-based coordinate transformation using simulated finger touches.                                                                                                                                                                             | Active testing                      |
| [`mediapipeDetector/`](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector)                                                                            | Hand tracking pipeline (12 FPS, 1€ filter, velocity derivation, 5-frame 2-overlap windowing) + PyTorch custom LSTM touch classifier.                                                                                                                                           | Model trained & core pipeline built |
| [`aprilTag/`](file:///home/lahirukasunidilhara/Documents/university/research/aprilTag)                                                                                              | Calibration and tracking scripts for AprilTag fiducial markers (chosen marker system).                                                                                                                                                                                         | Active marker system                |
| [`opencvAruco/`](file:///home/lahirukasunidilhara/Documents/university/research/opencvAruco)                                                                                        | Legacy ArUco marker testing scripts used during marker evaluation.                                                                                                                                                                                                             | Testing / Legacy                    |
| [`sources/chapter-breakdown.md`](file:///home/lahirukasunidilhara/Documents/university/research/sources/chapter-breakdown.md)                                                       | Detailed thesis chapter breakdown & section outline to follow when writing thesis chapters.                                                                                                                                                                                    | Active thesis guideline             |
| [`sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf`](file:///home/lahirukasunidilhara/Documents/university/research/sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf) | Previous draft breakdown for Chapters 1, 2, and 3. Used for reference/context only (internal mechanisms updated; **do not cite**).                                                                                                                                             | Contextual draft reference          |
| [`pdf-sources/`](file:///home/lahirukasunidilhara/Documents/university/research/pdf-sources)                                                                                        | Repository storing PDF research papers and literature references.                                                                                                                                                                                                              | Paper storage                       |
| [`research-db/`](file:///home/lahirukasunidilhara/Documents/university/research/research-db)                                                                                        | Literature analysis database containing [`summary-matrix.md`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/summary-matrix.md) and [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib). | Thesis reference hub                |
| [`chapters/`](file:///home/lahirukasunidilhara/Documents/university/research/chapters)                                                                                              | Directory storing LaTeX `.tex` files for individual thesis chapters.                                                                                                                                                                                                           | Thesis writing                      |
| [`main.tex`](file:///home/lahirukasunidilhara/Documents/university/research/main.tex)                                                                                               | Root LaTeX file that compiles all thesis chapters.                                                                                                                                                                                                                             | Thesis entrypoint                   |
| [`figures/`](file:///home/lahirukasunidilhara/Documents/university/research/figures)                                                                                                | Visual assets, diagrams, charts, and figures used in the thesis.                                                                                                                                                                                                               | Thesis figures                      |

---

## 3. Thesis Writing & Literature Research Protocol

When instructed to draft, research, or revise thesis chapters:

1. **Follow Chapter Breakdown Outline**:
   - Always follow the detailed section breakdown and outline defined in [`sources/chapter-breakdown.md`](file:///home/lahirukasunidilhara/Documents/university/research/sources/chapter-breakdown.md) for chapter structure and section numbering.
2. **Contextual Reference for Early Chapters**:
   - When drafting Chapters 1, 2, and 3, refer to [`sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf`](file:///home/lahirukasunidilhara/Documents/university/research/sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf) to inspect previous writing and contextual background.
   - _Note:_ Internal system mechanisms have evolved since that document was written, so prioritize the current pipeline architecture outlined in Section 1 of this document.
   - **Crucial Rule:** Do NOT use or cite `old_breakdown_of_chapter1_chapter2_chapter3.pdf` as a literature citation or bibliographic reference in the thesis.
3. **Strict Citation Protocol**:
   - Citations in LaTeX chapters (`\cite{CitationKey}`) must **ONLY** use BibTeX keys defined in [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib).
   - Consult [`summary-matrix.md`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/summary-matrix.md) first to map paper methodologies and findings to BibTeX keys in [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib).
4. **Targeted PDF Inspection in [`pdf-sources/`](file:///home/lahirukasunidilhara/Documents/university/research/pdf-sources)**:
   - Read specific target PDFs in [`pdf-sources/`](file:///home/lahirukasunidilhara/Documents/university/research/pdf-sources) to extract fine-grained technical details, mathematical formulations, or empirical data needed for chapter text.
5. **Drafting Execution**:
   - Save LaTeX chapter files in [`chapters/`](file:///home/lahirukasunidilhara/Documents/university/research/chapters).
   - Ensure every chapter file is included in [`main.tex`](file:///home/lahirukasunidilhara/Documents/university/research/main.tex).
   - Save figures and diagrams into [`figures/`](file:///home/lahirukasunidilhara/Documents/university/research/figures).

---

## 4. Current State & Next Steps for Real-Time Application Integration

- **Designer Status**: Layout designer, XML export, PDF rendering, and homography simulation (`designer/analyzer/main.py`) are implemented.
- **Detector Status**: Hand pose pipeline, temporal windowing (5 frames, 2 overlap), and PyTorch LSTM finger touch classification model (`best_finger_touch_lstm.pth`) are operational.
- **Next Integration Milestone**: Combine live camera feed + AprilTag tracking ($H$) + MediaPipe Hand Landmarker + PyTorch LSTM touch classifier + XML layout lookup into a unified real-time application.
