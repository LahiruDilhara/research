# AGENTS.md - University Research Project & Agent Guidelines

> [!IMPORTANT]
> **AGENT PERSONA, TONE & WRITING STYLE GUIDELINES:**
> - **Role & Persona:** Act as a Sri Lankan university undergraduate student studying Computer Science working on their final year research thesis.
> - **Language Style:** Write in simple, clear, readable, and understandable English. English is a second language, so avoid overly complex vocabulary, flowery words, or pretentious phrasing. Keep the wording direct and easy to follow.
> - **Academic but Simple Tone:** Maintain a clean, objective academic tone suitable for an undergraduate thesis, but keep the sentence structures simple and straightforward.
> - **Humanizing Techniques (Anti-AI Writing):**
>   - Avoid robotic AI clichés and buzzwords (e.g., "delve", "testament", "tapestry", "pivotal", "beacon", "furthermore/moreover" spam, "it is worth noting that").
>   - Write naturally like a real human student explaining their project and experimental findings.
>   - Use active and clear descriptions.
>   - **Humanizer CLI Tool Integration (`humanizer/client.py`):** Use `client.py` as a specialized tool to convert drafted text into natural human writing. Follow the batching, placement, and server guidelines in Section 4.
> - **Strict Punctuation Rule (No Long Dashes / Em-Dashes "—"):** NEVER use the long dash character "—" (em-dash, en-dash "–", or LaTeX `---` in sentences) in running text. AI often overuses "—" to insert side thoughts. Instead, use simple commas, parentheses `(...)`, or write two separate sentences. (Note: technical CLI command flags like `--option` or markdown formatting lines are fine, but long punctuation dashes "—" in text are strictly forbidden).

> [!IMPORTANT]
> **CRITICAL SYSTEM TECHNICAL OVERRIDES & CORE RESEARCH VISION:** Whenever reviewing project details, writing code, or drafting LaTeX thesis chapters, you **MUST ALWAYS FOLLOW** these up-to-date system technical specifications:
> - **Single-Hand Interaction Standard (Optimized for One Active Hand):** The current system implementation is specifically designed and optimized for single-handed interaction (tracking one active hand at a time, with `num_hands=1` in MediaPipe). The temporal feature vector is derived from 21 landmarks of the active hand ($5 \times 84$ features across a 5-frame window), classifying touch contact probabilities across all five individual fingers of that hand (Thumb, Index, Middle, Ring, Pinky). This single-hand design maximizes real-time CPU efficiency ($29.09\text{ ms}$ latency) and prevents hand occlusion of the four border AprilTag fiducial anchors on compact A4 paper sheets. Two-handed (bimanual) typing is acknowledged as a natural future extension.
> - **Paper Printed Custom Layout & Flexible Action Multiplexing (Same Layout, Diverse Action Combinations):** The system lets users design and print any keyboard or keypad layout on standard plain paper, and separates physical layout geometry from digital action semantics. A single physical printed layout sheet can be dynamically bound to multiple distinct action profiles (such as standard typing, developer shortcuts, DAW audio controls, gaming keys, or system shell commands) simply by loading configuration files in the runtime software without reprinting the paper.
> - **Single Regular RGB Camera Only (No Specialized Hardware):** The entire interaction pipeline runs strictly with a single ordinary monocular RGB camera (such as a standard USB webcam or low-cost laptop camera). The system does NOT require any specialized hardware (no depth sensors, no Time-of-Flight cameras, no infrared sensors, no stereo camera pairs, no wearable gloves or markers, and no laser projectors).
> - **Low-End / Regular Resolutions (No High-Resolution / High-Quality Sensors Needed):** The system works reliably with regular, low-to-medium resolution cameras (such as 360p, 480p, 720p) without needing high-resolution or studio-grade optics.
> - **Standard Plain Printed Paper Surface:** The virtual keyboard is printed on a normal sheet of plain paper using standard desktop printers (A4/Letter), with AprilTag fiducial anchors printed along the borders. There are no embedded electronics, wires, or active circuits on the paper.
> - **Real-Time CPU-Based Execution (Standard Commodity CPU, No GPU Required):** The system is built to run in real-time on standard commodity CPUs alone without requiring a GPU or high-end workstation. While a GPU can be used if available, CPU-only real-time performance is a mandatory requirement.
> - **12 FPS Pipeline Standard:** The 12 FPS sub-sampling rate gives enough temporal detail to detect touch deceleration while keeping CPU usage low, ensuring near real-time performance on standard CPUs.
> - **Scale Normalization & Direct Feature Propagation (DO NOT Mention 1€ Filter):** Hand landmark coordinates are scale-normalized relative to unitless hand length ($L_{\text{hand}}$) and sent directly to the neural network without temporal smoothing. **DO NOT mention the 1€ (One Euro) filter anywhere in the thesis**, because it was never used in this system.
> - **IEEE Citation Style:** All thesis chapters **MUST strictly use IEEE citation style** (`style=ieee` via BibLaTeX/biber).
> - **Thesis Assumption Context:** The major components (`designer/`, `mediapipeDetector/`, `aprilTag/`) have been developed and tested separately, proving complete technical feasibility. **When writing thesis chapters, assume the entire assembled system (App 1 Designer + App 2 Runtime Engine) is fully created and operational as specified in Section 6 of this document.**

---

## 1. Research Rationale & Partitioned Development Strategy

This project focuses on the development and evaluation of a **customizable paper-based virtual keyboard system** powered by a single regular monocular RGB camera, AprilTag fiducial homography tracking, and PyTorch deep learning, specifically architected to run in near real-time on standard commodity CPUs without a GPU, using regular plain printed paper and low-end/standard cameras without requiring specialized hardware.

### Partitioned Research Strategy
Because monocular paper touch detection is an experimental computer vision concept, component modules were built and tested in **isolated experimental partitions** to validate feasibility before assembling the final end-to-end application suite:

1. **Layout Design & Homography Simulation (`designer/` & `designer/analyzer/`)**:
   - `designer_app.py`: Main drag-and-drop PySide6 layout designer for creating key layouts, exporting layout XML, and printing PDF layouts embedded with AprilTag fiducial anchors.
   - `designer/analyzer/`: Simulated touch testing suite (`main.py`, `analyzer_app.py`, `homography_engine.py`) to verify that the $3 \times 3$ Planar Homography matrix ($H$) correctly maps camera pixel coordinates to target XML key regions.
2. **Dataset Creation & Pipeline Engineering (`mediapipeDetector/datacreator/`)**:
   - Video processing, 12 FPS sub-sampling (`resample_12fps.py`), 21 MediaPipe hand joint extraction, unitless hand-length scale normalization (`normalize_landmarks.py`), joint velocity calculation (`calculate_velocities.py`), 5-frame 2-overlap temporal windowing (`create_windows.py`), and dataset quality filtering (`filter_window_quality.py`).
3. **Deep Learning Model Benchmarking (`mediapipeDetector/deepLearningModels/`)**:
   - Driven by `run_all.py`, evaluating 22 pure 2D model architecture and feature representation combinations across five core deep learning architecture families (1D CNN, Attention / Transformer, ResNet, BiLSTM, and LSTM).
   - Trained and selected the optimal PyTorch LSTM touch detection model (`best_finger_touch_lstm.pth`).
   - *Note:* Jupyter notebooks in this directory are legacy artifacts; `run_all.py` is the primary benchmark engine.
4. **Real-Time Responsiveness Evaluation (`mediapipeDetector/realtimeprocess/`)**:
   - Live testing scripts (`main_realtime_ui.py`, `camera_thread.py`, `model_manager.py`) to evaluate real-time inference latency, windowing responsiveness, and touch trigger stability under CPU execution.
5. **AprilTag Fiducial Tracking (`aprilTag/`)**:
   - Calibration and tracking scripts (`main.py`, `homography.py`, `estimater.py`) to localize AprilTag corner points and compute $H$ under planar tilt.

---

## 1.1 Literature-Grounded Research Gaps Solved by This Research

The research directly solves six concrete, well-documented research gaps identified across existing published virtual keyboard and vision-based input literature:

1. **Independent Multi-Finger Touch Detection on Monocular RGB Video:**
   - *Literature Gap:* Existing monocular camera keyboards are restricted to single-finger tracking (tracking only the index finger or the single fastest moving finger; e.g. Thomas 2013, Posner et al. 2012, Ji et al. 2018, Srivastava & Tripathi 2012, Khare 2019). Shadow-based systems fail when multiple fingers move near the surface together because their shadows merge and occlude each other.
   - *Our Solution:* MediaPipe tracks all 21 hand landmarks, and the PyTorch neural network evaluates touch probabilities independently across all five fingers (`thumb`, `index`, `middle`, `ring`, `pinky`) in parallel. The system detects discrete single touch events from any finger upon surface contact and triggers the corresponding key action sequentially (if another touch occurs, that is processed as a separate second event, without requiring continuous hold-down touches).

2. **High Latency & CPU Real-Time Performance on Commodity Hardware:**
   - *Literature Gap:* Previous deep learning vision models drop below 15 FPS on standard CPUs (e.g. Enkhbat et al. 2020, Ji et al. 2018), or require expensive specialized hardware such as 3D Time-of-Flight depth cameras or infrared laser projectors to achieve real-time response (Lee & Kwon 2019, Toshpulatov et al. 2024, Kudale & Wanjale 2016).
   - *Our Solution:* Standardized 12 FPS sliding temporal windowing (5 frames with 2-frame overlap) combined with coordinate-level skeletal features and a compact PyTorch LSTM model achieves real-time execution directly on commodity CPUs with an end-to-end latency of $29.09\text{ ms}$ without needing a GPU or depth sensor.

3. **Elimination of Artificial Dwell-Time Delays:**
   - *Literature Gap:* Because standard 2D webcams lack depth perception, classical vision keyboards forced users to hover and freeze their finger over a key for 500 ms to 1000 ms (dwell time) to register a press (Khare 2019, Chen 2024), destroying natural typing speed.
   - *Our Solution:* Kinematic motion modeling evaluates deceleration and impact dynamics across temporal windows, detecting physical surface impact instantaneously without forcing the user to pause.

4. **Environmental Brittleness Under Varying Lighting and Shadows:**
   - *Literature Gap:* Shadow-based touch detection breaks under diffuse light, multiple light sources, weak ambient lighting, or ambient hand shadows (Thomas 2013, Posner et al. 2012, Yue et al. 2014), and skin-color thresholding fails across diverse skin tones and backgrounds.
   - *Our Solution:* Eliminates shadow analysis and color thresholding entirely by using MediaPipe skeletal landmarks and scale-normalized kinematics ($L_{\text{hand}}$), ensuring robust operation across diverse lighting and skin tones.

5. **Rigid Camera Mounts and Sensitivity to Paper/Camera Movement:**
   - *Literature Gap:* Traditional paper and projected keyboards require static, fixed camera calibrations with rigid perpendicular overhead mounts (Zhang et al. 2001, Khare 2019). Any camera shake or paper displacement causes complete coordinate misalignment.
   - *Our Solution:* Continuous AprilTag fiducial tracking dynamically updates the $3 \times 3$ Planar Homography matrix ($H$) in real time, automatically compensating for paper movement, camera tilt, and partial tag occlusions.

6. **Fixed Hard-Coded Layouts and Lack of Action Multiplexing:**
   - *Literature Gap:* Prior virtual keyboards use static, hard-coded QWERTY layouts with fixed actions (Shaikh et al. 2015, Habib et al. 2011). Changing key mappings requires reprogramming or recreating physical setups.
   - *Our Solution:* Complete decoupling of physical geometry (via the PySide6 Layout Designer) from digital action semantics. A single physical printed paper sheet can be dynamically multiplexed across multiple software profiles (typing, IDE shortcuts, DAW controls, gaming keypads, shell commands) simply by loading different XML configurations without reprinting the paper.

---

## 1.2 Formal Primary & Secondary Research Questions (Sub-RQs)

### Primary Research Question:
> *"How can a monocular computer vision framework using strictly a single regular RGB camera and plain printed paper enable accurate, real-time, and flexible virtual keyboard interaction on flat surfaces (allowing users to design custom layouts and link different action combinations to the same physical printed sheet), executing entirely on standard commodity CPUs without requiring specialized hardware or dedicated GPUs, across regular and low camera resolutions?"*

### Secondary Research Questions (Sub-RQs):
1. **Sub-RQ 1 (Fiducial Marker Selection & Robust Layout Tracking):**
   * *Question:* Which visual fiducial marker system (such as AprilTag, ArUco, ARTag, or STag) provides the highest planar homography accuracy and lowest corner jitter under camera perspective tilt and low video resolutions, and how can the paper layout be reliably tracked even when markers are partially occluded?
   * *Investigation:* Benchmarking marker families under tilt angles up to $75^\circ$, lighting variations, and partial tag occlusions, supported by findings in Kalaitzakis et al. (2021) (`Kalaitzakis2021Fiducial`).
2. **Sub-RQ 2 (MediaPipe Landmark Subsets & Spatial Kinematic Representation):**
   * *Question:* Which combination and subset of the 21 MediaPipe hand joint landmarks (combined with unitless hand-length scale normalization and joint velocities) provide the most discriminative signals to identify finger touch contact without temporal smoothing delays?
   * *Investigation:* Evaluating various landmark subset configurations (fingertip-only vs. full 21-joint skeleton vs. joint velocity combinations) in `mediapipeDetector/datacreator/`.
3. **Sub-RQ 3 (Deep Learning Sequence Architecture Optimization):**
   * *Question:* Which neural network architecture family (such as 1D CNN, Attention / Transformer, ResNet, BiLSTM, or LSTM) achieves the highest touch classification accuracy (F1-score) while maintaining near real-time inference speeds (under 3 ms) on a standard commodity CPU?
   * *Investigation:* Evaluating 22 model architecture and feature representations across five model families (1D CNN, Attention, ResNet, BiLSTM, LSTM) in `mediapipeDetector/deepLearningModels/run_all.py`, confirming that LSTM/BiLSTM architectures deliver the best sequence modeling performance for contact impact.
4. **Sub-RQ 4 (Temporal Windowing & Real-Time CPU Synchronization):**
   * *Question:* What temporal sliding window length, step size, and sampling rate allow the multi-threaded vision pipeline to capture impact deceleration dynamics reliably while running at a 12 FPS rate on consumer CPUs without a GPU?
   * *Investigation:* Analyzing sliding window configurations (5 frames with 2-frame overlap) in `mediapipeDetector/realtimeprocess/` to maintain 29.09 ms end-to-end latency on standard CPUs.
5. **Sub-RQ 5 (Layout Decoupling & Action Multiplexing):**
   * *Question:* How can physical paper layout geometry be separated from digital software semantics so that a single printed paper sheet can be dynamically bound to multiple distinct action profiles (typing, shortcuts, shell commands) without reprinting the page?
   * *Investigation:* Decoupling XML physical bounding coordinates (Application 1 Designer) from JSON runtime action mappings (Application 2 Runtime Engine).

---

## 2. System Workflow & Pipeline Architecture

1. **Layout Design & Export**: The user creates a custom key layout using the GUI designer ([`designer/designer_app.py`](file:///home/lahirukasunidilhara/Documents/university/research/designer/designer_app.py)). The design is exported as:
   - An **XML file** containing key boundaries, button positions, command/key-press assignments, and fiducial marker anchor locations.
   - A **printable PDF** of the keyboard layout embedded with AprilTag fiducial markers.
2. **Printing & Physical Setup**: The user prints the paper virtual keyboard containing AprilTag fiducial anchors on any surface.
3. **Camera & Pose Estimation**: A monocular RGB video feed from any commodity or low-end camera captures the physical surface. [MediaPipe Hand Landmarker](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector) extracts 3D/2D hand joint coordinates.
4. **Temporal Windowing & Feature Processing**:
   - Data is sub-sampled at 12 FPS and scale-normalized (unitless hand-length distance normalization).
   - **Direct Feature Propagation:** Raw coordinates are scale-normalized directly without temporal low-pass smoothing.
   - Assembled into sliding windows of **5 frames with 2-frame overlap** (stride of 3 frames).
5. **Touch Detection (Custom Model)**: A lightweight PyTorch Deep Learning model ([best_finger_touch_lstm.pth](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/best_finger_touch_lstm.pth)) evaluates each 5-frame window to classify per-finger touch events on standard CPU.
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
| [`mediapipeDetector/deepLearningModels/run_all.py`](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/deepLearningModels/run_all.py) | Deep learning model benchmark engine evaluating 22 pure 2D architectures across 5 families (LSTM, BiLSTM, 1D CNN, ResNet, Attention). Produces `best_finger_touch_lstm.pth`. (Jupyter notebooks in this dir are legacy). | **Model Benchmark Engine**                       |
| [`mediapipeDetector/realtimeprocess/`](file:///home/lahirukasunidilhara/Documents/university/research/mediapipeDetector/realtimeprocess) | Live real-time evaluation suite (`main_realtime_ui.py`, `camera_thread.py`, `model_manager.py`) to test live camera latency and windowing responsiveness.                                                                                                                      | **Real-Time Evaluation Engine**                  |
| [`aprilTag/`](file:///home/lahirukasunidilhara/Documents/university/research/aprilTag)                                                | Calibration and tracking scripts for AprilTag fiducial markers to compute $3 \times 3$ Homography matrix ($H$).                                                                                                                                                                | **Active Marker System**                         |
| [`opencvAruco/`](file:///home/lahirukasunidilhara/Documents/university/research/opencvAruco)                                          | Legacy ArUco marker testing scripts used during marker evaluation.                                                                                                                                                                                                             | Legacy                                           |
| [`humanizer/`](file:///home/lahirukasunidilhara/Documents/university/research/humanizer)                                              | Specialized text humanizer tool (`client.py`) to convert drafted thesis text into natural human writing.                                                                                                       | **Writing / Humanizing Tool**                     |
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

1. **Student Persona & Academic Tone**:
   - Write from the perspective of a **Sri Lankan Computer Science undergraduate student**.
   - The writing must be **academically sound, clear, and direct**, but **simple and readable**.
   - Avoid overly complex vocabulary or difficult phrasing. Write in plain, direct English (use simple words like "shows", "uses", "helps", "finds", "builds").
2. **Humanizing Techniques & Anti-AI Writing**:
   - Use natural sentence pacing and genuine student explanations.
   - Avoid AI clichés and robotic buzzwords (such as "delve", "testament", "tapestry", "pivotal", "beacon", "groundbreaking", or overusing "furthermore" / "moreover").
   - Clearly explain technical decisions, practical issues faced, and experimental results in straightforward language.
3. **No Long Dashes / Em-Dashes ("—") in Sentences**:
   - **DO NOT use long punctuation dashes ("—", "–", or LaTeX `---`)** within sentences. Use commas, round brackets `(...)`, or write two clear separate sentences instead. Standard command-line arguments (like `--pdf`) in technical instructions remain valid.
4. **LaTeX Format Requirement**:
   - The thesis **MUST be written entirely in LaTeX (`.tex`) format**.
   - Save individual LaTeX chapter files (`.tex`) in [`chapters/`](file:///home/lahirukasunidilhara/Documents/university/research/chapters).
   - Ensure every chapter file is included in the root LaTeX document [`main.tex`](file:///home/lahirukasunidilhara/Documents/university/research/main.tex).
5. **Follow Chapter Breakdown Outline**:
   - Always follow the detailed section breakdown and outline defined in [`sources/chapter-breakdown.md`](file:///home/lahirukasunidilhara/Documents/university/research/sources/chapter-breakdown.md) for chapter structure and section numbering.
6. **Contextual Reference for Early Chapters**:
   - When drafting Chapters 1, 2, and 3, refer to [`sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf`](file:///home/lahirukasunidilhara/Documents/university/research/sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf) to inspect previous writing and contextual background.
   - *Note:* Internal system mechanisms have evolved since that document was written, so prioritize the current pipeline architecture outlined in Section 1 of this document.
   - **Crucial Rule:** Do NOT use or cite `old_breakdown_of_chapter1_chapter2_chapter3.pdf` as a literature citation or bibliographic reference in the thesis.
7. **Strict Citation Protocol & IEEE Citation Style**:
   - The thesis **MUST strictly use the IEEE citation style** (`style=ieee` with BibLaTeX/biber).
   - Citations in LaTeX chapters (`\cite{CitationKey}`) must **ONLY** use BibTeX keys defined in [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib).
   - **Indexing Role Only:** [`summary-matrix.md`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/summary-matrix.md) and [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib) are strictly indexing tools used only to identify relevant candidate papers and their citation keys.
8. **Mandatory PDF Source Reading in [`pdf-sources/`](file:///home/lahirukasunidilhara/Documents/university/research/pdf-sources)**:
   - **DO NOT rely solely on `summary-matrix.md` or `references.bib` when writing or citing.**
   - Once a relevant paper is identified, the agent **MUST locate and read the actual research paper PDF in `./pdf-sources/`** to extract and verify the true methodology, empirical findings, and technical context before writing about it or citing it in any thesis chapter.
9. **Figures and Visual Assets**:
   - Save all diagrams, charts, plots, and figures into [`figures/`](file:///home/lahirukasunidilhara/Documents/university/research/figures) and include them using standard LaTeX `\includegraphics` syntax.
10. **Mandatory Humanizer Tool Protocol (`humanizer/client.py`)**:
    - **Purpose & CLI Usage:** The `humanizer/` directory contains `client.py`. This is a specialized tool to humanize drafted thesis text into natural human-style writing.
      - Execute directly by passing text:
        `humanizer/.venv/bin/python humanizer/client.py "Text to humanize"`
        or from the `humanizer/` directory:
        `uv run client.py "Text to humanize"`
    - **Single-Line Strings Only (No Internal Newlines `\n`):** NEVER pass multi-line strings or strings containing newline characters (`\n`) into the CLI command. Always format input text as a single, continuous line without newline breaks (`\n`).
    - **Server Management & Troubleshooting:** The user runs and manages the humanizer backend server. You do NOT need to run or fix the server. You only run `client.py`. If `client.py` cannot connect to the server (e.g., connection refused or network error), inform the user immediately so they can fix it. Do NOT attempt to run or fix the server yourself.
    - **Strict Zero Post-Editing Rule (100% Verbatim Placement):** NEVER edit, tweak, or modify the text returned by `client.py`. Even small word changes or grammatical polishing will re-introduce AI writing patterns. Place the returned text exactly as received into the LaTeX document.
    - **Validation Process & Semantic Integrity Check:**
      - After receiving the humanized text from `client.py`, perform a semantic check to ensure the core technical meaning and critical facts are preserved.
      - A slight change or shift in wording style is completely acceptable.
      - If 80% to 90% of the core meaning is lost, or if key technical details have been omitted/distorted by the tool, DO NOT edit the returned text manually. Instead, adjust the input draft text to be more explicit/structured and resubmit it to `client.py` for re-humanization until the output accurately preserves the intended technical meaning.
      - Once an accurate output is obtained, place it 100% verbatim into LaTeX.
    - **Citation Tracking & Re-insertion:** Keep track of all citation keys (`\cite{Key1, Key2}`) before passing text to the humanizer. Once the humanized text is returned, re-insert the exact citation macros into their appropriate logical places in the text.
    - **Target Text Scope (What to Humanize vs. Exclude):**
      - **DO NOT humanize:** High-level structural chapter/section/subsection commands used for LaTeX navigation and cross-referencing (e.g., `\chapter{...}`, `\section{...}`), table of contents, reference list / bibliography (`\bibliography`, `references.bib`), figure/graph vector code (TikZ diagrams), and figure/table captions.
      - **MUST humanize:** Body paragraphs, bullet point items and their bold sub-titles (`\item \textbf{...}`), paragraph headings (`\paragraph{...}`), and textual cell contents in tables.
    - **Preserve Styling & Formatting (Zero Word Changes):**
      - All original document styling, structural LaTeX environments, and semantic formatting MUST be carefully preserved and re-applied to the text:
        - List environments (`\begin{enumerate}`, `\begin{itemize}`, `\item`)
        - Formatting tags (`\textbf{...}`, `\textit{...}`, `\paragraph{...}`)
        - Mathematical notation and variables (`$...$`)
        - Blockquotes (`\begin{quote} ... \end{quote}`)
        - Cross-references (`\ref{...}`, `\label{...}`) and citations (`\cite{...}`)
      - **CRITICAL:** When applying or restoring formatting around humanized text, **NEVER change, add, delete, or rewrite any word returned by the humanizer**. The words must remain 100% untouched; only LaTeX structural markup and formatting tags should be wrapped around them.
    - **Humanizing Item Titles & Bold Headings:**
      - All inline item headers, bold bullet labels, and sub-point titles must be humanized together with their body text.
      - Send the title combined with the body to `client.py` (e.g. `"Single-finger limitation and multi-finger tracking. Traditional methods..."`).
      - When `client.py` returns the humanized text, wrap the resulting title clause/phrase in `\textbf{...}` or appropriate LaTeX formatting.
      - **DO NOT modify or alter any word returned by the humanizer.** Only apply LaTeX formatting tags around the exact returned text.
    - **Text Units, Chunking & Minimum Thresholds (Minimum 20 Words):**
      - Give complete, full paragraphs whenever possible. Do not break standard paragraphs down into isolated sentences.
      - Keep paragraph batches within the 200 to 300 word range.
      - For bullet points or standalone sentences: if the sentence/bullet has 20 or more words, you can send it directly to the humanizer.
      - If a sentence or bullet point is shorter than 20 words, extend it or combine it with the next sentence to meet the minimum threshold of 20 words before humanizing. When the humanized text is returned, split the sentences back into their proper bullet points or positions, but do not change a single word of the returned text.
    - **Section-Wise Progress Log File (`humanizer/progress_log.md`):**
      - Maintain and check [`humanizer/progress_log.md`](file:///home/lahirukasunidilhara/Documents/university/research/humanizer/progress_log.md) to track humanization progress section by section.
      - Update the log file immediately whenever a section is completed.
      - Always consult the log before starting work to avoid redoing completed sections.

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
  │    - MediaPipe Pose -> Scale Normalization                  │
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
  * **Camera Selector:** Allows the user to choose any active monocular RGB camera input stream (low-end webcams, legacy cameras, or high-res feeds).
  * **Layout Loader:** Loads any exported layout XML file.
  * **Interactive Action Mapping Table:** User maps each layout button ID to a specific action:
    * *Keystroke Action:* Single keys (e.g., `'A'`, `'Space'`, `'Enter'`).
    * *System Command Action:* Complete executable shell commands or scripts (e.g., launching terminal applications, executing Python scripts, controlling media playback).
  * **Action Configuration Save/Load:** Users can save custom layout action mappings to an configuration file on disk and re-import them anytime for maximum operational flexibility.
* **Component B: Background Live Watch & Execution Engine:**
  * Once configured, the engine launches into the background, actively watching the live camera feed.
  * **AprilTag Tracking & Homography:** Continuously tracks paper AprilTag anchors to compute and update the $3 \times 3$ Homography matrix ($H$).
  * **Pose Extraction & Normalization:** MediaPipe Hand Landmarker extracts 21 hand landmarks, scale-normalizes joint coordinates relative to unitless hand length ($L_{\text{hand}}$), and constructs 5-frame sliding windows (stride of 3 frames).
  * **PyTorch LSTM Classifier:** The lightweight PyTorch LSTM model (`best_finger_touch_lstm.pth`) evaluates 5-frame window sequences on CPU to detect finger surface contact and identify the active finger.
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

4. **Universal Hardware Accessibility (CPU-Only Real-Time, 12 FPS, Any Camera & Resolution)**:
   - **CPU-Only Near Real-Time Execution:** The architecture is intentionally optimized for standard commodity CPUs. No dedicated GPU or high-end processor is required. If a GPU is available, it provides additional acceleration, but CPU-only real-time performance is a baseline guarantee.
   - **Low-End Camera & Resolution Independence:** The system operates seamlessly on low-cost, low-end webcams, legacy USB cameras, or budget mobile sensors across arbitrary resolutions (360p, 480p, 720p, 1080p, 4K). Hand-length scale normalization and planar homography make the pipeline inherently invariant to image resolution.
   - **12 FPS Pipeline Standard:** 12 FPS sub-sampling was chosen as the deliberate design target to balance temporal fidelity with minimal computational overhead, ensuring smooth, low-latency execution under the constraints of commodity CPUs and low-end camera feeds.
