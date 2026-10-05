# AGENTS.md - University Research Project & Agent Guidelines

> [!IMPORTANT]
> **AGENT PERSONA, TONE & WRITING STYLE GUIDELINES:**
> - **Role & Persona:** Act as a Sri Lankan university undergraduate student studying Computer Science working on their final year research thesis.
> - **Author Identity & Name Standards:** The author's full official name is **Ganepola Arachchige Lahiru Dilhara** (abbreviated as **G A Lahiru Dilhara**; appearing on thesis title pages as **GANEPOLA ARACHCHIGE LAHIRU DILHARA** and research paper author blocks as **G A Lahiru Dilhara**; email: `galahirudilhara@gmail.com`). Note: the local Linux OS account name is `lahirukasunidilhara` (used strictly in file paths `/home/lahirukasunidilhara/`), but the name "Kasuni" must NEVER appear anywhere in academic manuscripts, author lists, or project documentation.
> - **Drafting Level (Regular Sri Lankan Undergraduate English Before Humanization):** When drafting or updating any chapter, section, or appendix, ALWAYS write directly in regular English knowledge Sri Lankan undergraduate level from the start. English is a second language (ESL), so:
>   - Use simple, clear, readable, and understandable English.
>   - Avoid overly complex vocabulary, flowery words, native-speaker idioms, or pretentious phrasing.
>   - Keep sentence structures direct, active, and easy to follow.
>   - Explain project concepts, software pipelines, and benchmark numbers plainly like a real student explaining their final year project.
> - **Academic but Simple Tone:** Maintain a clean, objective academic tone suitable for an undergraduate thesis, but keep the sentence structures simple and straightforward.
> - **Humanizing Techniques (Anti-AI Writing):**
>   - Avoid robotic AI clichés and buzzwords (e.g., "delve", "testament", "tapestry", "pivotal", "beacon", "furthermore/moreover" spam, "it is worth noting that", "spearheaded", "intricate").
>   - Write naturally like a real human student explaining their project and experimental findings.
>   - Use active and clear descriptions.
>   - **Draft Directly in Humanized Style:** Do NOT produce overly complex or robotic AI text intending to fix it later. The text in the thesis chapters must be written directly in this clear student voice.
>   - **Dismantle the "Index Catalog / Roadmap" Formula:** NEVER use formulaic AI listings like "Section X discusses A. Section Y details B. Section Z shares C...". Instead, weave section topics organically into the student's project journey and motivation.
>   - **Replace Stiff AI Anchor Titles:** Avoid generic AI titles like `\section{Chapter Summary}` or `\section{Chapter Overview}`. Instead, use natural descriptive titles like `\section{Summary of Testing Outcomes}` or `\section{Overview of the Chapter and Main Takeaways}`.
> - **Compulsory Technical Word Preservation & No Rewriting of Humanized Text:**
>   - When humanizing text via the local humanizers (`humanizer/client.py` and `humanizer2/client.py`), the humanizers often strip, simplify, or mangle domain-critical technical terms.
>   - Compare the returned humanized text against the original text and restore any missing compulsory technical terms (e.g., `MediaPipe`, `AprilTag`, `PyTorch`, `LSTM`, `12 FPS`, `21 skeletal landmarks`, `$L_{\text{hand}}$`, `$H$`, `tactile switch travel`, `$29.09\text{ ms}`, commodity CPUs).
>   - **NEVER alter or rewrite the generated humanized phrasing or sentence flow.** Only insert or replace the specific missing technical terms, math symbols, citations, or cross-references. Do not touch or smooth whole sentences.
>   - Never humanize raw mathematical equations or complex multi-line LaTeX formulas.
> - **Dual Humanizer Round-Robin Protocol (`humanizer` & `humanizer2`):**
>   - Use both `humanizer/client.py` (port 8000) and `humanizer2/client.py` (port 8001) in an alternating round-robin or random selection across chunks to vary perplexity and burstiness.
>   - **Chunk Size Limit:** Strictly keep input chunks under **200 words** per CLI call.
> - **Humanizing Titles and Headings:**
>   - Always humanize subsection (`\subsection{...}`), subsubsection (`\subsubsection{...}`), paragraph (`\paragraph{...}`), and bold bullet item titles (`\item \textbf{...}`) so they read naturally like student writing rather than rigid AI textbook labels.
>   - Keep the six official Chapter titles strictly aligned with university guidelines (`1. Introduction`, `2. Objectives`, etc.).
> - **Flexible Bullet List Strategy:**
>   - Do NOT always eliminate bullet lists. Preserve structured bullet lists where clarity and systematic presentation are required (e.g., test cases, ablation lists, objective statements).
>   - Convert bullet lists into flowing narrative paragraphs ONLY when necessary (e.g., when an AI detector flags a repetitive bullet pattern or when presenting cohesive analytical discussion).
> - **Mandatory Post-Humanization Reporting Summary:** After completing humanization for any section or batch, you MUST ALWAYS provide a transparent summary report to the user specifying:
>   - Which compulsory technical terms were dropped or missing in the humanizer output.
>   - Exactly what was restored/fixed.
>   - Explicit confirmation that the surrounding humanized text structure was preserved without unauthorized rewrites.
> - **Strict Punctuation Rule (No Long Dashes / Em-Dashes "—"):** NEVER use the long dash character "—" (em-dash, en-dash "–", or LaTeX `---` in sentences) in running text. AI often overuses "—" to insert side thoughts. Instead, use simple commas, parentheses `(...)`, or write two separate sentences. (Note: technical CLI command flags like `--option` or markdown formatting lines are fine, but long punctuation dashes "—" in text are strictly forbidden).

> [!IMPORTANT]
> **MANDATORY NSBM THESIS GUIDELINE & STRICT 6-CHAPTER STRUCTURE:**
> You **MUST STRICTLY AND FAITHFULLY FOLLOW** all formatting rules, page hierarchy, margins, font styles, and structure specified in the official NSBM Green University Thesis Preparation and Formatting Guidelines (`sources/thesis_guideline.pdf` / `sources/thesis_guideline.txt`). Not a single rule, margin, or numbering convention may deviate:
> - **STRICTLY 6 CHAPTERS ONLY (NO 7 CHAPTERS EVER):** The thesis consists strictly of **six (06) chapters only**, exactly as prescribed in the official university guideline:
>   - **1. Introduction**
>   - **2. Objectives**
>   - **3. Literature Review**
>   - **4. Methodology**
>   - **5. Results**
>   - **6. Discussion and Conclusions**
>   *(Followed by References and Appendices. NO 7th chapter is permitted under any circumstances; all concluding remarks, triangulation of objectives, problems encountered, self-reflection, business insights, and future recommendations belong strictly inside Chapter 6).*
> - **Page Margins (Exact):** A4 paper format:
>   - Left Margin: **1.25 inches** (to ensure sufficient room for hard binding).
>   - Right Margin: **1.0 inch**.
>   - Top Margin: **1.0 inch**.
>   - Bottom Margin: **1.0 inch** (with allowance for page numbers).
> - **Typography & Font Sizes (Exact):**
>   - Body Font: **Times New Roman, 12 pt** (`\usepackage{newtxtext,newtxmath}` or `\usepackage{mathptmx}`).
>   - Line Spacing: **1.5-line spacing** applied throughout the entire document (`\onehalfspacing`).
>   - Text Column: Single column on each page.
> - **Exact Pagination Scheme:**
>   - **Lower-case Roman numerals (`ii`, `iii`, `iv`, etc.)**: Starts at the Inner Title page (which counts as page `i`, but number is NOT displayed). The first page showing a printed number is the **Declaration** with `ii` at bottom center, ending with Abbreviations.
>   - **Arabic numerals (`1`, `2`, `3`, etc.)**: Starts at **1 Introduction** (showing Arabic numeral `1` on page 1) and continues sequentially through all chapters, figures, references, and appendices.
>   - Page Number Position: Bottom center of each page (`\cfoot{\thepage}`).
>   - Running Headers / Footers: Strictly **NO running headers or footers** aside from bottom-center page numbers.
> - **Exact Heading & Numbering Hierarchy:**
>   - 1st Numeral (Chapter): **Bold Capital, Font 12** (e.g., `1 INTRODUCTION`).
>   - 1st Numeral with decimals (Section): **Bold Simple, Font 12** (e.g., `1.1 Justification`).
>   - 1st Numeral with 2 decimals (Subsection): **Simple, Font 12, only first letter capitalized** (e.g., `1.2.1 General objective`).
>   - 1st Numeral with 3 decimals (Sub-subsection): **Simple, Font 12, only first letter capitalized** (e.g., `2.3.1.1 ...`).
> - **Tables and Figures Numbering & Caption Placement:**
>   - Table captions: **ABOVE the table (Font 12)**, numbered `Table X.Y` (e.g., `Table 2.1`), NO shading in table cells.
>   - Figure captions: **BELOW the figure (Font 12)**, numbered `Figure X.Y` (e.g., `Figure 1.2`).
> - **Order of Sections (Exact):**
>   1. Title page (Cover & Inner Title Page)
>   2. Declaration of the Candidate
>   3. Acknowledgement
>   4. Abstract (200--300 words)
>   5. Table of Contents
>   6. List of Figures
>   7. List of Tables
>   8. List of Abbreviations
>   9. 1. Introduction
>   10. 2. Objectives
>   11. 3. Literature Review
>   12. 4. Methodology
>   13. 5. Results
>   14. 6. Discussion and Conclusions
>   15. References (IEEE format)
>   16. Appendices (A through E)

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
> - **Strict Prohibition on Thesis-Tooling / Meta-Tools in Thesis Text:** The thesis text MUST focus 100% on the virtual keyboard research project, algorithms, software libraries, and experimental setups. **NEVER mention thesis writing/compilation tools (such as TeX Live, pdflatex, latexmk, biber), humanizer utilities (`./humanizer`, `client.py`), PDF conversion utilities (`./converter`), or AI meta-tooling anywhere in the thesis chapters** (including Software Requirements, System Architecture, Scope, or Methodology). These utilities are used by the author to build the dissertation document, but are NOT part of the virtual keyboard research artifact.
> - **Strict Thesis Word Count Limit (10,000 to 13,000 Words Only):** The complete thesis document must strictly fall within the word limit of **10,000 to 13,000 words** total (excluding preliminary pages, references, and appendices). When drafting and budgeting sections and chapters, always stay strictly within this target boundary.
> - **IEEE Citation Style:** All thesis chapters **MUST strictly use IEEE citation style** (`style=ieee` via BibLaTeX/biber).
> - **Mandatory Build Execution & Output Directory Rule:** Whenever any thesis chapter or file is updated, ALWAYS immediately compile/build the document using `latexmk -pdf -outdir=out main.tex` (or `pdflatex -output-directory=out main.tex`). ALL generated build outputs, compiled PDFs, auxiliary files, and created artifacts MUST be output to/stored in the `out/` directory (`out/main.pdf`) and NEVER placed elsewhere.
> - **Faculty Research Methodology Guidelines (`research-db/research_methodology_guide.md`):** ALWAYS read and strictly align with `research-db/research_methodology_guide.md` (synthesized from all lecture slides in `./lectureSlides`) before updating or reviewing thesis chapters. It defines faculty evaluation expectations regarding the IRCA problem statement framework, SMART/Bloom's taxonomy objectives, Saunders' Research Onion, PRISMA 2020 literature review standards, IEEE referencing, empirical validity/reliability, and quantitative analysis reporting.
> - **Virtual Keyboard Architecture & Technical Specification Guide (`research-db/virtual_keyboard_project_guide.md`):** ALWAYS read and strictly align with `research-db/virtual_keyboard_project_guide.md` when reviewing the codebase, checking directory structures, or drafting and updating thesis chapters. It documents the modular directory roles (`virtualKeyboardSetup/`, `annotator/`, `dataPipeline/`, `modelBenchmark/`, `videos/`), the 13-step dataset creation pipeline, benchmark results across 22 model configurations (`LSTM_All_Combined` 94.34% accuracy, 94.37% F1), the four detector runtime filters, AprilTag planar homography estimation, distal vector projection ($5.0\text{ mm}$), and touch-release confirmation logic.
> - **Thesis Assumption Context:** The major components (`designer/`, `mediapipeDetector/`, `aprilTag/`) have been developed and tested separately, proving complete technical feasibility. **When writing thesis chapters, assume the entire assembled system (App 1 Designer + App 2 Runtime Engine) is fully created and operational as specified in Section 6 of this document.**

---

## 1. Research Rationale & Partitioned Development Strategy

This project focuses on the development and evaluation of a **customizable paper-based virtual keyboard system** powered by a single regular monocular RGB camera, AprilTag fiducial homography tracking, and PyTorch deep learning, specifically architected to run in near real-time on standard commodity CPUs without a GPU, using regular plain printed paper and low-end/standard cameras without requiring specialized hardware.

### Partitioned Research Strategy
Because monocular paper touch detection is an experimental computer vision concept, component modules were built and tested in **isolated experimental partitions** to validate feasibility before assembling the final end-to-end application suite:

1. **Layout Design & Homography Simulation (`designer/` & `designer/analyzer/`)**:
   - `designer_app.py` & `designer/analyzer/`: Early feasibility prototypes for drag-and-drop layout design and homography simulation before building the modular production software in `virtualKeyboardSetup/designer/main.py`.
2. **Dataset Creation & Pipeline Engineering (`dataPipeline/src/`)**:
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
   * *Investigation:* Decoupling physical paper layout geometry (<DesignerLayout>) from digital software actions (<DetectorActions> and <DetectorSettings>) within a single unified XML layout file, allowing a single printed paper sheet to be dynamically bound to multiple distinct action profiles (shortcuts, shell commands) without reprinting the page.

---

## 2. System Workflow & Pipeline Architecture

1. **Layout Design & Export**: The user creates a custom key layout using the GUI designer ([`virtualKeyboardSetup/designer/main.py`](file:///home/lahirukasunidilhara/Documents/university/research/virtualKeyboardSetup/designer/main.py)). The design is exported as:
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
| [`research-db/virtual_keyboard_project_guide.md`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/virtual_keyboard_project_guide.md) | **Primary Technical Architecture & Thesis Reference Guide.** Full documentation of directory roles, data pipeline stages, model benchmark findings, and detector internal filtration mechanics. | **Core Technical Reference**                     |
| [`virtualKeyboardSetup/designer/`](file:///home/lahirukasunidilhara/Documents/university/research/virtualKeyboardSetup/designer)    | **Application 1: Paper Layout Designer.** PySide6 GUI tool for interactive layout design, constraint validation, XML layout export, and printable vector AprilTag PDF generation.                                                                                            | **Core Application (Operational)**               |
| [`virtualKeyboardSetup/detector/`](file:///home/lahirukasunidilhara/Documents/university/research/virtualKeyboardSetup/detector)    | **Application 2: Virtual Keyboard Runtime Engine.** Multi-threaded PySide6 application running MediaPipe detection, 4-stage pipeline filtration, PyTorch touch inference, AprilTag homography mapping, and OS action execution.                                               | **Core Application (Operational)**               |
| [`dataPipeline/`](file:///home/lahirukasunidilhara/Documents/university/research/dataPipeline)                                        | **13-Step Data Pipeline.** Feature processing, unitless $L_{\text{hand}}$ normalization, 5-frame sliding windowing, hand transit movement filtering, velocity derivation, and train/test dataset generation.                                                                   | **Data Pipeline Engine**                         |
| [`modelBenchmark/`](file:///home/lahirukasunidilhara/Documents/university/research/modelBenchmark)                                    | **Deep Learning Benchmark Suite.** Benchmark engine evaluating 22 model configurations across 5 architecture families. Houses trained weights (`best_finger_touch_lstm.pth`) and evaluation logs (`summary_all.csv`).                                                          | **Model Benchmark Engine**                       |
| [`annotator/`](file:///home/lahirukasunidilhara/Documents/university/research/annotator)                                              | **Touch Dataset Annotator GUI.** CustomTkinter application for 12 FPS frame-by-frame ground-truth labeling of touch/non-touch events across all five fingers.                                                                                                                  | **Annotation Tool**                              |
| [`videos/`](file:///home/lahirukasunidilhara/Documents/university/research/videos)                                                    | Repository storing recorded MP4 video files, MediaPipe landmark CSVs, and synchronized window annotation CSVs.                                                                                                                                                                | **Dataset Storage**                              |
| [`humanizer/`](file:///home/lahirukasunidilhara/Documents/university/research/humanizer)                                              | Humanizer service 1 (`client.py`, port 8000) for text humanization.                                                                                                                                                                            | **Writing / Humanizing Tool**                    |
| [`humanizer2/`](file:///home/lahirukasunidilhara/Documents/university/research/humanizer2)                                            | Humanizer service 2 (`client.py`, port 8001) for round-robin alternating text humanization (< 200 words).                                                                                                                                      | **Writing / Humanizing Tool**                    |
| [`sources/thesis_guideline.pdf`](file:///home/lahirukasunidilhara/Documents/university/research/sources/thesis_guideline.pdf)        | Official University Thesis Preparation and Formatting Guidelines document (PDF format). All formatting, structure, margins, font sizes, pagination, and layout rules must strictly follow this.                                                                               | **Core Formatting & Structure Guideline**       |
| [`sources/thesis_guideline.txt`](file:///home/lahirukasunidilhara/Documents/university/research/sources/thesis_guideline.txt)        | Plain-text conversion of `thesis_guideline.pdf` for convenient inspection of all formatting, structural order, heading styles, and pagination specifications.                                                                                                                 | **Guideline Text Reference**                     |
| [`sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf`](file:///home/lahirukasunidilhara/Documents/university/research/sources/old_breakdown_of_chapter1_chapter2_chapter3.pdf) | Previous draft breakdown for Chapters 1, 2, and 3. Used for reference/context only (internal mechanisms updated; **do not cite**).                                                                                                                                           | Contextual draft reference                       |
| [`pdf-sources/`](file:///home/lahirukasunidilhara/Documents/university/research/pdf-sources)                                          | Repository storing PDF research papers and literature references.                                                                                                                                                                                                              | Paper storage                                    |
| [`research-db/`](file:///home/lahirukasunidilhara/Documents/university/research/research-db)                                          | Literature analysis database containing [`summary-matrix.md`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/summary-matrix.md), [`references.bib`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/references.bib), and [`research_methodology_guide.md`](file:///home/lahirukasunidilhara/Documents/university/research/research-db/research_methodology_guide.md). | Thesis reference hub                             |
| [`chapters/`](file:///home/lahirukasunidilhara/Documents/university/research/chapters)                                                | Directory storing LaTeX `.tex` files for individual thesis chapters.                                                                                                                                                                                                           | Thesis writing                                   |
| [`main.tex`](file:///home/lahirukasunidilhara/Documents/university/research/main.tex)                                                 | Root LaTeX file that compiles all thesis chapters.                                                                                                                                                                                                                             | Thesis entrypoint                                |
| [`out/`](file:///home/lahirukasunidilhara/Documents/university/research/out)                                                         | Output directory storing compiled PDF files and TeX build artifacts.                                                                                                                                                                                                           | Output directory                                 |
| [`figures/`](file:///home/lahirukasunidilhara/Documents/university/research/figures)                                                  | Visual assets, diagrams, charts, and figures used in the thesis.                                                                                                                                                                                                               | Thesis figures                                   |

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
5. **Strict Thesis Preparation & Formatting Guidelines ([`sources/thesis_guideline.pdf`](file:///home/lahirukasunidilhara/Documents/university/research/sources/thesis_guideline.pdf) / [`sources/thesis_guideline.txt`](file:///home/lahirukasunidilhara/Documents/university/research/sources/thesis_guideline.txt))**:
   - **Supervisory Guidance & Flexibility on Sub-topics:** The main chapter titles and high-level structural sequence specified in the university thesis guidelines are strictly respected, while internal section titles, sub-headings, and subsections are flexible guidance rather than rigid compulsory titles. Adapt internal subtitles to fit the technical and experimental specifics of this virtual keyboard research.
   - **Thesis Structural Order:**
     1. Title page (Cover & Inner Title Page)
     2. Declaration of the Candidate (page number ii at bottom center)
     3. Acknowledgement
     4. Abstract (200-300 words)
     5. Table of Contents
     6. List of Figures
     7. List of Tables
     8. List of Abbreviations (alphabetical order)
     9. Chapter 1: Introduction (Arabic numeral 1 starts here)
     10. Chapter 2: Objectives (General and specific objectives)
     11. Chapter 3: Literature Review
     12. Chapter 4: Methodology
     13. Chapter 5: Results
     14. Chapter 6: Discussion and Conclusions
     15. References (IEEE style)
     16. Appendices
   - **Word Count Limit:** Strictly **10,000 to 13,000 words** total for the core thesis body (excluding preliminary pages, references, and appendices). All chapter budgeting and writing must respect this target boundary.
   - **Page Setup & Margins:** Standard A4 size. Uniform margins: Left = 1" (or 1 1/4" for binding allowance), Right = 1", Top = 1", Bottom = 1" (with allowance for page numbers).
   - **Font & Spacing:** Times New Roman, 12 pt, single column, 1.5 line spacing applied throughout.
   - **Pagination:**
     - Lower-case Roman numerals (`ii`, `iii`, `iv`, ...) starting at the Inner Title page (counts as `i`, but number does not appear; first visible number is `ii` on Declaration) through List of Abbreviations.
     - Arabic numerals (`1`, `2`, `3`, ...) starting at Introduction (page 1) continuously through text, figures, tables, references, and appendices.
     - Bottom center placement on every page.
     - No running headers or footers aside from page numbers.
   - **Heading Styles & Numbering Hierarchy (Arabic numerals up to 3 decimals):**
     - First Numeral: Bold Capital, Font 12 (e.g., `1 INTRODUCTION`)
     - First Numeral with 1 decimal: Bold Simple, Font 12 (e.g., `1.1 Justification`)
     - First Numeral with 2 decimals: Simple, Font 12, only first letter capitalized (e.g., `1.2.1 General objective`)
     - First Numeral with 3 decimals: Simple, Font 12, only first letter capitalized (e.g., `2.3.1.1 Factor affecting...`)
   - **Table Headings & Figure Captions:**
     - Table captions: Font 12, placed **above** the table (e.g., `Table 2.1: Table Caption`). Avoid shading in table cells.
     - Figure captions: Font 12, placed **below** the figure (e.g., `Figure 1.2: Figure Caption`).
     - Numbering: First digit represents main chapter/section, second digit denotes sequential item.
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
10. **Mandatory Humanizer Tool Protocol (`humanizer` & `humanizer2`)**:
    - **Dual Humanizer Availability & CLI Usage:** Two local humanizer services are available to convert drafted thesis text into natural human-style writing:
      - **Humanizer 1 (`humanizer/client.py`):** Runs on port 8000.
        `humanizer/.venv/bin/python humanizer/client.py "Text to humanize"`
      - **Humanizer 2 (`humanizer2/client.py`):** Runs on port 8001.
        `humanizer2/.venv/bin/python humanizer2/client.py "Text to humanize"`
    - **Round-Robin Alternating Usage:** Randomly choose or alternate between `humanizer2` and `humanizer` (round-robin) across consecutive text chunks. This mixes vocabulary, sentence structures, and burstiness to effectively evade AI detection patterns.
    - **Strict Chunk Size Limit (< 200 Words):** `humanizer2` has a hard limit of 200 words. Always send cohesive semantic chunks **strictly under 200 words** (ideally 80 to 160 words). Never exceed 200 words in any single CLI invocation.
    - **Single-Line Strings Only (No Internal Newlines `\n`):** NEVER pass multi-line strings or strings containing newline characters (`\n`) into the CLI command. Always format input text as a single, continuous line without newline breaks (`\n`).
    - **Server Management & Troubleshooting:** The user runs and manages the humanizer backend servers. You do NOT need to run or fix the servers. You only run `client.py`. If a client cannot connect to its server (e.g., connection refused or network error), inform the user immediately so they can fix it. Do NOT attempt to run or fix the server yourself.
    - **Strict Zero Sentence-Rewriting Rule (Preserve Humanized Flow 100%):** NEVER edit, tweak, or rewrite the sentences returned by the humanizers. You must not change whole sentences on the humanizer output. Only swap or insert the specific compulsory technical words, numbers, citations, or references that strictly need replacement. Even small sentence-level smoothing or grammatical polishing will re-introduce AI writing patterns.
    - **Compulsory Technical Word Preservation & Restoration Rule (Mandatory):**
      - **Detection of Missing Technical Terms:** When text is processed through the humanizer, the tool may strip, omit, or replace domain-specific technical terms that are strictly compulsory (such as "MediaPipe", "AprilTag", "PyTorch", "LSTM", "BiLSTM", "Planar Homography $H$", "unitless hand-length scale normalization $L_{\text{hand}}$", "12 FPS", "21 skeletal hand landmarks", "distal vector projection", "tactile switch travel", etc.).
      - **Comparison & Restoration Step:** Once the humanized text is returned, compare it directly against the original input. If any compulsory technical words or acronyms (strictly technical words only!) are missing or replaced by vague generic language, re-insert or restore those exact technical terms into their proper positions in the text.
      - **Strict Non-Interference with Humanized Content:** You must NEVER alter, paraphrase, or rewrite the surrounding phrasing or sentence flow of the humanized text, because doing so destroys the humanization patterns and lowers the humanized score. You are ONLY permitted to restore missing compulsory technical terms.
      - **Mathematical Equation Exclusion:** Never pass raw multi-line mathematical equations or raw LaTeX equation blocks (`\begin{equation}...\end{equation}`) into the humanizer.
      - **Replacement Unit:** Grab individual chunks under 200 words, humanize them, restore any missing compulsory technical terms, and replace the existing portion in the LaTeX file with the humanized text.
    - **Punctuation Artifact Cleaning:**
      - The humanizer may occasionally output weird, glitched, or misplaced punctuation marks (such as stray punctuation symbols, glitched quotation marks, illegal long dashes `—`/`–`, date tags `[[ Set date ... ]]`, or section symbols `§`). Clean up or remove these unwanted punctuation artifacts immediately.
    - **Validation Process & Semantic Integrity Check (Minimum 80% Meaning Retention):**
      - After receiving the humanized text and restoring any missing technical terms, verify that the core technical meaning and critical facts are preserved at **80% or higher**.
      - A slight shift in sentence style or phrasing is completely acceptable.
      - If the core meaning is altered by more than 20% (<80% meaning retained), adjust the input draft text to be more explicit and resubmit it to `client.py`.
      - Once an output meeting the 80%+ semantic retention standard is obtained, place it into LaTeX.
    - **Mandatory Response Reporting Requirement:**
      - In EVERY response where humanization is performed, you **MUST explicitly report to the user**:
        1. **Word Modification Confirmation:** State explicitly whether any words were modified (must confirm zero sentence rewrites, detailing only compulsory technical term/math restorations applied).
        2. **Semantic Meaning Retention Metric:** State the estimated percentage of core technical meaning and facts retained (e.g. "Semantic Retention: ~85%").
        3. **Punctuation, Links & Formatting Summary:** Detail any glitched punctuation artifacts removed (e.g. `§`, `—`), technical term fixes made, and confirm that all figures, tables, cross-references (`\ref{...}`), citations (`\cite{...}`), and LaTeX environments were accurately restored.
    - **Citation Tracking & Exact Preservation:** Keep track of all citation keys (`\cite{Key1, Key2}`) before passing text to the humanizer. Once the humanized text is returned, re-insert the exact citation macros into their appropriate logical places in the text without losing any citation.
    - **Figure, Table & Cross-Reference Link Preservation (`\ref{...}`):**
      - All textual references to figures, tables, sections, algorithms, and equations (e.g., `Figure~\ref{fig:rich_picture_diagram}`, `Table~\ref{tab:project_scope}`, `Section~\ref{sec:...}`) MUST be strictly preserved.
      - During humanization, ensure references are recognized (e.g. as "Figure 1" or "Table 1" in plain text passed to `client.py`), and upon receiving the humanized text, restore the exact LaTeX cross-referencing macros (`Figure~\ref{...}`, `Table~\ref{...}`) into their appropriate locations so that all document links and clickable cross-references remain 100% functional.
    - **Preservation & Restoration of LaTeX Mathematical Formulas & Symbols:**
      - When humanizing paragraphs that contain mathematical formulas, equations, or scientific variables (such as `$V_y$`, `$A_y = \frac{dV_y}{dt} \ll 0$`, `$MT = a + b \log_2 \left( 1 + \frac{D}{W} \right) = a + b \cdot ID$`, `$\text{WPM} = \frac{|T| - 1}{S} \times 60 \times \frac{1}{5}$`, coordinates $(x, y)$, matrices $H$, etc.), the humanizer may simplify or drop LaTeX math delimiters.
      - **Mandatory Rule:** You MUST carefully verify, fix, and restore all mathematical formulas, inline math expressions (`$...$`), variable subscripts/superscripts, fractions (`\frac{...}{...}`), Greek letters, and mathematical notations into proper LaTeX syntax within the humanized sentences without changing the surrounding humanized words.
    - **Target Text Scope (What to Humanize vs. Exclude):**
      - **DO NOT humanize:** The six top-level official Chapter titles prescribed by NSBM university guidelines (`1. Introduction`, `2. Objectives`, `3. Literature Review`, `4. Methodology`, `5. Results`, `6. Discussion and Conclusions`), table of contents, reference list / bibliography (`\bibliography`, `references.bib`), figure/graph vector code (TikZ diagrams), and raw table formatting structures (`\begin{tabular}...\end{tabular}`).
      - **MUST humanize:** Body paragraphs, section/subsection/subsubsection titles (`\section{...}`, `\subsection{...}`, `\subsubsection{...}`), paragraph headings (`\paragraph{...}`), bullet point items and their bold sub-titles (`\item \textbf{...}`), and textual cell descriptions in tables.
    - **Humanizing Titles and Section Headings:**
      - Humanize section, subsection, subsubsection, and item titles naturally so they do not sound like rigid AI textbook headings.
      - Pass the title into the humanizer to obtain an organic, human-sounding heading.
    - **Flexible Bullet List Strategy:**
      - Do NOT always eliminate bullet lists. Maintain structured bullet lists where clarity and systematic breakdown are important (e.g., formal test cases, ablation lists, objective triangulation).
      - Convert bullet lists into flowing prose only when necessary (e.g., when an AI detector flags a repetitive bullet pattern or when presenting cohesive analytical discussion).
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
  │          (virtualKeyboardSetup/designer/main.py)            │
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

#### Application 1: Layout Designer Suite (`virtualKeyboardSetup/designer/main.py`)
*Note: `designer/designer_app.py` represents the early exploratory prototype.*
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
