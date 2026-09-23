# Research Methodology & Writing Requirements Guide

This document synthesizes core guidelines, lecture notes, frameworks, and academic expectations from the research methodology lecture materials (`./lectureSlides`). These principles define how faculty evaluators review the thesis and must be rigorously respected across all thesis chapters.

---

## 1. Research Problem Formulation: The IRCA Framework

A strong research problem in computing avoids vague descriptions and converts generalized challenges into a structured, investigable problem using the **IRCA** model:

| Component | Definition | Application in This Virtual Keyboard Thesis |
| :--- | :--- | :--- |
| **I — Ideally** | What the ideal technical or interaction state should be. | Users should be able to turn any flat surface into a customizable, responsive, multi-finger input device using low-cost commodity hardware (paper, standard webcam, standard CPU). |
| **R — Reality** | How current real-world systems fail or contradict this ideal. | Existing monocular virtual keyboards are limited to single-finger tracking, suffer from high CPU latency (>100 ms) or require dedicated GPUs, rely on clumsy dwell-time delays (500–1000 ms), require expensive specialized hardware (depth cameras, laser projectors), or fail under lighting changes and paper displacement. |
| **C — Consequences** | The negative impact of this contradiction. | Practical adoption of monocular virtual input remains stalled; users cannot achieve natural touch typing speeds, interaction is physically fatiguing, and deployments remain brittle and cost-prohibitive. |
| **A — Aim** | The specific intention of this study to address the gap. | To develop and evaluate a monocular RGB paper-based virtual keyboard using MediaPipe skeletal tracking, unitless scale normalization, lightweight PyTorch LSTM sequence classification, and dynamic AprilTag homography tracking running in real-time on commodity CPUs. |

---

## 2. Research Aims, Objectives, Questions, and Hypotheses

Clear differentiation between project management tasks and research knowledge objectives is mandatory:
- **Research Aim:** The overarching broad purpose (exactly one overarching aim).
- **Research Objectives (SMART + Bloom's Taxonomy):** Specific, actionable milestones using precise action verbs (Design, Develop, Analyze, Evaluate, Benchmark) rather than vague verbs ("understand", "study"). Objectives must not describe project management chores (e.g., "conduct 10 interviews" or "write code").
- **Research Questions:** Clear, focused interrogative forms corresponding directly to research objectives. Avoid simple Yes/No questions.
- **Hypotheses:** Quantitative, testable predictions linking independent and dependent variables.

### The SMART Criteria Checklist
- **Specific:** Clearly defines what technical artifact or phenomenon is investigated.
- **Measurable:** Quantifiable metrics (e.g., F1-score $\ge 0.95$, end-to-end latency $< 30\text{ ms}$, homography error $< 2\text{ mm}$).
- **Achievable:** Technically feasible with commodity hardware and CPU-only inference.
- **Relevant:** Directly solves the identified literature gaps in HCI and computer vision.
- **Time-bound / Scope-bounded:** Validated through empirical benchmarking and controlled user evaluation.

---

## 3. Methodological Grounding: Saunders' Research Onion

The thesis methodology must align with the layers of Saunders' Research Onion (Saunders, Lewis & Thornhill, 2019):

1. **Research Philosophy:** **Pragmatism / Positivism**
   - Emphasizes practical utility, empirical measurability, and engineering problem-solving. Factual, quantifiable performance metrics (latency, F1-score, frames per second, typing accuracy) establish empirical validity.
2. **Research Approach:** **Deductive**
   - Formulates hypotheses and kinematic models (temporal deceleration profiles, homography projective geometry), developing an algorithmic pipeline and testing it against empirical benchmark datasets.
3. **Research Strategy:** **Design Science Research Methodology (DSRM) & Experimental Benchmarking**
   - Iterative design, development, and quantitative evaluation of software artifacts (`designer_app.py`, `datacreator/`, `run_all.py`, `realtimeprocess/`, `aprilTag/`).
4. **Methodological Choice:** **Quantitative / Mixed-Methods (Dominantly Quantitative Empirical)**
   - Quantitative evaluation of model accuracy, benchmark latency, and interaction speed; supplemented by subjective user feedback dimensions.
5. **Time Horizon:** **Cross-Sectional**
   - Controlled laboratory performance evaluations and benchmark trials conducted across varied camera resolutions, angles, and participant interaction sessions.
6. **Data Collection & Analysis:**
   - Primary data collection: High-speed video capture of finger interactions, skeletal landmark coordinate logging, real-time CPU execution profiling, and standardized UX surveys.

---

## 4. Literature Review Standards: Critical Synthesis & PRISMA 2020

A literature review is **both a process and a product**. It must not be a mere descriptive catalog of past papers ("X said this, Y said that"), but a **critical synthesis** connecting themes, evaluating methodological validity, and pinpointing unanswered questions.

### Synthesis & Critique Model
- **Paragraph Structure:** Topic sentence $\rightarrow$ Empirical evidence from literature $\rightarrow$ Comparison and contrast across approaches $\rightarrow$ Critical evaluation of limitations $\rightarrow$ Direct link to our research gap and proposed solution.
- **Thematic Organization:** Group literature thematically (e.g., markerless optical tracking, shadow-based contact detection, specialized depth-sensor keyboards, fiducial planar tracking).
- **PRISMA 2020 Systematic Review Principles:**
  - Explicit search parameters: databases (IEEE Xplore, ACM Digital Library, Google Scholar, Scopus), date range (2010–2025), Boolean search strings (`("virtual keyboard" OR "paper keyboard") AND ("computer vision" OR "monocular RGB") AND ("touch detection" OR "homography")`).
  - Clear inclusion and exclusion criteria (e.g., excluding systems requiring wearable data gloves, depth sensors, or active projected surfaces).
  - Summary matrix comparing existing systems against key benchmark dimensions (sensing modality, multi-finger capability, latency, lighting resilience, layout flexibility).

---

## 5. Computing Paper & Thesis Structure (IMRaD & IEEE Standards)

The research document follows the standard academic structure expected in computing disciplines:
1. **Introduction:** Context, problem statement (IRCA), research aim, significance, scope, and technical contributions.
2. **Objectives:** General objective, specific SMART objectives, and corresponding research questions.
3. **Literature Review:** Thematic appraisal of prior work, comparative synthesis matrix, and derivation of the 6 foundational research gaps.
4. **Methodology:** Research design (DSRM / Saunders' Onion), system architecture, dataset pipeline, mathematical formulations (scale normalization, planar homography, temporal windowing), and deep learning sequence modeling.
5. **Results and Analysis:** Empirical findings presented through structured tables and figures (22-model benchmark, landmark ablation, latency profiling across CPU resolutions, user typing metrics).
6. **Discussion and Conclusions:** Objective triangulation matrix linking every objective to empirical evidence, critical self-reflection, business/practical applications, and future research directions.

### Academic Citation & Referencing Integrity
- **Style:** Strictly **IEEE numbered format** (`style=ieee` with brackets `[#]`).
- **Quoting:** Exact phrases in quotation marks with citation and page number: `"..." [1, p. 15]`. Use sparingly.
- **Paraphrasing:** Rewrite source concepts in original words and cite `[2]`.
- **Summarizing:** Condense larger studies into concise findings with citation `[3]`.
- **Reference Completeness:** All in-text citations must correspond to the numbered bibliography (`research-db/references.bib`).

---

## 6. Empirical Rigor, Validity, and Reliability in Computing Research

Evaluators expect thorough validation of experimental claims:
- **Internal Validity:** Controlled variables during testing (fixed lighting levels, identical CPU hardware across runs, standard 12 FPS sub-sampling rate).
- **Construct Validity:** Ensuring that metrics truly reflect the phenomenon (e.g., F1-score for contact classification balance; end-to-end latency including capture, inference, and key dispatch).
- **External Validity & Generalisability:** Testing across diverse resolutions (360p, 480p, 720p), camera tilt angles ($0^\circ$ to $75^\circ$), and diverse hand sizes via unitless scale normalization ($L_{\text{hand}}$).
- **Reliability:** Consistency of results across repeated trials; 5-fold cross-validation during model training; verified deterministic behavior of homography matrix $H$.
- **Triangulation:**
  - *Data Triangulation:* Simulated touches, recorded benchmark datasets, live webcam sessions.
  - *Methodological Triangulation:* Offline deep learning benchmarks across 5 model families + live CPU pipeline profiling.
  - *Theoretical Triangulation:* Computer vision projective geometry (homography) combined with kinematic biomechanical motion modeling.

---

## 7. Quantitative Analysis & Usability Standards

Where quantitative and user experience evaluations are reported:
- **Descriptive Statistics:** Report measures of central tendency (Mean, Median) and dispersion (Standard Deviation $\sigma$) for typing speeds (WPM), character error rates (CER), and pipeline latencies.
- **Inferential Statistics & Significance:** When comparing configurations, report test statistics (e.g., $t$-test or ANOVA, $p$-values where $p < 0.05$ denotes significance).
- **UX Benchmarking (UEQ / SUS Standards):** Where subjective experience is evaluated, align with standardized dimensions: Attractiveness, Perspicuity, Efficiency, Dependability, Stimulation, and Novelty.
