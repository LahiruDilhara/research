# Thesis Humanization Progress Log

This log tracks the humanization progress across all sections and subsections of the thesis.
Sections are processed through `humanizer/client.py` and reviewed to restore any missing compulsory technical terms. No section is processed twice unless explicitly requested.

---

## Chapter 1: Introduction (`chapters/chapter01.tex`)
- [x] 1.1 Chapter Overview
- [x] 1.2 Problem Background
- [x] 1.3 Problem Statement
  - [x] 1.3.1 Ideally
  - [x] 1.3.2 Reality
  - [x] 1.3.3 Consequences
  - [x] 1.3.4 Aim
- [x] 1.4 Research Question
- [x] 1.5 Research Motivation
- [x] 1.6 Research Aim
- [x] 1.7 Research Objectives Overview
- [x] 1.8 Rich Picture of the Proposed Solution
- [x] 1.9 Resource Requirements
  - [x] 1.9.1 Hardware Requirements
  - [x] 1.9.2 Software Requirements
- [x] 1.10 Project Scope
  - [x] 1.10.1 In-Scope Technical Boundaries
  - [x] 1.10.2 Out-of-Scope Boundaries and Future Extensions
- [x] 1.11 Chapter Summary

---

## Chapter 2: Objectives (`chapters/chapter02.tex`)
- [x] 2.1 General Objective
- [x] 2.2 Specific Objectives
  - [x] 2.2.1 Objective 1: Theoretical Analysis and Baseline Requirements
  - [x] 2.2.2 Objective 2: Hand Kinematic Representation and Scale Normalization
  - [x] 2.2.3 Objective 3: Deep Learning Sequence Model Architecture Benchmark
  - [x] 2.2.4 Objective 4: System Architecture Engineering and Two-Application Suite Design
  - [x] 2.2.5 Objective 5: Empirical System Evaluation and Functional Benchmarking

---

## Chapter 3: Literature Review (`chapters/chapter03.tex`)
- [x] 3.1 Chapter Overview
- [x] 3.2 Conceptual Map of the Literature
  - [x] 3.2.1 Literature Search Strategy and PRISMA 2020 Systematic Protocol
- [x] 3.3 Domain Overview
  - [x] 3.3.1 Evolution of Text Entry and Spatial HCI Hardware
  - [x] 3.3.2 Kinematic Typing Dynamics and Biomechanical Surface Impact
  - [x] 3.3.3 Ergonomics and Human Input Performance Metrics
  - [x] 3.3.4 Multi-Finger Tendon Coupling
- [x] 3.4 Existing Systems and Sensing Modalities
  - [x] 3.4.1 Projection-Based and Infrared Sensor Systems
  - [x] 3.4.2 Shadow-Based Monocular RGB Keyboards
  - [x] 3.4.3 Hardware-Assisted Depth Sensors (RGB-D and ToF)
  - [x] 3.4.4 3D Mid-Air Typing and AR/VR Virtual Keyboards
  - [x] 3.4.5 Paper-Anchored Monocular RGB Keyboards
  - [x] 3.4.6 Specialized and Assistive Modalities
  - [x] 3.4.7 Fiducial Tracking in Planar Robotics
  - [x] 3.4.8 Comparative Literature Analysis Matrix
- [x] 3.5 Technological Analysis
  - [x] 3.5.1 Hand Pose Estimation and Skeletal Joint Tracking
  - [x] 3.5.2 Kinematic Scale Normalization and Raw Feature Propagation
  - [x] 3.5.3 Planar Homography and Fiducial Tracking
  - [x] 3.5.4 Deep Sequential Touch Detection Models
  - [x] 3.5.5 System Architecture Sensing Trade-Offs
- [x] 3.6 Reflection and Synthesis of Research Gaps
  - [x] 3.6.1 Synthesis of Prior Literature Limitations
  - [x] 3.6.2 Formal Research Gap Justification
- [x] 3.7 Chapter Summary

---

## Chapter 4: Methodology (`chapters/chapter04.tex`)
- [x] 4.1 Chapter Overview
- [x] 4.2 Research Framework and Scientific Methodology
  - [x] 4.2.1 Research Paradigm and Deductive Reasoning
  - [x] 4.2.2 Methodological Grounding via Saunders' Research Onion
  - [x] 4.2.3 Design Science Research Methodology (DSRM)
  - [x] 4.2.4 Partitioned Development and Experimental Strategy
- [x] 4.3 System Architecture and Software Design
- [x] 4.4 Application 1: Layout Designer Suite
  - [x] 4.4.1 Key Geometry and Fiducial Anchor Positioning
  - [x] 4.4.2 Synchronized Export Engine
- [x] 4.5 Data Collection and Kinematic Preprocessing Pipeline
  - [x] 4.5.1 Dataset Collection Protocol and Methodological Rigor
  - [x] 4.5.2 12 FPS Sub-Sampling and Single-Hand Tracking
  - [x] 4.5.3 Unitless Hand-Length Scale Normalization
  - [x] 4.5.4 Temporal Sliding Window Construction
  - [x] 4.5.5 CPU Time Budget for 12 FPS Real-Time Processing
- [x] 4.6 Deep Learning Sequence Model Architecture
  - [x] 4.6.1 Multi-Finger Touch Contact Classification
  - [x] 4.6.2 PyTorch LSTM Sequence Classifier
- [x] 4.7 Application 2: Runtime Virtual Keyboard Engine
  - [x] 4.7.1 Planar Homography Derivation ($H$)
  - [x] 4.7.2 Fingertip Coordinate Transformation and Key Lookup
  - [x] 4.7.3 Action Multiplexing and Command Dispatch
- [x] 4.8 Chapter Summary

---

## Chapter 5: Results (`chapters/chapter05.tex`)
- [x] 5.1 Chapter Overview
- [x] 5.2 Test Plan and Structured Test Cases
  - [x] 5.2.1 Functional Testing Suite
  - [x] 5.2.2 Non-Functional Testing Suite
- [x] 5.3 Testing and Evaluation Workflow
- [x] 5.4 Detailed Empirical Results and Strategy Review
  - [x] 5.4.1 Deep Learning Model Architecture Benchmark Evaluation
  - [x] 5.4.2 Evaluation of MediaPipe Landmark Subsets and Kinematic Information Content
  - [x] 5.4.3 Temporal Window Length and Stride Sensitivity Analysis
  - [x] 5.4.4 Scale Normalization Ablation Study Across Distance and Users
  - [x] 5.4.5 AprilTag Homography Tracking Accuracy and Tilt Sensitivity
  - [x] 5.4.6 Real-Time Pipeline Latency and Resource Profiling
  - [x] 5.4.7 Macro Command Trigger Execution and Multi-Profile Action Benchmark
- [x] 5.5 Chapter Summary

---

## Chapter 6: Discussion and Conclusions (`chapters/chapter06.tex`)
- [x] 6.1 Chapter Overview
- [x] 6.2 Accomplishment of Research Objectives
  - [x] 6.2.1 Triangulation Evaluation of Objective 1: Theoretical Analysis and Baseline Requirements
  - [x] 6.2.2 Triangulation Evaluation of Objective 2: Hand Kinematic Representation and Scale Normalization
  - [x] 6.2.3 Triangulation Evaluation of Objective 3: Deep Learning Sequence Model Architecture Benchmark
  - [x] 6.2.4 Triangulation Evaluation of Objective 4: System Architecture Engineering and Two-Application Suite Design
  - [x] 6.2.5 Triangulation Evaluation of Objective 5: Empirical System Evaluation and Functional Benchmarking
- [x] 6.3 Problems Encountered and Technical Mitigations
  - [x] 6.3.1 Webcam Image Quality and the 13 FPS Hardware Discovery
  - [x] 6.3.2 Failure of Hand-Drawn Paper Layouts and Freehand Vision
  - [x] 6.3.3 Corner Marker Occlusion and the Coordinate Projection Breakthrough
  - [x] 6.3.4 False Touch Triggers During Hand Translation and the 82% Model Ceiling
  - [x] 6.3.5 Tendon Coupling and Absence of Mechanical Tactile Feedback
- [x] 6.4 Self-Reflection
  - [x] 6.4.1 Your Ideology About the Research Carried Out
  - [x] 6.4.2 Benefits Gained
  - [x] 6.4.3 Learning Curves
- [x] 6.5 Business Insight of the Proposed Concept
  - [x] 6.5.1 Commercial Viability and Cost Efficiency
  - [x] 6.5.2 Real-World Application Possibilities
- [x] 6.6 Future Recommendations
- [x] 6.7 Chapter Summary
