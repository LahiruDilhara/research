import re

# ---------------------------------------------------------
# 1. Update Figure 1.1 in chapters/chapter01.tex
# ---------------------------------------------------------
with open('chapters/chapter01.tex', 'r') as f:
    c1 = f.read()

fig11_old = re.search(r'\\begin\{figure\}\[H\].*?\\caption\{Rich Picture of the Proposed Solution and Operational Pipeline Architecture\.\}.*?\\end\{figure\}', c1, re.DOTALL).group(0)

fig11_new = r"""\begin{figure}[H]
\centering
\resizebox{0.95\textwidth}{!}{%
\begin{tikzpicture}[
    box/.style={draw=blue!60, rectangle, rounded corners=4pt, text width=3.4cm, minimum height=1.1cm, align=center, fill=blue!5, font=\small\sffamily, thick},
    process/.style={draw=green!60!black, rectangle, rounded corners=4pt, text width=3.4cm, minimum height=1.1cm, align=center, fill=green!5, font=\small\sffamily, thick},
    model/.style={draw=orange!80!black, rectangle, rounded corners=4pt, text width=3.4cm, minimum height=1.1cm, align=center, fill=orange!10, font=\small\sffamily, thick},
    output/.style={draw=purple!60, rectangle, rounded corners=4pt, text width=3.4cm, minimum height=1.1cm, align=center, fill=purple!5, font=\small\sffamily, thick},
    arrow/.style={-Stealth, thick, draw=blue!80!black}
]

% Phase 1: Designer Suite (Row 1)
\node [box, at={(0, 6.0)}] (designer) {\textbf{PySide6 Layout GUI}\\Layout \& Anchor Design};
\node [output, at={(4.2, 6.0)}] (xml) {\textbf{XML Layout File}\\Bounding Boxes \& Actions};
\node [output, at={(8.4, 6.0)}] (pdf) {\textbf{Printable Layout PDF}\\Embedded AprilTag Anchors};

% Phase 2: Physical Setup & Camera (Row 2)
\node [process, at={(8.4, 4.0)}] (paper) {\textbf{Physical Paper Setup}\\Printed Sheet on Workspace};
\node [process, at={(4.2, 4.0)}] (camera) {\textbf{Monocular RGB Camera}\\Live Video Stream Capture};

% Phase 3: Vision Engine & Pose Normalization (Row 3)
\node [box, at={(0, 2.0)}] (apriltag) {\textbf{AprilTag Tracker}\\Sub-Pixel Quad Localization};
\node [model, at={(4.2, 2.0)}] (homography) {\textbf{Homography Matrix ($H$)}\\$3\times3$ Planar Transformation};

\node [box, at={(0, 0.0)}] (mediapipe) {\textbf{MediaPipe Landmarker}\\21 Hand Skeleton Keypoints};
\node [process, at={(4.2, 0.0)}] (filter) {\textbf{Hand Scale Normalization}\\Unitless $L_{\text{hand}}$ (No Filter)};
\node [process, at={(8.4, 0.0)}] (window) {\textbf{Temporal Windowing}\\5 Frames, 2-Frame Overlap};

% Phase 4: Model Classifier & System Action Dispatcher (Row 4)
\node [model, at={(8.4, -2.0)}] (lstm) {\textbf{PyTorch LSTM Model}\\Touch Event Classifier};
\node [process, at={(4.2, -2.0)}] (mapping) {\textbf{Planar Point Mapping}\\$P_{\text{XML}} = H \cdot P_{\text{pixel}}$};
\node [output, at={(0, -2.0)}] (execution) {\textbf{System Key Execution}\\OS Keystroke / Shell Action};

% Clean Non-Intersecting Connections
\draw [arrow] (designer) -- (xml);
\draw [arrow] (xml) -- (pdf);
\draw [arrow] (pdf) -- (paper);
\draw [arrow] (paper) -- (camera);

\draw [arrow] (camera) -- (apriltag);
\draw [arrow] (apriltag) -- (homography);

\draw [arrow] (camera) -- (mediapipe);
\draw [arrow] (mediapipe) -- (filter);
\draw [arrow] (filter) -- (window);

\draw [arrow] (window) -- (lstm);
\draw [arrow] (lstm) -- (mapping);
\draw [arrow] (homography) -- (mapping);
\draw [arrow] (mapping) -- (execution);

% XML schema connection to execution
\draw [arrow] (xml.west) -- ++(-0.6, 0) |- (execution.north);

\end{tikzpicture}%
}
\caption{Rich Picture of the Proposed Solution and Operational Pipeline Architecture.}
\label{fig:rich_picture_diagram}
\end{figure}"""

c1 = c1.replace(fig11_old, fig11_new)
with open('chapters/chapter01.tex', 'w') as f:
    f.write(c1)
print("Updated Figure 1.1 in chapter01.tex")


# ---------------------------------------------------------
# 2. Update Figure 2.1 and Figure 2.2 and Figure 2.4 in chapters/chapter02.tex
# ---------------------------------------------------------
with open('chapters/chapter02.tex', 'r') as f:
    c2 = f.read()

# Figure 2.1
fig21_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{Conceptual taxonomy and structural organization of the literature review\.\}.*?\\end\{figure\}', c2, re.DOTALL).group(0)

fig21_new = r"""\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    header/.style = {rectangle, draw=blue!80!black, fill=blue!25!white, thick, rounded corners=5pt, align=center, font=\sffamily\bfseries\small, inner sep=8pt, text width=16.5cm},
    box/.style = {rectangle, draw=blue!70!black, fill=blue!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\footnotesize, inner sep=6pt, text width=3.7cm, minimum height=1.2cm},
    subbox/.style = {rectangle, draw=gray!70, fill=gray!5!white, rounded corners=3pt, align=center, font=\sffamily\scriptsize, inner sep=5pt, text width=3.7cm},
    arrow/.style = {draw=blue!80!black, ->, >=Stealth, thick}
]

% Root Node
\node (root) [header, at={(0, 0)}] {\textbf{Monocular Vision-Based Virtual Keyboard Literature Taxonomy}};

% Level 1 Nodes - Perfectly leveled at y = -1.8cm across 4 distinct columns
\node (l1_domain) [box, at={(-6.45, -1.8)}] {1. HCI \& Input Paradigms\\(\S\ref{sec:lit_domain_overview})};
\node (l1_systems) [box, at={(-2.15, -1.8)}] {2. Sensing Modalities\\(\S\ref{sec:lit_existing_systems})};
\node (l1_tech) [box, at={(2.15, -1.8)}] {3. Technological Analysis\\(\S\ref{sec:lit_technological_analysis})};
\node (l1_reflect) [box, at={(6.45, -1.8)}] {4. Research Synthesis\\(\S\ref{sec:lit_reflection})};

% Connections to Level 1
\draw [arrow] (root.south -| l1_domain.north) -- (l1_domain.north);
\draw [arrow] (root.south -| l1_systems.north) -- (l1_systems.north);
\draw [arrow] (root.south -| l1_tech.north) -- (l1_tech.north);
\draw [arrow] (root.south -| l1_reflect.north) -- (l1_reflect.north);

% Column 1: Domain Overview
\node (l2_history) [subbox, below=0.4cm of l1_domain] {Evolution of Input Hardware\\\cite{ReviewVirtualKeyboard2020, Lee2022Virtual}};
\node (l2_drivers) [subbox, below=0.3cm of l2_history] {Paper Rationale \& Ergonomics\\\cite{Zhang2001Visual, Srivastava2012RealTime, Khare2019QWERTY}};

% Column 2: Existing Systems
\node (l2_proj) [subbox, below=0.4cm of l1_systems] {Projection \& IR Sensors\\\cite{Cheng2015Fingertip, Kudale2016RealTime}};
\node (l2_shadow) [subbox, below=0.3cm of l2_proj] {Shadow Analysis Methods\\\cite{Thomas2013Camera, Posner2012Single, Yue2014Blind}};
\node (l2_depth) [subbox, below=0.3cm of l2_shadow] {RGB-D / ToF Depth Sensing\\\cite{Lee2019Virtual, Toshpulatov2024RealTime}};
\node (l2_air) [subbox, below=0.3cm of l2_depth] {3D Mid-Air Typing / VR\\\cite{Boletsis2019TextInput, Enkhbat2020HandKey, Lee2022Virtual, Yoo2026WordLevel}};
\node (l2_paper) [subbox, below=0.3cm of l2_air] {Paper / Monocular RGB\\\cite{Zhang2001Visual, Srivastava2012RealTime, Khare2019QWERTY, Maman2023TypeNet}};
\node (l2_attack) [subbox, below=0.3cm of l2_paper] {Video Attack Side-Channels\\\cite{Yue2014Blind, Yang2022Towards}};

% Column 3: Technological Analysis
\node (l2_pose) [subbox, below=0.4cm of l1_tech] {MediaPipe Hand Landmarker\\\cite{Zhang2020MediaPipe, GilMartin2023Hand, Andriyanov2025Improving}};
\node (l2_scale) [subbox, below=0.3cm of l2_pose] {Hand Scale Normalization\\\cite{Doan2022Efficient, Kumar2026Hybrid}};
\node (l2_fiducial) [subbox, below=0.3cm of l2_scale] {AprilTag Homography $H$\\\cite{Wang2016AprilTag2, Kallwies2020Determining, Pirchheim2011Homography}};
\node (l2_deep) [subbox, below=0.3cm of l2_fiducial] {PyTorch LSTM Classifiers\\\cite{Hochreiter1997LSTM, Nunez2018Convolutional, Gammulle2021TMMF}};

% Column 4: Research Synthesis
\node (l2_gap) [subbox, below=0.4cm of l1_reflect] {Prior Art Limitations\\\cite{Thomas2013Camera, Lee2019Virtual, Maman2023TypeNet}};
\node (l2_solution) [subbox, below=0.3cm of l2_gap] {Proposed Architecture\\Synthesis (\S\ref{sec:lit_reflection})};

% Vertical Column Connections
\draw [arrow] (l1_domain) -- (l2_history);
\draw [arrow] (l2_history) -- (l2_drivers);

\draw [arrow] (l1_systems) -- (l2_proj);
\draw [arrow] (l2_proj) -- (l2_shadow);
\draw [arrow] (l2_shadow) -- (l2_depth);
\draw [arrow] (l2_depth) -- (l2_air);
\draw [arrow] (l2_air) -- (l2_paper);
\draw [arrow] (l2_paper) -- (l2_attack);

\draw [arrow] (l1_tech) -- (l2_pose);
\draw [arrow] (l2_pose) -- (l2_scale);
\draw [arrow] (l2_scale) -- (l2_fiducial);
\draw [arrow] (l2_fiducial) -- (l2_deep);

\draw [arrow] (l1_reflect) -- (l2_gap);
\draw [arrow] (l2_gap) -- (l2_solution);

\end{tikzpicture}%
}
\caption{Conceptual taxonomy and structural organization of the literature review.}
\label{fig:lit_conceptual_map_diagram}
\end{figure}"""

# Figure 2.2
fig22_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{MediaPipe 21 hand landmark anatomical topology\.\}.*?\\end\{figure\}', c2, re.DOTALL).group(0)

fig22_new = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    joint/.style = {circle, draw=red!80!black, fill=red!60!white, inner sep=0pt, minimum size=6pt},
    tip/.style = {circle, draw=blue!80!black, fill=blue!60!white, inner sep=0pt, minimum size=7pt},
    wrist/.style = {circle, draw=black, fill=gray!80, inner sep=0pt, minimum size=8pt},
    bone/.style = {draw=black!70, thick}
]

% Wrist
\node (J0) [wrist, label=below:{\scriptsize 0: Wrist}] at (0, 0) {};

% Thumb (Spread to Left)
\node (J1) [joint] at (-1.0, 0.6) {};
\node (J2) [joint] at (-1.8, 1.2) {};
\node (J3) [joint] at (-2.4, 1.8) {};
\node (J4) [tip, label=left:{\scriptsize 4: Thumb Tip}] at (-2.8, 2.4) {};

% Index Finger
\node (J5) [joint] at (-1.0, 2.0) {};
\node (J6) [joint] at (-1.2, 2.9) {};
\node (J7) [joint] at (-1.35, 3.6) {};
\node (J8) [tip, label=above left:{\scriptsize 8: Index Tip}] at (-1.45, 4.3) {};

% Middle Finger
\node (J9) [joint] at (0.0, 2.2) {};
\node (J10) [joint] at (0.0, 3.2) {};
\node (J11) [joint] at (0.0, 4.0) {};
\node (J12) [tip, label=above:{\scriptsize 12: Middle Tip}] at (0.0, 4.8) {};

% Ring Finger
\node (J13) [joint] at (1.0, 2.0) {};
\node (J14) [joint] at (1.2, 2.9) {};
\node (J15) [joint] at (1.35, 3.7) {};
\node (J16) [tip, label=above right:{\scriptsize 16: Ring Tip}] at (1.45, 4.4) {};

% Pinky Finger (Spread to Right)
\node (J17) [joint] at (1.8, 1.6) {};
\node (J18) [joint] at (2.1, 2.3) {};
\node (J19) [joint] at (2.3, 2.9) {};
\node (J20) [tip, label=right:{\scriptsize 20: Pinky Tip}] at (2.5, 3.5) {};

% Bones Connections
\draw [bone] (J0) -- (J1) -- (J2) -- (J3) -- (J4);
\draw [bone] (J0) -- (J5) -- (J6) -- (J7) -- (J8);
\draw [bone] (J0) -- (J9) -- (J10) -- (J11) -- (J12);
\draw [bone] (J0) -- (J13) -- (J14) -- (J15) -- (J16);
\draw [bone] (J0) -- (J17) -- (J18) -- (J19) -- (J20);
\draw [bone] (J5) -- (J9) -- (J13) -- (J17);

\end{tikzpicture}
\caption{MediaPipe 21 hand landmark anatomical topology.}
\label{fig:mediapipe_hand_topology}
\end{figure}"""

# Figure 2.4 (Both diagrams in section 2.5)
fig24a_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{Data creation, feature engineering, and model training workflow pipeline\.\}.*?\\end\{figure\}', c2, re.DOTALL).group(0)

fig24a_new = r"""\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    block/.style = {rectangle, draw=blue!80!black, fill=blue!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\footnotesize, text width=3.3cm, minimum height=1.1cm, inner sep=4pt},
    arrow/.style = {draw=blue!80!black, ->, >=Stealth, thick}
]

% Row 1: Steps 1 to 4
\node (step1) [block, at={(0,0)}] {\textbf{1. Video Capture}\\Raw RGB Video Feed};
\node (step2) [block, at={(3.7,0)}] {\textbf{2. 12 FPS Resampling}\\\texttt{resample\_12fps.py}};
\node (step3) [block, at={(7.4,0)}] {\textbf{3. MediaPipe Pose}\\21 Keypoint Extraction};
\node (step4) [block, at={(11.1,0)}] {\textbf{4. Scale Normalization}\\\texttt{normalize\_landmarks.py}\\(Unitless $L_{\text{hand}}$, No Filter)};

% Row 2: Steps 5 to 8
\node (step5) [block, at={(11.1,-2.0)}] {\textbf{5. Velocity Derivation}\\\texttt{calculate\_velocities.py}\\Spatial Joint Derivation};
\node (step6) [block, at={(7.4,-2.0)}] {\textbf{6. Temporal Windowing}\\\texttt{create\_windows.py}\\(5 Frames, 2-Frame Overlap)};
\node (step7) [block, at={(3.7,-2.0)}] {\textbf{7. Quality Filtering}\\\texttt{filter\_window\_quality.py}\\Window Validation};
\node (step8) [block, at={(0,-2.0)}] {\textbf{8. PyTorch LSTM Training}\\\texttt{run\_all.py} Benchmark\\\texttt{best\_finger\_touch\_lstm.pth}};

% Clean Non-Intersecting Connections
\draw [arrow] (step1) -- (step2);
\draw [arrow] (step2) -- (step3);
\draw [arrow] (step3) -- (step4);
\draw [arrow] (step4) -- (step5);
\draw [arrow] (step5) -- (step6);
\draw [arrow] (step6) -- (step7);
\draw [arrow] (step7) -- (step8);

\end{tikzpicture}%
}
\caption{Data creation, feature engineering, and model training workflow pipeline.}
\label{fig:workflow_dataset_pipeline}
\end{figure}"""

fig24b_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{Real-time runtime vision, touch classification, homography mapping, and key execution workflow\.\}.*?\\end\{figure\}', c2, re.DOTALL).group(0)

fig24b_new = r"""\begin{figure}[htbp]
\centering
\resizebox{0.95\textwidth}{!}{%
\begin{tikzpicture}[
    block/.style = {rectangle, draw=green!60!black, fill=green!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\footnotesize, text width=3.3cm, minimum height=1.1cm, inner sep=4pt},
    decision/.style = {diamond, draw=orange!80!black, fill=orange!10!white, thick, align=center, font=\sffamily\scriptsize, inner sep=2pt, aspect=1.8},
    arrow/.style = {draw=green!70!black, ->, >=Stealth, thick}
]

% Top Branch: Camera & Homography
\node (cam) [block, at={(0,0)}] {\textbf{Live RGB Camera Feed}\\\texttt{camera\_thread.py}};
\node (april) [block, at={(4.0,0)}] {\textbf{AprilTag Tracker}\\Extract Quad Corner Points};
\node (homog) [block, at={(8.0,0)}] {\textbf{SVD Homography Engine}\\Compute Matrix $H \in \mathbb{R}^{3 \times 3}$};

% Middle Branch: Pose & Touch Detection
\node (mp) [block, at={(0,-2.0)}] {\textbf{MediaPipe Hand Pose}\\21 Joint Keypoint Tracker};
\node (norm) [block, at={(4.0,-2.0)}] {\textbf{Hand Scale Normalization}\\Unitless $L_{\text{hand}}$ (No 1€ Filter)};
\node (lstm) [block, at={(8.0,-2.0)}] {\textbf{PyTorch LSTM Evaluator}\\5-Frame Window Classifier};

% Bottom Branch: Decision & Action Dispatch
\node (touch_dec) [decision, at={(8.0,-4.2)}] {Touch Event\\Confirmed?};
\node (map_coord) [block, at={(4.0,-4.2)}] {\textbf{Planar Point Mapping}\\$\mathbf{P}_{\text{XML}} = H \cdot \mathbf{P}_{\text{pixel}}$};
\node (lookup) [block, at={(0,-4.2)}] {\textbf{XML Key Lookup}\\Execute OS Action / Shell};

% Clean Connections
\draw [arrow] (cam) -- (april);
\draw [arrow] (april) -- (homog);
\draw [arrow] (cam) -- (mp);
\draw [arrow] (mp) -- (norm);
\draw [arrow] (norm) -- (lstm);
\draw [arrow] (lstm) -- (touch_dec);

\draw [arrow] (touch_dec) -- node[above, font=\sffamily\tiny]{\textbf{Yes ($p > 0.90$)}} (map_coord);
\draw [arrow] (touch_dec.east) -- ++(0.8,0) |- node[right, pos=0.25, font=\sffamily\tiny]{\textbf{No (Hovering)}} (lstm.east);

\draw [arrow] (homog.south) -- (lstm.north -| homog.south) -- (map_coord.north -| homog.south) -- (map_coord.north);
\draw [arrow] (map_coord) -- (lookup);

\end{tikzpicture}%
}
\caption{Real-time runtime vision, touch classification, homography mapping, and key execution workflow.}
\label{fig:workflow_runtime_pipeline}
\end{figure}"""

c2 = c2.replace(fig21_old, fig21_new)
c2 = c2.replace(fig22_old, fig22_new)
c2 = c2.replace(fig24a_old, fig24a_new)
c2 = c2.replace(fig24b_old, fig24b_new)

with open('chapters/chapter02.tex', 'w') as f:
    f.write(c2)
print("Updated Figure 2.1, Figure 2.2, Figure 2.4a, Figure 2.4b in chapter02.tex")


# ---------------------------------------------------------
# 3. Update Figure 3.1 in chapters/chapter03.tex
# ---------------------------------------------------------
with open('chapters/chapter03.tex', 'r') as f:
    c3 = f.read()

fig31_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{Design Science Research Methodology \(DSRM\) execution framework\.\}.*?\\end\{figure\}', c3, re.DOTALL).group(0)

fig31_new = r"""\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    phase/.style = {rectangle, draw=blue!80!black, fill=blue!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\footnotesize, minimum height=1.1cm, text width=2.4cm, inner sep=4pt},
    arrow/.style = {draw=blue!80!black, ->, >=Stealth, thick}
]

\node (p1) [phase, at={(0,0)}] {Phase 1\\Problem\\Identification};
\node (p2) [phase, at={(2.9,0)}] {Phase 2\\Define Objective\\of Solution};
\node (p3) [phase, at={(5.8,0)}] {Phase 3\\Design \&\\Development};
\node (p4) [phase, at={(8.7,0)}] {Phase 4\\Demonstration};
\node (p5) [phase, at={(11.6,0)}] {Phase 5\\Evaluation};
\node (p6) [phase, at={(14.5,0)}] {Phase 6\\Communication};

\draw [arrow] (p1) -- (p2);
\draw [arrow] (p2) -- (p3);
\draw [arrow] (p3) -- (p4);
\draw [arrow] (p4) -- (p5);
\draw [arrow] (p5) -- (p6);

% Clean Orthogonal Loop Arrow (No Overlap)
\draw [arrow] (p5.south) -- ++(0,-0.6) -| node[below, font=\sffamily\scriptsize, pos=0.5]{\textbf{Iterative Refinement Loop}} (p3.south);

\end{tikzpicture}%
}
\caption{Design Science Research Methodology (DSRM) execution framework.}
\label{fig:dsrm_process_flow}
\end{figure}"""

c3 = c3.replace(fig31_old, fig31_new)
with open('chapters/chapter03.tex', 'w') as f:
    f.write(c3)
print("Updated Figure 3.1 in chapter03.tex")

