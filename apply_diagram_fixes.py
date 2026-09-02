import re

# 1. Update Figure 2.1 in chapters/chapter02.tex
with open('chapters/chapter02.tex', 'r') as f:
    c2 = f.read()

fig21_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{Conceptual taxonomy and structural organization of the literature review\.\}.*?\\end\{figure\}', c2, re.DOTALL).group(0)

fig21_new = r"""\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    header/.style = {rectangle, draw=blue!80!black, fill=blue!25!white, thick, rounded corners=5pt, align=center, font=\sffamily\bfseries\small, inner sep=8pt, text width=16cm},
    box/.style = {rectangle, draw=blue!70!black, fill=blue!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\footnotesize, inner sep=6pt, text width=3.8cm, minimum height=1.2cm},
    subbox/.style = {rectangle, draw=gray!70, fill=gray!5!white, rounded corners=3pt, align=center, font=\sffamily\scriptsize, inner sep=5pt, text width=3.8cm},
    arrow/.style = {draw=blue!80!black, ->, >=Stealth, thick}
]

% Root Node
\node (root) [header] {\textbf{Monocular Vision-Based Virtual Keyboard Literature Taxonomy}};

% Level 1 Nodes - Placed in 4 clear non-overlapping columns
\node (l1_domain) [box, at={(-6.3,-1.8)}] {1. HCI \& Input Paradigms\\(\S\ref{sec:lit_domain_overview})};
\node (l1_systems) [box, at={(-2.1,-1.8)}] {2. Sensing Modalities\\(\S\ref{sec:lit_existing_systems})};
\node (l1_tech) [box, at={(2.1,-1.8)}] {3. Technological Analysis\\(\S\ref{sec:lit_technological_analysis})};
\node (l1_reflect) [box, at={(6.3,-1.8)}] {4. Research Synthesis\\(\S\ref{sec:lit_reflection})};

% Connections to Level 1
\draw [arrow] (root.south) -- (l1_domain.north);
\draw [arrow] (root.south) -- (l1_systems.north);
\draw [arrow] (root.south) -- (l1_tech.north);
\draw [arrow] (root.south) -- (l1_reflect.north);

% Column 1: Domain
\node (l2_history) [subbox, below=0.5cm of l1_domain] {Evolution of Input Hardware\\\cite{ReviewVirtualKeyboard2020, Lee2022Virtual}};
\node (l2_drivers) [subbox, below=0.3cm of l2_history] {Paper Rationale \& Ergonomics\\\cite{Zhang2001Visual, Srivastava2012RealTime, Khare2019QWERTY}};

% Column 2: Systems
\node (l2_proj) [subbox, below=0.5cm of l1_systems] {Projection \& IR Sensors\\\cite{Cheng2015Fingertip, Kudale2016RealTime}};
\node (l2_shadow) [subbox, below=0.3cm of l2_proj] {Shadow Analysis Methods\\\cite{Thomas2013Camera, Posner2012Single, Yue2014Blind}};
\node (l2_depth) [subbox, below=0.3cm of l2_shadow] {RGB-D / ToF Depth Sensing\\\cite{Lee2019Virtual, Toshpulatov2024RealTime}};
\node (l2_air) [subbox, below=0.3cm of l2_depth] {3D Mid-Air Typing / VR\\\cite{Boletsis2019TextInput, Enkhbat2020HandKey, Lee2022Virtual, Yoo2026WordLevel}};
\node (l2_paper) [subbox, below=0.3cm of l2_air] {Paper / Monocular RGB\\\cite{Zhang2001Visual, Srivastava2012RealTime, Khare2019QWERTY, Maman2023TypeNet}};
\node (l2_attack) [subbox, below=0.3cm of l2_paper] {Video Attack Side-Channels\\\cite{Yue2014Blind, Yang2022Towards}};

% Column 3: Tech Analysis
\node (l2_pose) [subbox, below=0.5cm of l1_tech] {MediaPipe Hand Landmarker\\\cite{Zhang2020MediaPipe, GilMartin2023Hand, Andriyanov2025Improving}};
\node (l2_scale) [subbox, below=0.3cm of l2_pose] {Hand Scale Normalization\\\cite{Doan2022Efficient, Kumar2026Hybrid}};
\node (l2_fiducial) [subbox, below=0.3cm of l2_scale] {AprilTag Homography $H$\\\cite{Wang2016AprilTag2, Kallwies2020Determining, Pirchheim2011Homography}};
\node (l2_deep) [subbox, below=0.3cm of l2_fiducial] {PyTorch LSTM Classifiers\\\cite{Hochreiter1997LSTM, Nunez2018Convolutional, Gammulle2021TMMF}};

% Column 4: Reflection
\node (l2_gap) [subbox, below=0.5cm of l1_reflect] {Prior Art Limitations\\\cite{Thomas2013Camera, Lee2019Virtual, Maman2023TypeNet}};
\node (l2_solution) [subbox, below=0.3cm of l2_gap] {Proposed Architecture\\Synthesis (\S\ref{sec:lit_reflection})};

% Connections to Level 2
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

c2 = c2.replace(fig21_old, fig21_new)
with open('chapters/chapter02.tex', 'w') as f:
    f.write(c2)
print("Updated Figure 2.1 in chapter02.tex")


# 2. Update Figure 4.1 and Figure 4.9 in chapters/chapter04.tex
with open('chapters/chapter04.tex', 'r') as f:
    c4 = f.read()

fig41_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{Stakeholder Power-Interest Matrix governing project requirement prioritization\.\}.*?\\end\{figure\}', c4, re.DOTALL).group(0)

fig41_new = r"""\begin{figure}[htbp]
\centering
\resizebox{0.85\textwidth}{!}{%
\begin{tikzpicture}[
    box/.style = {rectangle, draw=blue!80!black, fill=blue!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\footnotesize, text width=3.8cm, minimum height=2.2cm, inner sep=4pt},
    gridline/.style = {draw=gray!60, dashed, ultra thick}
]

% Axes
\draw [->, >=Stealth, ultra thick] (0,0) -- (9.5,0) node[right, font=\sffamily\bfseries\small] {Interest};
\draw [->, >=Stealth, ultra thick] (0,0) -- (0,7.0) node[above, font=\sffamily\bfseries\small] {Power / Influence};

% Quadrant Lines
\draw [gridline] (4.75, 0) -- (4.75, 6.5);
\draw [gridline] (0, 3.25) -- (9.0, 3.25);

% Quadrant Labels
\node (q1) [box] at (2.375, 4.875) {\textbf{Keep Satisfied}\\[2pt]\scriptsize Clinical \& Healthcare Staff\\\scriptsize Cleanroom Administrators};
\node (q2) [box, fill=green!15!white, draw=green!80!black] at (7.125, 4.875) {\textbf{Manage Closely (Key Players)}\\[2pt]\scriptsize System Engineers \& Admins\\\scriptsize Domain Macro Specialists};
\node (q3) [box] at (2.375, 1.625) {\textbf{Monitor (Minimum Effort)}\\[2pt]\scriptsize General Public Kiosk Users\\\scriptsize Hardware Suppliers};
\node (q4) [box, fill=orange!15!white, draw=orange!80!black] at (7.125, 1.625) {\textbf{Keep Informed}\\[2pt]\scriptsize Primary Typists \& Accessibility\\\scriptsize HCI Academic Community};

\end{tikzpicture}%
}
\caption{Stakeholder Power-Interest Matrix governing project requirement prioritization.}
\label{fig:stakeholder_power_interest}
\end{figure}"""

fig49_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{Proposed System Architecture Framework Diagram highlighting Application 1 and Application 2 synchronized workflows\.\}.*?\\end\{figure\}', c4, re.DOTALL).group(0)

fig49_new = r"""\begin{figure}[htbp]
\centering
\resizebox{0.95\textwidth}{!}{%
\begin{tikzpicture}[
    appbox/.style = {rectangle, draw=blue!80!black, fill=blue!20!white, thick, rounded corners=5pt, align=center, font=\sffamily\bfseries\small, text width=14cm, inner sep=6pt},
    subbox/.style = {rectangle, draw=blue!60!black, fill=blue!5!white, rounded corners=3pt, align=left, font=\sffamily\scriptsize, text width=13.2cm, inner sep=6pt},
    artbox/.style = {rectangle, draw=orange!80!black, fill=orange!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\scriptsize, text width=5.8cm, minimum height=1.3cm},
    compbox/.style = {rectangle, draw=green!70!black, fill=green!5!white, thick, rounded corners=4pt, align=left, font=\sffamily\scriptsize, text width=6.2cm, minimum height=2.4cm, inner sep=6pt},
    arrow/.style = {draw=blue!80!black, ->, >=Stealth, thick}
]

% Tier 1: Application 1 Container
\node (app1_hdr) [appbox] {\textbf{APPLICATION 1: LAYOUT DESIGNER SUITE (\texttt{designer/designer\_app.py})}};
\node (app1_body) [subbox, below=0.2cm of app1_hdr] {
\textbf{PySide6 Desktop Layout Designer GUI}\\
$\bullet$ Interactive drag-and-drop key creation \& grid snapping ($5.0\text{ mm}$ grid)\\
$\bullet$ Automated AprilTag visual anchor placement along paper border margins\\
$\bullet$ Button ID configuration \& dual artifact export engine
};

% Tier 2: Synchronized Artifacts
\node (art_pdf) [artbox, at={(-3.6, -3.8)}] {Printable Layout PDF Sheet\\{\normalfont\tiny (Embedded AprilTag Fiducial Anchors)}};
\node (art_xml) [artbox, at={(3.6, -3.8)}] {Layout Specification XML File\\{\normalfont\tiny (Key Bounding Boxes \& Action Mapping Schemas)}};

% Tier 3: Application 2 Container
\node (app2_hdr) [appbox, at={(0, -6.0)}] {\textbf{APPLICATION 2: RUNTIME VIRTUAL KEYBOARD ENGINE \& COMMAND MAPPER}};

% App 2 Components
\node (comp_a) [compbox, at={(-3.6, -8.2)}] {
\textbf{Component A: Setup \& Action Mapper GUI}\\
$\bullet$ Input Camera Feed Selector \& Live Preview\\
$\bullet$ Layout XML Loader \& Interactive Binding Table\\
$\bullet$ Map Key IDs to OS Keystrokes / System Commands\\
$\bullet$ Save / Load JSON Profile Configurations
};

\node (comp_b) [compbox, at={(3.6, -8.2)}] {
\textbf{Component B: Background Runtime Watch Engine}\\
$\bullet$ AprilTag Tracker $\to$ Planar Homography Matrix $H$\\
$\bullet$ MediaPipe 21 Pose $\to$ Scale Norm ($L_{\text{hand}}$, No 1€ Filter)\\
$\bullet$ 5-Frame Window $\to$ PyTorch LSTM Classifier\\
$\bullet$ Planar Coordinate Mapping: $P_{\text{XML}} = H \cdot P_{\text{pixel}}$\\
$\bullet$ XML Key Lookup $\to$ Dispatch OS Action / Shell Script
};

% Connecting Arrows
\draw [arrow] (app1_body.south -| art_pdf.north) -- (art_pdf.north);
\draw [arrow] (app1_body.south -| art_xml.north) -- (art_xml.north);

\draw [arrow] (art_pdf.south) -- node[right, font=\sffamily\tiny]{Physical Paper} (app2_hdr.north -| art_pdf.south);
\draw [arrow] (art_xml.south) -- node[left, font=\sffamily\tiny]{XML Schema} (app2_hdr.north -| art_xml.south);

\draw [arrow] (app2_hdr.south -| comp_a.north) -- (comp_a.north);
\draw [arrow] (app2_hdr.south -| comp_b.north) -- (comp_b.north);

\end{tikzpicture}%
}
\caption{Proposed System Architecture Framework Diagram highlighting Application 1 and Application 2 synchronized workflows.}
\label{fig:system_architecture_framework}
\end{figure}"""

c4 = c4.replace(fig41_old, fig41_new)
c4 = c4.replace(fig49_old, fig49_new)

with open('chapters/chapter04.tex', 'w') as f:
    f.write(c4)
print("Updated Figure 4.1 and Figure 4.9 in chapter04.tex")


# 3. Update Figure 5.1 in chapters/chapter05.tex
with open('chapters/chapter05.tex', 'r') as f:
    c5 = f.read()

fig51_old = re.search(r'\\begin\{figure\}\[htbp\].*?\\caption\{End-to-End multi-module dataflow pipeline block diagram\.\}.*?\\end\{figure\}', c5, re.DOTALL).group(0)

fig51_new = r"""\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    node distance = 0.8cm and 0.5cm,
    block/.style = {rectangle, draw=blue!80!black, fill=blue!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\tiny, minimum height=1.1cm, text width=2.4cm, inner sep=4pt},
    artblock/.style = {rectangle, draw=orange!80!black, fill=orange!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\tiny, minimum height=1.1cm, text width=2.4cm, inner sep=4pt},
    engineblock/.style = {rectangle, draw=green!80!black, fill=green!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\tiny, minimum height=1.1cm, text width=2.4cm, inner sep=4pt},
    arrow/.style = {draw=blue!80!black, ->, >=Stealth, thick}
]

% Row 1: Application 1 Layout Designer Suite
\node (b1) [block] {Application 1\\PySide6 Designer\\(\texttt{designer\_app.py})};
\node (b2) [artblock, right=0.6cm of b1] {Printable PDF Sheet\\(AprilTag Anchors)};
\node (b3) [artblock, right=0.6cm of b2] {Layout XML File\\(Bounding Boxes)};

% Row 2: Application 2 Setup GUI & Mapper
\node (b4) [block, below=1.0cm of b1] {Application 2\\Setup \& Mapper GUI\\(Component A)};
\node (b5) [block, right=0.6cm of b4] {Camera Stream\\Selector \& Preview};
\node (b6) [block, right=0.6cm of b5] {Action Mapping\\Table (Keystrokes/Shell)};

% Row 3: Background Vision & Touch Engine Pipeline
\node (b7) [engineblock, below=1.4cm of b4] {Background Camera Thread\\(\texttt{camera\_thread.py})};
\node (b8) [engineblock, right=0.5cm of b7] {MediaPipe 21 Pose \&\\Unitless Scale Norm ($L_{\text{hand}}$)};
\node (b9) [engineblock, right=0.5cm of b8] {5-Frame Window Matrix\\($\mathbf{X}_W \in \mathbb{R}^{5 \times 84}$)};
\node (b10) [engineblock, right=0.5cm of b9] {PyTorch LSTM Classifier\\(\texttt{best\_finger\_touch\_lstm.pth})};

% Row 4: Parallel AprilTag Homography & Final Dispatcher
\node (b11) [engineblock, below=0.9cm of b8] {AprilTag Tracker \& SVD\\Homography Engine ($H$)};
\node (b12) [engineblock, below=0.9cm of b10] {Planar Transformation\\($P_{\text{XML}} = H \cdot P_{\text{pixel}}$) \& Dispatch};

% Arrows
\draw [arrow] (b1) -- (b2);
\draw [arrow] (b1) -- (b3);
\draw [arrow] (b2) -- (b4);
\draw [arrow] (b3) -- (b4);
\draw [arrow] (b4) -- (b5);
\draw [arrow] (b5) -- (b6);

\draw [arrow] (b6) |- (b7);
\draw [arrow] (b7) -- (b8);
\draw [arrow] (b8) -- (b9);
\draw [arrow] (b9) -- (b10);

\draw [arrow] (b7) |- (b11);
\draw [arrow] (b10) -- (b12);
\draw [arrow] (b11) -- (b12);

\end{tikzpicture}%
}
\caption{End-to-End multi-module dataflow pipeline block diagram.}
\label{fig:imp_block_diagram}
\end{figure}"""

c5 = c5.replace(fig51_old, fig51_new)

with open('chapters/chapter05.tex', 'w') as f:
    f.write(c5)
print("Updated Figure 5.1 in chapter05.tex")

