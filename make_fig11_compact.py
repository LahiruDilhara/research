import re

with open('chapters/chapter01.tex', 'r') as f:
    c1 = f.read()

fig11_old = re.search(r'\\begin\{figure\}\[H\].*?\\caption\{Rich Picture of the Proposed Solution and Operational Pipeline Architecture\.\}.*?\\end\{figure\}', c1, re.DOTALL).group(0)

fig11_new = r"""\begin{figure}[htbp]
\centering
\resizebox{0.78\textwidth}{!}{%
\begin{tikzpicture}[
    box/.style={draw=blue!70!black, rectangle, rounded corners=3pt, text width=2.7cm, minimum height=0.85cm, align=center, fill=blue!8!white, font=\sffamily\tiny, thick},
    process/.style={draw=teal!70!black, rectangle, rounded corners=3pt, text width=2.7cm, minimum height=0.85cm, align=center, fill=teal!8!white, font=\sffamily\tiny, thick},
    model/.style={draw=orange!80!black, rectangle, rounded corners=3pt, text width=2.7cm, minimum height=0.85cm, align=center, fill=orange!10!white, font=\sffamily\tiny, thick},
    output/.style={draw=purple!70!black, rectangle, rounded corners=3pt, text width=2.7cm, minimum height=0.85cm, align=center, fill=purple!8!white, font=\sffamily\tiny, thick},
    arrow/.style={-Stealth, thick, draw=blue!80!black}
]

% Phase 1: Designer Suite (Row 1)
\node [box, at={(0, 4.2)}] (designer) {\textbf{PySide6 Layout GUI}\\Layout \& Anchor Design};
\node [output, at={(3.3, 4.2)}] (xml) {\textbf{XML Layout File}\\Key Bounds \& Actions};
\node [output, at={(6.6, 4.2)}] (pdf) {\textbf{Printable Layout PDF}\\Embedded AprilTag Anchors};

% Phase 2: Physical Setup & Camera (Row 2)
\node [process, at={(6.6, 2.8)}] (paper) {\textbf{Physical Paper Setup}\\Printed Sheet on Workspace};
\node [process, at={(3.3, 2.8)}] (camera) {\textbf{Monocular RGB Camera}\\Live Video Stream Capture};

% Phase 3: Vision Engine & Pose Normalization (Row 3)
\node [box, at={(0, 1.4)}] (apriltag) {\textbf{AprilTag Tracker}\\Sub-Pixel Quad Localization};
\node [model, at={(3.3, 1.4)}] (homography) {\textbf{Homography Matrix ($H$)}\\$3\times3$ Planar Transformation};

\node [box, at={(0, 0.0)}] (mediapipe) {\textbf{MediaPipe Landmarker}\\21 Hand Skeleton Keypoints};
\node [process, at={(3.3, 0.0)}] (filter) {\textbf{Hand Scale Normalization}\\Unitless $L_{\text{hand}}$ (No Filter)};
\node [process, at={(6.6, 0.0)}] (window) {\textbf{Temporal Windowing}\\5 Frames, 2-Frame Overlap};

% Phase 4: Model Classifier & System Action Dispatcher (Row 4)
\node [model, at={(6.6, -1.4)}] (lstm) {\textbf{PyTorch LSTM Model}\\Touch Event Classifier};
\node [process, at={(3.3, -1.4)}] (mapping) {\textbf{Planar Point Mapping}\\$P_{\text{XML}} = H \cdot P_{\text{pixel}}$};
\node [output, at={(0, -1.4)}] (execution) {\textbf{System Key Execution}\\OS Keystroke / Shell Action};

% Connections
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
\draw [arrow] (xml.west) -- ++(-0.5, 0) |- (execution.north);

\end{tikzpicture}%
}
\caption{Rich Picture of the Proposed Solution and Operational Pipeline Architecture.}
\label{fig:rich_picture_diagram}
\end{figure}"""

c1 = c1.replace(fig11_old, fig11_new)
with open('chapters/chapter01.tex', 'w') as f:
    f.write(c1)

print("Updated Figure 1.1 to be compact and elegant in chapter01.tex!")
