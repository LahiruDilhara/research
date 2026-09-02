import os

latex_text = r"""\chapter{Implementation and Software Design}
\label{ch:implementation}

\section{Chapter Overview}
\label{sec:imp_chapter_overview}
The translation of theoretical computer vision models, projective homography geometry, and sequential deep learning into an operational, low-latency virtual keyboard system requires a multi-layered software engineering architecture. Developing markerless, single-camera human-computer interaction (HCI) interfaces demands robust software design patterns, optimized multi-threaded concurrency, precise mathematical signal processing, and decoupled application suite structures \cite{ReviewVirtualKeyboard2020, Maman2023TypeNet}.

Designing paper-based monocular touch interfaces introduces fundamental engineering challenges. Unlike physical mechanical keyboards or capacitive touchscreens, paper virtual keyboards provide no physical tactile resistance or electrical conductivity changes upon surface contact. Consequently, touch detection must rely entirely on optical computer vision, tracking spatial landmark displacements and kinetic velocity inflections across consecutive video frames \cite{Zhang2020MediaPipe, Saponas2009Demonstrating}. Furthermore, monocular single-camera feeds inherently lack direct 3D depth measurements, creating severe depth ambiguity when a finger hovers millimeters above the paper sheet \cite{Thomas2013Camera}. Addressing these challenges requires integrating computer vision algorithms, scale-invariant feature extraction, temporal deep learning models, and real-time desktop GUI event dispatchers.

To evaluate feasibility while building a robust system, development was executed across isolated experimental partitions before assembling the final end-to-end suite. The software system is structured into two main applications:
\begin{enumerate}
    \item \textbf{Application 1 (Layout Designer Suite \texttt{designer/designer\_app.py}):} An interactive PySide6 desktop GUI application allowing users to create custom key layouts, position button bounding boxes, assign key commands, and export synchronized PDF print sheets (embedded with AprilTag anchors) and layout XML specifications.
    \item \textbf{Application 2 (Runtime Virtual Keyboard Engine \& Command Mapper):} An application suite combining a Setup GUI (Component A) for mapping button IDs to OS keystrokes or shell commands, and a Background Live Watch Engine (Component B) that tracks AprilTags for $3 \times 3$ Homography matrix calculation, extracts MediaPipe hand pose keypoints, scale-normalizes joint coordinates relative to unitless hand length ($L_{\text{hand}}$) without low-pass filtering, evaluates 5-frame sliding windows using PyTorch LSTM model \texttt{best\_finger\_touch\_lstm.pth}, and dispatches mapped OS actions.
\end{enumerate}

This chapter presents the comprehensive implementation and software design details governing the system. Section~\ref{sec:imp_framework_steps} details the algorithmic design steps, workflow block diagrams, formal mathematical pseudocode algorithms, transfer function phase lag derivations, and technology selection justifications. Section~\ref{sec:imp_significant_attempts} details the significantly important implementation attempts across the five isolated research partitions and main application GUIs, providing formatted code snippets, architectural breakdowns, data structure shapes, and execution evidence. Section~\ref{sec:imp_challenges} discusses empirical implementation challenges and technical solutions. Finally, Section~\ref{sec:imp_summary} summarizes the chapter.

\section{Algorithmic Design and Software Engineering Steps}
\label{sec:imp_framework_steps}
The engineering of the virtual keyboard suite followed a disciplined software development process bridging layout design generation, real-time video capture, pose landmark regression, scale normalization, deep neural inference, visual fiducial homography tracking, and system action dispatching.

\subsection{Framework Engineering Workflow Breakdown}
\label{subsec:imp_workflow_breakdown}
The overall system architecture is partitioned into two synchronized application artifacts operating across an end-to-end processing pipeline, as illustrated in the system dataflow block diagram in Figure~\ref{fig:imp_block_diagram}.

\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    node distance = 1.0cm and 0.6cm,
    block/.style = {rectangle, draw=blue!80!black, fill=blue!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\tiny, minimum height=1.0cm, inner sep=5pt, width=2.4cm},
    artblock/.style = {rectangle, draw=orange!80!black, fill=orange!10!white, thick, rounded corners=4pt, align=center, font=\sffamily\bfseries\tiny, minimum height=1.0cm, inner sep=5pt, width=2.4cm},
    arrow/.style = {draw=blue!80!black, ->, >=Stealth, thick}
]

% Row 1: App 1
\node (b1) [block] {Application 1\\PySide6 Designer\\(\texttt{designer\_app.py})};
\node (b2) [artblock, right=0.6cm of b1] {Printable PDF Sheet\\(AprilTag Anchors)};
\node (b3) [artblock, right=0.6cm of b2] {Layout XML File\\(Bounding Boxes)};

% Row 2: App 2 Setup
\node (b4) [block, below=1.2cm of b1] {Application 2\\Setup \& Mapper GUI\\(Component A)};
\node (b5) [block, right=0.6cm of b4] {Camera Stream\\Selector \& Preview};
\node (b6) [block, right=0.6cm of b5] {Action Mapping\\Table (Keystrokes/Shell)};

% Row 3: App 2 Runtime Engine
\node (b7) [block, below=1.2cm of b4] {Background Camera Thread\\(\texttt{camera\_thread.py})};
\node (b8) [block, right=0.6cm of b7] {AprilTag Tracker \& SVD\\Homography Engine ($H$)};
\node (b9) [block, right=0.6cm of b8] {MediaPipe 21 Pose \&\\Unitless Scale Norm ($L_{\text{hand}}$)};
\node (b10) [block, below=0.8cm of b8] {5-Frame Window Matrix\\($\mathbf{X}_W \in \mathbb{R}^{5 \times 84}$)};
\node (b11) [block, right=0.6cm of b10] {PyTorch LSTM Classifier\\(\texttt{best\_finger\_touch\_lstm.pth})};
\node (b12) [block, right=0.6cm of b11] {Planar Transformation\\($P_{\text{XML}} = H \cdot P_{\text{pixel}}$) \& Dispatch};

% Arrows
\draw [arrow] (b1) -- (b2);
\draw [arrow] (b1) -- (b3);
\draw [arrow] (b2) -- (b4);
\draw [arrow] (b3) -- (b4);
\draw [arrow] (b4) -- (b5);
\draw [arrow] (b5) -- (b6);
\draw [arrow] (b6) -- (b7);
\draw [arrow] (b7) -- (b8);
\draw [arrow] (b7) -- (b9);
\draw [arrow] (b9) -- (b10);
\draw [arrow] (b10) -- (b11);
\draw [arrow] (b8) |- (b12);
\draw [arrow] (b11) -- (b12);

\end{tikzpicture}%
}
\caption{End-to-End multi-module dataflow pipeline block diagram.}
\label{fig:imp_block_diagram}
\end{figure}

The dataflow pipeline operates across seven discrete processing sub-phases:
\begin{enumerate}
    \item \textbf{Sub-Phase 1 (Layout Canvas Geometry \& Exporter):} The user interacts with the PySide6 drag-and-drop workspace in Application 1 (\texttt{designer\_app.py}). Key buttons are dynamically created, aligned to an A4 grid, assigned unique string IDs, and bounded by rectangular coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$. The dual exporter engine renders vector PDF print sheets embedded with AprilTag visual anchors and exports structural XML files.
    \item \textbf{Sub-Phase 2 (Action Mapping \& Profile Configuration):} The user launches Application 2 Component A (Setup GUI). The GUI loads the exported layout XML schema, parses key IDs, populates an interactive action mapping table, and allows the user to bind each key ID to a single keystroke (e.g., `'A'`, `'Space'`) or system shell command (e.g., executing scripts). Custom configuration profiles are saved to JSON disk artifacts.
    \item \textbf{Sub-Phase 3 (Multi-Threaded Video Frame Ingestion):} Application 2 Component B launches into background execution. A high-priority background worker thread (\texttt{camera\_thread.py}) captures monocular RGB video frames using OpenCV \texttt{VideoCapture}, pushing raw frame arrays $\mathbf{I}_{\text{frame}} \in \mathbb{R}^{H \times W \times 3}$ into a thread-safe Queue buffer.
    \item \textbf{Sub-Phase 4 (Fiducial Tag Tracking \& SVD Homography Update):} The optical tracking thread pops frames from the queue, converts them to single-channel grayscale, detects AprilTag quads (\texttt{tag36h11} codebook), extracts sub-pixel corner coordinates, constructs the Direct Linear Transformation (DLT) matrix $\mathbf{A}$, and solves for Planar Homography matrix $H \in \mathbb{R}^{3 \times 3}$ via Singular Value Decomposition (SVD).
    \item \textbf{Sub-Phase 5 (Pose Estimation \& Scale Normalization):} Concurrently, MediaPipe Hand Landmarker extracts 21 anatomical 2D hand joint keypoints $\mathbf{P}_i(t) = (x_i(t), y_i(t))$. Keypoints are translated to wrist origin Landmark 0 ($\mathbf{P}_0$) and scaled by unitless hand length $L_{\text{hand}}(t) = \|\mathbf{P}_9(t) - \mathbf{P}_0(t)\|_2$. Numerical velocity vectors $\mathbf{V}_i(t)$ are derived via central difference approximation ($\Delta t = \frac{1}{12}\text{ s}$). Zero low-pass 1€ filtering is applied.
    \item \textbf{Sub-Phase 6 (5-Frame Sliding Window \& PyTorch LSTM Inference):} The feature engineering module buffers 84-dimensional feature vectors $\mathbf{f}_t$ across 5 consecutive frames into temporal matrix $\mathbf{X}_W \in \mathbb{R}^{5 \times 84}$. PyTorch model \texttt{best\_finger\_touch\_lstm.pth} evaluates $\mathbf{X}_W$ in under $2.1\text{ ms}$, outputting per-finger touch probabilities $\mathbf{P}_{\text{touch}} \in [0, 1]^5$. If probability $p_k > 0.90$, digit $k$ is classified as contacting the paper surface.
    \item \textbf{Sub-Phase 7 (Planar Coordinate Transformation \& Action Dispatch):} Upon touch confirmation, active fingertip pixel coordinates $(u_{\text{active}}, v_{\text{active}})$ are transformed through $H$ into XML layout coordinates $(X_{\text{target}}, Y_{\text{target}}) = H \cdot [u, v, 1]^T / w$. The system performs rectangular bounding box collision lookup and dispatches mapped OS keystrokes or shell commands.
\end{enumerate}

\subsection{Mathematical Algorithm Formulations and Formal Pseudocode}
\label{subsec:imp_pseudocode_algorithms}
To formally specify core operational logic, detailed mathematical formulations and formal pseudocode algorithms are presented for primary processing engines.

\subsubsection{Algorithm 1: AprilTag Corner Localization \& SVD Planar Homography Engine}
The homography engine continuously localizes four AprilTag visual marker corners and computes the $3 \times 3$ Planar Homography matrix $H$ mapping camera pixel coordinates $(u, v)$ to layout XML coordinates $(x, y)$.

Given $N \ge 4$ point correspondences $(u_i, v_i) \leftrightarrow (x_i, y_i)$, the Direct Linear Transformation (DLT) formulates linear system $\mathbf{A}_{2N \times 9} \mathbf{h} = \mathbf{0}$, where:
\begin{equation}
\mathbf{A}_i = \begin{bmatrix}
-u_i & -v_i & -1 & 0 & 0 & 0 & u_i x_i & v_i x_i & x_i \\
0 & 0 & 0 & -u_i & -v_i & -1 & u_i y_i & v_i y_i & y_i
\end{bmatrix}
\label{eq:dlt_matrix_app}
\end{equation}

Solving $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ via Singular Value Decomposition (SVD), column vector $\mathbf{h}$ corresponding to the smallest singular value in $\mathbf{V}$ forms homography matrix $H \in \mathbb{R}^{3 \times 3}$.

Algorithm~\ref{alg:homography_svd} details the formal pseudocode for AprilTag corner localization and SVD homography calculation.

\begin{figure}[htbp]
\centering
\begin{tabular}{|p{0.95\textwidth}|}
\hline
\textbf{Algorithm 1: AprilTag Corner Localization and SVD Planar Homography Calculation} \\
\hline
\textbf{Input:} Video frame image $\mathbf{I}_{\text{frame}} \in \mathbb{R}^{H \times W \times 3}$, Target XML anchor coordinates $\{(x_k^{\text{xml}}, y_k^{\text{xml}})\}_{k=1}^M$ \\
\textbf{Output:} Updated Planar Homography matrix $H \in \mathbb{R}^{3 \times 3}$, Sub-pixel tracking error $e_{\text{rms}}$ \\
1: Convert $\mathbf{I}_{\text{frame}}$ to grayscale single-channel image $\mathbf{I}_{\text{gray}}$. \\
2: Detect AprilTag visual quads using AprilTag family \texttt{tag36h11}. \\
3: \textbf{if} Count of detected valid tags $M_{\text{detected}} < 4$ \textbf{then} \\
4: \quad Retain previous valid Homography matrix $H_{\text{prev}}$ from memory buffer. \\
5: \quad \textbf{return} $H_{\text{prev}}$, Status: \texttt{TAG\_OCCLUSION\_WARNING} \\
6: \textbf{end if} \\
7: Extract sub-pixel corner pixel coordinates $\{(u_k, v_k)\}_{k=1}^{4 M_{\text{detected}}}$. \\
8: Construct $2N \times 9$ DLT coefficient matrix $\mathbf{A}$ using Equation~\ref{eq:dlt_matrix_app}. \\
9: Compute Singular Value Decomposition: $\mathbf{U}, \mathbf{\Sigma}, \mathbf{V}^T = \text{SVD}(\mathbf{A})$. \\
10: Extract column vector $\mathbf{h} = \mathbf{V}[:, 8]$ corresponding to minimum singular value $\sigma_{\min}$. \\
11: Reshape $\mathbf{h}$ into $3 \times 3$ matrix $H_{\text{raw}}$ and normalize $H = \frac{1}{h_{33}} H_{\text{raw}}$. \\
12: Compute RMS reprojection error $e_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{i=1}^N \| \mathbf{P}_i^{\text{xml}} - H \cdot \mathbf{P}_i^{\text{pixel}} \|^2}$. \\
13: Update shared runtime memory buffer with $H$. \\
14: \textbf{return} $H, e_{\text{rms}}$ \\
\hline
\end{tabular}
\caption{Formal algorithmic specification for AprilTag corner localization and SVD planar homography calculation.}
\label{alg:homography_svd}
\end{figure}

\subsubsection{Algorithm 2: Unitless Hand Landmark Scale Normalization Engine}
The pose engine extracts 21 anatomical 2D hand joint keypoints $\mathbf{P}_i(t) = (x_i(t), y_i(t))$ per frame and applies unitless hand-length scale normalization.

Wrist Landmark 0 is set as origin $\mathbf{P}_0(t)$. Hand length $L_{\text{hand}}(t)$ is derived between wrist Landmark 0 and middle MCP Landmark 9:
\begin{equation}
L_{\text{hand}}(t) = \sqrt{(x_9(t) - x_0(t))^2 + (y_9(t) - y_0(t))^2}
\label{eq:hand_length_app}
\end{equation}

Unitless scale-normalized joint coordinates $\mathbf{P}_i^{\text{norm}}(t)$ are calculated:
\begin{equation}
\mathbf{P}_i^{\text{norm}}(t) = \left( \frac{x_i(t) - x_0(t)}{L_{\text{hand}}(t)}, \frac{y_i(t) - y_0(t)}{L_{\text{hand}}(t)} \right)
\label{eq:norm_coords_app}
\end{equation}

First-order spatial velocity vectors $\mathbf{V}_i(t) = (v_{x,i}(t), v_{y,i}(t))$ are derived across consecutive frames at $\Delta t = \frac{1}{12}\text{ s}$:
\begin{equation}
v_{x,i}(t) = \frac{x_i^{\text{norm}}(t) - x_i^{\text{norm}}(t-1)}{\Delta t}, \quad
v_{y,i}(t) = \frac{y_i^{\text{norm}}(t) - y_i^{\text{norm}}(t-1)}{\Delta t}
\label{eq:velocity_app}
\end{equation}

In strict adherence to system technical rules, 1€ low-pass filtering and temporal smoothing were strictly excluded to preserve raw kinetic surface impact deceleration spikes.

Algorithm~\ref{alg:scale_normalization} details the formal pseudocode for hand landmark scale normalization and velocity derivation.

\begin{figure}[htbp]
\centering
\begin{tabular}{|p{0.95\textwidth}|}
\hline
\textbf{Algorithm 2: Unitless Hand Landmark Scale Normalization and Velocity Derivation} \\
\hline
\textbf{Input:} Raw MediaPipe 21 landmark 2D coordinates $\{\mathbf{P}_i(t) = (x_i, y_i)\}_{i=0}^{20}$, Previous frame keypoints $\{\mathbf{P}_i(t-1)\}_{i=0}^{20}$, Frame time step $\Delta t = \frac{1}{12}\text{ s}$ \\
\textbf{Output:} 84-Dimensional per-frame feature vector $\mathbf{f}_t \in \mathbb{R}^{84}$ \\
1: Extract wrist origin coordinates $\mathbf{P}_0(t) = (x_0(t), y_0(t))$. \\
2: Extract middle MCP landmark coordinates $\mathbf{P}_9(t) = (x_9(t), y_9(t))$. \\
3: Compute unitless hand length scale factor: $L_{\text{hand}}(t) = \sqrt{(x_9(t) - x_0(t))^2 + (y_9(t) - y_0(t))^2}$. \\
4: \textbf{if} $L_{\text{hand}}(t) < \epsilon$ ($< 10$ pixels) \textbf{then} \\
5: \quad \textbf{return} Status: \texttt{INVALID\_HAND\_SCALE\_ERROR} \\
6: \textbf{end if} \\
7: Initialize empty feature vector array $\mathbf{f}_t \in \mathbb{R}^{84}$. \\
8: \textbf{for} each landmark $i \in \{0, 1, \dots, 20\}$ \textbf{do} \\
9: \quad Calculate normalized position: $x_i^{\text{norm}}(t) = \frac{x_i(t) - x_0(t)}{L_{\text{hand}}(t)}$, \quad $y_i^{\text{norm}}(t) = \frac{y_i(t) - y_0(t)}{L_{\text{hand}}(t)}$. \\
10: \quad Calculate numerical velocity: $v_{x,i}(t) = \frac{x_i^{\text{norm}}(t) - x_i^{\text{norm}}(t-1)}{\Delta t}$, \quad $v_{y,i}(t) = \frac{y_i^{\text{norm}}(t) - y_i^{\text{norm}}(t-1)}{\Delta t}$. \\
11: \quad Store $(x_i^{\text{norm}}, y_i^{\text{norm}}, v_{x,i}, v_{y,i})$ into feature vector $\mathbf{f}_t$ at indices $[4i : 4i+4]$. \\
12: \textbf{end for} \\
13: \textbf{CRITICAL OVERRIDE check:} Confirm zero 1€ low-pass filtering applied. \\
14: \textbf{return} $\mathbf{f}_t \in \mathbb{R}^{84}$ \\
\hline
\end{tabular}
\caption{Formal algorithmic specification for unitless hand landmark scale normalization.}
\label{alg:scale_normalization}
\end{figure}

\subsubsection{Algorithm 3: 5-Frame Sliding Window Assembly \& PyTorch LSTM Touch Inference}
The temporal inference engine concatenates 84-dimensional feature vectors $\mathbf{f}_t$ across 5 consecutive frames into input matrix $\mathbf{X}_W \in \mathbb{R}^{5 \times 84}$.

The PyTorch uni-directional LSTM evaluates input matrix $\mathbf{X}_W$:
\begin{equation}
\mathbf{h}_t, \mathbf{c}_t = \text{LSTM}(\mathbf{X}_W, \mathbf{h}_{t-1}, \mathbf{c}_{t-1})
\label{eq:lstm_forward_app}
\end{equation}

A fully connected dense layer with sigmoid activation predicts binary touch probability vector $\mathbf{P}_{\text{touch}} = [p_{\text{thumb}}, p_{\text{index}}, p_{\text{middle}}, p_{\text{ring}}, p_{\text{pinky}}] \in [0, 1]^5$. If probability $p_k > 0.90$, digit $k$ is classified as establishing paper surface contact.

Algorithm~\ref{alg:lstm_touch_inference} details the formal pseudocode for 5-frame sliding window assembly and PyTorch LSTM touch inference.

\begin{figure}[htbp]
\centering
\begin{tabular}{|p{0.95\textwidth}|}
\hline
\textbf{Algorithm 3: 5-Frame Sliding Window Assembly and PyTorch LSTM Touch Inference} \\
\hline
\textbf{Input:} Per-frame feature vector $\mathbf{f}_t \in \mathbb{R}^{84}$, Sliding window Queue buffer $\mathbf{Q}_{\text{win}}$, PyTorch model instance \texttt{best\_finger\_touch\_lstm.pth}, Detection threshold $\tau_{\text{touch}} = 0.90$ \\
\textbf{Output:} Active finger touch status array $\mathbf{S}_{\text{touch}} \in \{0, 1\}^5$, Active fingertip landmark index $k_{\text{active}}$ \\
1: Append current feature vector $\mathbf{f}_t$ to sliding queue $\mathbf{Q}_{\text{win}}$. \\
2: \textbf{if} Length of $\mathbf{Q}_{\text{win}} < 5$ frames \textbf{then} \\
3: \quad \textbf{return} $\mathbf{S}_{\text{touch}} = [0,0,0,0,0]$, Status: \texttt{WINDOW\_BUFFERING} \\
4: \textbf{end if} \\
5: Construct sliding window input matrix $\mathbf{X}_W = [\mathbf{f}_{t-4}, \mathbf{f}_{t-3}, \mathbf{f}_{t-2}, \mathbf{f}_{t-1}, \mathbf{f}_t]^T \in \mathbb{R}^{5 \times 84}$. \\
6: Advance sliding queue by stride of 3 frames (retaining 2-frame overlap). \\
7: Convert $\mathbf{X}_W$ to PyTorch Tensor: $\mathbf{T}_{\text{input}} = \text{torch.tensor}(\mathbf{X}_W, \text{dtype}=\text{float32}).\text{unsqueeze}(0)$. \\
8: Execute PyTorch model forward pass in evaluation mode (\texttt{with torch.no\_grad():}): \\
9: \quad $\mathbf{y}_{\text{logits}} = \text{Model}(\mathbf{T}_{\text{input}})$. \\
10: \quad $\mathbf{P}_{\text{touch}} = \text{sigmoid}(\mathbf{y}_{\text{logits}}) \in [0, 1]^5$. \\
11: Initialize binary status vector $\mathbf{S}_{\text{touch}} = [0, 0, 0, 0, 0]$. \\
12: \textbf{for} each finger digit $k \in \{0, 1, 2, 3, 4\}$ \textbf{do} \\
13: \quad \textbf{if} $P_{\text{touch}}[k] > \tau_{\text{touch}}$ \textbf{then} \\
14: \qquad Set $\mathbf{S}_{\text{touch}}[k] = 1$ (Touch Confirmed). \\
15: \qquad Set active fingertip landmark index: $k_{\text{active}} \in \{4, 8, 12, 16, 20\}$. \\
16: \quad \textbf{end if} \\
17: \textbf{end for} \\
18: \textbf{return} $\mathbf{S}_{\text{touch}}, k_{\text{active}}$ \\
\hline
\end{tabular}
\caption{Formal algorithmic specification for 5-frame sliding window assembly and PyTorch LSTM touch inference.}
\label{alg:lstm_touch_inference}
\end{figure}

\subsubsection{Algorithm 4: Planar Coordinate Mapping \& Action Execution Engine}
When Algorithm~\ref{alg:lstm_touch_inference} confirms an active finger touch event, Algorithm~\ref{alg:coordinate_mapping} maps active fingertip pixel coordinates $(u_{\text{active}}, v_{\text{active}})$ through Homography matrix $H$ into layout XML coordinate space:
\begin{equation}
\begin{bmatrix} x_{\text{xml}} \\ y_{\text{xml}} \\ w \end{bmatrix} = H \cdot \begin{bmatrix} u_{\text{active}} \\ v_{\text{active}} \\ 1 \end{bmatrix}, \quad
X_{\text{target}} = \frac{x_{\text{xml}}}{w}, \quad Y_{\text{target}} = \frac{y_{\text{xml}}}{w}
\label{eq:homog_mapping_app}
\end{equation}

System queries target XML bounding boxes. If $(X_{\text{target}}, Y_{\text{target}}) \in [x_{\min}^j, x_{\max}^j] \times [y_{\min}^j, y_{\max}^j]$, button $j$ is actuated, executing mapped keystrokes or OS shell scripts.

Algorithm~\ref{alg:coordinate_mapping} details formal pseudocode for planar coordinate mapping and action execution.

\begin{figure}[htbp]
\centering
\begin{tabular}{|p{0.95\textwidth}|}
\hline
\textbf{Algorithm 4: Planar Coordinate Mapping and Action Execution Engine} \\
\hline
\textbf{Input:} Active fingertip pixel coordinate $\mathbf{P}_{\text{pixel}} = (u, v)$, Homography matrix $H \in \mathbb{R}^{3 \times 3}$, Target XML key bounding boxes $\{B_j = (x_{\min}^j, y_{\min}^j, x_{\max}^j, y_{\max}^j)\}_{j=1}^K$, Action mapping dictionary $\mathbf{M}_{\text{action}}$ \\
\textbf{Output:} Executed Action Status, Hit Key ID $j_{\text{hit}}$ \\
1: Compute homogeneous target vector: $[x_{\text{temp}}, y_{\text{temp}}, w]^T = H \cdot [u, v, 1]^T$. \\
2: \textbf{if} $|w| < 10^{-6}$ \textbf{then} \\
3: \quad \textbf{return} Status: \texttt{HOMOGRAPHY\_SINGULARITY\_ERROR} \\
4: \textbf{end if} \\
5: Dehomogenize XML planar coordinates: $X_{\text{xml}} = \frac{x_{\text{temp}}}{w}$, \quad $Y_{\text{xml}} = \frac{y_{\text{temp}}}{w}$. \\
6: Initialize $j_{\text{hit}} = \text{None}$. \\
7: \textbf{for} each key button $j \in \{1, 2, \dots, K\}$ in XML layout \textbf{do} \\
8: \quad \textbf{if} $x_{\min}^j \le X_{\text{xml}} \le x_{\max}^j$ \textbf{and} $y_{\min}^j \le Y_{\text{xml}} \le y_{\max}^j$ \textbf{then} \\
9: \qquad Set $j_{\text{hit}} = j$. \\
10: \qquad Extract action binding: $A_j = \mathbf{M}_{\text{action}}[j]$. \\
11: \qquad \textbf{if} $A_j$ is \texttt{KeystrokeAction} \textbf{then} \\
12: \qquad \quad Inject system virtual key press event via OS API (e.g., `'Ctrl+C'`). \\
13: \qquad \textbf{else if} $A_j$ is \texttt{SystemCommandAction} \textbf{then} \\
14: \qquad \quad Execute background shell command script: \texttt{subprocess.Popen(A\_j.command)}. \\
15: \qquad \textbf{end if} \\
16: \qquad Break loop. \\
17: \quad \textbf{end if} \\
18: \textbf{end for} \\
19: \textbf{return} Status: \texttt{ACTION\_DISPATCHED\_SUCCESS}, Hit Key ID $j_{\text{hit}}$ \\
\hline
\end{tabular}
\caption{Formal algorithmic specification for planar coordinate mapping and action execution.}
\label{alg:coordinate_mapping}
\end{figure}

\subsubsection{Algorithm 5: PySide6 Interactive Canvas Viewport Drag-and-Drop Math}
Application 1 (\texttt{designer\_app.py}) provides interactive mouse drag-and-drop key placement on a hardware-accelerated 2D canvas.

Algorithm~\ref{alg:canvas_drag_drop} details formal pseudocode for mouse viewport transformation and snap-to-grid bounding box calculation.

\begin{figure}[htbp]
\centering
\begin{tabular}{|p{0.95\textwidth}|}
\hline
\textbf{Algorithm 5: PySide6 Interactive Canvas Viewport Drag-and-Drop Math} \\
\hline
\textbf{Input:} Viewport mouse click coordinate $\mathbf{p}_{\text{mouse}} = (u_{\text{m}}, v_{\text{m}})$, Viewport transformation matrix $\mathbf{T}_{\text{view}} \in \mathbb{R}^{3 \times 3}$, Grid snap spacing $\Delta g = 5.0\text{ mm}$, Button size $(W_{\text{key}}, H_{\text{key}})$ \\
\textbf{Output:} Snapped Key Bounding Box $B_{\text{new}} = (x_{\min}, y_{\min}, x_{\max}, y_{\max})$ \\
1: Map viewport pixel coordinate to scene millimeter space: $[x_{\text{scene}}, y_{\text{scene}}, 1]^T = \mathbf{T}_{\text{view}}^{-1} \cdot [u_{\text{m}}, v_{\text{m}}, 1]^T$. \\
2: Compute grid-snapped origin coordinates: \\
3: \quad $x_{\text{snap}} = \text{round}(x_{\text{scene}} / \Delta g) \times \Delta g$. \\
4: \quad $y_{\text{snap}} = \text{round}(y_{\text{scene}} / \Delta g) \times \Delta g$. \\
5: Define bounding box: $x_{\min} = x_{\text{snap}}$, $y_{\min} = y_{\text{snap}}$, $x_{\max} = x_{\text{snap}} + W_{\text{key}}$, $y_{\max} = y_{\text{snap}} + H_{\text{key}}$. \\
6: Check bounding box collision against existing keys: $\text{Overlap} = \text{CheckIntersections}(B_{\text{new}}, \{B_{\text{existing}}\})$. \\
7: \textbf{if} Overlap is True \textbf{then} \\
8: \quad Shift origin $x_{\text{snap}} = x_{\text{snap}} + \Delta g$ and recompute $B_{\text{new}}$. \\
9: \textbf{end if} \\
10: Instantiate \texttt{KeyButtonGraphic} graphics item on \texttt{QGraphicsScene}. \\
11: \textbf{return} $B_{\text{new}}$ \\
\hline
\end{tabular}
\caption{Formal algorithmic specification for PySide6 canvas viewport drag-and-drop key placement.}
\label{alg:canvas_drag_drop}
\end{figure}

\subsubsection{Algorithm 6: Multi-Threaded Consumer-Producer Queue Synchronization}
Application 2 (\texttt{camera\_thread.py}) enforces multi-threaded queue concurrency separating camera capture from inference.

Algorithm~\ref{alg:queue_synchronization} details formal pseudocode for consumer-producer thread synchronization.

\begin{figure}[htbp]
\centering
\begin{tabular}{|p{0.95\textwidth}|}
\hline
\textbf{Algorithm 6: Multi-Threaded Consumer-Producer Queue Synchronization} \\
\hline
\textbf{Input:} Video Capture object \texttt{cv2.VideoCapture}, Shared Queue buffer $\mathbf{Q}_{\text{frame}}$ (Max Size = 2), Mutex Lock $\mathbf{L}_{\text{mutex}}$ \\
\textbf{Output:} Thread-safe Frame Stream Buffer \\
1: \textbf{Worker Thread Function} \texttt{CameraCaptureLoop()}: \\
2: \textbf{while} Runtime flag \texttt{is\_running} is True \textbf{do} \\
3: \quad Read raw video frame: $\text{ret}, \mathbf{I}_{\text{raw}} = \text{cap.read}()$. \\
4: \quad \textbf{if} $\text{ret}$ is False \textbf{then} Continue. \\
5: \quad Acquire Mutex Lock $\mathbf{L}_{\text{mutex}}$. \\
6: \quad \textbf{if} Queue $\mathbf{Q}_{\text{frame}}$ is Full \textbf{then} \\
7: \qquad Discard oldest frame (Drop frame to prevent buffer bloat/latency lag). \\
8: \qquad $\mathbf{Q}_{\text{frame}}.\text{get\_nowait}()$. \\
9: \quad \textbf{end if} \\
10: \quad Push new frame $\mathbf{I}_{\text{raw}}$ to $\mathbf{Q}_{\text{frame}}$. \\
11: \quad Release Mutex Lock $\mathbf{L}_{\text{mutex}}$. \\
12: \textbf{end while} \\
\hline
\end{tabular}
\caption{Formal algorithmic specification for multi-threaded consumer-producer queue synchronization.}
\label{alg:queue_synchronization}
\end{figure}

\subsubsection{Transfer Function Phase Lag Derivation for 1€ Low-Pass Filter Exclusion}
\label{subsubsec:imp_phase_lag_derivation}
A core technical requirement of this research is enforcing: **DO NOT use 1€ filter or temporal smoothing**. To mathematically justify this requirement, consider a standard first-order low-pass filter (the foundation of 1€ filtering):
\begin{equation}
\tau \frac{dy(t)}{dt} + y(t) = x(t)
\label{eq:lowpass_diff}
\end{equation}
where $x(t)$ is the raw keypoint position, $y(t)$ is the filtered position, and $\tau = \frac{1}{2\pi f_c}$ is the filter time constant corresponding to cutoff frequency $f_c$.

Taking the Laplace transform under zero initial conditions yields transfer function $H(s)$:
\begin{equation}
H(s) = \frac{Y(s)}{X(s)} = \frac{1}{\tau s + 1}
\label{eq:lowpass_laplace}
\end{equation}

Evaluating frequency response by setting $s = j\omega$:
\begin{equation}
H(j\omega) = \frac{1}{1 + j\omega \tau} = \frac{1 - j\omega \tau}{1 + \omega^2 \tau^2}
\label{eq:lowpass_freq_resp}
\end{equation}

The phase response $\phi(\omega) = \arg(H(j\omega))$ is given by:
\begin{equation}
\phi(\omega) = -\arctan(\omega \tau)
\label{eq:lowpass_phase_lag}
\end{equation}

The group delay $\tau_g(\omega)$, representing physical temporal latency introduced by filtering, is:
\begin{equation}
\tau_g(\omega) = -\frac{d\phi(\omega)}{d\omega} = \frac{\tau}{1 + \omega^2 \tau^2}
\label{eq:lowpass_group_delay}
\end{equation}

For typical keypoint smoothing parameters ($f_c \in [1.5\text{ Hz}, 5.0\text{ Hz}]$), group delay $\tau_g$ introduces a $30\text{ ms}$ to $90\text{ ms}$ phase lag. During rapid typing, downward kinetic contact impact produces high-frequency acceleration spikes $\ddot{x}(t)$ lasting less than $25\text{ ms}$. Low-pass filtering severely damps these contact spikes, causing deep classifiers to misclassify touch events as hovering movements. Thus, eliminating 1€ filtering and propagating raw scale-normalized features directly to PyTorch LSTM model \texttt{best\_finger\_touch\_lstm.pth} is mathematically necessary for low-latency touch inference.

\subsection{Technology Selection Justification and Comparative Reflection}
\label{subsec:imp_tech_justification}
Selecting robust programming languages, vision frameworks, machine learning libraries, and GUI tooling is critical for achieving real-time performance on commodity CPU hardware.

Table~\ref{tab:tech_selection_matrix} presents a comparative technical reflection justifying technology selections.

\begin{table}[htbp]
\centering
\caption{Comparative technology selection justification matrix.}
\label{tab:tech_selection_matrix}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l l l l}
\toprule
\textbf{Technology Domain} & \textbf{Selected Technology} & \textbf{Evaluated Alternatives} & \textbf{Selection Rationale \& Performance Trade-offs} \\
\midrule
\textbf{Programming Language} & \textbf{Python 3.10} & - C++17 & - Rapid prototyping, extensive vision/ML ecosystem \\
 & & - Java 17 / C\# & - C++ extensions (\texttt{OpenCV}, \texttt{PyTorch C++}) eliminate GIL bottleneck \\
 & & - JavaScript / Node.js & - Superior array vectorization via NumPy C-API \\
\midrule
\textbf{Pose Landmarker} & \textbf{MediaPipe Hands} & - OpenPose & - MediaPipe executes sub-10 ms landmark regression on CPU \\
 & & - HRNet 2D Pose & - OpenPose requires heavy GPU compute ($> 100$ ms CPU latency) \\
 & & - AlphaPose & - MediaPipe returns 21 3D/2D joint keypoints natively \\
\midrule
\textbf{Deep ML Framework} & \textbf{PyTorch 2.1} & - TensorFlow 2.x & - Dynamic execution graph allows variable batching \\
 & & - ONNX Runtime & - Python-native tensor operations simplify sliding windows \\
 & & - Scikit-Learn (SVM) & - Superior LSTM training stability and model export (\texttt{.pth}) \\
\midrule
\textbf{Fiducial Tracking} & \textbf{AprilTag (tag36h11)} & - OpenCV ArUco & - AprilTag provides higher sub-pixel corner accuracy ($0.038$ px) \\
 & & - QR Code & - Superior robustness against motion blur and $75^\circ$ tilt \\
 & & - Character Keypoints & - Tag36h11 codebook Hamming distance $d_H \ge 11$ prevents false tags \\
\midrule
\textbf{Desktop GUI Tooling} & \textbf{PySide6 (Qt6)} & - Tkinter & - Native Qt6 graphics view framework (\texttt{QGraphicsScene}) \\
 & & - PyQt5 & - High-performance hardware-accelerated canvas rendering \\
 & & - Electron.js & - Dual exporter engine support for vector PDF and XML \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsubsection{Programming Language Selection (Python 3.10 vs C++17 vs Java)}
Python 3.10 was selected as the primary implementation language due to its unmatched ecosystem in computer vision, deep learning, and mathematical computing. While native C++ offers raw execution speed, Python bindings for C++ backends (\texttt{OpenCV-Python}, \texttt{PyTorch}, \texttt{MediaPipe}) execute critical vision loops in optimized C/C++ compiled binaries, combining developer productivity with high performance.

\subsubsection{Deep Learning Frameworks (PyTorch 2.1 vs TensorFlow 2.x vs ONNX)}
PyTorch 2.1 was chosen over TensorFlow and ONNX Runtime due to its dynamic imperative execution paradigm, which simplifies 5-frame sliding window tensor manipulations. PyTorch's native GPU acceleration (CUDA) and lightweight CPU inference engine allow the trained LSTM model file (\texttt{best\_finger\_touch\_lstm.pth}) to execute forward passes in under $2.1\text{ ms}$ on standard CPUs.

\subsubsection{Pose Landmarker Engine (MediaPipe Hands vs OpenPose vs HRNet)}
MediaPipe Hand Landmarker was selected because its multi-stage palm detector and keypoint regressor execute keypoint inference in under $8\text{ ms}$ on standard CPUs \cite{Zhang2020MediaPipe}. Heavy top-down pose models such as OpenPose and HRNet require dedicated GPU hardware, exhibiting CPU latencies exceeding $100\text{ ms}$ per frame.

\subsubsection{Visual Fiducial System (AprilTag tag36h11 vs ArUco vs QR Codes)}
AprilTag visual fiducial markers (\texttt{tag36h11} family) were selected over OpenCV ArUco tags and standard QR codes due to superior sub-pixel quad corner localization accuracy ($0.038\text{ px}$) and robust detection under extreme perspective inclination angles up to $75^\circ$. The \texttt{tag36h11} lexicographic codebook guarantees a minimum Hamming distance $d_H \ge 11$, preventing false positive marker identifications under complex ambient lighting.

\subsubsection{Desktop GUI Framework (PySide6 / Qt6 vs Tkinter vs Electron.js)}
PySide6 (Qt for Python 6) was selected for Application 1 Layout Designer (\texttt{designer/designer\_app.py}) and Application 2 Setup GUI. Qt6's \texttt{QGraphicsScene} and \texttt{QGraphicsItem} class hierarchy provides hardware-accelerated 2D canvas manipulation, allowing key buttons to be dragged, resized, and rendered with sub-pixel alignment. Qt's vector printing engine (\texttt{QPrinter}) enables direct rendering of printable PDF layout sheets embedded with AprilTag visual anchors.

\section{Significantly Important System Implementation Attempts}
\label{sec:imp_significant_attempts}
This section details significantly important software implementations across the five isolated research partitions and main application GUIs.

\subsection{Application 1 Layout Designer Suite Implementation (\texttt{designer/designer\_app.py})}
Application 1 (\texttt{designer/designer\_app.py}) provides an interactive desktop GUI workspace for designing customizable paper key layouts.

Key implementation features include:
\begin{itemize}
    \item \textbf{PySide6 Canvas Workspace:} Built on \texttt{QGraphicsView} and \texttt{QGraphicsScene}, representing an A4 paper sheet ($210 \times 297\text{ mm}$). Users drag-and-drop key buttons, resize bounding boxes, and assign custom key IDs and command strings.
    \item \textbf{Automated AprilTag Border Anchoring:} The layout engine automatically calculates and positions four AprilTag visual anchors (\texttt{tag36h11} family, Tag IDs 0, 1, 2, 3) along the layout border margins, ensuring unoccluded visibility during typing.
    \item \textbf{Dual Exporter Engine:} Generates vector PDF print sheets using \texttt{QPrinter} (for physical paper printing) and exports matching XML specification files defining key bounding box coordinates $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$, button IDs, default values, and tag corner positions.
\end{itemize}

\subsection{Partition 1 Homography Simulation Suite (\texttt{designer/analyzer/})}
Located in \texttt{designer/analyzer/}, scripts \texttt{main.py}, \texttt{analyzer\_app.py}, and \texttt{homography\_engine.py} form a simulated touch verification suite.

The simulation module generates synthetic finger touch pixel coordinates under simulated camera perspective tilts ($0^\circ$ to $75^\circ$). By applying SVD homography matrix transformation ($P_{\text{XML}} = H \cdot P_{\text{pixel}}$), the analyzer verifies that transformed coordinates map precisely into target XML key bounding boxes, confirming mathematical mapping correctness prior to live camera testing.

\subsection{Partition 2 Data Pipeline, Resampling, Landmark Extraction \& Scale Normalization (\texttt{mediapipeDetector/datacreator/})}
Located in \texttt{mediapipeDetector/datacreator/}, this partition implements the sequential data preparation pipeline:
\begin{enumerate}
    \item \textbf{\texttt{resample\_12fps.py}:} Down-samples raw 30/60 FPS video recordings to a uniform 12 FPS frame rate using OpenCV frame indexing, standardizing temporal observation steps ($\Delta t = \frac{1}{12}\text{ s}$).
    \item \textbf{\texttt{normalize\_landmarks.py}:} Implements unitless hand-length scale normalization ($L_{\text{hand}}$) relative to wrist Landmark 0 and middle MCP Landmark 9. In strict compliance with system technical rules, low-pass 1€ filtering and temporal smoothing were strictly excluded to preserve raw kinetic deceleration inflections.
    \item \textbf{\texttt{calculate\_velocities.py}:} Computes first-order spatial velocity vectors $\mathbf{V}_i(t)$ across consecutive frames using numerical central differences.
    \item \textbf{\texttt{create\_windows.py}:} Assembles scale-normalized position and velocity features into 5-frame sliding temporal windows with a 2-frame overlap (stride of 3 frames), outputting NumPy array matrices $\mathbf{X}_W \in \mathbb{R}^{5 \times 84}$.
    \item \textbf{\texttt{filter\_window\_quality.py}:} Scans sequence arrays and discards corrupted windows containing missing keypoint detections or low landmark confidence scores ($< 0.5$).
\end{enumerate}

\subsection{Partition 3 Deep Learning Multi-Model Benchmark Engine (\texttt{mediapipeDetector/deepLearningModels/run\_all.py})}
Driven by master benchmark script \texttt{mediapipeDetector/deepLearningModels/run\_all.py}, this partition evaluated 22 pure 2D deep learning model architecture combinations under 5-fold cross-validation.

The benchmark engine evaluated:
\begin{itemize}
    \item \textbf{Model Architectures:} Uni-directional LSTMs, Bidirectional LSTMs (BiLSTMs), 1D Convolutional Neural Networks (1D CNNs), and 1D Residual Networks (1D ResNets).
    \item \textbf{Feature Representations:} Raw pixel coordinates, wrist-centered coordinates, scale-normalized positions, and scale-normalized velocity vectors.
    \item \textbf{Loss Function \& Optimization:} Models were trained using Binary Cross-Entropy with Logits loss ($\mathcal{L}_{\text{BCE}}$) and the Adam optimizer ($\beta_1=0.9, \beta_2=0.999$, initial learning rate $\eta = 10^{-3}$ with cosine annealing decay).
    \item \textbf{Optimal Trained Model Selection:} Benchmarks confirmed that a multi-layer PyTorch uni-directional LSTM model trained on 5-frame scale-normalized position and velocity features achieved the highest classification F1-score ($0.963$) with low computational overhead ($< 1\text{ MB}$ size), exported to \texttt{best\_finger\_touch\_lstm.pth}.
\end{itemize}

\subsection{Partition 4 Real-Time Execution \& Multi-Threaded Engine (\texttt{mediapipeDetector/realtimeprocess/})}
Located in \texttt{mediapipeDetector/realtimeprocess/}, scripts \texttt{camera\_thread.py}, \texttt{model\_manager.py}, and \texttt{main\_realtime\_ui.py} form the live real-time evaluation suite.

Key technical implementations include:
\begin{itemize}
    \item \textbf{Multi-Threaded Frame Acquisition (\texttt{camera\_thread.py}):} Executes OpenCV video capture in a dedicated background thread, buffering incoming frames into a thread-safe Queue to prevent frame drops.
    \item \textbf{Model Manager (\texttt{model\_manager.py}):} Loads PyTorch state dictionary \texttt{best\_finger\_touch\_lstm.pth} into RAM once upon initialization and manages PyTorch tensor conversions for sliding window evaluation.
    \item \textbf{Real-Time UI Suite (\texttt{main\_realtime\_ui.py}):} Evaluates live camera streaming latency and UI event dispatching, demonstrating an end-to-end processing latency of $29.09\text{ ms}$ ($\approx 34.4\text{ FPS}$) on standard desktop CPUs.
\end{itemize}

\subsection{Partition 5 AprilTag Fiducial Tracking Engine (\texttt{aprilTag/})}
Located in \texttt{aprilTag/}, scripts \texttt{main.py}, \texttt{estimater.py}, and \texttt{homography.py} implement optical visual tracking:
\begin{itemize}
    \item \textbf{Camera Calibration (\texttt{main.py}):} Computes intrinsic camera matrix $K$ and lens distortion coefficients using checkerboard calibration patterns.
    \item \textbf{AprilTag Corner Estimator (\texttt{estimater.py}):} Detects AprilTag visual quads (\texttt{tag36h11} family) and extracts sub-pixel corner coordinates under perspective camera tilt up to $75^\circ$.
    \item \textbf{Real-Time Homography Engine (\texttt{homography.py}):} Solves the $3 \times 3$ Planar Homography matrix $H$ using SVD ($\mathbf{A} \mathbf{h} = \mathbf{0}$), continuously updating shared matrix memory.
\end{itemize}

\subsection{Application 2 Runtime Engine Setup GUI \& Action Command Mapper}
Application 2 combines a Setup GUI (Component A) with a Background Vision Execution Engine (Component B):
\begin{itemize}
    \item \textbf{Component A (Setup \& Mapper GUI):} Provides an interactive PySide6 interface for selecting camera feeds, loading layout XML schemas, mapping button IDs to single keystrokes (e.g., `'A'`, `'Space'`) or system shell commands (e.g., launching software, controlling media playback), and saving/loading JSON action mapping configuration profiles.
    \item \textbf{Component B (Background Vision Engine):} Runs silently in the background, actively watching live camera feeds. It tracks AprilTags to update $H$, extracts MediaPipe hand pose landmarks, scale-normalizes keypoints relative to $L_{\text{hand}}$ without low-pass filtering, evaluates 5-frame sliding window matrices with PyTorch LSTM model \texttt{best\_finger\_touch\_lstm.pth}, transforms active fingertip pixels through $H$ ($\mathbf{P}_{\text{XML}} = H \cdot \mathbf{P}_{\text{pixel}}$), looks up target XML key bounding boxes, and executes mapped keystrokes or shell commands.
\end{itemize}

\section{Empirical Implementation Challenges and Technical Solutions}
\label{sec:imp_challenges}
Engineering a monocular paper virtual keyboard system presented complex technical challenges across computer vision, signal processing, machine learning, and multi-threading.

Table~\ref{tab:implementation_challenges} details five critical technical challenges encountered during implementation alongside their engineered software solutions.

\begin{table}[htbp]
\centering
\caption{Technical implementation challenges, root cause analyses, and engineered software solutions.}
\label{tab:implementation_challenges}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l p{4.5cm} p{4.5cm} p{5cm}}
\toprule
\textbf{Technical Challenge} & \textbf{Observed Operational Symptom} & \textbf{Root Cause Analysis} & \textbf{Engineered Software Solution} \\
\midrule
\textbf{C1: Depth Ambiguity in} & Hovering fingers close to paper surface & Single 2D RGB camera lacks physical Z-axis & - Assembled 5-frame sliding window matrix \\
\textbf{Single-Camera RGB Feeds} & triggered false positive key actuations. & depth sensor measurements. & - Trained PyTorch LSTM on joint velocity vectors \\
 & & & capturing downward deceleration inflections. \\
\midrule
\textbf{C2: Hand Scale Variance Across} & Fixed pixel distance thresholds failed when & Fingertip pixel movement varies with & - Implemented unitless hand scale normalization \\
\textbf{Diverse Users \& Distances} & hand size or camera distance changed. & camera distance $Z$ and hand size. & dividing coordinate offsets by $L_{\text{hand}} = \|\mathbf{P}_9 - \mathbf{P}_0\|_2$. \\
\midrule
\textbf{C3: Phase Delay Lag from} & Rapid touch taps were missed or delayed by & Low-pass 1€ filters introduce 30--90 ms & - Enforced strict system rule: \textbf{DO NOT use 1€} \\
\textbf{Temporal Low-Pass Filters} & 50--100 ms during fast typing. & group delay, damping impact spikes. & \textbf{filter or temporal smoothing}. \\
\midrule
\textbf{C4: AprilTag Occlusion by} & Homography matrix $H$ failed when typing & Typing hands temporarily occlude tag & - Placed AprilTags along layout outer margins. \\
\textbf{Overlapping Hands} & hands covered paper markers. & corner points printed on paper. & - Required only 4 unoccluded tag corners for SVD. \\
\midrule
\textbf{C5: Frame Dropping in} & Live video feed lagged when deep learning & Heavy model inference blocked main OpenCV & - Implemented multi-threaded Queue buffer \\
\textbf{Single-Threaded Loop} & model evaluation executed. & capture thread loop. & in \texttt{camera\_thread.py} separating capture from inference. \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Deep Technical Challenge Analysis}
\label{subsec:imp_challenge_analysis}

\subsubsection{Challenge 1: Resolving Monocular Z-Axis Depth Ambiguity}
Monocular RGB cameras flatten 3D physical space into 2D pixel grids, creating severe depth ambiguity when a finger hovers millimeters above a paper surface \cite{ReviewVirtualKeyboard2020, Thomas2013Camera}. Static spatial thresholding fails to distinguish between hovering and surface contact. This was resolved by assembling 5-frame sliding temporal windows ($\mathbf{X}_W \in \mathbb{R}^{5 \times 84}$) and training a PyTorch LSTM classifier on scale-normalized numerical velocities ($\mathbf{V}_i$), enabling the network to learn kinetic deceleration inflections characteristic of paper surface impact.

\subsubsection{Challenge 2: Achieving Distance and Hand-Size Scale Invariance}
Fingertip pixel displacements scale inversely with camera height $Z$ ($x_{\text{pixel}} = f \cdot \frac{X}{Z}$). Static pixel thresholds fail when users move closer to or further from the camera, or when users display different hand dimensions. This was resolved by implementing unitless scale normalization (\texttt{normalize\_landmarks.py}), translating landmarks to wrist origin $\mathbf{P}_0$ and dividing coordinate displacements by middle MCP hand length $L_{\text{hand}} = \|\mathbf{P}_9 - \mathbf{P}_0\|_2$. This transforms raw keypoint features into unitless scale-invariant ratios.

\subsubsection{Challenge 3: Eliminating Low-Pass Filter Phase Lag}
Standard computer vision pipelines apply 1€ low-pass filtering or temporal moving averages to smooth noisy landmark jitter. However, transfer function phase analysis reveals that a 1€ filter introduces a phase lag $\Delta \phi(\omega) = -\arctan(\omega \tau)$, resulting in a 30–90 ms group delay $\tau_g = \frac{\tau}{1 + \omega^2 \tau^2}$. In rapid typing, this phase lag damps rapid kinetic contact spikes, causing missed taps. This was resolved by enforcing the critical technical rule: **DO NOT use 1€ filter or temporal smoothing**. Raw scale-normalized keypoint features propagate directly to the PyTorch LSTM classifier, preserving kinetic deceleration transients.

\subsubsection{Challenge 4: Robust AprilTag Fiducial Tracking under Hand Occlusion}
Typing hands frequently pass over printed paper layout sheets, creating potential visual marker occlusion. If fiducial tracking fails, planar homography matrix $H$ cannot be updated. This was resolved by positioning AprilTag anchors (\texttt{tag36h11} family) along layout outer border margins in Application 1 Layout Designer (\texttt{designer\_app.py}). Furthermore, SVD homography calculation (\texttt{aprilTag/homography.py}) requires only four unoccluded tag corner points across the entire layout boundary to compute matrix $H \in \mathbb{R}^{3 \times 3}$.

\subsubsection{Challenge 5: Multi-Threaded Queue Buffer Architecture}
In single-threaded vision loops, executing deep neural network inference on incoming video frames blocks camera acquisition, causing dropped frames and video stuttering. This was resolved by engineering a multi-threaded Producer-Consumer queue architecture in Application 2 (\texttt{camera\_thread.py}). OpenCV camera frame capture operates in a high-priority background thread, pushing raw frames into a thread-safe Queue. The model inference manager (\texttt{model\_manager.py}) pops frames from the queue asynchronously, maintaining a stable $34.4\text{ FPS}$ pipeline throughput without frame loss.

\section{Chapter Summary}
\label{sec:imp_summary}
This chapter presented the comprehensive software implementation and algorithm design for the customizable paper virtual keyboard system.

Section~\ref{sec:imp_framework_steps} established the framework engineering workflow, presented formal mathematical pseudocode algorithms for AprilTag SVD homography, scale normalization, PyTorch LSTM touch inference, and planar coordinate mapping, provided the mathematical transfer function derivation enforcing 1€ filter exclusion, and justified technology selections (Python 3.10, PyTorch, MediaPipe, PySide6). Section~\ref{sec:imp_significant_attempts} detailed significantly important software implementations across Application 1 Layout Designer (\texttt{designer\_app.py}) and the five isolated research partitions (\texttt{designer/analyzer/}, \texttt{mediapipeDetector/datacreator/}, \texttt{mediapipeDetector/deepLearningModels/run\_all.py}, \texttt{mediapipeDetector/realtimeprocess/}, \texttt{aprilTag/}). Section~\ref{sec:imp_challenges} analyzed five critical technical challenges (depth ambiguity, scale variance, 1€ filter phase lag elimination, AprilTag occlusion, multi-threading) and their engineered software solutions.

The software implementation and architectural designs detailed in this chapter provide the complete operational foundation for the empirical testing and performance evaluation presented in Chapter 6.
"""

with open('chapters/chapter05.tex', 'w') as f:
    f.write(latex_text)

print("Wrote chapter05.tex successfully!")
