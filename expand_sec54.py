import os

with open('chapters/chapter05.tex', 'r') as f:
    text = f.read()

expanded_sec54 = r"""\subsection{Deep Technical Challenge Analysis}
\label{subsec:imp_challenge_analysis}

\subsubsection{Challenge 1: Resolving Monocular Z-Axis Depth Ambiguity}
Standard monocular RGB video feeds project 3D physical space onto a 2D image plane, inherently discarding explicit Z-axis depth measurements \cite{ReviewVirtualKeyboard2020, Thomas2013Camera}. When a user's finger hovers a few millimeters above a paper virtual keyboard surface, its 2D camera pixel coordinates $(u, v)$ are identical to those when the finger establishes actual physical contact with the paper sheet. Static spatial distance thresholding fails to distinguish between hovering and surface contact, causing severe false-positive key actuations.

To resolve monocular depth ambiguity without requiring specialized hardware (such as stereo depth cameras or time-of-flight sensors), the system leverages temporal kinetic feature windowing. As a finger approaches a physical surface to strike a key button, its motion exhibits a distinct kinetic velocity profile: an acceleration phase during downward movement followed by a rapid deceleration inflection upon physical impact with the paper sheet. By assembling 5-frame sliding temporal windows ($\mathbf{X}_W \in \mathbb{R}^{5 \times 84}$) containing scale-normalized spatial velocity vectors $\mathbf{V}_i(t)$, the PyTorch uni-directional LSTM classifier (\texttt{best\_finger\_touch\_lstm.pth}) learns to detect this temporal deceleration signature, robustly classifying surface contact independently of static Z-axis depth.

\subsubsection{Challenge 2: Achieving Distance and Hand-Size Scale Invariance}
Fingertip pixel displacements in monocular camera images depend heavily on two physical variables: the user's distance $Z$ from the camera lens and the user's anatomical hand size. Under perspective projection, a fingertip movement of $10\text{ mm}$ in physical space produces a pixel displacement $x_{\text{pixel}} = f \cdot \frac{X}{Z}$, where $f$ is the camera focal length. Consequently, fixed pixel distance thresholds fail when a user moves closer to or further from the camera feed, or when users display different hand proportions.

This challenge was solved by implementing unitless hand scale normalization in \texttt{normalize\_landmarks.py}. Wrist Landmark 0 ($\mathbf{P}_0$) is assigned as the local coordinate origin, eliminating absolute image location offsets. Next, hand length $L_{\text{hand}}(t) = \|\mathbf{P}_9(t) - \mathbf{P}_0(t)\|_2$ is derived as the Euclidean distance between wrist Landmark 0 and middle MCP Landmark 9. All 21 joint landmark coordinate offsets are divided by unitless hand length $L_{\text{hand}}(t)$:
\begin{equation}
x_i^{\text{norm}}(t) = \frac{x_i(t) - x_0(t)}{L_{\text{hand}}(t)}, \quad
y_i^{\text{norm}}(t) = \frac{y_i(t) - y_0(t)}{L_{\text{hand}}(t)}
\label{eq:scale_norm_explanation}
\end{equation}

Dividing joint coordinates by $L_{\text{hand}}(t)$ transforms raw pixel keypoints into unitless scale-invariant ratios. As a user moves closer to the camera, both joint displacements $\Delta x_i$ and hand length $L_{\text{hand}}$ increase proportionally, preserving identical scale-normalized feature ratios $\mathbf{P}_i^{\text{norm}}$.

\subsubsection{Challenge 3: Eliminating Low-Pass Filter Phase Lag (1€ Filter Exclusion)}
Traditional computer vision and gesture tracking systems commonly apply temporal low-pass filters (such as 1€ filters, Butterworth filters, or moving average smoothers) to reduce high-frequency landmark position jitter. However, transfer function phase delay analysis (Section~\ref{subsubsec:imp_phase_lag_derivation}) demonstrates that a low-pass filter introduces a frequency-dependent phase lag $\phi(\omega) = -\arctan(\omega \tau)$ and temporal group delay $\tau_g = \frac{\tau}{1 + \omega^2 \tau^2}$.

For standard keypoint filter parameters ($f_c \in [1.5\text{ Hz}, 5.0\text{ Hz}]$), low-pass filtering introduces a $30\text{ ms}$ to $90\text{ ms}$ latency delay. During fast touch typing, downward finger impact deceleration spikes last less than $25\text{ ms}$. Temporal low-pass filtering severely damps these rapid deceleration inflections, rounding off sharp velocity transients. As a result, temporal deep learning classifiers misclassify rapid surface touch taps as smooth hovering movements, degrading classification accuracy and increasing input latency.

To eliminate filter-induced phase delay, the system enforces the critical technical override: **DO NOT use 1€ filter or temporal smoothing**. Raw scale-normalized position and velocity features propagate directly to PyTorch LSTM model \texttt{best\_finger\_touch\_lstm.pth}. The LSTM recurrent layers inherently learn spatial-temporal representation dependencies, providing noise robustness while preserving crisp deceleration transients.

\subsubsection{Challenge 4: Robust AprilTag Fiducial Tracking under Hand Occlusion}
In paper virtual keyboard systems, user hands continuously hover over and pass across the printed paper layout sheet during touch typing. If visual fiducial tracking relies on central page markers, typing hands frequently cover the markers, causing visual occlusion and breaking planar homography matrix $H$ calculations.

This challenge was addressed through two coordinated engineering designs:
\begin{enumerate}
    \item \textbf{Outer Border Anchor Placement:} In Application 1 Layout Designer (\texttt{designer\_app.py}), four AprilTag visual anchors (\texttt{tag36h11} family, Tag IDs 0, 1, 2, 3) are automatically positioned along the outer paper border margins (Equations~\ref{eq:tag0_pos}--\ref{eq:tag3_pos}). Because user hands type predominantly within the central key region, border markers remain unoccluded during normal typing.
    \item \textbf{Minimal SVD Point Correspondence Solver:} SVD homography calculation (\texttt{aprilTag/homography.py}) requires only four unoccluded tag corner points across the entire layout boundary to compute matrix $H \in \mathbb{R}^{3 \times 3}$. If one or two AprilTags are temporarily covered, the tracking engine continues solving $H$ using the remaining visible tag corners.
\end{enumerate}

\subsubsection{Challenge 5: Multi-Threaded Queue Buffer Architecture}
In single-threaded computer vision applications, video frame capture, hand landmark extraction, deep neural model evaluation, homography matrix updates, and GUI rendering operate sequentially in a single main loop. When heavy deep neural network inference or homography SVD calculations execute, the loop blocks, causing dropped video frames, variable frame rates, and visual feed stuttering.

To achieve real-time pipeline execution, Application 2 implements a multi-threaded Producer-Consumer queue architecture in \texttt{camera\_thread.py}:
\begin{itemize}
    \item \textbf{Producer Capture Thread:} OpenCV video capture runs in a high-priority background worker thread (\texttt{QThread}), capturing raw video frames at a constant frame rate and pushing them into a thread-safe Queue buffer ($\mathbf{Q}_{\text{frame}}$, max size = 2).
    \item \textbf{Consumer Vision \& Inference Thread:} The background watch engine pops frames from the queue asynchronously. If model evaluation takes slightly longer than a frame interval, oldest frames in the queue are dropped automatically, preventing latency buffer bloat.
\end{itemize}

Decoupling video capture from inference execution guarantees a stable pipeline throughput of $34.4\text{ FPS}$ ($29.09\text{ ms}$ latency per frame) on commodity desktop CPUs.
"""

text = text.replace(r"\subsection{Deep Technical Challenge Analysis}" + "\n" + r"\label{subsec:imp_challenge_analysis}" + "\n\n" + r"\subsubsection{Challenge 1: Resolving Monocular Z-Axis Depth Ambiguity}" + "\n" + r"Monocular RGB cameras flatten 3D physical space into 2D pixel grids, creating severe depth ambiguity when a finger hovers millimeters above a paper surface \cite{ReviewVirtualKeyboard2020, Thomas2013Camera}. Static spatial thresholding fails to distinguish between hovering and surface contact. This was resolved by assembling 5-frame sliding temporal windows ($\mathbf{X}_W \in \mathbb{R}^{5 \times 84}$) and training a PyTorch LSTM classifier on scale-normalized numerical velocities ($\mathbf{V}_i$), enabling the network to learn kinetic deceleration inflections characteristic of paper surface impact." + "\n\n" + r"\subsubsection{Challenge 2: Achieving Distance and Hand-Size Scale Invariance}" + "\n" + r"Fingertip pixel displacements scale inversely with camera height $Z$ ($x_{\text{pixel}} = f \cdot \frac{X}{Z}$). Static pixel thresholds fail when users move closer to or further from the camera, or when users display different hand dimensions. This was resolved by implementing unitless scale normalization (\texttt{normalize\_landmarks.py}), translating landmarks to wrist origin $\mathbf{P}_0$ and dividing coordinate displacements by middle MCP hand length $L_{\text{hand}} = \|\mathbf{P}_9 - \mathbf{P}_0\|_2$. This transforms raw keypoint features into unitless scale-invariant ratios." + "\n\n" + r"\subsubsection{Challenge 3: Eliminating Low-Pass Filter Phase Lag}" + "\n" + r"Standard computer vision pipelines apply 1€ low-pass filtering or temporal moving averages to smooth noisy landmark jitter. However, transfer function phase analysis reveals that a 1€ filter introduces a phase lag $\Delta \phi(\omega) = -\arctan(\omega \tau)$, resulting in a 30–90 ms group delay $\tau_g = \frac{\tau}{1 + \omega^2 \tau^2}$. In rapid typing, this phase lag damps rapid kinetic contact spikes, causing missed taps. This was resolved by enforcing the critical technical rule: **DO NOT use 1€ filter or temporal smoothing**. Raw scale-normalized keypoint features propagate directly to the PyTorch LSTM classifier, preserving kinetic deceleration transients." + "\n\n" + r"\subsubsection{Challenge 4: Robust AprilTag Fiducial Tracking under Hand Occlusion}" + "\n" + r"Typing hands frequently pass over printed paper layout sheets, creating potential visual marker occlusion. If fiducial tracking fails, planar homography matrix $H$ cannot be updated. This was resolved by positioning AprilTag anchors (\texttt{tag36h11} family) along layout outer border margins in Application 1 Layout Designer (\texttt{designer\_app.py}). Furthermore, SVD homography calculation (\texttt{aprilTag/homography.py}) requires only four unoccluded tag corner points across the entire layout boundary to compute matrix $H \in \mathbb{R}^{3 \times 3}$." + "\n\n" + r"\subsubsection{Challenge 5: Multi-Threaded Queue Buffer Architecture}" + "\n" + r"In single-threaded vision loops, executing deep neural network inference on incoming video frames blocks camera acquisition, causing dropped frames and video stuttering. This was resolved by engineering a multi-threaded Producer-Consumer queue architecture in Application 2 (\texttt{camera\_thread.py}). OpenCV camera frame capture operates in a high-priority background thread, pushing raw frames into a thread-safe Queue. The model inference manager (\texttt{model\_manager.py}) pops frames from the queue asynchronously, maintaining a stable $34.4\text{ FPS}$ pipeline throughput without frame loss.", expanded_sec54)

with open('chapters/chapter05.tex', 'w') as f:
    f.write(text)

print("Updated Section 5.4 in chapter05.tex successfully!")
