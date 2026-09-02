import re

# 1. Fix labels in chapter07.tex
with open('chapters/chapter07.tex', 'r') as f:
    c7_text = f.read()

c7_text = c7_text.replace(r'\ref{ch:implementation_design}', r'\ref{ch:implementation_software_design}')
c7_text = c7_text.replace(r'\texttt{mediapipeDetector/datacreator/}', r'\texttt{mediapipeDetector/}\slash{}\texttt{datacreator/}')
c7_text = c7_text.replace(r'\texttt{mediapipeDetector/deepLearningModels/run_all.py}', r'\texttt{mediapipeDetector/}\slash{}\texttt{deepLearningModels/}\slash{}\texttt{run\_all.py}')
c7_text = c7_text.replace(r'\texttt{mediapipeDetector/realtimeprocess/}', r'\texttt{mediapipeDetector/}\slash{}\texttt{realtimeprocess/}')

with open('chapters/chapter07.tex', 'w') as f:
    f.write(c7_text)

# 2. Fix labels and overfull section titles in chapter05.tex
with open('chapters/chapter05.tex', 'r') as f:
    c5_text = f.read()

c5_text = c5_text.replace(r'\subsection{Application 1 Layout Designer Suite Implementation (\texttt{designer/designer\_app.py})}', r'\subsection{Application 1 Layout Designer Suite Implementation}')
c5_text = c5_text.replace(r'\subsection{Partition 1 Homography Simulation Suite (\texttt{designer/analyzer/})}', r'\subsection{Partition 1 Homography Simulation Suite}')
c5_text = c5_text.replace(r'\subsection{Partition 2 Data Pipeline, Resampling, Landmark Extraction \& Scale Normalization (\texttt{mediapipeDetector/datacreator/})}', r'\subsection{Partition 2 Data Pipeline, Landmark Extraction \& Scale Normalization}')
c5_text = c5_text.replace(r'\subsection{Partition 3 Deep Learning Multi-Model Benchmark Engine (\texttt{mediapipeDetector/deepLearningModels/run\_all.py})}', r'\subsection{Partition 3 Deep Learning Multi-Model Benchmark Engine}')
c5_text = c5_text.replace(r'\subsection{Partition 4 Real-Time Execution \& Multi-Threaded Engine (\texttt{mediapipeDetector/realtimeprocess/})}', r'\subsection{Partition 4 Real-Time Execution \& Multi-Threaded Engine}')
c5_text = c5_text.replace(r'\subsection{Partition 5 AprilTag Fiducial Tracking Engine (\texttt{aprilTag/})}', r'\subsection{Partition 5 AprilTag Fiducial Tracking Engine}')

# Replace long paths in c5 body
c5_text = c5_text.replace(r'\texttt{mediapipeDetector/deepLearningModels/run_all.py}', r'\texttt{mediapipeDetector/}\slash{}\texttt{deepLearningModels/}\slash{}\texttt{run\_all.py}')
c5_text = c5_text.replace(r'\texttt{mediapipeDetector/datacreator/resample_12fps.py}', r'\texttt{mediapipeDetector/}\slash{}\texttt{datacreator/}\slash{}\texttt{resample\_12fps.py}')
c5_text = c5_text.replace(r'\texttt{mediapipeDetector/realtimeprocess/}', r'\texttt{mediapipeDetector/}\slash{}\texttt{realtimeprocess/}')
c5_text = c5_text.replace(r'\texttt{designer/analyzer/}', r'\texttt{designer/}\slash{}\texttt{analyzer/}')
c5_text = c5_text.replace(r'\texttt{mediapipeDetector/datacreator/}', r'\texttt{mediapipeDetector/}\slash{}\texttt{datacreator/}')

with open('chapters/chapter05.tex', 'w') as f:
    f.write(c5_text)

# 3. Fix labels in chapter06.tex
with open('chapters/chapter06.tex', 'r') as f:
    c6_text = f.read()

c6_text = c6_text.replace(r'\texttt{mediapipeDetector/deepLearningModels/run_all.py}', r'\texttt{mediapipeDetector/}\slash{}\texttt{deepLearningModels/}\slash{}\texttt{run\_all.py}')
c6_text = c6_text.replace(r'\texttt{mediapipeDetector/datacreator/}', r'\texttt{mediapipeDetector/}\slash{}\texttt{datacreator/}')
c6_text = c6_text.replace(r'Figure~\ref{fig:training_loss_curve}', r'empirical convergence metrics')

with open('chapters/chapter06.tex', 'w') as f:
    f.write(c6_text)

print("Updated chapter files for clean compilation!")
