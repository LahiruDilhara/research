# Research Literature Summary Matrix

This document maintains structured technical summaries of research papers stored in `pdf-sources/` for literature review, methodology analysis, and thesis chapter drafting.

---

### 1. A Camera Based Virtual Keyboard with Touch Detection by Shadow Analysis
- **Authors & Year**: Joseph Thomas (2013)
- **Citation Key**: `Thomas2013Camera`
- **DOI / URL**: [Link](https://citeseerx.ist.psu.edu)
- **Technical Summary & Methodology**:
  - Proposes a monocular camera-based virtual keyboard system using shadow analysis to infer physical keystrokes.
  - Detects fingertip positions and tracks shadow displacement under fingertips during downward pressing motions.
  - Calculates intensity gradients and pixel proximity between fingertip contours and projected shadows.
  - Operates with a standard overhead or angled RGB web camera on any flat workspace surface.
  - Eliminates the requirement for dedicated depth sensors, stereo vision, or physical touch hardware.

### 2. A Local Fingertips Movement and Fingertips Clustering Based Virtual Keyboard Adopting a Camera
- **Authors & Year**: Hui Ji, Jianxin Chen, Qingyu Lin, and Ang Li (2018)
- **Citation Key**: `Ji2018Local`
- **DOI / URL**: [10.1145/3232829.3232840](https://doi.org/10.1145/3232829.3232840)
- **Technical Summary & Methodology**:
  - Develops a camera-based virtual keyboard using localized fingertip motion tracking and spatial clustering.
  - Uses skin-color segmentation and contour analysis to locate hands and segment individual fingertips.
  - Tracks local displacement vectors of fingertips to differentiate active typing motions from idle movements.
  - Applies k-means clustering on extracted fingertip locations to establish dynamic key region boundaries.
  - Achieves real-time keystroke input classification on flat surfaces using a standard webcam feed.

### 3. A Single Camera Based Floating Virtual Keyboard with Improved Touch Detection
- **Authors & Year**: Erez Posner, Nick Starzicki, and Eyal Katz (2012)
- **Citation Key**: `Posner2012Single`
- **DOI / URL**: [10.1109/EEEI.2012.6377033](https://doi.org/10.1109/EEEI.2012.6377033)
- **Technical Summary & Methodology**:
  - Introduces a single-camera floating virtual keyboard interface designed for plain paper or desk surfaces.
  - Utilizes dynamic shadow analysis between finger tips and shadow centroids to resolve Z-axis touch depth.
  - Confirms key contact when the Euclidean distance between fingertip and shadow falls below a calibrated threshold.
  - Implements adaptive illumination thresholding to handle variable ambient light and shadow contrast.
  - Demonstrates reduced false-trigger rates and robust key press detection compared to static thresholding.

### 4. An Efficient Patient Activity Recognition using LSTM Network and High-Fidelity Body Pose Tracking
- **Authors & Year**: Thanh-Nghi Doan (2022)
- **Citation Key**: `Doan2022Efficient`
- **DOI / URL**: [10.14569/IJACSA.2022.0130800](https://doi.org/10.14569/IJACSA.2022.0130800)
- **Technical Summary & Methodology**:
  - Presents a real-time patient activity monitoring framework combining MediaPipe pose tracking with an LSTM network.
  - Extracts 3D skeletal landmark sequences directly from standard RGB camera video without wearable sensors.
  - Applies scale normalization and temporal coordinate smoothing to achieve body-size and camera-distance invariance.
  - Trains a multi-layer LSTM network on sequential pose features for dynamic gesture and activity classification.
  - Demonstrates high classification accuracy and low latency for automated monitoring of abnormal movements.

### 5. ArUco Nano: a simpler, faster, and more reliable fiducial marker detector (Preprint)
- **Authors & Year**: Sergio Garrido-Jurado, Francisco J. Romero-Ramirez, and Rafael Muñoz-Salinas (2024)
- **Citation Key**: `GarridoJurado2024ArUcoPreprint`
- **DOI / URL**: [Link](https://arxiv.org)
- **Technical Summary & Methodology**:
  - Introduces ArUco Nano, a minimalist (~500 LOC), header-only C++ library for squared fiducial marker detection.
  - Replaces standard OpenCV contour extraction with a Visited-Aware contour algorithm to eliminate redundant pixel scans.
  - Performs direct sub-pixel code sampling without executing complete perspective image warping for candidates.
  - Substantially reduces memory allocation and execution overhead, achieving up to 6.5x speedup over OpenCV.
  - Demonstrates superior marker detection reliability on high-resolution images and embedded edge devices.

### 6. ArUco Nano: a simpler, faster, and more reliable fiducial marker detector
- **Authors & Year**: Sergio Garrido-Jurado, Francisco J. Romero-Ramirez, and Rafael Muñoz-Salinas (2026)
- **Citation Key**: `GarridoJurado2026ArUco`
- **DOI / URL**: [10.1016/j.softx.2026.102690](https://doi.org/10.1016/j.softx.2026.102690)
- **Technical Summary & Methodology**:
  - Presents the official release of ArUco Nano published in SoftwareX as an open-source header-only library.
  - Optimizes squared fiducial marker localization pipelines specifically tailored for resource-constrained robotics.
  - Combines Visited-Aware contour traversal with fast bit-decoding to eliminate computational bottlenecks.
  - Provides full backward compatibility with standard ArUco dictionary formats and marker sizes.
  - Achieves state-of-the-art throughput on ultra-high-resolution imagery (up to 16 Megapixels).

### 8. Blind Recognition of Touched Keys: Attack and Countermeasures
- **Authors & Year**: Qinggang Yue, Zhen Ling, Benyuan Liu, Xinwen Fu, and Wei Zhao (2014)
- **Citation Key**: `Yue2014Blind`
- **DOI / URL**: [10.1109/TIFS.2015.2428612](https://doi.org/10.1109/TIFS.2015.2428612)
- **Technical Summary & Methodology**:
  - Introduces a computer vision side-channel attack that infer touch screen inputs from side-view videos.
  - Uses optical flow to identify touch interaction frames by detecting finger motion deceleration.
  - Derives a homography matrix H mapping screen surface edges in video frames to a target keyboard template.
  - Analyzes fingertip shadow formations and applies k-means clustering to locate exact contact points.
  - Evaluates attack effectiveness across smartphone devices and proposes key layout randomization defenses.

### 9. CNN+RNN Depth and Skeleton based Dynamic Hand Gesture Recognition
- **Authors & Year**: Kenneth Lai and Svetlana N. Yanushkevich (2018)
- **Citation Key**: `Lai2018CNN`
- **DOI / URL**: [10.1109/ICPR.2018.8545718](https://doi.org/10.1109/ICPR.2018.8545718)
- **Technical Summary & Methodology**:
  - Combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) for dynamic gesture recognition.
  - Fuses multimodal sensor data including depth map images and 3D joint skeleton trajectories.
  - Extracts spatial features using CNNs and models temporal joint sequences using multi-layer RNN/LSTM units.
  - Applies late-fusion network strategies to combine spatial and temporal gesture prediction probabilities.
  - Achieves high classification accuracy on public benchmark datasets for human gesture interaction.

### 10. CNN-Based Real-time Hand and Fingertip Recognition for the Design of a Virtual Keyboard
- **Authors & Year**: Yan-Mei Li, Tae-Ho Lee, Jin-Sung Kim, and Hyuk-Jae Lee (2021)
- **Citation Key**: `Li2021CNN`
- **DOI / URL**: [10.1109/ITC-CSCC52171.2021.9501471](https://doi.org/10.1109/ITC-CSCC52171.2021.9501471)
- **Technical Summary & Methodology**:
  - Presents a real-time CNN-based hand and fingertip detector for virtual keyboard interaction.
  - Uses lightweight CNN architectures to localize hands under varying lighting and complex backgrounds.
  - Extracts 2D spatial coordinate locations for each fingertip without relying on skin-color thresholding.
  - Maps recognized fingertip positions to virtual key grids for real-time keystroke event triggering.
  - Demonstrates real-time frame rates and robust detection performance on embedded hardware platforms.

### 11. Camera pose estimation with moving Aruco-board
- **Authors & Year**: Jakob Isaksson and Lucas Magnusson (2020)
- **Citation Key**: `Isaksson2020Camera`
- **DOI / URL**: [10.1109/ISC2.2017.8090811](https://urn.kb.se/resolve?urn=urn:nbn:se:hj:diva-48419)
- **Technical Summary & Methodology**:
  - Investigates camera pose calibration in stereo vision systems using a moving ArUco marker board.
  - Applies Perspective-n-Point (PnP) solvers with pre-calibrated intrinsic camera matrices to compute extrinsic pose.
  - Tracks 3D-to-2D point correspondences across dynamic vehicle-mounted ArUco marker configurations.
  - Evaluates pose accuracy, rotational drift, and translational error under real-world vehicle movements.
  - Demonstrates reliable camera pose re-calibration for outdoor automated tolling and monitoring infrastructure.

### 12. Deep vision-based real-time hand gesture recognition: a review
- **Authors & Year**: Cui Cui, Mohd Shahrizal Sunar, and Goh Eg Su (2020)
- **Citation Key**: `Cui2020Deep`
- **DOI / URL**: [10.7717/peerj-cs.2921](https://doi.org/10.7717/peerj-cs.2921)
- **Technical Summary & Methodology**:
  - Provides a systematic review of deep learning techniques for real-time vision-based hand gesture recognition.
  - Compares CNN spatial feature extraction, LSTM temporal modeling, and transformer attention mechanisms.
  - Analyzes gesture recognition challenges including background clutter, motion blur, occlusion, and latency.
  - Evaluates dataset representations across 2D RGB, 3D depth, thermal, and 21-keypoint skeletal modalities.
  - Outlines optimization strategies for deploying deep gesture recognition models on edge computing devices.

### 13. DynaKey: Dynamic Keystroke Tracking Using a Head-Mounted Camera Device
- **Authors & Year**: Hao Zhang, Yafeng Yin, Lei Xie, Tao Gu, Minghui You, and Sanglu Lu (2022)
- **Citation Key**: `Zhang2022DynaKey`
- **DOI / URL**: [10.1109/JIOT.2021.3114224](https://doi.org/10.1109/JIOT.2021.3114224)
- **Technical Summary & Methodology**:
  - Proposes DynaKey, a dynamic virtual keyboard tracking framework for smart glasses and head-mounted cameras.
  - Applies perspective transformations and homography to compensate for natural user head motion between frames.
  - Tracks printed or drawn keyboard boundaries on physical surfaces dynamically using camera and gyroscope data.
  - Detects fingertip trajectories across frame sequences to locate downward keystroke contact events.
  - Achieves high typing accuracy and robust head-motion resistance without fixed camera mounts.

### 14. Edge Computing Approach to AI-Based Gesture for Human–Robot Interaction and Control
- **Authors & Year**: Nikola Ivačko, Ivan Ćirić, and Miloš Simonović (2024)
- **Citation Key**: `Ivacko2024Edge`
- **DOI / URL**: [10.3390/computers15040241](https://doi.org/10.3390/computers15040241)
- **Technical Summary & Methodology**:
  - Presents an edge-deployable vision system for human-robot interaction using an wrist-mounted RGB camera.
  - Combines MediaPipe Hands for skeletal joint tracking with YOLO for real-time target object detection.
  - Uses ArUco planar calibration to compute transformations between hand landmarks and robot workspace coordinates.
  - Implements low-pass alpha filtering to smooth hand trajectory signals and prevent robot jerk.
  - Demonstrates touchless, real-time cobot manipulator control with low latency on edge compute nodes.

### 15. Estimation of Fingertip Force Direction With Computer Vision
- **Authors & Year**: Yu Sun, John M. Hollerbach, and Stephen A. Mascaro (2009)
- **Citation Key**: `Sun2009Estimation`
- **DOI / URL**: [10.1109/TRO.2009.2032954](https://doi.org/10.1109/TRO.2009.2032954)
- **Technical Summary & Methodology**:
  - Presents a vision method to infer fingertip force magnitude and direction from fingernail coloration patterns.
  - Registers fingernail images using RANSAC and warps sub-regions to a reference atlas with elastic registration.
  - Applies Linear Discriminant Analysis (LDA) to extract force-dependent blood volume change features.
  - Classifies 4 shear force directions and normal contact force without mounting physical force sensors on fingers.
  - Achieves 90% uncalibrated and 94% individually calibrated force direction recognition accuracy.

### 17. Finger-Gesture Recognition for Visible Light Communication Systems Using Machine Learning
- **Authors & Year**: Julian Webber, Abolfazl Mehbodniya, Rui Teng, Ahmed Arafa, and Ahmed Alwakeel (2021)
- **Citation Key**: `Webber2021Finger`
- **DOI / URL**: [10.3390/app112411582](https://doi.org/10.3390/app112411582)
- **Technical Summary & Methodology**:
  - Proposes finger gesture recognition using existing Visible Light Communication (VLC) luminaire systems.
  - Measures shadow disruptions and optical intensity variations at photodiode receivers caused by finger movement.
  - Extracts time-series light intensity features without requiring computationally expensive video cameras.
  - Trains machine learning classifiers (SVM, KNN, Random Forest) to distinguish distinct finger gestures.
  - Achieves reliable gesture detection with low power consumption for smart office and indoor HCI applications.

### 18. Fingertip-based interactive projector–camera system
- **Authors & Year**: Jun Cheng, Qun Wang, Rui Song, and Xinyu Wu (2015)
- **Citation Key**: `Cheng2015Fingertip`
- **DOI / URL**: [10.1016/j.sigpro.2014.08.043](https://doi.org/10.1016/j.sigpro.2014.08.043)
- **Technical Summary & Methodology**:
  - Presents an interactive projector-camera system enabling bare-hand touch on arbitrary flat surfaces.
  - Performs foreground hand extraction by subtracting predicted background images via geometric/photometric calibration.
  - Detects fingertip positions and uses shadow analysis to estimate Z-axis finger-to-surface distance.
  - Establishes a 3D touch decision model to detect surface contact without requiring dual camera setups.
  - Evaluates system responsiveness and touch accuracy across interactive projected GUI applications.

### 19. Gaze-Based Text Entry with Common RGB Cameras
- **Authors & Year**: Yuchi Chen (2024)
- **Citation Key**: `Chen2024Gaze`
- **DOI / URL**: [Link](https://summit.sfu.ca)
- **Technical Summary & Methodology**:
  - Develops a eye-gaze tracking text entry system operating on commodity webcams without eye trackers.
  - Extracts facial keypoints and pupil positions using deep neural networks to estimate 2D gaze vectors.
  - Maps estimated gaze fixation coordinates to on-screen virtual keyboard key boundaries.
  - Implements dwell-time and eye-blink triggers to perform key selection while mitigating Midas touch errors.
  - Demonstrates accessible text input functionality for users with motor impairments using standard RGB cameras.

### 20. HYBRID TEMPORAL FILTERING FOR STABLE TOUCHLESS DRAWING: NOISE REDUCTION IN MEDIAPIPE-BASED AIR CANVAS SYSTEMS
- **Authors & Year**: Sumit Kumar (2026)
- **Citation Key**: `Kumar2026Hybrid`
- **DOI / URL**: [Link](https://www.tijer.org)
- **Technical Summary & Methodology**:
  - Introduces a hybrid temporal filtering framework to eliminate jitter in MediaPipe-based touchless drawing systems.
  - Combines 1€ Filter jitter reduction with Moving Average exponential smoothing for 2D index fingertip trajectories.
  - Processes raw MediaPipe hand landmark coordinates in real-time to suppress micro-tremors during static pauses.
  - Maintains low latency during rapid hand movements by adaptively scaling filter cutoff frequencies.
  - Demonstrates smooth stroke rendering and reduced false line disconnections in air canvas interfaces.

### 21. HandKey: An Efficient Hand Typing Recognition using CNN for Virtual Keyboard
- **Authors & Year**: Avirmed Enkhbat, Timothy K. Shih, Noorkholis Luthfil Hakim, Wisnu Aditya, and Tipajin Thaipisutikul (2020)
- **Citation Key**: `Enkhbat2020HandKey`
- **DOI / URL**: [10.1109/SMC42975.2020.9283182](https://doi.org/10.1109/SMC42975.2020.9283182)
- **Technical Summary & Methodology**:
  - Proposes HandKey, a 3D two-hand typing motion recognition framework using a single RGB webcam.
  - Extracts multi-finger skeletal joint positions to track natural two-handed touch typing gestures in air.
  - Trains CNN classification models on spatial finger joint configurations to recognize individual key presses.
  - Eliminates the requirement for printed layouts or physical touch surfaces by defining air typing zones.
  - Demonstrates effective typing recognition accuracy with low computational hardware overhead.

### 22. Homography-Based Planar Mapping and Tracking for Mobile Phones
- **Authors & Year**: Christian Pirchheim and Gerhard Reitmayr (2011)
- **Citation Key**: `Pirchheim2011Homography`
- **DOI / URL**: [10.1109/ISMAR.2011.6092388](https://doi.org/10.1109/ISMAR.2011.6092388)
- **Technical Summary & Methodology**:
  - Presents a real-time camera tracking and mapping system based on planar homography assumptions for mobile phones.
  - Uses keyframe-based planar mapping to compute relative homographies across video frames efficiently.
  - Applies an image rectification pipeline to solve planar 3D camera pose reconstruction without full bundle adjustment.
  - Tracks continuously updated planar feature point maps to deliver robust 6DOF pose estimations.
  - Reduces computational complexity substantially while preserving acceptable localization accuracy on mobile CPUs.

### 23. Implementation of Zhang's Camera Calibration Algorithm on a Single Camera for Accurate Pose Estimation Using ArUco Markers
- **Authors & Year**: Junardo Herdiansyah, Febi Ariefka Septian Putra, and Dwi Septiyanto (2024)
- **Citation Key**: `Herdiansyah2024Implementation`
- **DOI / URL**: [10.59247/jfsc.v2i3.256](https://doi.org/10.59247/jfsc.v2i3.256)
- **Technical Summary & Methodology**:
  - Applies Zhang's camera calibration algorithm to optimize 3D pose estimation accuracy using ArUco markers.
  - Computes exact camera intrinsic matrices and radial/tangential lens distortion coefficients.
  - Evaluates corner localization error and Perspective-n-Point (PnP) pose estimation across distance ranges.
  - Achieves pose estimation accuracy exceeding 95% for single-camera autonomous navigation setups.
  - Reduces reprojection errors in marker corner extraction for reliable physical surface tracking.

### 24. Improved Pose Estimation of Aruco Tags Using a Novel 3D Placement Strategy
- **Authors & Year**: Petr Oščádal, Dominik Heczko, Aleš Vysocký, Jakub Mlotek, Petr Novák, Ivan Virgala, Marek Sukop, and Zdenko Bobovský (2020)
- **Citation Key**: `Oscadal2020Improved`
- **DOI / URL**: [10.3390/s20174825](https://doi.org/10.3390/s20174825)
- **Technical Summary & Methodology**:
  - Proposes a 3D non-coplanar placement strategy for ArUco tags to improve monocular camera pose accuracy.
  - Evaluates standard OpenCV ArUco pose estimation errors under coplanar versus 3D multi-tag layouts.
  - Mitigates ambiguity issues in homography and PnP pose estimation caused by coplanar marker degeneration.
  - Validates position and orientation accuracy using Intel RealSense RGB-D sensors and robotic ground truth.
  - Demonstrates significant reduction in orientation jitter and Z-axis distance estimation errors.

### 25. Improving Gesture Recognition Efficiency with MediaPipe and YOLO-Pose
- **Authors & Year**: Nikita Andriyanov and Svetlana Mikhailova (2025)
- **Citation Key**: `Andriyanov2025Improving`
- **DOI / URL**: [10.5194/isprs-archives-XLVIII-2-W9-2025-13-2025](https://doi.org/10.5194/isprs-archives-XLVIII-2-W9-2025-13-2025)
- **Technical Summary & Methodology**:
  - Combines MediaPipe landmark tracking with YOLO-Pose keypoint estimation for efficient gesture recognition.
  - Uses MediaPipe for fast initial hand localization and YOLO-Pose for robust multi-person pose extraction.
  - Evaluates performance on the HaGRID benchmark dataset for accuracy, frame rate, and parameter count.
  - Reduces computational complexity compared to traditional heavy 3D CNN gesture pipelines.
  - Demonstrates high inference speed suitable for real-time human-computer interaction applications.

### 26. Integrating optical finger motion tracking with surface touch events
- **Authors & Year**: Jennifer MacRitchie and Andrew P. McPherson (2015)
- **Citation Key**: `MacRitchie2015Integrating`
- **DOI / URL**: [10.3389/fpsyg.2015.00702](https://doi.org/10.3389/fpsyg.2015.00702)
- **Technical Summary & Methodology**:
  - Presents a synchronized sensing framework integrating optical motion capture with surface touch sensors.
  - Correlates 3D optical finger trajectories with capacitive/piezoelectric contact events in piano playing.
  - Analyzes finger velocity profiles before, during, and after key impact to characterize expressive touch.
  - Provides high temporal precision for mapping optical joint tracking to physical surface contact.
  - Demonstrates applications in music performance analysis and tactile human-computer interface design.

### 27. Machine Perception Of Visual Motion
- **Authors & Year**: H. Christopher Longuet-Higgins (1985)
- **Citation Key**: `LonguetHiggins1985Machine`
- **DOI / URL**: [Link](https://ceur-ws.org)
- **Technical Summary & Methodology**:
  - Formulates fundamental mathematical foundations for visual motion perception and optical flow interpretation.
  - Derives equations relating 2D image velocity vectors to 3D translational and rotational camera movement.
  - Establishes key properties of epipolar geometry, essential matrices, and planar homography constraints.
  - Analyzes ambiguity conditions when recovering 3D structure from monocular image velocity fields.
  - Provides foundational visual motion principles widely applied in modern computer vision algorithms.

### 28. Markerless Motion Capture for Pianists' Hand Movements Analysis
- **Authors & Year**: Ivan Pilkov (2024)
- **Citation Key**: `Pilkov2024Markerless`
- **DOI / URL**: [10.3389/fpsyg.2020.01159](https://jku.at)
- **Technical Summary & Methodology**:
  - Develops a markerless multi-camera motion capture system for capturing pianists' hand and finger motions.
  - Uses 3 RGB cameras and state-of-the-art hand keypoint estimation models without physical markers.
  - Creates a multimodal dataset of piano performances linking audio, MIDI, and 3D finger joint kinematics.
  - Evaluates fingertip velocity, articulation height, and key-striking temporal synchronization.
  - Demonstrates non-intrusive gesture capture for music performance analysis and skill assessment.

### 29. QWERTY Keyboard in Virtual Domain Using Image Processing
- **Authors & Year**: Pallavi Khare (2019)
- **Citation Key**: `Khare2019QWERTY`
- **DOI / URL**: [10.1109/ICICCS46578.2019.9037042](https://doi.org/10.1109/ICICCS46578.2019.9037042)
- **Technical Summary & Methodology**:
  - Proposes a paper-drawn QWERTY virtual keyboard system using classical image processing techniques.
  - Detects keyboard region boundaries and individual key boundaries using thresholding and edge detection.
  - Tracks color-segmented index fingertip centroids to identify targeted keyboard coordinate locations.
  - Detects keypress actions when fingertip motion pauses over a specific key region boundary.
  - Demonstrates a portable low-cost alternative to physical keypads for mobile computing devices.

### 30. Real Time Mono-vision Based Customizable Virtual Keyboard Using Finger Tip Speed Analysis
- **Authors & Year**: Sumit Srivastava and Ramesh Chandra Tripathi (2012)
- **Citation Key**: `Srivastava2012RealTime`
- **DOI / URL**: [10.1145/2160125.2160154](https://doi.org/10.1145/2160125.2160154)
- **Technical Summary & Methodology**:
  - Presents a monocular vision customizable virtual keyboard system based on fingertip speed analysis.
  - Allows users to define custom keyboard size, key shapes, orientation, and layout assignments on paper.
  - Uses relative fingertip velocity analysis—the typing finger moves fastest before decelerating at surface contact.
  - Handles uneven or paper surface inclinations using quadrilateral contour detection and warping.
  - Achieves accurate real-time typing recognition with single-camera hardware on unconstrained surfaces.

### 31. Real Time Webcam based Infrared Tracking for Projection Display System
- **Authors & Year**: Aniket Kudale and Kirti Wanjale (2016)
- **Citation Key**: `Kudale2016RealTime`
- **DOI / URL**: [10.5815/ijmsc.2016.04.05](https://doi.org/10.5815/ijmsc.2016.04.05)
- **Technical Summary & Methodology**:
  - Develops an infrared pen tracking projection system using an IR-filter-modified webcam and projector.
  - Tracks high-intensity IR LED emissions on flat surfaces (walls, whiteboards, tables) in real-time.
  - Applies 4-point homography calibration to map camera pixel coordinates to projected display space.
  - Detects pen tap contact and continuous drawing paths with low latency and high spatial precision.
  - Provides an affordable touch-interactive whiteboard solution using low-cost commodity components.

### 32. Real-Time Multimodal Fingertip Contact Detection via Depth and Motion Fusion for Vision-Based Human-Computer Interaction
- **Authors & Year**: Mukhiddin Toshpulatov, Wookey Lee, Suan Lee, and Geehyuk Lee (2024)
- **Citation Key**: `Toshpulatov2024RealTime`
- **DOI / URL**: [Link](https://openaccess.thecvf.com)
- **Technical Summary & Methodology**:
  - Proposes multimodal depth and motion fusion for detecting real-time fingertip surface contact events in VR/AR.
  - Fuses 3D depth map geometry with 2D motion velocity vector fields extracted from finger trajectories.
  - Eliminates false touch detections caused by hovering fingers or rapid directional hand changes.
  - Trains a temporal fusion classifier that outputs sub-frame accurate surface impact timestamps.
  - Achieves robust touch detection across arbitrary desk surfaces and virtual object manipulation tasks.

### 33. Real-time object detection method for embedded devices
- **Authors & Year**: Zicong Jiang, Liquan Zhao, Shuaiyang Li, and Yanfei Jia (2021)
- **Citation Key**: `Jiang2021RealTime`
- **DOI / URL**: [10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140)
- **Technical Summary & Methodology**:
  - Proposes a lightweight object detection model optimized for resource-constrained embedded edge devices.
  - Modifies YOLOv4-tiny by replacing CSPBlock modules with ResBlock-D structures to decrease MAC operations.
  - Introduces an auxiliary residual feature extraction network to retain small object detection accuracy.
  - Reduces total model parameter size and memory footprint while increasing frame rate efficiency.
  - Demonstrates real-time inference capability on embedded ARM and mobile GPU compute hardware.

### 34. Relative Pose Estimation and Planar Reconstruction via Superpixel-Driven Multiple Homographies
- **Authors & Year**: Xi Wang, Marc Christie, and Eric Marchand (2020)
- **Citation Key**: `Wang2020Relative`
- **DOI / URL**: [10.1109/IROS45743.2020.9341707](https://doi.org/10.1109/IROS45743.2020.9341707)
- **Technical Summary & Methodology**:
  - Proposes a method for relative camera pose estimation and planar reconstruction from RGB image pairs.
  - Extracts and matches superpixel regions across images to fit multiple homography matrices via multi-model RANSAC.
  - Introduces a voting mechanism to resolve dual-solution ambiguities during homography matrix decomposition.
  - Implements a non-linear joint optimization (bundle adjustment) over multiple planar homographies.
  - Provides accurate planar surface models and 6DOF camera pose tracking for visual SLAM.

### 35. Review Of Virtual Keyboard
- **Authors & Year**: Research Survey Authors (2020)
- **Citation Key**: `ReviewVirtualKeyboard2020`
- **DOI / URL**: [Link](https://www.ijert.org)
- **Technical Summary & Methodology**:
  - Presents a comprehensive review of virtual keyboard modalities, including optical, acoustic, and magnetic technologies.
  - Categorizes touch detection approaches into shadow analysis, color segmentation, depth sensing, and deep learning.
  - Analyzes user typing ergonomics, speed (WPM), error rates, and tactile feedback limitations.
  - Compares single-camera RGB setups with stereo vision, infrared projection, and head-mounted cameras.
  - Identifies future research directions in deep learning keypoint tracking and customized layout adaptation.

### 36. Skeleton Based Dynamic Hand Gesture Recognition using LSTM and CNN
- **Authors & Year**: Aaahm Ikram and Yue Liu (2020)
- **Citation Key**: `Ikram2020Skeleton`
- **DOI / URL**: [10.1145/3421558.3421568](https://doi.org/10.1145/3421558.3421568)
- **Technical Summary & Methodology**:
  - Proposes a dynamic hand gesture recognition framework combining 1D CNNs and LSTM networks.
  - Processes 3D skeletal joint coordinate sequences along with joint velocity vectors from Leap Motion Controller.
  - Uses CNN layers to extract spatial joint features and LSTM modules to capture temporal gesture dynamics.
  - Evaluates classification accuracy on public benchmark datasets for human-computer interaction.
  - Demonstrates low computational latency suitable for interactive AR/VR and game control.

### 37. Small Object Detection with YOLO: A Performance Analysis Across Model Versions and Hardware
- **Authors & Year**: Muhammad Fasih Tariq and Muhammad Azeem Javed (2025)
- **Citation Key**: `Tariq2025Small`
- **DOI / URL**: [Link](https://arxiv.org/abs/2504.09900)
- **Technical Summary & Methodology**:
  - Provides an extensive benchmark of YOLO object detection models (v5, v8, v9, v10, v11) for small objects.
  - Evaluates inference latency across Intel/AMD CPUs (ONNX, OpenVINO) and NVIDIA GPUs (TensorRT).
  - Analyzes detection sensitivity for objects occupying tiny image areas (1%, 2.5%, and 5% of resolution).
  - Quantifies trade-offs between backbone depth, feature pyramid network (FPN) resolution, and FPS.
  - Offers optimal model selection guidelines for deploying small-object detectors on edge hardware.

### 38. Text Input in Virtual Reality: A Preliminary Evaluation of the Drum-Like VR Keyboard
- **Authors & Year**: Costas Boletsis and Stian Kongsvik (2019)
- **Citation Key**: `Boletsis2019TextInput`
- **DOI / URL**: [10.3390/technologies7020031](https://doi.org/10.3390/technologies7020031)
- **Technical Summary & Methodology**:
  - Evaluates a drum-like controller-based virtual reality keyboard interface utilizing a drum stick metaphor.
  - Users trigger keystrokes via downward controller striking motions onto a 3D virtual keyboard plane.
  - Measures typing entry speed (WPM), character error rates, System Usability Scale (SUS) scores.
  - Reports high user engagement and positive experiential feedback due to intuitive physical movement.
  - Discusses design improvements for reducing arm fatigue during extended VR text entry sessions.

### 39. Towards a General Video-based Keystroke Inference Attack
- **Authors & Year**: Zhuolin Yang, Yuxin Chen, and Ben Y. Zhao (2022)
- **Citation Key**: `Yang2022Towards`
- **DOI / URL**: [Link](https://arxiv.org)
- **Technical Summary & Methodology**:
  - Demonstrates a video-based keystroke inference attack using a single commodity smartphone camera.
  - Operates without prior training data, keyboard layout calibration, or physical side-channel sensors.
  - Uses self-supervised finger tracking from video to label and filter keystroke motion trajectory features.
  - Trains deep neural network inference models directly on target video frames to reconstruct typed text.
  - Validates attack success across diverse environments, keyboards, and unconstrained user typing styles.

### 40. TypeNet: Towards Camera Enabled Touch Typing on Flat Surfaces through Self-Refinement
- **Authors & Year**: Ben Maman and Amit Bermano (2023)
- **Citation Key**: `Maman2023TypeNet`
- **DOI / URL**: [Link](https://arxiv.org)
- **Technical Summary & Methodology**:
  - Presents TypeNet, a real-time computer vision framework enabling touch typing on uncalibrated flat surfaces.
  - Uses a single monocular camera placed at an arbitrary front-facing angle without surface geometry calibration.
  - Adopts a deep classification model based on finger joint spatial configurations during touch impact.
  - Integrates a language model decoding layer specifically designed for real-time keystroke correction.
  - Applies self-refinement training to adapt typing models to unconstrained individual hand movement styles.

### 41. Utilizing Inpainting for Keypoint Detection for Vision-Based Control of Robotic Manipulators
- **Authors & Year**: C. Schenck et al. (2024)
- **Citation Key**: `Schenck2024Utilizing`
- **DOI / URL**: [Link](https://arxiv.org)
- **Technical Summary & Methodology**:
  - Proposes an inpainting-assisted dataset generation pipeline for markerless keypoint detection on robot manipulators.
  - Attaches ArUco markers along robot links to generate automated ground truth keypoint spatial annotations.
  - Uses image inpainting to remove ArUco markers and reconstruct clean background pixels automatically.
  - Trains deep neural keypoint detectors on synthetic markerless images for visual servoing control.
  - Demonstrates accurate robot joint tracking without requiring physical fiducial markers at runtime.

### 42. Virtual Keyboards With Real-Time and Robust Deep Learning-Based Gesture Recognition
- **Authors & Year**: Tae-Ho Lee, Sunwoong Kim, Taehyun Kim, Jin-Sung Kim, and Hyuk-Jae Lee (2022)
- **Citation Key**: `Lee2022Virtual`
- **DOI / URL**: [10.1109/THMS.2022.3165165](https://doi.org/10.1109/THMS.2022.3165165)
- **Technical Summary & Methodology**:
  - Presents a real-time deep learning gesture recognition virtual keyboard algorithm for AR/VR HMDs.
  - Designs ambidextrous virtual keyboard layouts to minimize total finger travel distance during typing.
  - Introduces a fast typing gesture action optimized for sequential adjacent keypress transitions.
  - Develops an automated dataset generation pipeline to train 7-class deep gesture networks efficiently.
  - Achieves a 1.5x speedup in typing throughput and robust performance across variable background lighting.

### 43. Virtual Touch Sensor Using a Depth Camera
- **Authors & Year**: Dong-seok Lee and Soon-kak Kwon (2019)
- **Citation Key**: `Lee2019Virtual`
- **DOI / URL**: [10.3390/s19040885](https://doi.org/10.3390/s19040885)
- **Technical Summary & Methodology**:
  - Proposes a virtual touch sensor using 3D depth cameras to transform flat surfaces into touch panels.
  - Detects touch candidate regions by isolating depth pixels within calibrated distance thresholds from the surface.
  - Identifies touch impact points by locating local minimum depth distance pixels within active regions.
  - Applies a dynamic weight filtering algorithm to correct depth noise while preserving rapid touch movement.
  - Demonstrates accurate virtual touch panel interactions for wide-area display systems.

### 44. Visual Panel: Virtual Mouse, Keyboard and 3D Controller with an Ordinary Piece of Paper
- **Authors & Year**: Zhengyou Zhang, Ying Wu, Ying Shan, and Steven Shafer (2001)
- **Citation Key**: `Zhang2001Visual`
- **DOI / URL**: [Link](https://www.microsoft.com/en-us/research)
- **Technical Summary & Methodology**:
  - Presents Visual Panel, an early interactive system transforming ordinary quad-paper into a virtual input device.
  - Uses monocular computer vision to track paper boundary quadrilaterals and calculate homography transformations.
  - Tracks fingertip tip locations relative to quad paper coordinates to emulate mouse movement and typing.
  - Enables 3D rotation and translation control by computing 3D spatial orientation of the paper panel.
  - Pioneers markerless and paper-anchored visual human-computer interaction frameworks.

### 45. Word-Level Motion Learning for Contactless QWERTY Typing with a Single Camera
- **Authors & Year**: Sung-Sic Yoo and Heung-Shik Lee (2026)
- **Citation Key**: `Yoo2026WordLevel`
- **DOI / URL**: [10.3390/s26041087](https://doi.org/10.3390/s26041087)
- **Technical Summary & Methodology**:
  - Proposes a word-level motion learning framework for contactless QWERTY typing using a single RGB camera.
  - Models complete typing gestures as spatiotemporal hand joint displacement trajectories instead of keypresses.
  - Segments typing motions temporally and accumulates direction-aware finger displacement feature vectors.
  - Achieves robust word recognition insensitive to minor hand position drift or variable typing speed.
  - Provides high word recognition accuracy for contactless text entry in AR/VR environments.

### 46. Convolutional Neural Networks and Long Short-Term Memory for skeleton-based human activity and hand gesture recognition
- **Authors & Year**: Juan C. Núñez, Raúl Cabido, Juan J. Pantrigo, Antonio S. Montemayor, and José F. Vélez (2018)
- **Citation Key**: `Nunez2018Convolutional`
- **DOI / URL**: [10.1016/j.patcog.2017.10.033](https://doi.org/10.1016/j.patcog.2017.10.033)
- **Technical Summary & Methodology**:
  - Addresses 3D skeleton-based human activity and hand gesture recognition using combined CNN and LSTM networks.
  - Uses CNN layers to extract spatial joint features and LSTM modules to capture temporal pose dynamics.
  - Introduces a two-stage training strategy: initial CNN spatial optimization followed by joint end-to-end tuning.
  - Evaluates performance on 3D full-body and hand skeleton datasets, demonstrating high classification accuracy.
  - Achieves real-time inference throughput for continuous skeletal activity recognition.

### 47. MediaPipe Hands: On-device Real-time Hand Tracking
- **Authors & Year**: Fan Zhang, Valentin Bazarevsky, Andrey Vakunov, Andrei Tkachenka, George Sung, Chuo-Ling Chang, and Matthias Grundmann (2020)
- **Citation Key**: `Zhang2020MediaPipe`
- **DOI / URL**: [Link](https://arxiv.org/abs/2006.10214)
- **Technical Summary & Methodology**:
  - Presents Google's real-time on-device hand tracking solution predicting a 21 3D/2D hand skeleton from a single RGB camera.
  - Architecture consists of a single-shot detector (palm detector) and a subsequent 21-hand-landmark keypoint regression model.
  - Achieves high real-time inference throughput on mobile GPUs and desktop CPUs without specialized depth sensors.
  - Open-sourced through MediaPipe framework, enabling ubiquitous vision-based gesture and hand interaction.

### 48. Long Short-Term Memory
- **Authors & Year**: Sepp Hochreiter and Jürgen Schmidhuber (1997)
- **Citation Key**: `Hochreiter1997LSTM`
- **DOI / URL**: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
- **Technical Summary & Methodology**:
  - Introduces the seminal Long Short-Term Memory (LSTM) recurrent neural network architecture.
  - Solves the vanishing/exploding gradient problem in traditional RNNs using constant error carousels and multiplicative gate units.
  - Enforces constant error flow across extended temporal discrete time steps.
  - Provides the mathematical foundation for temporal sequential gesture classification and time-series feature modeling.

### 49. AprilTag 2: Efficient and robust fiducial detection
- **Authors & Year**: John Wang and Edwin Olson (2016)
- **Citation Key**: `Wang2016AprilTag2`
- **DOI / URL**: [10.1109/IROS.2016.7759617](https://doi.org/10.1109/IROS.2016.7759617)
- **Technical Summary & Methodology**:
  - Redesigns the AprilTag visual fiducial marker detector for superior efficiency, higher detection rates, and lower false positive rates.
  - Introduces adaptive thresholding, sub-pixel quad boundary fitting, and accelerated decimation for high-resolution images.
  - Demonstrates high localization precision under severe perspective tilt, low resolution, and partial occlusion.

### 50. Determining and Improving the Localization Accuracy of AprilTag Detection
- **Authors & Year**: Jan Kallwies, Bianca Forkel, and Hans-Joachim Wuensche (2020)
- **Citation Key**: `Kallwies2020Determining`
- **DOI / URL**: [10.1109/ICRA40945.2020.9197314](https://doi.org/10.1109/ICRA40945.2020.9197314)
- **Technical Summary & Methodology**:
  - Evaluates localization accuracy across AprilTag 3, AprilTags C++, ArUco, and OpenCV fiducial marker libraries.
  - Proposes novel post-processing edge refinement and partial border occlusion filtering algorithms.
  - Achieves median sub-pixel corner localization error of 0.017 px under extreme perspective inclination.

### 51. ARTag, AprilTag and CALTag Fiducial Systems Comparison in a Presence of Partial Rotation
- **Authors & Year**: Ksenia Shabalina, Artur Sagitov, Leysan Sabirova, Hongbing Li, and Evgeni Magid (2018)
- **Citation Key**: `Shabalina2018ARTag`
- **DOI / URL**: [10.1007/978-3-319-93818-9_16](https://doi.org/10.1007/978-3-319-93818-9_16)
- **Technical Summary & Methodology**:
  - Conducts a comparative experimental evaluation of ARTag, AprilTag, and CALTag marker systems under partial rotation.
  - Measures detection rates, corner extraction stability, and homography error under varying camera angles.
  - Demonstrates that AprilTag yields superior robustness against rotational drift and partial marker tilt.

### 52. Fiducial Markers for Pose Estimation: Overview, Applications and Experimental Comparison
- **Authors & Year**: Michail Kalaitzakis, Brennan Cain, Sabrina Carroll, Anand Ambrosi, Camden Whitehead, and Nikolaos Vitzilaios (2021)
- **Citation Key**: `Kalaitzakis2021Fiducial`
- **DOI / URL**: [10.1007/s10846-020-01307-9](https://doi.org/10.1007/s10846-020-01307-9)
- **Technical Summary & Methodology**:
  - Provides a comprehensive survey and benchmark of ARTag, AprilTag, ArUco, and STag marker families.
  - Evaluates pose estimation accuracy, computational latency, and noise resistance across single tags and multi-tag bundles.
  - Analyzes planar homography mapping accuracy under optical noise, shadows, and motion blur.

### 53. Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks
- **Authors & Year**: Manuel Gil-Martín, Marco Raoul Marini, Iván Martín-Fernández, Sergio Esteban-Romero, and Luigi Cinque (2023)
- **Citation Key**: `GilMartin2023Hand`
- **DOI / URL**: [10.5220/0011689200003417](https://doi.org/10.5220/0011689200003417)
- **Technical Summary & Methodology**:
  - Develops a hand gesture classification framework using 21 MediaPipe hand landmark coordinates.
  - Evaluates coordinate normalization strategies, gesture representation lengths, and deep learning network architectures.
  - Demonstrates high classification accuracy on benchmark gesture datasets using skeletal joint features.

### 54. Lightweight real-time hand segmentation leveraging MediaPipe landmark detection
- **Authors & Year**: Guillermo Sánchez-Brizuela, Ana Cisnal, Eusebio de la Fuente-López, Juan-Carlos Fraile, and Javier Pérez-Turiel (2023)
- **Citation Key**: `SanchezBrizuela2023Lightweight`
- **DOI / URL**: [10.1007/s10055-023-00858-0](https://doi.org/10.1007/s10055-023-00858-0)
- **Technical Summary & Methodology**:
  - Presents a real-time algorithm leveraging MediaPipe hand landmarks for skin-tone and lighting invariant processing.
  - Processes MediaPipe joint coordinates using morphological and logical operators to generate dynamic skin masks.
  - Achieves robust real-time performance at 90 FPS on standard CPUs without specialized hardware acceleration.

### 55. Monocular Tracking of Human Hand on a Smart Phone Camera using MediaPipe and its Application in Robotics
- **Authors & Year**: Sreehari Sreenath, D. Ivan Daniels, Apparaju S. D. Ganesh, Yashaswi S. Kuruganti, and Rajeevlochana G. Chittawadigi (2021)
- **Citation Key**: `Sreenath2021Monocular`
- **DOI / URL**: [10.1109/R10-HTC53172.2021.9641542](https://doi.org/10.1109/R10-HTC53172.2021.9641542)
- **Technical Summary & Methodology**:
  - Demonstrates real-time monocular hand tracking on commodity smartphone RGB cameras using MediaPipe.
  - Maps 21 joint landmark coordinates to 2D spatial workspace coordinates with low latency.
  - Validates low-cost camera accessibility for touchless interface control without specialized hardware.

### 56. Unsupervised Gesture Segmentation by Motion Detection of a Real-Time Data Stream
- **Authors & Year**: Miguel A. Simão, Pedro Neto, and Olivier Gibaru (2016)
- **Citation Key**: `Simao2016Unsupervised`
- **DOI / URL**: [10.1109/TII.2016.2613683](https://doi.org/10.1109/TII.2016.2613683)
- **Technical Summary & Methodology**:
  - Proposes an unsupervised threshold-based gesture segmentation algorithm for real-time continuous data streams.
  - Derives velocity and acceleration vectors numerically from keypoint positions to detect movement direction inversions.
  - Segment continuous motion streams into dynamic active gestures and static idle pauses using sliding windows.

### 57. TMMF: Temporal Multi-Modal Fusion for Single-Stage Continuous Gesture Recognition
- **Authors & Year**: Harshala Gammulle, Simon Denman, Sridha Sridharan, and Clinton Fookes (2021)
- **Citation Key**: `Gammulle2021TMMF`
- **DOI / URL**: [10.1109/TIP.2021.3108420](https://doi.org/10.1109/TIP.2021.3108420)
- **Technical Summary & Methodology**:
  - Introduces a single-stage continuous gesture recognition framework (TMMF) learning gesture transitions without pre-segmentation.
  - Fuses temporal feature maps across variable-length gesture streams to detect and classify gestures concurrently.

### 58. Designing Highly Reliable Fiducial Markers
- **Authors & Year**: Mark Fiala (2010)
- **Citation Key**: `Fiala2010Designing`
- **DOI / URL**: [10.1109/TPAMI.2009.146](https://doi.org/10.1109/TPAMI.2009.146)
- **Technical Summary & Methodology**:
  - Introduces the seminal ARTag digital planar fiducial marker system in IEEE TPAMI.
  - Formulates the two-stage detection architecture: edge-gradient based candidate quad hypothesis generation and digital coding verification with Hamming distance and CRC checks.
  - Overcomes the lighting sensitivity and high false-positive rates of earlier correlation-based templates (such as ARToolkit).
  - Establishes fundamental evaluation metrics for planar fiducial systems, including false positive rates, occlusion tolerance, and minimal resolution requirements.

### 59. DeepTag: A General Framework for Fiducial Marker Design and Detection
- **Authors & Year**: Zhuming Zhang, Yongtao Hu, Guoxing Yu, and Jingwen Dai (2023)
- **Citation Key**: `Zhang2022DeepTag`
- **DOI / URL**: [10.1109/TPAMI.2022.3174603](https://doi.org/10.1109/TPAMI.2022.3174603)
- **Technical Summary & Methodology**:
  - Proposes a deep learning based general framework for fiducial marker design, keypoint regression, and digital decoding using end-to-end CNNs.
  - Utilizes dense internal keypoints (such as all internal grid cell centroids) rather than just the four outer corners to significantly improve pose and homography accuracy.
  - Supports existing marker families (AprilTag, ArUco, TopoTag, RuneTag) and allows customized local patterns.
  - Implements an on-the-fly synthetic data generation pipeline to train models without manual annotations.
  - Demonstrates superior detection robustness and pose accuracy under steep viewing angles, severe motion blur, and low image resolutions.
