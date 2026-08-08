import cv2
import cv2.aruco as aruco

# Load the image
image = cv2.imread('./img/aruco15.png')
if image is None:
    raise FileNotFoundError("Could not find the image file.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Adaptive Thresholding
# This helps separate the marker grid from the background in variable lighting
# 11 is the block size (must be odd), 2 is the constant subtracted from the mean
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Get all available ArUco dictionary constants
dictionary_names = [name for name in dir(aruco) if name.startswith("DICT_")]

print(f"Testing {len(dictionary_names)} dictionaries using adaptive thresholding...")

for name in dictionary_names:
    dict_type = getattr(aruco, name)
    dictionary = aruco.getPredefinedDictionary(dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    # Detect markers on the thresholded image
    corners, ids, rejected = detector.detectMarkers(thresh)
    
    if ids is not None and len(ids) > 0:
        print(f"Found markers in dictionary: {name}")
        print(f"Detected IDs: {ids.flatten()}")
        
        # Optional: Visualization for debugging
        # aruco.drawDetectedMarkers(image, corners, ids)
        # cv2.imshow('Detection', image)
        # cv2.waitKey(0)