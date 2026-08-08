import cv2
import cv2.aruco as aruco

diconary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(diconary, parameters)

image = cv2.imread("./img/aruco14.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the same adaptive thresholding used in your scanner
thresh = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 11, 2
)

# Use the thresholded image for detection
corners, ids, rejected = detector.detectMarkers(thresh)

if ids is not None:
    aruco.drawDetectedMarkers(image, corners, ids)
    print(f"Detected {len(ids)} markers.")
else:
    print("No markers detected.")

cv2.imshow('Detected Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

