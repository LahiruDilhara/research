import cv2
from dt_apriltags import Detector

# 1. Initialize the detector
detector = Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=1.0, 
    quad_sigma=0.0,
    refine_edges=1
)

# 2. Initialize camera capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open video source.")

print("Starting video feed. Press 'q' to exit.")

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detect tags
    tags = detector.detect(gray)

    # 4. Draw the results
    for tag in tags:
        corners = tag.corners.astype(int)
        
        # Draw green polygon
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw red center
        center = tuple(tag.center.astype(int))
        cv2.circle(frame, center, radius=4, color=(0, 0, 255), thickness=-1)
        
        # Label ID
        cv2.putText(frame, f"ID: {tag.tag_id}", (center[0] - 15, center[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 5. Display the stream
    cv2.imshow('AprilTag Live Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()