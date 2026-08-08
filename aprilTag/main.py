import cv2
import numpy as np
from dt_apriltags import Detector

# 1. Initialize the robust AprilTag detector
detector = Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=1.0, 
    quad_sigma=0.0,
    refine_edges=1
)

# 2. Load the image
image = cv2.imread('./img/aprilTags24.jpeg')
# blurred_image = cv2.GaussianBlur(image, (21,21), 0)  # Apply Gaussian blur to reduce noise
blurred_image = image
if image is None:
    raise FileNotFoundError("Could not find the image.")

gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

# 3. Detect tags
tags = detector.detect(gray)
print(f"Detected {len(tags)} tags.")

# 4. Draw the results manually
for tag in tags:
    # Extract the four corners and convert them to integers for OpenCV drawing
    corners = tag.corners.astype(int)
    
    # Draw a green polygon around the tag
    cv2.polylines(blurred_image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw a red dot at the center
    center = tuple(tag.center.astype(int))
    cv2.circle(blurred_image, center, radius=4, color=(0, 0, 255), thickness=-1)
    
    # Write the Tag ID near the center
    cv2.putText(blurred_image, f"ID: {tag.tag_id}", (center[0] - 15, center[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 5. Display the final image
cv2.imshow('dt-apriltags Detection', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()