import cv2
import numpy as np

def check_duplicates(src=0):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot open camera.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_frames = 1
    duplicate_count = 0

    print("Analyzing 300 frames for duplicates... Keep the camera still or move slowly.")
    
    for _ in range(300):
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean squared difference between consecutive frames
        err = np.sum((prev_gray.astype("float") - gray.astype("float")) ** 2)
        err /= float(prev_gray.shape[0] * prev_gray.shape[1])
        
        # If error is near 0, the frame is a duplicate
        if err < 0.1:
            duplicate_count += 1
            
        prev_gray = gray

    cap.release()
    print(f"\n--- Diagnostic Results ---")
    print(f"Total Frames Checked: {total_frames}")
    print(f"Duplicate Frames Detected: {duplicate_count} ({(duplicate_count/total_frames)*100:.1f}%)")

if __name__ == "__main__":
    check_duplicates()