import cv2
import time

def measure_actual_fps(src=0, width=1280, height=720, target_fps=30):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    
    # Set requested parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Measuring actual FPS for 5 seconds at requested {target_fps} FPS ({width}x{height})...")
    print("Please keep the camera running.")

    frame_count = 0
    start_time = time.time()
    duration = 5.0  # test duration in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        frame_count += 1
        
        # Stop after 5 seconds
        if time.time() - start_time >= duration:
            break

    total_time = time.time() - start_time
    cap.release()

    # Calculate actual FPS
    actual_fps = frame_count / total_time

    print(f"\n--- Results ---")
    print(f"Total Frames Captured: {frame_count}")
    print(f"Time Elapsed: {total_time:.2f} seconds")
    print(f"Actual Measured FPS: {actual_fps:.2f}")

if __name__ == "__main__":
    # You can change width, height, or target_fps here
    measure_actual_fps(src=0, width=1280, height=720, target_fps=30)