import cv2
import time

def record_video(output_filename="camera_record.mp4", target_fps=30.0, width=1280, height=720):
    # Initialize camera capture
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Configure camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Warm-up camera exposure and sensor
    time.sleep(1.0)

    # Fetch actual frame dimensions to initialize VideoWriter correctly
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera stream opened: {actual_width}x{actual_height} @ {target_fps} FPS target")

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, target_fps, (actual_width, actual_height))

    print(f"[INFO] Recording started. Output will be saved to '{output_filename}'.")
    print("[INFO] Press 'q' in the preview window to stop recording.")

    frame_interval = 1.0 / target_fps
    next_frame_time = time.time() + frame_interval
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to grab frame.")
            break

        # Write frame to file
        out.write(frame)
        frame_count += 1

        # Display live preview window
        cv2.imshow("Recording (Press 'q' to stop)", frame)

        # Precise timing control so recording matching real-time (not fast-forwarded)
        current_time = time.time()
        sleep_time = next_frame_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        next_frame_time += frame_interval

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Cleanup resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Recording finished and saved to '{output_filename}'.")
    print(f"Total Frames: {frame_count}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Actual FPS Recorded: {actual_fps:.2f}")

if __name__ == "__main__":
    record_video()
