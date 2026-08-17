import cv2
import time
import threading

class VideoStreamCapture:
    def __init__(self, src=0, width=1280, height=720, fps=30):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        
        # Configure camera resolution and target FPS
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        
        # Force MJPG format to allow high resolutions at 30fps over USB
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    # Camera index (0 is usually /dev/video0)
    camera_source = 0
    target_fps = 30.0
    width, height = 1280, 720

    print("[INFO] Starting video stream...")
    video_capture = VideoStreamCapture(src=camera_source, width=width, height=height, fps=int(target_fps)).start()
    time.sleep(2) # Warm-up time for auto-exposure/sensor lock

    # Query actual dimensions from captured frame to ensure exact match
    grabbed, sample_frame = video_capture.read()
    if grabbed and sample_frame is not None:
        height, width = sample_frame.shape[:2]

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, target_fps, (width, height))

    print("[INFO] Recording... Press 'q' in the video window to stop.")
    
    frame_count = 0
    start_time = time.time()
    
    # Target frame interval for 30 FPS (33.33 milliseconds)
    frame_interval = 1.0 / target_fps
    next_frame_time = time.time() + frame_interval

    try:
        while True:
            grabbed, frame = video_capture.read()
            if not grabbed or frame is None:
                print("[WARNING] Frame dropped or camera disconnected.")
                break

            # Write frame to file
            out.write(frame)
            frame_count += 1

            # Display live preview
            cv2.imshow('Recording (Press Q to stop)', frame)

            # Control loop timing to maintain exact 30 FPS pacing
            current_time = time.time()
            sleep_time = next_frame_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_frame_time += frame_interval

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        elapsed_time = time.time() - start_time
        video_capture.stop()
        out.release()
        cv2.destroyAllWindows()
        
        actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"\n[INFO] Recording finished.")
        print(f"Total Frames: {frame_count}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Achieved Average FPS: {actual_fps:.2f}")

if __name__ == "__main__":
    main()