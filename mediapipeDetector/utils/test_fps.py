import cv2

cap = cv2.VideoCapture('hi.webm')
print("FPS prop:", cap.get(cv2.CAP_PROP_FPS))
print("Frame count prop:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if count < 5 or count > 138:
        print(f"Frame {count}: {msec} ms")
    count += 1
print("Total frames read:", count)
cap.release()
