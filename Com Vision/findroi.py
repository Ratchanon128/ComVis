import cv2

video_path = "Com Vision/IMG_5112.MOV"  # เปลี่ยนเป็น path ของวิดีโอจริง
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)

cv2.namedWindow("Select Table ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Table ROI", 800, 600)


ret, frame = cap.read()
if ret:
    roi = cv2.selectROI("Select Table ROI", frame, fromCenter=False, showCrosshair=True)
    print("Selected ROI:", roi)  # ค่า ROI จะอยู่ในรูปแบบ (x, y, width, height)


cap.release()
cv2.destroyAllWindows()