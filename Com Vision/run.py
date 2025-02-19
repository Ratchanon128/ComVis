import cv2
import time
from ultralytics import YOLO

# โหลดโมเดล
model = YOLO("runs/detect/train/weights/best.pt")

# เปิดวิดีโอ
# cap = cv2.VideoCapture("video/test1.mp4")
cap = cv2.VideoCapture("video/test2.MOV")

# ลดขนาดบัฟเฟอร์เพื่อลดดีเลย์
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# สร้างหน้าต่างที่สามารถปรับขนาดได้
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 800, 600)

# กำหนด ROI สำหรับแต่ละโต๊ะ
roi_list = [
    ((0, 800, 600, 100), 5),   # โต๊ะ 1
    ((0, 725, 750, 80), 5),   # โต๊ะ 2
    ((125, 675, 775 , 80), 5), # โต๊ะ 3
    ((225, 625, 775 , 80), 5),   # โต๊ะ 4
    ((325, 600, 775 , 70), 5),   # โต๊ะ 5
    ((425, 590, 775 , 60), 5),   # โต๊ะ 6
    ((700, 565, 525 , 60), 5),   # โต๊ะ 7
    ((775, 545, 525 , 60), 5),   # โต๊ะ 8
    ((950, 530, 400 , 60), 5),   # โต๊ะ 9
    ((1000, 520, 400 , 60), 5),   # โต๊ะ 10

    ((750, 925, 1000, 100), 10),   # โต๊ะ 11
    ((850, 825, 900, 100), 10),   # โต๊ะ 12
    ((950, 775, 900 , 100), 8), # โต๊ะ 13
    ((1150, 700, 700 , 100), 8),   # โต๊ะ 14
    ((1150, 650, 700 , 100), 5),   # โต๊ะ 15
    ((1250, 625, 600 , 80), 5),   # โต๊ะ 16
    ((1300, 610, 600 , 60), 5),   # โต๊ะ 17
    ((1350, 590, 550 , 60), 5),   # โต๊ะ 18
    ((1430, 570, 475 , 60), 5),   # โต๊ะ 19
    ((1450, 550, 475 , 60), 5),   # โต๊ะ 20
]

prev_time = 0  # ตัวแปรเก็บเวลาของเฟรมก่อนหน้า

while True:
    start_time = time.time()  # เวลาที่เริ่มประมวลผลเฟรม

    ret, frame = cap.read()
    if not ret:
        break
    
    # ตรวจจับวัตถุบนเฟรม
    results = model(frame)

    annotated_frame = results[0].plot()
    # คำนวณ FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # เก็บจำนวนคนที่อยู่ในแต่ละ ROI
    roi_counts = {i: 0 for i in range(len(roi_list))}

    # ตรวจสอบว่า bounding box ของคนอยู่ใน ROI ไหน
    for result in results[0].boxes.data:
        x1, y1, x2, y2 = result[:4].numpy()  # ขอบเขตของคน (bounding box)
        
        # เช็คว่า bounding box อยู่ใน ROI ไหน
        for idx, (roi, _) in enumerate(roi_list):
            roi_x, roi_y, roi_w, roi_h = roi
            if roi_x < x1 < roi_x + roi_w and roi_y < y1 < roi_y + roi_h:
                roi_counts[idx] += 1

    # หาว่า ROI ไหนที่มีคนอยู่มากที่สุด
    max_people_roi = max(roi_counts, key=roi_counts.get)
    
    # แสดงข้อความว่ามีคนอยู่ใน ROI ไหนมากที่สุด
    cv2.putText(annotated_frame, f"Most crowded ROI: Table {max_people_roi + 1}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # แสดงค่า FPS ลงบนวิดีโอ
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # แสดงภาพในหน้าต่าง
    cv2.imshow("Detection", annotated_frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
