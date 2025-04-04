import cv2
import numpy as np
from ultralytics import YOLO


# โหลดโมเดล
model = YOLO("runs/detect/train/weights/best.pt")

# เปิดวิดีโอ
cap = cv2.VideoCapture("Com Vision/IMG_5112.MOV")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 800, 600)

# สร้าง heatmap เปล่า
heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)

# โหลดภาพพื้นหลังที่ต้องการซ้อน heatmap
background_img = cv2.imread("Com Vision/shop_backgroud.jpg")
background_img = cv2.resize(background_img, (int(cap.get(3)), int(cap.get(4))))

# กำหนด ROI สำหรับแต่ละโต๊ะ
roi_list = [
    ((84, 549, 698, 211),2),   # โต๊ะ 1
    ((1032, 587, 377, 201),2),   # โต๊ะ 2
    ((1430, 599, 363, 206),2), # โต๊ะ 3
    
]

# ตัวแปรเพื่อเก็บจำนวนคนในแต่ละโต๊ะ
table_people_count = [0] * len(roi_list)


# ตัวแปรเพื่อเก็บพิกัดของคนที่ตรวจจับในแต่ละเฟรม
detected_people = {}

frame_counter = 0  # ตัวแปรนับเฟรม

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    
    # คำนวณทุกๆ 5 เฟรม
    if frame_counter % 5 == 0:
        results = model(frame)

        for result in results[0].boxes.data:
            x1, y1, x2, y2, confidence = result[:5].numpy()
            frame_counter = 0  
            if confidence > 0.5:  # ตรวจสอบว่า confidence สูงพอที่จะเพิ่มเข้าไปใน heatmap
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # ตรวจสอบให้ค่าพิกัดอยู่ในช่วงของ heatmap
            if 0 <= center_x < heatmap.shape[1] and 0 <= center_y < heatmap.shape[0]:
                # heatmap[center_y:center_y+1, center_x:center_x+1] += 1
                heatmap[center_y-5:center_y+5, center_x-5:center_x+5] += 5

            # ตรวจสอบว่าแต่ละคนอยู่ใน ROI ใด
            for idx, (roi, angle) in enumerate(roi_list):
                roi_x, roi_y, roi_w, roi_h = roi
                if roi_x <= center_x <= roi_x + roi_w and roi_y <= center_y <= roi_y + roi_h:
                    table_people_count[idx] += 1  # นับจำนวนคนในแต่ละ ROI

    heatmap_blur = cv2.GaussianBlur(heatmap, (25, 25), 0)
    # heatmap_blur = cv2.GaussianBlur(heatmap, (5, 5), 0)
    heatmap_max = np.max(heatmap_blur)
    if heatmap_max > 0:
        heatmap_norm = (heatmap_blur / heatmap_max * 255).astype(np.uint8)
    else:
        heatmap_norm = heatmap_blur.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # หาค่าจำนวนคนมากที่สุด
    max_people_count = max(table_people_count)
    
    for idx, (roi, angle) in enumerate(roi_list):
        roi_x, roi_y, roi_w, roi_h = roi
        roi_area = heatmap_norm[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        mean_intensity = np.mean(roi_area)  # ค่าความหนาแน่นเฉลี่ยใน ROI

        # แสดงค่า mean_intensity
        print(f"Mean intensity for Table {idx + 1}: {mean_intensity:.2f}")

        # กำหนดสีตามค่าความหนาแน่น
        if mean_intensity > 0.7:
            if table_people_count[idx] == max_people_count:  # ถ้าจำนวนคนในกล่องสูงสุด
                color = (0, 255, 255)  # สีเหลือง (สำหรับกล่องที่มีคนมากที่สุด)
        else:
            color = (255, 0, 0)  # สีฟ้า (ไม่มีความหนาแน่น)

        # วาดกล่อง
        center = (roi_x + roi_w / 2, roi_y + roi_h / 2)
        size = (roi_w, roi_h)
        rect = (center, size, angle)
        box_points = cv2.boxPoints(rect).astype(int)
        cv2.polylines(frame, [box_points], isClosed=True, color=color, thickness=2)

        # แสดงหมายเลขโต๊ะ
        cv2.putText(frame, f"Table {idx + 1}", (roi_x, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # สร้างสำเนาของภาพพื้นหลัง
    heatmap_on_background = background_img.copy()

    # ซ้อน heatmap เฉพาะบริเวณ ROI ลงบนภาพพื้นหลัง
    for idx, (roi, _) in enumerate(roi_list):
        roi_x, roi_y, roi_w, roi_h = roi

        # ดึงพื้นที่จาก heatmap สีที่ต้องการวาง
        roi_heatmap = heatmap_colored[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # วาง ROI heatmap ลงบน background
        heatmap_on_background[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = cv2.addWeighted(
        background_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w], 0.5,
        roi_heatmap, 0.5, 0)

    # แสดงผล
    output_frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    cv2.imshow("Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite("heatmap_result.png", heatmap_on_background)

cap.release()
cv2.destroyAllWindows()