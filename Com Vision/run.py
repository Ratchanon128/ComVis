import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# Open video
cap = cv2.VideoCapture("IMG_5112.MOV")

background_img = cv2.imread("shop_backgroud.jpg")

if not cap.isOpened():
    print("ไม่สามารถเปิดวิดีโอได้! ตรวจสอบ path หรือฟอร์แมตไฟล์อีกครั้ง")
else:
    print("เปิดวิดีโอได้! กำลังทำการวิเคราะห์...")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 800, 600)

# Create two heatmaps: one for display (with decay) and one to store total data
display_heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)
total_heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)

# Define ROIs for each table/store
roi_list = [
    ((140, 544, 580, 217), 2),   # Table/Store 1
    ((1032, 587, 377, 201), 2),   # Table/Store 2
    ((1430, 599, 363, 206), 2),   # Table/Store 3
]

# Variables to store people count in each table
table_people_count = [0] * len(roi_list)

# Data storage for analysis
density_data = {
    'timestamp': [],
    'frame_number': [],
}

# Initialize dictionaries for each table
for i in range(len(roi_list)):
    density_data[f'table_{i+1}_intensity'] = []
    density_data[f'table_{i+1}_people'] = []

# Store the visit duration data
visit_data = defaultdict(lambda: defaultdict(list))
person_tracking = {}
next_person_id = 0

frame_counter = 0
start_time = time.time()
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Decay factor for display heatmap (lower = faster decay)
decay_factor = 0.95

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    current_time = start_time + (frame_counter / fps)
    timestamp = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
    
    # Apply decay to display heatmap
    display_heatmap = display_heatmap * decay_factor
    
    # Process every 10 frames
    if frame_counter % 10 == 0:
        results = model(frame)
        
        # Reset table people count for this frame
        table_people_at_frame = [0] * len(roi_list)
        current_people = set()
        
        for result in results[0].boxes.data:
            x1, y1, x2, y2, confidence = result[:5].numpy()
            if confidence > 0.5:
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Add to both heatmaps
                if 0 <= center_x < display_heatmap.shape[1] and 0 <= center_y < display_heatmap.shape[0]:
                    # Add to display heatmap (with decay)
                    display_heatmap[center_y-5:center_y+5, center_x-5:center_x+5] += 5
                    
                    # Add to total heatmap (accumulates over time)
                    total_heatmap[center_y-5:center_y+5, center_x-5:center_x+5] += 5
                
                # Check which ROI this person belongs to
                person_found = False
                for idx, (roi, angle) in enumerate(roi_list):
                    roi_x, roi_y, roi_w, roi_h = roi
                    if roi_x <= center_x <= roi_x + roi_w and roi_y <= center_y <= roi_y + roi_h:
                        table_people_at_frame[idx] += 1
                        
                        # Simple tracking - assign an ID based on position
                        person_id = None
                        min_distance = float('inf')
                        
                        # Try to match with existing people
                        for pid, info in person_tracking.items():
                            if info['active']:
                                p_x, p_y = info['position']
                                distance = np.sqrt((center_x - p_x)**2 + (center_y - p_y)**2)
                                if distance < min_distance and distance < 50:  # 50px threshold
                                    min_distance = distance
                                    person_id = pid
                        
                        # If no match, create new person ID
                        if person_id is None:
                            person_id = next_person_id
                            next_person_id += 1
                            person_tracking[person_id] = {
                                'position': (center_x, center_y),
                                'first_seen': frame_counter,
                                'last_seen': frame_counter,
                                'current_table': idx,
                                'tables_visited': {idx: [(frame_counter, frame_counter)]},
                                'active': True,
                                'visit_confirmed': False
                            }
                        else:
                            # Update existing person
                            person_tracking[person_id]['position'] = (center_x, center_y)
                            person_tracking[person_id]['last_seen'] = frame_counter
                            
                            # If table changed, record it
                            if person_tracking[person_id]['current_table'] != idx:
                                person_tracking[person_id]['current_table'] = idx
                                if idx not in person_tracking[person_id]['tables_visited']:
                                    person_tracking[person_id]['tables_visited'][idx] = []
                                person_tracking[person_id]['tables_visited'][idx].append((frame_counter, frame_counter))
                            else:
                                # Update end time of current visit
                                current_visits = person_tracking[person_id]['tables_visited'][idx]
                                if current_visits:
                                    start_frame, _ = current_visits[-1]
                                    current_visits[-1] = (start_frame, frame_counter)
                        
                        current_people.add(person_id)
                        person_found = True
                        break
        
        # Mark people as inactive if not seen in this frame
        for pid in person_tracking:
            if pid not in current_people and person_tracking[pid]['active']:
                if frame_counter - person_tracking[pid]['last_seen'] > 15:  # Lost for 15 frames
                    person_tracking[pid]['active'] = False
        
        # Update table people counts
        table_people_count = table_people_at_frame
        
        # Calculate and store density data using the total heatmap
        total_heatmap_blur = cv2.GaussianBlur(total_heatmap, (25, 25), 0)
        total_max = np.max(total_heatmap_blur)
        if total_max > 0:
            total_norm = (total_heatmap_blur / total_max * 255).astype(np.uint8)
        else:
            total_norm = total_heatmap_blur.astype(np.uint8)
        
        # Record data
        density_data['timestamp'].append(timestamp)
        density_data['frame_number'].append(frame_counter)
        
        for idx, (roi, angle) in enumerate(roi_list):
            roi_x, roi_y, roi_w, roi_h = roi
            roi_area = total_norm[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            mean_intensity = np.mean(roi_area)
            
            # Store the data
            density_data[f'table_{idx+1}_intensity'].append(mean_intensity)
            density_data[f'table_{idx+1}_people'].append(table_people_at_frame[idx])
    
    # Visual display (for monitoring) using display_heatmap with decay
    display_heatmap_blur = cv2.GaussianBlur(display_heatmap, (25, 25), 0)
    display_max = np.max(display_heatmap_blur)
    if display_max > 0:
        display_norm = (display_heatmap_blur / display_max * 255).astype(np.uint8)
    else:
        display_norm = display_heatmap_blur.astype(np.uint8)
    display_colored = cv2.applyColorMap(display_norm, cv2.COLORMAP_JET)
    
    # Display ROIs and statistics
    for idx, (roi, angle) in enumerate(roi_list):
        roi_x, roi_y, roi_w, roi_h = roi
        roi_area = display_norm[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        mean_intensity = np.mean(roi_area)
        
        # Determine color based on density
        if mean_intensity > 0.5:
            color = (0, 165, 255)      # Orange for high density
        else:
            color = (255, 0, 0)        # Blue for low density
        
        # Draw box
        center = (roi_x + roi_w / 2, roi_y + roi_h / 2)
        size = (roi_w, roi_h)
        rect = (center, size, angle)
        box_points = cv2.boxPoints(rect).astype(int)
        cv2.polylines(frame, [box_points], isClosed=True, color=color, thickness=2)
        
        # Show table number and metrics
        cv2.putText(frame, f"Store {idx + 1}: {table_people_count[idx]} people", 
                   (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Density: {mean_intensity:.1f}", 
                   (roi_x, roi_y + roi_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add progress information
    progress = int((frame_counter / total_frames) * 100)
    cv2.putText(frame, f"Processing: {progress}%", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combined output
    output_frame = cv2.addWeighted(frame, 0.7, display_colored, 0.3, 0)
    cv2.imshow("Detection", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save collected data
df = pd.DataFrame(density_data)
df.to_csv('store_density_data.csv', index=False)

# Create summary statistics
summary = {
    'store': [],
    'avg_intensity': [],
    'max_intensity': [],
    'avg_people': [],
    'max_people': [],
    'total_visits': [],
    'avg_visit_duration': []
}

# Process tracking data for visit statistics
MIN_VISIT_DURATION_SECONDS = 5

# Process tracking data for visit statistics
for idx in range(len(roi_list)):
    # Count total visits per table
    total_visits = 0
    visit_durations = []
    
    for person_id, info in person_tracking.items():
        if idx in info['tables_visited']:
            # Filter visits by minimum duration
            valid_visits = []
            for start_frame, end_frame in info['tables_visited'][idx]:
                duration_frames = end_frame - start_frame
                duration_seconds = duration_frames / fps
                if duration_seconds > MIN_VISIT_DURATION_SECONDS:
                    valid_visits.append((start_frame, end_frame))
                    visit_durations.append(duration_seconds)
            
            # Only count visits that meet the minimum duration
            total_visits += len(valid_visits)

    # Calculate statistics
    avg_visit_duration = np.mean(visit_durations) if visit_durations else 0
    
    summary['store'].append(f'Store {idx+1}')
    summary['avg_intensity'].append(np.mean(density_data[f'table_{idx+1}_intensity']))
    summary['max_intensity'].append(np.max(density_data[f'table_{idx+1}_intensity']))
    summary['avg_people'].append(np.mean(density_data[f'table_{idx+1}_people']))
    summary['max_people'].append(np.max(density_data[f'table_{idx+1}_people']))
    summary['total_visits'].append(total_visits)
    summary['avg_visit_duration'].append(avg_visit_duration)

summary_df = pd.DataFrame(summary)
summary_df.to_csv('store_summary_statistics.csv', index=False)

# Generate visualization of store statistics
plt.figure(figsize=(15, 10))

# Plot average densities
plt.subplot(2, 2, 1)
plt.bar(summary['store'], summary['avg_intensity'])
plt.title('Average Density by Store')
plt.ylabel('Average Intensity')

# Plot people count
plt.subplot(2, 2, 2)
plt.bar(summary['store'], summary['avg_people'])
plt.title('Average People Count by Store')
plt.ylabel('Average People Count')

# Plot total visits
plt.subplot(2, 2, 3)
plt.bar(summary['store'], summary['total_visits'])
plt.title('Total Visits by Store')
plt.ylabel('Number of Visits')

# Plot average visit duration
plt.subplot(2, 2, 4)
plt.bar(summary['store'], summary['avg_visit_duration'])
plt.title('Average Visit Duration by Store')
plt.ylabel('Duration (seconds)')

plt.tight_layout()
plt.savefig('store_statistics.png')

# Create a separate visualization for the final heatmap with background image
plt.figure(figsize=(12, 8))

# Resize background image to match frame dimensions if needed
bg_resized = cv2.resize(background_img, (int(cap.get(3)), int(cap.get(4))))
bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)

# Display background image first
plt.imshow(bg_rgb)

# Then overlay the heatmap with transparency
total_heatmap_blur = cv2.GaussianBlur(total_heatmap, (25, 25), 0)
total_max = np.max(total_heatmap_blur)
if total_max > 0:
    total_normalized = total_heatmap_blur / total_max
else:
    total_normalized = total_heatmap_blur

# Display the heatmap with alpha transparency
heatmap_overlay = plt.imshow(total_normalized, cmap='jet', alpha=0.7)
plt.colorbar(heatmap_overlay, label='Normalized Density')
plt.title('Total Accumulated Density Heatmap')

# Draw ROI boundaries on the heatmap
for idx, (roi, angle) in enumerate(roi_list):
    roi_x, roi_y, roi_w, roi_h = roi
    # Convert to matplotlib rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle((roi_x, roi_y), roi_w, roi_h, 
                     linewidth=2, edgecolor='white', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(roi_x + 3, roi_y + 3, f"Store {idx+1}", 
         color='white', ha='left', va='top', fontweight='bold',)

plt.savefig('total_heatmap.png')

# Display results using the original print statements
print("\n==== Store Traffic Analysis ====")
print(summary_df.to_string(index=False))
print("\nMost popular store by average density:", 
      summary_df.loc[summary_df['avg_intensity'].idxmax()]['store'])
print("Most popular store by average people count:", 
      summary_df.loc[summary_df['avg_people'].idxmax()]['store'])
print("Most popular store by total visits:", 
      summary_df.loc[summary_df['total_visits'].idxmax()]['store'])
print("Store with longest average visits:", 
      summary_df.loc[summary_df['avg_visit_duration'].idxmax()]['store'])

# Show plots at the end
plt.show()

cap.release()
cv2.destroyAllWindows()
print("\nAnalysis complete. Data saved to CSV files.")
print("Visualizations saved as:")
print("- 'store_statistics.png' (Overall statistics)")
print("- 'total_heatmap.png' (Complete accumulated density heatmap)")