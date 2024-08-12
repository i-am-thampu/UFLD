import cv2
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import time

# Path to the model and video
model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = True
video_path = "road.mp4"

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Function to calculate lane angle and space area
def calculate_lane_info(lanes_points):
    angles = []
    space_areas = []
    for lane_points in lanes_points:
        if len(lane_points) > 1:
            # Calculate angle
            dx = lane_points[-1][0] - lane_points[0][0]
            dy = lane_points[-1][1] - lane_points[0][1]
            angle = np.arctan2(dy, dx) * 180.0 / np.pi
            angles.append(angle)
            # Calculate space area (simple approximation)
            space_area = cv2.contourArea(np.array(lane_points))
            space_areas.append(space_area)
    return angles, space_areas

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time.time()
    
    # Detect the lanes
    output_frame = lane_detector.detect_lanes(frame)
    
    # Calculate lane info
    angles, space_areas = calculate_lane_info(lane_detector.lanes_points)
    
    # Display angles and space areas on the frame
    for i, (angle, space_area) in enumerate(zip(angles, space_areas)):
        text = f"Lane {i+1}: Angle={angle:.2f}, Area={space_area:.2f}"
        cv2.putText(output_frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time
    cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Write the frame to the output video
    out.write(output_frame)
    
    # Display the resulting frame
    cv2.imshow('Detected lanes', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

