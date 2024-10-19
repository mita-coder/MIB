import cv2
import numpy as np
import pandas as pd
import time

# Load YOLO model and coco labels (classes)
weights_path = "yolov4.weights"
config_path = "yolov4.cfg"
coco_names_path = "coco.names"

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Vehicle types of interest and their respective weights
vehicle_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle']
vehicle_weights = {'car': 1.0, 'truck': 2.0, 'motorbike': 0.5, 'bus': 3.0, 'bicycle': 0.3}

# Create a list to store the data temporarily
vehicle_data_list = []

# Track vehicles and their entry/exit times
start_time = {}
end_time = {}

def detect_vehicles(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    vehicle_types = []
    bounding_boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in vehicle_classes:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                bounding_boxes.append([x, y, w, h])

                # Vehicle type
                vehicle_types.append(classes[class_id])

    return vehicle_types, bounding_boxes

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_skip = 3  # Skip every 3rd frame for processing
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:  # Process only every nth frame
            vehicle_types, bounding_boxes = detect_vehicles(frame)
            current_time = time.time()

            for i, vehicle_type in enumerate(vehicle_types):
                vehicle_id = i  # For simplicity, using index as vehicle ID
                if vehicle_id not in start_time:
                    start_time[vehicle_id] = current_time

                if is_leaving_frame(bounding_boxes[i], frame.shape):
                    end_time[vehicle_id] = current_time
                    time_taken = end_time[vehicle_id] - start_time[vehicle_id]
                    weight = vehicle_weights.get(vehicle_type, 1.0)

                    vehicle_data_list.append({
                        'vehicle_type': vehicle_type,
                        'entry_time': start_time[vehicle_id],
                        'exit_time': end_time[vehicle_id],
                        'time_taken': time_taken,
                        'weight': weight
                    })

        frame_count += 1

    cap.release()

def is_leaving_frame(bbox, frame_shape):
    x, y, w, h = bbox
    frame_width, frame_height = frame_shape[1], frame_shape[0]
    return (x + w > frame_width - 20 or y + h > frame_height - 20)

# Path to your video file
video_path = "215258_small.mp4"
process_video(video_path)

# Convert the list to a DataFrame and save to CSV
vehicle_data = pd.DataFrame(vehicle_data_list)
vehicle_data.to_csv('vehicle_traffic_data.csv', index=False)
print("Vehicle data saved successfully!")
