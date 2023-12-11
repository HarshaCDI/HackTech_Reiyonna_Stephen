from boxmot import StrongSORT
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Function to create a video writer
def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

# Function to estimate speed
def estimate_speed(location1, location2, frame_rate=30, ppm=8):
    d_pixel = np.sqrt(np.power(location2[0] - location1[0], 2) + np.power(location2[1] - location1[1], 2))
    d_meters = d_pixel / ppm
    time_constant = 1 / frame_rate
    speed = d_meters / time_constant
    return round(speed)

if __name__ == "__main__":
    # Configuration
    device = "cuda:0"  # Choose appropriate device
    fp16 = True  # True if GPU available

    # Path to pre-trained YOLOv8 model
    yolo_model_path = '../runs/detect/train4/weights/best.pt'

    # Path to ReID model weights
    reid_model_weights = 'mobilenetv2_x1_4_dukemtmcreid.pt'

    # Video source path
    video_source = '../Test datasets/scene01_01.mp4'

    # Output video path
    output_video_path = '../TrackingResults/StrongSORT.mp4'

    # Output text file path
    result_file_path = '../speedResult.txt'

    # Initialize YOLO model
    yolo_model = YOLO(yolo_model_path)

    # Initialize tracking model
    tracker = StrongSORT(
        model_weights=Path(reid_model_weights),
        device=device,
        fp16=fp16,
    )
    tracker.n_init = 1

    # Open video file
    vid = cv2.VideoCapture(video_source)

    # Create video writer
    writer = create_video_writer(vid, output_video_path)

    frame_rate = 30  # Assuming the videos are 30 fps
    total_detections = 0
    speed_detections = {}

    while True:
        ret, frame = vid.read()

        if not ret:
            break

        # Predict detections using YOLO
        results = yolo_model(frame)
        detections = results[0]['boxes'].xyxy if results and 'boxes' in results[0] else []

        # Initialize the list of bounding boxes and confidences
        bounding_boxes = []

        if not detections:
            detections = [[0, 0, 0, 0, 0.0922948837280273, 0]]  # Dummy detection if none

        # Update the tracker with the detected bounding boxes
        valid_detections = []

        for detection in detections:
            x1, y1, x2, y2 = detection[:4]

            # Ensure the bounding box coordinates are within image dimensions
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))

            if x1 < x2 and y1 < y2:
                valid_detections.append([x1, y1, x2, y2] + detection[4:])

        # Reshape the array
        valid_detections = np.array(valid_detections).reshape(-1, 6)

        ts = tracker.update(valid_detections, frame)

        if ts.size > 0:
            xyxys = ts[0][:, :4].astype('int')  # Assuming the tracking results are in the first element
            ids = ts[0][:, 4].astype('int')
            confs = ts[0][:, 5]

            for xyxy, obj_id, conf in zip(xyxys, ids, confs):
                # Estimate vehicle speed (replace this with the actual speed estimation logic)
                speed = estimate_speed((0, 0), (0, 0))

                result_line = f"{video_source} {vid.get(cv2.CAP_PROP_POS_FRAMES)} {obj_id} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]} {speed} {conf}\n"
                with open(result_file_path, "a") as result_file:
                    result_file.write(result_line)

                total_detections += 1

                # Draw bounding box and information on the frame
                frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                cv2.putText(frame, f'id: {obj_id}, conf: {conf}', (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show the frame
        writer.write(frame)

    # Release video capture and writer
    vid.release()
    writer.release()
