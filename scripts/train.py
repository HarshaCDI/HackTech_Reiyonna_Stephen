import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../data/data.yaml")
model_path = os.path.join(script_dir, "../models/yolov8n.pt")

# Run the yolo train command
command = f"yolo train model={model_path} data={data_path} epochs=5 imgsz=640"
os.system(command)
