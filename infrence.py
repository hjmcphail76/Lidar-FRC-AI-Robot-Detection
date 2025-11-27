from ultralytics import YOLO
from PIL import Image
import os

# Load your trained YOLOv8 weights
model = YOLO('best.pt')

# Input and output directories
input_dir = 'photo-output-results'
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)

# Iterate over all image files
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)

        # Open image and convert to RGB (important for grayscale LiDAR images)
        img = Image.open(img_path).convert('RGB')

        # Run inference with lower confidence to catch small predictions
        # 💀 2%, we underfitting sooo bad
        results = model.predict(img, conf=0.02)

        # Number of detected objects
        num_boxes = len(results[0].boxes)
        print(f"{filename}: {num_boxes} objects detected")

        # Save annotated image
        results[0].save(os.path.join(output_dir, filename))
