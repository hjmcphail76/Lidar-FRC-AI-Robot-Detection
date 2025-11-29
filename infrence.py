import infrence
import numpy as np
import asyncio
from ultralytics import YOLO
from PIL import Image
import os
import plot
import nt_interface

# Load your trained YOLOv8 weights
model = YOLO('best.pt')

# # Input and output directories
# input_dir = 'photo-output-results'
# output_dir = 'predictions'
# os.makedirs(output_dir, exist_ok=True)


def make_inference(img):
    results = model.predict(img, conf=.5, verbose=False)[0]

    img_h, img_w = img.shape[:2]

    # The pygame screen size
    screen_w = 800
    screen_h = 600

    # for box in results.boxes:
    if len(results.boxes) == 0:
        return results
    x1, y1, x2, y2 = results.boxes[0].xyxy[0].tolist()

    # SCALE YOLO coordinates from image-size → screen-size
    sx1 = x1 * (screen_w / img_w)
    sx2 = x2 * (screen_w / img_w)
    sy1 = y1 * (screen_h / img_h)
    sy2 = y2 * (screen_h / img_h)

    plot.enqueue_box(sx1, sy1, sx2, sy2)

    nt_interface.publish_pose(
        (x1 + x2) / 2 / 1000,  # convert to meters
        (y1 + y2) / 2 / 1000,
        0
    )

    return results


image_queue = asyncio.Queue()


def enqueue_image(img):
    image_queue.put_nowait(img)


async def run_inference_detector():
    while True:
        latest = None

        # Drain the queue, keeping only the latest batch
        while not image_queue.empty():
            latest = await image_queue.get()

        if latest is not None:

            results = make_inference(latest)
            if results is None:
                return

            # results.save(
            #     "predictions/live_prediction.png")

        await asyncio.sleep(0.1)


def get_start_data_collection():
    """Return the coroutine for use in asyncio.gather."""
    return run_inference_detector()
