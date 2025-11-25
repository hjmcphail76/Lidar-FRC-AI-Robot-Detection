import asyncio
from PIL import Image
import numpy as np
import os

# Queue to store batches of points NOT lines here
point_queue = asyncio.Queue()


def enqueue_points(xs, ys):
    """Add a new batch of points to the queue."""
    point_queue.put_nowait((xs, ys))


async def run_save_data():
    lidar_batches = []
    dataCount = 0

    while True:
        latest = None

        # Drain the queue, keeping only the latest batch
        while not point_queue.empty():
            xs, ys = await point_queue.get()
            latest = (xs, ys)

        if latest is not None:
            xs = latest[0]
            ys = latest[1]

            print(f"Saving batch {dataCount} with {len(xs)} points")
            data = lidar_to_image(xs, ys)

            # Convert to PIL format image
            img = Image.fromarray(data)

            # Save to disk
            img.save(f"photo-output-results/output_image_pil{dataCount}.png")
            dataCount += 1

        else:
            print("No new data")

        await asyncio.sleep(2)


def lidar_to_image(xs, ys,
                   img_width=360, img_height=256, max_range=200):

    img = np.zeros((img_height, img_width), dtype=np.uint8)

    x = np.array(xs).flatten()
    y = np.array(ys).flatten()

    x_min, x_max = -max_range, max_range

    # invert Y for image coordinates by swapping +/- for y min/max
    y_min, y_max = max_range, -max_range

    # Map XY to image coordinates
    cols = ((x - x_min) / (x_max - x_min) * (img_width - 1)).astype(int)
    rows = ((y_max - y) / (y_max - y_min) * (img_height - 1)).astype(int)

    # Clamp indices
    cols = np.clip(cols, 0, img_width - 1)
    rows = np.clip(rows, 0, img_height - 1)

    # Draw points as white on the black background
    img[rows, cols] = 255

    # img[img_height//2, img_width//2] = 128 #marks center as sensor position

    return img


def get_start_data_collection():
    """Return the coroutine for use in asyncio.gather."""
    return run_save_data()
