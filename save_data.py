import asyncio
from PIL import Image
import numpy as np
import os

# Queue to store batches of points NOT lines here
point_queue = asyncio.Queue()


def enqueue_points(angles_mm, distances_mm):
    """Add a new batch of points to the queue."""
    point_queue.put_nowait((angles_mm, distances_mm))


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
            angles_mm, distances_mm = latest
            lidar_batches = [latest]  # store only most recent revolution
            print("Saved batch:", len(angles_mm), "points")

            # Convert XY → polar occupancy image
            data = polar_lidar_to_image(angles_mm, distances_mm)

            # Convert to PIL image
            img = Image.fromarray(data)

            # Save to disk
            img.save(f"photo-output-results/output_image_pil{dataCount}.png")
            dataCount += 1

        else:
            print("No new data")

        await asyncio.sleep(2)


# Fully AI created, goofy hallucination of a function and is making me kinda annoyed.
def polar_lidar_to_image(angles_deg, distances_mm,
                         img_width=360, img_height=256, max_range=200):

    img = np.zeros((img_height, img_width), dtype=np.uint8)

    # Flatten inputs
    angles_deg = np.array(angles_deg).flatten()
    distances_mm = np.array(distances_mm).flatten()

    # Filter valid distances
    valid = (distances_mm > 0) & (distances_mm <= max_range)
    angles_deg = angles_deg[valid]
    distances_mm = distances_mm[valid]

    # Convert to radians
    angles_rad = np.deg2rad(angles_deg)

    # Polar -> Cartesian (front = +Y, left = -X)
    x = distances_mm * np.sin(angles_rad)  # left/right
    y = distances_mm * np.cos(angles_rad)  # front/back

    # Map XY to image coordinates
    x_min, x_max = -max_range, max_range
    y_min, y_max = 0, max_range  # front at top

    cols = ((x - x_min) / (x_max - x_min) * (img_width - 1)).astype(int)
    rows = ((y_max - y) / (y_max - y_min) *
            (img_height - 1)).astype(int)  # flip Y

    # Clamp indices
    cols = np.clip(cols, 0, img_width - 1)
    rows = np.clip(rows, 0, img_height - 1)

    # Set points
    img[rows, cols] = 255

    return img


def get_start_data_collection():
    """Return the coroutine for use in asyncio.gather."""
    return run_save_data()
