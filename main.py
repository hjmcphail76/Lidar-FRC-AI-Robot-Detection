import os
import shutil
import cv2
import math
from rplidarc1 import RPLidar
import asyncio
from time import sleep
import numpy as np
import infrence
import save_data
from PIL import Image
import plot

max_scan_distance_mm = 5000  # 5 meters

debug_mode = False


# Lists to store LIDAR points
rev_temp_xs = []
rev_temp_ys = []

rev_temp_angles = []
rev_temp_distances = []


async def process_scan_data():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(process_queue(lidar.output_queue, lidar.stop_event))
            tg.create_task(lidar.simple_scan(make_return_dict=True))

            tg.create_task(plot.get_start_plot())
            tg.create_task(save_data.get_start_data_collection(
                debugging=debug_mode))
            tg.create_task(infrence.get_start_data_collection())
    finally:
        lidar.reset()


async def process_queue(queue, stop_event):
    global rev_temp_xs, rev_temp_ys, rev_temp_angles, rev_temp_distances
    while not stop_event.is_set():
        while not queue.empty():
            data = await queue.get()
            angle_deg = data["a_deg"]
            distance_mm = data["d_mm"]

            if distance_mm is not None:
                if distance_mm < max_scan_distance_mm:
                    angle_rad = math.radians(angle_deg)
                    x = distance_mm * math.cos(angle_rad - math.pi / 2)
                    y = distance_mm * math.sin(angle_rad - math.pi / 2)

                    rev_temp_xs.append(x)  # converted to cartesian
                    rev_temp_ys.append(y)

                    rev_temp_angles.append(angle_deg)
                    rev_temp_distances.append(distance_mm)

                if angle_deg >= 358.5:

                    if len(rev_temp_xs) > 5:

                        img = lidar_to_image(
                            rev_temp_xs,
                            rev_temp_ys,
                            img_width=360,
                            img_height=256,
                            max_range=max_scan_distance_mm,
                        )

                        save_data.enqueue_images(Image.fromarray(img))

                        infrence.enqueue_image(img)

                        # results = infrence.make_inference(
                        #     img)  # returns Result object

                        plot.enqueue_image(img)

                    # reset for next revolution
                    rev_temp_xs = []
                    rev_temp_ys = []

                    rev_temp_angles = []
                    rev_temp_distances = []

        await asyncio.sleep(0.01)


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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img_rgb


try:
    # Initialize LIDAR
    lidar = RPLidar("COM4", 460800)
    sleep(1)

    folder = "photo-output-results"

    # Delete everything *inside* the folder
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

    asyncio.run(process_scan_data())
except KeyboardInterrupt:
    print("Scan interrupted... WHY??? 😂")
    lidar.reset()
finally:
    lidar.reset()
