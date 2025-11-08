import math
from rplidarc1 import RPLidar
import asyncio
from time import sleep

import plot


# Initialize LIDAR
lidar = RPLidar("COM4", 460800)
sleep(1)  # Allow time for the device to initialize

# Lists to store LIDAR points
rev_temp_xs = []
rev_temp_ys = []


async def process_scan_data():
    try:
        async with asyncio.TaskGroup() as tg:
            # No wait task needed; just scan and process indefinitely
            tg.create_task(process_queue(lidar.output_queue, lidar.stop_event))
            tg.create_task(lidar.simple_scan(make_return_dict=True))
            tg.create_task(plot.get_start_plot())
    finally:
        lidar.reset()


async def process_queue(queue, stop_event):
    global rev_temp_xs, rev_temp_ys
    while not stop_event.is_set():
        while not queue.empty():
            data = await queue.get()
            angle_deg = data["a_deg"]
            distance_mm = data["d_mm"]

            if distance_mm is not None:
                if distance_mm < 250:
                    angle_rad = math.radians(angle_deg)
                    x = distance_mm * math.cos(angle_rad)
                    y = distance_mm * math.sin(angle_rad)
                    rev_temp_xs.append(x)
                    rev_temp_ys.append(y)

                    # print(rev_temp_xs[-1], rev_temp_ys[-1]) debugging

                    plot.enqueue_points(rev_temp_xs, rev_temp_ys)
                if angle_deg >= 358.5:
                    rev_temp_xs = []
                    rev_temp_ys = []
                    # print("New Revolution")
        await asyncio.sleep(0.1)


try:
    asyncio.run(process_scan_data())
except KeyboardInterrupt:
    print("Scan interrupted... WHY??? 😂")
    lidar.reset()
finally:
    lidar.reset()
