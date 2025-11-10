import math
from rplidarc1 import RPLidar
import asyncio
from time import sleep
import plot
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression


def fit_ransac_line(xs, ys):
    X = np.array(xs).reshape(-1, 1)
    Y = np.array(ys)
    if len(X) < 2:
        raise ValueError("Need at least 2 points for RANSAC line fitting")

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=0.2,
        min_samples=2,
        max_trials=1000,
    )
    ransac.fit(X, Y)
    line_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    line_y = ransac.predict(line_x)
    return line_x, line_y


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
                    x = distance_mm * math.cos(angle_rad - math.pi / 2)
                    y = distance_mm * math.sin(angle_rad - math.pi / 2)
                    rev_temp_xs.append(x)
                    rev_temp_ys.append(y)

                    # print(rev_temp_xs[-1], rev_temp_ys[-1]) debugging

                    plot.enqueue_points(rev_temp_xs, rev_temp_ys)
                if angle_deg >= 358.5:
                    # if len(rev_temp_xs) > 5:
                    #     try:
                    #         line_x, line_y = fit_ransac_line(rev_temp_xs, rev_temp_ys)
                    #         plot.enqueue_points(line_x.flatten(), line_y, is_line=True)
                    #     except Exception as e:
                    #         print("RANSAC failed:", e)

                    rev_temp_xs = []
                    rev_temp_ys = []

        await asyncio.sleep(0.01)


try:
    asyncio.run(process_scan_data())
except KeyboardInterrupt:
    print("Scan interrupted... WHY??? 😂")
    lidar.reset()
finally:
    lidar.reset()
