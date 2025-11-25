import math
from rplidarc1 import RPLidar
import asyncio
from time import sleep
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
import save_data

# Initialize LIDAR
lidar = RPLidar("COM4", 460800)
sleep(1)

max_scan_distance_mm = 5000  # 5 meters
pygame_visualization = False

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
            if pygame_visualization:
                import plot
                tg.create_task(plot.get_start_plot())
            tg.create_task(save_data.get_start_data_collection(
                max_scan_distance_mm))
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
                if distance_mm < max_scan_distance_mm:  # max range
                    angle_rad = math.radians(angle_deg)
                    x = distance_mm * math.cos(angle_rad - math.pi / 2)
                    y = distance_mm * math.sin(angle_rad - math.pi / 2)

                    rev_temp_xs.append(x)  # converted to cartesian
                    rev_temp_ys.append(y)

                    rev_temp_angles.append(angle_deg)
                    rev_temp_distances.append(distance_mm)

                if angle_deg >= 358.5:  # every time a revolution completes
                    # if len(rev_temp_xs) > 5:
                    #     try:
                    #         lines = await asyncio.to_thread(
                    #             find_all_ransac_lines,  # multi-line sync function
                    #             rev_temp_xs,
                    #             rev_temp_ys,
                    #         )

                    #         # Plot each detected line
                    #         for line_x, line_y, inliers in lines:
                    #             plot.enqueue_points(
                    #                 line_x.flatten(), line_y, is_line=True
                    #             )

                    #     except Exception as e:
                    #         print("RANSAC failed:", e)

                    # Save polar data for AI model training

                    if len(rev_temp_xs) > 5:
                        save_data.enqueue_points(
                            list(rev_temp_xs), list(rev_temp_ys))
                        if pygame_visualization:
                            plot.enqueue_points(rev_temp_xs, rev_temp_ys)

                        # reset for next revolution
                    rev_temp_xs = []
                    rev_temp_ys = []

                    rev_temp_angles = []
                    rev_temp_distances = []

        await asyncio.sleep(0.01)


def find_all_ransac_lines(xs, ys, min_inliers=30, max_lines=2):
    # Convert to arrays
    X_all = np.array(xs).reshape(-1, 1)
    Y_all = np.array(ys)

    lines = []  # List of (line_x, line_y, inlier_mask)

    for _ in range(max_lines):
        if len(X_all) < 2:
            break

        # Fit one RANSAC line
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=0.35,
            min_samples=2,
            max_trials=50,
        )
        ransac.fit(X_all, Y_all)

        inliers = ransac.inlier_mask_
        num_inliers = inliers.sum()

        # Confidence threshold: reject weak lines
        if num_inliers < min_inliers:
            break

        # Build smooth line prediction
        line_x = np.linspace(X_all.min(), X_all.max(), 200).reshape(-1, 1)
        line_y = ransac.predict(line_x)

        lines.append((line_x, line_y, inliers))

        # Remove inliers and keep only outliers for next iteration
        X_all = X_all[~inliers]
        Y_all = Y_all[~inliers]
    # if len(lines) > 0:
    #     print(f"Detected {len(lines)} lines")
    return lines


try:
    asyncio.run(process_scan_data())

except KeyboardInterrupt:
    print("Scan interrupted... WHY??? 😂")
    lidar.reset()
finally:
    lidar.reset()
