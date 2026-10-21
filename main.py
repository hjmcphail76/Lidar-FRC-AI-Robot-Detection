import asyncio
import math
import time
from time import sleep

import cv2
import numpy as np
from rplidarc1 import RPLidar
from sklearn.cluster import DBSCAN

import plot
import nt_interface

LIDAR_PORT = "COM3"
LIDAR_BAUDRATE = 460800

MAX_SCAN_DISTANCE_MM = 8000

# Keep the Pygame overlay (clusters, boxes) on the same scale as the image.
plot.MAX_RANGE_MM = MAX_SCAN_DISTANCE_MM

MIN_SCAN_ANGLE_DEG = 0
MAX_SCAN_ANGLE_DEG = 360

CLUSTER_EPS_MM = 100
CLUSTER_MIN_POINTS = 5
MIN_TARGET_POINTS = 10

IMAGE_WIDTH = 360
IMAGE_HEIGHT = 256

QUEUE_POLL_INTERVAL = 0.0005

# Keep only the newest scan waiting for clustering.
# Items are (xs, ys, timestamp_seconds).
scan_queue = asyncio.Queue(maxsize=1)

# ============================================================
# LIDAR TARGET CLUSTERING
# ============================================================

def find_lidar_targets(xs, ys):
    """
    Cluster Cartesian LIDAR points using DBSCAN.

    Returns a list of target dictionaries containing:
        points, point_count, center_x, center_y,
        width_mm, height_mm, bbox_mm
    """
    if len(xs) < CLUSTER_MIN_POINTS:
        return []

    points = np.column_stack((xs, ys)).astype(np.float32)

    clustering = DBSCAN(
        eps=CLUSTER_EPS_MM,
        min_samples=CLUSTER_MIN_POINTS,
    ).fit(points)

    labels = clustering.labels_
    targets = []

    for label in np.unique(labels):
        # DBSCAN labels noise as -1.
        if label == -1:
            continue

        cluster = points[labels == label]

        if len(cluster) < MIN_TARGET_POINTS:
            continue

        x_min, y_min = cluster.min(axis=0)
        x_max, y_max = cluster.max(axis=0)

        targets.append({
            "points": cluster,
            "point_count": len(cluster),

            "center_x": float((x_min + x_max) / 2),
            "center_y": float((y_min + y_max) / 2),

            "width_mm": float(x_max - x_min),
            "height_mm": float(y_max - y_min),
            "bbox_mm": (  # Bounding box info in mm, not really used
                float(x_min),
                float(y_min),
                float(x_max),
                float(y_max),
            ),
        })

    return targets


# ============================================================
# LIDAR IMAGE GENERATION
# ============================================================

def lidar_to_image(
    xs,
    ys,
    img_width=IMAGE_WIDTH,
    img_height=IMAGE_HEIGHT,
    max_range=MAX_SCAN_DISTANCE_MM,
):
    """
    Convert Cartesian LIDAR points into an RGB image.

    World coordinates:
        X: left/right
        Y: forward/back according to the existing coordinate mapping
    """
    img = np.zeros(
        (img_height, img_width),
        dtype=np.uint8,
    )

    if len(xs) == 0:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    x_min, x_max = -max_range, max_range
    y_min, y_max = max_range, -max_range

    cols = (
        (x - x_min) / (x_max - x_min) * (img_width - 1)
    ).astype(int)

    rows = (
        (y_max - y) / (y_max - y_min) * (img_height - 1)
    ).astype(int)

    # Ignore out-of-range points instead of clamping them
    # to the image border.
    valid = (
        (x >= -max_range)
        & (x <= max_range)
        & (y >= -max_range)
        & (y <= max_range)
    )

    cols = cols[valid]
    rows = rows[valid]

    img[rows, cols] = 255

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# ============================================================
# BACKGROUND CLUSTERING WORKER
# ============================================================

async def clustering_algorithm_worker():
    """
    Cluster scans outside the main asyncio event loop.

    If clustering falls behind, old pending scans are discarded
    so processing stays close to real time.
    """
    while True:
        xs, ys, scan_timestamp = await scan_queue.get()

        try:
            start_time = time.perf_counter()

            targets = await asyncio.to_thread(
                find_lidar_targets,
                xs,
                ys,
            )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            # Publish the newest available target set to Pygame.
            plot.enqueue_targets(targets)

            # Publish to NetworkTables. An empty target list still publishes
            # (timestamp only), which clears the previous poses.
            nt_interface.publish_poses(targets, scan_timestamp)

            # print(
            #     f"Clustering: {elapsed_ms:.1f} ms | "
            #     f"Points: {len(xs)} | "
            #     f"Targets: {len(targets)}"
            # )

        except Exception as exc:
            print(f"Target clustering error: {exc}")

        finally:
            scan_queue.task_done()


def enqueue_latest_scan(xs, ys, timestamp):
    """
    Queue a snapshot of a complete revolution.

    Discard the waiting scan if the worker has not caught up.
    The scan currently being processed is not affected.
    """
    if scan_queue.full():
        try:
            scan_queue.get_nowait()
            scan_queue.task_done()
        except asyncio.QueueEmpty:
            pass

    try:
        scan_queue.put_nowait((xs.copy(), ys.copy(), timestamp))
    except asyncio.QueueFull:
        # A newer scan was queued concurrently.
        pass


# ============================================================
# LIDAR QUEUE PROCESSING
# ============================================================

async def process_queue(queue, stop_event):
    rev_temp_xs = []
    rev_temp_ys = []

    prev_angle_deg = None

    while not stop_event.is_set():
        processed_data = False

        while not queue.empty():
            try:
                data = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            processed_data = True

            angle_deg = data["a_deg"]
            distance_mm = data["d_mm"]

            # A revolution ends when the angle wraps around (e.g. 359 -> 1).
            # Detecting the wrap fires exactly once per revolution, and the
            # new sample is handled as the first point of the next one.
            if prev_angle_deg is not None and angle_deg < prev_angle_deg - 180:
                # Stamp the scan when it completes, not after clustering.
                scan_timestamp = nt_interface.get_nt_timestamp_seconds()

                # Snapshot this revolution before resetting buffers.
                xs = rev_temp_xs.copy()
                ys = rev_temp_ys.copy()

                # Render the image immediately.
                img = lidar_to_image(
                    xs,
                    ys,
                    img_width=IMAGE_WIDTH,
                    img_height=IMAGE_HEIGHT,
                    max_range=MAX_SCAN_DISTANCE_MM,
                )

                plot.enqueue_image(img)

                # Cluster asynchronously. Sparse or empty scans are queued
                # too, so previously published targets get cleared.
                enqueue_latest_scan(xs, ys, scan_timestamp)

                # Reset buffers for the next revolution.
                rev_temp_xs = []
                rev_temp_ys = []

            prev_angle_deg = angle_deg

            if MIN_SCAN_ANGLE_DEG <= angle_deg < MAX_SCAN_ANGLE_DEG:

                # Retain only valid returns within the maximum range.
                if distance_mm is not None and 0 < distance_mm < MAX_SCAN_DISTANCE_MM:
                    angle_rad = math.radians(angle_deg)

                    x = distance_mm * math.cos(
                        angle_rad - math.pi / 2
                    )

                    y = distance_mm * math.sin(
                        angle_rad - math.pi / 2
                    )

                    rev_temp_xs.append(x)
                    rev_temp_ys.append(y)

        if not processed_data:
            await asyncio.sleep(QUEUE_POLL_INTERVAL)


# ============================================================
# ASYNC TASK MANAGEMENT
# ============================================================

async def process_scan_data(lidar):
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                process_queue(
                    lidar.output_queue,
                    lidar.stop_event,
                )
            )

            tg.create_task(
                lidar.simple_scan(make_return_dict=True)
            )

            tg.create_task(
                plot.get_start_plot()
            )

            tg.create_task(
                clustering_algorithm_worker()
            )

            tg.create_task(
                nt_interface.start_nt_publisher()
            )

    finally:
        lidar.reset()


def main():
    lidar = None

    try:
        print(f"Connecting to LIDAR on {LIDAR_PORT}...")

        lidar = RPLidar(
            LIDAR_PORT,
            LIDAR_BAUDRATE,
        )

        sleep(0.25)

        print("LIDAR connected. Starting scan processing.")

        asyncio.run(process_scan_data(lidar))

    except KeyboardInterrupt:
        print("\nStopping LIDAR processing...")

    finally:
        if lidar is not None:
            try:
                lidar.reset()
            except Exception:
                pass

        print("LIDAR stopped.")


if __name__ == "__main__":
    main()