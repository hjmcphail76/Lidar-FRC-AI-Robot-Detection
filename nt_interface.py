import asyncio
import math
import threading
import ntcore

_inst = None
_pose_pub = None
_lock = threading.Lock()


def _init():
    """Connect to NT and create the publisher once. Safe to call repeatedly."""
    global _inst, _pose_pub
    with _lock:
        if _pose_pub is not None:
            return
        _inst = ntcore.NetworkTableInstance.getDefault()
        _inst.startClient4("LidarProcessorClient")
        _inst.setServer("127.0.0.1")  # or robot IP

        table = _inst.getTable("LidarProcessor")
        _pose_pub = table.getDoubleArrayTopic("poses").publish(
            ntcore.PubSubOptions(keepDuplicates=True))


MERGE_DISTANCE_MM = 1000.0  # targets whose centers are closer than this are merged


def merge_close_targets(targets, merge_distance_mm=MERGE_DISTANCE_MM):
    """
    Combine targets whose centers are less than `merge_distance_mm` apart.

    Returns a list of (center_x_mm, center_y_mm) tuples. The closest pair under
    the threshold is replaced by the XY midpoint of the two, and this repeats
    until no remaining pair is that close (so chains of overlapping targets
    collapse into one).
    """
    centers = [(t["center_x"], t["center_y"]) for t in targets]

    while len(centers) > 1:
        best = None
        best_dist = merge_distance_mm
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = math.hypot(
                    centers[i][0] - centers[j][0],
                    centers[i][1] - centers[j][1],
                )
                if dist < best_dist:
                    best_dist = dist
                    best = (i, j)

        if best is None:
            break

        i, j = best
        midpoint = (
            (centers[i][0] + centers[j][0]) / 2.0,
            (centers[i][1] + centers[j][1]) / 2.0,
        )
        centers = [c for k, c in enumerate(centers) if k not in (i, j)]
        centers.append(midpoint)

    return centers


def publish_poses(targets, timestamp):
    """
    Publish the current targets as a flat array. Call once per scan.

    Targets whose centers are less than 1 m apart are merged into a single
    target at their XY midpoint before publishing (see merge_close_targets).

    Structure of the published array:
    [timestamp, x1, y1, rotation1, x2, y2, rotation2, ...]
    where:
        - timestamp: float, seconds (NT server time) at which the scan completed
        - x: float, meters (forward)
        - y: float, meters (left)
        - rotation: float, radians (placeholder for now)

    An empty `targets` list publishes [timestamp] on its own, which tells the
    consumer that a fresh scan found no targets (clearing the previous list).
    """
    _init()

    poses = [timestamp]
    for center_x, center_y in merge_close_targets(targets):
        poses += [
            -center_y / 1000.0,   # x (forward), mm -> m
            -center_x / 1000.0,   # y (left), mm -> m
            0.0,                  # placeholder rotation (radians)
        ]

    _pose_pub.set(poses)
    _inst.flush()


def get_nt_timestamp_seconds():
    _init()
    offset = _inst.getServerTimeOffset()  # Optional, microseconds; None if not connected
    now_server_us = ntcore._now() + (offset or 0)
    return now_server_us / 1e6


async def start_nt_publisher():
    """Connects in a worker thread. Can be used directly in an asyncio.TaskGroup."""
    await asyncio.to_thread(_init)