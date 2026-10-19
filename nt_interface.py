import asyncio
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


def publish_poses(targets):
    """
    Publish the current targets as a flat [x, y, rotation_rad, ...] array.
    Call once per scan. An empty `targets` publishes an empty array, which clears the list.
    
    Structure of the published array:
    [timestamp, x1, y1, rotation1, x2, y2, rotation2, ...]
    where:
        - timestamp: float, seconds since epoch
        - x: float, meters (forward)
        - y: float, meters (left)
        - rotation: float, radians (placeholder for now)
    """
    _init()

    poses = [targets[0]["timestamp"]]
    for target in targets:
        poses += [
            -target["center_y"] / 1000.0,   # x (forward), mm -> m
            -target["center_x"] / 1000.0,   # y (left), mm -> m
            0.0,                            # placeholder rotation (radians)
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