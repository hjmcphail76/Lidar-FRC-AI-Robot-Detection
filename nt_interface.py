import asyncio
import threading
import time
from ntcore import NetworkTableInstance
from wpimath.geometry import Pose2d, Rotation2d

# Shared pose list. Starts empty; it is fully rebuilt on every update cycle,
# so it can hold any number of poses and never contains stale entries.
shared_poses: list[Pose2d] = []
pose_lock = threading.Lock()


def nt_publisher_thread():
    """Background NT publisher thread."""
    inst = NetworkTableInstance.getDefault()
    inst.startClient4("LidarProcessorClient")
    inst.setServer("127.0.0.1")  # or robot IP

    table = inst.getTable("LidarProcessor")
    pose_pub = table.getStructArrayTopic("poses", Pose2d).publish()

    while True:
        with pose_lock:
            pose_list_copy = list(shared_poses)

        pose_pub.set(pose_list_copy)
        inst.flush()
        time.sleep(0.02)  # 50 Hz


def publish_poses(targets):
    """
    Thread-safe update of the pose list.

    Call once per update cycle with the current targets. The previous list is
    discarded entirely (purging stale poses), and one pose is created per
    target with no upper limit. An empty `targets` clears all published poses.
    """
    new_poses = [
        Pose2d(
            -target["center_y"] / 1000.0,   # mm -> m
            -target["center_x"] / 1000.0,  # mm -> m
            Rotation2d.fromDegrees(0.0),   # placeholder angle
        )
        for target in targets
    ]

    with pose_lock:
        shared_poses[:] = new_poses  # swap contents in place under the lock


async def start_nt_publisher():
    """
    Starts the NT publisher thread in the background.
    Can be directly used in asyncio.TaskGroup.
    """
    def starter():
        threading.Thread(target=nt_publisher_thread, daemon=True).start()

    await asyncio.to_thread(starter)