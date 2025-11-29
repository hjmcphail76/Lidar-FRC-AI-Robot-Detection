import asyncio
import threading
import time
from ntcore import NetworkTableInstance
from wpimath.geometry import Pose2d, Rotation2d

# Shared pose list (start with a single default pose at origin)
shared_poses = [Pose2d(0, 0, Rotation2d.fromDegrees(0))]
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
            # Always publish the single pose in the list
            pose_list_copy = list(shared_poses)

        pose_pub.set(pose_list_copy)
        inst.flush()
        time.sleep(0.02)  # 50 Hz


def publish_pose(x, y, angle_deg):
    """Thread-safe update of the single pose."""
    with pose_lock:
        shared_poses[0] = Pose2d(x, y, Rotation2d.fromDegrees(angle_deg))


async def start_nt_publisher():
    """
    Starts the NT publisher thread in the background.
    Can be directly used in asyncio.TaskGroup.
    """
    def starter():
        threading.Thread(target=nt_publisher_thread, daemon=True).start()

    # Run thread starter in a background OS thread
    await asyncio.to_thread(starter)
