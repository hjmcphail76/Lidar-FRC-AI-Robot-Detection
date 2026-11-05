import asyncio
import math
import queue
import threading
import time

import cv2
import numpy as np
from rplidarc1 import RPLidar
import serial
from sklearn.cluster import DBSCAN

import nt_interface
import ui


class LidarProcessor(threading.Thread):
    """
    Owns the LIDAR connection, scan processing, clustering and NT publishing
    on its own thread (with its own asyncio event loop).

    Usage:
        lidar_proc = LidarProcessor()
        lidar_proc.start()
        ...
        lidar_proc.is_connected()
        lidar_proc.get_status()
        lidar_proc.stop()
    """

    # Connection states
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"


    def __init__(
        self,
        port="/dev/ttyUSB0",
        baudrate=460800,
        max_scan_distance_mm=8000,
        min_scan_angle_deg=0,
        max_scan_angle_deg=360,
        cluster_eps_mm=100,
        cluster_min_points=5,
        min_target_points=10,
        image_width=360,
        image_height=256,
        queue_poll_interval=0.0005,
        reconnect_delay=2.0,
        data_timeout=3.0,
    ):
        super().__init__(name="LidarProcessor", daemon=True)

        self.port = port
        self.baudrate = baudrate
        self.max_scan_distance_mm = max_scan_distance_mm
        self.min_scan_angle_deg = min_scan_angle_deg
        self.max_scan_angle_deg = max_scan_angle_deg
        self.cluster_eps_mm = cluster_eps_mm
        self.cluster_min_points = cluster_min_points
        self.min_target_points = min_target_points
        self.image_width = image_width
        self.image_height = image_height
        self.queue_poll_interval = queue_poll_interval
        self.reconnect_delay = reconnect_delay
        self.data_timeout = data_timeout

        ui.MAX_RANGE_MM = self.max_scan_distance_mm

        # --- Thread-shared state (guarded by _state_lock) ---
        self._state_lock = threading.Lock()
        self._state = self.DISCONNECTED
        self._last_error = None
        self._last_sample_time = None   # time.monotonic()
        self._last_scan_time = None     # time.monotonic()
        self._scan_count = 0
        self._last_target_count = 0

        # --- Control ---
        self._stop_requested = threading.Event()
        self._reconnect_requested = threading.Event()
        self._loop = None               # set in run()
        self._session_task = None       # current asyncio task, set in run()


        self._scan_queue = None   # created per session in _session()
        self._lidar = None

    # ============================================================
    # PUBLIC API (safe to call from any thread, e.g. the UI)
    # ============================================================

    def is_connected(self):
        """True if connected and samples have arrived recently."""
        with self._state_lock:
            if self._state != self.CONNECTED:
                return False
            if self._last_sample_time is None:
                return False
            return (time.monotonic() - self._last_sample_time) < self.data_timeout

    def get_state(self):
        with self._state_lock:
            return self._state

    def get_last_error(self):
        with self._state_lock:
            return self._last_error
    
    def set_com_port(self, port):
        """Change the serial port and reconnect."""
        self.port = port
        self.reconnect()
    
    def get_com_port(self):
        """Get the current serial port."""
        return self.port

    def get_status(self):
        """Snapshot of everything the UI might want to display."""
        now = time.monotonic()
        with self._state_lock:
            return {
                "state": self._state,
                "connected": (
                    self._state == self.CONNECTED
                    and self._last_sample_time is not None
                    and now - self._last_sample_time < self.data_timeout
                ),
                "port": self.port,
                "baudrate": self.baudrate,
                "last_error": self._last_error,
                "scan_count": self._scan_count,
                "last_target_count": self._last_target_count,
                "seconds_since_scan": (
                    None if self._last_scan_time is None
                    else now - self._last_scan_time
                ),
            }

    def reconnect(self):
        """Drop the current session and reconnect (e.g. a UI button)."""
        self._close_lidar(self._lidar)
        self._reconnect_requested.set()
        self._cancel_session()

    
    def _close_lidar(self, lidar):
        """Stop scanning, reset the device, and release the serial port."""
        try:
            lidar.stop_event.set()
        except Exception:
            pass
        try:
            lidar.reset()
        except Exception:
            pass

        def find_serials(obj, depth=0):
            if isinstance(obj, serial.Serial):
                yield obj
            elif depth < 2 and hasattr(obj, "__dict__"):
                for v in vars(obj).values():
                    yield from find_serials(v, depth + 1)

        found = False
        for ser in find_serials(lidar):
            found = True
            try:
                if ser.is_open:
                    ser.close()
            except Exception as exc:
                print(f"Failed to close serial port: {exc}")

        if not found:
            print(f"No serial.Serial found. Attributes: {list(vars(lidar))}")
            
            
    def stop(self, timeout=5.0):
        """Stop the thread and release the LIDAR."""
        self._stop_requested.set()
        self._cancel_session()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)

    # ============================================================
    # THREAD ENTRY POINT
    # ============================================================

    def run(self):
        try:
            asyncio.run(self._main())
        finally:
            self._set_state(self.DISCONNECTED)

    async def _main(self):
        """Connect / run / retry loop."""
        self._loop = asyncio.get_running_loop()

        while not self._stop_requested.is_set():
            self._reconnect_requested.clear()
            self._session_task = asyncio.create_task(self._session())

            try:
                await self._session_task
            except asyncio.CancelledError:
                # Cancelled by stop() / reconnect(); the loop condition decides.
                pass
            except Exception as exc:
                self._set_state(self.DISCONNECTED, error=str(exc))
                print(f"LIDAR session error: {exc}")
            finally:
                self._session_task = None

            if self._stop_requested.is_set():
                break

            if self._reconnect_requested.is_set():
                continue

            self._set_state(self.DISCONNECTED)
            await asyncio.sleep(self.reconnect_delay)

    async def _session(self):
        """One connection attempt plus scanning until it fails or is cancelled."""
        lidar = None
        self._scan_queue = asyncio.Queue(maxsize=1)

        try:
            self._set_state(self.DISCONNECTED)
            print(f"Connecting to LIDAR on {self.port}...")

            lidar = RPLidar(self.port, self.baudrate)
            self._lidar = lidar

            lidar.healthcheck()  # raises if not connected / not healthy

            print("LIDAR connected. Starting scan processing.")
            with self._state_lock:
                self._last_sample_time = time.monotonic()
            self._set_state(self.CONNECTED)

            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._process_queue(lidar.output_queue, lidar.stop_event))
                tg.create_task(lidar.simple_scan(make_return_dict=True))
                tg.create_task(self._clustering_worker())
                tg.create_task(nt_interface.start_nt_publisher())
                tg.create_task(self._watchdog())
        finally:
            self._lidar = None
            if lidar is not None:
                try:
                    lidar.reset()
                    time.sleep(.5)
                except Exception:
                    pass

    async def _watchdog(self):
        """Raise if samples stop arriving so the outer loop reconnects."""
        while True:
            await asyncio.sleep(0.5)
            with self._state_lock:
                last = self._last_sample_time
            if last is not None and time.monotonic() - last > self.data_timeout:
                raise TimeoutError("No LIDAR data received")

    # ============================================================
    # INTERNAL STATE HELPERS
    # ============================================================

    def _set_state(self, state, error=None):
        with self._state_lock:
            self._state = state
            if error is not None:
                self._last_error = error
            elif state == self.CONNECTED:
                self._last_error = None

    def _cancel_session(self):
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _cancel():
            if self._session_task is not None:
                self._session_task.cancel()

        try:
            loop.call_soon_threadsafe(_cancel)
        except RuntimeError:
            pass  # loop already closed

    # ============================================================
    # CLUSTERING
    # ============================================================

    def find_lidar_targets(self, xs, ys):
        """Cluster Cartesian LIDAR points using DBSCAN."""
        if len(xs) < self.cluster_min_points:
            return []

        points = np.column_stack((xs, ys)).astype(np.float32)

        labels = DBSCAN(
            eps=self.cluster_eps_mm,
            min_samples=self.cluster_min_points,
        ).fit(points).labels_

        targets = []

        for label in np.unique(labels):
            if label == -1:  # DBSCAN noise
                continue

            cluster = points[labels == label]

            if len(cluster) < self.min_target_points:
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
                "bbox_mm": (
                    float(x_min),
                    float(y_min),
                    float(x_max),
                    float(y_max),
                ),
            })

        return targets

    async def _clustering_worker(self):
        """Cluster scans off the event loop; stale scans are dropped."""
        while True:
            xs, ys, scan_timestamp = await self._scan_queue.get()

            try:
                targets = await asyncio.to_thread(self.find_lidar_targets, xs, ys)

                ui.enqueue_targets(targets)

                # An empty target list still publishes, clearing old poses.
                nt_interface.publish_poses(targets, scan_timestamp)

                with self._state_lock:
                    self._last_target_count = len(targets)

            except Exception as exc:
                print(f"Target clustering error: {exc}")

            finally:
                self._scan_queue.task_done()

    def _enqueue_latest_scan(self, xs, ys, timestamp):
        """Queue a revolution, discarding the waiting one if the worker is behind."""
        if self._scan_queue.full():
            try:
                self._scan_queue.get_nowait()
                self._scan_queue.task_done()
            except asyncio.QueueEmpty:
                pass

        try:
            self._scan_queue.put_nowait((xs.copy(), ys.copy(), timestamp))
        except asyncio.QueueFull:
            pass

    # ============================================================
    # IMAGE GENERATION
    # ============================================================

    def lidar_to_image(self, xs, ys):
        """Convert Cartesian LIDAR points into an RGB image."""
        max_range = self.max_scan_distance_mm
        img_width = self.image_width
        img_height = self.image_height

        img = np.zeros((img_height, img_width), dtype=np.uint8)

        if len(xs) == 0:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        x = np.asarray(xs, dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32)

        x_min, x_max = -max_range, max_range
        y_min, y_max = max_range, -max_range

        cols = ((x - x_min) / (x_max - x_min) * (img_width - 1)).astype(int)
        rows = ((y_max - y) / (y_max - y_min) * (img_height - 1)).astype(int)

        valid = (
            (x >= -max_range) & (x <= max_range)
            & (y >= -max_range) & (y <= max_range)
        )

        img[rows[valid], cols[valid]] = 255

        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # ============================================================
    # LIDAR SAMPLE PROCESSING
    # ============================================================

    async def _process_queue(self, queue, stop_event):
        rev_xs = []
        rev_ys = []
        prev_angle_deg = None

        while not stop_event.is_set():
            processed_data = False

            while not queue.empty():
                try:
                    data = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                processed_data = True

                with self._state_lock:
                    self._last_sample_time = time.monotonic()

                angle_deg = data["a_deg"]
                distance_mm = data["d_mm"]

                # A revolution ends when the angle wraps (e.g. 359 -> 1).
                if prev_angle_deg is not None and angle_deg < prev_angle_deg - 180:
                    scan_timestamp = nt_interface.get_nt_timestamp_seconds()

                    xs = rev_xs
                    ys = rev_ys
                    rev_xs = []
                    rev_ys = []

                    xs_arr = np.asarray(xs, dtype=np.float32)
                    ys_arr = np.asarray(ys, dtype=np.float32)

                    ui.enqueue_image(self.lidar_to_image(xs_arr, ys_arr))
                    self._enqueue_latest_scan(xs_arr, ys_arr, scan_timestamp)

                    with self._state_lock:
                        self._scan_count += 1
                        self._last_scan_time = time.monotonic()

                prev_angle_deg = angle_deg

                if self.min_scan_angle_deg <= angle_deg < self.max_scan_angle_deg:
                    if distance_mm is not None and 0 < distance_mm < self.max_scan_distance_mm:
                        angle_rad = math.radians(angle_deg)
                        rev_xs.append(distance_mm * math.cos(angle_rad - math.pi / 2))
                        rev_ys.append(distance_mm * math.sin(angle_rad - math.pi / 2))

            if not processed_data:
                await asyncio.sleep(self.queue_poll_interval)