import base64
import threading

import cv2
import numpy as np
from nicegui import ui

# Screen/canvas dimensions (SVG overlay coordinate space)
WIDTH, HEIGHT = 800, 600

# Colors
GREEN = "#00dc00"
WHITE = "#ffffff"
CYAN = "#00dcff"

# Must match MAX_SCAN_DISTANCE_MM in main.py (main.py overwrites this at import).
MAX_RANGE_MM = 8000

SHOW_LABELS = True
REFRESH_HZ = 30

_lock = threading.Lock()
_state = {"image": None, "targets": [], "box": None}
_version = {"image": 0, "targets": 0, "box": 0}


def enqueue_image(image_np):
    """
    Publish the newest numpy image.
    Expected shape: (H, W, 3), dtype uint8, RGB.
    """
    with _lock:
        _state["image"] = image_np
        _version["image"] += 1


def enqueue_box(x1, y1, x2, y2):
    """Legacy YOLO box in screen pixel coordinates."""
    with _lock:
        _state["box"] = (x1, y1, x2, y2)
        _version["box"] += 1


def enqueue_targets(targets):
    """
    Publish targets returned by find_lidar_targets().
    An empty list is a real update and clears the overlay.
    """
    with _lock:
        _state["targets"] = targets
        _version["targets"] += 1


def _snapshot(seen):
    """Return {key: value} for every slot newer than the versions in `seen`."""
    changed = {}
    with _lock:
        for key, ver in _version.items():
            if ver != seen[key]:
                seen[key] = ver
                changed[key] = _state[key]
    return changed


# ============================================================
# RENDERING HELPERS
# ============================================================

def world_to_screen(x, y):
    """
    Convert LIDAR Cartesian coordinates (mm) to canvas pixels.
    Matches the coordinate mapping used by lidar_to_image().
    """
    px = int((x + MAX_RANGE_MM) / (2 * MAX_RANGE_MM) * WIDTH)
    py = int((y + MAX_RANGE_MM) / (2 * MAX_RANGE_MM) * HEIGHT)
    return (
        max(0, min(WIDTH - 1, px)),
        max(0, min(HEIGHT - 1, py)),
    )


def image_to_data_url(img_np):
    """Encode an RGB uint8 numpy image as a PNG data URL."""
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def targets_to_svg(targets):
    """Clustered points, bounding boxes and labels as SVG markup."""
    parts = []
    for i, target in enumerate(targets):
        x_min, y_min, x_max, y_max = target["bbox_mm"]

        # Cluster points
        for x, y in target["points"]:
            px, py = world_to_screen(x, y)
            parts.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{CYAN}"/>')

        # Bounding box
        p1 = world_to_screen(x_min, y_max)
        p2 = world_to_screen(x_max, y_min)
        left, right = min(p1[0], p2[0]), max(p1[0], p2[0])
        top, bottom = min(p1[1], p2[1]), max(p1[1], p2[1])
        parts.append(
            f'<rect x="{left}" y="{top}" width="{max(1, right - left)}" '
            f'height="{max(1, bottom - top)}" fill="none" '
            f'stroke="{GREEN}" stroke-width="2"/>'
        )

        if SHOW_LABELS:
            label = (
                f"T{i + 1} | {target['point_count']} pts | "
                f"{target['width_mm']:.0f}x{target['height_mm']:.0f} mm"
            )
            parts.append(
                f'<text x="{left}" y="{max(12, top - 6)}" fill="{GREEN}" '
                f'font-size="14" font-family="monospace">{label}</text>'
            )
    return "".join(parts)


def box_to_svg(box):
    """Legacy YOLO box (screen pixel coordinates) as SVG markup."""
    if box is None:
        return ""
    x1, y1, x2, y2 = box
    return (
        f'<rect x="{int(x1)}" y="{int(y1)}" width="{int(x2 - x1)}" '
        f'height="{int(y2 - y1)}" fill="none" stroke="{WHITE}" stroke-width="2"/>'
    )


# ============================================================
# GUI
# ============================================================


def set_control_callbacks(on_start=None, on_stop=None):
    global _on_start, _on_stop
    _on_start, _on_stop = on_start, on_stop


def main():
    ui.label("Lidar Coprocessor Control Panel").classes("text-h4 m-4 font-bold")
    status_label = ui.label("System Status: Disconnected").classes("text-lg m-4 text-amber-500")

    system_status = {"nt_connected": False, "visualization_active": False}


    def toggle_visualization():
        system_status["nt_connected"] = not system_status["nt_connected"]
        if system_status["value"]:
            button.text = "Stop Visualiization"
            status_label.classes(replace="text-lg m-4 text-green-500")
            if _on_start:
                _on_start()
        else:
            button.text = "Start Visualization"
            status_label.classes(replace="text-lg m-4 text-amber-500")
            if _on_stop:
                _on_stop()

    button = ui.button("Start Lidar Detection", on_click=toggle_visualization)
    button.classes("m-4 bg-primary text-white")

    # Lidar view: raster image underneath, SVG overlay (targets + boxes) on top.
    view = ui.interactive_image(size=(WIDTH, HEIGHT)).classes("m-4 bg-black")
    view.props("fit=fill")
    view.style(f"width: {WIDTH}px; height: {HEIGHT}px; image-rendering: pixelated;")

    # Per-client drawing state
    seen = {"image": 0, "targets": 0, "box": 0}
    drawn = {"targets": "", "box": ""}

    def refresh():
        #EA
        changed = _snapshot(seen)
        if not changed:
            return

        if "image" in changed and changed["image"] is not None:
            view.set_source(image_to_data_url(changed["image"]))

        if "targets" in changed:
            drawn["targets"] = targets_to_svg(changed["targets"])
        if "box" in changed:
            drawn["box"] = box_to_svg(changed["box"])

        if "targets" in changed or "box" in changed:
            view.content = drawn["targets"] + drawn["box"]

    ui.timer(1 / REFRESH_HZ, refresh)


def run_gui():
    """Build the page and start the web server (blocks; call from the main thread)."""
    main()
    # reload=False: the auto-reloader would re-import this module and
    # try to open the LIDAR serial port a second time.
    ui.run(title="FRC Lidar GUI", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    run_gui()