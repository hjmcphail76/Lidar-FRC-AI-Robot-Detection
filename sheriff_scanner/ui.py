import base64
import threading

import cv2
import numpy as np
from nicegui import ui

import nt_interface

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

system_status = {
    "sensor_connected": False,
    "nt_connected": False,
    "visualization_active": False,
}


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
# GUI
# ============================================================


def set_nt_ip(ip):
    """Called when the user sets the NetworkTables IP address."""
    pass  # TODO: implement


def main(lidar_processor=None):
    ui.dark_mode().enable()
    ui.colors(primary="#3b82f6", positive="#22c55e", negative="#ef4444", warning="#f59e0b")
    ui.query("body").classes("bg-slate-950")

    CARD = "bg-slate-900 border border-slate-800 rounded-xl shadow-lg p-4 w-full"
    SECTION = "text-xs font-semibold tracking-widest text-slate-400 uppercase"

    # ---------- Header ----------
    with ui.header().classes("bg-slate-900 border-b border-slate-800 items-center px-6 py-3"):
        ui.icon("radar", size="28px").classes("text-primary")
        ui.label("Lidar Coprocessor").classes("text-xl font-semibold text-white")
        ui.label("Control Panel").classes("text-xl font-light text-slate-400")
        ui.space()
        if lidar_processor is not None:
            ui.label(f"{lidar_processor.port} @ {lidar_processor.baudrate}").classes(
                "text-sm font-mono text-slate-400"
            )

    # ---------- Helpers ----------
    def status_row(title):
        """A row with a colored dot, a title, and a right-aligned state label."""
        with ui.row().classes("w-full items-center no-wrap"):
            dot = ui.icon("circle", size="12px").props("color=negative")
            ui.label(title).classes("text-slate-200")
            ui.space()
            state = ui.label("Disconnected").classes("text-sm font-medium text-negative")
        return dot, state

    def set_status(row, connected):
        dot, state = row
        dot.props(f"color={'positive' if connected else 'negative'}")
        state.text = "Connected" if connected else "Disconnected"
        state.classes(
            replace="text-sm font-medium "
            + ("text-positive" if connected else "text-negative")
        )

    
    def submit_configuration(lidar_processor):
        port = com_port_input.value.strip()
        if port and port != lidar_processor.get_com_port():
            lidar_processor.set_port(port)
            ui.notify(f"COM port set to {port}", type="positive", position="top")
        
        ip = nt_ip_input.value.strip()
        if ip:
            set_nt_ip(ip)
            ui.notify(f"NetworkTables address set to {ip}", type="positive", position="top")
            
    def toggle_visualization():
        active = not system_status["visualization_active"]
        system_status["visualization_active"] = active
        if active:
            toggle_btn.text = "Pause Visualization"
            toggle_btn.props("icon=pause color=warning")
        else:
            toggle_btn.text = "Start Visualization"
            toggle_btn.props("icon=play_arrow color=primary")

    # ---------- Body ----------
    with ui.row().classes("w-full no-wrap items-start gap-6 p-6"):

        # Left column: controls
        with ui.column().classes("gap-4").style("width: 320px; min-width: 320px;"):

            with ui.card().classes(CARD):
                ui.label("Connections").classes(SECTION)
                ui.separator().classes("bg-slate-800")
                sensor_row = status_row("Sensor")
                nt_row = status_row("NetworkTables")

            with ui.card().classes(CARD):
                ui.label("NetworkTables").classes(SECTION)
                ui.separator().classes("bg-slate-800")
                nt_ip_input = (
                    ui.input(label="Server IP", value=nt_interface.ip_address, placeholder="e.g. 10.TE.AM.2")
                    .props("outlined dense dark")
                    .classes("w-full")
                    .on("keydown.enter", lambda: submit_configuration(lidar_processor))
                )
                com_port_input = (
                    ui.input(label="Port", value=lidar_processor.port if lidar_processor else "", placeholder="e.g. COM3")
                    .props("outlined dense dark")
                    .classes("w-full")
                    .on("keydown.enter", lambda: submit_configuration(lidar_processor))
                )
                ui.button("Apply", icon="check", on_click=lambda: submit_configuration(lidar_processor)).props(
                    "unelevated no-caps"
                ).classes("w-full")

            with ui.card().classes(CARD):
                ui.label("Visualization").classes(SECTION)
                ui.separator().classes("bg-slate-800")
                toggle_btn = ui.button(
                    "Start Visualization", icon="play_arrow", on_click=toggle_visualization
                ).props("unelevated no-caps color=primary").classes("w-full")

            with ui.card().classes(CARD):
                ui.label("Statistics").classes(SECTION)
                ui.separator().classes("bg-slate-800")
                with ui.grid(columns=2).classes("w-full gap-3"):
                    with ui.column().classes("gap-0"):
                        ui.label("Targets").classes("text-xs text-slate-400")
                        target_label = ui.label("0").classes("text-2xl font-mono text-white")

        # Right column: lidar view
        with ui.column().classes("gap-4").style("flex: 1 1 0; min-width: 0;"):
            with ui.card().classes(CARD):
                with ui.row().classes("w-full items-center"):
                    ui.label("Lidar View").classes(SECTION)
                    ui.space()
                    ui.label(f"Range: {MAX_RANGE_MM / 1000:g} m").classes(
                        "text-xs font-mono text-slate-400"
                    )
                view = ui.interactive_image(size=(WIDTH, HEIGHT)).classes(
                    "w-full rounded-lg bg-black border border-slate-800"
                )
                view.props("fit=fill")
                view.style(
                    f"max-width: {WIDTH}px; aspect-ratio: {WIDTH} / {HEIGHT}; "
                    "image-rendering: pixelated;"
                )

    # ---------- Refresh loop ----------
    seen = {"image": 0, "targets": 0, "box": 0}
    drawn = {"targets": "", "box": ""}

    def refresh():
        if lidar_processor is not None:
            system_status["nt_connected"] = nt_interface.get_nt_connected()
            system_status["sensor_connected"] = lidar_processor.is_connected()

            stats = lidar_processor.get_status()
            target_label.text = str(stats["last_target_count"])

        set_status(sensor_row, system_status["sensor_connected"])
        set_status(nt_row, system_status["nt_connected"])

        if system_status["visualization_active"]:
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


def run_gui(lidar_processor=None):
    """Build the page and start the web server (blocks; call from the main thread)."""

    @ui.page("/")
    def index():
        main(lidar_processor)  # runs per browser client; the processor is shared

    # reload=False: the auto-reloader would re-import this module and
    # try to open the LIDAR serial port a second time.
    ui.run(title="FRC Lidar GUI", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    run_gui()




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