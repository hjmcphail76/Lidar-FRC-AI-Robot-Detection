
import pygame
import asyncio
import numpy as np

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LIDAR Targets + YOLO Boxes")

# Colors
GREEN = (0, 220, 0)
WHITE = (255, 255, 255)
CYAN = (0, 220, 255)
BLACK = (0, 0, 0)

# Match your LIDAR configuration
MAX_RANGE_MM = 5000

# Queues
image_queue = asyncio.Queue()
box_queue = asyncio.Queue()
target_queue = asyncio.Queue()


def enqueue_image(image_np):
    """
    Push a numpy image into the queue.
    Expected shape: (H, W, 3), dtype uint8, RGB.
    """
    image_queue.put_nowait(image_np)


def enqueue_box(x1, y1, x2, y2):
    """
    Legacy YOLO box in screen pixel coordinates.
    """
    box_queue.put_nowait((x1, y1, x2, y2))


def enqueue_targets(targets):
    """
    Push targets returned by find_lidar_targets().

    Each target contains:
        points: Nx2 array of (x, y) in mm
        point_count: number of points
        center_x, center_y: center in mm
        width_mm, height_mm: bounding dimensions
        bbox_mm: (x_min, y_min, x_max, y_max)
    """
    target_queue.put_nowait(targets)


def world_to_screen(x, y):
    """
    Convert LIDAR Cartesian coordinates (mm) to screen pixels.

    Matches the coordinate mapping used by lidar_to_image().
    """
    px = int(
        (x + MAX_RANGE_MM) / (2 * MAX_RANGE_MM) * WIDTH
    )

    py = int(
        (y + MAX_RANGE_MM) / (2 * MAX_RANGE_MM) * HEIGHT
    )

    return (
        max(0, min(WIDTH - 1, px)),
        max(0, min(HEIGHT - 1, py))
    )


def draw_targets(targets):
    """
    Draw clustered points and their bounding boxes.
    Bounding boxes are defined in world coordinates (mm).
    """
    for i, target in enumerate(targets):
        points = target["points"]
        x_min, y_min, x_max, y_max = target["bbox_mm"]

        # Draw points belonging to this cluster in cyan
        for x, y in points:
            px, py = world_to_screen(x, y)
            pygame.draw.circle(screen, CYAN, (px, py), 3)

        # Convert bounding-box corners to screen coordinates
        p1 = world_to_screen(x_min, y_max)
        p2 = world_to_screen(x_max, y_min)

        left = min(p1[0], p2[0])
        top = min(p1[1], p2[1])
        right = max(p1[0], p2[0])
        bottom = max(p1[1], p2[1])

        # Draw bounding box
        pygame.draw.rect(
            screen,
            GREEN,
            pygame.Rect(
                left,
                top,
                max(1, right - left),
                max(1, bottom - top)
            ),
            2
        )

        # Target label
        label = (
            f"T{i + 1} | {target['point_count']} pts | "
            f"{target['width_mm']:.0f}x"
            f"{target['height_mm']:.0f} mm"
        )

        font = pygame.font.Font(None, 20)
        text_surface = font.render(label, True, GREEN)

        label_y = max(0, top - 20)

        # # Small black background for label readability
        # label_bg = text_surface.get_rect(
        #     topleft=(left, label_y)
        # )
        # pygame.draw.rect(screen, BLACK, label_bg)

        # screen.blit(text_surface, (left, label_y))


def draw_yolo_boxes(boxes):
    """
    Draw legacy YOLO boxes in screen pixel coordinates.
    """
    for x1, y1, x2, y2 in boxes:
        pygame.draw.rect(
            screen,
            WHITE,
            pygame.Rect(
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ),
            2
        )


async def run_plot():
    latest_image_surface = None
    latest_targets = []
    latest_boxes = []

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get the latest LIDAR image
        while not image_queue.empty():
            img_np = await image_queue.get()

            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)

            surface = pygame.surfarray.make_surface(
                np.transpose(img_np, (1, 0, 2))
            )

            latest_image_surface = pygame.transform.scale(
                surface, (WIDTH, HEIGHT)
            )

        # Get the latest DBSCAN target groupings
        while not target_queue.empty():
            latest_targets = await target_queue.get()

        # Get latest YOLO boxes
        while not box_queue.empty():
            latest_boxes.append(await box_queue.get())

            # Preserve the existing single-box behavior
            if len(latest_boxes) > 1:
                latest_boxes.pop(0)

        # Redraw at the display refresh rate
        if latest_image_surface is not None:
            screen.blit(latest_image_surface, (0, 0))
        else:
            screen.fill(BLACK)

        # Draw detected LIDAR targets
        draw_targets(latest_targets)

        # Draw YOLO boxes, if provided
        draw_yolo_boxes(latest_boxes)

        pygame.display.flip()

        clock.tick(30)
        await asyncio.sleep(0.01)

    pygame.quit()


def get_start_plot():
    return run_plot()
