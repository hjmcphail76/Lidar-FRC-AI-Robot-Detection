import pygame
import asyncio
import numpy as np

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LIDAR Image + YOLO Boxes")

# Colors
GREEN = (0, 200, 0)

# Queues
image_queue = asyncio.Queue()
box_queue = asyncio.Queue()


def enqueue_image(image_np):
    """
    Push a numpy image into the queue.
    Expected shape: (H, W, 3), dtype uint8 (RGB)
    """
    image_queue.put_nowait(image_np)


def enqueue_box(x1, y1, x2, y2):
    box_queue.put_nowait((x1, y1, x2, y2))


async def run_plot():
    latest_image_surface = None
    box_batches = []

    running = True
    redraw_needed = False
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Pull image (lidar image from lidar_to_image)
        while not image_queue.empty():
            img_np = await image_queue.get()

            # Ensure RGB uint8
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)

            # Convert numpy image → pygame surface
            surface = pygame.surfarray.make_surface(
                np.transpose(img_np, (1, 0, 2)))

            # Scale to screen
            surface = pygame.transform.scale(surface, (WIDTH, HEIGHT))

            latest_image_surface = surface
            redraw_needed = True

        # YOLO boxes
        if box_queue.empty():
            redraw_needed = True
        while not box_queue.empty():
            box = await box_queue.get()
            box_batches.append(box)
            if len(box_batches) > 1:
                box_batches.pop(0)
            redraw_needed = True

        if redraw_needed:
            # Draw the lidar image
            if latest_image_surface is not None:
                screen.blit(latest_image_surface, (0, 0))

            # Draw YOLO boxes on top
            for (x1, y1, x2, y2) in box_batches:
                pygame.draw.rect(
                    screen,
                    GREEN,
                    pygame.Rect(x1, y1, x2 - x1, y2 - y1),
                    2
                )

            pygame.display.flip()
            redraw_needed = False

        clock.tick(30)
        await asyncio.sleep(0.05)


def get_start_plot():
    return run_plot()
