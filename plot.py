import pygame
import asyncio

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LIDAR + RANSAC Plot")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Queue to store batches of points (or lines)
point_queue = asyncio.Queue()


def enqueue_points(xs, ys, is_line=False):
    """Add a new batch of points to the queue."""
    point_queue.put_nowait((xs, ys, is_line))


async def run_plot():
    lidar_batches = []  # store recent lidar scans
    line_batches = []  # store recent fitted lines
    running = True
    redraw_needed = False

    clock = pygame.time.Clock()

    while running:
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check for new points (non-blocking)
        while not point_queue.empty():
            xs, ys, is_line = await point_queue.get()
            if is_line:
                line_batches.append((xs, ys))
                if len(line_batches) > 10:  # keep memory bounded
                    line_batches.pop(0)
            else:
                lidar_batches.append((xs, ys))
                if len(lidar_batches) > 3:  # keep only last few revolutions
                    lidar_batches.pop(0)
            redraw_needed = True  # only redraw when something changed

        # Only redraw if new data arrived
        if redraw_needed:
            screen.fill(WHITE)

            # Draw latest lidar revolution
            if lidar_batches:
                xs, ys = lidar_batches[-1]
                for i in range(len(xs)):
                    pygame.draw.circle(
                        screen,
                        BLACK,
                        (int(xs[i] + WIDTH / 2), int(ys[i] + HEIGHT / 2)),
                        3,
                    )

            # Draw latest RANSAC line
            if line_batches:
                xs, ys = line_batches[-1]
                for i in range(len(xs) - 1):
                    pygame.draw.line(
                        screen,
                        RED,
                        (int(xs[i] + WIDTH / 2), int(ys[i] + HEIGHT / 2)),
                        (int(xs[i + 1] + WIDTH / 2), int(ys[i + 1] + HEIGHT / 2)),
                        2,
                    )

            pygame.display.flip()
            redraw_needed = False

        clock.tick(30)  # limit to 30 FPS max to keep CPU low
        await asyncio.sleep(0.01)


def get_start_plot():
    """Return the coroutine for use in asyncio.gather."""
    return run_plot()
