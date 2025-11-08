import pygame
import random
import asyncio

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Scatter Plot")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Queue to store points
point_queue = asyncio.Queue()


def enqueue_points(xs, ys):
    """Add a new point to the queue"""
    point_queue.put_nowait((xs, ys))


async def run_plot():
    points = []  # all rev points to draw
    running = True

    while running:
        # Drain the queue and add new points to the main list
        # @TODO fix this since we don't technicaly need this :D
        while not point_queue.empty():
            point = await point_queue.get()
            points.append(points)

        # Draw the points
        screen.fill(WHITE)

        latest_rev_points_in_queue = await point_queue.get()

        for i in range(len(latest_rev_points_in_queue[0])):
            pygame.draw.circle(
                screen,
                BLACK,
                (
                    int(latest_rev_points_in_queue[0][i] + WIDTH / 2),
                    int(latest_rev_points_in_queue[1][i] + HEIGHT / 2),
                ),
                4,
            )
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        await asyncio.sleep(0.01)  # yield control


async def main():
    # Run producer and plotter concurrently
    await asyncio.gather(run_plot())


def get_start_plot():
    return main()
