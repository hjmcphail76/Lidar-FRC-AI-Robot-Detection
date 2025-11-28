import asyncio
from PIL import Image
import numpy as np
import os
import infrence

# Queue to store batches of points NOT lines here
image_queue = asyncio.Queue()


def enqueue_images(img):
    """Add a new batch of points to the queue."""
    image_queue.put_nowait(img)


async def run_save_data(debugging=False):
    dataCount = 0

    while True:
        latest = None

        # Drain the queue, keeping only the latest batch
        while not image_queue.empty():
            latest = await image_queue.get()

        if latest is not None:
            latest.save(
                f"photo-output-results/output_image_pil{dataCount}.png")

            if not debugging:
                dataCount += 1

        else:
            print("No new data")
        if not debugging:
            await asyncio.sleep(2)  # save data cycle delay
        else:
            await asyncio.sleep(0.05)


def get_start_data_collection(debugging=False):
    """Return the coroutine for use in asyncio.gather."""
    return run_save_data(debugging=debugging)
