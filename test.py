from PIL import Image
import numpy as np

# Create a sample NumPy array (e.g., a grayscale image)
data = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

# Convert the NumPy array to a Pillow Image object
img = Image.fromarray(data)

# Save the image
img.save("output_image_pil.png")
