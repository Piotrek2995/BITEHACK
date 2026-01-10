import rasterio
import numpy as np
from PIL import Image

with rasterio.open("snow_binary.tif") as src:
    data = src.read(1)

h, w = data.shape
rgba = np.zeros((h, w, 4), dtype=np.uint8)

# śnieg = biały
rgba[data == 1] = [255, 255, 255, 200]  # lekka przezroczystość

img = Image.fromarray(rgba, mode="RGBA")
img.save("snow_binary.png")

print("✅ snow_binary.png gotowe")
