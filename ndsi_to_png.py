import rasterio
import numpy as np
from PIL import Image
import os

NDSI_THRESHOLD = 0.4

IN_DIR = "ndsi_daily"
OUT_DIR = "png_daily"
os.makedirs(OUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(IN_DIR)):
    if not fname.endswith(".tif"):
        continue

    date = fname.replace("ndsi_", "").replace(".tif", "")
    print(f"🖼️ {date}")

    with rasterio.open(f"{IN_DIR}/{fname}") as src:
        ndsi = src.read(1)

    snow = ndsi >= NDSI_THRESHOLD

    h, w = snow.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # śnieg = biały
    rgba[snow] = [255, 255, 255, 220]

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(f"{OUT_DIR}/snow_{date}.png")

print("✅ Wszystkie PNG gotowe")
