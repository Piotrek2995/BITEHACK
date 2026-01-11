import rasterio
import numpy as np
from PIL import Image
import os
from collections import defaultdict

NDSI_THRESHOLD = 0.4

IN_DIR = "ndsi_daily2"
OUT_DIR = "png_daily2"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# 1. TIFF-y dzienne → PNG-i
# ===============================
all_ndsi_data = {}
snow_frequency = None
shape = None

for fname in sorted(os.listdir(IN_DIR)):
    if not fname.endswith(".tif"):
        continue

    date = fname.replace("ndsi_", "").replace(".tif", "")

    with rasterio.open(os.path.join(IN_DIR, fname)) as src:
        ndsi = src.read(1)
        transform = src.transform
        crs = src.crs

    shape = ndsi.shape
    all_ndsi_data[date] = ndsi

    h, w = ndsi.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    snow = ndsi >= NDSI_THRESHOLD

    # ❄️ ŚNIEG
    rgba[snow, 0] = 255  # R
    rgba[snow, 1] = 255  # G
    rgba[snow, 2] = 255  # B
    rgba[snow, 3] = 255  # ALPHA (WAŻNE!)

    # 🌍 BRAK ŚNIEGU → alpha = 0 (domyślnie)

    Image.fromarray(rgba, "RGBA").save(
        f"{OUT_DIR}/snow_{date}.png"
    )

print(f"✅ PNG dzienne wygenerowane ({len(all_ndsi_data)} dni)")

# ===============================
# 2. AGREGACJA SEZONOWA
# ===============================
if all_ndsi_data and shape:
    h, w = shape
    
    # Policz częstość śniegu dla każdego pixela
    snow_count = np.zeros((h, w), dtype=np.float32)
    total_valid = np.zeros((h, w), dtype=np.float32)

    for date, ndsi in all_ndsi_data.items():
        # Policz tylko pixele z danymi (!=NaN i !=-9999)
        valid = np.isfinite(ndsi) & (ndsi != -9999)
        total_valid[valid] += 1

        # Policz pixele ze śniegiem
        snow = (ndsi >= NDSI_THRESHOLD) & valid
        snow_count[snow] += 1

    # Prawdopodobieństwo śniegu
    snow_prob = np.divide(
        snow_count, 
        total_valid, 
        where=total_valid > 0, 
        out=np.zeros_like(snow_count)
    )

    # ===============================
    # 3. TIFF SEZONOWY
    # ===============================
    # Progi dla kolorowania:
    THRESHOLD_BLUE = 0.3    # od 30% prawdopodobieństwa → niebieski
    THRESHOLD_WHITE = 0.7   # od 70% → biały (pewny śnieg)

    seasonal_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # NIEBIESKI = śnieg od 30-70% (częsty)
    moderate_snow = (snow_prob >= THRESHOLD_BLUE) & (snow_prob < THRESHOLD_WHITE)
    seasonal_rgba[moderate_snow, 0] = 100   # R (ciemniejszy niebieski)
    seasonal_rgba[moderate_snow, 1] = 180   # G
    seasonal_rgba[moderate_snow, 2] = 255   # B
    seasonal_rgba[moderate_snow, 3] = 220   # ALPHA

    # BIAŁY = śnieg od 70%+ (pewny)
    certain_snow = snow_prob >= THRESHOLD_WHITE
    seasonal_rgba[certain_snow, 0] = 255    # R
    seasonal_rgba[certain_snow, 1] = 255    # G
    seasonal_rgba[certain_snow, 2] = 255    # B
    seasonal_rgba[certain_snow, 3] = 255    # ALPHA

    # Zapisz PNG
    Image.fromarray(seasonal_rgba, "RGBA").save(
        f"{OUT_DIR}/snow_seasonal_probability.png"
    )

    # Zapisz też jako TIFF (raw data)
    with rasterio.open(
        f"{OUT_DIR}/snow_seasonal_probability.tif",
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-1,
        compress="deflate",
    ) as dst:
        dst.write(snow_prob.astype(np.float32), 1)

    # Statystyka
    certain_percent = np.round(100 * np.sum(certain_snow) / np.sum(total_valid > 0), 1)
    moderate_percent = np.round(100 * np.sum(moderate_snow) / np.sum(total_valid > 0), 1)

    print(f"\n🎨 MAPA SEZONOWA:")
    print(f"  ⚪ Biały (pewny śnieg 70%+):   {certain_percent}%")
    print(f"  🔵 Niebieski (częsty 30-70%): {moderate_percent}%")
    print(f"✅ Zapisano: snow_seasonal_probability.png + .tif")
