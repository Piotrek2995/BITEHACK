import planetary_computer as pc
from pystac_client import Client
import stackstac
import rioxarray
import numpy as np
import json

# --- Parametry ---
BBOX = [19.00001667, 49.66669, 19.06264, 49.70834]
DATE_RANGE = "2024-01-01/2024-02-28"
MAX_CLOUD = 10

NDSI_THRESHOLD = 0.4
SNOW_FREQ_THRESHOLD = 0.4  # poniżej NIE pokazujemy

# --- Połączenie STAC ---
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime=DATE_RANGE,
    query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
)

items = list(search.items())
print(f"Znaleziono scen: {len(items)}")

# --- Stack ---
stack = stackstac.stack(
    items,
    assets=["B03", "B11"],
    resolution=20,
    bounds_latlon=BBOX,
    epsg=32634,
)

# --- NDSI ---
green = stack.sel(band="B03")
swir = stack.sel(band="B11")
ndsi = (green - swir) / (green + swir)

# --- ŚNIEG (per data) ---
snow_mask = ndsi > NDSI_THRESHOLD

# --- CZĘSTOŚĆ ---
snow_frequency = snow_mask.mean(dim="time").compute()

# --- BINARNA MAPA DO WIZUALIZACJI ---
snow_binary = (snow_frequency >= SNOW_FREQ_THRESHOLD).astype("uint8")

# --- PROCENT ŚNIEGU ---
total_pixels = np.isfinite(snow_frequency.values).sum()
snow_pixels = (snow_binary.values == 1).sum()

snow_percent = round(100 * snow_pixels / total_pixels, 1)

print(f"❄️ Pokrycie śniegiem: {snow_percent}%")

# --- ZAPIS RASTRA ---
snow_binary.rio.to_raster(
    "snow_binary.tif",
    compress="lzw"
)

# --- ZAPIS METADATA (dla Cesium UI) ---
with open("stats.json", "w") as f:
    json.dump(
        {
            "snow_percent": snow_percent,
            "date_range": DATE_RANGE,
        },
        f,
        indent=2
    )

print("✅ Zapisano: snow_binary.tif + stats.json")
