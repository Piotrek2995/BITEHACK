import planetary_computer as pc
from pystac_client import Client
import stackstac
import rioxarray  # enables .rio accessor

# --- Parametry ---
BBOX = [19.03, 49.69, 19.15, 49.78]  # Szczyrk
DATE_RANGE = "2024-01-01/2024-02-28"
MAX_CLOUD = 10  # %

# --- Połączenie z Planetary Computer ---
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

# --- Wyszukiwanie Sentinel-2 ---
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime=DATE_RANGE,
    query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
)

items = list(search.items())
print(f"Znaleziono scen: {len(items)}")

# --- Stackowanie pasm ---
stack = stackstac.stack(
    items,
    assets=["B03", "B11"],
    resolution=20,
    bounds_latlon=BBOX,
    epsg=32634,  # ← KLUCZOWA LINIA
)

# --- NDSI ---
green = stack.sel(band="B03")
swir = stack.sel(band="B11")

ndsi = (green - swir) / (green + swir)

# --- Maskowanie śniegu ---
snow_mask = ndsi > 0.4

# --- Częstość występowania śniegu ---
snow_frequency = snow_mask.mean(dim="time").compute()  # wymuś obliczenie

# --- Zapis do GeoTIFF ---
output_tiff = "snow_frequency.tif"
snow_frequency.rio.to_raster(output_tiff, compress="lzw")
print(f"Zapisano: {output_tiff}")
