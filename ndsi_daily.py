import planetary_computer as pc
from pystac_client import Client
import stackstac
import numpy as np
import rasterio
from datetime import datetime, timedelta
import os

# ---------------- PARAMETRY ----------------
BBOX = [19.00001667, 49.66669, 19.06264, 49.70834]
EPSG = 32634
RESOLUTION = 20
MAX_CLOUD = 30
NDSI_THRESHOLD = 0.4

START_YEAR = 2023
END_YEAR = 2024

OUT_DIR = "ndsi_daily"
os.makedirs(OUT_DIR, exist_ok=True)
# -------------------------------------------

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

def daterange(start, end):
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)

for year in range(START_YEAR, END_YEAR):
    print(f"\n❄️ Sezon {year}/{year+1}")

    start_date = datetime(year, 12, 1)
    end_date   = datetime(year + 1, 2, 28)

    for day in daterange(start_date, end_date):
        date_str = day.strftime("%Y-%m-%d")
        print(f" ▶ {date_str}")

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=BBOX,
            datetime=f"{date_str}/{date_str}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        )

        items = list(search.items())
        if not items:
            print("   ⚠️ brak scen")
            continue

        stack = stackstac.stack(
            items,
            assets=["B03", "B11"],
            resolution=RESOLUTION,
            bounds_latlon=BBOX,
            epsg=EPSG,
            fill_value=np.nan,
        )

        green = stack.sel(band="B03")
        swir  = stack.sel(band="B11")

        ndsi = (green - swir) / (green + swir)
        ndsi = ndsi.where(np.isfinite(ndsi))

        # median jeśli >1 scena
        ndsi_day = ndsi.median(dim="time", skipna=True).values

        out_path = f"{OUT_DIR}/ndsi_{date_str}.tif"

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=ndsi_day.shape[0],
            width=ndsi_day.shape[1],
            count=1,
            dtype="float32",
            crs=f"EPSG:{EPSG}",
            transform=stack.transform,
            nodata=-9999,
            compress="deflate",
        ) as dst:
            dst.write(np.nan_to_num(ndsi_day, nan=-9999), 1)

        print(f"   ✅ {out_path}")
