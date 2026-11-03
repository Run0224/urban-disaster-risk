import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

folder = Path.home() / "satellite_geo/fy4a"
dem_folder = folder / "sources/dem"

coordinates = np.load(folder / "coordinates_approx.npz")
lat = coordinates["latitude"]
lon = coordinates["longitude"]

height, width = len(lat), len(lon)
dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])

transform = from_origin(
    float(lon[0] - dx / 2),
    float(lat[0] + dy / 2),
    dx, dy,
)

manifest = json.loads(
    (dem_folder / "download_manifest.json").read_text()
)

elevation = np.full((height, width), np.nan, dtype="float32")

for tile in manifest["completed"]:
    with rasterio.open(dem_folder / f"{tile}.tif") as src:
        source = src.read(1, masked=True).astype("float32").filled(np.nan)
        part = np.full_like(elevation, np.nan)

        reproject(
            source=source,
            destination=part,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        use = np.isfinite(part) & ~np.isfinite(elevation)
        elevation[use] = part[use]

valid = np.isfinite(elevation)
if not valid.any():
    raise RuntimeError("没有匹配到有效高程，暂停保存")

output = folder / "static_geography_approx.npz"
np.savez_compressed(
    output,
    latitude=lat,
    longitude=lon,
    surface_elevation_center_m=elevation,
    elevation_valid_mask=valid,
    county_id=np.load(folder / "county_id.npy"),
)

metadata = {
    "geolocation_status": "approximate_from_readme",
    "shape": [height, width],
    "elevation_source": "Copernicus DEM GLO-90, AWS 2021 release",
    "elevation_type": "digital_surface_model",
    "elevation_units": "metres",
    "vertical_reference": "EGM2008 height",
    "sampling": "nearest source sample at approximate satellite pixel center",
    "missing_value": "NaN; not automatically interpreted as ocean",
    "county_lookup": str(folder / "county_lookup.json"),
    "note": "Not pixel-area mean elevation; geolocation is approximate."
}

(folder / "static_geography_approx.metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("已保存：", output)
print("高程数组形状：", elevation.shape)
print("有效高程像素：", int(valid.sum()))
print("无有效高程像素：", int((~valid).sum()))
print("有效高程范围（米）：",
      float(elevation[valid].min()),
      float(elevation[valid].max()))
