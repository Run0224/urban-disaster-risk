import json
from pathlib import Path
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin

folder = Path.home() / "satellite_geo/fy4a"
boundaries = gpd.read_file(
    folder / "sources/china_county.geojson"
).to_crs("EPSG:4326")

if boundaries.empty or not boundaries.geometry.is_valid.all():
    raise RuntimeError("边界为空或存在无效几何，暂停处理")

# 按 README 推算的近似经纬度，行列从 0 开始。
lat = 26.0319 - (np.arange(518) - 131) * 700 / 256 * 0.01
lon = 108.505 + (np.arange(460) - 102) * 900 / 256 * 0.01

dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
transform = from_origin(
    float(lon[0] - dx / 2),
    float(lat[0] + dy / 2),
    dx, dy,
)

boundaries = boundaries.sort_values("shapeID").reset_index(drop=True)
shapes = []
lookup = {}

for index, row in boundaries.iterrows():
    number = index + 1
    shapes.append((row.geometry, number))
    lookup[str(number)] = {
        "source_name": str(row["shapeName"]),
        "source_id": str(row["shapeID"]),
    }

county_id = rasterize(
    shapes,
    out_shape=(518, 460),
    transform=transform,
    fill=0,
    all_touched=False,
    dtype="int32",
)

np.save(folder / "county_id.npy", county_id)
np.savez_compressed(
    folder / "coordinates_approx.npz",
    latitude=lat,
    longitude=lon,
)

used = [int(value) for value in np.unique(county_id) if value != 0]
metadata = {
    "geolocation_status": "approximate_from_readme",
    "boundary_source": "geoBoundaries gbOpen CHN ADM2",
    "boundary_year": 2017,
    "assignment": "pixel_center",
    "unmatched_value": 0,
    "regions": {str(value): lookup[str(value)] for value in used},
}
(folder / "county_lookup.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("生成完成")
print("区县编号图：", folder / "county_id.npy")
print("名称对照表：", folder / "county_lookup.json")
print("近似坐标：", folder / "coordinates_approx.npz")
print("匹配到的区县数量：", len(used))
