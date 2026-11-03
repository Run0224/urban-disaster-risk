import json
from pathlib import Path
import numpy as np
import rasterio

folder = Path.home() / "satellite_geo/fy4a"
dem = folder / "sources/dem"

with np.load(folder / "coordinates_approx.npz") as data:
    lat = data["latitude"]
    lon = data["longitude"]

height, width = len(lat), len(lon)
size = height * width
dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
west = float(lon[0] - dx / 2)
north = float(lat[0] + dy / 2)

total = np.zeros(size, dtype="float64")
count = np.zeros(size, dtype="int64")
low = np.full(size, np.inf)
high = np.full(size, -np.inf)

manifest = json.loads((dem / "download_manifest.json").read_text())
if manifest["failed"]:
    raise RuntimeError("存在下载失败的瓦片，请先处理")
tiles = manifest["completed"]

for index, tile in enumerate(tiles, 1):
    with rasterio.open(dem / f"{tile}.tif") as src:
        t = src.transform
        if src.crs.to_epsg() != 4326 or t.b != 0 or t.d != 0:
            raise RuntimeError(f"不支持的源网格：{tile}")

        a = src.read(1, masked=True).astype("float64")
        valid = ~np.ma.getmaskarray(a) & np.isfinite(a.data)

        # 原始高程像素中心的经纬度。
        x = t.c + (np.arange(src.width) + 0.5) * t.a
        y = t.f + (np.arange(src.height) + 0.5) * t.e

        columns = np.floor((x - west) / dx).astype("int64")
        rows = np.floor((north - y) / dy).astype("int64")

        inside = (
            (rows[:, None] >= 0) & (rows[:, None] < height)
            & (columns[None, :] >= 0) & (columns[None, :] < width)
        )
        use = valid & inside
        rr, cc = np.nonzero(use)
        ids = rows[rr] * width + columns[cc]
        values = a.data[rr, cc]

        total += np.bincount(ids, weights=values, minlength=size)
        count += np.bincount(ids, minlength=size)
        np.minimum.at(low, ids, values)
        np.maximum.at(high, ids, values)

    if index % 20 == 0 or index == len(tiles):
        print(f"已处理 {index}/{len(tiles)} 块", flush=True)

valid = count > 0
mean = np.full(size, np.nan)
mean[valid] = total[valid] / count[valid]
low[~valid] = np.nan
high[~valid] = np.nan
relief = high - low

if not valid.any():
    raise RuntimeError("没有有效统计")
if not (
    np.all(mean[valid] >= low[valid] - 1e-6)
    and np.all(mean[valid] <= high[valid] + 1e-6)
):
    raise RuntimeError("统计一致性检查失败，未保存")

output = folder / "terrain_statistics_approx_v2.npz"
np.savez_compressed(
    output,
    latitude=lat,
    longitude=lon,
    surface_elevation_mean_m=mean.reshape(height, width),
    surface_elevation_min_m=low.reshape(height, width),
    surface_elevation_max_m=high.reshape(height, width),
    terrain_relief_m=relief.reshape(height, width),
    valid_mask=valid.reshape(height, width),
    source_sample_count=count.reshape(height, width),
)

metadata = {
    "source": "Copernicus DEM GLO-90 downloaded tiles",
    "geolocation_status": "approximate_from_readme",
    "height_type": "digital_surface_model",
    "units": "metres",
    "method": "Assign valid source pixel centers to satellite cells; compute all statistics from the same samples.",
    "mean": "Arithmetic mean of assigned valid source samples, not exact ground-area weighted mean.",
    "missing": "NaN",
    "coverage_note": "Sample count is not area coverage; partial coverage remains possible."
}
output.with_suffix(".metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("统计一致性检查通过")
print("有效像素：", int(valid.sum()))
print("已保存：", output)
