import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

root = Path.home() / "satellite_geo/fy4a"
library = root / "geographic_library"

# 样例：卫星 UTC 时间、裁剪起始行列、行列数量。
timestamp = "201801010000"
row, col, height, width = 131, 102, 256, 256
year = int(timestamp[:4])

records = [
    json.loads(line)
    for line in (library / "datasets.jsonl").read_text().splitlines()
    if line.strip()
]
matches = [
    item for item in records
    if item["category"] == "population"
    and item.get("representative_year") == year
]
if len(matches) != 1:
    raise RuntimeError(f"没有唯一适用的 {year} 年人口数据")

population = matches[0]

with np.load(root / "coordinates_approx.npz") as coordinates:
    lat = coordinates["latitude"]
    lon = coordinates["longitude"]

dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
west = float(lon[col] - dx/2)
east = float(lon[col + width - 1] + dx/2)
north = float(lat[row] + dy/2)
south = float(lat[row + height - 1] - dy/2)

total = 0.0
valid_count = 0
missing_count = 0

with rasterio.open(population["file"]) as src:
    if src.crs.to_epsg() != 4326:
        raise RuntimeError("人口数据坐标系不是预期的 EPSG:4326")

    # 选择中心落在查询范围内的源人口像元。
    window = from_bounds(west, south, east, north, src.transform)
    r0 = max(0, int(np.ceil(window.row_off - 0.5)))
    r1 = min(src.height, int(np.ceil(
        window.row_off + window.height - 0.5)))
    c0 = max(0, int(np.ceil(window.col_off - 0.5)))
    c1 = min(src.width, int(np.ceil(
        window.col_off + window.width - 0.5)))

    if r1 <= r0 or c1 <= c0:
        raise RuntimeError("查询范围与人口文件不相交")

    # 分块读取，避免一次加载整个区域。
    for start in range(r0, r1, 256):
        rows = min(256, r1 - start)
        values = src.read(
            1, window=Window(c0, start, c1-c0, rows), masked=True
        )
        mask = np.ma.getmaskarray(values)
        usable = ~mask & np.isfinite(values.data) & (values.data >= 0)
        total += float(values.data[usable].sum(dtype="float64"))
        valid_count += int(usable.sum())
        missing_count += int((~usable).sum())

result = {
    "satellite_time_utc": "2018-01-01T00:00:00Z",
    "crop": {
        "row": row, "column": col,
        "height": height, "width": width,
        "index_origin": 0
    },
    "geolocation_status": "approximate_from_readme",
    "bounds_west_south_east_north": [west, south, east, north],
    "population_dataset_id": population["dataset_id"],
    "population_year": year,
    "population_estimate_over_valid_source_pixels": (
        total if valid_count else None
    ),
    "valid_source_pixels": valid_count,
    "missing_or_invalid_source_pixels": missing_count,
    "method": "Sum source pixels whose centers fall within query bounds; boundary pixels are not fractionally allocated.",
    "limitation": "Missing coverage is excluded, not treated as zero; this is not guaranteed complete population for the whole region."
}

output = root / "population_match_example.jsonl"
output.write_text(
    json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2))
print("样例已保存：", output)
