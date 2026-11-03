import json
from pathlib import Path
import numpy as np

ROOT = Path.home() / "satellite_geo/fy4a"
FOLDER = ROOT / "risk_features_trial"

with (FOLDER / "regional_features.jsonl").open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

with np.load(FOLDER / "past_hour_arrays.npz", allow_pickle=False) as d:
    lat = d["latitude"].copy()
    lon = d["longitude"].copy()
    coverage = d["radar_coverage_mask"].copy()

with np.load(ROOT / "terrain_statistics_approx_v2.npz",
             allow_pickle=False) as d:
    tlat = d["latitude"].copy()
    tlon = d["longitude"].copy()
    valid = d["valid_mask"].copy()
    fields = {
        key: d[key].copy() for key in (
            "surface_elevation_mean_m",
            "surface_elevation_min_m",
            "surface_elevation_max_m",
            "terrain_relief_m",
        )
    }

if not (np.all(np.diff(tlat) < 0) and np.all(np.diff(tlon) > 0)):
    raise ValueError("地形坐标方向异常")
if coverage.shape != (lat.size, lon.size):
    raise ValueError("雷达覆盖尺寸异常")
if valid.shape != (tlat.size, tlon.size):
    raise ValueError("地形掩膜尺寸异常")

def nearest_indices(axis, targets):
    # axis必须递增；返回最近的源像素中心索引。
    upper = np.clip(np.searchsorted(axis, targets), 1, axis.size-1)
    lower = upper-1
    return np.where(
        np.abs(targets-axis[lower]) <= np.abs(targets-axis[upper]),
        lower, upper,
    )

if (lat.min() < tlat.min() or lat.max() > tlat.max()
    or lon.min() < tlon.min() or lon.max() > tlon.max()):
    raise ValueError("雷达目标中心超出地形坐标中心范围")

ri = tlat.size-1-nearest_indices(tlat[::-1], lat)
ci = nearest_indices(tlon, lon)
source_ids = ri[:, None]*tlon.size + ci[None, :]

for record in records:
    region = record["region_id"]
    top = int(region.split("_")[0][1:])
    left = int(region.split("_")[1][1:])
    sl = (slice(top, top+70), slice(left, left+90))
    # 同一地形源格只统计一次，避免重采样重复计数。
    selected = np.unique(source_ids[sl][coverage[sl]])
    usable = selected[valid.ravel()[selected]]

    terrain = {
        "method": "unique_nearest_source_cells_at_covered_radar_centers",
        "source_grid_shape": list(valid.shape),
        "selected_source_cells": int(selected.size),
        "valid_source_cells": int(usable.size),
        "note": (
            "近似坐标下关联的原地形网格统计；源格可能跨区域边界。"
            "不是新的高分辨率DEM，不提供坡度或完整覆盖保证。"
        ),
    }
    for key, array in fields.items():
        values = array.ravel()[usable]
        values = values[np.isfinite(values)]
        terrain[key] = {
            "finite_source_cells": int(values.size),
            "median_of_source_cell_statistics":
                float(np.median(values)) if values.size else None,
            "minimum_of_source_cell_statistics":
                float(values.min()) if values.size else None,
            "maximum_of_source_cell_statistics":
                float(values.max()) if values.size else None,
        }
    record["terrain_reference"] = terrain

target = FOLDER / "regional_features_with_terrain.jsonl"
with target.open("x", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(
            record, ensure_ascii=False, allow_nan=False
        ) + "\n")

print("区域总数：", len(records))
print("有关联有效地形的区域：", sum(
    r["terrain_reference"]["valid_source_cells"] > 0 for r in records
))
print("已保存：", target)
print("原特征文件保留；未加入灾害标签，未生成概率或损失。")
