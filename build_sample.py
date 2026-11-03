import json
from pathlib import Path
import numpy as np

folder = Path.home() / "satellite_geo/fy4a"
population = json.loads(
    (folder / "population_match_example.jsonl")
    .read_text(encoding="utf-8").strip()
)
lookup = json.loads(
    (folder / "county_lookup.json").read_text(encoding="utf-8")
)

crop = population["crop"]
r, c = crop["row"], crop["column"]
h, w = crop["height"], crop["width"]
if (r, c, h, w) != (131, 102, 256, 256):
    raise ValueError("人口样例范围与预期不一致")

region = np.s_[r:r+h, c:c+w]

with np.load(folder / "static_geography_approx.npz") as geo:
    ids = geo["county_id"][region]
    counties = [
        lookup["regions"][str(int(number))]
        for number in np.unique(ids) if number != 0
    ]
    unmatched_county = int((ids == 0).sum())

with np.load(folder / "terrain_statistics_approx_v2.npz") as terrain:
    valid = terrain["valid_mask"][region]
    low = terrain["surface_elevation_min_m"][region]
    high = terrain["surface_elevation_max_m"][region]
    relief = terrain["terrain_relief_m"][region]

    terrain_summary = {
        "valid_satellite_cells": int(valid.sum()),
        "missing_satellite_cells": int((~valid).sum()),
        "minimum_surface_elevation_m": (
            float(low[valid].min()) if valid.any() else None
        ),
        "maximum_surface_elevation_m": (
            float(high[valid].max()) if valid.any() else None
        ),
        "maximum_within_cell_relief_m": (
            float(relief[valid].max()) if valid.any() else None
        ),
        "file": str(folder / "terrain_statistics_approx_v2.npz"),
        "note": "Statistics use available source samples; valid cells may have partial coverage."
    }

sample = {
    "sample_id": "fy4a_201801010000_center256",
    "time_utc": population["satellite_time_utc"],
    "satellite": {
        "host": "192.168.48.29",
        "user": "ices",
        "path": "/mnt/data/zk/diffcast_3d/data/sat_6min/2018/01/01/201801010000/201801010000.npy",
        "crop": crop
    },
    "geolocation_status": "approximate_from_readme",
    "bounds_west_south_east_north": population[
        "bounds_west_south_east_north"
    ],
    "administrative_regions": {
        "source": lookup["boundary_source"],
        "year": lookup["boundary_year"],
        "assignment": "satellite_pixel_center",
        "regions": counties,
        "unmatched_satellite_cells": unmatched_county
    },
    "terrain": terrain_summary,
    "population": population,
    "limitations": [
        "Satellite geolocation is approximate.",
        "County boundaries represent 2017, not necessarily current boundaries.",
        "Population is a 2018 modelled estimate over valid source pixels.",
        "No disaster labels or land-cover data have been added."
    ]
}

output = folder / "sample_2018_center256.jsonl"
output.write_text(
    json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)

print("样例已保存：", output)
print("裁剪范围：", crop)
print("匹配区县数：", len(counties))
print("地形摘要：", terrain_summary)
print("人口估计（仅有效来源像元）：",
      population["population_estimate_over_valid_source_pixels"])
