import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, help="UTC日期，YYYY-MM-DD")
parser.add_argument("--row", type=int, required=True)
parser.add_argument("--column", type=int, required=True)
args = parser.parse_args()

date = datetime.strptime(args.date, "%Y-%m-%d")
r, c = args.row, args.column
if not (0 <= r < 518 and 0 <= c < 460):
    raise ValueError("行号应为0–517，列号应为0–459")

root = Path.home() / "satellite_geo/fy4a"
index = json.loads(
    (root / "geography_index.jsonl").read_text().strip()
)
lookup = json.loads(
    Path(index["county_lookup_file"]).read_text()
)

result = {
    "date_utc": args.date,
    "grid_id": index["grid_id"],
    "row": r,
    "column": c,
    "index_origin": 0,
    "geolocation_status": "approximate_from_readme"
}

with np.load(index["geography_file"]) as geo:
    lat = geo["latitude"].copy()
    lon = geo["longitude"].copy()
    result["latitude_approx"] = float(lat[r])
    result["longitude_approx"] = float(lon[c])
    number = int(geo["county_id"][r, c])
    result["county_reference"] = {
        "boundary_year": lookup["boundary_year"],
        "source": lookup["boundary_source"],
        "match": lookup["regions"].get(str(number)),
        "note": "Historical boundary reference; not verified as the boundary at the query date."
    }

with np.load(index["terrain_statistics"]["file"]) as terrain:
    if not (
        np.array_equal(terrain["latitude"], lat)
        and np.array_equal(terrain["longitude"], lon)
    ):
        raise ValueError("地形网格与公共网格不同")

    valid = bool(terrain["valid_mask"][r, c])
    result["terrain"] = {
        "has_valid_data": valid,
        "units": "m",
        "note": "Surface-height statistics; partial source coverage is possible. Extreme values have not all been quality reviewed."
    }
    for name in (
        "surface_elevation_mean_m",
        "surface_elevation_min_m",
        "surface_elevation_max_m",
        "terrain_relief_m",
    ):
        result["terrain"][name] = (
            float(terrain[name][r, c]) if valid else None
        )

for category, section in (
    ("population", "annual_population"),
    ("land_cover", "annual_land_cover"),
):
    entry = index.get(section, {}).get(str(date.year))
    if entry is None:
        result[category] = {
            "status": "missing_matching_year",
            "requested_year": date.year,
            "available_years": sorted(index.get(section, {}))
        }
        continue

    with np.load(entry["file"], allow_pickle=False) as data:
        if not (
            np.array_equal(data["latitude"], lat)
            and np.array_equal(data["longitude"], lon)
        ):
            raise ValueError(f"{category} 网格与公共网格不同")

        valid = bool(data["has_valid_data"][r, c])
        item = {
            "status": "has_valid_data" if valid else "no_valid_data",
            "year": date.year,
            "source_file": entry["file"],
            "valid_source_pixel_count":
                int(data["valid_source_pixel_count"][r, c]),
            "note": "Has valid data does not guarantee complete coverage."
        }

        if category == "population":
            item["population_over_valid_source_pixels"] = (
                float(data["population_over_valid_source_pixels"][r, c])
                if valid else None
            )
            item["source_extent_covers_cell"] = bool(
                data["source_extent_covers_cell"][r, c]
            )
            item["all_selected_source_pixels_valid"] = bool(
                data["all_selected_source_pixels_valid"][r, c]
            )
            item["units"] = "people, annual modelled estimate"
        else:
            item["class_proportions"] = (
                {
                    str(name): float(data["class_proportions"][i, r, c])
                    for i, name in enumerate(data["class_names"])
                }
                if valid else None
            )
            item["missing_or_unclassified_source_pixel_count"] = int(
                data["missing_or_unclassified_source_pixel_count"][r, c]
            )

        result[category] = item

output_dir = root / "pixel_queries"
output_dir.mkdir(exist_ok=True)
output = output_dir / f"{args.date}_r{r}_c{c}.jsonl"
output.write_text(
    json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
print("已保存：", output)
