import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument(
    "--bounds", type=float, nargs=4, required=True,
    metavar=("WEST", "SOUTH", "EAST", "NORTH")
)
args = parser.parse_args()

west, south, east, north = args.bounds
if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
    raise ValueError("经纬度范围不合法")

library = Path.home() / "satellite_geo/fy4a/geographic_library"
records = [
    json.loads(line)
    for line in (library / "datasets.jsonl").read_text().splitlines()
    if line.strip()
]

result = {
    "query_year": args.year,
    "query_bounds": args.bounds,
    "geolocation_status": "approximate_from_readme",
    "categories": {}
}

for category in ("terrain", "administrative_boundary",
                 "population", "land_cover"):
    entries = [r for r in records if r["category"] == category]
    candidates = []
    rejected = []

    for item in entries:
        dataset_id = item["dataset_id"]

        if not Path(item["file"]).is_file():
            rejected.append({"dataset_id": dataset_id,
                             "reason": "file_missing"})
            continue

        if item["time_type"] == "representative_year":
            if item.get("representative_year") != args.year:
                rejected.append({
                    "dataset_id": dataset_id,
                    "reason": "year_mismatch",
                    "available_year": item.get("representative_year")
                })
                continue

        coverage = "not_verified"
        bounds = item.get("bounds_west_south_east_north")

        if bounds:
            w, s, e, n = bounds
            intersects = west < e and east > w and south < n and north > s
            if not intersects:
                rejected.append({"dataset_id": dataset_id,
                                 "reason": "outside_extent"})
                continue
            coverage = (
                "within_file_extent"
                if w <= west and east <= e and s <= south and north <= n
                else "partial_extent"
            )

        candidates.append({
            "dataset_id": dataset_id,
            "file": item["file"],
            "spatial_coverage": coverage,
            "note": "File extent does not guarantee valid data at every location."
        })

    result["categories"][category] = {
        "status": "candidate_found" if candidates else "no_matching_data",
        "candidates": candidates,
        "rejected": rejected
    }

output = library / f"query_result_{args.year}.json"
output.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

for category, info in result["categories"].items():
    print(category, "：", info["status"])
    for candidate in info["candidates"]:
        print("  ", candidate["dataset_id"],
              candidate["spatial_coverage"])
    for rejected in info["rejected"]:
        print("  未采用：", rejected)

print("详细结果：", output)
