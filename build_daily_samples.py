import json
import subprocess
from datetime import datetime
from pathlib import Path

root = Path.home() / "satellite_geo/fy4a"
index = json.loads(
    (root / "geography_index.jsonl").read_text().strip()
)

day = "20180101"
year = day[:4]
remote_dir = (
    "/mnt/data/zk/diffcast_3d/data/sat_6min/"
    f"{day[:4]}/{day[4:6]}/{day[6:8]}"
)

# 只列出远程文件，不修改或下载卫星数据。
response = subprocess.run(
    [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "ices@192.168.48.29",
        f"find -L {remote_dir} -type f -name '*.npy'"
    ],
    check=True, capture_output=True, text=True,
)

def annual_reference(section):
    entry = index.get(section, {}).get(year)
    if entry is None:
        return {
            "status": "missing_matching_year",
            "requested_year": int(year)
        }
    if not Path(entry["file"]).is_file():
        raise FileNotFoundError(entry["file"])
    return {
        "status": "file_available_check_pixel_masks_when_reading",
        "year": int(year),
        "file": entry["file"],
        "metadata_file": entry["metadata_file"]
    }

population = annual_reference("annual_population")
land_cover = annual_reference("annual_land_cover")

for path in (
    index["geography_file"],
    index["county_lookup_file"],
    index["terrain_statistics"]["file"],
):
    if not Path(path).is_file():
        raise FileNotFoundError(path)

records = []
seen = set()

for path in sorted(response.stdout.splitlines()):
    stamp = Path(path).stem
    if len(stamp) != 12 or not stamp.isdigit():
        continue
    time = datetime.strptime(stamp, "%Y%m%d%H%M")
    if stamp[:8] != day:
        continue
    if stamp in seen:
        raise RuntimeError(f"同一时刻出现多个文件：{stamp}")
    seen.add(stamp)

    records.append({
        "sample_id": f"fy4a_{stamp}_center256",
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "satellite": {
            "host": "192.168.48.29",
            "user": "ices",
            "path": path,
            "expected_shape": [8, 518, 460],
            "contents_verified_in_this_step": False
        },
        "grid_id": index["grid_id"],
        "geolocation_status": "approximate_from_readme",
        "crop": {
            "row": 131, "column": 102,
            "height": 256, "width": 256,
            "index_origin": 0
        },
        "geography": {
            "static_file": index["geography_file"],
            "terrain_file": index["terrain_statistics"]["file"],
            "county_lookup_file": index["county_lookup_file"],
            "county_boundary_year": 2017,
            "county_usage": "historical_reference_only",
            "population": population,
            "land_cover": land_cover
        },
        "reading_rule": (
            "Apply the same crop to satellite and spatial geography arrays; "
            "read validity masks; convert crop-local pixel indices to "
            "global indices by adding crop row and column."
        )
    })

if not records:
    raise RuntimeError("没有找到符合日期的卫星文件，未生成索引")

folder = root / "samples"
folder.mkdir(exist_ok=True)
output = folder / f"fy4a_{day}_center256.jsonl"
temporary = output.with_suffix(".jsonl.tmp")
temporary.write_text(
    "".join(
        json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
        for item in records
    ),
    encoding="utf-8",
)
temporary.replace(output)

print("当天实际找到的帧数：", len(records))
print("首个时刻：", records[0]["time_utc"])
print("最后时刻：", records[-1]["time_utc"])
print("人口：", population["status"])
print("土地覆盖：", land_cover["status"])
print("已保存：", output)
