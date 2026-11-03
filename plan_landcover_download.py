import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

root = Path.home() / "satellite_geo/fy4a"
folder = root / "geographic_library/land_cover"
folder.mkdir(parents=True, exist_ok=True)

with np.load(root / "coordinates_approx.npz") as geo:
    lat, lon = geo["latitude"], geo["longitude"]

dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
west, east = float(lon[0]-dx/2), float(lon[-1]+dx/2)
south, north = float(lat[-1]-dy/2), float(lat[0]+dy/2)

base = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
pattern = re.compile(
    r"ESA_WorldCover_10m_2020_v100_([NS])(\d+)([EW])(\d+)_Map\.tif$"
)

tiles = []
token = None
while True:
    parameters = {
        "list-type": "2",
        "prefix": "v100/2020/map/",
        "max-keys": "1000"
    }
    if token:
        parameters["continuation-token"] = token

    url = base + "/?" + urllib.parse.urlencode(parameters)
    with urllib.request.urlopen(url, timeout=120) as response:
        document = ET.fromstring(response.read())

    for item in document.findall("s:Contents", ns):
        key = item.findtext("s:Key", namespaces=ns)
        match = pattern.search(key)
        if not match:
            continue

        a, y, b, x = match.groups()
        y = int(y) * (1 if a == "N" else -1)
        x = int(x) * (1 if b == "E" else -1)

        # WorldCover 分类图瓦片为 3°×3°。
        if x < east and x+3 > west and y < north and y+3 > south:
            size = int(item.findtext("s:Size", namespaces=ns))
            local = folder / Path(key).name
            tiles.append({
                "key": key,
                "url": base + "/" + key,
                "size_bytes": size,
                "file": str(local),
                "existing_size_matches": (
                    local.exists() and local.stat().st_size == size
                )
            })

    token = document.findtext("s:NextContinuationToken", namespaces=ns)
    if not token:
        break

if not tiles:
    raise RuntimeError("没有查询到匹配瓦片，未生成计划")

plan = {
    "dataset": "ESA WorldCover 2020 v100",
    "year": 2020,
    "geolocation_status": "approximate_from_readme",
    "satellite_bounds": [west, south, east, north],
    "tiles": sorted(tiles, key=lambda t: t["key"])
}

output = folder / "download_plan_2020.json"
output.write_text(
    json.dumps(plan, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

remaining = [t for t in tiles if not t["existing_size_matches"]]
print("需要的瓦片数量：", len(tiles))
print("已有且大小匹配：", len(tiles)-len(remaining))
print("还需下载：", len(remaining))
print("还需下载大小：",
      round(sum(t["size_bytes"] for t in remaining)/1024**3, 2), "GiB")
print("计划已保存：", output)
print("本次没有下载地图瓦片。")
