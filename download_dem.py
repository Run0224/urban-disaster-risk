import json
import re
import urllib.request
from pathlib import Path
import numpy as np
import rasterio

folder = Path.home() / "satellite_geo/fy4a/sources/dem"
folder.mkdir(parents=True, exist_ok=True)
base = "https://copernicus-dem-90m.s3.amazonaws.com"

# 与前面区县图使用同一套近似卫星坐标。
lat = 26.0319 - (np.arange(518) - 131) * 700 / 256 * 0.01
lon = 108.505 + (np.arange(460) - 102) * 900 / 256 * 0.01
dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
west, east = float(lon[0] - dx/2), float(lon[-1] + dx/2)
south, north = float(lat[-1] - dy/2), float(lat[0] + dy/2)

print("读取官方瓦片目录……", flush=True)
with urllib.request.urlopen(base + "/tileList.txt", timeout=120) as r:
    listing = r.read().decode("utf-8")
(folder / "tileList.txt").write_text(listing)

pattern = r"Copernicus_DSM_COG_30_([NS])(\d+)_00_([EW])(\d+)_00_DEM"
tiles = set()

for match in re.finditer(pattern, listing):
    ns, y, ew, x = match.groups()
    y = int(y) * (1 if ns == "N" else -1)
    x = int(x) * (1 if ew == "E" else -1)

    # 使用略宽的候选范围，避免源瓦片半像元偏移造成遗漏。
    if x < east + 0.01 and x + 1 > west - 0.01:
        if y < north + 0.01 and y + 1 > south - 0.01:
            tiles.add(match.group())

manifest = {
    "dataset": "Copernicus DEM GLO-90",
    "source": base,
    "satellite_geolocation": "approximate_from_readme",
    "satellite_bounds": [west, south, east, north],
    "candidate_tiles": sorted(tiles),
    "completed": [],
    "failed": [],
}
print("需要检查/下载的瓦片数：", len(tiles), flush=True)

for index, tile in enumerate(sorted(tiles), 1):
    target = folder / f"{tile}.tif"
    temporary = folder / f"{tile}.part"
    url = f"{base}/{tile}/{tile}.tif"

    try:
        valid = False
        if target.exists():
            try:
                with rasterio.open(target) as src:
                    valid = src.crs is not None and src.count == 1
            except Exception:
                pass

        if not valid:
            with urllib.request.urlopen(url, timeout=120) as response:
                with temporary.open("wb") as file:
                    while chunk := response.read(1024 * 1024):
                        file.write(chunk)
            with rasterio.open(temporary) as src:
                if src.crs is None or src.count != 1:
                    raise RuntimeError("瓦片格式不符合预期")
            temporary.replace(target)

        manifest["completed"].append(tile)
        print(f"[{index}/{len(tiles)}] 完成 {tile}", flush=True)

    except Exception as error:
        manifest["failed"].append({"tile": tile, "error": str(error)})
        print(f"[{index}/{len(tiles)}] 失败 {tile}: {error}", flush=True)

    (folder / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

print("成功：", len(manifest["completed"]))
print("失败：", len(manifest["failed"]))
print("下载记录：", folder / "download_manifest.json")
