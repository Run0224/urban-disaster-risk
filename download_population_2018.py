import json
import urllib.request
from pathlib import Path
import rasterio

folder = Path.home() / "satellite_geo/fy4a/geographic_library/population"
catalog = json.loads(
    (folder / "worldpop_china_catalog.json").read_text(encoding="utf-8")
)

matches = [
    item for item in catalog["data"]
    if str(item.get("popyear")) == "2018"
]
if len(matches) != 1:
    raise RuntimeError("2018 年记录不是唯一一条，暂停下载")

record = matches[0]
url = record["files"][0]
if not url.startswith("https://data.worldpop.org/"):
    raise ValueError("下载地址不符合预期")

output = folder / "chn_ppp_2018.tif"
temporary = folder / "chn_ppp_2018.part.tif"

if output.exists():
    raise FileExistsError(f"文件已经存在，不覆盖：{output}")

print("开始下载全国人口栅格，可能需要较长时间……", flush=True)
downloaded = 0
next_report = 100 * 1024 * 1024

with urllib.request.urlopen(url, timeout=180) as response:
    with temporary.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                print(f"已下载 {downloaded / 1024**2:.0f} MiB", flush=True)
                next_report += 100 * 1024 * 1024

with rasterio.open(temporary) as src:
    if src.crs is None or src.count != 1:
        raise RuntimeError("人口文件格式不符合预期")
    print("坐标系统：", src.crs)
    print("尺寸：", src.height, src.width)
    print("缺测标记：", src.nodata)

temporary.replace(output)
(folder / "chn_ppp_2018.source.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("下载完成：", output)
