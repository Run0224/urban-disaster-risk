import json
import urllib.request
from pathlib import Path
import rasterio
from rasterio.windows import Window

folder = Path.home() / "satellite_geo/fy4a/geographic_library/land_cover"
plan = json.loads((folder / "download_plan_2020.json").read_text())
report = {"completed": [], "failed": []}
total = len(plan["tiles"])

for index, tile in enumerate(plan["tiles"], 1):
    target = Path(tile["file"])
    part = target.with_suffix(".tif.part")
    expected = tile["size_bytes"]

    try:
        ready = target.exists() and target.stat().st_size == expected

        if not ready:
            offset = part.stat().st_size if part.exists() else 0
            if offset > expected:
                raise RuntimeError("临时文件大于预期")

            if offset < expected:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                request = urllib.request.Request(tile["url"], headers=headers)

                with urllib.request.urlopen(request, timeout=180) as response:
                    if offset and response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise RuntimeError("续传起点不符合预期")
                        mode = "ab"
                    elif response.status == 200:
                        mode = "wb"
                    else:
                        raise RuntimeError(f"HTTP状态异常：{response.status}")

                    print(f"[{index}/{total}] 下载 {target.name}", flush=True)
                    with part.open(mode) as file:
                        while chunk := response.read(1024 * 1024):
                            file.write(chunk)

            if part.stat().st_size != expected:
                raise RuntimeError("下载不完整，可重新运行续传")

        check_path = target if ready else part
        with rasterio.open(check_path) as src:
            if src.crs is None or src.count != 1:
                raise RuntimeError("文件格式异常")
            src.read(1, window=Window(0, 0, 64, 64))

        if not ready:
            part.replace(target)

        report["completed"].append(str(target))
        print(f"[{index}/{total}] 完成或已存在", flush=True)

    except Exception as error:
        report["failed"].append({"file": str(target), "error": str(error)})
        print(f"[{index}/{total}] 失败：{error}", flush=True)

    (folder / "download_report_2020.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

print("成功：", len(report["completed"]))
print("失败：", len(report["failed"]))
