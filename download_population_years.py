import json
import subprocess
from pathlib import Path

folder = Path.home() / "satellite_geo/fy4a/geographic_library/population/R2025A_v1"
plan = json.loads((folder / "download_plan.json").read_text())
failed = []

for item in plan:
    year = item["year"]
    target = Path(item["file"])
    expected = item.get("size_bytes")
    part = target.with_suffix(".tif.part")

    if not expected or item.get("http_status") != 200:
        failed.append(year)
        print(year, "下载计划异常，跳过", flush=True)
        continue

    if target.exists():
        if target.stat().st_size == expected:
            print(year, "文件已存在，大小匹配", flush=True)
            continue
        failed.append(year)
        print(year, "已有文件大小异常，保留待检查", flush=True)
        continue

    print(year, "开始下载", flush=True)
    result = subprocess.run([
        "curl", "--fail", "--location",
        "--retry", "5", "--retry-delay", "5",
        "--connect-timeout", "20",
        "--silent", "--show-error",
        "--continue-at", "-",
        "--output", str(part),
        item["url"],
    ])

    if result.returncode != 0 or not part.exists():
        failed.append(year)
        print(year, "下载失败，可重新运行续传", flush=True)
        continue

    if part.stat().st_size != expected:
        failed.append(year)
        print(year, "大小不完整，保留续传文件", flush=True)
        continue

    part.replace(target)
    print(year, "下载完成，大小检查通过", flush=True)

print("本轮结束，失败年份：", failed, flush=True)
print("尚未进行栅格内容检查和卫星网格对齐。", flush=True)
