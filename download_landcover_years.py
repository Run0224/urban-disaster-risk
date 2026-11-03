import hashlib
import json
import subprocess
from pathlib import Path

folder = Path.home() / "satellite_geo/fy4a/geographic_library/land_cover/CLCD_2018_2024"
plan = json.loads((folder / "download_plan.json").read_text())

def verified(path, item):
    if not path.exists() or path.stat().st_size != item["size_bytes"]:
        return False
    algorithm, expected = item["checksum"].split(":", 1)
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected

failed = []
for item in plan:
    year = item["year"]
    target = Path(item["file"])
    part = target.with_suffix(".tif.part")

    if target.exists():
        if verified(target, item):
            print(year, "已有文件校验通过", flush=True)
            continue
        print(year, "已有文件校验失败，保留待检查", flush=True)
        failed.append(year)
        continue

    print(year, "开始下载", flush=True)
    result = subprocess.run([
        "curl", "--fail", "--location",
        "--retry", "5", "--retry-delay", "5",
        "--connect-timeout", "20",
        "--speed-limit", "1024", "--speed-time", "120",
        "--continue-at", "-", "--progress-bar",
        "--output", str(part), item["url"],
    ])
    if result.returncode:
        print(year, "下载失败，临时文件保留供续传", flush=True)
        failed.append(year)
        continue

    print(year, "正在检查大小和校验值", flush=True)
    if not verified(part, item):
        print(year, "校验失败，临时文件保留待检查", flush=True)
        failed.append(year)
        continue

    part.replace(target)
    print(year, "下载完成，完整性校验通过", flush=True)

print("本轮结束，失败年份：", failed, flush=True)
print("尚未进行土地覆盖分类检查和卫星网格对齐。", flush=True)
