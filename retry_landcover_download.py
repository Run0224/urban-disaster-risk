import hashlib
import json
import subprocess
import time
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
    expected = item["size_bytes"]

    if target.exists():
        if verified(target, item):
            print(year, "正式文件校验通过，跳过", flush=True)
        else:
            print(year, "正式文件异常，保留待检查", flush=True)
            failed.append(year)
        continue

    success = False
    for attempt in range(1, 21):
        size = part.stat().st_size if part.exists() else 0
        if size >= expected:
            if verified(part, item):
                part.replace(target)
                success = True
                print(year, "下载完成，官方校验通过", flush=True)
            else:
                print(year, "文件大小或校验异常，保留待检查", flush=True)
            break

        print(
            year, f"第{attempt}/20次连接",
            f"已有 {size/expected:.1%}",
            flush=True,
        )
        result = subprocess.run([
            "curl", "--fail", "--location", "--http1.1",
            "--connect-timeout", "20", "--max-time", "600",
            "--speed-limit", "1024", "--speed-time", "120",
            "--continue-at", "-", "--silent", "--show-error",
            "--output", str(part), item["url"],
        ])

        size = part.stat().st_size if part.exists() else 0
        print(year, f"当前进度 {size/expected:.1%}",
              "连接返回码:", result.returncode, flush=True)

        if size == expected:
            if verified(part, item):
                part.replace(target)
                success = True
                print(year, "下载完成，官方校验通过", flush=True)
            else:
                print(year, "校验失败，保留待检查", flush=True)
            break
        if size > expected or result.returncode == 33:
            print(year, "文件异常或服务器拒绝续传，停止该年", flush=True)
            break
        if attempt < 20:
            time.sleep(10)

    if not success:
        failed.append(year)

print("本轮结束，失败年份：", failed, flush=True)
