import argparse
import hashlib
import io
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path.home() / "satellite_geo/fy4a"
SOURCE = ROOT / "training/combined_candidates_v1/input_inventory.jsonl"
OUT = ROOT / "training/raw_input_cache_v1"
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=3)
args = parser.parse_args()

pairs = {}
with SOURCE.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        sample = json.loads(line)
        if sample["files_complete"]:
            for frame in sample["input_frames"]:
                pairs[(frame["satellite"], frame["radar"])] = frame

entries = list(pairs.values())
if args.limit:
    entries = entries[:args.limit]

def validate(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as d:
        sat = d["satellite"]
        radar = d["radar"]
        if sat.shape != (8, 518, 460):
            raise ValueError(f"卫星尺寸异常：{sat.shape}")
        if radar.shape != (700, 900):
            raise ValueError(f"雷达尺寸异常：{radar.shape}")
        if not np.issubdtype(sat.dtype, np.number):
            raise ValueError("卫星类型异常")
        return int(np.sum(~np.isfinite(sat)))

failed = []
for index, frame in enumerate(entries, 1):
    digest = hashlib.sha256(
        json.dumps([frame["satellite"], frame["radar"]]).encode()
    ).hexdigest()[:20]
    target = OUT / f"{digest}.npz"
    metadata = OUT / f"{digest}.json"
    print(f"[{index}/{len(entries)}] {frame['time_utc']}", flush=True)

    try:
        if target.exists():
            invalid = validate(target.read_bytes())
            print("已有缓存，检查通过", flush=True)
        else:
            code = (
                "import io,sys,numpy as np;"
                "from pathlib import Path;"
                f"a=np.load({frame['satellite']!r},allow_pickle=False);"
                f"r=np.frombuffer(Path({frame['radar']!r}).read_bytes(),dtype=np.uint8);"
                "b=io.BytesIO();"
                "np.savez_compressed(b,satellite=a,radar_png=r);"
                "sys.stdout.buffer.write(b.getvalue())"
            )
            response = subprocess.run(
                ["ssh", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=10",
                 "ices@192.168.48.29",
                 "/home/ices/miniconda3/bin/python -c "
                 + shlex.quote(code)],
                capture_output=True, timeout=180,
            )
            if response.returncode:
                raise RuntimeError(
                    response.stderr.decode(errors="replace")
                )
            from PIL import Image
            with np.load(io.BytesIO(response.stdout), allow_pickle=False) as d:
                satellite = d["satellite"].copy()
                png = d["radar_png"].tobytes()
            with Image.open(io.BytesIO(png)) as im:
                if im.mode != "L":
                    raise ValueError(f"雷达模式异常：{im.mode}")
                radar = np.array(im)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, satellite=satellite, radar=radar)
            blob = buffer.getvalue()
            invalid = validate(blob)
            temporary = target.with_suffix(".npz.part")
            temporary.write_bytes(blob)
            temporary.replace(target)
            print("读取、解码、保存完成", flush=True)

        metadata.write_text(json.dumps({
            **frame,
            "cache_file": str(target),
            "satellite_nonfinite_values": invalid,
            "note": "原始数值缓存，尚未清洗或生成特征。",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    except Exception as error:
        failed.append({**frame, "error": str(error)})
        print("失败：", str(error)[:400], flush=True)

report = OUT / "last_run_report.json"
report.write_text(json.dumps({
    "requested_pairs": len(entries),
    "failed_pairs": len(failed),
    "failures": failed,
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("本轮结束，失败配对：", len(failed))
print("报告：", report)
if failed:
    raise SystemExit(1)
