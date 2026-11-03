import json
import shlex
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path.home() / "satellite_geo/fy4a"
SOURCE = ROOT / (
    "training/regional_extraction_plan_v1/"
    "positive_extraction_plan.jsonl"
)
OUT = ROOT / "training/input_availability_v1"

with SOURCE.open(encoding="utf-8") as f:
    samples = [
        json.loads(line) for line in f if line.strip()
    ]
samples = [s for s in samples if s["target_kind"] == "single_day"]

requests_by_sample = {}
unique = {}

for sample in samples:
    issue = datetime.fromisoformat(sample["issue_time_utc"])
    entries = []
    for minutes in range(60, 0, -6):
        moment = issue - timedelta(minutes=minutes)
        stamp = moment.strftime("%Y%m%d%H%M")
        satellite = (
            "/mnt/data/zk/diffcast_3d/data/sat_6min/"
            + moment.strftime("%Y/%m/%d/")
            + stamp + "/" + stamp + ".npy"
        )
        radar = (
            "/home/ices/data/radar_gd/"
            + (moment + timedelta(hours=8)).strftime(
                "%Y/%m/%d/%Y%m%d%H%M.png"
            )
        )
        entry = {
            "time_utc": moment.isoformat(),
            "satellite": satellite,
            "radar": radar,
        }
        entries.append(entry)
        unique[satellite] = None
        unique[radar] = None
    requests_by_sample[sample["sample_id"]] = entries

code = (
    "import json,sys;from pathlib import Path;"
    "paths=json.load(sys.stdin);"
    "result={p:Path(p).is_file() for p in paths};"
    "print(json.dumps(result))"
)
print("检查单日候选：", len(samples), flush=True)
print("去重后文件路径数：", len(unique), flush=True)

response = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
     "ices@192.168.48.29",
     "/home/ices/miniconda3/bin/python -c " + shlex.quote(code)],
    input=json.dumps(list(unique)),
    capture_output=True, text=True, timeout=180,
)
if response.returncode:
    raise RuntimeError(response.stderr)

exists = json.loads(response.stdout)
if set(exists) != set(unique):
    raise ValueError("远程检查返回的路径不完整")

results = []
for sample in samples:
    frames = requests_by_sample[sample["sample_id"]]
    missing = []
    for frame in frames:
        for kind in ("satellite", "radar"):
            if not exists[frame[kind]]:
                missing.append({
                    "kind": kind,
                    "time_utc": frame["time_utc"],
                    "path": frame[kind],
                })
    results.append({
        **sample,
        "input_frames": frames,
        "missing_files": missing,
        "files_complete": not missing,
        "feature_status": "pending",
        "training_ready": False,
        "availability_note": (
            "仅检查当前文件存在，不代表数组完整或历史实时可用；"
            "插值卫星帧的原始观测截止时间仍需核实。"
        ),
    })

OUT.mkdir(parents=True, exist_ok=False)
target = OUT / "single_day_input_inventory.jsonl"
with target.open("x", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

complete = [r for r in results if r["files_complete"]]
print("文件齐全候选：", len(complete))
print("有文件缺失候选：", len(results)-len(complete))
print("齐全候选涉及过程数：",
      len({r["event_group_id"] for r in complete}))
print("缺失文件数（去重）：", sum(not v for v in exists.values()))
print("已保存：", target)
print("本次未下载数组、未训练，也未生成负样本。")
