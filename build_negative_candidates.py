import calendar
import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path.home() / "satellite_geo/fy4a"
SOURCE = ROOT / (
    "training/input_availability_v1/single_day_input_inventory.jsonl"
)
DISASTERS = ROOT / (
    "geographic_library/disasters/"
    "Guangdong_Disaster_Records_2018_2024Jan/local_observations.jsonl"
)
OUT = ROOT / "training/negative_candidates_v1"
BJT = timezone(timedelta(hours=8))
BUFFER = 7

def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

positives = [r for r in read(SOURCE) if r["files_complete"]]
records = read(DISASTERS)

# 保守地排除所有已知灾种的日期窗口，不只排除目标灾种。
blocked = set()
unknown_time = []
for r in records:
    found = False
    for field in ("time", "matching_time_window"):
        t = r.get(field) or {}
        start = t.get("start_date_local")
        end = t.get("end_date_local") or start
        if not start:
            continue
        try:
            first, last = date.fromisoformat(start), date.fromisoformat(end)
            if last < first:
                continue
            day = first - timedelta(days=BUFFER)
            finish = last + timedelta(days=BUFFER)
            while day <= finish:
                blocked.add(day)
                day += timedelta(days=1)
            found = True
        except (ValueError, TypeError):
            continue
    if not found:
        unknown_time.append(r["record_id"])

groups = {}
for positive in positives:
    start = datetime.fromisoformat(
        positive["target_start_utc"]
    ).astimezone(BJT).date()
    candidates = [
        date(start.year, start.month, d)
        for d in range(
            1, calendar.monthrange(start.year, start.month)[1] + 1
        )
        if date(start.year, start.month, d) not in blocked
    ]
    # 确定性抽样，重复运行时选择一致。
    candidates.sort(key=lambda d: hashlib.sha256(
        (positive["sample_id"] + d.isoformat()).encode()
    ).hexdigest())

    for day in candidates[:3]:
        region = positive["region"]
        key = (
            positive["hazard_type"],
            region.get("province"), region.get("city"),
            region.get("county"), day.isoformat(),
        )
        if key not in groups:
            issue = datetime.combine(day, time.min, tzinfo=BJT)
            utc = issue.astimezone(timezone.utc)
            groups[key] = {
                "hazard_type": positive["hazard_type"],
                "region": region,
                "issue_time_utc": utc.isoformat(),
                "input_window_start_utc":
                    (utc-timedelta(hours=1)).isoformat(),
                "input_window_end_utc_exclusive": utc.isoformat(),
                "target_start_utc": utc.isoformat(),
                "target_end_utc_exclusive":
                    (utc+timedelta(days=1)).isoformat(),
                "forecast_horizon_hours": 24,
                "proposed_label": 0,
                "label_quality": "weak_negative_unreported",
                "confirmed_no_disaster": False,
                "loss_label": None,
                "matched_positive_sample_ids": [],
                "matched_positive_event_groups": [],
                "files_complete": None,
                "training_ready": False,
            }
        row = groups[key]
        for field, value in (
            ("matched_positive_sample_ids", positive["sample_id"]),
            ("matched_positive_event_groups", positive["event_group_id"]),
        ):
            if value not in row[field]:
                row[field].append(value)

rows = list(groups.values())
for i, row in enumerate(rows):
    row["sample_id"] = f"weak_negative_{i:04d}"

OUT.mkdir(parents=True, exist_ok=False)
with (OUT / "negative_candidates.jsonl").open("x", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

summary = {
    "positive_candidates": len(positives),
    "weak_negative_candidates": len(rows),
    "positives_with_at_least_one_negative": len({
        sid for row in rows for sid in row["matched_positive_sample_ids"]
    }),
    "records_without_usable_date": unknown_time,
    "exclusion_buffer_days": BUFFER,
    "note": (
        "7天缓冲是实验采样规则，不是无灾保证；"
        "缺失时间、漏报及持续灾害可能污染弱负例。"
        "弱负例不自动获得零损失标签；尚未划分训练测试集。"
    ),
}
(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("已保存：", OUT / "negative_candidates.jsonl")
