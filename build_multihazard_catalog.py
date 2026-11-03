import json
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "satellite_geo/fy4a"
SOURCE = ROOT / (
    "geographic_library/disasters/"
    "Guangdong_Disaster_Records_2018_2024Jan/local_observations.jsonl"
)
OUT = ROOT / "training/multihazard_labels_v1"

with SOURCE.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

ids = [r["record_id"] for r in records]
if len(ids) != len(set(ids)):
    raise ValueError("record_id重复，停止生成")

catalog = []
metrics = []
hazard_counts = Counter()
impact_counts = Counter()

for r in records:
    hazard = r["hazard_type"]
    impact = r["record_type"] == "confirmed_impact_report"
    hazard_counts[hazard] += 1
    impact_counts[hazard] += int(impact)

    catalog.append({
        "record_id": r["record_id"],
        "event_group_id": r["event_group_id"],
        "hazard_type": hazard,
        "record_type": r["record_type"],
        "reported_impact": True if impact else None,
        "negative_label_available": False,
        "time": r.get("time"),
        "location": r.get("location"),
        "evidence": r.get("evidence"),
        "matching_time_window": r.get("matching_time_window"),
        "forecast_target_ready": False,
        "review_required": [
            "确认发生时间与预测窗口",
            "确认区域与统计范围",
            "检查同过程重复报道和续报",
        ],
    })

    for index, metric in enumerate(r.get("metrics") or []):
        metrics.append({
            "metric_id": f"{r['record_id']}_metric_{index}",
            "record_id": r["record_id"],
            "event_group_id": r["event_group_id"],
            "hazard_type": hazard,
            "record_type": r["record_type"],
            "metric_original": metric,
            "location_original": r.get("location"),
            "time_original": r.get("time"),
            "evidence": r.get("evidence"),
            "training_ready": False,
            "note": (
                "保留单位、范围、截至时间及限定词；"
                "未去重，不能直接相加或分摊到像素。"
            ),
        })

# 新建版本目录，避免覆盖已有结果。
OUT.mkdir(parents=True, exist_ok=False)

for name, rows in (
    ("hazard_records.jsonl", catalog),
    ("impact_metrics.jsonl", metrics),
):
    with (OUT / name).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(
                row, ensure_ascii=False, allow_nan=False
            ) + "\n")

summary = {
    "source": str(SOURCE),
    "record_count": len(catalog),
    "hazard_type_count": len(hazard_counts),
    "nonempty_metric_count": len(metrics),
    "records_with_nonempty_metrics":
        len({m["record_id"] for m in metrics}),
    "record_count_by_hazard": dict(hazard_counts),
    "impact_report_count_by_hazard": dict(impact_counts),
    "metric_count_by_name": dict(Counter(
        m["metric_original"].get("name", "unknown") for m in metrics
    )),
    "note": "标签候选库，尚不是可直接训练的概率或损失数据集。",
}
with (OUT / "summary.json").open("x", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n已保存：", OUT)
print("原始记录未修改；没有把未知标记为未受灾或零损失。")
