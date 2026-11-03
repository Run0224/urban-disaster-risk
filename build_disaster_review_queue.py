import json
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "satellite_geo/fy4a"
SOURCE = ROOT / (
    "geographic_library/disasters/"
    "Guangdong_Disaster_Records_2018_2024Jan"
)
OUT = ROOT / "training/multihazard_review_v1"

def read(name):
    with (SOURCE / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

records = read("local_observations.jsonl")
admin_rows = read("disaster_admin_references.jsonl")
admins = {r["record_id"]: r for r in admin_rows}
if len(admins) != len(admin_rows):
    raise ValueError("行政关联表record_id重复")

queue = []
for r in records:
    time = r.get("time") or {}
    location = r.get("location") or {}
    metrics = r.get("metrics") or []
    admin = admins.get(r["record_id"], {})
    impact = r["record_type"] == "confirmed_impact_report"

    issues = []
    if not (
        time.get("occurrence_start_utc")
        and time.get("occurrence_end_utc")
    ):
        issues.append("缺少明确的UTC发生区间，不能直接标注一小时预测目标")
    if time.get("precision") != "minute":
        issues.append("时间精度不足，需核实或采用更粗的预测时间窗")
    if not location.get("geometry"):
        issues.append("缺少受灾几何范围，报道行政区不等于全域受灾")
    if not impact:
        issues.append("当前为天气现象观测，不能直接作为实际受灾标签")
    if not metrics:
        issues.append("没有非空损失指标，不能作为零损失")
    else:
        issues.append("损失需核实统计范围、截至时间、单位、限定词和续报重复")

    # 优先审核已有损失、地点较明确的记录；不是风险或可信度评分。
    priority = (
        10 * bool(metrics)
        + 3 * bool(location.get("county"))
        + 2 * bool(location.get("city"))
        + 2 * (time.get("precision") == "minute")
        + int(impact)
    )

    queue.append({
        "record_id": r["record_id"],
        "event_group_id": r["event_group_id"],
        "hazard_type": r["hazard_type"],
        "record_type": r["record_type"],
        "review_priority": priority,
        "reported_time_information": time,
        "candidate_search_window": r.get("matching_time_window"),
        "reported_location": location,
        "administrative_references": admin.get("administrative_regions", []),
        "reported_metrics": metrics,
        "evidence": r.get("evidence"),
        "review_issues": issues,
        "review_decision": {
            "status": "pending",
            "verified_occurrence_start_utc": None,
            "verified_occurrence_end_utc": None,
            "verified_target_region": None,
            "supported_prediction_time_scale": None,
            "impact_label": None,
            "approved_metric_ids": [],
            "supporting_evidence": [],
            "reviewer": None,
        },
        "training_ready": False,
    })

ids = [r["record_id"] for r in queue]
if len(ids) != len(set(ids)):
    raise ValueError("记录ID重复")

queue.sort(key=lambda r: (-r["review_priority"], r["record_id"]))
OUT.mkdir(parents=True, exist_ok=False)
path = OUT / "review_queue.jsonl"
with path.open("x", encoding="utf-8") as f:
    for row in queue:
        f.write(json.dumps(
            row, ensure_ascii=False, allow_nan=False
        ) + "\n")

summary = {
    "records": len(queue),
    "hazard_types": len({r["hazard_type"] for r in queue}),
    "records_with_nonempty_metrics": sum(
        bool(r["reported_metrics"]) for r in queue
    ),
    "review_issue_counts": dict(Counter(
        issue for r in queue for issue in r["review_issues"]
    )),
    "training_ready": 0,
}
(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n优先审核的前5条：")
for row in queue[:5]:
    print(json.dumps({
        "record_id": row["record_id"],
        "hazard_type": row["hazard_type"],
        "time": row["reported_time_information"],
        "location": row["reported_location"],
        "metrics": row["reported_metrics"],
        "evidence": row["evidence"],
    }, ensure_ascii=False, indent=2))

print("\n已保存：", path)
print("原数据未修改；本次未生成确定的受灾或未受灾标签。")
