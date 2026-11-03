import json
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "satellite_geo/fy4a"
FOLDER = ROOT / (
    "geographic_library/disasters/"
    "Guangdong_Disaster_Records_2018_2024Jan"
)
OUT = ROOT / "training/regional_labels_v1"

def read(name):
    with (FOLDER / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

records = read("local_observations.jsonl")
admin_rows = read("disaster_admin_references.jsonl")
admins = {r["record_id"]: r for r in admin_rows}
if len(admins) != len(admin_rows):
    raise ValueError("行政关联记录ID重复")

labels = []
unresolved = []

for record in records:
    rid = record["record_id"]
    impact = record["record_type"] == "confirmed_impact_report"
    regions = admins.get(rid, {}).get("administrative_regions") or []

    if not regions:
        unresolved.append({
            "record_id": rid,
            "location": record.get("location"),
            "reason": "没有可用行政区域关联，保留原地点待处理",
        })
        continue

    seen = set()
    for region in regions:
        province = region.get("province")
        city = region.get("city")
        county = region.get("county")
        key = (province, city, county)
        if key in seen:
            continue
        seen.add(key)

        if not any(key):
            continue

        level = "county" if county else "city" if city else "province"
        labels.append({
            "label_id": f"{rid}_region_{len(seen)}",
            "record_id": rid,
            "event_group_id": record["event_group_id"],
            "hazard_type": record["hazard_type"],
            "record_type": record["record_type"],
            "region": {
                "province": province,
                "city": city,
                "county": county,
                "level": level,
            },
            "regional_impact_label": 1 if impact else None,
            "hazard_observed": True,
            "label_definition": (
                "按报道提及地点建立区域标签；区域内有受灾报道，"
                "不代表全区域、所有像素或人口均受灾。"
                if impact else
                "区域内有天气现象观测；是否造成损害未知。"
            ),
            "reported_time": record.get("time"),
            "candidate_search_window": record.get("matching_time_window"),
            "forecast_window_label": None,
            "time_alignment_status": "pending",
            "evidence": record.get("evidence"),
            "loss_reference": {
                "record_id": rid,
                "has_nonempty_metrics": bool(record.get("metrics")),
                "note": "按原记录统计范围另行关联，不复制或分摊损失。",
            },
            "split_group": record["event_group_id"],
            "training_ready": False,
        })

OUT.mkdir(parents=True, exist_ok=False)
for name, rows in (
    ("regional_labels.jsonl", labels),
    ("unresolved_locations.jsonl", unresolved),
):
    with (OUT / name).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(
                row, ensure_ascii=False, allow_nan=False
            ) + "\n")

summary = {
    "source_records": len(records),
    "regional_label_rows": len(labels),
    "impact_positive_rows": sum(
        r["regional_impact_label"] == 1 for r in labels
    ),
    "impact_unknown_rows": sum(
        r["regional_impact_label"] is None for r in labels
    ),
    "unresolved_records": len(unresolved),
    "regional_rows_by_hazard": dict(Counter(
        r["hazard_type"] for r in labels
    )),
    "note": "区域行可能来自同一事件，不是独立灾害样本数；尚未确定预测窗口。",
}
(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n已保存：", OUT)
