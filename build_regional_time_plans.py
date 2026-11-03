import json
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path.home() / "satellite_geo/fy4a/training"
SOURCE = ROOT / "regional_labels_v1/regional_labels.jsonl"
OUT = ROOT / "regional_time_plans_v1"
BJT = timezone(timedelta(hours=8))

with SOURCE.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

plans = []
for row in records:
    t = row.get("reported_time") or {}
    search = row.get("candidate_search_window") or {}
    start = t.get("start_date_local")
    end = t.get("end_date_local")
    basis = t.get("time_basis") or ""

    # 只把明确标注为发生日期的日级记录用于日级候选。
    explicit = (
        t.get("precision") == "day"
        and "occurrence" in basis
        and bool(start)
    )
    status = "explicit_occurrence_date_candidate" if explicit else "needs_time_review"

    plan = {
        "label_id": row["label_id"],
        "record_id": row["record_id"],
        "event_group_id": row["event_group_id"],
        "hazard_type": row["hazard_type"],
        "region": row["region"],
        "regional_impact_label": row["regional_impact_label"],
        "time_status": status,
        "reported_time": t,
        "candidate_search_window": search,
        "issue_time_utc": None,
        "target_start_utc": None,
        "target_end_utc_exclusive": None,
        "training_ready": False,
    }

    if explicit:
        try:
            first = date.fromisoformat(start)
            last = date.fromisoformat(end) if end else first
            if last < first:
                raise ValueError("结束日期早于开始日期")
            begin = datetime.combine(first, time.min, tzinfo=BJT)
            finish = datetime.combine(
                last + timedelta(days=1), time.min, tzinfo=BJT
            )
            plan.update({
                "issue_time_utc": begin.astimezone(timezone.utc).isoformat(),
                "target_start_utc": begin.astimezone(timezone.utc).isoformat(),
                "target_end_utc_exclusive":
                    finish.astimezone(timezone.utc).isoformat(),
                "input_window_start_utc":
                    (begin-timedelta(hours=1)).astimezone(timezone.utc).isoformat(),
                "input_window_end_utc_exclusive":
                    begin.astimezone(timezone.utc).isoformat(),
                "target_semantics":
                    "报道支持的日期或日期段内，区域有受灾报道；不是逐帧受灾标签",
                "review_note":
                    "缺少结束日期时只建立起始日候选，不声称事件当天结束；"
                    "多日目标不能拆成每天均受灾。仍需检查记录及数据可用性。",
            })
        except (ValueError, TypeError) as error:
            plan["time_status"] = "invalid_date_needs_review"
            plan["error"] = str(error)

    plans.append(plan)

OUT.mkdir(parents=True, exist_ok=False)
with (OUT / "sample_time_plans.jsonl").open("x", encoding="utf-8") as f:
    for plan in plans:
        f.write(json.dumps(plan, ensure_ascii=False, allow_nan=False) + "\n")

counts = Counter(p["time_status"] for p in plans)
print("区域记录总数：", len(plans))
print("时间计划分类：", dict(counts))
print("明确发生日期候选中的受灾正记录：", sum(
    p["time_status"] == "explicit_occurrence_date_candidate"
    and p["regional_impact_label"] == 1
    for p in plans
))
print("已保存：", OUT / "sample_time_plans.jsonl")
print("未将推测检索窗口自动改成发生区间；未生成负样本。")
