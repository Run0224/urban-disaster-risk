import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path.home() / "satellite_geo/fy4a/training"
SOURCE = ROOT / "baseline_features_approx_v2/features_all.jsonl"
OUT = ROOT / "risk_baseline_v1"
OUT.mkdir(parents=True, exist_ok=True)

with SOURCE.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
if len({r["sample_id"] for r in rows}) != len(rows):
    raise ValueError("样本ID重复")

# 将相关样本连接成不可跨训练/测试的分组。
parent = list(range(len(rows)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(a, b):
    a, b = find(a), find(b)
    if a != b:
        parent[b] = a

seen = {}
for i, row in enumerate(rows):
    tokens = []
    if row.get("event_group_id"):
        tokens.append("event:" + row["event_group_id"])
    tokens += [
        "event:" + g
        for g in row.get("matched_positive_event_groups", [])
    ]
    tokens += ["cache:" + k for k in row["input_cache_keys"]]
    tokens.append("issue:" + row["issue_time_utc"])
    tokens.append("sample:" + row["sample_id"])
    tokens += [
        "sample:" + sid
        for sid in row.get("matched_positive_sample_ids", [])
    ]
    for token in tokens:
        if token in seen:
            union(i, seen[token])
        else:
            seen[token] = i

groups = np.array([find(i) for i in range(len(rows))])
names = sorted({k for row in rows for k in row["features"]})
X = np.array([
    [np.nan if row["features"].get(k) is None
     else float(row["features"][k]) for k in names]
    for row in rows
])
X[~np.isfinite(X)] = np.nan
y = np.array([int(r["label"]) for r in rows])
if not np.isin(y, [0, 1]).all():
    raise ValueError("标签必须为0或1")

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

def fit(x, target):
    # 缺失值处理和标准化只依据训练部分。
    columns = np.flatnonzero(np.isfinite(x).any(axis=0))
    if not len(columns):
        raise ValueError("没有可用特征")
    a = x[:, columns]
    medians = np.array([
        np.median(col[np.isfinite(col)]) for col in a.T
    ])
    a = np.where(np.isfinite(a), a, medians)
    center = a.mean(axis=0)
    scale = a.std(axis=0)
    scale[scale < 1e-8] = 1
    a = (a-center)/scale
    weights = np.zeros(a.shape[1])
    intercept = float(np.log(
        (target.sum()+0.5)/(len(target)-target.sum()+0.5)
    ))
    # 固定L2正则，不使用测试组调参。
    for _ in range(2000):
        error = sigmoid(a @ weights + intercept)-target
        weights -= 0.05*(a.T @ error/len(target) + 0.1*weights)
        intercept -= 0.05*error.mean()
    return columns, medians, center, scale, weights, intercept

def predict(model, x):
    columns, medians, center, scale, weights, intercept = model
    a = x[:, columns]
    a = np.where(np.isfinite(a), a, medians)
    return sigmoid(((a-center)/scale) @ weights + intercept)

def metrics(target, score):
    positive, negative = score[target == 1], score[target == 0]
    auc = (
        float(np.mean(
            (positive[:, None] > negative[None, :])
            + 0.5*(positive[:, None] == negative[None, :])
        )) if len(positive) and len(negative) else None
    )
    return {
        "tested_samples": int(len(target)),
        "positive": int(target.sum()),
        "weak_negative": int((target == 0).sum()),
        "roc_auc_against_candidate_labels": auc,
        "brier_against_candidate_labels":
            float(np.mean((score-target)**2)),
        "accuracy_at_0_5":
            float(np.mean((score >= 0.5) == target)),
    }

report = {
    "candidate_samples": len(rows),
    "connected_groups": len(set(groups.tolist())),
    "feature_count": len(names),
    "trained": {},
    "skipped": {},
    "limitations": [
        "负例为未报道弱负例，不是核实无灾。",
        "分数基于抽样数据，未校准为自然发生概率。",
        "年地理数据为回顾性产品；插值帧历史可用性未完全核实。",
        "区域范围和卫星定位近似；不同范围提取方法可能带来偏差。",
        "这是固定正则逻辑回归基线，不是GLM微调，不预测损失。",
    ],
}
predictions = []

print("候选数：", len(rows), "关联分组数：", report["connected_groups"],
      flush=True)

for hazard in sorted({r["hazard_type"] for r in rows}):
    idx = np.array([
        i for i, r in enumerate(rows) if r["hazard_type"] == hazard
    ])
    target, g = y[idx], groups[idx]
    positive_groups = set(g[target == 1].tolist())
    negative_groups = set(g[target == 0].tolist())
    if len(positive_groups) < 3 or len(negative_groups) < 3:
        report["skipped"][hazard] = {
            "positive_samples": int(target.sum()),
            "weak_negative_samples": int((target == 0).sum()),
            "positive_groups": len(positive_groups),
            "negative_groups": len(negative_groups),
            "reason": "本轮要求正负类分别至少涉及3个关联分组",
        }
        print(hazard, "暂不训练：关联分组不足", flush=True)
        continue

    oof = np.full(len(idx), np.nan)
    tested_groups = 0
    for group in sorted(set(g.tolist())):
        train = g != group
        test = ~train
        if len(set(target[train].tolist())) < 2:
            continue
        model = fit(X[idx[train]], target[train])
        oof[test] = predict(model, X[idx[test]])
        tested_groups += 1

    valid = np.isfinite(oof)
    if not valid.any() or len(set(target[valid].tolist())) < 2:
        report["skipped"][hazard] = {
            "reason": "留组测试未覆盖正负两类，不能报告评估"
        }
        continue

    result = metrics(target[valid], oof[valid])
    result["tested_groups"] = tested_groups
    # 用每个测试组训练部分的正例比例建立常数分数对照。
    reference = np.full(len(idx), np.nan)
    for group in set(g.tolist()):
        train, test = g != group, g == group
        if len(set(target[train].tolist())) == 2:
            reference[test] = target[train].mean()
    result["constant_baseline_brier"] = float(np.mean(
        (reference[valid]-target[valid])**2
    ))
    report["trained"][hazard] = result

    for local in np.flatnonzero(valid):
        row = rows[idx[local]]
        predictions.append({
            "sample_id": row["sample_id"],
            "hazard_type": hazard,
            "candidate_label": int(target[local]),
            "label_quality": row["label_quality"],
            "held_out_group": int(g[local]),
            "experimental_score": float(oof[local]),
            "probability_calibrated": False,
        })

    full_model = fit(X[idx], target)
    columns, medians, center, scale, weights, intercept = full_model
    np.savez_compressed(
        OUT / f"{hazard}_model.npz",
        feature_names=np.array(names),
        columns=columns,
        medians=medians,
        center=center,
        scale=scale,
        weights=weights,
        intercept=np.array(intercept),
        probability_calibrated=np.array(False),
    )
    print(hazard, json.dumps(result, ensure_ascii=False), flush=True)

with (OUT / "held_out_predictions.jsonl").open("w", encoding="utf-8") as f:
    for row in predictions:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

(OUT / "training_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)
(OUT / "sample_groups.json").write_text(json.dumps({
    r["sample_id"]: int(groups[i]) for i, r in enumerate(rows)
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n训练完成")
print("已训练灾种：", list(report["trained"]))
print("暂未训练灾种：", list(report["skipped"]))
print("留组预测条数：", len(predictions))
print("结果目录：", OUT)
print("模型分数尚未校准，不用于正式预警或损失估计。")
