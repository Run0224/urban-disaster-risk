import base64
import hashlib
import io
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

from align_satellite_to_radar import resample

ROOT = Path.home() / "satellite_geo/fy4a"
TRAIN = ROOT / "training"
OUT = ROOT / "glm_predictions_v1"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "images").mkdir(exist_ok=True)
(OUT / "responses").mkdir(exist_ok=True)
(OUT / "predictions").mkdir(exist_ok=True)

def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

rows = read(TRAIN / "baseline_features_approx_v2/features_all.jsonl")
inventory = {
    r["sample_id"]: r
    for r in read(TRAIN / "combined_candidates_v1/input_inventory.jsonl")
}
with np.load(ROOT / "coordinates_approx.npz", allow_pickle=False) as d:
    slat, slon = d["latitude"].copy(), d["longitude"].copy()

lat = 26.0319 - np.arange(700)*0.01
lon = 108.505 + np.arange(900)*0.01
coverage = np.load(
    ROOT / "alignment_checks/radar_valid_mask.npy",
    allow_pickle=False,
).astype(bool)
extent = [lon[0]-.005, lon[-1]+.005, lat[-1]-.005, lat[0]+.005]

def make_image(sample, row, target):
    frames = sorted(sample["input_frames"], key=lambda f: f["time_utc"])
    # 展示过去一小时的早、中、末三帧，数值特征来自全部10帧。
    chosen = [frames[0], frames[len(frames)//2], frames[-1]]
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 8), constrained_layout=True
    )
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgray")
    box = row.get("region_extent", {}).get(
        "bounds_west_south_east_north"
    )
    for i, frame in enumerate(chosen):
        digest = hashlib.sha256(json.dumps(
            [frame["satellite"], frame["radar"]]
        ).encode()).hexdigest()[:20]
        with np.load(
            TRAIN / "raw_input_cache_v1" / f"{digest}.npz",
            allow_pickle=False,
        ) as d:
            sat = d["satellite"].copy()
            radar = d["radar"].astype(np.float32)

        bt, _, _, _ = resample(sat[5:6], slat, slon, lat, lon)
        radar[(radar < 15) | (radar >= 74)] = 0
        radar[~coverage] = np.nan

        a = axes[0, i].imshow(
            np.ma.masked_invalid(bt[0]),
            origin="upper", extent=extent,
            cmap="gray_r", vmin=180, vmax=320,
        )
        b = axes[1, i].imshow(
            np.ma.masked_invalid(radar),
            origin="upper", extent=extent,
            cmap=cmap, vmin=0, vmax=74,
        )
        # 使用相对时间，降低模型依靠历史日期记忆回答的风险。
        axes[0, i].set_title(["Earlier", "Middle", "Latest"][i])
        for ax in axes[:, i]:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            if box:
                west, south, east, north = box
                ax.plot(
                    [west, east, east, west, west],
                    [south, south, north, north, south],
                    color="red", linewidth=1.2,
                )
                # 留出周边天气背景。
                ax.set_xlim(max(extent[0], west-.3),
                            min(extent[1], east+.3))
                ax.set_ylim(max(extent[2], south-.3),
                            min(extent[3], north+.3))

    fig.colorbar(a, ax=list(axes[0]), label="C12 brightness temperature (K)",
                 shrink=.65)
    fig.colorbar(b, ax=list(axes[1]), label="Radar reflectivity (dBZ)",
                 shrink=.65)
    fig.suptitle("Past observations | top: satellite; bottom: radar")
    fig.savefig(target, dpi=110)
    plt.close(fig)

def extract_json(content):
    if "<think>" in content:
        if "</think>" not in content:
            raise ValueError("思考段未结束")
        content = content.split("</think>", 1)[1]
    content = content.replace("<|begin_of_box|>", "")
    content = content.replace("<|end_of_box|>", "")
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content)
    return json.loads(content.strip())

failed = []
completed = 0

for index, row in enumerate(rows, 1):
    sid = row["sample_id"]
    prediction_file = OUT / "predictions" / f"{sid}.json"
    print(f"[{index}/{len(rows)}] {sid}", flush=True)
    try:
        if prediction_file.exists():
            saved = json.loads(prediction_file.read_text(encoding="utf-8"))
            if saved.get("model") != "glm-4.6v":
                raise ValueError("已有结果模型不一致")
            completed += 1
            print("已有大模型结果，跳过", flush=True)
            continue

        sample = inventory[sid]
        image_file = OUT / "images" / f"{sid}.png"
        if not image_file.exists():
            make_image(sample, row, image_file)

        # 白名单构造输入：不包含标签、事件名称、记录编号和实际损失。
        context = {
            "target_hazard": row["hazard_type"],
            "target_region": row["region"],
            "forecast_horizon_hours": 24,
            "regional_features": row["features"],
            "region_extent": row.get("region_extent"),
        }

        prompt = """
你负责根据给定的观测和地理信息，预测目标区域未来24小时内，
目标灾种造成实际损害的风险，并在有依据时估计损失。

输入图上排是卫星C12亮温，越亮越冷，不是地面温度；
下排是雷达反射率，灰色为覆盖区外。
从左到右为过去一小时的早、中、末三帧。
数值特征来自全部10帧。图像提供周边背景，
目标区域数值以regional_features为准，不把其他地区回波算入目标区。
若有红色范围框，它是近似参考范围，不是受灾范围。

请使用你的气象和灾害知识作综合估计。
允许基于不完整证据给出主观概率，不要求先提供另一套灾害模型。
但概率是未经校准的模型估计，必须说明依据和关键不确定性。
不得将回波面积占比直接等同于受灾概率。
没有强回波不等于不存在所有灾种。
人口和土地覆盖表示暴露背景，不应单独推高天气危险性。
人口统计不是实际受灾人数；年度产品为回顾性数据。
地形统计不是坡度或排水能力。
如果确实无法支持任何概率判断，允许null并解释，
不要为了填数字而编造事实。

如果缺少灾害影响范围、资产价值或脆弱性依据，
受灾人口和经济损失返回null，不得把总人口当受灾人口。
不检索、回忆或猜测具体历史事件，不输出思考过程。

只返回以下JSON：
{
  "hazard_type": "目标灾种",
  "assessment_status": "model_estimate或insufficient_evidence",
  "impact_probability": null,
  "probability_calibrated": false,
  "expected_affected_people": null,
  "expected_direct_loss_cny": null,
  "evidence": ["本次输入支持的依据"],
  "uncertainties": ["不确定性"],
  "loss_basis": "损失估计依据，或不能估计的原因"
}
非null概率必须在0到1之间；人数、金额必须非负。
不得声称估计经过训练校准或具有已验证准确率。

本次输入：
""" + json.dumps(context, ensure_ascii=False, allow_nan=False)

        (OUT / "responses" / f"{sid}_prompt.txt").write_text(
            prompt, encoding="utf-8"
        )
        encoded = base64.b64encode(image_file.read_bytes()).decode()
        response = requests.post(
            "http://127.0.0.1:8002/v1/chat/completions",
            json={
                "model": "glm-4.6v",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + encoded
                        }},
                    ],
                }],
                "temperature": 0.1,
                "max_tokens": 2200,
            },
            timeout=600,
        )
        response.raise_for_status()
        raw = response.json()
        (OUT / "responses" / f"{sid}.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("输出未正常结束，完整响应已保存")

        result = extract_json(choice["message"].get("content") or "")
        required = {
            "hazard_type", "assessment_status", "impact_probability",
            "probability_calibrated", "expected_affected_people",
            "expected_direct_loss_cny", "evidence", "uncertainties",
            "loss_basis",
        }
        if not isinstance(result, dict) or not required.issubset(result):
            raise ValueError("结果字段不完整")
        if result["hazard_type"] != row["hazard_type"]:
            raise ValueError("目标灾种不一致")
        if result["probability_calibrated"] is not False:
            raise ValueError("模型错误声称概率已校准")
        if result["assessment_status"] not in (
            "model_estimate", "insufficient_evidence"
        ):
            raise ValueError("状态异常")
        for key in (
            "impact_probability", "expected_affected_people",
            "expected_direct_loss_cny",
        ):
            value = result[key]
            if value is not None:
                if type(value) not in (int, float) or not np.isfinite(value):
                    raise ValueError(f"{key}不是有限数值")
                if value < 0 or (key == "impact_probability" and value > 1):
                    raise ValueError(f"{key}超出范围")
        if result["assessment_status"] == "insufficient_evidence" and any(
            result[k] is not None for k in (
                "impact_probability", "expected_affected_people",
                "expected_direct_loss_cny",
            )
        ):
            raise ValueError("证据不足状态与数值输出矛盾")

        # 预测已完成后才保存对照标签，标签从未送入请求。
        saved = {
            "sample_id": sid,
            "model": "glm-4.6v",
            "prediction": result,
            "reference_candidate_label": row["label"],
            "reference_label_quality": row["label_quality"],
        }
        prediction_file.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        completed += 1
        print("GLM受灾概率估计：", result["impact_probability"], flush=True)

    except Exception as error:
        failed.append({"sample_id": sid, "error": str(error)})
        print("本条未完成：", str(error)[:300], flush=True)

    (OUT / "run_report.json").write_text(json.dumps({
        "requested": len(rows),
        "completed": completed,
        "failed": failed,
        "model": "glm-4.6v",
        "probability_calibrated": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n本轮结束：完成", completed, "失败", len(failed))
print("结果目录：", OUT)
