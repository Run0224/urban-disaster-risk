import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

from load_radar_grid_sample import load_sample

ROOT = Path.home() / "satellite_geo/fy4a"
OUT = ROOT / "glm_multihazard_rehearsal"
OUT.mkdir(parents=True, exist_ok=True)

manifest = ROOT / "training/satellite_disaster_weak_samples_v6_geography.jsonl"
with manifest.open(encoding="utf-8") as f:
    item = json.loads(next(line for line in f if line.strip()))

print("读取联合样本……", flush=True)
data = load_sample(item)
coverage = data["radar_coverage_mask"]
radar = data["radar_dbz"]
lat, lon = data["latitude"], data["longitude"]

extent = [
    lon[0] - 0.005, lon[-1] + 0.005,
    lat[-1] - 0.005, lat[0] + 0.005,
]
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

satellite = np.ma.masked_invalid(data["satellite_bt"][5])
im = axes[0].imshow(
    satellite, origin="upper", extent=extent,
    cmap="gray_r", vmin=180, vmax=320,
)
axes[0].set_title("Satellite C12 brightness temperature")
fig.colorbar(im, ax=axes[0], label="K", shrink=0.75)

radar_cmap = plt.get_cmap("turbo").copy()
radar_cmap.set_bad("lightgray")
im = axes[1].imshow(
    np.ma.masked_invalid(radar), origin="upper", extent=extent,
    cmap=radar_cmap, vmin=0, vmax=74,
)
axes[1].set_title("Radar: gray = outside coverage")
fig.colorbar(im, ax=axes[1], label="dBZ", shrink=0.75)

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))

fig.suptitle("UTC " + item["time_utc"] + " | approximate satellite geolocation")
image_path = OUT / f"{item['sample_id']}_input.png"
fig.savefig(image_path, dpi=140)
plt.close(fig)

pop_valid = coverage & data["population_valid"]
land_valid = coverage & data["land_cover_valid"]
stats = {
    "time_utc": item["time_utc"],
    "target": "未来一小时雷达覆盖区域的强降水风险",
    "coordinate_status": "卫星定位基于README近似坐标，未独立验证",
    "radar_coverage_cells": int(coverage.sum()),
    "radar_cells_ge_35_dbz": int(np.sum(coverage & (radar >= 35))),
    "radar_cells_ge_50_dbz": int(np.sum(coverage & (radar >= 50))),
    "population_cells_with_valid_data": int(pop_valid.sum()),
    "population_sum_over_valid_source_pixels":
        float(data["population"][pop_valid].sum()),
    "population_note": "仅有效源像素的人口估计之和，不能视为完整覆盖人口",
    "land_cover_mean_proportions_over_valid_covered_cells": {
        str(name): float(data["land_cover_proportions"][i, land_valid].mean())
        if land_valid.any() else None
        for i, name in enumerate(data["land_cover_class_names"])
    },
    "annual_covariate_note": "同年回顾性产品，未验证历史预测时点是否可用",
}
prompt = """
你正在进行多灾种受灾风险和损失预测的流程预演。
依据附图和统计，评估观测时刻之后未来一小时、雷达覆盖范围内的风险。
不得查询或回忆该历史日期的实际灾情。不能把已有报道当作输入证据。

输入说明：
左图为卫星C12亮温，越亮越冷；右图为雷达反射率，
灰色为覆盖区外。输入只有一帧，没有天气演变序列。
卫星定位为近似坐标。人口和土地覆盖是同年回顾性产品。
人口统计是大范围有效像素之和，不是受灾人口，也不是完整区域人口。
土地覆盖比例是有效覆盖格子的平均，不能代替局部风险区统计。
≥35或≥50 dBZ的阈值统计不能直接决定是否受灾。
没有强回波或缺少观测，不等于所有灾种风险低。

逐项评估以下18类，不得漏项：
cold_freeze, debris_flow, drought, flash_flood, flood, hail,
heatwave, heavy_rain, landslide, river_flood, snow_sleet,
storm_surge, tornado, tropical_cyclone, tropical_depression,
urban_flood, waterspout, wildfire。

要求：
1. 区分天气现象发生与造成实际损害；此处关注造成损害的风险。
2. 没有充分依据时，状态为insufficient_data，数值为null，
   不能把未知填0，也不能统一标为低风险。
3. 如给出受灾概率估计，必须标记为未经校准的模型主观估计。
4. 受影响人口、直接经济损失缺少灾害范围、资产和脆弱性依据时，
   返回null，禁止直接拿总人口当受灾人口或编造金额。
5. 干旱等慢发灾害不适合仅靠单帧和一小时窗口评估，应明确指出。
6. 地域只用经纬度描述，不虚构市县名称。
7. 不输出思考过程，只输出一个JSON对象。

JSON结构：
{
  "mode": "pipeline_rehearsal",
  "forecast_horizon_minutes": 60,
  "probability_calibrated": false,
  "hazards": [
    {
      "hazard_type": "上述英文灾种之一",
      "assessment_status": "experimental_estimate或insufficient_data",
      "impact_probability_estimate": null,
      "probability_basis": "说明数值依据或无法估计原因",
      "higher_risk_region": null,
      "expected_affected_people": null,
      "expected_direct_economic_loss_cny": null,
      "loss_basis": "说明损失估计依据或缺失信息",
      "missing_inputs": []
    }
  ],
  "overall_limitations": []
}
非null概率必须在0到1之间。各灾种可以并发，概率不要求加总为1。
不要将不同灾种的损失直接相加。

统计信息：
""" + json.dumps(stats, ensure_ascii=False, allow_nan=False)

# 仅发送明确构造的统计和图像，不发送原清单中的灾害标签。
payload = {
    "model": "glm-4.6v",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," +
                       base64.b64encode(image_path.read_bytes()).decode()
            }},
        ],
    }],
    "temperature": 0.1,
    "max_tokens": 6000,
}
(OUT / f"{item['sample_id']}_prompt.txt").write_text(
    prompt, encoding="utf-8",
)

print("调用 GLM-4.6V，可能需要几分钟……", flush=True)
response = requests.post(
    "http://127.0.0.1:8002/v1/chat/completions",
    json=payload, timeout=600,
)
if not response.ok:
    raise RuntimeError(
        f"模型返回 {response.status_code}：{response.text[:3000]}"
    )
result = response.json()
response_path = OUT / f"{item['sample_id']}_response.json"
response_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
)
answer = result["choices"][0]["message"].get("content")
print("\n模型输出：\n", answer)
print("\n结束原因：", result["choices"][0].get("finish_reason"))
print("输入图：", image_path)
print("完整响应：", response_path)
