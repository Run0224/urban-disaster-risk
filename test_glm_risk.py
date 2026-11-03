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
OUT = ROOT / "glm_risk_trial"
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
请根据附图和统计信息，进行一次历史样本的未来一小时强降水风险试验。
只使用提供的观测，不查询或回忆该日期实际灾情。
左图为C12亮温，越亮越冷；右图为雷达反射率，灰色为覆盖区外，
零值包含清洗后的弱回波，不代表覆盖区外。
两图时间对应同一时刻。不得把红外亮温直接换算为降雨量。
只有单帧，没有运动趋势、未来观测和地面雨量，不能确认未来降水量。
人口、土地覆盖仅辅助描述暴露程度，不能作为降雨发生的证据。
不要因有人口就提高天气危险性评分；不要虚构市县名称。
评分是未校准主观风险指数，不是灾害发生概率。
请仅返回JSON，字段如下：
hazard、forecast_horizon_minutes、risk_level、
uncalibrated_risk_score_0_100（证据不足时可为null）、
higher_risk_regions（使用经纬度范围描述）、
evidence、exposure_interpretation、limitations、probability_calibrated（false）。
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
    "max_tokens": 1800,
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
