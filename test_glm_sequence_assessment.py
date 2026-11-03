import base64
import json
import re
from pathlib import Path

import requests

ROOT = Path.home() / "satellite_geo/fy4a"
OLD = ROOT / "glm_multihazard_rehearsal"
OUT = ROOT / "glm_sequence_assessment"
OUT.mkdir(parents=True, exist_ok=True)
NAME = "201801071600_full_grid_weak"

text = (OLD / f"{NAME}_prompt.txt").read_text(encoding="utf-8")
stats = json.loads(text.rsplit("统计信息：", 1)[1].strip())
sequence_folder = ROOT / "glm_sequence_trial"
sequence = json.loads(
    (sequence_folder / "sequence_statistics.json").read_text(encoding="utf-8")
)
if sequence["available_frames"] != 6 or sequence["missing"]:
    raise ValueError("时序不完整")
if sequence["anchor_time_utc"] != "2018-01-07T16:00:00+00:00":
    raise ValueError("时序与当前试验样本时间不一致")
stats = {
    "static_and_anchor_statistics": stats,
    "past_hour_sequence": sequence,
}
image = base64.b64encode(
    (sequence_folder / "past_hour_sequence.png").read_bytes()
).decode()

hazards = [
    "cold_freeze", "debris_flow", "drought", "flash_flood",
    "flood", "hail", "heatwave", "heavy_rain", "landslide",
    "river_flood", "snow_sleet", "storm_surge", "tornado",
    "tropical_cyclone", "tropical_depression", "urban_flood",
    "waterspout", "wildfire",
]
combined = []

for start in range(0, len(hazards), 3):
    batch = hazards[start:start+3]
    print("正在评估：", batch, flush=True)

    prompt = """
这是历史样本的多灾种受灾预测流程预演。
先说明观测支持的趋势，再判断是否有量化受灾概率和损失的依据。
即使提供连续多帧，没有量化关系也允许返回null。
三个数值均为null时，status必须为insufficient_data。
雷达反射率不能直接当作地面雨量；已有C12亮温不能说成完全没有云顶信息。
只根据附图和统计评估未来一小时，不使用该日期的历史灾情记忆。
附图共2行6列，列从左到右为过去一小时的6个时刻。
上排为卫星C12亮温，越亮越冷，不是地面温度；
下排为对应时刻的雷达，灰色为覆盖区外。
所有输入均不晚于预测起报时刻，预测未来一小时。
分析回波位置、面积和强度变化，但不能把面积变化直接换算成受灾概率。
不能仅凭整体面积减少就判定所有局部地区风险下降。
这些是过去观测，不是未来气象预报。
卫星定位为近似坐标。
人口是大范围有效像素人口之和，不是受灾人口。
土地覆盖是大范围均值，不代表某个风险区域。

只评估指定的3类。直接输出JSON数组，不输出推理过程。
每项仅包含：
hazard_type、status、probability、affected_people、
loss_cny、reason、missing_inputs。
status只能为experimental_estimate或insufficient_data。
reason不超过50个汉字；missing_inputs最多3项。

禁止把回波面积占比、土地覆盖占比当作事件发生或受灾概率。
不得仅因没有强回波就判定灾害不会发生。
没有量化依据的概率、人数、损失必须为null。
没有依据就返回insufficient_data；不能为了输出数字而猜测。
概率若有依据只能在0至1之间，且仍未经校准。
慢发灾害不适用仅过去一小时观测时应说明。
""" + "\n指定灾种：" + json.dumps(batch) + "\n统计：" + json.dumps(
        stats, ensure_ascii=False, allow_nan=False
    )

    response = requests.post(
        "http://127.0.0.1:8002/v1/chat/completions",
        json={
            "model": "glm-4.6v",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + image
                    }},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 3000,
        },
        timeout=600,
    )
    response.raise_for_status()
    raw = response.json()
    (OUT / f"batch_{start//3+1}_response.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    choice = raw["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise RuntimeError("本批未正常结束，完整响应已保存，停止合并")

    answer = choice["message"].get("content") or ""
    if "<think>" in answer and "</think>" not in answer:
        raise ValueError("思考段未结束，不能提取最终结果")
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    answer = answer.replace("<|begin_of_box|>", "")
    answer = answer.replace("<|end_of_box|>", "")
    answer = answer.strip()
    if answer.startswith("```"):
        answer = re.sub(r"^```(?:json)?\s*", "", answer)
        answer = re.sub(r"\s*```$", "", answer)

    rows = json.loads(answer)
    if not isinstance(rows, list) or len(rows) != len(batch):
        raise ValueError("返回条目数异常")
    if sorted(r["hazard_type"] for r in rows) != sorted(batch):
        raise ValueError("灾种遗漏或重复")

    required = {
        "hazard_type", "status", "probability", "affected_people",
        "loss_cny", "reason", "missing_inputs",
    }
    for row in rows:
        if not required.issubset(row):
            raise ValueError("字段不完整")
        if row["status"] not in ("experimental_estimate", "insufficient_data"):
            raise ValueError("状态异常")
        for key in ("probability", "affected_people", "loss_cny"):
            value = row[key]
            if value is not None:
                if (type(value) not in (int, float)
                    or not (0 <= value < float("inf"))):
                    raise ValueError(f"{key}数值异常")
        p = row["probability"]
        if p is not None and p > 1:
            raise ValueError("概率超范围")
        if row["status"] == "insufficient_data" and any(
            row[k] is not None
            for k in ("probability", "affected_people", "loss_cny")
        ):
            raise ValueError("证据不足状态却输出数值")
    combined.extend(rows)
    print("本批结构检查通过", flush=True)

result = {
    "mode": "pipeline_rehearsal",
    "probability_calibrated": False,
    "forecast_horizon_minutes": 60,
    "hazards": combined,
}
target = OUT / f"{NAME}_validated_structure.json"
target.write_text(
    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\n18类输出结构完整，已保存：", target)
print("结构通过不代表概率和损失准确，仍需独立验证。")
