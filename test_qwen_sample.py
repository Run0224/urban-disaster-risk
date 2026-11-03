import base64
import json
import urllib.request
import urllib.error
from pathlib import Path

folder = Path.home() / "satellite_geo/fy4a"
sample = json.loads(
    (folder / "sample_2018_center256_with_image.jsonl")
    .read_text(encoding="utf-8").strip()
)

image = Path(sample["image"]["path"])
encoded = base64.b64encode(image.read_bytes()).decode("ascii")

# 控制输入长度；不把几百个区县名称全部发送。
context = {
    "time_utc": sample["time_utc"],
    "geolocation_status": sample["geolocation_status"],
    "bounds": sample["bounds_west_south_east_north"],
    "image": {
        "channel": "C12",
        "wavelength_um": 10.8,
        "display": "Lower brightness temperatures appear brighter."
    },
    "terrain": {
        "maximum_surface_elevation_m":
            sample["terrain"]["maximum_surface_elevation_m"],
        "valid_satellite_cells":
            sample["terrain"]["valid_satellite_cells"],
        "missing_satellite_cells":
            sample["terrain"]["missing_satellite_cells"],
        "minimum_elevation": "Pending quality review; do not use."
    },
    "population": {
        "year": sample["population"]["population_year"],
        "estimate_over_valid_source_pixels":
            sample["population"]["population_estimate_over_valid_source_pixels"],
        "limitation": sample["population"]["limitation"]
    }
}

prompt = (
    "这是一个数据接入测试。根据所附单时刻红外卫星图和资料，"
    "用中文分三点回答：1.图上可直接观察到的亮温分布；"
    "2.资料提供的地理与人口背景；3.目前不能得出的结论。"
    "不要把人口总量说成广东省人口；不要猜测具体区县；"
    "不要声称可以凭这一帧确定降雨量、灾害发生或发布预警。"
    "坐标是近似的，人口不是实时人数。请简短回答。\n"
    + json.dumps(context, ensure_ascii=False)
)

request_body = {
    "model": "qwen-next",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + encoded
                }
            }
        ]
    }],
    "max_tokens": 1200,
    "temperature": 0.1,
    "stream": False
}

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(request_body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

print("正在请求本机模型……", flush=True)
try:
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    print("服务返回错误：", error.code)
    print(error.read().decode("utf-8", errors="replace"))
    raise SystemExit(1)

output = folder / "qwen_sample_response.json"
output.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
message = result["choices"][0]["message"]
print("模型回复：")
print(message.get("content") or "未返回最终正文，请查看保存的完整响应。")
print("完整响应已保存：", output)
