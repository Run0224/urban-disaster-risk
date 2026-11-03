import io
import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path.home() / "satellite_geo/fy4a"
sample_path = folder / "sample_2018_center256.jsonl"
sample = json.loads(sample_path.read_text(encoding="utf-8").strip())

satellite = sample["satellite"]
result = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
     f'{satellite["user"]}@{satellite["host"]}',
     f'cat {satellite["path"]}'],
    check=True,
    stdout=subprocess.PIPE,
)
data = np.load(io.BytesIO(result.stdout), allow_pickle=False)
if data.shape != (8, 518, 460):
    raise ValueError(f"卫星数组尺寸异常：{data.shape}")

crop = satellite["crop"]
r, c = crop["row"], crop["column"]
h, w = crop["height"], crop["width"]
image = data[5, r:r+h, c:c+w]

if image.shape != (h, w) or not np.isfinite(image).all():
    raise ValueError("裁剪尺寸异常或存在非有限亮温")

output = folder / "sample_2018_center256_C12.png"

# 无坐标轴、无边框，保存为 256×256 像素图片。
plt.imsave(
    output, image, cmap="gray_r",
    vmin=180, vmax=320, origin="upper",
)

sample["image"] = {
    "path": str(output),
    "channel": "C12",
    "wavelength_um": 10.8,
    "brightness_temperature_range_k": [180, 320],
    "colormap": "gray_r",
    "note": "Rendered preview of one channel; does not preserve all eight channels or exact brightness temperatures."
}
sample["quality_notes"] = [
    "Geolocation remains approximate.",
    "Minimum surface elevation -76.92 m in the crop is pending investigation; do not treat it as a verified terrain fact."
]

new_sample = folder / "sample_2018_center256_with_image.jsonl"
new_sample.write_text(
    json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)
print("图片已保存：", output)
print("配套样例已保存：", new_sample)
