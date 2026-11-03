import io
import json
import shlex
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path.home() / "satellite_geo/fy4a"
MANIFEST = ROOT / "training/satellite_disaster_weak_samples_v5_population.jsonl"

def load_sample(item):
    """读取完整卫星帧、同年人口及有效性掩膜，不裁剪。"""
    ref = item["population_reference"]
    year = int(item["time_utc"][:4])
    if ref["year"] != year or item["year"] != year:
        raise ValueError("人口年份与卫星年份不一致")
    if item["grid_id"] != ref["grid_id"]:
        raise ValueError("网格标识不一致")

    sat = item["satellite"]
    code = (
        "import io,numpy as np,sys;"
        f"a=np.load({sat['path']!r},allow_pickle=False);"
        "b=io.BytesIO();np.save(b,a,allow_pickle=False);"
        "sys.stdout.buffer.write(b.getvalue())"
    )
    command = "/home/ices/miniconda3/bin/python -c " + shlex.quote(code)
    response = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            f"{sat['user']}@{sat['host']}", command,
        ],
        capture_output=True, timeout=180,
    )
    if response.returncode:
        raise RuntimeError(response.stderr.decode(errors="replace"))
    satellite = np.load(io.BytesIO(response.stdout), allow_pickle=False)

    with np.load(ref["file"], allow_pickle=False) as data:
        population = data[ref["value_key"]].copy()
        valid = data[ref["valid_mask_key"]].copy()
        extent = data[ref["extent_mask_key"]].copy()
        complete = data[ref["complete_selected_pixels_mask_key"]].copy()

    if satellite.shape != (8, 518, 460):
        raise ValueError(f"卫星尺寸异常：{satellite.shape}")
    for array in (population, valid, extent, complete):
        if array.shape != satellite.shape[-2:]:
            raise ValueError("人口或掩膜尺寸与卫星不一致")
    if not np.array_equal(np.isfinite(population), valid):
        raise ValueError("人口有效性掩膜不一致")
    if np.any(population[valid] < 0):
        raise ValueError("存在负人口值")
    if not np.isfinite(satellite).all():
        raise ValueError("卫星包含非有限值")

    return {
        "satellite": satellite,
        "population": population,
        "population_valid": valid,
        "population_extent": extent,
        "population_selected_pixels_complete": complete,
        "sample_metadata": item,
    }

if __name__ == "__main__":
    examples = {}
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                examples.setdefault(item["year"], item)

    for year, item in sorted(examples.items()):
        print(f"{year} 年：正在读取 {item['sample_id']}", flush=True)
        result = load_sample(item)
        print(
            "通过；卫星:", result["satellite"].shape,
            "人口:", result["population"].shape,
            "有效格子:", int(result["population_valid"].sum()),
            flush=True,
        )
    print("逐年抽样联合读取完成；未裁剪、未启动训练。", flush=True)
