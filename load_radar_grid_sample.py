import io
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path.home() / "satellite_geo/fy4a"
GRID = ROOT / "aligned/radar_grid_approx"

def load_sample(item):
    moment = datetime.fromisoformat(
        item["time_utc"].replace("Z", "+00:00")
    )
    # 清单时间为UTC；雷达文件名为北京时间。
    utc = moment.utcoffset()
    if utc != timedelta(0):
        raise ValueError("样本时间必须明确为UTC")
    bjt = moment + timedelta(hours=8)
    year = moment.year

    satellite_file = GRID / f"{item['sample_id']}_satellite.npz"
    with np.load(satellite_file, allow_pickle=False) as d:
        satellite = d["satellite_bt"].copy()
        satellite_valid = d["finite_interpolation_mask"].copy()
        lat = d["latitude"].copy()
        lon = d["longitude"].copy()

    layers = {}
    for kind in ("population", "land_cover"):
        path = GRID / "geography" / f"{kind}_{year}.npz"
        with np.load(path, allow_pickle=False) as d:
            if not (np.array_equal(d["latitude"], lat)
                    and np.array_equal(d["longitude"], lon)):
                raise ValueError(f"坐标不一致：{path}")
            layers[kind] = {key: d[key].copy() for key in d.files}

    radar_path = (
        "/home/ices/data/radar_gd/"
        + bjt.strftime("%Y/%m/%d/%Y%m%d%H%M.png")
    )
    print("读取雷达：", radar_path, flush=True)
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         "ices@192.168.48.29", f"cat {radar_path}"],
        capture_output=True, timeout=120,
    )
    if response.returncode:
        raise RuntimeError(response.stderr.decode(errors="replace"))

    with Image.open(io.BytesIO(response.stdout)) as image:
        if image.mode != "L":
            raise ValueError(f"预期单通道雷达，实际模式：{image.mode}")
        radar = np.array(image).astype(np.float32)

    mask_path = ROOT / "alignment_checks/radar_valid_mask.npy"
    coverage = np.load(mask_path, allow_pickle=False)
    if not np.isin(coverage, [0, 1]).all():
        raise ValueError("雷达覆盖掩膜必须为0/1")
    coverage = coverage.astype(bool)
    if radar.shape != (700, 900) or coverage.shape != radar.shape:
        raise ValueError("雷达或覆盖掩膜尺寸异常")
    if satellite.shape != (8, 700, 900):
        raise ValueError("卫星尺寸异常")

    radar[(radar < 15) | (radar >= 74)] = 0
    radar[~coverage] = np.nan

    population = layers["population"]
    land = layers["land_cover"]
    return {
        "satellite_bt": satellite,
        "satellite_finite_mask": satellite_valid,
        "radar_dbz": radar,
        "radar_coverage_mask": coverage,
        "population": population["population_over_valid_source_pixels"],
        "population_valid": population["has_valid_data"],
        "land_cover_proportions": land["class_proportions"],
        "land_cover_valid": land["has_valid_data"],
        "land_cover_class_names": land["class_names"],
        "latitude": lat,
        "longitude": lon,
        # 原清单完整保留，包括灾害关联；旧网格引用仍指向旧文件。
        "source_sample_metadata": item,
        "target_grid_id": "radar_700x900_approx_mapping_v1",
        "geolocation_status": "approximate_from_readme",
    }

if __name__ == "__main__":
    manifest = ROOT / "training/satellite_disaster_weak_samples_v6_geography.jsonl"
    with manifest.open(encoding="utf-8") as f:
        item = json.loads(next(line for line in f if line.strip()))

    result = load_sample(item)
    for key in (
        "satellite_bt", "radar_dbz", "population",
        "land_cover_proportions",
    ):
        print(key, result[key].shape)
    print("雷达覆盖格子：", int(result["radar_coverage_mask"].sum()))
    print("样本：", item["sample_id"])
    print("联合读取完成；原样本元数据保留，未启动训练。")
