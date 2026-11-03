import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "satellite_geo/fy4a"
sys.path.insert(0, str(ROOT))
from load_satellite_population import load_sample

def resample(values, source_lat, source_lon, target_lat, target_lon):
    """根据像素中心经纬度进行双线性插值；不向范围外外推。"""
    if values.shape[-2:] != (source_lat.size, source_lon.size):
        raise ValueError("数据尺寸与源坐标不一致")
    if not (np.all(np.diff(source_lat) < 0)
            and np.all(np.diff(source_lon) > 0)):
        raise ValueError("要求源纬度递减、经度递增")
    if not (np.isfinite(source_lat).all()
            and np.isfinite(source_lon).all()):
        raise ValueError("源坐标存在非有限值")

    if (target_lat.min() < source_lat.min()
        or target_lat.max() > source_lat.max()
        or target_lon.min() < source_lon.min()
        or target_lon.max() > source_lon.max()):
        raise ValueError("目标网格超出源坐标范围")

    rows = np.interp(
        target_lat, source_lat[::-1],
        np.arange(source_lat.size)[::-1],
    )
    cols = np.interp(
        target_lon, source_lon, np.arange(source_lon.size),
    )
    r0 = np.clip(np.floor(rows).astype(int), 0, source_lat.size - 2)
    c0 = np.clip(np.floor(cols).astype(int), 0, source_lon.size - 2)
    wr = (rows - r0)[:, None]
    wc = (cols - c0)[None, :]

    output = np.full(
        (values.shape[0], target_lat.size, target_lon.size),
        np.nan, dtype=np.float32,
    )
    valid = np.zeros(output.shape, dtype=bool)

    for channel, a in enumerate(values):
        p00 = a[r0[:, None], c0[None, :]]
        p01 = a[r0[:, None], (c0 + 1)[None, :]]
        p10 = a[(r0 + 1)[:, None], c0[None, :]]
        p11 = a[(r0 + 1)[:, None], (c0 + 1)[None, :]]

        ok = (np.isfinite(p00) & np.isfinite(p01)
              & np.isfinite(p10) & np.isfinite(p11))
        result = (
            (1 - wr) * (1 - wc) * p00
            + (1 - wr) * wc * p01
            + wr * (1 - wc) * p10
            + wr * wc * p11
        )
        output[channel] = np.where(ok, result, np.nan)
        valid[channel] = ok

    return output, valid, rows, cols

def main():
    manifest = ROOT / "training/satellite_disaster_weak_samples_v6_geography.jsonl"
    with manifest.open(encoding="utf-8") as f:
        item = json.loads(next(line for line in f if line.strip()))

    folder = ROOT / "aligned/radar_grid_approx"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{item['sample_id']}_satellite.npz"
    if target.exists():
        raise FileExistsError(f"结果已存在，保留原文件：{target}")

    with np.load(ROOT / "coordinates_approx.npz", allow_pickle=False) as data:
        source_lat = data["latitude"].copy()
        source_lon = data["longitude"].copy()

    latitude = 26.0319 - np.arange(700) * 0.01
    longitude = 108.505 + np.arange(900) * 0.01

    print("读取卫星：", item["sample_id"], flush=True)
    satellite = load_sample(item)["satellite"]
    if satellite.shape != (8, 518, 460):
        raise ValueError(f"卫星尺寸异常：{satellite.shape}")

    aligned, valid, rows, cols = resample(
        satellite, source_lat, source_lon, latitude, longitude,
    )
    if aligned.shape != (8, 700, 900):
        raise ValueError("输出尺寸异常")
    if not np.array_equal(np.isfinite(aligned), valid):
        raise ValueError("输出与有效掩膜不一致")

    metadata = {
        "sample_id": item["sample_id"],
        "time_utc": item["time_utc"],
        "source_satellite": item["satellite"],
        "source_coordinates": str(ROOT / "coordinates_approx.npz"),
        "target_grid_id": "radar_700x900_approx_mapping_v1",
        "method": "bilinear_from_pixel_center_latitude_longitude",
        "geolocation_status": "approximate_from_readme",
        "spatial_registration_verified": False,
        "mask_meaning": "all_four_source_neighbors_are_finite",
        "radar_coverage_mask_applied": False,
        "geographic_layers_included": False,
    }

    with target.open("xb") as f:
        np.savez_compressed(
            f,
            satellite_bt=aligned,
            finite_interpolation_mask=valid,
            latitude=latitude,
            longitude=longitude,
            source_row=rows,
            source_column=cols,
            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
        )

    print("卫星尺寸：", aligned.shape)
    print("各通道有限插值像素数：", valid.sum(axis=(1, 2)).tolist())
    print("卫星源行范围：", rows.min(), rows.max())
    print("卫星源列范围：", cols.min(), cols.max())
    print("已保存：", target)
    print("本次只处理卫星；人口、土地覆盖等尚未转换到此网格。")

if __name__ == "__main__":
    main()
