import io
import json
import shlex
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from align_satellite_to_radar import resample

ROOT = Path.home() / "satellite_geo/fy4a"
OUT = ROOT / "case_replay/202207040624"
OUT.mkdir(parents=True, exist_ok=True)

manifest = ROOT / "training/satellite_disaster_weak_samples_v6_geography.jsonl"
with manifest.open(encoding="utf-8") as f:
    item = json.loads(next(line for line in f if line.strip()))

# 固定历史起报时刻；不包含灾害名称、标签或损失。
item = {
    "sample_id": "202207040624_replay",
    "time_utc": "2022-07-04T06:24:00+00:00",
    "satellite": {
        "host": "192.168.48.29",
        "user": "ices",
        "path": (
            "/mnt/data/zk/diffcast_3d/data/sat_6min/"
            "2022/07/04/202207040624/202207040624.npy"
        ),
    },
}
anchor = datetime.fromisoformat(item["time_utc"])
times = [anchor - timedelta(minutes=m) for m in (60, 48, 36, 24, 12, 0)]
sat_root = Path(item["satellite"]["path"]).parents[4]
host = item["satellite"]["host"]
user = item["satellite"]["user"]

with np.load(ROOT / "coordinates_approx.npz", allow_pickle=False) as d:
    src_lat, src_lon = d["latitude"].copy(), d["longitude"].copy()
lat = 26.0319 - np.arange(700)*0.01
lon = 108.505 + np.arange(900)*0.01
coverage = np.load(
    ROOT / "alignment_checks/radar_valid_mask.npy",
    allow_pickle=False,
).astype(bool)
if coverage.shape != (700, 900):
    raise ValueError("雷达覆盖掩膜尺寸异常")

frames, stats, missing = [], [], []

for moment in times:
    stamp = moment.strftime("%Y%m%d%H%M")
    sat_path = sat_root / moment.strftime("%Y/%m/%d") / stamp / f"{stamp}.npy"
    radar_path = (
        Path("/home/ices/data/radar_gd")
        / (moment + timedelta(hours=8)).strftime("%Y/%m/%d/%Y%m%d%H%M.png")
    )
    print("读取UTC：", stamp, flush=True)
    code = (
        "import io,sys,numpy as np;"
        "from pathlib import Path;"
        f"a=np.load({str(sat_path)!r},allow_pickle=False);"
        f"r=np.frombuffer(Path({str(radar_path)!r}).read_bytes(),dtype=np.uint8);"
        "b=io.BytesIO();np.savez_compressed(b,satellite=a,radar_png=r);"
        "sys.stdout.buffer.write(b.getvalue())"
    )
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         f"{user}@{host}",
         "/home/ices/miniconda3/bin/python -c " + shlex.quote(code)],
        capture_output=True, timeout=180,
    )
    if response.returncode:
        missing.append({
            "time_utc": moment.isoformat(),
            "error": response.stderr.decode(errors="replace"),
        })
        print("该时刻读取失败，保留缺失记录")
        continue

    with np.load(io.BytesIO(response.stdout), allow_pickle=False) as d:
        sat = d["satellite"].copy()
        png = d["radar_png"].tobytes()
    if sat.shape != (8, 518, 460):
        raise ValueError(f"卫星尺寸异常：{sat.shape}")
    # 当前只展示C12；不重复转换其余7个通道。
    bt, valid, _, _ = resample(
        sat[5:6], src_lat, src_lon, lat, lon
    )
    with Image.open(io.BytesIO(png)) as im:
        if im.mode != "L":
            raise ValueError(f"雷达图像模式异常：{im.mode}")
        radar = np.array(im).astype(np.float32)
    if radar.shape != (700, 900):
        raise ValueError("雷达尺寸异常")
    radar[(radar < 15) | (radar >= 74)] = 0
    radar[~coverage] = np.nan

    frames.append((moment, bt[0], radar))
    stats.append({
        "time_utc": moment.isoformat(),
        "radar_cells_ge_35_dbz": int(np.sum(coverage & (radar >= 35))),
        "radar_cells_ge_50_dbz": int(np.sum(coverage & (radar >= 50))),
        "cold_cloud_cells_below_250K_within_radar_coverage":
            int(np.sum(coverage & valid[0] & (bt[0] < 250))),
    })

report = {
    "anchor_time_utc": anchor.isoformat(),
    "requested_frames": len(times),
    "available_frames": len(frames),
    "missing": missing,
    "frame_statistics": stats,
    "note": "回波或冷云面积占比不是受灾概率；卫星定位为近似坐标。",
}
(OUT / "sequence_statistics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)

# 不静默替换时间或用零填充缺失帧。
if len(frames) != len(times):
    raise RuntimeError(
        f"仅找到{len(frames)}/{len(times)}帧；缺失详情已保存，先不调用模型"
    )

extent = [lon[0]-.005, lon[-1]+.005, lat[-1]-.005, lat[0]+.005]
fig, axes = plt.subplots(
    2, len(frames), figsize=(20, 8), constrained_layout=True
)
cmap = plt.get_cmap("turbo").copy()
cmap.set_bad("lightgray")
for i, (moment, bt, radar) in enumerate(frames):
    s = axes[0, i].imshow(
        np.ma.masked_invalid(bt), extent=extent, origin="upper",
        cmap="gray_r", vmin=180, vmax=320,
    )
    r = axes[1, i].imshow(
        np.ma.masked_invalid(radar), extent=extent, origin="upper",
        cmap=cmap, vmin=0, vmax=74,
    )
    axes[0, i].set_title(moment.strftime("%m-%d %H:%M UTC"))
    for ax in axes[:, i]:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect(1/np.cos(np.deg2rad(lat.mean())))
fig.colorbar(s, ax=list(axes[0]), label="C12 brightness temperature (K)",
             shrink=.7)
fig.colorbar(r, ax=list(axes[1]), label="Radar reflectivity (dBZ)",
             shrink=.7)
fig.suptitle("Past-hour observations | top: satellite; bottom: radar")
image = OUT / "past_hour_sequence.png"
fig.savefig(image, dpi=130)
plt.close(fig)

print(json.dumps(report, ensure_ascii=False, indent=2))
print("时序图已保存：", image)
print("准备完成；尚未调用GLM。")



from rasterio.features import geometry_mask
from rasterio.transform import from_origin

# 缓存此次案例，不包含灾情标签。
radars = np.stack([f[2] for f in frames])
brightness = np.stack([f[1] for f in frames])
np.savez_compressed(
    OUT / "observations.npz",
    radar_dbz=radars,
    satellite_c12_bt=brightness,
    radar_coverage_mask=coverage,
    latitude=lat, longitude=lon,
    times_utc=np.array([f[0].isoformat() for f in frames]),
)

disaster_folder = ROOT / (
    "geographic_library/disasters/"
    "Guangdong_Disaster_Records_2018_2024Jan"
)
boundary = disaster_folder / (
    "Guangdong_County_Boundary_Matching_29/"
    "county_boundaries_wgs84_approx.geojson"
)
geojson = json.loads(boundary.read_text(encoding="utf-8"))
features = geojson.get("features", [])

selected = []
for feature in features:
    values = {str(v) for v in (feature.get("properties") or {}).values()}
    if values.intersection({"440607", "440607.0", "三水区"}):
        selected.append(feature)

if len(selected) != 1:
    raise ValueError(
        f"三水区边界匹配到{len(selected)}个，停止，避免选错区域"
    )

county = geometry_mask(
    [selected[0]["geometry"]],
    out_shape=(700, 900),
    transform=from_origin(lon[0]-.005, lat[0]+.005, .01, .01),
    invert=True,
    all_touched=False,
)
observed = county & coverage
if not county.any() or not observed.any():
    raise ValueError("三水区未落入当前有效网格")

result = {
    "target_region": "佛山市三水区",
    "anchor_time_utc": anchor.isoformat(),
    "forecast_horizon_minutes": 120,
    "boundary_status": "行政参考边界，坐标近似，历史年份未核实；不是受灾范围",
    "county_grid_cells": int(county.sum()),
    "radar_covered_county_cells": int(observed.sum()),
    "radar_coverage_fraction": float(observed.sum()/county.sum()),
    "frame_statistics": [],
}
for i, (moment, _, _) in enumerate(frames):
    r, bt = radars[i], brightness[i]
    rv = observed & np.isfinite(r)
    bv = observed & np.isfinite(bt)
    result["frame_statistics"].append({
        "time_utc": moment.isoformat(),
        "radar_max_dbz": float(r[rv].max()) if rv.any() else None,
        "radar_cells_ge_35_dbz": int(np.sum(rv & (r >= 35))),
        "radar_cells_ge_50_dbz": int(np.sum(rv & (r >= 50))),
        "cold_cloud_cells_below_250K": int(np.sum(bv & (bt < 250))),
        "valid_radar_cells": int(rv.sum()),
        "valid_c12_cells": int(bv.sum()),
    })

folder = ROOT / "aligned/radar_grid_approx/geography"
for kind in ("population", "land_cover"):
    with np.load(folder / f"{kind}_2022.npz", allow_pickle=False) as d:
        if not (np.array_equal(d["latitude"], lat)
                and np.array_equal(d["longitude"], lon)):
            raise ValueError(f"{kind}坐标不一致")
        valid = county & d["has_valid_data"]
        if kind == "population":
            result["population"] = {
                "valid_cells": int(valid.sum()),
                "sum_over_valid_source_pixels": float(
                    d["population_over_valid_source_pixels"][valid].sum()
                ) if valid.any() else None,
                "note": "行政范围内有效源像素的人口估计之和，可能覆盖不完整，不是受灾人口",
            }
        else:
            result["land_cover"] = {
                "valid_cells": int(valid.sum()),
                "mean_proportions": {
                    str(name): float(d["class_proportions"][k][valid].mean())
                    if valid.any() else None
                    for k, name in enumerate(d["class_names"])
                },
            }

result["limitations"] = [
    "卫星位置和行政边界均存在近似误差",
    "年人口和土地覆盖为回顾性产品，历史可用性未核实",
    "没有输入灾害记录或实际损失",
    "尚未接入本案例的地形、风场和定量受灾模型",
]
target = OUT / "sanshui_input_features.json"
target.write_text(
    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)
np.save(OUT / "sanshui_admin_reference_mask.npy", county)
print("\n三水区输入特征：")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("已保存：", target)
