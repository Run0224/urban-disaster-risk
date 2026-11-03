import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from align_satellite_to_radar import resample

ROOT = Path.home() / "satellite_geo/fy4a"
TRAIN = ROOT / "training"
OUT = TRAIN / "baseline_features_v1"
OUT.mkdir(parents=True, exist_ok=True)

def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

samples = read(TRAIN / "combined_candidates_v1/input_inventory.jsonl")
coverage = np.load(
    ROOT / "alignment_checks/radar_valid_mask.npy",
    allow_pickle=False,
).astype(bool)
lat = 26.0319 - np.arange(700)*0.01
lon = 108.505 + np.arange(900)*0.01

with np.load(ROOT / "coordinates_approx.npz", allow_pickle=False) as d:
    slat, slon = d["latitude"].copy(), d["longitude"].copy()

with np.load(ROOT / "terrain_statistics_approx_v2.npz",
             allow_pickle=False) as d:
    terrain = {k: d[k].copy() for k in d.files}

folder = ROOT / (
    "geographic_library/disasters/"
    "Guangdong_Disaster_Records_2018_2024Jan"
)
boundary = folder / (
    "Guangdong_County_Boundary_Matching_29/"
    "county_boundaries_wgs84_approx.geojson"
)
features = json.loads(boundary.read_text(encoding="utf-8"))["features"]

# 从已有地点关联中取得行政代码。
codes = {}
for record in read(folder / "disaster_admin_references.jsonl"):
    for mention in record.get("existing_boundary_mentions") or []:
        ref = mention.get("web_boundary_reference") or {}
        key = (
            ref.get("city") or mention.get("city"),
            ref.get("county") or mention.get("county"),
        )
        if ref.get("adcode"):
            codes.setdefault(key, set()).add(str(ref["adcode"]))

masks = {}
annual = {}

def region_mask(region):
    city, county = region.get("city"), region.get("county")
    key = (region.get("province"), city, county)
    if key in masks:
        return masks[key]

    matches = []
    for feature in features:
        props = feature.get("properties") or {}
        values = {str(v) for v in props.values()}
        if county:
            matched = county in values or bool(
                codes.get((city, county), set()) & values
            )
            parent = props.get("city") or props.get("city_name")
            if parent and city and parent != city:
                matched = False
        else:
            # 不用不完整的县区集合拼成整个城市。
            matched = (
                city and props.get("name") == city
                and props.get("level") == "city"
            )
        if matched:
            matches.append(feature)

    if len(matches) != 1:
        raise ValueError(f"完整行政边界未匹配：{key}，匹配数={len(matches)}")

    mask = geometry_mask(
        [matches[0]["geometry"]],
        out_shape=(700, 900),
        transform=from_origin(lon[0]-.005, lat[0]+.005, .01, .01),
        invert=True,
    )
    if not mask.any():
        raise ValueError("区域未落入目标网格")
    masks[key] = mask
    return mask

def mean(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None

def nearest(axis, targets):
    upper = np.clip(np.searchsorted(axis, targets), 1, len(axis)-1)
    lower = upper-1
    return np.where(
        abs(targets-axis[lower]) <= abs(targets-axis[upper]),
        lower, upper,
    )

tlat, tlon = terrain["latitude"], terrain["longitude"]
if not (np.all(np.diff(tlat) < 0) and np.all(np.diff(tlon) > 0)):
    raise ValueError("地形坐标方向异常")
if (lat.min() < tlat.min() or lat.max() > tlat.max()
    or lon.min() < tlon.min() or lon.max() > tlon.max()):
    raise ValueError("地形坐标范围不足")
ri = len(tlat)-1-nearest(tlat[::-1], lat)
ci = nearest(tlon, lon)
terrain_ids = ri[:, None]*len(tlon)+ci[None, :]

success, failed = [], []

for index, sample in enumerate(samples, 1):
    sid = sample["sample_id"]
    print(f"[{index}/{len(samples)}] {sid}", flush=True)
    try:
        mask = region_mask(sample["region"])
        observed = mask & coverage
        if not observed.any():
            raise ValueError("区域不在雷达覆盖范围内")

        frames = sorted(sample["input_frames"], key=lambda x: x["time_utc"])
        issue = datetime.fromisoformat(sample["issue_time_utc"])
        times = [datetime.fromisoformat(f["time_utc"]) for f in frames]
        if len(times) != 10 or len(set(times)) != 10:
            raise ValueError("不是10个不同的观测时刻")
        if any(t >= issue or t < issue-timedelta(hours=1) for t in times):
            raise ValueError("观测超出起报前一小时")
        if any((b-a).total_seconds() != 360 for a, b in zip(times, times[1:])):
            raise ValueError("观测间隔不是6分钟")

        history, cache_keys = [], []
        for frame in frames:
            digest = hashlib.sha256(json.dumps(
                [frame["satellite"], frame["radar"]]
            ).encode()).hexdigest()[:20]
            cache_keys.append(digest)
            with np.load(
                TRAIN / "raw_input_cache_v1" / f"{digest}.npz",
                allow_pickle=False,
            ) as d:
                radar = d["radar"].astype(np.float32)
                sat = d["satellite"].copy()
            if radar.shape != (700, 900) or sat.shape != (8, 518, 460):
                raise ValueError("缓存数组尺寸异常")

            radar[(radar < 15) | (radar >= 74)] = 0
            bt, valid, _, _ = resample(
                sat[5:6], slat, slon, lat, lon
            )
            r = radar[observed]
            cloud_valid = observed & valid[0]
            history.append({
                "radar_max_dbz": float(r.max()),
                "radar_ge35_fraction": float(np.mean(r >= 35)),
                "radar_ge50_fraction": float(np.mean(r >= 50)),
                "c12_mean_k": mean(bt[0][cloud_valid]),
                "c12_finite_fraction":
                    float(cloud_valid.sum()/observed.sum()),
            })

        vector = {
            "radar_coverage_fraction": float(observed.sum()/mask.sum()),
        }
        for key in history[0]:
            values = [h[key] for h in history]
            vector[key+"_last"] = values[-1]
            vector[key+"_hour_mean"] = mean([
                v for v in values if v is not None
            ])
            vector[key+"_change"] = (
                values[-1]-values[0]
                if values[-1] is not None and values[0] is not None
                else None
            )

        year = (
            datetime.fromisoformat(sample["target_start_utc"])
            + timedelta(hours=8)
        ).year
        if year not in annual:
            annual[year] = {}
            for kind in ("population", "land_cover"):
                path = ROOT / (
                    f"aligned/radar_grid_approx/geography/{kind}_{year}.npz"
                )
                with np.load(path, allow_pickle=False) as d:
                    if not (
                        np.array_equal(d["latitude"], lat)
                        and np.array_equal(d["longitude"], lon)
                    ):
                        raise ValueError("年度图层经纬度不一致")
                    annual[year][kind] = {
                        k: d[k].copy() for k in d.files
                    }

        pop = annual[year]["population"]
        pv = mask & pop["has_valid_data"]
        vector["population_valid_fraction"] = float(pv.sum()/mask.sum())
        vector["population_sum_valid"] = (
            float(pop["population_over_valid_source_pixels"][pv].sum())
            if pv.any() else None
        )

        land = annual[year]["land_cover"]
        lv = mask & land["has_valid_data"]
        vector["land_cover_valid_fraction"] = float(lv.sum()/mask.sum())
        for k, name in enumerate(land["class_names"]):
            vector["land_"+str(name)] = mean(
                land["class_proportions"][k][lv]
            )

        ids = np.unique(terrain_ids[mask])
        ids = ids[terrain["valid_mask"].ravel()[ids]]
        vector["terrain_valid_source_cells"] = int(ids.size)
        for key in ("surface_elevation_mean_m", "terrain_relief_m"):
            vector[key] = mean(terrain[key].ravel()[ids])

        label = sample.get(
            "regional_impact_label", sample.get("proposed_label")
        )
        if label not in (0, 1):
            raise ValueError("候选标签异常")

        success.append({
            "sample_id": sid,
            "hazard_type": sample["hazard_type"],
            "label": label,
            "label_quality": sample.get(
                "label_quality", "reported_impact_candidate"
            ),
            "region": sample["region"],
            "issue_time_utc": sample["issue_time_utc"],
            "event_group_id": sample.get("event_group_id"),
            "matched_positive_event_groups":
                sample.get("matched_positive_event_groups", []),
            "matched_positive_sample_ids":
                sample.get("matched_positive_sample_ids", []),
            "input_cache_keys": cache_keys,
            "features": vector,
        })
        print("特征提取完成", flush=True)

    except Exception as error:
        failed.append({
            "sample_id": sid,
            "region": sample["region"],
            "reason": str(error),
        })
        print("未提取：", str(error), flush=True)

    # 每处理一条即保存，已有天气缓存不会改变。
    with (OUT / "features.jsonl").open("w", encoding="utf-8") as f:
        for row in success:
            f.write(json.dumps(
                row, ensure_ascii=False, allow_nan=False
            ) + "\n")
    (OUT / "report.json").write_text(json.dumps({
        "processed": index,
        "requested": len(samples),
        "success": len(success),
        "failed": len(failed),
        "success_by_label": dict(Counter(str(r["label"]) for r in success)),
        "failures": failed,
        "note": "探索性特征；近似定位、回顾性年度图层、弱负例和插值时点限制仍保留。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n本轮完成")
print("成功：", len(success), "未提取：", len(failed))
print("标签分布：", dict(Counter(str(r["label"]) for r in success)))
print("输出目录：", OUT)
