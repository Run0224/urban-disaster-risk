import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window, from_bounds

ROOT = Path.home() / "satellite_geo/fy4a"
H, W = 700, 900
STEP = 0.01
LAT = 26.0319 - np.arange(H) * STEP
LON = 108.505 + np.arange(W) * STEP
WEST, EAST = LON[0] - STEP/2, LON[-1] + STEP/2
SOUTH, NORTH = LAT[-1] - STEP/2, LAT[0] + STEP/2

def aggregate(path, kind):
    valid_count = np.zeros(H * W, dtype=np.int64)
    missing_count = np.zeros(H * W, dtype=np.int64)
    total = np.zeros(H * W, dtype=np.float64)
    classes = np.zeros((9, H * W), dtype=np.int64)

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"原始文件缺少坐标系：{path}")
        if src.transform.b != 0 or src.transform.d != 0:
            raise ValueError("当前脚本要求原始栅格无旋转")

        bounds = transform_bounds(
            "EPSG:4326", src.crs, WEST, SOUTH, EAST, NORTH,
            densify_pts=101,
        )
        win = from_bounds(*bounds, transform=src.transform)
        c0 = max(0, int(np.floor(win.col_off)) - 2)
        r0 = max(0, int(np.floor(win.row_off)) - 2)
        c1 = min(src.width, int(np.ceil(win.col_off + win.width)) + 2)
        r1 = min(src.height, int(np.ceil(win.row_off + win.height)) + 2)
        if c0 >= c1 or r0 >= r1:
            raise ValueError("原始文件与雷达范围不相交")

        for start in range(r0, r1, 32):
            stop = min(start + 32, r1)
            data = src.read(
                1, window=Window(c0, start, c1-c0, stop-start),
                masked=True,
            )
            rr, cc = np.meshgrid(
                np.arange(start, stop) + 0.5,
                np.arange(c0, c1) + 0.5,
                indexing="ij",
            )
            x = src.transform.a * cc + src.transform.c
            y = src.transform.e * rr + src.transform.f
            lon, lat = transform(
                src.crs, "EPSG:4326", x.ravel(), y.ravel(),
            )
            lon, lat = np.asarray(lon), np.asarray(lat)
            inside = (
                np.isfinite(lon) & np.isfinite(lat)
                & (lon >= WEST) & (lon < EAST)
                & (lat > SOUTH) & (lat <= NORTH)
            )
            selected = np.flatnonzero(inside)
            rows = np.floor((NORTH-lat[selected])/STEP).astype(int)
            cols = np.floor((lon[selected]-WEST)/STEP).astype(int)
            keep = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
            selected, rows, cols = selected[keep], rows[keep], cols[keep]
            ids = rows * W + cols

            values = np.asarray(data.data).ravel()[selected]
            masked = np.ma.getmaskarray(data).ravel()[selected]
            good = ~masked & np.isfinite(values)
            if kind == "population":
                if np.any(good & (values < 0)):
                    raise ValueError("发现未被掩膜标记的负人口值")
            else:
                good &= np.isin(values, np.arange(1, 10))

            valid_count += np.bincount(ids[good], minlength=H*W)
            missing_count += np.bincount(ids[~good], minlength=H*W)

            if kind == "population":
                total += np.bincount(
                    ids[good], weights=values[good], minlength=H*W,
                )
            else:
                for code in range(1, 10):
                    classes[code-1] += np.bincount(
                        ids[good & (values == code)], minlength=H*W,
                    )

            progress = 100 * (stop-r0) / (r1-r0)
            print(f"\r{kind}：{progress:.1f}%", end="", flush=True)

    print()
    valid = valid_count.reshape(H, W) > 0
    result = {
        "latitude": LAT,
        "longitude": LON,
        "has_valid_data": valid,
        "valid_source_pixel_count": valid_count.reshape(H, W),
        "missing_or_unclassified_source_pixel_count":
            missing_count.reshape(H, W),
    }
    if kind == "population":
        population = total.reshape(H, W)
        population[~valid] = np.nan
        result["population_over_valid_source_pixels"] = population
        if not np.isfinite(population[valid]).all():
            raise ValueError("人口输出存在异常")
    else:
        fractions = np.full((9, H*W), np.nan, dtype=np.float32)
        np.divide(
            classes, valid_count[None, :],
            out=fractions, where=valid_count[None, :] > 0,
        )
        fractions = fractions.reshape(9, H, W)
        if not np.allclose(fractions[:, valid].sum(axis=0), 1):
            raise ValueError("土地覆盖比例之和异常")
        result.update(
            class_proportions=fractions,
            class_codes=np.arange(1, 10),
            class_names=np.array([
                "cropland", "forest", "shrubland", "grassland",
                "water", "snow_ice", "barren", "impervious", "wetland",
            ]),
        )
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True,
                        choices=range(2018, 2025))
    args = parser.parse_args()
    year = args.year
    base = ROOT / "geographic_library"
    jobs = [
        ("population", base / (
            f"population/R2025A_v1/"
            f"chn_pop_{year}_CN_100m_R2025A_v1.tif"
        )),
        ("land_cover", base / (
            f"land_cover/CLCD_2018_2024/CLCD_v01_{year}_albert.tif"
        )),
    ]
    folder = ROOT / "aligned/radar_grid_approx/geography"
    folder.mkdir(parents=True, exist_ok=True)

    for kind, source in jobs:
        target = folder / f"{kind}_{year}.npz"
        if target.exists():
            raise FileExistsError(f"结果已存在，未覆盖：{target}")
        print("正在处理：", source, flush=True)
        result = aggregate(source, kind)
        metadata = {
            "year": year,
            "source_file": str(source),
            "target_grid_id": "radar_700x900_approx_mapping_v1",
            "method": "source_pixel_center_assignment",
            "boundary_pixels_split_by_area": False,
            "coverage_completeness_verified": False,
            "note": "有效数据不代表完整覆盖；无有效数据保留NaN。",
        }
        result["metadata_json"] = np.array(
            json.dumps(metadata, ensure_ascii=False)
        )
        temporary = target.with_suffix(".npz.part")
        with temporary.open("wb") as f:
            np.savez_compressed(f, **result)
        temporary.replace(target)
        print("有效格子：", int(result["has_valid_data"].sum()))
        print("已保存：", target, flush=True)

if __name__ == "__main__":
    main()
