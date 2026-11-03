import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform, transform_bounds

root = Path.home() / "satellite_geo/fy4a"
source = root / "geographic_library/land_cover/CLCD_2018_2024"
output = root / "aligned/land_cover_CLCD"
output.mkdir(parents=True, exist_ok=True)

with np.load(root / "coordinates_approx.npz") as geo:
    lat = geo["latitude"].copy()
    lon = geo["longitude"].copy()

H, W = len(lat), len(lon)
assert (H, W) == (518, 460)
dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
assert dx > 0 and dy > 0
assert np.allclose(np.diff(lon), dx)
assert np.allclose(np.diff(lat), -dy)
west, north = float(lon[0]-dx/2), float(lat[0]+dy/2)
east, south = west+W*dx, north-H*dy
N = H*W

names = np.array([
    "cropland", "forest", "shrubland", "grassland",
    "water", "snow_ice", "barren", "impervious", "wetland",
])
reports = []

for year in range(2018, 2025):
    target = output / f"land_cover_{year}_approx.npz"
    metadata_path = target.with_suffix(".metadata.json")
    if target.exists() or metadata_path.exists():
        raise FileExistsError(f"已有结果，停止以免覆盖：{target}")

    counts = np.zeros((9, N), dtype=np.int64)
    selected = np.zeros(N, dtype=np.int64)
    print(f"\n===== {year} 年开始 =====", flush=True)

    with rasterio.open(source / f"CLCD_v01_{year}_albert.tif") as src:
        t = src.transform
        if src.crs is None or t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError("源投影或网格方向不符合预期")

        bounds = transform_bounds(
            "EPSG:4326", src.crs,
            west, south, east, north, densify_pts=101,
        )
        win = from_bounds(*bounds, transform=t)
        r0 = max(0, int(np.floor(win.row_off))-2)
        r1 = min(src.height, int(np.ceil(win.row_off+win.height))+2)
        c0 = max(0, int(np.floor(win.col_off))-2)
        c1 = min(src.width, int(np.ceil(win.col_off+win.width))+2)
        if r1 <= r0 or c1 <= c0:
            raise ValueError("源数据与卫星范围不相交")

        xs = t.c + (np.arange(c0, c1)+0.5)*t.a
        last_report = -1

        for start in range(r0, r1, 16):
            stop = min(start+16, r1)
            data = src.read(
                1, window=Window(c0, start, c1-c0, stop-start),
                masked=True,
            )
            ys = t.f + (np.arange(start, stop)+0.5)*t.e
            xx, yy = np.meshgrid(xs, ys)
            xlon, ylat = transform(
                src.crs, "EPSG:4326", xx.ravel(), yy.ravel(),
            )
            xlon = np.asarray(xlon)
            ylat = np.asarray(ylat)
            finite = np.isfinite(xlon) & np.isfinite(ylat)
            inside = (
                finite & (xlon >= west) & (xlon < east)
                & (ylat <= north) & (ylat > south)
            )
            positions = np.flatnonzero(inside)
            cols = np.floor((xlon[positions]-west)/dx).astype(np.int64)
            rows = np.floor((north-ylat[positions])/dy).astype(np.int64)
            keep = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
            positions = positions[keep]
            ids = rows[keep]*W + cols[keep]
            selected += np.bincount(ids, minlength=N)

            values = data.data.ravel()[positions]
            masked = np.ma.getmaskarray(data).ravel()[positions]
            usable = ~masked & np.isfinite(values) & (values >= 1) & (values <= 9)
            if np.any(usable & (values != np.floor(values))):
                raise ValueError("发现非整数类别编码")
            labels = values[usable].astype(np.int64)-1
            packed = labels*N + ids[usable]
            counts += np.bincount(packed, minlength=9*N).reshape(9, N)

            percent = int(100*(stop-r0)/(r1-r0))
            if percent != last_report:
                print(year, f"行进度：{percent}%", flush=True)
                last_report = percent

        source_crs = src.crs.to_wkt()

    valid_count = counts.sum(axis=0)
    valid = valid_count > 0
    fractions = np.full((9, N), np.nan, dtype=np.float64)
    np.divide(
        counts, valid_count[None, :],
        out=fractions, where=valid[None, :],
    )
    assert valid.any()
    assert np.allclose(fractions[:, valid].sum(axis=0), 1)
    assert np.all(selected >= valid_count)

    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(
            f,
            latitude=lat, longitude=lon,
            class_codes=np.arange(1, 10),
            class_names=names,
            class_proportions=fractions.reshape(9, H, W),
            has_valid_data=valid.reshape(H, W),
            valid_source_pixel_count=valid_count.reshape(H, W),
            selected_source_pixel_count=selected.reshape(H, W),
            missing_or_unclassified_source_pixel_count=(
                selected-valid_count
            ).reshape(H, W),
        )
    with np.load(temporary, allow_pickle=False) as check:
        assert check["class_proportions"].shape == (9, 518, 460)
        assert np.array_equal(check["latitude"], lat)
        assert np.array_equal(check["longitude"], lon)
    temporary.replace(target)

    metadata = {
        "year": year,
        "source_product": "CLCD",
        "source_record": "https://zenodo.org/records/15853565",
        "source_file": str(source / f"CLCD_v01_{year}_albert.tif"),
        "source_crs": source_crs,
        "grid_id": "fy4a_readme_approx_v1",
        "geolocation_status": "approximate_from_readme",
        "method": "Transform source pixel centers to WGS84 and assign to satellite cells.",
        "proportion_denominator": "valid classified source pixels only",
        "boundary_handling": "No fractional splitting of source pixels.",
        "nodata_is_zero": False,
        "cross_year_substitution": False,
        "note_zh": "比例为有效来源像元计数比例，近似面积比例；不保证整个格子完整覆盖。",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reports.append({"year": year, "valid_cells": int(valid.sum()), "file": str(target)})
    print(year, "保存及结果检查通过，有效格子：", int(valid.sum()), flush=True)

(output / "alignment_report.json").write_text(
    json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8",
)
print("七年CLCD对齐完成；尚未更新地理索引及训练清单。", flush=True)
