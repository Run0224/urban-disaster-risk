import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

root = Path.home() / "satellite_geo/fy4a"
library = root / "geographic_library"
output_dir = root / "aligned"
output_dir.mkdir(exist_ok=True)

with np.load(root / "coordinates_approx.npz") as geo:
    lat = geo["latitude"]
    lon = geo["longitude"]

H, W = len(lat), len(lon)
N = H * W
dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
west = float(lon[0] - dx/2)
north = float(lat[0] + dy/2)
east = west + W*dx
south = north - H*dy

codes = np.array([10,20,30,40,50,60,70,80,90,95,100])
names = [
    "tree_cover", "shrubland", "grassland", "cropland",
    "built_up", "bare_sparse_vegetation", "snow_ice",
    "permanent_water", "herbaceous_wetland",
    "mangroves", "moss_lichen"
]

def process(path, kind, year):
    valid_count = np.zeros(N, dtype="int64")
    selected_count = np.zeros(N, dtype="int64")
    population_sum = np.zeros(N, dtype="float64")
    class_counts = np.zeros((len(codes), N), dtype="int64")

    with rasterio.open(path) as src:
        t = src.transform
        if (src.crs.to_epsg() != 4326 or t.b != 0 or t.d != 0
                or t.a <= 0 or t.e >= 0):
            raise RuntimeError("源网格不符合预期，暂停处理")

        b = src.bounds
        if east <= b.left or west >= b.right or north <= b.bottom or south >= b.top:
            raise RuntimeError("源文件与卫星范围不相交")

        window = from_bounds(
            max(west,b.left), max(south,b.bottom),
            min(east,b.right), min(north,b.top), t
        )
        r0 = max(0, int(np.floor(window.row_off)))
        r1 = min(src.height, int(np.ceil(window.row_off+window.height)))
        c0 = max(0, int(np.floor(window.col_off)))
        c1 = min(src.width, int(np.ceil(window.col_off+window.width)))

        x = t.c + (np.arange(c0,c1)+0.5)*t.a
        destination_col = np.floor((x-west)/dx).astype("int64")
        processed = 0
        next_report = 2_000_000

        for start in range(r0,r1,64):
            stop = min(start+64,r1)
            a = src.read(
                1, window=Window(c0,start,c1-c0,stop-start),
                masked=True
            )
            y = t.f + (np.arange(start,stop)+0.5)*t.e
            destination_row = np.floor((north-y)/dy).astype("int64")
            inside = (
                (destination_row[:,None] >= 0)
                & (destination_row[:,None] < H)
                & (destination_col[None,:] >= 0)
                & (destination_col[None,:] < W)
            )

            rr, cc = np.nonzero(inside)
            ids = destination_row[rr]*W + destination_col[cc]
            selected_count += np.bincount(ids,minlength=N)

            usable = inside & ~np.ma.getmaskarray(a) & np.isfinite(a.data)
            if kind == "population":
                usable &= a.data >= 0
            else:
                usable &= np.isin(a.data,codes)

            rr, cc = np.nonzero(usable)
            ids = destination_row[rr]*W + destination_col[cc]
            values = a.data[rr,cc]
            valid_count += np.bincount(ids,minlength=N)

            if kind == "population":
                population_sum += np.bincount(
                    ids,weights=values,minlength=N
                )
            elif len(ids):
                labels = np.searchsorted(codes,values)
                packed = labels*N + ids
                class_counts += np.bincount(
                    packed,minlength=len(codes)*N
                ).reshape(len(codes),N)

            processed += a.size
            if processed >= next_report or stop == r1:
                print(
                    f"{kind}: 已处理 {processed:,} 个源像元，"
                    f"行进度 {stop-r0}/{r1-r0}", flush=True
                )
                next_report = processed + 2_000_000

        # 仅检查源文件矩形是否完全覆盖目标格子；
        # 与“该格子内所有源数据有效”分别保存。
        full_extent = (
            (lat[:,None]+dy/2 <= b.top)
            & (lat[:,None]-dy/2 >= b.bottom)
            & (lon[None,:]-dx/2 >= b.left)
            & (lon[None,:]+dx/2 <= b.right)
        )

    has_data = valid_count > 0
    common = dict(
        latitude=lat,
        longitude=lon,
        valid_source_pixel_count=valid_count.reshape(H,W),
        missing_or_invalid_source_pixel_count=(
            selected_count-valid_count
        ).reshape(H,W),
        source_extent_covers_cell=full_extent,
        has_valid_data=has_data.reshape(H,W),
        all_selected_source_pixels_valid=(
            has_data & (selected_count==valid_count)
        ).reshape(H,W),
    )

    if kind == "population":
        population_sum[~has_data] = np.nan
        common["population_over_valid_source_pixels"] = (
            population_sum.reshape(H,W)
        )
        total = float(np.nansum(population_sum))
        print("有效来源人口估计合计：",total,flush=True)
    else:
        fractions = np.full((len(codes),N),np.nan,dtype="float64")
        np.divide(
            class_counts,valid_count[None,:],
            out=fractions,where=has_data[None,:]
        )
        if has_data.any() and not np.allclose(
            fractions[:,has_data].sum(axis=0),1.0
        ):
            raise RuntimeError("类别比例检查失败")
        common["class_proportions"] = fractions.reshape(len(codes),H,W)
        common["class_codes"] = codes
        common["class_names"] = np.array(names)

    target = output_dir / f"{kind}_{year}_approx.npz"
    np.savez_compressed(target,**common)

    metadata = {
        "year":year,
        "source_file":str(path),
        "shape":[H,W],
        "geolocation_status":"approximate_from_readme",
        "method":"Assign each source pixel by its center to one satellite cell.",
        "boundary_handling":"No fractional splitting of source pixels.",
        "missing_value":"NaN",
        "population_units":"people per satellite cell over valid source pixels",
        "landcover_proportions":"Valid source-pixel count fractions; approximate area fractions.",
        "limitation":"Has-valid-data does not guarantee complete coverage. Check extent and validity masks. No cross-year substitution."
    }
    target.with_suffix(".metadata.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )
    print("已保存：",target,flush=True)
    print("有有效数据的卫星像素：",int(has_data.sum()),flush=True)

process(
    library/"population/chn_ppp_2018.tif",
    "population",2018
)
process(
    library/"land_cover/ESA_WorldCover_10m_2020_v100_N21E111_Map.tif",
    "land_cover",2020
)
