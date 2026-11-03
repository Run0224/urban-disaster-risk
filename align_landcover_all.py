import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

root = Path.home() / "satellite_geo/fy4a"
folder = root / "geographic_library/land_cover"
out = root / "aligned"
out.mkdir(exist_ok=True)

plan = json.loads((folder / "download_plan_2020.json").read_text())
with np.load(root / "coordinates_approx.npz") as geo:
    lat, lon = geo["latitude"], geo["longitude"]

H, W = len(lat), len(lon)
N = H * W
dx, dy = float(lon[1]-lon[0]), float(lat[0]-lat[1])
west, north = float(lon[0]-dx/2), float(lat[0]+dy/2)
east, south = west+W*dx, north-H*dy

codes = np.array([10,20,30,40,50,60,70,80,90,95,100])
names = np.array([
    "tree_cover","shrubland","grassland","cropland","built_up",
    "bare_sparse_vegetation","snow_ice","permanent_water",
    "herbaceous_wetland","mangroves","moss_lichen"
])

counts = np.zeros((len(codes),N), dtype="int64")
selected = np.zeros(N, dtype="int64")

for index, tile in enumerate(plan["tiles"], 1):
    path = Path(tile["file"])
    if not path.exists() or path.stat().st_size != tile["size_bytes"]:
        raise RuntimeError(f"文件缺失或大小异常：{path}")

    with rasterio.open(path) as src:
        t = src.transform
        if (src.crs.to_epsg() != 4326 or t.b != 0 or t.d != 0
                or t.a <= 0 or t.e >= 0):
            raise RuntimeError(f"源网格异常：{path}")

        b = src.bounds
        window = from_bounds(
            max(west,b.left), max(south,b.bottom),
            min(east,b.right), min(north,b.top), t
        )
        r0 = max(0,int(np.floor(window.row_off)))
        r1 = min(src.height,int(np.ceil(window.row_off+window.height)))
        c0 = max(0,int(np.floor(window.col_off)))
        c1 = min(src.width,int(np.ceil(window.col_off+window.width)))

        xs = t.c+(np.arange(c0,c1)+0.5)*t.a
        cols = np.floor((xs-west)/dx).astype("int64")

        for start in range(r0,r1,64):
            stop = min(start+64,r1)
            a = src.read(
                1, window=Window(c0,start,c1-c0,stop-start),
                masked=True
            )
            ys = t.f+(np.arange(start,stop)+0.5)*t.e
            rows = np.floor((north-ys)/dy).astype("int64")
            inside = (
                (rows[:,None]>=0)&(rows[:,None]<H)
                &(cols[None,:]>=0)&(cols[None,:]<W)
            )

            rr,cc = np.nonzero(inside)
            ids = rows[rr]*W+cols[cc]
            unique, number = np.unique(ids,return_counts=True)
            selected[unique] += number

            valid = (
                inside & ~np.ma.getmaskarray(a)
                & np.isin(a.data,codes)
            )
            rr,cc = np.nonzero(valid)
            ids = rows[rr]*W+cols[cc]
            if ids.size:
                unique,inverse = np.unique(ids,return_inverse=True)
                labels = np.searchsorted(codes,a.data[rr,cc])
                local = np.bincount(
                    labels*len(unique)+inverse,
                    minlength=len(codes)*len(unique)
                ).reshape(len(codes),len(unique))
                counts[:,unique] += local

            if (start-r0)//64 % 100 == 0:
                print(
                    f"瓦片 {index}/{len(plan['tiles'])}："
                    f"行进度 {stop-r0}/{r1-r0}",
                    flush=True
                )

    print(f"瓦片 {index}/{len(plan['tiles'])} 完成",flush=True)

valid_count = counts.sum(axis=0)
valid = valid_count > 0
fractions = np.full(counts.shape,np.nan,dtype="float64")
np.divide(
    counts,valid_count[None,:],
    out=fractions,where=valid[None,:]
)
if not valid.any():
    raise RuntimeError("没有有效分类，暂停保存")
if not np.allclose(fractions[:,valid].sum(axis=0),1):
    raise RuntimeError("类别比例检查失败")

output = out / "land_cover_2020_full_approx.npz"
np.savez_compressed(
    output,
    latitude=lat, longitude=lon,
    class_codes=codes, class_names=names,
    class_proportions=fractions.reshape(len(codes),H,W).astype("float32"),
    valid_source_pixel_count=valid_count.reshape(H,W),
    selected_source_pixel_count=selected.reshape(H,W),
    missing_or_unclassified_source_pixel_count=(
        selected-valid_count
    ).reshape(H,W),
    has_valid_data=valid.reshape(H,W),
)

metadata = {
    "source":"ESA WorldCover 2020 v100",
    "year":2020,
    "geolocation_status":"approximate_from_readme",
    "tiles":len(plan["tiles"]),
    "method":"Assign source pixel centers to satellite cells; combine class counts across all tiles before computing proportions.",
    "proportions":"Valid source-pixel count fractions, approximating area fractions.",
    "missing":"NaN where no valid classification exists.",
    "coverage_note":"Valid data may cover only part of a cell. Selected-source counts do not prove complete coverage.",
    "download_plan":str(folder / "download_plan_2020.json"),
}
output.with_suffix(".metadata.json").write_text(
    json.dumps(metadata,ensure_ascii=False,indent=2),
    encoding="utf-8"
)

print("类别比例检查通过",flush=True)
print("有有效数据的卫星像素：",int(valid.sum()),flush=True)
print("无有效分类的卫星像素：",int((~valid).sum()),flush=True)
print("已保存：",output,flush=True)
