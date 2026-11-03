import json
from pathlib import Path
import numpy as np
from load_satellite_population import load_sample as load_population_sample

ROOT = Path.home() / "satellite_geo/fy4a"
MANIFEST = ROOT / "training/satellite_disaster_weak_samples_v6_geography.jsonl"

def load_sample(item):
    ref = item["land_cover_reference"]
    year = int(item["time_utc"][:4])
    if ref["year"] != year or ref["grid_id"] != item["grid_id"]:
        raise ValueError("土地覆盖年份或网格不一致")

    result = load_population_sample(item)
    with np.load(ref["file"], allow_pickle=False) as data:
        fractions = data[ref["value_key"]].copy()
        valid = data[ref["valid_mask_key"]].copy()
        names = data[ref["class_names_key"]].copy()
        codes = data[ref["class_codes_key"]].copy()
        counts = data[ref["valid_count_key"]].copy()
        missing = data[ref["missing_count_key"]].copy()
        latitude = data["latitude"].copy()
        longitude = data["longitude"].copy()

    with np.load(item["population_reference"]["file"], allow_pickle=False) as pop:
        if not (
            np.array_equal(latitude, pop["latitude"])
            and np.array_equal(longitude, pop["longitude"])
        ):
            raise ValueError("人口与土地覆盖坐标不一致")

    shape = result["satellite"].shape[-2:]
    if fractions.shape != (9, *shape):
        raise ValueError("土地覆盖比例尺寸异常")
    for array in (valid, counts, missing):
        if array.shape != shape:
            raise ValueError("土地覆盖掩膜或计数尺寸异常")
    if not np.array_equal(codes, np.arange(1, 10)):
        raise ValueError("类别编码异常")
    if len(names) != 9 or not valid.any():
        raise ValueError("类别名称或有效数据异常")
    if not np.array_equal(counts > 0, valid):
        raise ValueError("有效计数与掩膜不一致")
    if not np.isfinite(fractions[:, valid]).all():
        raise ValueError("有效位置存在非有限比例")
    if not np.all((fractions[:, valid] >= 0) & (fractions[:, valid] <= 1)):
        raise ValueError("比例超出0到1")
    if not np.allclose(fractions[:, valid].sum(axis=0), 1):
        raise ValueError("类别比例之和不为1")
    if not np.isnan(fractions[:, ~valid]).all():
        raise ValueError("缺失位置应保留NaN")

    result.update({
        "land_cover_proportions": fractions,
        "land_cover_valid": valid,
        "land_cover_class_names": names,
        "land_cover_class_codes": codes,
        "land_cover_valid_count": counts,
        "land_cover_missing_count": missing,
    })
    return result

if __name__ == "__main__":
    examples = {}
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                examples.setdefault(item["year"], item)
    for year, item in sorted(examples.items()):
        print(year, "年：正在联合读取", flush=True)
        data = load_sample(item)
        print(
            "通过；卫星:", data["satellite"].shape,
            "人口:", data["population"].shape,
            "土地覆盖:", data["land_cover_proportions"].shape,
            flush=True,
        )
    print("逐年抽样联合读取完成；完整网格、未启动训练。", flush=True)
