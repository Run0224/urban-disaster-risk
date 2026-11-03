import ast
import json
from pathlib import Path
import numpy as np

root = Path.home() / "satellite_geo/fy4a"
original = root / "align_population_landcover.py"
output = root / "aligned/population_R2025A_v1"
output.mkdir(parents=True, exist_ok=True)

# 复用原脚本的网格和函数，移除末尾旧数据的处理调用。
tree = ast.parse(original.read_text(encoding="utf-8"))
tree.body = [
    node for node in tree.body
    if not (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "process"
    )
]
namespace = {"__name__": "population_alignment"}
exec(compile(tree, str(original), "exec"), namespace)
namespace["output_dir"] = output

lat = namespace["lat"]
lon = namespace["lon"]
assert (len(lat), len(lon)) == (518, 460)
assert np.all(np.diff(lat) < 0)
assert np.all(np.diff(lon) > 0)
assert np.allclose(np.diff(lat), np.diff(lat)[0])
assert np.allclose(np.diff(lon), np.diff(lon)[0])

source = root / "geographic_library/population/R2025A_v1"
for year in range(2018, 2025):
    path = source / f"chn_pop_{year}_CN_100m_R2025A_v1.tif"
    if not path.is_file():
        raise FileNotFoundError(path)

reports = []
for year in range(2018, 2025):
    print(f"\n===== 正在对齐 {year} 年 =====", flush=True)
    path = source / f"chn_pop_{year}_CN_100m_R2025A_v1.tif"
    namespace["process"](path, "population", year)

    target = output / f"population_{year}_approx.npz"
    with np.load(target) as data:
        values = data["population_over_valid_source_pixels"]
        valid = data["has_valid_data"]
        assert values.shape == (518, 460)
        assert np.array_equal(data["latitude"], lat)
        assert np.array_equal(data["longitude"], lon)
        assert np.array_equal(np.isfinite(values), valid)
        assert np.all(values[valid] >= 0)
        assert valid.any()
        report = {
            "year": year,
            "file": str(target),
            "valid_cells": int(valid.sum()),
            "sum_over_valid_source_pixels": float(np.nansum(values)),
        }

    metadata_path = target.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "source_product": "WorldPop R2025A v1 constrained 100m",
        "grid_id": "fy4a_readme_approx_v1",
        "temporal_resolution": "annual",
        "cross_year_substitution": False,
        "nodata_is_zero": False,
        "interpretation_zh": (
            "有效来源像元人口估计之和；缺失值不补零。"
            "有有效数据不代表整个卫星格子完整覆盖。"
            "卫星定位沿用README近似网格。"
        ),
    })
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reports.append(report)
    print(year, "对齐与结果检查通过", flush=True)

(output / "alignment_report.json").write_text(
    json.dumps(reports, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("\n七年人口对齐完成：", output, flush=True)
print("尚未修改训练数据加载逻辑。", flush=True)
