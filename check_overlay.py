import io
import subprocess
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

folder = Path.home() / "satellite_geo/fy4a"
path = (
    "/mnt/data/zk/diffcast_3d/data/sat_6min/"
    "2018/01/01/201801010000/201801010000.npy"
)

# 只读取远程一帧，图片输出到 c055。
result = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "ices@192.168.48.29",
     f"cat {path}"],
    check=True, stdout=subprocess.PIPE,
)
sat = np.load(io.BytesIO(result.stdout), allow_pickle=False)
if sat.shape != (8, 518, 460):
    raise ValueError(f"卫星形状异常：{sat.shape}")

with np.load(folder / "static_geography_approx.npz") as geo:
    lat = geo["latitude"]
    lon = geo["longitude"]

dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
west, east = lon[0] - dx/2, lon[-1] + dx/2
south, north = lat[-1] - dy/2, lat[0] + dy/2

counties = gpd.read_file(
    folder / "sources/china_county.geojson"
).to_crs("EPSG:4326")
region = box(west, south, east, north)
counties = counties[counties.intersects(region)].copy()
counties.geometry = counties.geometry.intersection(region)

fig, ax = plt.subplots(figsize=(12, 9))
image = ax.imshow(
    sat[5], origin="upper",
    extent=[west, east, south, north],
    cmap="gray_r", vmin=180, vmax=320,
)
counties.boundary.plot(
    ax=ax, color="cyan", linewidth=0.25, alpha=0.6
)

# 卫星中心窗口：[131:387, 102:358]。
left = lon[102] - dx/2
bottom = lat[386] - dy/2
ax.add_patch(Rectangle(
    (left, bottom), 256*dx, 256*dy,
    fill=False, edgecolor="yellow", linewidth=2,
    label="Nominal radar coverage",
))

ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    "FY-4A 2018-01-01 00:00 UTC | 10.8 um\n"
    "Approximate coordinates; 2017 county boundaries"
)
ax.legend(loc="lower left")
fig.colorbar(image, ax=ax, label="Brightness temperature (K)")
output = folder / "satellite_county_overlay.png"
fig.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)
print("图片已保存：", output)
