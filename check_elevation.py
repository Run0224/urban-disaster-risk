from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path.home() / "satellite_geo/fy4a"

with np.load(folder / "static_geography_approx.npz") as data:
    lat = data["latitude"]
    lon = data["longitude"]
    elevation = data["surface_elevation_center_m"]
    valid = data["elevation_valid_mask"]

dx = float(lon[1] - lon[0])
dy = float(lat[0] - lat[1])
extent = [
    lon[0] - dx / 2,
    lon[-1] + dx / 2,
    lat[-1] - dy / 2,
    lat[0] + dy / 2,
]

cmap = plt.get_cmap("terrain").copy()
cmap.set_bad("lightgray")

fig, ax = plt.subplots(figsize=(10, 8))
image = ax.imshow(
    np.ma.masked_where(~valid, elevation),
    extent=extent,
    origin="upper",
    cmap=cmap,
    aspect="equal",
)
ax.set_xlabel("Longitude (degrees east)")
ax.set_ylabel("Latitude (degrees north)")
ax.set_title(
    "FY-4A grid: surface elevation at pixel centers\n"
    "Approximate geolocation | Gray = no valid elevation"
)
ax.grid(alpha=0.25)
fig.colorbar(image, ax=ax, label="Surface elevation (m)")

output = folder / "elevation_check.png"
fig.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)
print("检查图已保存：", output)
