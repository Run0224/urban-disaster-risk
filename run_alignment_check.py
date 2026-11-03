import ast
import io
import json
import shlex
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

root = Path.home() / "satellite_geo/fy4a"
folder = root / "alignment_checks"
folder.mkdir(exist_ok=True)
project = "/mnt/data/deminy/RemoteWorkspace/proj_CI"

def remote_bytes(path):
    result = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "ices@192.168.48.29", "cat " + shlex.quote(str(path)),
        ],
        capture_output=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout

print("读取检查代码和雷达覆盖掩膜……", flush=True)
source = remote_bytes(
    project + "/tools/check_sat_radar_alignment.py"
).decode("utf-8")
(folder / "original_check_script.py").write_text(source, encoding="utf-8")
mask = folder / "radar_valid_mask.npy"
mask.write_bytes(remote_bytes(
    project + "/Dataset_process/our_dataset/assets/radar_valid_mask.npy"
))
assert np.load(mask, allow_pickle=False).shape == (700, 900)

# 保留原检查算法，替换项目依赖及文件读取方式。
tree = ast.parse(source)
tree.body = [
    node for node in tree.body
    if not (
        isinstance(node, ast.ImportFrom)
        and node.module in ("data_provider.sat_reader", "utils.main_util")
    )
    and not (
        isinstance(node, ast.If)
        and "__name__" in ast.unparse(node.test)
    )
]

class Reader:
    def __init__(self, sat_dir, num_channels, offset):
        self.sat_dir = Path(sat_dir)
        self.num_channels = num_channels

cfg = {
    "radar_dir": "/home/ices/data/radar_gd",
    "sat_dir": "/mnt/data/zk/diffcast_3d/data/sat_6min",
    "sat_num_channels": 8,
    "sat_utc_offset_hours": 8,
}
namespace = {
    "__name__": "alignment_check",
    "__file__": str(folder / "original_check_script.py"),
    "RADAR_H": 700, "RADAR_W": 900,
    "SAT_H": 256, "SAT_W": 256,
    "SAT_PAD_H": 131, "SAT_PAD_W": 102,
    "SatReader": Reader,
    "load_config": lambda _: cfg,
    "project_path": lambda p: Path(p),
}
exec(compile(tree, "original_check_script.py", "exec"), namespace)

availability = []

def read_radar(directory, dt):
    stamp = dt.strftime("%Y%m%d%H%M")
    path = Path(directory) / stamp[:4] / stamp[4:6] / stamp[6:8] / (stamp+".png")
    print("读取雷达：", stamp, flush=True)
    try:
        raw = remote_bytes(path)
    except RuntimeError as error:
        print("雷达读取失败：", error, flush=True)
        return None
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("雷达PNG无法解码")
    if image.shape != (700, 900):
        raise ValueError(f"雷达尺寸异常：{image.shape}")
    image = image.astype(np.float32)
    image[(image < 15) | (image >= 74)] = 0
    return image

def read_sat(reader, dt, channel):
    stamp = dt.strftime("%Y%m%d%H%M")
    path = reader.sat_dir / stamp[:4] / stamp[4:6] / stamp[6:8] / stamp / (stamp+".npy")
    print("读取卫星UTC：", stamp, flush=True)
    try:
        array = np.load(io.BytesIO(remote_bytes(path)), allow_pickle=False)
    except RuntimeError as error:
        print("卫星读取失败：", error, flush=True)
        availability.append({"time_utc": stamp, "available": False})
        return None
    if array.shape != (8, 518, 460):
        raise ValueError(f"卫星尺寸异常：{array.shape}")
    field = array[channel].astype(np.float32)
    ok = bool(np.isfinite(field).all())
    availability.append({"time_utc": stamp, "available": ok})
    return field if ok else None

namespace["read_radar"] = read_radar
namespace["read_sat_full"] = read_sat
sys.argv = [
    "check_sat_radar_alignment.py",
    "--config_file", "runtime_override",
    "--valid_mask", str(mask),
    "--times", "201907301500,201907301530,201907301600",
    "--offsets_min=-12:12:6,480",
    "--search", "8", "--skip_diurnal",
    "--out", str(folder / "pilot_20190730.json"),
]
namespace["main"]()
(folder / "pilot_20190730_availability.json").write_text(
    json.dumps(availability, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("本次只做诊断，未修改卫星、雷达或坐标。", flush=True)
