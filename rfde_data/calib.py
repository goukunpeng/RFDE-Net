"""Read TJ4DRadSet calibration files (KITTI-style text format).
"""
import numpy as np
from pathlib import Path

# Rows removed from the top of the 1280 x 960 image to obtain the 1280 x 640 crop.
CROP_TOP = 320


def _to_4x4(matrix):
    output = np.eye(4, dtype=np.float32)
    if matrix.shape == (3, 3):
        output[:3, :3] = matrix
    elif matrix.shape == (3, 4):
        output[:3, :] = matrix
    elif matrix.shape == (4, 4):
        output = matrix.astype(np.float32)
    else:
        raise ValueError(f"Unsupported matrix shape: {matrix.shape}")
    return output


def read_calib(path, crop_top=0):
    entries = {}
    with open(Path(path), "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, values = line.split(":", 1)
            entries[key.strip()] = np.array(values.split(), dtype=np.float32)

    p2 = entries["P2"].reshape(3, 4).copy()
    p2[1, 2] -= float(crop_top)
    r0 = entries.get("R0_rect", np.eye(3, dtype=np.float32).reshape(-1)).reshape(3, 3)
    tr = entries["Tr_velo_to_cam"].reshape(3, 4)
    return {
        "P2": p2.astype(np.float32),
        "R0_rect": _to_4x4(r0),
        "Tr_velo_to_cam": _to_4x4(tr),
    }


def camera_to_sensor(calib):
    return np.linalg.inv(calib["R0_rect"] @ calib["Tr_velo_to_cam"]).astype(np.float32)
