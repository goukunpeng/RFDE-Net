"""Project an image-plane freespace mask and a dense depth map to a BEV grid.
"""
import math
import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageDraw
from .calib import CROP_TOP, camera_to_sensor, read_calib


@dataclass(frozen=True)
class BEVConfig:
    forward_range: tuple = (1.0, 60.0)   # metres, [min, max)
    lateral_range: tuple = (-10.0, 10.0)  # metres, [min, max)
    resolution: float = 0.2               # metres per cell
    min_depth: float = 0.5                # valid depth for back-projection, metres
    max_depth: float = 80.0
    pixel_step: int = 1

    @property
    def shape(self):
        rows = int(math.ceil((self.forward_range[1] - self.forward_range[0]) / self.resolution))
        cols = int(math.ceil((self.lateral_range[1] - self.lateral_range[0]) / self.resolution))
        return rows, cols

    def rows_up_to(self, distance):
        return int(round((distance - self.forward_range[0]) / self.resolution))


def load_bev_calib(calib_path):
    calib = read_calib(calib_path, crop_top=CROP_TOP)
    return calib["P2"], camera_to_sensor(calib)


def mask_depth_to_points(mask, depth, p2, cam_to_sensor, cfg):
    step = max(int(cfg.pixel_step), 1)
    rows = np.arange(0, mask.shape[0], step, dtype=np.int32)
    cols = np.arange(0, mask.shape[1], step, dtype=np.int32)
    m = mask[np.ix_(rows, cols)]
    d = depth[np.ix_(rows, cols)]
    valid = m & np.isfinite(d) & (d >= cfg.min_depth) & (d <= cfg.max_depth)
    origin = (cam_to_sensor @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))[:3].astype(np.float32)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), origin

    r, c = np.nonzero(valid)
    v = rows[r].astype(np.float32)
    u = cols[c].astype(np.float32)
    z = d[valid].astype(np.float32)
    fx, fy, cx, cy = p2[0, 0], p2[1, 1], p2[0, 2], p2[1, 2]
    pts_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z, np.ones_like(z)], axis=0)
    pts = (cam_to_sensor @ pts_cam).T[:, :3].astype(np.float32)
    return pts, origin


def rasterize(points, origin, cfg):
    height, width = cfg.shape
    bev = np.zeros((height, width), dtype=np.uint8)
    if points.size == 0:
        return bev
    f_min, f_max = cfg.forward_range
    l_min, l_max = cfg.lateral_range
    fwd, lat = points[:, 0], points[:, 1]
    valid = (np.isfinite(fwd) & np.isfinite(lat) & (fwd >= f_min) & (fwd < f_max)
             & (lat >= l_min) & (lat < l_max))
    if not np.any(valid):
        return bev

    end_rows = np.floor((fwd[valid] - f_min) / cfg.resolution).astype(np.int32)
    end_cols = np.floor((lat[valid] - l_min) / cfg.resolution).astype(np.int32)
    cells = np.unique(end_rows * width + end_cols)
    end_rows, end_cols = cells // width, cells % width

    # The camera centre lies outside the grid; its projection is clipped to the nearest cell.
    o_row = int(np.clip(np.floor((origin[0] - f_min) / cfg.resolution), 0, height - 1))
    o_col = int(np.clip(np.floor((origin[1] - l_min) / cfg.resolution), 0, width - 1))
    image = Image.fromarray(bev, mode="L")
    draw = ImageDraw.Draw(image)
    for row, col in zip(end_rows.tolist(), end_cols.tolist()):
        draw.line((o_col, o_row, int(col), int(row)), fill=1, width=1)
    return np.array(image, dtype=np.uint8)


def resize_depth(depth, shape):
    if depth.shape == tuple(shape):
        return depth
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)


def mask_depth_to_bev(mask, depth, calib_path, cfg=BEVConfig()):
    p2, cam_to_sensor = load_bev_calib(calib_path)
    depth = resize_depth(depth, mask.shape)
    points, origin = mask_depth_to_points(mask.astype(bool), depth, p2, cam_to_sensor, cfg)
    return rasterize(points, origin, cfg)
