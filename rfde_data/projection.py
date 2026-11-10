"""Project LiDAR or 4D radar points onto the TJ4DRadSet camera image.
"""
import numpy as np
from PIL import Image

# Number of float32 values per point in the TJ4DRadSet binary files.
POINT_DIMS = {"lidar": 4, "radar": 8}


def read_points(path, sensor):
    return np.fromfile(path, dtype=np.float32).reshape(-1, POINT_DIMS[sensor])


def project_points(points_xyz, calib, image_shape):
    xyz = points_xyz[:, :3].astype(np.float32)
    xyz = xyz[xyz[:, 0] >= 0]
    if xyz.shape[0] == 0:
        return np.empty((0, 2), np.int32), np.empty((0,), np.float32)

    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)]).T
    cam = calib["P2"] @ calib["R0_rect"] @ calib["Tr_velo_to_cam"] @ xyz_h
    cam = cam[:, cam[2, :] > 0]
    if cam.shape[1] == 0:
        return np.empty((0, 2), np.int32), np.empty((0,), np.float32)

    cam[:2] /= cam[2, :]
    img_h, img_w = image_shape
    u, v, z = cam
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    pixels = np.stack([u[inside].astype(np.int32), v[inside].astype(np.int32)], axis=1)
    return pixels, z[inside].astype(np.float32)


def nearest_per_pixel(pixels, depths, image_shape):
    img_w = image_shape[1]
    linear = pixels[:, 1].astype(np.int64) * img_w + pixels[:, 0].astype(np.int64)
    order = np.lexsort((depths, linear))
    keep = np.ones(order.shape[0], dtype=bool)
    keep[1:] = linear[order][1:] != linear[order][:-1]
    return order[keep]


def sparse_depth_map(points_xyz, calib, image_shape):
    depth = np.zeros(image_shape, dtype=np.float32)
    mask = np.zeros(image_shape, dtype=np.uint8)
    pixels, depths = project_points(points_xyz, calib, image_shape)
    if len(pixels):
        idx = nearest_per_pixel(pixels, depths, image_shape)
        chosen = pixels[idx]
        depth[chosen[:, 1], chosen[:, 0]] = depths[idx]
        mask[chosen[:, 1], chosen[:, 0]] = 1
    return depth, mask


def save_depth_png(depth, path):
    value = (depth * 256).clip(0, 65535).astype(np.uint16)
    Image.fromarray(value, mode="I;16").save(path)


def read_depth_png(path):
    return np.asarray(Image.open(path), dtype=np.float32) / 256.0


def save_mask_png(mask, path):
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)
