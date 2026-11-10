"""LiDAR-completed dense depth targets with BP-Net.
"""
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfde_data.calib import CROP_TOP, read_calib

KITTI_MEAN = (90.9950, 96.2278, 94.3213)
KITTI_STD = (79.2382, 80.5267, 82.1483)


def read_ids(path):
    return [line.strip().split(".")[0] for line in Path(path).read_text().splitlines() if line.strip()]


def load_inputs(rgb_path, sparse_path, calib_path):
    """Return BP-Net inputs (rgb, lidar, K) as float32 numpy arrays (C, H, W)."""
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8).astype(np.float32)
    rgb = (rgb - np.array(KITTI_MEAN)) / np.array(KITTI_STD)
    lidar = (np.array(Image.open(sparse_path), dtype=int) / 256.0).astype(np.float32)
    rgb = rgb.transpose(2, 0, 1).astype(np.float32)[:, CROP_TOP:, :]
    lidar = lidar[None, CROP_TOP:, :]
    k = read_calib(calib_path, crop_top=CROP_TOP)["P2"][:, :3].astype(np.float32)
    return rgb, lidar, k


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bpnet-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tj4d-root", type=Path, required=True, help="folder containing image_2")
    parser.add_argument("--lidar-calib-dir", type=Path, required=True,
                        help="LiDAR calibration folder, e.g. TJ4DRadSet_LiDAR/training/calib")
    parser.add_argument("--sparse-dir", type=Path, required=True, help="output of project_points.py --sensor lidar")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(args.bpnet_root.resolve()))
    import torch
    from models import Pre_MF_Post  # BP-Net model (configs/net/PMP.yaml)

    net = Pre_MF_Post()
    state = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(state["net"], strict=True)
    net.to(args.device).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = read_ids(args.split_file)
    with torch.no_grad():
        for i, sid in enumerate(ids):
            rgb, lidar, k = load_inputs(args.tj4d_root / "image_2" / f"{sid}.png",
                                        args.sparse_dir / f"{sid}.png",
                                        args.lidar_calib_dir / f"{sid}.txt")
            tensors = [torch.from_numpy(x)[None].to(args.device) for x in (rgb, lidar, k)]
            output = net(*tensors)
            if isinstance(output, (list, tuple)):
                output = output[-1]
            depth = (output[0, 0] * 256.0).cpu().numpy().astype("uint16")
            Image.fromarray(depth).save(args.out_dir / f"{sid}.png")
            if (i + 1) % 200 == 0 or i + 1 == len(ids):
                print(f"BP-Net: {i + 1}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main()
